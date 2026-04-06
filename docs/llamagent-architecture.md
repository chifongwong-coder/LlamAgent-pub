# LlamAgent 架构设计文档

> 本文档是 LlamAgent 框架的自包含架构参考，涵盖核心层、模块系统（含事件 Hook v1.8）、工具系统、执行策略、功能模块（含技能系统 v1.4、Job 系统 v1.5、共享检索层 v1.7）、接口层、安全体系（含三层区域系统）、配置参考、沙箱执行系统、子 Agent 控制系统与 Continuous Runner（v2.0）。

---

## 1. 项目概述与设计哲学

### 1.1 项目简介

- **名称**：LlamAgent（一只超级能干的羊驼 AI Agent）
- **定位**：模块化、可扩展的 AI Agent 框架
- **技术栈**：Python + LiteLLM + ChromaDB + FastAPI + Gradio
- **代码路径**：`llamagent/`

```
llamagent/
├── __init__.py
├── main.py                  # 入口，动态加载模块
├── core/                    # 核心层（独立可运行）
│   ├── agent.py             # LlamAgent + Module 基类
│   ├── config.py            # 统一配置
│   ├── authorization.py     # 授权引擎 + 策略（v1.9）
│   ├── contract.py          # Task Mode 合同与状态（v1.9.1）
│   ├── controller.py        # Task Mode 驱动控制器（v1.9.6）
│   ├── runner.py            # ContinuousRunner + Trigger（v2.0）
│   ├── hooks.py             # 事件 Hook 系统（v1.8）
│   ├── llm.py               # LLMClient（LiteLLM 封装）
│   └── persona.py           # 角色系统
├── modules/                 # 可插拔模块
│   ├── tools/               # 工具调用
│   ├── memory/              # 记忆系统
│   ├── rag/                 # RAG 知识检索
│   ├── reasoning/           # 任务规划（推理规划）
│   ├── reflection/          # 自我反思
│   ├── multi_agent/         # 多 Agent 协作
│   ├── mcp/                 # MCP 外部集成
│   ├── safety/              # 安全护栏
│   ├── skill/               # v1.4 技能系统（playbook 层）
│   ├── job/                 # v1.5 Job 执行生命周期管理
│   └── retrieval/           # v1.7 共享检索服务包（非 Module）
└── interfaces/              # 交互接口
    ├── cli.py               # 终端界面
    ├── web_ui.py            # Web 界面
    └── api_server.py        # HTTP API
```

### 1.2 设计哲学

| 原则 | 说明 |
|------|------|
| **核心独立** | 不加载任何模块就能对话——LlamAgent 只依赖 Config + LLMClient |
| **模块可插拔** | 继承 Module 基类，通过 `register_module()` 挂载，按需组合 |
| **松耦合** | 模块间用 `agent.has_module()` 软检查，不硬依赖 |
| **优雅降级** | 可选依赖（chromadb / gradio / fastapi / mcp）未安装时 `try/except`，不影响核心 |
| **角色隔离** | 记忆和自定义工具按 `persona_id` 隔离 |
| **单一工具字典** | core 维护 `_tools` 扁平字典，模块通过 `register_tool()` 注册；ToolsModule 内部另有自己的 `ToolRegistry` 实例用于组织管理 |
| **执行策略可插拔** | 默认 SimpleReAct，可替换为 PlanReAct（Plan + 逐步 ReAct） |
| **动态上下文分层** | Tool 负责"能做什么"，Skill 负责"该怎么做"——skill 通过 on_context 按需注入，不混入 tool 系统 |
| **配置驱动** | 统一 Config 类，支持 YAML 层次化配置，优先级：环境变量 > YAML > 代码默认值 |

### 1.3 核心架构原则（v1.9.6+）

六条原则约束所有核心架构变更：

| 原则 | 说明 |
|------|------|
| **P1 Agent 独立** | Agent 是 courier，不理解 controller/engine 内部逻辑 |
| **P2 Agent 不知内部** | Agent 只路由数据，不检查数据内容 |
| **P3 Controller 纯状态机** | 不引用 agent/engine，所有数据通过参数和返回值传递 |
| **P4 Engine 持有状态** | 所有授权状态由 AuthorizationEngine 管理 |
| **P5 单向依赖** | zone.py(L3) -> contract.py(L2) -> authorization.py/controller.py(L1) -> agent.py(L0) |
| **P6 不过度设计** | 不实现假设性的未来需求，但为可预见的需求预留扩展接口 |

---

## 2. 核心层（Core）

核心层包含 LlamAgent、Config、LLMClient、Persona/PersonaManager 四个组件，是框架最小可运行单元。

### 2.1 LlamAgent（`core/agent.py`）

Agent 主体。核心执行引擎采用 ReAct 循环（Thought → Action → Observation），工具调用内嵌于 Action 阶段。

#### 2.1.1 chat() Pipeline 完整流程

```
用户输入
  │
  ├─ -1. _abort = False    v2.0: 重置 abort 标志（清除上一任务的残留信号）
  │
  ├─ 0. SESSION_START    首次 chat 时触发一次（v1.8 事件 Hook）
  ├─    PRE_CHAT         每轮触发（v1.8 事件 Hook，观察性，不支持 SKIP）
  │
  ├─ *. Controller 分流   _controller is not None 时走 _run_controller_turn（v1.9.7）
  │     └─ TaskModeController 两步协议：handle_turn() → ModeAction → _run_pipeline() → on_pipeline_done()
  │
  ├─ 1. on_input        各模块预处理输入（如安全过滤）
  │     └─ 输入被清空 → blocked=True，跳到 POST_CHAT
  │
  ├─ 2. on_context       各模块增强上下文（如知识库指南、记忆指南注入）
  │
  ├─ 3. 构建 messages    system prompt + 对话历史 + 工具 schema
  │
  ├─ 4. 执行策略（可插拔）
  │     ├─ SimpleReAct（默认）：直接走 ReAct 循环
  │     └─ PlanReAct：先规划，再逐步 ReAct
  │
  │     ReAct 循环（底层引擎，无状态，不感知工具来源）：
  │
  │     接口：run_react(messages, tools_schema, tool_dispatch, *, should_continue=None)
  │       - messages:        发给 LLM 的消息列表
  │       - tools_schema:    工具 schema 列表（传入 function calling）
  │       - tool_dispatch:   工具调度函数 (name, args) → str
  │       - should_continue: 可选回调 () → str | None，每次工具调用后检查，
  │                          返回 str 则中止循环（该字符串作为中断原因），返回 None 则继续
  │
  │     ┌─────────────────────────────────────────────┐
  │     │  v2.0: abort 检查点 1（循环顶部，LLM 调用前）   │
  │     │    └─ _abort → ReactResult(status="aborted",  │
  │     │       terminal=True)                           │
  │     │                                               │
  │     │  LLM 调用（带 tools_schema）                  │
  │     │    ├─ 无 tool_calls → 直接返回文本响应         │
  │     │    └─ 有 tool_calls → 进入循环：               │
  │     │         │                                     │
  │     │         ├─ 1. 读取 LLM 返回的 tool_calls      │
  │     │         ├─ 2. 先追加 assistant tool_calls      │
  │     │         │    message（保证消息链完整）          │
  │     │         ├─ 3. 将 arguments 统一解析为 dict     │
  │     │         ├─ 4. Action: tool_dispatch(name, args)│
  │     │         ├─ 5. 工具异常统一捕获，转换为          │
  │     │         │    observation 字符串                │
  │     │         ├─ 6. 追加对应的 tool message          │
  │     │         │    ├─ 超长结果截断（max_observation_tokens）│
  │     │         │    └─ 结果作为 tool message 追加     │
  │     │         ├─ 6.5 v2.0: abort 检查点 2（工具调用后）│
  │     │         │    └─ _abort → ReactResult(aborted,  │
  │     │         │       terminal=True)                  │
  │     │         ├─ 7. should_continue 检查（若已设置）  │
  │     │         │    └─ 返回 str → 中止循环，返回       │
  │     │         │       ReactResult(status="interrupted",│
  │     │         │       reason=该字符串)                 │
  │     │         └─ 再次调用 LLM → 继续循环或返回       │
  │     │                                               │
  │     │  循环保护：                                    │
  │     │    - 最大步数（max_react_steps，-1=无限）       │
  │     │    - 重复检测（连续相同工具名+参数则中止，-1=关闭）│
  │     │    - 超时保护（react_timeout，每步独立计时）     │
  │     │    - observation 截断（max_observation_tokens） │
  │     │    - ContextWindowExceededError → 中止循环     │
  │     │    - v2.0: abort 机制（两处检查点）             │
  │     └─────────────────────────────────────────────┘
  │
  ├─ 5. on_output        各模块后处理输出（如反思评估、安全脱敏）
  │
  ├─ 6. 返回响应 + 更新对话历史
  │     注意：写入 history 的是经过 on_input 处理后的 processed，
  │     而非原始 user_input，确保安全过滤后的输入不被绕过。
  │
  └─    POST_CHAT        always-fire，所有退出路径均触发（v1.8 事件 Hook）
        data 包含 blocked / blocked_by / completed 字段
```

#### 2.1.2 ReactResult 结构化返回

`run_react()` 返回结构化结果，便于执行策略根据状态做分支处理：

```python
@dataclass
class ReactResult:
    """ReAct 循环的结构化返回结果。"""
    text: str                    # 最终文本响应
    status: Literal[
        "completed",             # 正常完成（LLM 返回无 tool_calls 的文本）
        "max_steps",             # 达到最大步数限制
        "timeout",               # 超时
        "error",                 # 错误导致中止（具体原因见 error 字段）
        "interrupted",           # 被 should_continue 回调中断（策略自定义中断条件）
        "context_overflow",      # 上下文窗口溢出
        "aborted",               # v2.0: 被 abort() 中止
    ]
    error: str | None = None     # 错误详情（status 非 completed 时）
    steps_used: int = 0          # 本次循环使用的步骤数
    reason: str | None = None    # 中断原因（status 为 interrupted 时，由 should_continue 回调返回）
    terminal: bool = False       # v2.0: 不可恢复结果标记，调用方应停止（不重试/不 replan）
```

**`terminal` 字段语义（v2.0）**：标记不可恢复的执行结果。当 `terminal=True` 时，调用方（如 PlanReAct）应终止整个执行流程，不尝试 replan 或重试。

| status | terminal | 语义 |
|--------|----------|------|
| `aborted` | `True` | 被外部 abort() 中止，不可恢复 |
| `context_overflow` | `True` | 上下文窗口溢出，不可恢复 |
| `error` / `timeout` / `max_steps` | `False` | 可能可恢复，PlanReAct 可尝试 replan |
| `completed` / `interrupted` | `False` | 正常结果 |

PlanReAct 通过 `result.terminal` 判断是否退出，与 abort 机制零耦合：

```python
# modules/reasoning/module.py
if result.terminal:
    step_i.status = "failed"
    break  # 终止整个计划执行
```

#### 2.1.3 工具注册表

core 维护一个单一的 `self._tools: dict`（扁平字典，key 为工具名，value 为工具元数据 dict），存储所有已注册工具：

```python
# 格式：{name: {"name": str, "func": callable, "description": str, "parameters": dict,
#               "tier": str, "safety_level": int, "execution_policy": ...,
#               "creator_id": str | None, "path_extractor": Callable | None,
#               "pack": str | None}}
self._tools: dict[str, dict[str, Any]] = {}
```

> **注意**：每个 tool dict 中包含 `"name"` 键（与 `_tools` 的 key 相同），便于在 `ToolExecutor` 等需要 tool dict 的场景中直接获取工具名称，无需依赖外部 key。

模块通过 `agent.register_tool()` 向 `self._tools` 注册工具。`_tools_version` 在每次 `register_tool()` / `remove_tool()` 时自增。

> **注意**：ToolsModule 内部维护自己的 `ToolRegistry` 实例（`global_registry`（模块级单例，存储 common 层工具）+ `agent_registry`）用于组织管理工具，但这些是 ToolsModule 的内部实现细节，core 层不感知。其他模块（如 MCP、Memory、RAG、Multi-Agent）直接通过 `agent.register_tool()` 将工具注册到 `self._tools`。

**`call_tool` 方法**——工具的查找、区域检查和执行：

```
agent.call_tool(name: str, args: dict) -> str:
    1. 查找工具：self._tools.get(name)
       IF 找不到:
           → 返回 "工具 '{name}' 不存在。可用工具：[tool1, tool2, ...]"
    2. PRE_TOOL_USE hook（v1.8）：
       - emit_hook(PRE_TOOL_USE, {tool_name, args, tool_info})
       - Hook 可返回 SKIP 阻止执行（data["skip_reason"] 作为拒绝消息）
       - Hook 可修改 data["args"] 改变工具参数
    3. AuthorizationEngine.evaluate()（v1.9）：
       a. 路径提取（path_extractor 或 auto 检测）
       b. zone 评估 → ZoneEvaluation（逐路径 ALLOW / CONFIRMABLE / HARD_DENY）
       c. 按 policy 决策：
          - InteractivePolicy（v1.9.0）：逐 item 确认 CONFIRMABLE，HARD_DENY 直接拒绝
          - TaskPolicy execute 阶段（v1.9.2, v1.9.8）：先匹配 task scope，再匹配 session scope（v1.9.8），未命中退回确认
          - ContinuousPolicy（v1.9.3, v1.9.9）：匹配 session scope 或直接拒绝，无交互。v1.9.9：无 seed_scopes 时默认开放 project 读写
          - v1.9.4：scope 过期/用量自动失效；AuthorizationResult 携带审计事件（SCOPE_USED/DENIED/ISSUED/REVOKED）
          - 确认通过 ConfirmRequest/ConfirmResponse 结构化接口
          - 确认等待时间不计入 react_timeout
    4. 执行工具（统一包裹两条路径）：
       - tool_executor 非 None → 路由到沙箱执行
       - 否则 tool["func"](**args) → 返回结果（转换为 str）
    5. POST_TOOL_USE hook（v1.8）：成功时触发，data 含 result/duration_ms
    6. TOOL_ERROR hook（v1.8）：异常时触发，data 含 error
```

工具未找到和区域拒绝都以 tool observation 的形式传回模型，模型可据此调整行为。

区域检查在 `call_tool()` 中统一执行，不在工具内部。路径提取通过 `path_extractor`（显式注册）或 `auto_path_extractor`（自动推断）实现，详见第 8.1 节三层区域系统。

#### 2.1.4 循环保护

防止 agent 陷入无限循环或 token 溢出，内置于 core：

**循环控制**：

| 参数 | 说明 | 默认值 | -1 语义（v2.0） |
|------|------|--------|-----------------|
| `max_react_steps` | 单次对话最多执行多少步工具调用 | 10 | 无限（continuous 模式） |
| `max_duplicate_actions` | 连续相同 action（工具名 + 参数完全一致）超过此数则中止 | 2 | 关闭重复检测（continuous 模式） |
| `react_timeout` | 单步 ReAct 循环执行时间上限（PlanReAct 每步独立计时） | 210s | N/A |

v2.0 引入 `-1` 哨兵值表示 unlimited/disabled。run_react 循环条件适配：
- `while max_react_steps == -1 or steps < max_react_steps:`
- `if max_duplicate_actions != -1 and duplicate_count >= max_duplicate_actions:`

**token 保护**（ReAct 循环内部）：

| 机制 | 触发条件 | 处理方式 | 额外 LLM 调用 |
|------|----------|----------|---------------|
| observation 截断 | 工具返回结果超过 `max_observation_tokens` | 截断到限制长度，追加"...（内容已截断）" | 无 |
| ContextWindowExceededError 兜底 | LLM 调用超出上下文窗口 | 中止循环，返回最后一次成功的 LLM 文本响应；若无则返回提示 | 无 |

observation 截断在工具返回结果后、追加到 messages 前执行。对话级压缩在 ReAct **之前**处理历史消息，两者互不干涉。

> PlanReAct 每步独立构建 messages（不带对话历史），配合 observation 截断后单步很难溢出，因此 ContextWindowExceededError 主要影响 SimpleReAct。

#### 2.1.5 上下文管理

两层机制协同工作：

| 机制 | 参数 | 触发条件 | 处理方式 |
|------|------|----------|----------|
| 轮数窗口 | `context_window_size`（默认 20） | 每轮对话后，对话历史超过 N 轮 | 丢弃最早的轮次 |
| token 压缩 | `context_compress_threshold`（默认 0.7） | 每轮对话前，token 总量超过 `max_context_tokens × 0.7` | 部分压缩（保留最近 N 轮，压缩旧对话为摘要） |

**数据模型**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `summary` | `str \| None` | 对话历史摘要，压缩时生成，独立于 history |
| `history` | `list[dict]` | 对话历史 |

**设计要点**：`summary` 不混入 `history`，trim 只操作 `history`，`summary` 独立保留。`build_messages()` 构建消息时，若 `summary` 存在，将其作为额外的 system message 拼入 messages 列表。

**完整流程**（每轮对话前后各检查一次）：

```
用户发送消息
  │
  ├─ 1. 对话前：检查 token
  │     │
  │     ├─ 估算 history 的 token 量
  │     │
  │     ├─ IF token < max_context_tokens × 0.7
  │     │     └─ 正常继续，无需压缩
  │     │
  │     └─ IF token ≥ max_context_tokens × 0.7
  │           │
  │           └─ 部分压缩（仅此一阶段）
  │                 ├─ 保留最近 compress_keep_turns 轮原文不动
  │                 ├─ 将更早的对话（含旧 summary）调用 LLM 压缩成新摘要
  │                 └─ self.summary = 新摘要, self.history = [最近 N 轮]
  │
  ├─ 2. 执行 pipeline（on_input → on_context → 执行策略 → on_output）
  │
  └─ 3. 对话后：检查轮数
        ├─ IF len(self.history) ≤ context_window_size * 2 → 正常保留
        └─ IF len(self.history) > context_window_size * 2 → 丢弃最早的轮次
```

> **注意**：没有第二阶段全量压缩，也没有对话级 ContextWindowExceededError 兜底。ContextWindowExceededError 仅在 `run_react()` 内部处理——捕获后中止循环，返回 `ReactResult(status="context_overflow")`。`chat()` 本身不捕获该异常。

#### 2.1.6 Abort 机制（v2.0）

支持外部调用方（如 ContinuousRunner）安全中止正在执行的任务。

**核心设计**：
- `agent._abort: bool` — 中止标志，`__init__` 时为 `False`
- `agent.abort()` — 公开方法，设置 `_abort = True`
- `chat()` 入口重置 `_abort = False`（唯一重置点）
- `run_react()` 在两处检查（只检查，不重置）：
  1. 外层 while 循环顶部（LLM 调用前）
  2. 内层 tool_call 循环（每个工具调用后）

**中止语义**：已完成操作保留（不回滚），当前原子操作让它完成，后续操作不再执行。

**chat() 为唯一重置点的原因**：`_run_controller_turn` 的驱动循环会多次调用 `_run_pipeline`（prepare → execute）。如果 `run_react` 在 prepare 阶段检测到 abort 后重置标志，execute 阶段会继续执行，违反 abort 语义。chat() 重置保证 abort 信号在整个 `_run_controller_turn` 期间持续有效。

**两种中止场景**：
- LLM 调用中：等 HTTP 返回后在循环顶部检查退出
- 工具执行中：等当前工具完成后在工具调用后检查退出

#### 2.1.7 关键接口

| 方法 | 说明 |
|------|------|
| `chat(user_input) -> str` | 主入口，走完整 pipeline（on_input → on_context → 执行策略 → on_output）。入口处重置 `_abort = False`。`user_input` 不应为空（空字符串在内部视为安全拦截信号） |
| `abort()` | v2.0: 信号中止当前任务。run_react 在两处检查点检测后返回 `ReactResult(status="aborted", terminal=True)` |
| `register_module(module)` | 注册模块，兼容层：先尝试 `module.on_attach(agent)`，若无则 fallback 到 `module.attach(agent)` |
| `get_module(name) / has_module(name)` | 查询模块 |
| `list_modules() -> list[str]` | 列出所有已注册模块 |
| `set_execution_strategy(strategy)` | 替换执行策略（默认 SimpleReAct） |
| `build_messages(query, context, *, include_history=True, extra_system="") -> list[dict]` | 统一消息构建：拼装 system prompt + summary + history + 当前 query |
| `run_react(messages, tools_schema, tool_dispatch, *, should_continue=None) -> ReactResult` | ReAct 循环引擎，无状态，由执行策略调用。返回结构化 ReactResult |
| `call_tool(name: str, args: dict) -> str` | 工具调用：从 `_tools` 查找 + 区域检查 + 执行 |
| `get_all_tool_schemas() -> list[dict]` | 从 `_tools` 过滤工具 schema，按 tier + 角色过滤 |
| `register_tool(name, func, description, parameters=None, tier="common", safety_level=1, execution_policy=None, creator_id=None, path_extractor=None, pack=None)` | 注册工具到 `_tools`。`path_extractor`: 路径提取回调；`pack`: 条件暴露 pack 名称 |
| `remove_tool(name)` | 从注册表移除工具，同时 `_tools_version += 1` |
| `clear_conversation()` | 清空对话历史 |
| `status() -> dict` | 当前状态 |

**`build_messages()` 参数说明**：

| 参数 | 说明 |
|------|------|
| `query` | 用户查询 |
| `context` | `on_context` 输出的上下文字符串 |
| `include_history` | 是否包含对话历史（SimpleReAct=True, PlanReAct 每步=False） |
| `extra_system` | 额外 system prompt（PlanReAct 用于注入步骤指令） |

构建逻辑：拼装 system prompt（`_build_system_prompt(context)` + extra_system）+ summary（若存在，作为额外 system message）+ history（若 `include_history=True`）+ 当前 query。

#### 2.1.8 System Prompt 构建

`_build_system_prompt(context)` 每轮调用时构建完整的 system prompt，由 3 个部分组成：

```
┌─────────────────────────────────────────────────────┐
│ 1. 角色身份                                          │
│    有 Persona → persona.to_system_prompt()           │
│    无 Persona → config.system_prompt（默认羊驼人设）  │
├─────────────────────────────────────────────────────┤
│ 2. 行为准则                                          │
│    固定文本，所有角色通用                              │
│    规范工具调用行为（结果真实不可编造、失败换方案、      │
│    不确定时询问用户、尊重权限边界）                     │
├─────────────────────────────────────────────────────┤
│ 3. 能力模块说明                                      │
│    遍历所有已注册模块，列出 name + description         │
│    格式：- {mod.name}: {mod.description}             │
│    无模块则跳过此段                                   │
└─────────────────────────────────────────────────────┘
```

此外，`on_context` 输出的动态上下文（context 参数）直接追加在末尾。

**构建规则**：
- 没有缓存机制——每轮调用 `_build_system_prompt()` 重新构建（`_tools_version` 字段存在但未用于缓存）
- 模块说明只列出模块名称和描述，不包含按 tier 分组的工具列表——工具 schema 通过 function calling 的 `tools` 参数单独传给 LLM
- 无内容的段落跳过不显示
- 消息构建逻辑统一在 `build_messages()` 方法中，执行策略不应重复实现消息拼装

### 2.2 Config（`core/config.py`）

统一配置，所有参数集中管理：

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 模型 | `model` | 自动检测 | DeepSeek > OpenAI > Anthropic > Ollama |
| 模型 | `max_context_tokens` | 自动获取 | 模型上下文窗口大小，通过 `litellm.get_max_tokens()` 自动获取，用户可手动覆盖 |
| 模型 | `api_retry_count` | 1 | LLM API 调用失败重试次数 |
| Agent | `system_prompt` | 羊驼人设 | 无 Persona 时的默认身份 |
| Agent | `context_compress_threshold` | 0.7 | 对话 token 占 `max_context_tokens` 的比例，超过则触发压缩 |
| Agent | `compress_keep_turns` | 3 | 压缩时保留最近 N 轮原文不压缩 |
| Agent | `max_react_steps` | 10 | ReAct 循环最大步数 |
| Agent | `max_duplicate_actions` | 2 | 连续相同 action 超过此数则中止 |
| Agent | `react_timeout` | 210s | 单步 ReAct 循环执行时间上限（PlanReAct 每步独立计时） |
| Agent | `max_observation_tokens` | 2000 | 单次工具返回结果的最大 token 数，超过则截断 |
| Agent | `context_window_size` | 20 | 对话上下文保留轮数，超出则丢弃最早的轮次 |
| 记忆 | `memory_mode` | "off" | 记忆模式："off" / "autonomous" / "hybrid" |
| 检索层 | `retrieval_persist_dir` | `{BASE_DIR}/data/chroma` | 检索层持久化根目录（`chroma_dir` 为 backward compat 别名） |
| RAG | `rag_top_k` / `chunk_size` | 3 / 500 | 检索数量 / 分块大小 |
| 角色 | `persona_file` | `data/personas.json` | 角色定义存储路径 |
| 工具 | `agent_tools_dir` | `data/agent_tools` | 自定义工具存储目录 |
| 规划 | `max_plan_adjustments` | 7 | PlanReAct 执行期间计划最大调整次数 |
| 反思 | `reflection_enabled` | false | 反思评估开关 |
| 反思 | `reflection_score_threshold` | 7.0 | 低于此分触发教训保存 |
| 安全 | `permission_level` | 1 | 无 Persona 时的兜底权限等级 |
| 输出 | `output_dir` | `output/` | 文件输出目录 |

模型检测优先级：环境变量 `MODEL_NAME` > API Key 检测 > Ollama 回退。

### 2.3 LLMClient（`core/llm.py`）

基于 LiteLLM 的统一 LLM 接口，所有 LLM 调用都通过此类。

#### 核心方法

| 方法 | 说明 | 调用方 |
|------|------|--------|
| `chat(messages, tools, tool_choice, temperature, max_tokens, response_format, timeout)` | 底层调用，返回完整 response 对象，支持 function calling。`timeout` 为可选参数（秒） | ReAct 循环（core） |
| `ask(prompt, system="", **kwargs) -> str` | 单轮问答，返回纯文本。无显式 `timeout` 参数，如需超时可通过 `**kwargs` 透传 | Persona 创建、Reflection 根因分析、上下文压缩等 |
| `ask_json(prompt, system="", **kwargs) -> dict` | 单轮问答，返回解析后的 JSON。无显式 `timeout` 参数，如需超时可通过 `**kwargs` 透传 | Planning 任务分解、Reflection 打分等 |

#### token 计数

| 方法 | 说明 |
|------|------|
| `count_tokens(messages: list[dict] \| str) -> int` | 估算 messages 列表的总 token 数（使用 `litellm.token_counter()`，不可用时按 1 字符 ≈ 1 token 估算） |
| `max_context_tokens -> int` | 当前模型的上下文窗口大小（property，从 LiteLLM 自动获取） |

#### 错误处理

- API 调用失败时按 `api_retry_count` 重试（指数退避，默认 1 次重试，共 2 次调用）
- 捕获 `litellm.ContextWindowExceededError`：仅在 `run_react()` 内部处理 → 中止循环，返回 `ReactResult(status="context_overflow")`
- JSON 解析失败时尝试从 Markdown 代码块中提取

### 2.4 Persona / PersonaManager（`core/persona.py`）

角色系统，支持创建多个不同身份的 LlamAgent。

**创建流程**：用户只需提供 `name` + 一句角色描述（如"前端开发专家"），系统调用模型自动扩展为完整的 system prompt（身份、擅长领域、行为风格、回答方式等），存入 Persona。

**Persona 数据结构**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 显示名称（如"代码羊驼"） |
| `role_description` | str | 用户输入的角色描述（如"前端开发专家"） |
| `system_prompt` | str | 模型自动扩展生成的完整人设 prompt |
| `role` | str | 权限角色："admin" 或 "user" |
| `permission_level` | int \| None | 权限等级（None 表示自动设置：admin=3, user=1，可手动调整） |
| `persona_id` | str | 唯一 ID（自动从 name 通过 `_slugify()` 生成：英文名 → 下划线连接小写，非 ASCII 名如中文 → MD5 hash 前缀 `persona_`） |
| `created_at` | str | 创建时间（ISO 格式，自动生成） |

**权限区分**：
- 管理员的 system prompt 额外包含管理员权限说明
- 普通角色的 system prompt 只包含自身身份和可用工具说明

**PersonaManager**：CRUD 操作 + JSON 文件持久化。`create()` 方法接受可选的 `llm` 参数（`LLMClient` 实例），用于调用模型扩展生成完整 system prompt。

### 2.5 核心层调用关系

```
  人类用户（创建阶段）
      │
      │ 提供 name + 角色描述
      ↓
  PersonaManager ──llm.ask()──→ LLMClient ──→ 生成完整 system prompt
      │
      └─ 存入 Persona 对象，持久化到 JSON


┌───────────────────── LlamAgent ────────────────────┐
│                                                       │
│  初始化（一次性）                                      │
│  ┌────────┐  注入配置   ┌────────────┐                │
│  │ Config  │──────────→│ 各组件      │                │
│  └────────┘            └────────────┘                │
│  ┌────────┐  缓存                                     │
│  │Persona │──────→ system_prompt                      │
│  └────────┘                                           │
│                                                       │
│  每轮对话                                              │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│    人类用户                                           │
│  │   │ ↑                                          │  │
│      │ │ 返回响应                                     │
│  │   │ 提问 / 请求                                │  │
│      ↓ │                                              │
│  │ ┌────────────┐                                 │  │
│    │ 构建        │ ← system_prompt（已缓存）           │
│  │ │ messages    │ ← summary + history（对话历史）  │  │
│    └─────┬──────┘                                     │
│  │       ↓                                        │  │
│    ┌────────────┐                                     │
│  │ │ 执行策略    │  chat()    ┌──────────┐         │  │
│    │ SimpleReAct │←─────────→│LLMClient │              │
│  │ │ / PlanReAct │  response  │          │         │  │
│    │ (含 ReAct)  │           │          │              │
│  │ └─────┬──────┘            │          │         │  │
│          │           ask()   │          │              │
│  │ ┌─────┴──────┐←─────────→│          │         │  │
│    │ 上下文压缩  │            │          │              │
│  │ └────────────┘count_tokens│          │         │  │
│                              └──────────┘              │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## 3. 模块系统

### 3.1 Module 基类

所有功能模块继承自 `Module` 基类（定义在 `core/agent.py`）。基类提供空实现，模块按需重写：

```python
class Module:
    name: str  # 模块名称，用于 has_module() 查询

    def on_attach(self, agent):    pass   # 注册时调用一次
    def on_shutdown(self):          pass   # agent 退出时调用
    def on_input(self, user_input) -> str:  return user_input
    def on_context(self, query, context) -> str:  return context
    def on_output(self, response) -> str:  return response
```

### 3.2 Pipeline Callback（洋葱模型）

**生命周期回调**（各调用一次）：

| 回调 | 触发时机 | 用途举例 |
|------|----------|----------|
| `on_attach(agent)` | `register_module()` 时 | 注册工具、注入执行策略、初始化存储、注入安全回调 |
| `on_shutdown()` | agent 退出时 | 关闭连接、释放资源 |

**Pipeline Callback**（每轮对话调用）：

| 回调 | 触发时机 | 用途举例 |
|------|----------|----------|
| `on_input(user_input) -> str` | ReAct 循环之前 | 安全过滤、输入改写 |
| `on_context(query, context) -> str` | 构建 messages 时 | 知识库指南注入、记忆指南注入、技能 playbook 注入 |
| `on_output(response) -> str` | ReAct 循环之后 | 反思评估、记忆保存、安全脱敏 |

**完整生命周期**：

```
on_attach → [on_input → on_context → 执行策略 → on_output]×N → on_shutdown
```

### 3.3 执行顺序（洋葱模型）

**推荐注册顺序**：

```
safety → sandbox → tools → job → rag → memory → skill → reflection → planning → mcp → multi_agent → child_agent
```

**执行规则**：
- `on_input` / `on_context`：按注册顺序**正序**执行
- `on_output` / `on_shutdown`：按注册顺序**逆序**执行

```
请求进入：  safety.on_input → reflection.on_input        （正序）
上下文增强：rag.on_context → memory.on_context → skill.on_context → reflection.on_context  （正序）
输出处理：  reflection.on_output → memory.on_output(hybrid) → safety.on_output  （逆序）
关闭清理：  mcp.on_shutdown                              （逆序）
```

理由：形成对称的洋葱结构——safety 最先过滤输入 / 最后脱敏输出；tools 紧跟其后提供基础设施；rag 和 memory 先增强上下文；skill 在 memory 之后注入任务 playbook（此时已有知识库和记忆上下文，skill 匹配可据此做更精准的判断）；reflection 在 planning 之前注册，planning 的 `on_attach` 可通过 `has_module("reflection")` 检查并注入反思引擎。

### 3.4 事件 Hook（v1.8）

Pipeline Callback 是**轮级**的（每轮对话触发一次），事件 Hook 是**工具级**和**生命周期级**的（每次工具调用触发）。两者共存。

| 层级 | Hook 类型 | 触发频率 | 用途 |
|------|-----------|----------|------|
| 轮级 | `on_input` / `on_context` / `on_output` | 每轮 1 次 | 模块逻辑（安全、上下文、记忆） |
| 工具级 | `PRE_TOOL_USE` / `POST_TOOL_USE` / `TOOL_ERROR` | 每次工具调用 | 审计、拦截、通知 |
| 生命周期 | `SESSION_START` / `SESSION_END` / `PRE_CHAT` / `POST_CHAT` | 会话/轮级 | 生命周期观察 |

事件 Hook 通过 `agent.register_hook()` 注册（代码 callable 或 YAML shell command）。只有 `PRE_TOOL_USE` 支持 SKIP 阻止执行。

处理器类型：`CallableHandler`（Python callable）和 `ShellHandler`（subprocess，通过 `$HOOK_*` 环境变量传递数据）。

详见 `docs/modules/hooks/overview.md`。

### 3.5 Task Mode（v1.9.6, v1.9.8, v1.9.9, v2.0）

Task mode 是一种前置协议层，在执行策略之外，用于复杂任务的前置摸底和授权。

v1.9.6 将驱动循环与控制逻辑分离：agent 只负责驱动，`TaskModeController` 负责状态判断和合同生成。

v1.9.9 新增 config-driven 模式初始化：`agent.__init__` 读取 `config.authorization_mode`，非 interactive 时自动调用 `set_mode()`。支持 YAML 配置和环境变量 `AUTHORIZATION_MODE`。

v1.9.8 引入 session scopes 共享授权：task 模式复用 continuous 模式的 seed_scopes 机制，支持预授权。

v2.0 新增：prepare-only 工具（open_questions 收集）、clarification_turns 硬限制、mode-aware 配置。

#### 两种执行路径

**有 session scopes（auto_execute=True）**：跳过 prepare/contract/confirm，直接执行。

```
set_mode("task") → engine 加载 seed_scopes 或询问用户
  → session_scopes 非空 → controller.auto_execute = True

chat() → _run_controller_turn(user_input)
  → controller.handle_turn() → ModeAction(run_execute)  # 跳过 prepare
  → _run_pipeline(mode="normal")
  → call_tool → TaskPolicy: task_scopes → session_scopes → fallback confirm
  → controller.on_pipeline_done() → reply + 清理 task_scopes
```

Session scopes 来源：
1. Config 的 seed_scopes — `_load_seed_scopes()` 自动加载（与 continuous 同一方法）
2. Engine 在 set_mode 时通过 `ConfirmRequest(kind="session_authorize")` 询问用户

**无 session scopes（auto_execute=False）**：完整 prepare → contract → confirm → execute 流程。

```
chat() → _controller is not None → _run_controller_turn(user_input)
  驱动循环（agent 侧）：
    1. action = controller.handle_turn(user_input) → ModeAction
    2. 处理 action.authorization_update（如有）
    3. 根据 action.kind 分派：
       - run_prepare  → _run_pipeline(mode="prepare")，再调 on_pipeline_done
       - run_execute  → _run_pipeline(mode="normal")，再调 on_pipeline_done
       - await_user   → 退出循环，展示合同文本给用户
       - reply        → 退出循环，返回最终结果
       - cancel       → 退出循环，任务取消
    4. action = controller.on_pipeline_done(action, outcome) → 新的 ModeAction
       - pending_scopes 通过 PipelineOutcome.metadata 传回 controller
       - controller 内部聚合 scopes → 生成 TaskContract
    5. 循环退出后：agent 手写 history
```

#### Controller 两步协议

- `handle_turn(user_input) -> ModeAction`：根据当前 phase 决定下一步动作
- `on_pipeline_done(action, outcome) -> ModeAction`：pipeline 执行完成后回调，收集 pending_scopes 并推进状态

#### Prepare-only 工具与 open_questions（v2.0）

prepare 阶段注入专用工具 `_report_question`，供 LLM 报告不确定项：

```python
# agent._register_prepare_tools()
agent.register_tool(
    name="_report_question",
    func=lambda question: self._open_questions_buffer.append(question) or f"Question recorded: {question}",
    description="Report an open question or uncertainty during task planning.",
    tier="default", safety_level=0,
)
```

**生命周期**：
1. `_run_pipeline(mode="prepare")` 入口：清空 `_open_questions_buffer`，调用 `_register_prepare_tools()` 注册工具
2. LLM 在 dry-run 期间调用 `_report_question(question)` → 问题追加到 buffer
3. pipeline 结束：buffer 内容写入 `outcome.metadata["open_questions"]`，调用 `_unregister_prepare_tools()` 移除工具

**Controller 处理**：`_on_prepare_done()` 从 `outcome.metadata` 提取 `open_questions`，写入 `TaskContract.open_questions` 字段。合同展示时在 scopes 列表后显示：

```
Open questions:
  ? <question 1>
  ? <question 2>
```

#### clarification_turns 硬限制（v2.0）

Controller 内置 `MAX_CLARIFICATION_TURNS = 3` 常量，限制 re-prepare 轮次。

awaiting_confirmation 状态下，用户输入既非 confirm 也非 cancel 时，触发 re-prepare。每次 re-prepare 时 `state.clarification_turns += 1`。超过上限后，controller 返回 `ModeAction(kind="await_user")`，强制用户做 yes/no 决定：

```
Maximum clarification rounds (3) reached. Reply 'yes' to execute with the current plan, or 'no' to cancel.
```

`reset()` 时 `clarification_turns` 归零。

#### Controlled dry-run 规则（prepare 阶段）

- `action="read"` → 真实执行（帮助 LLM 了解上下文）
- `action="write"/"execute"` → 不执行，记录为 pending scope
- `ask_user` → 始终允许（收集用户信息）
- `HARD_DENY` → 直接拒绝
- `_report_question` → 直接执行（v2.0，prepare-only 工具，safety_level=0）

#### Scope 匹配优先级（execute 阶段）

TaskPolicy._decide_execute 将两个 scope 列表拼接后交给 `_find_matching_scope()`，第一个匹配的 scope 被使用（v1.9.8）：

```python
scopes = task_scopes[task_id] + session_scopes  # task scopes 在前，优先被匹配
```

1. `task_scopes[task_id]` — 当前任务的 contract 批准的 scope（列表前段，优先匹配）
2. `session_scopes` — seed_scopes 或用户预授权的 scope（列表后段，兜底匹配）
3. 都不匹配 → fallback 到 confirm

#### Session scope 生命周期

- 任务完成/取消：只清理 task_scopes，session_scopes 保留
- 模式切换（set_mode）：`_clear_all_scopes` 清理所有

#### 循环保护

驱动循环有 `_MAX_MODE_STEPS` 上限（默认 10），防止 controller 状态机无限循环。正常流程下 auto_execute 路径 2 步完成（execute + reply），prepare 路径 3-4 步完成。耗尽时返回错误消息。

Task mode 不替代 PlanReAct。合同确认后，执行阶段内部仍可使用 SimpleReAct 或 PlanReAct。

### 3.6 Mode-Aware 配置（v2.0）

`set_mode()` 切换模式时自动调整效率参数。`_MODE_DEFAULTS` 作为 agent.py 的类级常量：

```python
_MODE_KEYS = {"max_react_steps", "max_duplicate_actions", "react_timeout",
              "max_observation_tokens"}

_MODE_DEFAULTS = {
    "task":       {"max_react_steps": 50, "react_timeout": 600,
                   "max_duplicate_actions": 5, "max_observation_tokens": 5000},
    "continuous": {"max_react_steps": -1, "react_timeout": 600,
                   "max_duplicate_actions": -1, "max_observation_tokens": 10000},
}
```

**三种模式的效率参数对比**：

| 参数 | Interactive（默认） | Task | Continuous |
|------|-------------------|------|-----------|
| max_react_steps | 10 | 50 | -1（无限） |
| max_duplicate_actions | 2 | 5 | -1（关闭） |
| react_timeout | 210s | 600s | 600s |
| max_observation_tokens | 2000 | 5000 | 10000 |

**Config 恢复逻辑**：
- `__init__` 中快照 interactive 配置：`self._interactive_config = {k: getattr(config, k) for k in _MODE_KEYS}`
- `set_mode("interactive")` → 恢复 snapshot（保留用户 YAML 自定义值，不丢失）
- `set_mode("task"/"continuous")` → 应用 `_MODE_DEFAULTS`
- 异常路径（set_mode 失败） → 恢复 snapshot

**`max_plan_adjustments` 不纳入 `_MODE_DEFAULTS`**：PlanReAct 在 `on_attach` 时快照 config 值，set_mode 之后修改 config 对 PlanReAct 无效。

### 3.7 ContinuousRunner（v2.0，`core/runner.py`）

ContinuousRunner 是外部组件，通过 Trigger 驱动 `agent.chat()`。Agent 不知道 Runner 的存在。

**依赖方向**：`Runner → Agent → Engine`。Trigger 不引用 Agent。Agent 不引用 Runner。

```
                  poll()           chat()
  Trigger -------> Runner -------> Agent -------> Engine
  (不知 Agent)    (外部驱动)      (不知 Runner)  (授权策略)
```

#### ContinuousRunner

```python
class ContinuousRunner:
    def __init__(self, agent, triggers: list[Trigger], *,
                 poll_interval: float = 1.0,
                 task_timeout: float = 0,      # 0 = 无应用超时
                 on_timeout: str | Callable = "abort"):
        ...

    def run(self) -> None:
        """主循环，阻塞直到 stop()。单个任务失败不杀主循环。"""

    def _run_task(self, task_input: str, trigger: Trigger) -> None:
        """执行单个任务，可选 watchdog 超时，记录 task_log。"""

    def stop(self) -> None:
        """从任意线程调用，优雅停止。同时调 agent.abort() 中止当前任务。"""

    def get_log(self) -> list[TaskLogEntry]:
        """返回 task_log 副本（线程安全）。"""

    def clear_log(self) -> None:
        """清空 task_log。"""
```

**主循环行为**：
- 轮询所有 trigger，有输入则调 `_run_task(task_input, trigger)`
- `_run_task` 记录 `TaskLogEntry` 到 `task_log`（trigger_type, input, output, status, duration）
- `_run_task` 异常时 `try/except` 捕获并 log，记录 status="error"，不退出主循环
- `_stopped.wait(poll_interval)` 实现可中断的等待

**Task Log（v2.0.2）**：
- `task_log: list[TaskLogEntry]`：内存中的结构化执行日志
- `get_log()` 返回 copy（线程安全），`clear_log()` 清空
- `TaskLogEntry` 纯 dataclass，不引用 agent/engine

**task_timeout + watchdog**：
- `task_timeout > 0` 时，启动 `threading.Timer` 作为 watchdog
- 超时触发 `on_timeout` 动作：默认 `"abort"` 调 `agent.abort()`；callable 自定义行为（记日志、发告警、降级等）
- 任务完成后 `timer.cancel()` 取消 watchdog

#### Trigger ABC 与内置实现

```python
class Trigger(ABC):
    """触发源：产出输入字符串，不知道 agent 的存在。"""
    @abstractmethod
    def poll(self) -> str | None: ...
```

| Trigger | 用途 | 实现 |
|---------|------|------|
| `TimerTrigger(interval, message)` | 定时执行固定任务 | `time.time() - last_fire >= interval` 时返回 message |
| `FileTrigger(watch_dir, message_template)` | 文件变化触发 | `os.listdir()` diff 检测新增文件，首次 poll 为快照不触发 |

**TimerTrigger** 首次 poll 初始化时间戳但不触发，避免启动即执行。
**FileTrigger** 首次 poll 快照当前目录内容但不触发，后续检测新增文件。只检测新增，不检测修改或删除。

---

## 4. 工具系统

### 4.1 四层工具体系与权限矩阵

| 层级 | 说明 | 注册位置 | 谁能看见 | 谁能调用 | 谁能创建/删除 |
|------|------|----------|----------|----------|---------------|
| `default` | 核心工具（元工具 + 记忆 + RAG + 协作 + MCP 桥接） | `agent_registry` | 所有角色 | 所有角色 | 仅系统 |
| `common` | 通用工具（预装 + 管理员创建） | `global_registry` | 所有角色 | 所有角色 | 预装：仅系统；通用：仅管理员 |
| `admin` | 管理员专属工具 | `agent_registry` | 仅管理员 | 仅管理员 | 仅系统 |
| `agent` | 自定义工具（角色自己创建） | `agent_registry` | 仅创建者 | 仅创建者 | 创建者自己 |

**可见性 = 可用性**：`get_all_tool_schemas()` 按 tier + 角色过滤后传给 LLM。能看到的工具就能调用，看不到的就调不了。tier 是唯一的访问控制维度。

### 4.2 平台预装工具（common 层）

所有角色可见可用：

| 工具 | 功能 | safety_level | path_extractor |
|------|------|------|------|
| `ask_user(question, choices?)` | 向用户提问获取信息（v1.8.2），tier="default"，始终可见 | 1 | 无 |
| `web_search(query)` | 真实网络搜索（v1.8.1，可切换后端：DuckDuckGo / SerpAPI / Tavily），pack="web" | 1 | 无 |
| `web_fetch(url)` | 抓取指定 URL 的页面内容 | 1 | 无 |
| `read_file(filename)` | 读文件（三层区域：playground/项目内直接读，外部需确认） | 1 | `lambda args: [args.get("filename")]` |
| `write_file(filename, content)` | 写文件（三层区域：playground 内直接写，项目内需确认，外部禁止） | 2 | `lambda args: [args.get("filename")]` |

管理员专属（tier=admin）：

| 工具 | 功能 | safety_level | path_extractor |
|------|------|------|------|
| `execute_command(command)` | 执行 shell 命令（shell=True，恢复完整 shell 能力） | 2 | 命令字符串路径扫描函数（扫描命令中的绝对路径） |

> **注意**：read_file、write_file、execute_command 已在 v1.5 中删除，由 read_files、write_files、apply_patch、start_job 替代。

### 4.3 核心工具（default 层）

所有角色可用，各模块在 `on_attach` 时注册：

| 工具 | 来源模块 | 功能 | safety_level |
|------|----------|------|------|
| `create_tool(name, description, code)` | tools | 创建自定义工具（区域系统自动保护） | 2 |
| `list_my_tools()` | tools | 查看自己的自定义工具 | 1 |
| `delete_tool(name)` | tools | 删除自定义工具 | 2 |
| `query_toolbox()` | tools | 查看通用工具箱 | 1 |
| `save_memory(content, category)` | memory | 保存重要信息到长期记忆 | 2 |
| `recall_memory(query)` | memory | 从长期记忆中语义检索 | 1 |
| `search_knowledge(query)` | rag | 从本地知识库检索相关文档 | 1 |
| `list_agents()` | multi_agent | 查看可用协作角色 | 1 |
| `create_agent(name, role_description)` | multi_agent | 创建临时协作角色 | 2 |
| `delegate(task, agent_name)` | multi_agent | 将子任务委派给指定角色执行 | 2 |

> MCP 桥接工具也注册为 default 层，具体工具取决于连接的 MCP Server。

### 4.4 管理员专属工具（admin 层）

| 工具 | 功能 | safety_level |
|------|------|------|
| `create_common_tool(name, description, code)` | 创建通用工具，所有角色可用 | 2 |
| `list_all_agent_tools()` | 查看所有角色的自定义工具 | 1 |
| `promote_tool(persona_id, tool_name)` | 提升自定义工具为通用工具 | 2 |
| `execute_command(command)` | 执行 shell 命令（shell=True） | 2 |

> **注意**：read_file、write_file、execute_command 已在 v1.5 中删除，由 read_files、write_files、apply_patch、start_job 替代。

### 4.5 工具持久化

| 层级 | 存储文件 | 说明 |
|------|----------|------|
| `common`（管理员创建） | `__common__.json` | 管理员通过 `create_common_tool` 创建的通用工具 |
| `agent` | `{persona_id}.json` | 各角色自己创建的自定义工具 |

存储目录由 `config.agent_tools_dir` 配置。加载时动态编译为可调用函数。

### 4.6 call_tool 调用流

```
LLM 决定调用工具
    │
    ▼
run_react 解析 tool_calls
    │
    ▼
tool_dispatch(name, args)
    │
    ├─ PlanReAct 步骤内：name=="replan" → replan_closure（不经 call_tool）
    │
    └─ 其他工具 → agent.call_tool(name, args)
        │
        ├─ 1. 查找工具：_tools[name]，不存在 → 返回错误字符串
        │
        ├─ 2. 区域检查（v1.3）：
        │     ├─ path_extractor / auto_path_extractor → 提取路径
        │     ├─ realpath() → 判断区域（playground / 项目 / 外部）
        │     ├─ 结合 safety_level 决定：直接执行 / 暂停确认 / 禁止
        │     └─ 需确认 → confirm_handler（等待时间不计入 react_timeout）
        │
        ├─ 3. 沙箱分发（tool_executor 非 None 时）→ 路由到沙箱
        │
        └─ 4. 执行：tool["func"](**args) → str
```

---

## 5. 执行策略

### 5.1 ExecutionStrategy 接口

core 的执行阶段采用可插拔的策略模式（Strategy Pattern），取代已废弃的 `on_execute` callback：

```
on_input → on_context → 【执行策略】 → on_output
```

**策略接口**：

| 方法 | 说明 |
|------|------|
| `execute(query, context, agent) -> str` | 执行用户请求，返回响应 |

策略的职责：每个策略负责组装 `tools_schema` 和 `tool_dispatch`，传给 `run_react()`。ReAct 引擎本身不感知工具来源，只做循环控制。

### 5.2 SimpleReAct（默认策略）

- `tools_schema` = `agent.get_all_tool_schemas()`（从 `_tools` 按 tier + 角色过滤）
- `tool_dispatch` = `agent.call_tool`（走注册表查找 + 权限检查）
- 等同于原来的行为：LLM 无 tool_calls 则直接回复，有则循环执行

```
SimpleReAct.execute(query, context, agent):
    tools_schema = agent.get_all_tool_schemas()
    tool_dispatch = agent.call_tool
    messages = agent.build_messages(query, context)
    result = agent.run_react(messages, tools_schema, tool_dispatch)
    return result.text
```

### 5.3 PlanReAct（规划策略）

由 Planning 模块在 `on_attach()` 时通过 `agent.set_execution_strategy()` 注入。

#### 5.3.1 复杂度路由

每次请求先做复杂度判断（一次 LLM 调用）：
- 简单任务（闲聊、单步操作）→ 直接走 ReAct 循环（与 SimpleReAct 一致）
- 复杂任务（多步骤、多工具协作）→ 进入规划流程

#### 5.3.2 DAG 规划

`TaskPlanner.plan()` 将复杂任务拆分为 3~8 个可执行步骤：

**步骤数据结构**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `step_id` | str | 唯一标识（如 "s1", "s2"），replan 不可修改已有 step_id |
| `order` | int | 显示顺序（1, 2, 3...），replan 可重排 |
| `action` | str | 具体要做什么 |
| `tool` | str \| null | 需要使用的工具 |
| `expected_output` | str | 预期产出 |
| `depends_on` | list[str] | 依赖的前序步骤的 step_id |
| `status` | str | `pending` / `running` / `completed` / `failed` / `skipped` |
| `result` | str \| null | 执行结果 |

**计划合法性校验**（`plan()` 或 `replan()` 返回后必须验证）：

1. `depends_on` 引用的 step_id 必须存在于当前 steps 中
2. 不允许自依赖
3. 必须是 DAG（有向无环图）——不允许循环依赖
4. 至少存在一个入口步骤（depends_on 为空）
5. 不允许重复的 step_id

不合法的计划：raise `ValueError`，`PlanReAct.execute()` 捕获后回退到简单执行路径。

#### 5.3.3 完整执行流程

```
PlanReAct.execute(query, context, agent)
  │
  ├─ 1. 复杂度判断
  │     ├─ 简单任务 → 直接走 ReAct 循环 → END
  │     └─ 复杂任务 → 继续 ↓
  │
  ├─ 2. 生成计划（TaskPlanner.plan）
  │
  ├─ 3. 初始化
  │     ├─ adjustment_count = 0
  │     ├─ interrupt_flag = False（nonlocal 变量）
  │     ├─ should_continue = lambda: "replanned" if interrupt_flag else None
  │     └─ replan_tool = 创建闭包
  │
  ├─ 4. 逐步执行（Python while 循环：存在 pending 步骤时继续）
  │     │
  │     ├─ 4a. 处理可跳过的步骤（依赖中有 failed/skipped → skip）
  │     ├─ 4b. 找 ready steps（依赖全部 completed）→ 按 order 选第一个
  │     │       无 ready step 但有 pending → 判定死锁 → 触发 replan 或中止
  │     ├─ 4c. 构建步骤上下文
  │     ├─ 4d. step_i.status = "running"
  │     ├─ 4e. 组装工具集 + 调用 ReAct
  │     │     interrupt_flag = False（每步开始前重置）
  │     │     tools_schema = agent.get_all_tool_schemas() + [replan_schema]
  │     │     step_messages = agent.build_messages(query, context,
  │     │                       include_history=False, extra_system=step_prompt)
  │     │     result = agent.run_react(step_messages, tools_schema,
  │     │                              tool_dispatch, should_continue=should_continue)
  │     │
  │     └─ 4f. 判断步骤结果（基于 ReactResult.status）
  │           ├─ completed → 记录结果，continue
  │           ├─ interrupted → 被 replan 中断，标记 failed，刷新 context，continue
  │           └─ max_steps/timeout/error/context_overflow → 标记 failed
  │                 ├─ adjustment_count < max → replan → continue
  │                 └─ adjustment_count >= max → 中止执行
  │
  ├─ 5. 结果汇总（llm.ask，不带工具）
  │
  └─ 6. 质量评估（可选，需挂载 Reflection 模块且开启）
        ├─ score ≥ threshold → 通过
        └─ score < threshold → replan 补充步骤 → 回到步骤 4
```

#### 5.3.4 三种 replan 路径

计划调整有三条路径，底层都调用 `TaskPlanner.replan()`，共享 `max_plan_adjustments` 计数器：

| | 模型主动 replan | 失败自动 replan | 质量驱动 replan |
|------|------|------|------|
| **触发者** | 模型（调用 `replan` 工具） | Python while 循环 | PlanReAct 质量评估 |
| **触发时机** | 步骤执行中 | 步骤执行后 | 所有步骤完成后 |
| **当前步骤** | 立即中止（`interrupted`） | 已结束（`failed`） | 追加补充步骤 |
| **前提条件** | 无 | 无 | 需挂载 Reflection 模块 |
| **典型场景** | "API 已废弃，换方案" | "工具超时" | "报告缺关键数据" |

**replan 闭包实现**：

```
replan_tool = 闭包 {
    持有引用：steps[]、TaskPlanner、adjustment_count、nonlocal interrupt_flag

    replan(feedback: str) → str:
        IF adjustment_count >= max_plan_adjustments:
            return "已达最大调整次数"
        adjustment_count += 1
        replan_result = TaskPlanner.replan(steps, completed, feedback)
        steps[:] = replan_result["steps"]
        interrupt_flag = True     # 通知 should_continue 中断当前 run_react
        return "计划已调整，剩余步骤已更新"
}
```

关键约束：
- `replan` 只修改数据，**不触发执行**——执行权始终在 while 循环
- `replan` 内部的 `TaskPlanner.replan()` 是普通 LLM 调用（`ask_json`），不带 tools，不会递归进入 ReAct
- 三条路径共享同一个 `adjustment_count`

#### 5.3.5 策略内部工具

策略可以向 `run_react()` 注入额外工具。这些工具不进注册表，不出现在 system prompt 的工具说明文本中，但通过 `tools_schema` 参数传入 LLM function calling，通过 `tool_dispatch` 处理调用：

| 工具来源 | 生命周期 | 经过 call_tool？ | 权限检查 |
|----------|----------|-----------------|----------|
| 注册表工具（四层体系） | 持久存在 | 是 | 区域系统检查 |
| 策略内部工具（如 `replan`） | 策略执行期间 | 否，tool_dispatch 直接调用 | 无需（受信代码注入） |

#### 5.3.6 处理 interrupted 状态

当 `run_react()` 返回 `ReactResult(status="interrupted")` 时，core 不关心中断原因——它只是通过 `should_continue` 回调检测到策略要求中止循环，并将回调返回的字符串透传到 `ReactResult.reason`。

PlanReAct 的处理策略：检查 `result.reason`，若为 `"replanned"` 则重新调用 `on_context` 刷新动态上下文，再从更新后的计划中选择下一个 ready step 继续执行。上下文刷新是 PlanReAct 的策略决策，不是 core 的机制。

---


---

## 6. 功能模块

LlamAgent 提供 12 个可插拔模块 + 1 个共享检索服务包。各模块通过 callback pipeline 与 Agent 交互，互不依赖。

详细文档参见 `docs/modules/<name>/overview.md`（功能概述）和 `api.md`（接口参考）。

### 模块一览

| 模块 | 目录 | 回调 | 核心职责 | 详细文档 |
|------|------|------|---------|---------|
| **Tools** | `modules/tools/` | on_input, on_context, on_shutdown | 工具注册（四层体系）、workspace 文件操作、project sync、pack 条件暴露 | [overview](modules/tools/overview.md) · [api](modules/tools/api.md) |
| **Memory** | `modules/memory/` | on_context, on_output | 结构化事实记忆、读写解耦、自动 recall、去重合并 | [overview](modules/memory/overview.md) · [api](modules/memory/api.md) |
| **RAG** | `modules/rag/` | on_context | Agentic RAG、Hybrid Search（vector + BM25 + rerank）、结构感知分块 | [overview](modules/rag/overview.md) · [api](modules/rag/api.md) |
| **Job** | `modules/job/` | on_attach, on_shutdown | 命令执行生命周期（start/inspect/wait/cancel）、依赖 SandboxModule | [overview](modules/job/overview.md) · [api](modules/job/api.md) |
| **Skill** | `modules/skill/` | on_input, on_context | 任务 playbook 注入、三级匹配（命令/tag/LLM）、pack 触发 | [overview](modules/skill/overview.md) · [api](modules/skill/api.md) |
| **Reasoning** | `modules/reasoning/` | on_attach | SimpleReAct / PlanReAct 执行策略、复杂度自动判断、DAG 任务分解 | [overview](modules/reasoning/overview.md) · [api](modules/reasoning/api.md) |
| **Reflection** | `modules/reflection/` | on_input, on_context, on_output | 质量评估、教训管理、质量驱动 replan | [overview](modules/reflection/overview.md) · [api](modules/reflection/api.md) |
| **Safety** | `modules/safety/` | on_input, on_output, on_shutdown | 输入过滤（注入检测）+ 输出脱敏（API key 遮蔽） | [overview](modules/safety/overview.md) · [api](modules/safety/api.md) |
| **Sandbox** | `modules/sandbox/` | on_attach, on_shutdown | 工具执行隔离、ExecutionPolicy、LocalProcessBackend | [overview](modules/sandbox/overview.md) · [api](modules/sandbox/api.md) |
| **Child Agent** | `modules/child_agent/` | on_attach | 受控子 Agent（预算、工具白名单、TaskBoard） | [overview](modules/child_agent/overview.md) · [api](modules/child_agent/api.md) |
| **Multi-Agent** | `modules/multi_agent/` | on_attach | 轻量角色协作（delegate 一次性调用） | [overview](modules/multi_agent/overview.md) · [api](modules/multi_agent/api.md) |
| **MCP** | `modules/mcp/` | on_attach, on_shutdown | Model Context Protocol 客户端、外部工具桥接 | [overview](modules/mcp/overview.md) · [api](modules/mcp/api.md) |

### 共享检索服务

| 服务包 | 目录 | 职责 | 详细文档 |
|--------|------|------|---------|
| **Retrieval** | `modules/retrieval/` | EmbeddingProvider + VectorBackend + LexicalBackend + RetrievalPipeline + Factory | [overview](modules/retrieval/overview.md) · [api](modules/retrieval/api.md) |

Retrieval 不是 Module（无 on_attach），是 Memory/RAG/Reflection 的共享基础设施。模块通过 `create_pipeline()` 工厂函数获取组件，不直接引用具体实现类。

### 模块间关系

```
ToolsModule ─── on_input: pack 重置
    │           on_context: 状态评估 + WORKSPACE_GUIDE + hint block
    │
SkillModule ─── on_context: skill 匹配 → pack 激活 → playbook 注入
    │           (必须在 ToolsModule 之后注册)
    │
MemoryModule ── on_context: 自动 recall（auto 模式）
    │           on_output: hybrid 事实提取
    │
RAGModule ───── on_context: 知识库使用指南注入
    │
ReflectionModule ── on_context: 教训注入
    │                on_output: 质量评估
    │
PlanningModule ─── on_attach: 注入 PlanReAct 执行策略
    │
SafetyModule ──── on_input: 输入过滤
    │              on_output: 输出脱敏
    │              on_shutdown: 关闭审计日志
    │
SandboxModule ── on_attach: 注入 tool_executor
    │
JobModule ────── start_job → tool_executor.run_command()
    │            (硬依赖 SandboxModule)
    │
ChildAgentModule ── spawn_child（默认公开面）
    │
MultiAgentModule ── delegate（multi-agent pack）
    │
MCPModule ────── on_attach: 桥接外部 MCP 工具
```

### 推荐注册顺序

```
safety → sandbox → tools → job → rag → memory → skill → reflection → planning → mcp → multi_agent → child_agent
```

关键依赖：
- **SandboxModule 在 JobModule 之前**（注入 tool_executor，否则 Job 不注册工具）
- **ToolsModule 在 SkillModule 之前**（pack 重置 → skill 追加）

### 已知限制

1. **Background thread 异常日志缺失**：JobHandle 丢弃 traceback
2. **path_extractor lambda 按引用捕获**：服务对象不重新赋值则无风险
3. **Workspace cleanup 失败无重试**：只 log warning
4. **Shell 命令内路径不受 changeset 追踪**：zone 只看 cwd，不解析命令内容

## 7. 接口层

### 7.1 CLI（`interfaces/cli.py`）

| 特性 | 说明 |
|------|------|
| Rich 终端 | 彩色输出 + Markdown 渲染（Rich 未安装时纯文本 fallback） |
| 斜杠命令 | `/quit` `/help` `/status` `/modules` `/clear` `/mode` `/abort` |
| Mode Switching（v2.0.1） | `/mode [name]` 切换/查看模式；`/abort` 中止任务 |
| Task Contract 展示（v2.0.1） | 检测 `[Task Contract]` 前缀，Rich Panel 黄色边框高亮 |
| Continuous 交互（v2.0.1） | trigger 配置 → runner.run() 阻塞 → Ctrl+C 返回 interactive |
| confirm_handler（v2.0.1） | 交互式授权确认提示 |
| ASCII 横幅 | 启动时显示欢迎画面 |
| 参数解析 | `--modules` 指定加载模块，`--ask` 单问题模式 |

### 7.2 Web UI（`interfaces/web_ui.py`）

| 特性 | 说明 |
|------|------|
| Gradio 聊天 | 多轮对话界面 + 消息历史 |
| Mode 选择（v2.0.1） | 构建时选择 interactive / task / continuous 模式 |
| Continuous Runner（v2.0.1） | trigger 配置面板 + Start/Stop/Refresh 按钮 |
| confirm_handler（v2.0.1） | auto-approve（task contract 提供确认环节） |
| 文件上传 | 拖拽上传文档到 RAG（.txt / .md / .pdf） |
| 状态面板 | 右侧显示 Agent 状态、已加载模块 |

### 7.3 HTTP API（`interfaces/api_server.py`）

| 特性 | 说明 |
|------|------|
| REST API | `POST /chat`、`GET /status`、`GET /modules`、`POST /upload`、`POST /clear` |
| Mode API（v2.0.1） | `POST /mode` 切换模式、`GET /mode` 查看模式、`POST /abort` 中止任务 |
| confirm_handler（v2.0.1） | auto-approve（调用者通过 contract 在应用层决策） |
| WebSocket | `/ws/chat` 流式对话 |
| 会话管理 | OrderedDict LRU 淘汰（默认最多 100 会话） |
| 安全 | Bearer Token 认证 + 速率限制（60 次/分钟/IP） |
| CORS | 默认允许所有来源 |

### 7.4 接口层通用行为

所有接口在创建 Agent 时**自动加载 safety 模块**，确保通过外部接口进入的请求始终受安全保护。用户通过 `--modules` 指定的模块在 safety 之后加载。

**confirm_handler 策略（v2.0.1）**：CLI 使用交互式确认提示（用户终端输入 yes/no），Web UI 和 API Server 使用 auto-approve（task mode 的 contract 流程已提供确认环节）。

---

## 8. 安全体系

### 8.0 威胁模型

**防护目标**：防止 LLM 幻觉导致的误操作（误删文件、误写关键配置、命令拼错导致破坏等）。

**非防护目标**：不防恶意用户。用户使用 agent 是为了提升效率，不是为了破坏自己的电脑。

**分层应对**：
- 正常误操作（99%）→ 区域系统 + 确认机制拦住
- 代码中的明文路径越界 → 字符串扫描拦住
- 恶意绕过 / LLM 被毒化 → 沙箱隔离兜底（Phase 2，用户在敏感场景下自行开启）

### 8.1 三层区域系统（Core）

以项目目录（`cwd`）为基准，划分三个安全区域：

| 区域 | 路径 | sl=1（只读/无副作用） | sl=2（有副作用） |
|------|------|-----------------|-----------------|
| **Playground** | `{cwd}/llama_playground/` | 直接执行 | 直接执行 |
| **项目目录** | `{cwd}/` 内（playground 除外） | 直接执行 | 暂停确认 |
| **外部** | `{cwd}/` 以外 | 暂停确认 | 禁止执行 |

- Playground 由框架启动时自动创建（如不存在），位于 `{cwd}/llama_playground/`
- 区域判定使用 `os.path.realpath()` 防止 symlink 绕过
- 区域检查在 `call_tool()` 中统一执行，不在工具内部

**路径提取机制（path_extractor）**：

每个工具可注册一个 `path_extractor` 函数，告诉 `call_tool` 如何从参数中提取需要检查的路径：

```python
# 内置工具：显式注册 path_extractor
agent.register_tool(
    name="read_file", func=read_file,
    path_extractor=lambda args: [args.get("filename")],
)

# execute_command：从命令字符串中扫描路径
agent.register_tool(
    name="execute_command", func=execute_command,
    path_extractor=lambda args: extract_paths_from_command(args.get("command", "")),
)

# delegate：无路径操作，不注册 path_extractor
agent.register_tool(name="delegate", func=delegate)
```

> **注意**：read_file、write_file、execute_command 已在 v1.5 中删除，由 read_files、write_files、apply_patch、start_job 替代。

**call_tool 区域检查流程**：

1. 有 `path_extractor` → 调用它 → 拿到路径列表 → 逐个 `realpath()` → 判断区域 → 结合 sl 决定执行/确认/拒绝
2. 无 `path_extractor` → 使用 `auto_path_extractor` 兜底：检查参数名中是否包含 `path`、`file`、`filepath` 等关键词，匹配到的参数值作为路径检查
3. 提取结果为空 → 跳过区域检查，直接执行

**auto_path_extractor**（兜底，用于创建的工具等未注册 path_extractor 的情况）：

```python
PATH_KEYWORDS = {"path", "file", "filepath"}

def auto_path_extractor(args):
    return [v for k, v in args.items()
            if any(kw in k.lower() for kw in PATH_KEYWORDS)
            and isinstance(v, str)]
```

**三种工具的覆盖方式**：

| 工具类型 | 路径提取方式 | 示例 |
|---------|------------|------|
| 内置工具 | 显式 `path_extractor` | read_file、write_file、execute_command |
| 创建的工具 | `auto_path_extractor` 自动推断 | 参数名含 path/file 的自动检查 |
| 无路径工具 | 提取为空，跳过 | delegate、list_agents、recall_memory |

> **注意**：read_file、write_file、execute_command 已在 v1.5 中删除，由 read_files、write_files、apply_patch、start_job 替代。

### 8.1.1 Playground 生命周期

- **启动时**：检测 `{cwd}/llama_playground/` 是否存在，不存在则自动创建
- **运行时**：playground 内所有操作无限制（sl=1 和 sl=2 均直接执行）
- **退出时**：不自动清理（保留用户产出的文件）
- **位置**：始终在当前工作目录下，跟随用户的项目位置

### 8.1.2 确认机制（confirm_handler）

在 LlamAgent 上设置 `confirm_handler` 可选回调（v1.9 升级为结构化接口）：

```python
self.confirm_handler: Callable[[ConfirmRequest], ConfirmResponse | bool] | None = None
```

`call_tool()` 中 `AuthorizationEngine` 判定为 CONFIRMABLE 时，通过 `ConfirmRequest` 发起确认：

```python
# InteractivePolicy 逐 item 确认
request = ConfirmRequest(
    kind="operation_confirm",
    tool_name=tool_name,
    action=item.action,    # "read" | "write" | "execute"
    zone=item.zone,        # "playground" | "project" | "external"
    target_paths=[item.path],
    message=item.message,
)
response = engine.confirm(request)  # → agent._ask_confirmation(request)
```

- `confirm_handler` 未设置时，需确认的操作默认拒绝
- **确认等待时间不计入 ReAct 超时**（`react_timeout` 只计算 LLM 调用和工具执行的时间）
- handler 返回 bool 时有 backward compat 自动包装为 ConfirmResponse

### 8.1.3 安全机制总览

| 层面 | 位置 | 检查对象 | 说明 |
|------|------|----------|------|
| 区域系统 | `call_tool()` | 工具操作的目标路径 | path_extractor 提取路径 + 区域判定 + safety_level 联动 |
| 输入过滤 | `on_input` callback | 用户的输入文本 | 拦截注入攻击和有害内容（SafetyModule） |
| 输出脱敏 | `on_output` callback | 模型的输出文本 | 遮盖 API Key、凭据等敏感信息（SafetyModule） |
| 沙箱隔离 | `ToolExecutor` | 带 execution_policy 的工具 | 由 SandboxModule 提供（Phase 2 完善） |

### 8.2 输入过滤（on_input）

检测用户输入中的安全威胁：
- **注入攻击检测**：识别 "忽略之前的指令"、"ignore previous instructions" 等 prompt injection
- **危险内容检测**：正则匹配武器、毒品、入侵等有害内容关键词
- **过长输入**：超过阈值则截断

**拒绝机制**：检测到注入攻击或危险内容时返回空字符串 `""`。`_run_pipeline()` 检查到空输入后返回 `PipelineOutcome(blocked=True)`，`chat()` 通过 `outcome.blocked` 判断是否被拦截（v1.9.7 改为结构化检测，不再依赖响应文本字符串比较）。

> **约定**：`agent.chat(user_input)` 的 `user_input` 不应为空字符串。用户空输入应在接口层拦截。空字符串在 agent 内部视为安全拦截信号。

### 8.3 权限等级

每个角色（Persona）携带 `permission_level`：

```python
@dataclass
class Persona:
    ...
    permission_level: int | None = None

    def __post_init__(self):
        if self.permission_level is None:
            self.permission_level = 3 if self.role == "admin" else 1
```

| 场景 | permission_level 来源 |
|------|------|
| 有 Persona | `persona.permission_level`（admin 默认 3，user 默认 1，均可手动调整） |
| 无 Persona | `config.permission_level`（兜底默认值） |

### 8.4 tier 与 safety_level

| 属性 | 职责 | 生效位置 |
|------|------|----------|
| tier | **访问控制**：控制谁能看到和使用工具（看见 = 能用） | `get_all_tool_schemas()` |
| safety_level | **区域联动**：决定在不同区域需要什么级别的保护 | `call_tool()` 区域检查 |

**tier 决定一切访问权限**。safety_level 不再是"能不能用"的门槛，而是与区域系统联动，决定"在不同区域需要什么级别的保护"。

| safety_level | 含义 | Playground 内 | 项目目录内 | 外部 |
|------|------|------|------|------|
| 1 | 无副作用（读取、查询） | 直接执行 | 直接执行 | 暂停确认 |
| 2 | 有副作用（写入、删除、执行） | 直接执行 | 暂停确认 | 禁止执行 |

**SafetyModule 的实际安全能力**：仅提供输入过滤（`on_input`，拦截注入攻击）和输出脱敏（`on_output`，遮盖 API Key 等敏感信息）。SafetyModule 不控制工具执行——工具执行安全由 core 层的三层区域系统独立负责，始终生效，不依赖 SafetyModule 是否加载。

**自定义工具**（create_tool）安全机制：

1. **builtins 黑名单**：仅禁止 `exec`/`eval`（防套娃——禁止在工具代码中嵌套执行代码。正常 import 允许，区域系统在运行时管路径安全）
2. **字符串字面量路径扫描**：创建工具时扫描代码中的字符串字面量，如果发现路径类字符串越界（超出项目目录）则拒绝创建
3. **auto_path_extractor 运行时保护**：创建的工具未显式注册 `path_extractor`，运行时由 `auto_path_extractor` 自动推断路径参数并做区域检查
4. **沙箱兜底**（Phase 2）：用户在敏感场景下自行开启 Docker 隔离

### 8.5 输出脱敏（on_output）

检查模型输出并自动遮蔽敏感信息，完整脱敏类别如下：

| 类别 | 匹配模式 | 脱敏方式 |
|------|----------|----------|
| API Keys | `sk-xxx` 模式（20+ 位字母数字） | 替换为 `[REDACTED]` |
| 凭证（带引号） | `key/token/secret/password = "value"`（引号内 10+ 字符） | 值替换为 `[REDACTED]` |
| 凭证（无引号） | `key/token/secret/password = value`（20+ 位字母数字） | 值替换为 `[REDACTED]` |
| 手机号 | 11 位中国手机号（1[3-9]开头） | 中间 4 位替换为 `****` |
| 身份证号 | 18 位中国身份证号（含末位 X） | 中间 8 位替换为 `********` |
| 银行卡号 | 16-19 位纯数字 | 保留前 4 位 + 后 4 位，中间替换为 `****` |

- 审计日志：所有违规事件记录到 `safety_audit.log`（通过 `_log_violation()` 写入审计 logger）

### 8.6 三层安全防线总结

| 层次 | 机制 | 防护范围 | 防护对象 | 状态 |
|------|------|---------|---------|------|
| 1. 区域系统 | path_extractor 路径检查 + confirm_handler | 内置工具的文件/命令越界 | LLM 幻觉误操作 | v1.3 实现 |
| 2. 字符串扫描 | create_tool 代码字面量扫描 + builtins 黑名单 | 创建的工具中明文路径越界 | LLM 幻觉误操作 | v1.3 实现 |
| 3. 沙箱隔离 | Docker/gVisor 进程隔离 | 所有绕过手段 | 恶意绕过 / LLM 被毒化 | Phase 2（用户自行开启） |

### 8.7 与 v1.2 的对比

| 方面 | v1.2 | v1.3 |
|------|------|------|
| safety_level 作用 | 无 SafetyModule 时 core 兜底拒绝 sl>=2 | 与区域联动，决定执行/确认/拒绝 |
| 保护边界 | 单一（output_dir 或 cwd） | 三层区域（playground / 项目 / 外部） |
| SafetyModule | 加载后放行所有工具 | 变为可选增强（on_input + on_output），不影响区域系统 |
| safety_loaded 标记 | 控制 core 兜底是否生效 | 移除，区域系统取代其职责 |
| create_tool 前置要求 | 需要 SandboxModule | 移除（区域系统取代） |
| execute_command shell | False（shlex.split） | True（恢复完整 shell 能力） |

---

## 9. 配置参考

### 配置优先级链（v1.7.1）

```
环境变量 > YAML 文件 > 代码默认值
```

- **环境变量**：最高优先级，部署时覆盖任何配置
- **YAML 文件**：`llamagent.yaml` 或 `.llamagent/config.yaml`，层次化配置。自动发现（单文件生效，不做 merge）。显式 `Config(config_path="xxx.yaml")` 失败时直接报错
- **代码默认值**：Config 类中的默认值

模块访问方式不变：`config.xxx` 扁平属性。YAML 层次结构只在 Config 内部解析。

以下为 `Config` 类全部可配置字段。部分参数支持通过环境变量和 YAML 覆盖，下表中标注了对应的环境变量名。

### 模型配置

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `model` | str | 自动检测 | `MODEL_NAME` | 模型名称。优先级：环境变量 > API Key 检测（DeepSeek > OpenAI > Anthropic）> Ollama 回退 |
| `max_context_tokens` | int | 自动获取 | — | 模型上下文窗口大小，通过 `litellm.get_max_tokens()` 自动获取 |
| `api_retry_count` | int | 1 | — | LLM API 调用失败重试次数（指数退避） |

### Agent 配置

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `system_prompt` | str | 羊驼人设 | — | 无 Persona 时的默认身份 |
| `context_window_size` | int | 20 | — | 对话上下文保留轮数 |
| `context_compress_threshold` | float | 0.7 | — | 对话 token 占 `max_context_tokens` 的比例阈值 |
| `compress_keep_turns` | int | 3 | — | 压缩时保留最近 N 轮原文 |
| `max_react_steps` | int | 10 | `MAX_REACT_STEPS` | ReAct 循环最大步数 |
| `max_duplicate_actions` | int | 2 | `MAX_DUPLICATE_ACTIONS` | 连续相同 action 中止阈值 |
| `react_timeout` | float | 210.0 | `REACT_TIMEOUT` | 单步 ReAct 执行时间上限（秒） |
| `max_observation_tokens` | int | 2000 | — | 单次工具返回结果最大 token 数 |

### 检索层配置（v1.7）

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `retrieval_persist_dir` | str | `data/chroma` | `RETRIEVAL_PERSIST_DIR`（fallback: `CHROMA_DIR`） | 检索层持久化根目录 |
| `embedding_provider` | str | "chromadb" | `EMBEDDING_PROVIDER` | Embedding 提供者 |
| `embedding_model` | str | "" | `EMBEDDING_MODEL` | Embedding 模型名（空=提供者默认） |

### 记忆配置（v1.7 读写解耦）

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `memory_mode` | str | "off" | `MEMORY_MODE` | 写入模式："off" / "autonomous" / "hybrid" |
| `memory_recall_mode` | str | "tool" | `MEMORY_RECALL_MODE` | 读取模式："off" / "tool" / "auto" |
| `memory_fact_fallback` | str | "text" | `MEMORY_FACT_FALLBACK` | 事实提取失败兜底："text"（存纯文本）/ "drop"（丢弃） |
| `memory_recall_top_k` | int | 5 | `MEMORY_RECALL_TOP_K` | recall 检索返回数量 |
| `memory_auto_recall_max_inject` | int | 3 | `MEMORY_AUTO_RECALL_MAX_INJECT` | auto recall 最大注入条数 |
| `memory_auto_recall_threshold` | float | 0.35 | `MEMORY_AUTO_RECALL_THRESHOLD` | auto recall 相似度阈值 |

### RAG 配置（v1.7 hybrid）

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `rag_top_k` | int | 3 | `RAG_TOP_K` | 检索结果数量 |
| `chunk_size` | int | 500 | `CHUNK_SIZE` | 文本分块大小 |
| `rag_retrieval_mode` | str | "hybrid" | `RAG_RETRIEVAL_MODE` | 检索模式："vector" / "lexical" / "hybrid" |
| `rag_rerank_enabled` | bool | true | `RAG_RERANK` | 是否启用 LLM reranking |

### 角色与工具配置

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `persona_file` | str | `data/personas.json` | `PERSONA_FILE` | 角色定义存储路径 |
| `agent_tools_dir` | str | `data/agent_tools` | `AGENT_TOOLS_DIR` | 自定义工具存储目录 |

### 规划配置

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `max_plan_adjustments` | int | 7 | `MAX_PLAN_ADJUSTMENTS` | PlanReAct 计划最大调整次数（含三种 replan 路径共享） |

### 反思配置

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `reflection_enabled` | bool | false | — | 反思评估开关 |
| `reflection_score_threshold` | float | 7.0 | `REFLECTION_SCORE_THRESHOLD` | 低于此分触发教训保存 |

### 技能配置（v1.4）

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `skill_dirs` | list[str] | `[]` | `SKILL_DIRS` | 额外的 skill 目录路径（追加到默认扫描路径之后） |
| `skill_max_active` | int | 2 | — | 每轮最多激活的 skill 数量 |
| `skill_llm_fallback` | bool | false | — | 开启 C 级兜底：B 级无候选时，让 LLM 扫描全量 metadata 做语义匹配 |

### 安全配置

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `permission_level` | int | 1 | `PERMISSION_LEVEL` | 无 Persona 时的兜底权限等级 |

### Job 配置（v1.5）

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `job_default_timeout` | float | 300.0 | `JOB_DEFAULT_TIMEOUT` | Job 默认超时时间（秒） |
| `job_max_active` | int | 10 | `JOB_MAX_ACTIVE` | 同时运行的最大 Job 数量 |
| `job_profiles` | dict | `{}` | — | Job 执行配置预设（profile name → config dict） |

### Workspace 配置（v1.5）

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `workspace_id` | str \| None | None | — | Workspace ID（可由 API server 传入 session_id，否则首次使用时懒生成） |

### 输出配置

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|------|------|--------|----------|------|
| `output_dir` | str | `output/` | `OUTPUT_DIR` | 文件输出目录 |

---

## 10. 沙箱执行系统（Sandbox Execution）

沙箱执行系统为高风险工具提供可选的隔离执行环境。整个系统作为独立模块实现，不修改 `run_react()`、`ExecutionStrategy` 接口或 `Module` 基类。core 层仅新增一个属性和一段分发逻辑，未加载沙箱模块时行为与原有完全一致。

### 10.1 ExecutionPolicy

描述一次工具执行**需要什么能力**，不描述用什么平台实现。所有字段都有默认值，默认组合等同于 host 直调。

```python
@dataclass
class ExecutionPolicy:
    """Backend-agnostic execution requirements for a tool."""

    # 运行时环境
    runtime: str = "python"           # "python" | "shell" | "wasm" (reserved) | "native"
    # 注：wasm 为预留值，计划与 Skills 系统一起实现（Wasm 插件作为 skill 的一种实现形态）

    # 隔离级别
    isolation: str = "none"           # "none" | "process" | "container" | "microvm"

    # 文件系统访问
    filesystem: str = "host"          # "host" | "read_only" | "task_workspace" | "isolated_rw"

    # 网络访问
    network: str = "full"             # "none" | "allowlist" | "full"

    # 生命周期
    session_mode: str = "one_shot"    # "one_shot" | "task_session"

    # 资源限制（None = 无限制）
    timeout_seconds: float | None = None
    max_memory_mb: int | None = None
    max_cpu_cores: float | None = None
    max_artifact_bytes: int | None = None
```

工具注册时通过 `execution_policy` 字段关联策略。`execution_policy` 为 `None` 表示不需要沙箱，走 host 直调。

**预置策略常量**（便利定义，用户也可自行组合）：

```python
POLICY_HOST = ExecutionPolicy()  # 默认，等同于 host 直调
POLICY_READONLY = ExecutionPolicy(
    isolation="process", filesystem="read_only",
    network="none", timeout_seconds=30,
)
POLICY_UNTRUSTED_CODE = ExecutionPolicy(
    isolation="process", filesystem="task_workspace",
    network="none", timeout_seconds=60, max_memory_mb=512,
)
POLICY_SHELL_LIMITED = ExecutionPolicy(
    runtime="shell", isolation="process",
    filesystem="task_workspace", network="none",
    timeout_seconds=30, max_memory_mb=256,
)
POLICY_SANDBOXED_CODER = ExecutionPolicy(
    runtime="python",
    isolation="process",
    filesystem="task_workspace",
    network="allowlist",      # pip install etc.
    session_mode="task_session",
    timeout_seconds=300,
    max_memory_mb=1024,
)

# 本地子进程沙箱：仅依赖 subprocess，无真正隔离，但有超时保护。
# 这是唯一与 LocalProcessBackend（默认后端）兼容的预置策略。
# 需要真正隔离的场景应使用更严格的预置策略 + Docker/gVisor 后端。
POLICY_LOCAL_SUBPROCESS = ExecutionPolicy(
    runtime="shell",
    isolation="none",
    filesystem="host",
    network="full",
    timeout_seconds=30,
)
```

> `POLICY_SANDBOXED_CODER` 专为 coder 子 Agent 设计，允许在任务目录中读写，有限网络访问（如 pip install），更长超时和更大内存限额。
>
> `POLICY_LOCAL_SUBPROCESS` 是 `SandboxModule` 的 `auto_assign` 默认分配策略（而非 `POLICY_SHELL_LIMITED`）。`POLICY_SHELL_LIMITED` 要求 `isolation="process"` + `filesystem="task_workspace"`，但 `LocalProcessBackend` 仅支持 `isolation="none"`，因此需要一个兼容预置。

### 10.2 ExecutionBackend 与 ExecutionSession 协议

后端实现两个抽象：`ExecutionBackend` 负责创建会话，`ExecutionSession` 负责执行。

```python
@dataclass
class ExecutionSpec:
    """单次执行请求。"""
    command: str                        # 要执行的内容（代码/命令）
    args: dict                          # 工具参数
    policy: ExecutionPolicy             # 执行策略
    workspace_path: str | None = None   # task_session 模式下复用的工作目录
    env_vars: dict | None = None        # 额外环境变量


@dataclass
class ExecutionResult:
    """执行结果的结构化返回。"""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    artifacts: list[str] = field(default_factory=list)   # 产出文件路径
    duration_ms: float = 0
    peak_memory_mb: float = 0
    timed_out: bool = False
    truncated: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_observation(self) -> str:
        """转换为 ReAct 循环的 tool observation 字符串。"""
        ...


class ExecutionSession:
    """可复用的执行会话（抽象容器/进程/VM 生命周期）。"""

    def run(self, spec: ExecutionSpec) -> ExecutionResult: ...
    def close(self) -> None: ...

    @property
    def workspace_path(self) -> str | None:
        return None


class ExecutionBackend:
    """创建会话的抽象后端。"""

    name: str = "base"

    def create_session(self, policy: ExecutionPolicy) -> ExecutionSession: ...

    def capabilities(self) -> dict:
        """自描述后端支持的能力。"""
        # 返回示例（LocalProcessBackend）：
        # {
        #     "supported_runtimes": ["python", "shell"],
        #     "supported_isolation": ["none"],  # LocalProcessBackend 仅支持 none
        #     "supports_network_isolation": False,
        #     "supports_persistent_session": True,
        #     "available": True,
        # }
        ...
```

**Session 的生命周期管理**：
- `one_shot`：每次 `create_session` → `run` → `close`
- `task_session`：同一个 task 内复用 session（解决冷启动 + workspace 复用）
- 底层无论是 subprocess、Docker、gVisor 还是 microVM，这个抽象都成立

### 10.3 BackendResolver

后端解析器通过自描述能力匹配选择最佳后端，新增后端不需要改执行主链路。

```python
class BackendResolver:
    """根据 policy 选择最佳可用后端。"""

    def register(self, backend: ExecutionBackend) -> None: ...

    def resolve(self, policy: ExecutionPolicy) -> ExecutionBackend:
        """遍历已注册后端，过滤 available + runtime + isolation + network 匹配项，
        按隔离级别排序（优先选隔离能力更强的），返回最佳匹配。
        若 policy.network != "full"，则跳过不支持网络隔离的后端（fail-closed）。
        无匹配则抛出 RuntimeError。"""
        ...
```

注册示例：

```python
resolver.register(LocalProcessBackend())
resolver.register(DockerBackend())      # 可选
resolver.register(GVisorBackend())      # 可选，Linux only
```

### 10.4 ToolExecutor

挂接在 `call_tool()` 和 `ExecutionBackend` 之间的分发层。根据工具的 `execution_policy` 决定走 host 直调还是沙箱执行。

```python
class ToolExecutor:
    """将工具执行分发到合适的后端。"""

    def __init__(self, resolver: BackendResolver):
        self.resolver = resolver
        self.current_task_id: str | None = None            # 由执行策略在任务开始前设置
        self._sessions: dict[str, ExecutionSession] = {}   # "task_id:backend_name:tool_name" -> session

    def execute(self, tool_info: dict, args: dict) -> str:
        """
        执行工具。根据 execution_policy 路由到沙箱或 host。

        - policy 为 None 或 isolation=="none" → host 直调（原有行为）
        - 否则 → 通过 BackendResolver 找到后端 → 创建/复用 session → 执行
        - one_shot 模式：执行后立即关闭 session
        - task_session 模式：按 self.current_task_id 复用 session
        """
        ...

    def close_task_sessions(self, task_id: str) -> None:
        """清理指定 task 的所有 session。"""
        ...

    def run_command(self, command: str, cwd: str, timeout: float = 300) -> str:
        """
        直接执行 shell 命令（不经过工具分发）。
        由 JobModule 调用，委托给当前 backend 执行。
        """
        ...

    def shutdown(self) -> None:
        """关闭所有活跃 session。"""
        ...
```

> **current_task_id 使用约定**：执行策略在任务开始前设置 `agent.tool_executor.current_task_id = task_id`，任务结束后置 None。SimpleReAct 不设置（默认 None = one_shot），PlanReAct 在每个 step 执行前设置。call_tool 签名不变，core 改动为零。

### 10.5 SandboxModule

标准 Module，`on_attach` 时创建 `BackendResolver` + `ToolExecutor`，注入到 agent，并自动为高风险工具分配沙箱策略。

```python
class SandboxModule(Module):
    name = "sandbox"
    description = "Sandbox execution for high-risk tools"

    def __init__(self, auto_assign: bool = True):
        self.auto_assign = auto_assign
        ...

    def on_attach(self, agent):
        self.resolver = BackendResolver()
        self.resolver.register(LocalProcessBackend())  # 默认后端
        self.executor = ToolExecutor(self.resolver)
        agent.tool_executor = self.executor             # 注入分发器

        # Auto-assign execution_policy to high-risk tools
        if self.auto_assign:
            for name, tool in agent._tools.items():
                if tool.get("safety_level", 1) >= 2 and tool.get("execution_policy") is None:
                    tool["execution_policy"] = POLICY_LOCAL_SUBPROCESS

    def on_shutdown(self):
        if self.executor:
            self.executor.shutdown()
```

**模块依赖方向**：SandboxModule 可以依赖 tools 模块的信息（如读取 safety_level），但反过来 tools 模块不应依赖 sandbox。这保证了 sandbox 是可选的——不加载 sandbox 时，其他模块行为完全不变。

具体约束：
- builtin.py 中的工具不 import sandbox 模块
- SandboxModule 的 on_attach 负责扫描已注册工具并注入 execution_policy
- auto_assign=True（默认）时，safety_level >= 2 的工具自动获得 `POLICY_LOCAL_SUBPROCESS` 沙箱策略
- auto_assign 只填充 execution_policy 为 None 且 safety_level >= 2 的工具。手动通过 register_tool() 设置的 execution_policy 不会被覆盖。
- 用户可设 auto_assign=False 手动控制
- auto_assign 与区域系统互不干扰——区域系统控制路径越界，sandbox 提供进程隔离，两者正交

### 10.6 Core 层集成

core 层的关键属性和 `call_tool` 流程：

**1. `__init__` 属性**：

```python
self.confirm_handler: Callable[[ConfirmRequest], ConfirmResponse | bool] | None = None  # 确认回调（v1.9 结构化，v1.9.7 类型收窄）
self.project_dir: str = os.path.realpath(os.getcwd())  # 项目目录快照（v1.3）
self.playground_dir: str = os.path.realpath(os.path.join(self.project_dir, "llama_playground"))  # Playground 路径（v1.3）
self.tool_executor: ToolExecutor | None = None  # 由 SandboxModule 注入（v1.2）
```

启动时自动创建 `llama_playground/` 目录（如不存在）。

**2. `call_tool()` 完整流程**：

```python
def call_tool(self, name: str, args: dict) -> str:
    # 1. 查找工具
    # 2. PRE_TOOL_USE hook（v1.8）
    # 3. AuthorizationEngine.evaluate()（v1.9）
    #    → 路径提取 → zone 评估（ZoneEvaluation）→ policy 决策
    #    → ALLOW: 继续 / CONFIRMABLE: 逐 item 确认 / HARD_DENY: 拒绝

    # 4. 执行（沙箱分发 或 直调）
    if self.tool_executor is not None:
        return self.tool_executor.execute(tool, args)

    # 4. Host 直调（原有路径）
    result = tool["func"](**args)
    return str(result) if result is not None else ""
```

未加载 SandboxModule 时，`tool_executor` 为 `None`，完全走原有 host 直调路径。区域检查始终生效，不依赖任何模块。

### 10.7 ExecutionResult 与 ReAct 的衔接

`ExecutionResult.to_observation()` 将结构化执行结果转换为字符串，作为 tool observation 传回 ReAct 循环：

- 执行成功 → 返回 `stdout`（无输出则返回 `"(no output)"`）
- 超时 → 返回 `"Execution timed out."` + stderr
- 失败 → 返回 stderr + exit code

这与原有工具返回字符串的接口完全一致，ReAct 循环无需任何修改。

### 10.8 临时文件与 Artifact 生命周期

Artifact 生命周期由父 Agent 统一管理，工具只负责创建，不负责清理：

1. **工具创建临时目录**：沙箱工具执行时创建临时工作目录（如 `/tmp/llamagent_sandbox_xxxx/`）
2. **注册到父 Agent**：工具通过 ToolExecutor 将临时目录路径注册到父 Agent 的 `_managed_workspaces` 列表中
3. **任务期间可访问**：同一 task 内的后续工具调用和子 Agent 可以访问这些目录中的产出文件
4. **shutdown 统一清理**：父 Agent 的 `shutdown()` 调用时，由 ToolExecutor 统一清理所有已注册的临时目录

这样设计的好处：
- 不需要额外的 "artifact export" 机制——文件就在父 Agent 管理的目录里
- 子 Agent 产出的文件通过 `artifact_refs` 传递路径引用，无需跨 workspace 拷贝
- 清理逻辑集中在一处，避免泄漏

### 10.9 文件结构

```
llamagent/modules/sandbox/
├── __init__.py
├── policy.py           # ExecutionPolicy, 预置常量
├── backend.py          # ExecutionBackend, ExecutionSession, ExecutionSpec, ExecutionResult
├── resolver.py         # BackendResolver
├── executor.py         # ToolExecutor
├── module.py           # SandboxModule
└── backends/           # 具体后端实现
    ├── __init__.py
    └── local_process.py  # LocalProcessBackend（默认后端）
```

---

## 11. 子 Agent 控制系统（Child Agent Control）

子 Agent 控制系统提供受控的子 Agent 生命周期管理，包括预算限制、工具过滤和任务追踪。作为独立模块实现，加载后不使用时对主 Agent 零影响。现有 `multi_agent` 模块的 `delegate()` 工具继续作为轻量快捷方式保留。

### 11.1 AgentExecutionPolicy

描述子 Agent 的能力边界——不只是 prompt，还包括工具访问、资源预算和委派控制。

```python
@dataclass
class AgentExecutionPolicy:
    """子 Agent 的能力和资源边界。"""

    # 工具访问控制
    tool_allowlist: list[str] | None = None    # None = 继承父 Agent 全部工具
    tool_denylist: list[str] | None = None     # 显式拒绝（优先于 allowlist）

    # 沙箱
    execution_policy: ExecutionPolicy | None = None  # 子 Agent 工具调用的默认沙箱策略

    # 预算
    budget: Budget | None = None

    # 委派控制
    can_spawn_children: bool = False           # 默认不允许创建子 Agent
    max_delegation_depth: int = 1              # 最大嵌套层级

    # 上下文
    history_mode: str = "none"                 # "none" | "summary" | "full"

    # 结果
    result_mode: str = "text"                  # "text" | "artifacts" | "structured"
```

### 11.2 Budget / BudgetTracker / BudgetedLLM

预算系统采用 wrapper 模式，不侵入 `LLMClient` 核心。

```python
@dataclass
class Budget:
    """子 Agent 或任务的资源预算。"""
    max_tokens: int | None = None
    max_time_seconds: float | None = None
    max_steps: int | None = None
    max_llm_calls: int | None = None
    max_artifact_bytes: int | None = None  # 存储预算（字节）

> 子 Agent 创建时一次性确定存储预算，运行期间自行管理，不需要回传给父 Agent。
> `max_artifact_bytes` 仅在沙箱执行模式下自动生效（由 `ToolExecutor` 统计 `ExecutionResult.artifacts` 文件大小并调用 `BudgetTracker.record_artifact()`）。Host 直调的工具不受此限制，与 `max_observation_tokens` 仅在 `run_react` 内生效是同一思路。


class BudgetTracker:
    """跟踪累计资源使用，与 Budget 对比检查。"""

    def __init__(self, budget: Budget): ...

    # 累计使用量
    artifact_bytes_used: int = 0

    def check(self) -> str | None:
        """在预算内返回 None，超出返回原因字符串。"""
        ...

    def record_llm_call(self, tokens: int = 0) -> None: ...
    def record_step(self) -> None: ...
    def record_artifact(self, size_bytes: int) -> None:
        """记录一次 artifact 写入的字节数。"""
        ...


class BudgetedLLM:
    """包装 LLMClient，每次调用前检查预算。不修改 LLMClient 本身。"""

    def __init__(self, llm: LLMClient, tracker: BudgetTracker): ...

    def chat(self, messages, **kwargs):
        """检查预算 → 调用 LLMClient.chat → 记录消耗。"""
        ...

    def ask(self, prompt, **kwargs): ...
    def ask_json(self, prompt, **kwargs): ...
    def count_tokens(self, messages): ...
```

预算超出时抛出 `BudgetExceededError`，由子 Agent 执行层捕获并记录为任务失败。

### 11.3 ChildAgentSpec

创建子 Agent 时的完整描述。

```python
@dataclass
class ChildAgentSpec:
    """创建子 Agent 的规格描述。"""
    task: str                                  # 要执行的任务
    role: str = "worker"                       # 角色名
    system_prompt: str = ""                    # 角色 prompt（空则按 role 查找）
    context: str = ""                          # 父 Agent 选择性传递的上下文
    policy: AgentExecutionPolicy | None = None # 能力边界
    parent_task_id: str | None = None          # 父任务 ID（用于 TaskBoard 关联）
    artifact_refs: list[dict] = field(default_factory=list)  # [{"path": str, "description": str, "type": "file"|"directory"}]
```

> 父 Agent 可以通过 `artifact_refs` 向子 Agent 传递结构化的产出引用。子 Agent 的 ExecutionSession 创建时，引用的文件会被挂载/拷贝到子 Agent 的工作目录中。

### 11.4 ChildAgentController

控制子 Agent 的完整生命周期：创建、等待、取消、收集结果。

```python
class ChildAgentController:
    """
    控制子 Agent 生命周期：spawn, wait, cancel, collect。

    不替代 MultiAgentOrchestrator.delegate() ——
    delegate() 保留为轻量快捷方式。
    此控制器用于需要预算、生命周期管理和结果收集的复杂多子 Agent 编排。
    """

    def __init__(self, runner: AgentRunnerBackend, task_board: TaskBoard,
                 max_children: int = 20): ...

    def spawn_child(self, spec: ChildAgentSpec, agent_factory) -> str:
        """创建子 Agent，在 TaskBoard 登记，返回 task_id。
        当已生成的子 Agent 数量达到 max_children 时抛出 RuntimeError。"""
        ...

    def wait_child(self, task_id: str, timeout: float | None = None) -> TaskRecord:
        """等待子 Agent 完成，更新 TaskBoard，返回结果。"""
        ...

    def cancel_child(self, task_id: str) -> bool:
        """取消运行中的子 Agent。"""
        ...

    def list_children(self, parent_id: str) -> list[TaskRecord]:
        """列出指定父任务的所有子任务。"""
        ...

    def collect_results(self, parent_id: str) -> list[TaskRecord]:
        """收集所有已完成/失败的子任务结果。"""
        ...
```

**`max_children` 限制**：`ChildAgentController.__init__` 接受 `max_children: int = 20` 参数。`spawn_child()` 在创建前检查当前父任务已有的子 Agent 数量，达到上限时抛出 `RuntimeError`。`ChildAgentModule._spawn_child()` 捕获该异常并返回错误信息作为 tool observation。

**Runner 结果清理**：`spawn_child()` 在将结果同步到 TaskBoard 后，立即清理 runner 内部的 `_results` 缓存（`self.runner._results.pop(task_id, None)`），防止长时间运行时 runner 内存无限增长。TaskBoard 是唯一的结果持久化点。

### 11.5 AgentRunnerBackend 协议

抽象后端，负责子 Agent 的实际执行。

```python
class AgentRunnerBackend:
    """执行子 Agent 的抽象后端。"""

    name: str = "base"

    def spawn(self, spec: ChildAgentSpec, agent_factory) -> str:
        """创建子 Agent，返回 child_id。"""
        ...

    def wait(self, child_id: str, timeout: float | None = None) -> TaskRecord:
        """等待子 Agent 完成，返回结果。"""
        ...

    def cancel(self, child_id: str) -> bool:
        """取消运行中的子 Agent。"""
        ...

    def status(self, child_id: str) -> str:
        """查询子 Agent 状态。"""
        ...
```

### 11.6 TaskBoard 与 TaskRecord

轻量内存任务板，追踪所有子任务的状态和结果。

```python
@dataclass
class TaskRecord:
    """任务板上的单条任务记录。"""
    task_id: str
    parent_id: str | None = None
    role: str = ""
    task: str = ""
    status: str = "pending"        # "pending" | "running" | "completed" | "failed" | "cancelled"
    result: str = ""
    artifacts: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)  # duration, tokens, steps 等
    input_snapshot: dict = field(default_factory=dict)  # {task, context, role, policy_summary, artifact_refs}
    created_at: float = 0
    completed_at: float = 0


class TaskBoard:
    """内存任务追踪板。"""

    def create(self, task_id, parent_id=None, **kwargs) -> TaskRecord: ...
    def update(self, task_id, **kwargs) -> None: ...
    def get(self, task_id) -> TaskRecord | None: ...
    def children_of(self, parent_id) -> list[TaskRecord]: ...
    def collect_results(self, parent_id) -> list[TaskRecord]: ...
```

> `input_snapshot` 记录子 Agent 实际收到的输入快照，用于事后调试和审计。由 `ChildAgentController.spawn_child()` 在创建时自动填充。

### 11.7 预设角色策略

为常见角色预设能力边界，模块可直接按角色名查找对应策略。

```python
ROLE_POLICIES = {
    "researcher": AgentExecutionPolicy(
        tool_allowlist=["web_search", "web_fetch", "search_knowledge", "search_text", "read_files"],
        budget=Budget(max_llm_calls=20, max_time_seconds=300),
        can_spawn_children=False,
    ),
    "writer": AgentExecutionPolicy(
        tool_allowlist=["read_files", "write_files", "apply_patch"],
        budget=Budget(max_llm_calls=15, max_time_seconds=300),
        can_spawn_children=False,
    ),
    "analyst": AgentExecutionPolicy(
        tool_allowlist=["read_files", "search_text", "web_search", "search_knowledge"],
        budget=Budget(max_llm_calls=15, max_time_seconds=300),
        can_spawn_children=False,
    ),
    "coder": AgentExecutionPolicy(
        tool_allowlist=["read_files", "write_files", "apply_patch", "start_job", "glob_files", "search_text"],
        execution_policy=POLICY_SANDBOXED_CODER if _SANDBOX_AVAILABLE else None,  # 沙箱可用时强制进沙箱，不可用时降级为 None
        budget=Budget(max_llm_calls=30, max_time_seconds=600),
        can_spawn_children=False,
    ),
}
```

### 11.8 ChildAgentModule

标准 Module，`on_attach` 时初始化控制器并注册子 Agent 控制工具。

```python
class ChildAgentModule(Module):
    name = "child_agent"
    description = "Spawn and control child agents with budget and capability boundaries"

    def on_attach(self, agent):
        self._parent_id = str(id(agent))              # 以 agent 实例 ID 字符串作为 scope key
        self.task_board = TaskBoard()
        runner = InlineRunnerBackend()               # 默认内联执行
        self.controller = ChildAgentController(runner, self.task_board)

        # 注册控制工具
        agent.register_tool("spawn_child", self._spawn_child,
                            "创建子 Agent 执行子任务（受控预算和能力边界）",
                            tier="default", safety_level=2)
        agent.register_tool("list_children", self._list_children,
                            "列出所有子 Agent 及其状态",
                            tier="default", safety_level=1)
        agent.register_tool("collect_results", self._collect_results,
                            "收集所有已完成子 Agent 的结果",
                            tier="default", safety_level=1)
```

**工具级语义（ReAct observation 返回值）**：

| 工具 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `spawn_child(task, role, context)` | task: 任务描述, role: 角色名, context: 上下文 | 子 Agent 执行结果文本（str） | 同步阻塞，等待子 Agent 完成后返回结果 |
| `list_children()` | 无（scope 为当前 agent 实例） | 当前 agent 所有已创建子 Agent 的状态列表 | 内部使用 `_parent_id` 作为 scope key |
| `collect_results()` | 无（scope 为当前 agent 实例） | 当前 agent 所有已完成子 Agent 的结果 | 内部使用 `_parent_id` 作为 scope key |

> **关键设计**：工具返回的是字符串（供 ReAct observation 使用），不是 task_id。Controller API 内部使用 task_id，但工具层对模型屏蔽了这些细节。Module 内部维护 `_parent_id = id(self.agent)` 作为 scope key，`list_children()` 和 `collect_results()` 无需传 parent_id 参数。

**子 Agent 创建流程（bridging 逻辑）**：

```
模型调用 spawn_child(task, role, context)
  │
  ├─ 1. 按 role 查找 ROLE_POLICIES，获取 AgentExecutionPolicy
  ├─ 2. 构建 ChildAgentSpec
  ├─ 3. task_id = controller.spawn_child(spec, agent_factory)
  │       │
  │       ├─ TaskBoard 登记任务
  │       └─ AgentRunnerBackend.spawn(spec, factory)
  │             │
  │             └─ agent_factory(spec) → 创建受限 LlamAgent
  │                   ├─ config = copy.copy(parent.config) 浅拷贝父 Config，然后覆写子 Agent 专属字段
  │                   │   （避免 __new__ 模式在 Config 新增属性时遗漏字段）
  │                   ├─ child = LlamAgent(config) 通过正常构造函数创建
  │                   ├─ 继承父 Agent 的 LLMClient（通过 BudgetedLLM 包装）
  │                   ├─ 继承父 Agent 的区域系统上下文（project_dir、playground_dir、confirm_handler）
  │                   ├─ 继承父 Agent 的 tool_executor（沙箱执行器）
  │                   ├─ 工具通过 copy.deepcopy(parent._tools) 深拷贝，防止子 Agent 修改工具（如注入 execution_policy）影响父 Agent
  │                   ├─ 按 tool_allowlist / tool_denylist 过滤工具
  │                   ├─ 使用 SimpleReAct 策略（不走规划）
  │                   └─ child.chat(task) → 返回结果
  │
  ├─ 4. record = controller.wait_child(task_id) → 获取 TaskRecord
  └─ 5. return record.result → 返回结果文本作为 tool observation
```

### 11.9 文件结构

```
llamagent/modules/child_agent/
├── __init__.py
├── policy.py           # AgentExecutionPolicy, ChildAgentSpec, ROLE_POLICIES
├── budget.py           # Budget, BudgetTracker, BudgetedLLM, BudgetExceededError
├── task_board.py       # TaskBoard, TaskRecord
├── runner.py           # AgentRunnerBackend
├── module.py           # ChildAgentModule, ChildAgentController
└── runners/            # 具体 runner 实现
    ├── __init__.py
    └── inline.py       # InlineRunnerBackend（默认后端）
```

---

## 12. 可选性与向后兼容

### 12.1 沙箱模块未加载时的行为

未加载 `SandboxModule` 时：

- `agent.tool_executor` 为 `None`
- `call_tool()` 跳过沙箱分发，直接走 host 直调路径
- 工具的 `execution_policy` 字段被忽略
- 行为与沙箱系统引入之前完全一致

### 12.2 子 Agent 模块未加载时的行为

未加载 `ChildAgentModule` 时：

- `spawn_child` / `list_children` / `collect_results` 工具不存在
- 现有 `multi_agent` 模块的 `delegate()` 工具继续正常工作
- 对 core 层零影响

### 12.3 子 Agent 模块已加载但未使用时的行为

`ChildAgentModule` 可加载但不调用——模型只是多了几个可用工具。未实际调用 `spawn_child` 时，`TaskBoard` 为空，无额外开销。

### 12.4 Core 层改动范围

core 层改动：

| 改动位置 | 内容 |
|----------|------|
| `agent.__init__` | 新增 `confirm_handler`、`playground_dir` 属性；启动时自动创建 `llama_playground/` |
| `agent.call_tool()` | 新增区域检查逻辑（替换 v1.2 的 safety_loaded 检查），基于 path_extractor + sl 判断执行/确认/拒绝 |
| `agent.register_tool()` | 新增 `path_extractor` 参数 |
| `agent.tool_executor` | 保留（v1.2 沙箱分发） |
| `react_timeout` 计时 | 确认等待时间不计入 |

**未改动的部分**：
- `run_react()` 签名和逻辑
- `ExecutionStrategy` 接口
- `Module` 基类接口
- `chat()` 主管道
- callback pipeline 执行顺序
- 现有模块的行为

### 12.5 沙箱依赖未安装时的处理

沙箱模块的可选后端依赖（如 Docker SDK）未安装时，`SandboxModule` 优雅降级——`import` 失败只打印提示，不影响其他功能。`LocalProcessBackend` 仅依赖标准库 `subprocess`，无额外安装要求。

### 12.6 Skill 模块未加载时的行为（v1.4）

未加载 `SkillModule` 时：

- 不扫描 skill 目录，不建立元数据索引
- `on_context` 链路中无 skill 注入环节，context 不含 `[Active Skill]` block
- 其他模块（tools / rag / memory 等）的行为不受影响
- 对 core 层零影响——SkillModule 不修改 `_tools`、不注入 `ExecutionStrategy`、不改动 `call_tool()`

### 12.7 Skill 模块已加载但无 skill 文件时的行为（v1.4）

`SkillModule` 可加载但扫描路径下无任何 skill 目录——索引为空，`on_context` 匹配结果始终为 0，不向 context 追加内容。无额外 token 开销。

### 12.8 Job 模块未加载时的行为（v1.5）

未加载 `JobModule` 时：

- `start_job` / `wait_job` 等 6 个 Job 工具不存在
- agent 无命令执行能力（`web_search`、`web_fetch` 等非命令工具不受影响）
- 对 core 层零影响

### 12.9 Job 模块已加载但无 SandboxModule 时的行为（v1.5.1）

`JobModule` 已加载但 `agent.tool_executor` 为 None（SandboxModule 未加载）时：

- `on_attach` 打印 error 日志警告，**跳过工具注册**
- 6 个 Job 工具不存在（LLM 不可见，无命令执行能力）
- 这是 secure by default：执行任意 shell 命令需要用户显式加载 SandboxModule

### 12.10 Job 模块已加载且有 SandboxModule 但未使用时的行为（v1.5）

`JobModule` 正常加载但未调用 `start_job`——JobService 内部 job 列表为空，follow-up 工具返回错误信息。无额外开销。

---

## 13. 默认实现

### 13.1 LocalProcessBackend（沙箱默认后端）

基于 `subprocess` 的执行后端，macOS / Linux / CI 环境均可运行。不提供真正的安全隔离，但完整实现 `ExecutionBackend` / `ExecutionSession` 协议接口。

**特性**：
- 支持 `python` 和 `shell` 两种 runtime
- 仅支持 `none` 一种 isolation 级别（无真正进程隔离——子进程以宿主权限运行）
- 每次 `create_session` 创建临时工作目录（`tempfile.mkdtemp`）
- 超时保护（`subprocess.run` 的 `timeout` 参数，捕获 `TimeoutExpired`）
- 异常兜底：非超时异常（如 `FileNotFoundError`、`PermissionError`）统一转换为 `ExecutionResult(exit_code=-1)`，不会穿透到上层
- 不支持网络隔离（`supports_network_isolation: False`）
- 支持持久会话（`supports_persistent_session: True`）

**环境隔离**：子进程使用最小环境变量集（仅 `PATH`、`HOME`、`LANG`、`TERM`），**不继承**宿主进程的完整环境变量（防止 API Key、凭据等泄漏到沙箱内）。如需向沙箱传递额外环境变量，可通过 `ExecutionSpec.env_vars` 显式指定。

### 13.2 InlineRunnerBackend（子 Agent 默认后端）

在当前线程内串行执行子 Agent，最简单的实现，无并发，但完整实现 `AgentRunnerBackend` 协议。

**特性**：
- `spawn()` 阻塞执行——调用 `agent_factory(spec)` 创建子 Agent，立即执行 `child.chat(task)`，返回时任务已完成
- `wait()` 直接返回已有结果（因为 `spawn` 已同步完成）
- `cancel()` 返回 `False`（无法取消已完成的同步执行）
- 捕获 `BudgetExceededError`，记录为任务失败
- 适合开发调试和简单场景，后续可替换为线程池或进程池后端

---

## 14. 测试策略

### 14.1 协议一致性测试（Backend Conformance Tests）

任何 `ExecutionBackend` 实现都必须通过同一组协议测试：
- 创建 session、执行命令、返回结构化结果
- 超时处理（`timed_out=True`）
- workspace 路径有效性
- session 复用（`task_session` 模式）
- session 关闭后资源释放

任何 `AgentRunnerBackend` 实现都必须通过同一组生命周期测试：
- spawn → wait → 获取结果
- cancel 正在执行的子 Agent
- status 查询

### 14.2 测试分层

| 层次 | 测什么 | 后端 | 环境 |
|------|--------|------|------|
| 协议测试 | Backend 接口合规 | MockBackend | 任何环境 |
| 集成测试 | call_tool → executor → backend → result | LocalProcessBackend | macOS / Linux / CI |
| 端到端测试 | chat() → ReAct → sandboxed tool | LocalProcess + Mock LLM | macOS / Linux / CI |
| 平台测试 | 特定后端功能 | Docker / gVisor | 按需 |

### 14.3 测试文件结构

```
tests/                              # 公开精选测试（CI 运行）
├── test_react.py                   # v1.1 ReAct 循环
├── test_pipeline.py                # v1.1 chat pipeline
├── test_integration.py             # v1.1 模块集成
├── test_planning.py                # v1.1 PlanReAct 流程
├── test_sandbox.py                 # v1.2 沙箱执行流程
└── test_child_agent.py             # v1.2 子 Agent 控制流程

tests_internal/                     # 内部详细测试（gitignored）
├── test_config.py                  # 配置
├── test_llm_mock.py                # LLMClient
├── test_agent_messages.py          # 消息构建
├── test_agent_tools.py             # 工具注册与调用
├── test_agent_modules.py           # 模块注册
├── test_persona.py                 # 角色系统
├── test_step.py                    # Step 数据结构
├── test_react_mock.py              # ReAct 循环详细
├── test_chat_pipeline_mock.py      # chat pipeline 详细
├── test_integration_mock.py        # 模块集成详细
├── test_plan_react_mock.py         # PlanReAct 详细
├── test_sandbox_mock.py            # v1.2 沙箱详细
├── test_child_agent_mock.py        # v1.2 子 Agent 详细
└── test_*_real.py                  # 真实 LLM 测试（需 Ollama）
```

### 14.4 平台相关测试标记

```python
requires_docker = pytest.mark.skipif(not _docker_available(), reason="Docker not available")
requires_gvisor = pytest.mark.skipif(not _gvisor_available(), reason="gVisor not available")
```

测试对象是"是否满足协议"，而不是"某个平台是否工作"。新增后端只需通过 conformance tests 即可验证正确性。

---

## 变更记录

### v1.1 → v1.2

| 变更 | 说明 |
|------|------|
| `Budget` 新增 `max_artifact_bytes` | 子 Agent 创建时确定存储预算，运行时自治 |
| `BudgetTracker` 新增 `artifact_bytes_used` / `record_artifact()` | 跟踪 artifact 存储使用量 |
| 新增 10.8 临时文件与 Artifact 生命周期 | Artifact 生命周期由父 Agent 统一管理，ToolExecutor 维护 `_managed_workspaces` |
| `SandboxModule` 新增 `auto_assign` 参数 | 默认 True，自动为高风险工具分配沙箱策略；明确模块依赖方向 |

### v1.2 → v1.3

| 变更 | 说明 |
|------|------|
| 新增三层区域系统 | playground / 项目目录 / 外部，基于路径判定 + safety_level 联动 |
| 移除 `safety_loaded` 标记 | 区域系统取代 core 兜底逻辑，始终生效，不依赖 SafetyModule |
| SafetyModule 职责收窄 | 仅提供 on_input + on_output，不再控制工具执行 |
| 新增 `confirm_handler` | LlamAgent 新增确认回调，区域检查需确认时调用 |
| 新增 `playground_dir` | LlamAgent 启动时自动创建 `llama_playground/` |
| `register_tool()` 新增 `path_extractor` 参数 | 工具注册时声明路径提取方式 |
| 新增 `auto_path_extractor` | 未注册 path_extractor 的工具自动推断路径参数 |
| `execute_command` 恢复 `shell=True` | 恢复完整 shell 能力，安全由区域系统保障 |
| `read_file` / `write_file` 边界改为三层区域 | 替代原有的 cwd / output_dir 单一边界 |
| `create_tool` 移除 SandboxModule 前置要求 | 区域系统取代 |
| `create_tool` 移除 AST 白名单 | 仅保留 builtins 黑名单（exec/eval/\_\_import\_\_）+ 字符串字面量路径扫描 |
| `react_timeout` 不含确认等待时间 | 确认等待时间独立于 ReAct 超时 |
| 子 Agent 继承区域系统上下文 | 替代 safety_loaded 继承 |

### v1.3 → v1.4

| 变更 | 说明 |
|------|------|
| 新增 Skill 模块 | 任务级 playbook 层，通过 on_context 动态注入流程指引 |
| 新增 `modules/skill/` 目录 | SkillModule + SkillIndex（元数据索引） |
| Skill 文件格式 | config.yaml（元数据）+ SKILL.md（纯 playbook）+ 可选 references/assets/ |
| 四级扫描路径 | 项目级 `.llamagent/skills/` + `.agents/skills/`，用户级 `~/.llamagent/skills/` + `~/.agents/skills/` |
| 三层渐进加载 | 启动只读 config.yaml → 激活时加载 SKILL.md → 按需读 references |
| 三级激活匹配（C 可选） | A: `/skill` 命令；B: tag 分词归一化 + 变体集交集匹配 + LLM 消歧；C: LLM 全量 metadata 扫描（默认关闭） |
| 独立 skill block 注入 | `[Active Skill: name]...[End Skill]` 格式追加到 context 末尾 |
| Skill 不进 history/summary | 每轮重新判断激活，不持久化到对话历史 |
| 子 Agent 不继承父 skill | child agent 独立做 skill 检索 |
| 推荐注册顺序更新 | `safety → tools → rag → memory → skill → reflection → planning → mcp → multi_agent → job` |
| Config 新增 `skill_dirs` | 额外 skill 目录路径 |
| Config 新增 `skill_max_active` | 每轮最多激活 skill 数（默认 2） |
| Config 新增 `skill_llm_fallback` | C 级兜底开关（默认 false） |

### v1.4 → v1.5

| 变更 | 说明 |
|------|------|
| 删除 `read_file`、`write_file`、`execute_command` | 由 read_files、write_files + apply_patch、start_job(wait=True) 完全替代 |
| 新增 JobModule（`modules/job/`） | 硬依赖 SandboxModule（执行）+ 软依赖 ToolsModule（cwd）+ SafetyModule（黑名单），start_job(wait=True/False) 统一同步/异步执行，无自有 subprocess |
| 新增 WorkspaceService（ToolsModule 内部服务） | workspace 生命周期管理、路径解析（相对→workspace、project:前缀→project_dir、绝对→zone 检查） |
| 新增 ProjectSyncService（ToolsModule 内部服务） | workspace→project 同步，结构化 search/replace patch，changeset 操作栈（逐步回滚），whole-file preimage/postimage sync |
| Workspace 目录结构 | `<playground_dir>/sessions/<workspace_id>/shared/` + `tasks/<task_id>/`，始终 Zone 1 |
| 新增 workspace 探索工具 | list_tree、glob_files、search_text、read_files、read_ranges、stat_paths |
| 新增 workspace 修改工具 | write_files、create_temp_file、move_path、copy_path、delete_path |
| 新增 project sync 工具 | apply_patch（search/replace）、preview_patch、replace_block、sync_workspace_to_project（auto/copy/patch）、revert_changes |
| apply_patch 格式 | 结构化 `edits: list[dict]`（`[{"match", "replace", "expected_count"}]`），不用 unified diff |
| core 层零改动 | call_tool()、ToolExecutor 均不变（新增 run_command() 方法供 JobModule 调用） |
| JobModule 硬依赖 SandboxModule + 软依赖 ToolsModule + SafetyModule | 命令执行通过 agent.tool_executor.run_command()（SandboxModule 注入），无自有 subprocess；ToolsModule 提供 cwd 解析；SafetyModule 提供命令黑名单 |
| 新工具注册模式 | Service 方法 + 声明式注册表 + 显式 parameters dict，不用 @tool 装饰器 |
| read_files 内部截断 | 自行管理 max_observation_tokens 预算，_truncate_observation 作为兜底 no-op |
| 推荐注册顺序更新 | 末尾加 job |
| Config 新增 `job_default_timeout` | Job 默认超时（默认 300.0，类型 float） |
| Config 新增 `job_max_active` | 同时运行最大 Job 数（默认 10） |
| Config 新增 `job_profiles` | Job 执行配置预设（profile name → config dict） |
| Config 新增 `workspace_id` | Workspace ID（可由 API server 传入 session_id，否则懒生成） |
| ToolsModule 新增 `on_context` callback | 注入 workspace 行为准则（WORKSPACE_GUIDE）到 LLM context |
| ToolsModule 新增 `on_shutdown` callback | 调用 WorkspaceService.cleanup() 清理 workspace session 目录 |
| Changeset 存储改为 in-memory | ProjectSyncService 实例上的 `list[Changeset]`，仅在 agent session 生命周期内持久 |

### v1.9.x → v2.0

| 变更 | 说明 |
|------|------|
| 新增 Abort 机制 | `agent._abort` 标志 + `abort()` 公开方法；`chat()` 入口重置（唯一重置点）；`run_react()` 两处检查点（循环顶部 + 工具调用后） |
| `ReactResult` 新增 `aborted` 状态 | `status="aborted"` 表示被外部中止 |
| `ReactResult` 新增 `terminal` 字段 | `terminal: bool = False`，标记不可恢复结果。`aborted` 和 `context_overflow` 为 terminal=True |
| PlanReAct 适配 terminal | 检查 `result.terminal` 决定是否终止整个计划，与 abort 机制零耦合 |
| 新增 Mode-aware 配置 | `_MODE_DEFAULTS` 类级常量，`set_mode()` 自动应用效率参数；`_interactive_config` 快照用于恢复 |
| -1 哨兵值 | `max_react_steps=-1` 表示无限，`max_duplicate_actions=-1` 表示关闭检测。`run_react` 循环条件适配 |
| 新增 `core/runner.py` | `ContinuousRunner` + `Trigger` ABC + `TimerTrigger` + `FileTrigger`。外部组件驱动 agent.chat()，依赖方向 Runner -> Agent |
| ContinuousRunner task_timeout + watchdog | `threading.Timer` 监控单任务超时，`on_timeout` 支持 `"abort"` 和 callable 两种模式 |
| Prepare-only 工具 | `_report_question` 在 prepare 阶段注册，结束后移除。buffer 内容通过 `outcome.metadata["open_questions"]` 传回 controller |
| `TaskContract.open_questions` 填充 | Controller 的 `_on_prepare_done` 从 outcome metadata 提取 open_questions 写入 contract |
| clarification_turns 硬限制 | `MAX_CLARIFICATION_TURNS=3`，超过后强制用户做 yes/no 决定 |
| 新增设计原则 P1-P6 | 架构文档新增 1.3 节，记录六条核心架构原则（v1.9.6+ 确立） |

### v2.0 → v2.0.1

| 变更 | 说明 |
|------|------|
| CLI mode switching | 新增 `/mode [name]`、`/abort` 命令；task contract 黄色面板展示；continuous 交互（trigger 配置 → runner 阻塞 → Ctrl+C 返回） |
| CLI confirm_handler | 交互式授权确认提示（显示 tool_name / action / zone / paths，用户 yes/no） |
| Web UI mode 支持 | Mode dropdown（interactive/task/continuous）；continuous runner 面板（Start/Stop/Refresh） |
| API Server mode endpoints | `POST /mode` 切换、`GET /mode` 查看、`POST /abort` 中止；continuous 返回 501 |
| confirm_handler 策略 | CLI=交互式提示，Web UI/API=auto-approve（contract 流程提供确认） |
| 版本号统一 | `__init__.__version__` 作为唯一来源，接口层导入而非硬编码 |
| Shutdown 一致性 | CLI direct mode + API lifespan + Web UI rebuild 均正确调用 shutdown |
| main.py CLI 入口修复 | 构造参数传错 bug 修复（先 create_agent 再传 agent） |
| 示例脚本 | `10_task_mode.py`（task 完整流程）、`11_continuous_mode.py`（runner + trigger 用法） |

### v2.0.1 → v2.0.2

| 变更 | 说明 |
|------|------|
| `ExecutionStrategy.execute_stream()` | 策略接口新增流式方法（默认 None）。Agent 通过返回值路由，不检查具体类型（P1/P2） |
| `SimpleReAct.execute_stream()` | 覆盖实现，调 `run_react_stream()` 返回 generator |
| `LLMClient.chat_stream()` | 传 `stream=True` 给 litellm，返回 chunk iterable |
| `agent.run_react_stream()` | 流式 ReAct 循环，纯文本 stream + tool call 状态消息。含 abort/duplicate/timeout/max_steps 保护 |
| `agent.chat_stream()` | 顶层流式入口，完整生命周期（hooks/on_input/strategy 路由/on_output 事后/history）|
| `_merge_tool_call_deltas()` | 按 index 累积合并 litellm stream tool_call 增量片段 |
| on_output 事后处理 | 用户看 raw stream，on_output 处理累积文本存入 history |
| `TaskLogEntry` dataclass | 结构化任务记录（trigger_type, input, output, status, error, started_at, duration）|
| Runner `task_log` + `get_log()` / `clear_log()` | 内存任务日志，get_log 返回 copy（线程安全）|
| `_run_task` 签名变更 | 新增 trigger 参数，记录 trigger_type |
| Web UI 删除 chat wrap hack | 改用 `runner.get_log()` 读取日志 |
| 接口层 streaming | CLI 逐字输出、Web UI generator yield、API WebSocket 真 streaming |

---

## 15. 已知风险与延后项

以下问题已在代码审计中识别，当前版本暂未修复，记录风险和后续计划。

### 15.1 S-A6：ExecutionPolicy.timeout_seconds 无默认上限

**现状**：`ExecutionPolicy.timeout_seconds` 默认值为 `None`，表示不限制执行时间。用户自定义 policy 时如果忘记设置 timeout，工具可以无限期执行。

**风险**：
- 恶意或有缺陷的工具代码（如死循环、网络阻塞）会导致 Agent 主线程永久挂起
- InlineRunnerBackend 是同步阻塞执行，子 Agent 内的无限工具也会阻塞父 Agent

**当前缓解措施**：
- `POLICY_LOCAL_SUBPROCESS` 预置策略已设 30 秒超时
- `POLICY_UNTRUSTED_CODE` 预置策略已设 60 秒超时
- `SandboxModule.auto_assign` 为高风险工具分配的默认策略自带 30 秒超时
- `run_react()` 有独立的 `react_timeout`（默认 210 秒），会从外层中止整个 ReAct 循环

**后续计划**：考虑在 `ToolExecutor.execute()` 层增加全局默认超时（如 300 秒），当 policy 未指定 timeout 时兜底。

### 15.2 S-B1：非正常退出时临时目录泄漏

**现状**：`ToolExecutor` 在 `shutdown()` 中清理 `_managed_workspaces` 下的所有临时目录。但如果进程非正常退出（SIGKILL、未捕获异常、Ctrl+C 后 shutdown 未执行），临时目录（`/tmp/llamagent_sandbox_*`）会残留。

**风险**：
- 长期运行的 Agent 服务如果反复异常重启，`/tmp` 下会积累大量残留目录
- 残留目录可能包含子 Agent 或沙箱工具产生的敏感中间文件

**当前缓解措施**：
- 临时目录使用 `tempfile.mkdtemp(prefix="llamagent_sandbox_")` 创建，前缀可识别
- 正常关闭路径（`agent.shutdown()` → `SandboxModule.on_shutdown()` → `ToolExecutor.shutdown()`）会完整清理
- 操作系统通常会在重启时清理 `/tmp`

**后续计划**：考虑在 `ToolExecutor.__init__()` 中注册 `atexit.register(self.shutdown)` 作为兜底清理，或在启动时扫描并清理前次残留的 `llamagent_sandbox_*` 目录。

### 15.3 execute_command 路径扫描不完美

**现状**：`execute_command` 的 `path_extractor` 通过扫描命令字符串中的绝对路径来检查区域越界。这种基于字符串扫描的方式只能挡住 LLM 生成的正常命令中的明文路径。

**风险**：
- 命令中的相对路径、环境变量展开（如 `$HOME`）、shell 变量拼接等方式可绕过扫描
- 命令执行后通过管道或子命令访问外部路径的情况无法检测

**当前缓解措施**：
- execute_command 仍为 tier=admin，仅管理员可见可用
- 区域系统挡住 99% 的正常 LLM 幻觉误操作（威胁模型的核心防护对象）
- 完整隔离留给 Phase 2 沙箱后端

**后续计划**：Phase 2 实现 Docker/gVisor 沙箱后端，通过进程隔离从根本上解决命令越界问题。

### 15.4 auto_path_extractor 覆盖有限

**现状**：`auto_path_extractor` 仅检查参数名中包含 `path`、`file`、`filepath` 关键词的参数。创建的工具如果使用其他参数名（如 `target`、`destination`、`source`）传递文件路径，则不会被区域检查拦截。

**风险**：
- LLM 创建的工具可能使用非标准参数名传递路径，绕过区域检查

**当前缓解措施**：
- builtins 黑名单禁止 `__import__`，创建的工具无法导入 os/subprocess 等模块直接操作文件系统
- 字符串字面量路径扫描在创建时检查代码中的明文路径
- 匹配宽松策略：关键词使用子串匹配（`any(kw in k.lower())`），`filepath`、`file_path`、`input_file` 等变体都能匹配

**后续计划**：考虑扩展 PATH_KEYWORDS 集合，或允许工具创建者在代码注释中声明 path 参数。
