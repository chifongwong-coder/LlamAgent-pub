# LlamAgent API 参考文档（Core 层）

> 本文档涵盖 Core 层（LlamAgent、Config、LLMClient、Persona、Hook 系统）和接口层的 API。
> 模块层 API 已分散到各模块文档：`docs/modules/<name>/api.md`。
> Hook 系统详细参考见 `docs/modules/hooks/api.md`。

---

## 1. 核心层（Core）

### 1.1 LlamAgent

LlamAgent 的主类，包含 `chat()` 入口和 `run_react()` 引擎。无需加载任何模块即可作为对话式 AI 助手使用；通过 `register_module()` 加载模块后获得工具调用、RAG、记忆等能力。

#### 构造函数

```python
class LlamAgent:
    def __init__(
        self,
        config: Config | None = None,
        persona: Persona | None = None,
    )
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| config | `Config \| None` | `None` | 配置对象；为 None 时自动创建默认 `Config()` |
| persona | `Persona \| None` | `None` | 角色对象；为 None 时使用 `config.system_prompt` 作为默认身份 |

**实例属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| config | `Config` | 配置对象 |
| persona | `Persona \| None` | 当前角色 |
| llm | `LLMClient` | LLM 客户端实例 |
| modules | `dict[str, Module]` | 已注册模块字典 |
| history | `list[dict]` | 对话历史 |
| summary | `str \| None` | 压缩后的历史摘要 |
| conversation | `list[dict]` | history 的别名（向后兼容） |
| confirm_handler | `Callable[[ConfirmRequest], ConfirmResponse \| bool] \| None` | v1.9 结构化区域确认回调（v1.9.7 类型收窄，接受 bool 返回用于向后兼容）。ConfirmRequest 含 kind/tool_name/action/zone/target_paths/message/mode + 可选 requested_scopes（v1.9.1 合同确认时携带）。未设置时，需确认的操作默认拒绝 |
| mode | `str` | v1.9 授权模式，默认 `"interactive"`。支持 `"interactive"` / `"task"` / `"continuous"` |
| _authorization_engine | `AuthorizationEngine` | v1.9 授权引擎，封装路径提取 + zone 评估 + policy 决策 |
| _controller | `ModeController \| None` | v1.9.6 模式控制器，set_mode("task") 时初始化 |
| project_dir | `str` | 项目目录快照（`os.path.realpath(os.getcwd())`），创建时固定，zone 检查的基准 |
| playground_dir | `str` | playground 目录路径（`{project_dir}/llama_playground`），启动时自动创建 |
| tool_executor | `ToolExecutor \| None` | 沙箱执行分发器，由 SandboxModule 在 on_attach 时注入；为 None 时工具在 host 上直接执行 |
| interaction_handler | `UserInteractionHandler \| None` | v1.8.2 用户交互处理器，由调用方在 `register_module` 前注入；为 None 时 `ask_user` 工具返回错误提示 |
| _active_packs | `set[str]` | v1.6 当前激活的 pack 集合。每轮 `ToolsModule.on_input()` 清空，`on_context` 和工具执行中重新填充 |
| _hooks | `dict[HookEvent, list[HookRegistration]]` | v1.8 事件 hook 注册表，按事件类型分组 |
| _session_started | `bool` | v1.8 首次 chat 标记，控制 SESSION_START 只触发一次 |
| _in_hook | `bool` | v1.8 重入保护标志，防止 hook 内 call_tool 递归触发 hook |
| _abort | `bool` | v2.0 中止标志，由 `abort()` 设置为 True，`chat()` 入口重置为 False |
| _interactive_config | `dict` | v2.0 快照 interactive 模式下的 `_MODE_KEYS` 配置值，set_mode("interactive") 时恢复 |

**使用示例：**

```python
from llamagent.core import LlamAgent, Config

# 最简用法
agent = LlamAgent()
response = agent.chat("你好")

# 带配置和角色
config = Config()
config.system_prompt = "你是一个专业的翻译助手"
agent = LlamAgent(config)
```

---

#### chat

```python
def chat(self, user_input: str) -> str
```

Agent 主入口：接收用户输入，返回响应。

两条执行路径（v1.9.7）：
- Controller 模式（`_controller is not None`）：委托 `_run_controller_turn()` 驱动 controller 两步协议
- 普通模式：`SESSION_START` -> `PRE_CHAT` -> `on_input` -> `on_context` -> `execution strategy` -> `on_output` -> `POST_CHAT`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| user_input | `str` | -- | 用户输入，不应为空字符串 |

**返回值：** `str` -- Agent 的响应文本。

**使用示例：**

```python
response = agent.chat("帮我查一下今天北京的天气")
print(response)
```

**注意事项：**
- 输入为空或被安全模块拦截时返回拒绝语，`PipelineOutcome.blocked=True`（v1.9.7 结构化检测）
- 每次调用后自动进行上下文窗口管理（裁剪最旧轮次）
- 对话前自动检查 token 压缩阈值，超阈值触发历史压缩
- v2.0：入口处重置 `_abort = False`，清除上次任务遗留的中止信号
- v1.8：`SESSION_START` 仅首次 chat 触发；`PRE_CHAT` / `POST_CHAT` 每轮触发；`POST_CHAT` 采用 always-fire 语义（safety 拦截也触发，data 含 `blocked` / `blocked_by` / `completed`）

---

#### chat_stream（v2.0.2）

```python
def chat_stream(self, user_input: str) -> Generator[str, None, None]
```

chat() 的流式版本。逐块 yield 文本。

- **纯文本回复**：LLM stream 逐 token yield
- **Tool call**：yield 状态消息（`"[Calling tool_name...]"`、`"[tool_name done]"`）
- **on_output**：stream 结束后事后处理累积文本，存入 history（用户看 raw stream）
- **策略路由**：通过 `execute_stream()` 接口判断，返回 None 时 fallback 到 `execute()` 一次性 yield
- **Task mode**：fallback 到 `_run_controller_turn()`（非 streaming）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| user_input | `str` | -- | 用户输入 |

**返回值：** `Generator[str, None, None]` -- 文本 chunk generator。

**使用示例：**

```python
for chunk in agent.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
print()
```

---

#### run_react

```python
def run_react(
    self,
    messages: list[dict],
    tools_schema: list[dict],
    tool_dispatch: Callable[[str, dict], str],
    *,
    should_continue: Callable[[], str | None] | None = None,
) -> ReactResult
```

ReAct 循环引擎。无状态，不感知工具来源，由执行策略调用。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| messages | `list[dict]` | -- | 发给 LLM 的消息列表（含系统提示 + 对话历史 + 当前查询） |
| tools_schema | `list[dict]` | -- | OpenAI 格式的工具 schema 列表 |
| tool_dispatch | `Callable[[str, dict], str]` | -- | 工具分发函数，签名 `(name, args) -> result_str` |
| should_continue | `Callable[[], str \| None] \| None` | `None` | 每次工具调用后检查的回调；返回 str 中断循环（该字符串作为中断原因），返回 None 继续 |

**返回值：** `ReactResult` -- 结构化返回结果。

**循环保护机制：**
- 最大步数（`max_react_steps`）
- 重复检测（连续相同工具名 + 参数则中止）
- 超时保护（`react_timeout`，每步独立计时）
- 观测截断（`max_observation_tokens`）
- `ContextWindowExceededError` -> 中止循环

**使用示例：**

```python
messages = agent.build_messages("搜索 Python 教程", "")
tools_schema = agent.get_all_tool_schemas()
result = agent.run_react(messages, tools_schema, agent.call_tool)
print(result.text, result.status)
```

**注意事项：**
- 没有工具时直接进行纯 LLM 对话
- `should_continue` 返回非 None 时，循环以 `interrupted` 状态结束
- 上下文窗口溢出时返回 `context_overflow` 状态（`terminal=True`），并给出友好提示
- v2.0：两个 abort 检查点（循环顶部 + 工具调用后），`_abort` 为 True 时返回 `aborted` 状态（`terminal=True`）

---

#### call_tool

```python
def call_tool(self, name: str, args: dict) -> str
```

注册表工具调用：查找 + 权限检查 + 执行。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| name | `str` | -- | 工具名称 |
| args | `dict` | -- | 工具参数（dict，内部展开调用） |

**返回值：** `str` -- 工具执行结果字符串，或错误描述字符串。

**执行流程：**
1. 查找工具，不存在则返回可用工具列表
2. PRE_TOOL_USE hook（v1.8）：可 SKIP 阻止执行或修改 args
3. AuthorizationEngine.evaluate()（v1.9）：路径提取 → zone 评估（ZoneEvaluation）→ policy 决策（InteractivePolicy 逐 item 确认 CONFIRMABLE，HARD_DENY 直接拒绝）
4. 执行（统一两条路径）：`tool_executor` 非 None → 沙箱执行；否则 `tool["func"](**args)`
5. POST_TOOL_USE hook（v1.8）：成功时触发
6. TOOL_ERROR hook（v1.8）：异常时触发

**区域 + safety_level 矩阵：**

| 区域 | sl=1（只读/无副作用） | sl=2（有副作用） |
|------|-------------|-----------------|
| Playground | 直接执行 | 直接执行 |
| 项目目录 | 直接执行 | 暂停确认 |
| 外部 | 暂停确认 | 禁止执行 |

**注意事项：**
- 确认等待时间不计入 `react_timeout`（仅计算 LLM 调用和工具执行的时间）
- 工具不存在和权限被拒都以字符串返回（作为 tool observation 反馈给模型）

---

#### register_tool

```python
def register_tool(
    self,
    name: str,
    func: Callable,
    description: str,
    parameters: dict | None = None,
    tier: str = "common",
    safety_level: int = 1,
    execution_policy: ExecutionPolicy | None = None,
    creator_id: str | None = None,
    path_extractor: Callable[[dict], list[str]] | None = None,
    pack: str | None = None,
) -> None
```

向注册表注册一个工具。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| name | `str` | -- | 工具名称 |
| func | `Callable` | -- | 工具执行函数 |
| description | `str` | -- | 工具描述 |
| parameters | `dict \| None` | `None` | JSON Schema 格式的参数定义；为 None 时从函数签名推断 |
| tier | `str` | `"common"` | 工具层级 `"default"` / `"common"` / `"admin"` / `"agent"` |
| safety_level | `int` | `1` | 安全等级 1=只读 2=有副作用 3=高风险 |
| execution_policy | -- | `None` | 沙箱执行策略；None 表示不需要沙箱，走 host 直调 |
| creator_id | `str \| None` | `None` | 创建者 persona_id（仅 agent 层级工具使用，用于可见性过滤） |
| path_extractor | `Callable[[dict], list[str]] \| None` | `None` | 路径提取函数，从工具参数中提取需要区域检查的路径列表；None 时使用 `auto_path_extractor` 兜底 |
| pack | `str \| None` | `None` | Pack 名称，None 表示默认公开面（始终可见），非 None 表示条件 pack（仅 pack 激活时可见） |

**注意事项：** 注册时，`name` 会被同时存储到 tool dict 中（`tool["name"] = name`），便于 `ToolExecutor` 等组件直接获取工具名称。

**使用示例：**

```python
def my_calc(expression: str) -> str:
    return str(eval(expression))

agent.register_tool(
    "calculator",
    my_calc,
    "计算数学表达式",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
    safety_level=1,
)
```

---

#### remove_tool

```python
def remove_tool(self, name: str) -> bool
```

从注册表中移除一个工具。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| name | `str` | -- | 工具名称 |

**返回值：** `bool` -- 成功移除返回 True，工具不存在返回 False。

---

#### get_all_tool_schemas

```python
def get_all_tool_schemas(self) -> list[dict]
```

合并注册表中的工具 schema，按 tier + persona role + pack 过滤。

**过滤规则：**
- `tier=default` + `tier=common`：对所有角色可见
- `tier=admin`：仅当 `persona.role == "admin"` 时包含
- `tier=agent`：当前角色的自定义工具
- v1.6 新增 pack 过滤。在 tier/persona 过滤之后，检查工具的 `pack` 字段：`pack` 为 None 的工具始终可见；`pack` 非 None 时仅当 `pack in agent._active_packs` 才可见。

**返回值：** `list[dict]` -- OpenAI function calling 格式的工具 schema 列表。

---

#### register_module

```python
def register_module(self, module: Module) -> None
```

注册一个能力模块。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| module | `Module` | -- | 要注册的模块实例 |

**兼容逻辑：**
- 模块重写了 `on_attach()` -> 调用 `on_attach()`
- 模块仅重写了 `attach()`（旧 API） -> 调用 `attach()`
- 都未重写 -> 调用基类 `on_attach()`

**注意事项：**
- 注册失败（`on_attach` 抛异常）会记录日志并重新抛出
- 模块以 `module.name` 为键存入 `self.modules`

---

#### get_module

```python
def get_module(self, name: str) -> Module | None
```

按名称获取已注册模块。未找到返回 None。

---

#### has_module

```python
def has_module(self, name: str) -> bool
```

检查指定模块是否已注册。

---

#### list_modules

```python
def list_modules(self) -> list[str]
```

列出所有已注册模块的名称。

---

#### authorization_status（v1.9.4）

```python
def authorization_status(self) -> dict
```

返回当前授权状态快照：mode + task_scopes 详情 + session_scopes 详情（含 uses/max_uses/source/expired）。

v1.9.6：scope 详情的格式化逻辑下沉到 `AuthorizationEngine.authorization_status()`，agent 只添加 `mode` 字段后转发。agent 不直接读 engine 内部状态。

---

#### get_active_task_id（v1.9.2）

```python
def get_active_task_id(self) -> str | None
```

获取当前活跃的 task ID，用于 scope 存储和查找。

**优先级**：`controller.state.task_id`（task mode）> `_current_task_id`（PlanReAct）。

**返回值**：`str | None` — task ID 或 None（无活跃 task）。

---

#### set_mode（v1.9.1）

```python
def set_mode(self, mode: str) -> None
```

切换授权模式。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| mode | `str` | -- | `"interactive"` / `"task"` / `"continuous"` |

**行为**（v1.9.6, v1.9.8, v1.9.9）：
- 切换序列：检查 idle → 准备 controller → 清除 scopes → 切换 policy → 配置 controller auto_execute → commit → 发射事件
- `"interactive"`：使用 InteractivePolicy，清除 controller 和所有 scopes
- `"task"`：使用 TaskPolicy，初始化 TaskModeController。加载 seed_scopes（复用 continuous 同一方法），无 seed_scopes 时通过 `ConfirmRequest(kind="session_authorize")` 询问用户是否开放 project 访问。有 session scopes 时 controller.auto_execute=True（跳过 prepare/contract），否则 auto_execute=False（原有流程）
- `"continuous"`：使用 ContinuousPolicy，从 `config.seed_scopes` 加载 session scopes。v1.9.9：无 seed_scopes 时默认创建 project 读写 scope，无交互
- 内部调用 `engine._switch_policy()` 完成 policy 切换（原 `engine.set_mode()` 已内部化）
- v1.9.9：`agent.__init__` 自动读取 `config.authorization_mode`，非 interactive 时调用 `set_mode()`。无效模式值 warning + 回退 interactive。支持环境变量 `AUTHORIZATION_MODE`

---

#### abort（v2.0）

```python
def abort(self) -> None
```

通知 agent 在当前原子操作完成后停止。设置 `_abort = True`。

`run_react()` 在两个检查点读取此标志：循环顶部（LLM 调用前）和每次工具调用后。命中时返回 `ReactResult(status="aborted", terminal=True)`。

`chat()` 在入口处重置标志（`_abort = False`），因此过时的中止信号不会影响新任务。

**线程安全**：可从任意线程调用。典型使用场景是 `ContinuousRunner.stop()` 或外部超时看门狗。

---

#### _MODE_KEYS / _MODE_DEFAULTS（v2.0）

模式感知配置常量，`set_mode()` 切换模式时用于调整 ReAct 引擎参数。

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

| 模式 | max_react_steps | react_timeout | max_duplicate_actions | max_observation_tokens |
|------|----------------|---------------|----------------------|----------------------|
| task | 50 | 600 | 5 | 5000 |
| continuous | -1（无限制） | 600 | -1（无限制） | 10000 |
| interactive | 恢复 `_interactive_config` 快照值 | | | |

**行为**：`__init__` 时保存 interactive 模式的原始值到 `_interactive_config`。`set_mode()` 时根据目标模式从 `_MODE_DEFAULTS` 读取并写入 `config`；切回 interactive 时恢复快照。`-1` 在 `run_react()` 中表示无限制。

---

#### set_execution_strategy

```python
def set_execution_strategy(self, strategy: ExecutionStrategy) -> None
```

替换当前执行策略。默认为 `SimpleReAct`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| strategy | `ExecutionStrategy` | -- | 新的执行策略实例 |

**注意事项：** 通常由规划模块在 `on_attach()` 时调用，注入 `PlanReAct` 策略。

---

#### build_messages

```python
def build_messages(
    self,
    query: str,
    context: str,
    *,
    include_history: bool = True,
    extra_system: str = "",
) -> list[dict]
```

统一消息构建方法。SimpleReAct 和 PlanReAct 都通过此方法构建 LLM 消息列表。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| query | `str` | -- | 用户查询 |
| context | `str` | -- | on_context 输出的上下文字符串 |
| include_history | `bool` | `True` | 是否包含对话历史（SimpleReAct=True，PlanReAct 逐步=False） |
| extra_system | `str` | `""` | 附加系统提示（PlanReAct 步骤指令等） |

**返回值：** `list[dict]` -- 消息列表。

**消息结构：**
1. 系统提示（角色身份 + 模块信息 + 动态上下文 + extra_system）
2. 摘要（如果有）
3. 对话历史（可选）
4. 当前查询

---

#### clear_conversation

```python
def clear_conversation(self) -> None
```

清空对话历史和摘要。

---

#### shutdown

```python
def shutdown(self) -> None
```

Agent 关闭：先触发 `SESSION_END` 事件 hook（v1.8），再逆序调用所有模块的 `on_shutdown()`（洋葱模型）。

---

#### register_hook（v1.8）

```python
def register_hook(
    self,
    event: HookEvent,
    handler: HookCallback | HookHandler,
    *,
    matcher: HookMatcher | None = None,
    priority: int = 100,
    source: str = "code",
) -> None
```

注册一个事件 hook。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| event | `HookEvent` | -- | 监听的事件类型 |
| handler | `HookCallback \| HookHandler` | -- | 处理函数或 Handler 实例；callable 自动包装为 `CallableHandler` |
| matcher | `HookMatcher \| None` | `None` | 过滤条件（AND 逻辑） |
| priority | `int` | `100` | 执行优先级，值越小越先（YAML 注册默认 200） |
| source | `str` | `"code"` | 来源标识 `"code"` / `"yaml"` |

**使用示例：**

```python
from llamagent.core import HookEvent, HookMatcher

def audit_hook(ctx):
    print(f"[AUDIT] {ctx.data['tool_name']}")

agent.register_hook(HookEvent.POST_TOOL_USE, audit_hook)
agent.register_hook(
    HookEvent.PRE_TOOL_USE, blocker,
    matcher=HookMatcher(tool_name="start_job"),
)
```

---

#### emit_hook（v1.8）

```python
def emit_hook(self, event: HookEvent, data: dict) -> HookResult
```

触发一个事件 hook，执行所有匹配的 handler。

| 参数 | 类型 | 说明 |
|------|------|------|
| event | `HookEvent` | 事件类型 |
| data | `dict` | 事件数据（handler 可修改） |

**返回值：** `HookResult` — SKIP 表示阻止（仅 PRE_TOOL_USE），其他事件始终 CONTINUE。

**注意事项：** 通常由框架内部调用（call_tool / chat / shutdown），模块一般不需要直接调用。重入保护：`_in_hook` 为 True 时直接返回 CONTINUE。

---

#### status

```python
def status(self) -> dict
```

返回 Agent 当前状态。

**返回值：** `dict` -- 包含以下字段：

```python
{
    "model": str,           # 使用的模型
    "persona": str | None,  # 角色名称
    "modules": {            # 已加载模块
        "name": "description",
    },
    "conversation_turns": int,  # 对话轮次
}
```

---

### 1.2 Module 基类

所有可插拔模块的基类。模块通过重写回调方法与 Agent 管道交互。

```python
class Module:
    name: str = "base"
    description: str = ""
```

#### 生命周期回调

##### on_attach

```python
def on_attach(self, agent: LlamAgent) -> None
```

模块被 `register_module()` 注册时调用。用于初始化存储、注册工具、注入执行策略、注入安全回调等。

基类默认实现保存 agent 引用：`self.agent = agent`。

##### attach

```python
def attach(self, agent: LlamAgent) -> None
```

向后兼容：旧模块使用 `attach()`，新模块应使用 `on_attach()`。

##### on_shutdown

```python
def on_shutdown(self) -> None
```

Agent 退出时调用。用于关闭连接、释放资源。

#### Pipeline Callback（管道回调）

##### on_input

```python
def on_input(self, user_input: str) -> str
```

输入预处理回调。按模块注册顺序（正序）调用。返回空字符串视为安全拦截。

##### on_context

```python
def on_context(self, query: str, context: str) -> str
```

上下文增强回调。按模块注册顺序（正序）调用。各模块依次追加上下文。

##### on_output

```python
def on_output(self, response: str) -> str
```

输出后处理回调。按模块注册的 **逆序** 调用（洋葱模型）。

##### on_execute（已弃用）

```python
def on_execute(self, query: str, context: str) -> str | None
```

[已弃用] 执行拦截回调；返回非 None 则跳过默认执行。目标架构中被 `ExecutionStrategy` 取代。

---

### 1.3 ExecutionStrategy

可插拔执行策略接口。负责组装 `tools_schema` 和 `tool_dispatch`，然后交给 `run_react()`。

```python
class ExecutionStrategy:
    def execute(self, query: str, context: str, agent: LlamAgent) -> str:
        raise NotImplementedError

    def execute_stream(self, query: str, context: str, agent: LlamAgent) -> Generator | None:
        """v2.0.2: 流式执行。返回 generator 或 None（不支持，fallback 到 execute）。"""
        return None
```

#### execute()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| query | `str` | -- | 经 on_input 处理后的用户输入 |
| context | `str` | -- | 经 on_context 增强后的上下文 |
| agent | `LlamAgent` | -- | Agent 实例 |

**返回值：** `str` -- 最终响应文本。

#### execute_stream()（v2.0.2）

参数同 `execute()`。

**返回值：** `Generator[str, None, None] | None` -- 文本 chunk generator，或 None（不支持 streaming）。

Agent 通过返回值判断策略是否支持 streaming，不检查具体类型（P1/P2）。

**内置策略：**
- `SimpleReAct`（默认）：`execute()` 运行 ReAct 循环，`execute_stream()` 运行 `run_react_stream()`
- `PlanReAct`：`execute()` 规划后逐步执行，`execute_stream()` 返回 None（不支持 streaming）

---

### 1.4 ReactResult

ReAct 循环的结构化返回结果。

```python
@dataclass
class ReactResult:
    text: str
    status: Literal["completed", "max_steps", "timeout", "error", "interrupted", "context_overflow", "aborted"]
    error: str | None = None
    steps_used: int = 0
    reason: str | None = None
    terminal: bool = False  # v2.0
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| text | `str` | -- | 最终文本响应 |
| status | `Literal[...]` | -- | 结束状态 |
| error | `str \| None` | `None` | 错误描述（仅 error/context_overflow/timeout 状态） |
| steps_used | `int` | `0` | 使用的步数 |
| reason | `str \| None` | `None` | 中断原因（仅 interrupted 状态） |
| terminal | `bool` | `False` | v2.0：不可恢复的结果标志。为 True 时调用方应停止（不应重试或重规划）。`"aborted"` 和 `"context_overflow"` 状态下为 True |

**status 值说明：**
- `completed`：正常完成
- `max_steps`：达到最大步数限制
- `timeout`：单步超时
- `error`：执行出错
- `interrupted`：被 `should_continue` 回调中断（如 replan 触发）
- `context_overflow`：上下文窗口溢出（`terminal=True`）
- `aborted`：v2.0 新增。被 `abort()` 信号中止（`terminal=True`）

---

### 1.5 PipelineOutcome（v1.9.6）

`_run_pipeline()` 的结构化返回值。定义在 `contract.py`。

```python
@dataclass
class PipelineOutcome:
    response: str
    task_id: str | None = None
    blocked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| response | `str` | -- | pipeline 产出的响应文本 |
| task_id | `str \| None` | `None` | 关联的 task ID |
| blocked | `bool` | `False` | v1.9.7：on_input 阻断时为 `True`，`chat()` 据此判断是否被安全拦截 |
| metadata | `dict[str, Any]` | `{}` | 不透明数据载体。prepare 模式下 engine 将 pending_scopes 存入此处，agent 不检查内容，controller 读取所需数据 |

---

### 1.6 ConfirmRequest / ConfirmResponse（v1.9, v1.9.8）

授权确认的结构化接口。定义在 `zone.py`。

```python
@dataclass
class ConfirmRequest:
    kind: str               # "operation_confirm" | "session_authorize"
    tool_name: str
    action: str             # "read" | "write" | "execute" | "read_write"
    zone: str               # "playground" | "project" | "external"
    target_paths: list[str]
    message: str
    mode: str = "interactive"
    requested_scopes: list[RequestedScope] | None = None
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| kind | `str` | -- | `"operation_confirm"`：单次工具操作确认；`"session_authorize"`（v1.9.8）：set_mode("task") 时询问是否开放 project 访问 |
| tool_name | `str` | -- | 触发确认的工具名。session_authorize 时为 `"*"` |
| action | `str` | -- | 操作类型。session_authorize 时为 `"read_write"` |
| zone | `str` | -- | 安全区域 |
| target_paths | `list[str]` | -- | 操作目标路径 |
| message | `str` | -- | 展示给用户的确认消息 |
| mode | `str` | `"interactive"` | 信息字段，不参与决策 |
| requested_scopes | `list[RequestedScope] \| None` | `None` | v1.9.1：合同请求�� scope 列表 |

```python
@dataclass
class ConfirmResponse:
    allow: bool
    approved_scopes: list[RequestedScope] | None = None
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| allow | `bool` | -- | 是否批准 |
| approved_scopes | `list[RequestedScope] \| None` | `None` | v1.9.2：用户缩窄的 scope（未使用时为 None） |

**数据流**：policy 构造 `ConfirmRequest` → `engine.confirm()` → `agent._ask_confirmation()` → `confirm_handler` → `ConfirmResponse`。confirm_handler 返回 bool 时自动包装为 ConfirmResponse（向后兼容）。

---

### 1.7 AuthorizationUpdate / AuthorizationUpdateResult（v1.9.6, v1.9.8）

授权变更请求与结果。`AuthorizationUpdate` 定义在 `contract.py`，`AuthorizationUpdateResult` 定义在 `authorization.py`。

```python
@dataclass
class AuthorizationUpdate:
    task_id: str | None = None
    approved_scopes: list[RequestedScope] | None = None
    clear_task_scope: bool = False
    clear_session_scopes: bool = False
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| task_id | `str \| None` | `None` | 目标 task ID |
| approved_scopes | `list[RequestedScope] \| None` | `None` | 待写入的授权范围（由 engine 转换为 ApprovalScope） |
| clear_task_scope | `bool` | `False` | 清理该 task_id 的 scopes |
| clear_session_scopes | `bool` | `False` | 清理所有 session scopes |

```python
@dataclass
class AuthorizationUpdateResult:
    events: list[tuple[str, dict]] = field(default_factory=list)
    changed: bool = False
    has_session_scopes: bool = False  # v1.9.8: 用于 set_mode 设置 controller.auto_execute
```

**数据流**：controller 创建 `AuthorizationUpdate` → agent 透传给 `engine.apply_update()` → engine 返回 `AuthorizationUpdateResult` → agent 从 events 发射 hook。agent 不理解授权语义，只做搬运。

---

### 1.8 ModeAction / ModeController / TaskModeController（v1.9.6, v1.9.8）

Task mode 控制器协议。定义在 `controller.py`。

```python
@dataclass
class ModeAction:
    kind: str                  # reply | run_prepare | run_execute | await_user | cancel
    query: str | None = None
    extra_system: str = ""
    task_id: str | None = None
    response: str | None = None
    authorization_update: AuthorizationUpdate | None = None
```

| kind 值 | 含义 |
|---------|------|
| run_prepare | 运行 prepare dry-run pipeline |
| run_execute | 运行正常 execute pipeline |
| await_user | 展示合同，等待用户确认/取消/补充 |
| reply | 任务完成，返回最终结果 |
| cancel | 任务取消 |

```python
class ModeController(ABC):
    def handle_turn(self, user_input: str) -> ModeAction: ...
    def on_pipeline_done(self, action: ModeAction, outcome: PipelineOutcome) -> ModeAction: ...
    def reset(self) -> list[tuple[str, dict]]: ...
    def is_idle(self) -> bool: ...
```

`TaskModeController` 是当前唯一实现。纯状态机，不持有 agent 或 engine 引用。所有数据通过方法参数输入、返回值输出。

v1.9.8 新增 `auto_execute` 属性（默认 False）：由 agent 在 set_mode 时根据 `_switch_policy` 返回的 `has_session_scopes` 设置。`auto_execute=True` 时，`handle_turn` 从 idle 直接返回 `run_execute`（跳过 prepare/contract/confirm）。

v2.0 新增 `MAX_CLARIFICATION_TURNS` 类常量（默认 3）：awaiting_confirmation 阶段，用户提供额外信息触发 re-prepare 的最大轮数。超过此限制后，controller 返回 `await_user` 并提示用户必须 confirm 或 cancel，不再接受补充信息。

```python
class TaskModeController(ModeController):
    MAX_CLARIFICATION_TURNS = 3  # v2.0: 最大 re-prepare 轮数
```

---

### 1.9 ContinuousRunner / Trigger（v2.0）

连续模式运行器。定义在 `runner.py`。外部组件，通过 `agent.chat()` 和 `agent.abort()` 公共 API 驱动 agent，不修改 agent 内部状态。

依赖方向：Runner -> Agent -> Engine。Trigger 无 agent 引用。

#### Trigger

```python
class Trigger(ABC):
    @abstractmethod
    def poll(self) -> str | None:
        """Return input string if triggered, None otherwise."""
        ...
```

触发源抽象基类。`poll()` 被 ContinuousRunner 周期性调用，返回非 None 字符串时触发一次 `agent.chat()`。

#### ContinuousRunner

```python
class ContinuousRunner:
    def __init__(
        self,
        agent: LlamAgent,
        triggers: list[Trigger],
        *,
        poll_interval: float = 1.0,
        task_timeout: float = 0,
        on_timeout: str | Callable = "abort",
    )
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| agent | `LlamAgent` | -- | Agent 实例（应处于 continuous 模式） |
| triggers | `list[Trigger]` | -- | 触发源列表 |
| poll_interval | `float` | `1.0` | 轮询周期（秒） |
| task_timeout | `float` | `0` | 单任务应用层超时（秒），0 = 无超时 |
| on_timeout | `str \| Callable` | `"abort"` | 超时动作：`"abort"` 调用 `agent.abort()`，或传入 callable 自定义行为 |

##### run

```python
def run(self) -> None
```

主循环。阻塞当前线程，顺序轮询所有 trigger，有输入时调用 `_run_task()`。直到 `stop()` 被调用。

##### stop

```python
def stop(self) -> None
```

通知 runner 停止。可从任意线程调用。内部设置停止事件并调用 `agent.abort()` 中止当前运行中的任务。

##### get_log（v2.0.2）

```python
def get_log(self) -> list[TaskLogEntry]
```

返回 task_log 副本（线程安全，可在 runner 运行时调用）。

##### clear_log（v2.0.2）

```python
def clear_log(self) -> None
```

清空 task_log。

##### _run_task（内部方法）

```python
def _run_task(self, task_input: str, trigger: Trigger) -> None
```

执行单次任务，记录 `TaskLogEntry` 到 `task_log`。若 `task_timeout > 0`，启动看门狗线程计时，超时后调用 `on_timeout` 动作。

#### TaskLogEntry（v2.0.2）

```python
@dataclass
class TaskLogEntry:
    trigger_type: str    # trigger 类名，如 "TimerTrigger"
    input: str           # trigger.poll() 返回的输入
    output: str          # agent.chat() 响应
    status: str          # "completed" | "error"
    error: str | None    # 错误信息（status="error" 时）
    started_at: float    # time.time() 开始时间
    duration: float      # 执行耗时（秒）
```

纯数据类，不引用 agent/engine。

**使用示例：**

```python
from llamagent.core.runner import ContinuousRunner, TimerTrigger

agent.set_mode("continuous")
runner = ContinuousRunner(agent, [TimerTrigger(60, "check system health")])

# 主线程阻塞运行：
runner.run()

# 从其他线程停止：
runner.stop()
```

#### TimerTrigger

```python
class TimerTrigger(Trigger):
    def __init__(self, interval: float, message: str)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| interval | `float` | -- | 触发间隔（秒） |
| message | `str` | -- | 触发时返回的固定输入字符串 |

定时触发器。首次 `poll()` 初始化时间戳（不触发），之后每隔 `interval` 秒返回 `message`。

#### FileTrigger

```python
class FileTrigger(Trigger):
    def __init__(self, watch_dir: str, *, message_template: str = "New files detected: {files}")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| watch_dir | `str` | -- | 监视目录路径 |
| message_template | `str` | `"New files detected: {files}"` | 消息模板，`{files}` 占位符会被替换为逗号分隔的新文件名 |

文件触发器。检测 `watch_dir` 中新增的文件。首次 `poll()` 快照当前文件列表（不触发），之后每次检查新增文件。仅检测新增，不检测修改或删除。

---

### 1.10 Config

全局配置，每个实例独立持有状态。优先级链：环境变量 > YAML 文件 > 代码默认值。

```python
class Config:
    def __init__(self, config_path: str | None = None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| config_path | `str \| None` | `None` | YAML 配置文件路径。None = 自动发现（`llamagent.yaml` / `.llamagent/config.yaml`）。显式指定但文件不存在时 raise FileNotFoundError |

**配置加载流程**：`_set_defaults()` → `_load_yaml()` → `_apply_env_overrides()` → `_post_process()`

**YAML 自动发现顺序**（第一个找到的生效，单文件策略）：
1. `LLAMAGENT_CONFIG` 环境变量
2. 项目根目录 `llamagent.yaml`
3. `.llamagent/config.yaml`
4. `~/.llamagent/config.yaml`

#### 配置项一览

| 配置项 | 类型 | 默认值 | 环境变量 | 说明 |
|--------|------|--------|----------|------|
| model | `str` | 自动检测 | `MODEL_NAME` | 模型标识符 |
| max_context_tokens | `int` | 自动检测 | -- | 模型上下文窗口大小，通过 litellm 自动检测 |
| api_retry_count | `int` | `1` | -- | LLM API 调用失败重试次数 |
| system_prompt | `str` | *内置默认* | -- | 无 Persona 时的默认身份提示 |
| context_window_size | `int` | `20` | -- | 保留的对话轮次数 |
| context_compress_threshold | `float` | `0.7` | -- | 对话 token 超过 max_context_tokens 的此比例时触发压缩 |
| compress_keep_turns | `int` | `3` | -- | 压缩时保留最近的未压缩轮次数 |
| max_react_steps | `int` | `10` | `MAX_REACT_STEPS` | ReAct 循环最大步数 |
| max_duplicate_actions | `int` | `2` | `MAX_DUPLICATE_ACTIONS` | 同一操作连续重复超过此次数则中止 |
| react_timeout | `float` | `210.0` | `REACT_TIMEOUT` | 每个 ReAct 步骤的时间限制（秒） |
| max_observation_tokens | `int` | `2000` | -- | 单次工具返回结果的最大 token 数 |
| memory_mode | `str` | `"off"` | `MEMORY_MODE` | 记忆模式：`"off"` / `"autonomous"` / `"hybrid"` |
| chroma_dir | `str` | *（`retrieval_persist_dir` 别名）* | `CHROMA_DIR` | backward compat 别名，值跟随 `retrieval_persist_dir` |
| rag_top_k | `int` | `3` | `RAG_TOP_K` | 检索结果返回数量 |
| chunk_size | `int` | `500` | `CHUNK_SIZE` | 文档分块大小 |
| persona_file | `str` | `{BASE_DIR}/data/personas.json` | `PERSONA_FILE` | 角色定义 JSON 文件路径 |
| agent_tools_dir | `str` | `{BASE_DIR}/data/agent_tools` | `AGENT_TOOLS_DIR` | 自定义工具存储目录 |
| max_plan_adjustments | `int` | `7` | `MAX_PLAN_ADJUSTMENTS` | PlanReAct 执行中最大计划调整次数 |
| reflection_enabled | `bool` | `False` | -- | 反思评估开关 |
| reflection_score_threshold | `float` | `7.0` | `REFLECTION_SCORE_THRESHOLD` | 低于此分数触发教训保存（及 PlanReAct 下的重规划） |
| permission_level | `int` | `1` | `PERMISSION_LEVEL` | 无 Persona 时的兜底权限等级 |
| output_dir | `str` | `{BASE_DIR}/output` | `OUTPUT_DIR` | 文件输出目录 |
| skill_dirs | `list[str]` | `[]` | `SKILL_DIRS` | 额外的 skill 目录路径（追加到默认扫描路径之后） |
| skill_max_active | `int` | `2` | -- | 每轮最多激活的 skill 数量 |
| skill_llm_fallback | `bool` | `False` | -- | 开启 C 级兜底：B 级无候选时让 LLM 扫描全量 metadata 做语义匹配 |
| job_default_timeout | `float` | `300.0` | `JOB_DEFAULT_TIMEOUT` | Job 默认超时时间（秒） |
| job_max_active | `int` | `10` | `JOB_MAX_ACTIVE` | 同时运行的最大 Job 数量 |
| job_profiles | `dict` | `{}` | -- | Job execution profile presets (profile name → config dict) |
| workspace_id | `str \| None` | `None` | -- | Workspace ID (optional, for API session reuse; lazy-generated on first use if not set) |
| retrieval_persist_dir | `str` | `{BASE_DIR}/data/chroma` | `RETRIEVAL_PERSIST_DIR`（fallback: `CHROMA_DIR`） | 检索层持久化根目录 |
| embedding_provider | `str` | `"chromadb"` | `EMBEDDING_PROVIDER` | Embedding 提供者 |
| embedding_model | `str` | `""` | `EMBEDDING_MODEL` | Embedding 模型名（空=提供者默认） |
| memory_recall_mode | `str` | `"tool"` | `MEMORY_RECALL_MODE` | 记忆读取模式：`"off"` / `"tool"` / `"auto"` |
| memory_fact_fallback | `str` | `"text"` | `MEMORY_FACT_FALLBACK` | 事实提取失败兜底策略：`"text"`（存为纯文本 episode）/ `"drop"`（丢弃） |
| memory_recall_top_k | `int` | `5` | `MEMORY_RECALL_TOP_K` | recall 检索返回数量 |
| memory_auto_recall_max_inject | `int` | `3` | `MEMORY_AUTO_RECALL_MAX_INJECT` | auto recall 最大注入条数 |
| memory_auto_recall_threshold | `float` | `0.35` | `MEMORY_AUTO_RECALL_THRESHOLD` | auto recall 相似度阈值（低于此值不注入） |
| rag_retrieval_mode | `str` | `"hybrid"` | `RAG_RETRIEVAL_MODE` | RAG 检索模式：`"vector"` / `"lexical"` / `"hybrid"` |
| rag_rerank_enabled | `bool` | `True` | `RAG_RERANK` | 是否启用 LLM reranking |
| web_search_provider | `str` | `""` | `WEB_SEARCH_PROVIDER` | 搜索后端：`""` (自动检测) / `"duckduckgo"` / `"serpapi"` / `"tavily"` |
| web_search_num_results | `int` | `5` | `WEB_SEARCH_NUM_RESULTS` | 默认搜索返回结果数 |
| authorization_mode | `str` | `"interactive"` | -- | v1.9 授权模式：`"interactive"` / `"task"` / `"continuous"` |
| seed_scopes | `list[dict] \| None` | `None` | -- | Continuous mode 预授权 scope 列表，YAML `authorization.seed_scopes` 独立解析 |
| hooks_config | `dict \| None` | `None` | -- | YAML `hooks:` 段的原始解析结果。独立解析逻辑，不走 `_YAML_MAP`。LlamAgent 初始化时读取并注册为 ShellHandler |

**模型自动检测优先级：** `MODEL_NAME` 环境变量 > API Key 检测（DEEPSEEK > OPENAI > ANTHROPIC）> Ollama 兜底（`ollama_chat/qwen3.5:latest`）

**使用示例：**

```python
config = Config()
config.max_react_steps = 15
config.memory_mode = "autonomous"
agent = LlamAgent(config)
```

---

### 1.11 LLMClient

统一 LLM 调用接口，基于 LiteLLM 支持多模型后端。

#### 构造函数

```python
class LLMClient:
    def __init__(self, model: str, api_retry_count: int = 1)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | `str` | -- | 模型标识符，如 `"openai/gpt-4o-mini"` |
| api_retry_count | `int` | `1` | API 调用失败后重试次数（重试 N 次，总共 N+1 次调用） |

---

#### chat

```python
def chat(
    self,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    timeout: float | None = None,
)
```

底层 LLM 调用，返回完整响应对象，支持 function calling。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| messages | `list[dict]` | -- | 消息列表 `[{"role": "...", "content": "..."}]` |
| temperature | `float` | `0.7` | 创造性参数 |
| max_tokens | `int \| None` | `None` | 最大输出 token 数 |
| response_format | `dict \| None` | `None` | 强制输出格式，如 `{"type": "json_object"}` |
| tools | `list[dict] \| None` | `None` | OpenAI 格式的工具 schema 列表 |
| tool_choice | `str \| None` | `None` | 工具选择策略 `"auto"` / `"none"` / `"required"` |
| timeout | `float \| None` | `None` | 请求超时（秒） |

**返回值：** litellm 原始响应对象。

**异常：**
- `RuntimeError`：litellm 未安装时抛出
- `ContextWindowExceededError`：上下文窗口溢出时立即抛出（不重试）

**注意事项：**
- 内置指数退避重试逻辑
- `ContextWindowExceededError` 不会重试，直接抛出

---

#### chat_stream（v2.0.2）

```python
def chat_stream(
    self,
    messages: list[dict],
    temperature: float = 0.7,
    tools: list[dict] | None = None,
    timeout: float | None = None,
)
```

流式 LLM 调用，返回 chunk iterable。

每个 chunk 结构：
- `chunk.choices[0].delta.content` — 文本片段（可能为 None）
- `chunk.choices[0].delta.tool_calls` — tool call 增量片段（可能为 None）
- `chunk.choices[0].finish_reason` — `"stop"` / `"tool_calls"` / None

重试逻辑同 `chat()`。

---

#### ask

```python
def ask(self, prompt: str, system: str = "", **kwargs) -> str
```

单轮问答便捷方法，返回纯文本。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| prompt | `str` | -- | 用户提问内容 |
| system | `str` | `""` | 系统提示（可选） |
| **kwargs | -- | -- | 传递给 `chat()` 的额外参数 |

**返回值：** `str` -- 模型回复文本。litellm 未安装时返回 `"[LLM 不可用] ..."` 格式的错误消息。

---

#### ask_json

```python
def ask_json(self, prompt: str, system: str = "", **kwargs) -> dict
```

单轮问答返回解析后的 JSON。自动设置 `response_format={"type": "json_object"}`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| prompt | `str` | -- | 用户提问内容 |
| system | `str` | `""` | 系统提示（可选） |
| **kwargs | -- | -- | 传递给 `chat()` 的额外参数 |

**返回值：** `dict` -- 解析后的字典。

**JSON 解析策略：**
1. 直接 `json.loads()` 解析
2. 失败后尝试从 Markdown 代码块 ` ```json ... ``` ` 中提取
3. 仍然失败则返回 `{"raw_response": str, "error": "JSON 解析失败"}`

---

#### count_tokens

```python
def count_tokens(self, messages: list[dict] | str) -> int
```

估算消息列表或文本的 token 数。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| messages | `list[dict] \| str` | -- | 消息列表或纯文本字符串 |

**返回值：** `int` -- 估算的 token 数。

**注意事项：** 优先使用 `litellm.token_counter()`，不可用时按字符粗略估算（1 字符 ~= 1 token）。

---

### 1.12 Persona

LlamAgent 角色定义。

```python
@dataclass
class Persona:
    name: str
    role_description: str = ""
    system_prompt: str = ""
    role: str = "user"
    permission_level: int | None = None
    persona_id: str = ""
    created_at: str = ""
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| name | `str` | -- | 显示名称 |
| role_description | `str` | `""` | 用户提供的角色描述，用于 LLM 自动扩展 |
| system_prompt | `str` | `""` | 完整角色提示（LLM 自动生成或手动指定） |
| role | `str` | `"user"` | 权限角色 `"admin"` \| `"user"` |
| permission_level | `int \| None` | `None` | 权限等级；None 时按 role 自动设置（admin=3, user=1） |
| persona_id | `str` | `""` | 唯一 ID；空则从 name 自动生成 |
| created_at | `str` | `""` | 创建时间（ISO 格式，自动生成） |

#### is_admin 属性

```python
@property
def is_admin(self) -> bool
```

是否为管理员角色。等价于 `self.role == "admin"`。

#### to_system_prompt

```python
def to_system_prompt(self) -> str
```

将角色信息转换为系统提示。优先使用 `system_prompt` 字段；为空时从 `role_description` 构建。

**使用示例：**

```python
persona = Persona(
    name="Code Llama",
    role_description="前端开发专家",
    role="user",
)
agent = LlamAgent(persona=persona)
```

---

### 1.13 PersonaManager

角色管理器：创建、列表、加载、删除角色。

#### 构造函数

```python
class PersonaManager:
    def __init__(self, storage_path: str, llm=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| storage_path | `str` | -- | 角色定义 JSON 文件路径 |
| llm | `LLMClient \| None` | `None` | 默认 LLM 客户端实例，用于 `create()` 时自动扩展 system_prompt |

---

#### create

```python
def create(
    self,
    name: str,
    role_description: str,
    role: str = "user",
    permission_level: int | None = None,
    persona_id: str = "",
    llm=None,
) -> Persona
```

创建并保存一个新角色。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| name | `str` | -- | 显示名称 |
| role_description | `str` | -- | 角色描述 |
| role | `str` | `"user"` | 权限角色 `"admin"` \| `"user"` |
| permission_level | `int \| None` | `None` | 权限等级；None 按 role 自动设置 |
| persona_id | `str` | `""` | 唯一 ID；空则从 name 自动生成 |
| llm | `LLMClient \| None` | `None` | LLM 客户端实例，用于自动扩展 system_prompt |

**返回值：** `Persona` -- 创建的角色对象。

**异常：** `ValueError` -- persona_id 已存在时抛出。

**使用示例：**

```python
manager = PersonaManager("data/personas.json")
coder = manager.create("Code Llama", role_description="前端开发专家")
agent = LlamAgent(persona=coder)
```

---

#### get

```python
def get(self, persona_id: str) -> Persona | None
```

按 ID 获取角色。未找到返回 None。

---

#### list

```python
def list(self) -> list[Persona]
```

列出所有已创建的角色。

---

#### delete

```python
def delete(self, persona_id: str) -> bool
```

删除角色。成功返回 True，未找到返回 False。

**注意事项：** 不会清除关联的记忆数据，需手动清理。

---

## 2. 模块层 API

模块层 API 已分散到各模块独立文档，按模块查阅：

| 模块 | API 文档 | 概览 |
|------|---------|------|
| Tools | [`docs/modules/tools/api.md`](modules/tools/api.md) | [`overview.md`](modules/tools/overview.md) |
| Memory | [`docs/modules/memory/api.md`](modules/memory/api.md) | [`overview.md`](modules/memory/overview.md) |
| RAG | [`docs/modules/rag/api.md`](modules/rag/api.md) | [`overview.md`](modules/rag/overview.md) |
| Retrieval (shared) | [`docs/modules/retrieval/api.md`](modules/retrieval/api.md) | [`overview.md`](modules/retrieval/overview.md) |
| Job | [`docs/modules/job/api.md`](modules/job/api.md) | [`overview.md`](modules/job/overview.md) |
| Skill | [`docs/modules/skill/api.md`](modules/skill/api.md) | [`overview.md`](modules/skill/overview.md) |
| Reasoning | [`docs/modules/reasoning/api.md`](modules/reasoning/api.md) | [`overview.md`](modules/reasoning/overview.md) |
| Reflection | [`docs/modules/reflection/api.md`](modules/reflection/api.md) | [`overview.md`](modules/reflection/overview.md) |
| Safety | [`docs/modules/safety/api.md`](modules/safety/api.md) | [`overview.md`](modules/safety/overview.md) |
| Sandbox | [`docs/modules/sandbox/api.md`](modules/sandbox/api.md) | [`overview.md`](modules/sandbox/overview.md) |
| Child Agent | [`docs/modules/child_agent/api.md`](modules/child_agent/api.md) | [`overview.md`](modules/child_agent/overview.md) |
| Multi-Agent | [`docs/modules/multi_agent/api.md`](modules/multi_agent/api.md) | [`overview.md`](modules/multi_agent/overview.md) |
| MCP | [`docs/modules/mcp/api.md`](modules/mcp/api.md) | [`overview.md`](modules/mcp/overview.md) |

---

<!-- Legacy module sections removed in v1.7.1. See docs/modules/<name>/api.md for details. -->

## 3. 接口层

### 3.1 CLI（LlamAgentCLI）

终端聊天界面。使用 Rich 库美化输出，未安装时自动降级为纯文本模式。

```python
class LlamAgentCLI:
    def __init__(
        self,
        module_names: list[str] | None = None,
        persona_name: str | None = None,
        agent=None,
    )
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| module_names | `list[str] \| None` | `None` | 要加载的模块列表，None 表示全部加载 |
| persona_name | `str \| None` | `None` | 角色名称 |
| agent | `LlamAgent \| None` | `None` | 传入已创建的 Agent 实例（最高优先级） |

#### chat_mode

```python
def chat_mode(self)
```

进入交互式聊天模式（主循环）。支持斜杠命令和常规对话。

#### ask

```python
def ask(self, question: str)
```

单问模式：问一个问题，得到回答。

**CLI 斜杠命令：**

| 命令 | 说明 |
|------|------|
| `/quit` (`/exit`, `/q`) | 退出对话 |
| `/help` | 显示帮助信息 |
| `/status` | 查看 Agent 运行状态 |
| `/modules` | 查看已加载模块 |
| `/clear` | 清空对话历史 |

**启动方式：**

```bash
python -m llamagent                                     # 交互式聊天（默认）
python -m llamagent.interfaces.cli ask "今天天气如何"      # 单问模式
python -m llamagent.interfaces.cli --modules tools,rag   # 指定模块
python -m llamagent.interfaces.cli --no-modules          # 纯聊天模式
```

---

### 3.2 Web UI

基于 Gradio 的 Web 聊天界面。

#### create_web_ui

```python
def create_web_ui(agent) -> gr.Blocks
```

构建 Gradio Web 界面。

| 参数 | 类型 | 说明 |
|------|------|------|
| agent | `LlamAgent` | 已创建的 Agent 实例 |

**返回值：** `gr.Blocks` -- Gradio 界面对象。

**异常：** `ImportError` -- Gradio 未安装时抛出。

**功能：**
- 聊天对话（自动管理对话历史）
- 文档上传（加载到 RAG 知识库）
- Agent 状态面板
- 示例问题快速启动

#### launch_web_ui

```python
def launch_web_ui(demo: gr.Blocks, port: int = 7860)
```

启动 Web UI 服务器。

**启动方式：**

```bash
python -m llamagent --mode web
python -m llamagent.interfaces.web_ui
```

---

### 3.3 HTTP API Server

基于 FastAPI 的 RESTful API + WebSocket 服务。

#### create_api_server

```python
def create_api_server(
    module_names: list[str] | None = None,
    persona_name: str | None = None,
) -> FastAPI
```

创建 FastAPI 应用实例。

**异常：** `ImportError` -- FastAPI 未安装时抛出。

#### launch_api_server

```python
def launch_api_server(
    module_names: list[str] | None = None,
    persona_name: str | None = None,
    host: str | None = None,
    port: int | None = None,
)
```

启动 API 服务器。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| module_names | `list[str] \| None` | `None` | 要加载的模块列表 |
| persona_name | `str \| None` | `None` | 角色名称 |
| host | `str \| None` | `None` | 绑定地址（默认 `API_HOST` 环境变量或 `0.0.0.0`） |
| port | `int \| None` | `None` | 监听端口（默认 `API_PORT` 环境变量或 `8000`） |

#### API 端点

##### POST /chat

发送聊天消息。

**请求体 (ChatRequest)：**

```python
{
    "message": str,              # 用户消息内容（1-10000 字符）
    "session_id": str | None,    # 会话 ID（可选，省略使用默认会话）
}
```

**响应体 (ChatResponse)：**

```python
{
    "reply": str,        # Agent 回复内容
    "session_id": str,   # 会话 ID
    "model": str,        # 使用的模型名称
}
```

**认证：** Bearer Token（`API_AUTH_TOKEN` 环境变量配置，为空时跳过认证）。

---

##### GET /status

Agent 状态和健康检查。无需认证。

**响应体 (StatusResponse)：**

```python
{
    "status": str,              # "healthy" | "degraded"
    "version": str,             # 服务版本
    "model": str,               # 当前使用的模型
    "uptime_seconds": float,    # 服务运行时长（秒）
    "modules": dict,            # 已加载模块
}
```

---

##### GET /modules

获取模块列表。

**响应体：** `list[ModuleInfo]`

```python
[
    {
        "name": str,          # 模块名称
        "description": str,   # 模块描述
        "loaded": bool,       # 是否已加载
    }
]
```

---

##### POST /upload

上传文件到 RAG 知识库。

**请求：** multipart/form-data，files 字段包含文件。

**响应体 (UploadResponse)：**

```python
{
    "message": str,             # 上传结果描述
    "files_processed": int,     # 已处理文件数
}
```

**注意事项：** 需要 RAG 模块已加载，否则返回 400 错误。

---

##### POST /clear

清空对话历史。

**请求体：**

```python
{
    "session_id": str | None,    # 会话 ID（可选，省略使用默认会话）
}
```

**响应体：**

```python
{
    "status": str,       # "ok"
    "message": str,      # 操作结果描述
}
```

---

##### POST /mode（v2.0.1）

切换 Agent 模式。

**请求体 (ModeRequest)：**

```python
{
    "mode": str,                 # 目标模式：interactive / task（continuous 返回 501）
    "session_id": str | None,    # 会话 ID（可选）
}
```

**响应体 (ModeResponse)：**

```python
{
    "mode": str,         # 当前模式
    "config": {          # mode-related 配置值
        "max_react_steps": int,
        "max_duplicate_actions": int,
        "react_timeout": float,
        "max_observation_tokens": int,
    },
}
```

**错误响应：**
- `400`：无效模式名或任务执行中无法切换
- `501`：continuous 模式不支持通过 API 切换

---

##### GET /mode（v2.0.1）

查看当前 Agent 模式和配置。

**查询参数：** `session_id`（可选）

**响应体：** 同 `POST /mode` 的 `ModeResponse`。

---

##### POST /abort（v2.0.1）

发送中止信号。当前原子操作完成后停止，后续操作不再执行。

**查询参数：** `session_id`（可选）

**响应体 (AbortResponse)：**

```python
{
    "success": bool,     # 是否成功发送中止信号
    "message": str,      # 状态消息
}
```

---

##### WS /ws/chat

WebSocket 流式聊天。

**协议：**
1. 连接后，若配置了认证，客户端发送 `{"type": "auth", "token": "<key>"}`
2. 客户端发送 `{"type": "message", "content": "你好"}`
3. 服务端流式返回 `{"type": "chunk", "content": "..."}`
4. 最终发送 `{"type": "done", "content": "完整回复"}`
5. 出错时返回 `{"type": "error", "content": "错误消息"}`

---

#### 中间件

- **CORS 中间件：** 默认允许所有来源（生产环境应替换为具体前端域名）
- **速率限制中间件：** 每 IP 每 60 秒最多 60 次请求（/status、/docs 等路径豁免）
- **全局异常处理：** 捕获未处理异常，防止堆栈信息泄漏

---

### 3.4 入口函数（create_agent / main）

#### create_agent

```python
def create_agent(
    module_names: list[str] | None = None,
    persona_name: str | None = None,
    config_path: str | None = None,
) -> LlamAgent
```

Agent 工厂函数，所有接口（CLI / Web / API）共享。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| module_names | `list[str] \| None` | `None` | 要加载的模块列表。`None` = 全部加载，`[]` = 不加载（纯聊天模式） |
| persona_name | `str \| None` | `None` | 角色名称，None 使用默认身份 |
| config_path | `str \| None` | `None` | YAML 配置文件路径，None 使用自动发现 |

**返回值：** 配置好的 `LlamAgent` 实例。

**可用模块注册表 (AVAILABLE_MODULES)：**

| 名称 | 导入路径 |
|------|----------|
| safety | `llamagent.modules.safety.SafetyModule` |
| tools | `llamagent.modules.tools.ToolsModule` |
| job | `llamagent.modules.job.JobModule` |
| rag | `llamagent.modules.rag.RAGModule` |
| memory | `llamagent.modules.memory.MemoryModule` |
| skill | `llamagent.modules.skill.SkillModule` |
| reflection | `llamagent.modules.reflection.ReflectionModule` |
| planning | `llamagent.modules.reasoning.PlanningModule` |
| mcp | `llamagent.modules.mcp.MCPModule` |
| multi_agent | `llamagent.modules.multi_agent.MultiAgentModule` |
| sandbox | `llamagent.modules.sandbox.SandboxModule` |
| child_agent | `llamagent.modules.child_agent.ChildAgentModule` |

#### main

```python
def main()
```

主入口：解析命令行参数并启动对应接口。

```bash
python -m llamagent                                    # 默认 CLI 交互聊天
python -m llamagent --mode web                         # 启动 Web UI
python -m llamagent --mode api                         # 启动 HTTP API 服务
python -m llamagent --modules tools,rag,memory         # 指定加载模块
python -m llamagent --no-modules                       # 纯聊天模式
python -m llamagent --persona CodeLlama                # 指定角色
python -m llamagent --config prod.yaml                 # 指定 YAML 配置文件
python -m llamagent --port 9000                        # 指定端口（Web/API）
```

---

