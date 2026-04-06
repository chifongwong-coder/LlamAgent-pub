# LlamAgent 版本变更记录

---

## v2.0.2 — Streaming + Runner Task Log

### Streaming（`chat_stream()`）

- **`ExecutionStrategy.execute_stream()`**：策略接口新增流式方法，默认返回 None（不支持）。Agent 通过返回值判断是否支持 streaming，不检查具体策略类型（符合 P1/P2）。
- **`SimpleReAct.execute_stream()`**：覆盖实现，调用 `run_react_stream()` 返回 generator。
- **`LLMClient.chat_stream()`**：传 `stream=True` 给 litellm，返回 chunk iterable。含重试逻辑。
- **`agent.run_react_stream()`**：流式 ReAct 循环，纯文本逐 token yield，tool call yield 状态消息（"[Calling tool...]"、"[tool done]"）。含完整循环保护（abort / duplicate detection / timeout / max_steps）。
- **`agent.chat_stream()`**：顶层入口，复制 chat() 完整生命周期（hooks、on_input、compression、on_output、history），执行部分通过策略接口路由。
- **on_output 事后处理**：用户看到 raw stream，on_output 在 stream 结束后处理累积文本存入 history。安全主防线 on_input 在 stream 前正常执行。
- **适用范围**：Interactive + SimpleReAct 完整 stream；PlanReAct / Task mode / Continuous mode fallback 到非 stream。
- **`_merge_tool_call_deltas()`**：静态辅助方法，按 index 累积合并 litellm stream tool_call 增量片段。

### Runner Task Log

- **`TaskLogEntry` dataclass**：结构化任务记录（trigger_type, input, output, status, error, started_at, duration）。
- **`ContinuousRunner.task_log`**：内存中的任务日志列表，每次 `_run_task` 自动记录。
- **`get_log()` / `clear_log()`**：公开方法，get_log 返回 copy（线程安全）。
- **`_run_task` 签名变更**：新增 trigger 参数，记录 `trigger_type=type(trigger).__name__`。
- **Web UI 适配**：删除 v2.0.1 的 agent.chat wrap hack，改用 `runner.get_log()` 读取日志。

### 接口层适配

- **CLI**：interactive 模式优先使用 `chat_stream` 逐字输出
- **Web UI**：`chat_respond` 改为 generator 支持 Gradio streaming
- **API Server**：WebSocket 改为真 streaming（通过 `chat_stream` 收集 chunks）

### 修改文件

- `llamagent/core/llm.py`：新增 `chat_stream()`
- `llamagent/core/agent.py`：新增 `execute_stream()` 接口 + `SimpleReAct.execute_stream()` + `chat_stream()` + `run_react_stream()` + `_merge_tool_call_deltas()`
- `llamagent/core/runner.py`：新增 `TaskLogEntry` + `task_log` + `get_log()` / `clear_log()` + `_run_task` 签名变更
- `llamagent/interfaces/cli.py`：`_process_chat` 使用 stream
- `llamagent/interfaces/web_ui.py`：chat_respond stream + 删除 chat wrap hack + 改用 get_log()
- `llamagent/interfaces/api_server.py`：WebSocket 真 streaming

### 测试

- 公共：新增 3 个流程测试（chat_stream text+tool / fallback+abort+controller / runner task_log）
- 内部：新增 4 个 mock 测试（merge delta + react_stream flow / on_output history / runner log internals）

---

## v2.0.1 — Interface Mode Switching + Examples + Bug Fixes

### 接口层 Mode Switching

三个接口（CLI、Web UI、API Server）现在支持 v2.0 的 mode switching、abort、task contract 交互。

#### CLI（`interfaces/cli.py`）
- **`/mode [name]`**：切换/查看模式（interactive / task / continuous）
- **`/abort`**：发送中止信号
- **Task Contract 展示**：检测 `[Task Contract]` 前缀，Rich Panel 黄色边框高亮展示
- **Continuous 交互**：选择 trigger 类型 → 配置参数 → `runner.run()` 阻塞主线程 → Ctrl+C 返回 interactive
- **confirm_handler**：交互式授权确认提示
- **`/status` 增强**：显示当前 mode

#### Web UI（`interfaces/web_ui.py`）
- **Mode dropdown**：interactive / task / continuous 三种模式
- **Continuous Runner 面板**：trigger 类型/参数配置 + Start/Stop/Refresh 按钮
- **confirm_handler**：auto-approve（Gradio 无法中途弹确认框，task contract 提供确认环节）

#### API Server（`interfaces/api_server.py`）
- **`POST /mode`**：切换模式（interactive / task，continuous 返回 501）
- **`GET /mode`**：查看当前模式和 mode-related config
- **`POST /abort`**：发送中止信号
- **confirm_handler**：auto-approve（调用者通过 contract 文本在应用层决策）

### 示例脚本

- **`examples/10_task_mode.py`**：Task mode 完整流程（set_mode → chat → contract → confirm → execute）
- **`examples/11_continuous_mode.py`**：ContinuousRunner + TimerTrigger + FileTrigger 用法

### Bug Fixes

- **版本号统一**：`__init__.py` 作为唯一版本来源（`"2.0.1"`），CLI/Web/API 全部导入，不再硬编码
- **main.py CLI 入口 bug**：`LlamAgentCLI` 构造函数参数传错，改为先 `create_agent()` 再传 agent
- **Shutdown 一致性**：CLI direct mode 补 shutdown、API Server lifespan shutdown 遍历调 `agent.shutdown()`、Web UI rebuild 时 shutdown 旧 agent
- **示例私有属性访问**：`examples/03_modules.py` 的 `agent._tools` 改用 `agent.get_all_tool_schemas()`

### 修改文件

- `llamagent/__init__.py`：版本号 → `"2.0.1"`
- `llamagent/main.py`：CLI 入口 bug 修复
- `llamagent/interfaces/cli.py`：mode/abort 命令 + task contract 展示 + continuous 交互 + confirm_handler
- `llamagent/interfaces/web_ui.py`：mode dropdown + continuous runner 面板 + confirm_handler
- `llamagent/interfaces/api_server.py`：mode/abort endpoints + confirm_handler + lifespan shutdown
- `examples/03_modules.py`：私有属性修复
- `examples/10_task_mode.py`：新文件
- `examples/11_continuous_mode.py`：新文件

---

## v2.0 — Mode-Aware Config + Continuous Runner + Abort Mechanism

### 设计动机

三种模式对应三种信任级别（Interactive=低、Task=中、Continuous=高），效率限制从"不信任 LLM"转向"按模式分级信任"。Continuous 模式需要外部驱动组件（Runner + Trigger），而非新的 Controller。Agent 核心保持不变，新能力通过外部组件实现。

### P0 — 核心改动

#### Mode-Aware 效率参数

- **`_MODE_DEFAULTS` 分级配置**：agent.py 类级常量，set_mode 时自动调整效率参数（max_react_steps, max_duplicate_actions, react_timeout, max_observation_tokens）到 config。Task 模式放宽限制（50 步 / 600s），Continuous 模式近乎无限（-1 = unlimited/disabled）。
- **-1 sentinel 语义**：`max_react_steps=-1` 表示无限步数，`max_duplicate_actions=-1` 表示关闭重复检测。run_react 循环条件适配：`while max_react_steps == -1 or steps < max_react_steps`。
- **Interactive config snapshot 恢复**：`__init__` 中存储 interactive 模式的 config 快照（`_interactive_config`），切回 interactive 时恢复用户 YAML 自定义值，而非硬编码默认值。异常路径同样恢复 snapshot。
- **`max_plan_adjustments` 不纳入**：PlanReAct 在 on_attach 时快照 config，set_mode 后修改对其无效，故不纳入 `_MODE_DEFAULTS`。

| 参数 | Interactive | Task | Continuous |
|------|------------|------|-----------|
| max_react_steps | 10 | 50 | -1（无限） |
| max_duplicate_actions | 2 | 5 | -1（关闭） |
| react_timeout | 210s | 600s | 600s |
| max_observation_tokens | 2000 | 5000 | 10000 |

#### ContinuousRunner + Trigger ABC

- **`ContinuousRunner` 外部驱动组件**：新建 `llamagent/core/runner.py`，不修改 agent 核心。Runner 通过 trigger polling 驱动 agent.chat()，主循环阻塞直到 stop()，单个任务异常不杀主循环。
- **`Trigger` 抽象基类**：`poll() -> str | None`，触发源产出输入，不知道 agent 的存在。依赖方向：Runner -> Agent -> Engine，Trigger 不引用 Agent。
- **`task_timeout` watchdog**：可选应用级超时，超时后执行 on_timeout 行为。`task_timeout=0` 表示无监控。
- **`on_timeout` 扩展接口**：默认 `"abort"` 调用 agent.abort()；部署者可传 callable 自定义行为（记日志、发告警、降级等）。

#### Abort 机制

- **`agent.abort()` 公开方法**：设置 `_abort = True`，语义为"已完成操作保留，当前原子操作完成后停止，后续操作不再执行"。
- **`chat()` 入口重置**：chat() 是 `_abort` 唯一重置点，保证 task mode 下 abort 跨 pipeline 传播（prepare abort 后 execute 也 abort）。
- **`run_react` 双检查**：外层 while 循环顶部（LLM 调用前）+ 内层 tool_call 循环（每个工具调用后），只检查不重置。
- **`ReactResult.terminal` 字段**：`terminal: bool = False`，标记不可恢复结果。`status="aborted"` 和 `status="context_overflow"` 设为 terminal=True，其余（error/timeout/max_steps）为 False（可 replan）。
- **PlanReAct terminal 检查**：通过 `result.terminal` 数据契约判断是否退出，与 abort 机制零耦合。

### P1 — 增强改动

- **内置 TimerTrigger**：定时执行固定任务，`time.time() - last_fire > interval` 判断触发。
- **内置 FileTrigger**：目录监听，`os.listdir()` diff 检测文件变化触发。
- **`TaskContract.open_questions`**：prepare 阶段注册 `_report_question` 专用工具，LLM 调用它报告不确定项，controller 在 `_on_prepare_done` 中提取并写入 contract。
- **`clarification_turns` 硬限制**：`MAX_CLARIFICATION_TURNS=3`，re-prepare 超过上限后强制用户选择取消或直接执行（yes/no）。

### 新增文件

- `llamagent/core/runner.py`：ContinuousRunner + Trigger ABC + TimerTrigger + FileTrigger

### 修改文件

- `llamagent/core/agent.py`：ReactResult.terminal 字段、_abort + abort() 方法、chat() 重置、_MODE_DEFAULTS + _interactive_config、_register_prepare_tools、run_react -1 条件适配
- `llamagent/core/controller.py`：MAX_CLARIFICATION_TURNS 硬限制、open_questions 提取
- `llamagent/modules/reasoning/module.py`：PlanReAct 内外层循环 terminal 检查

### 设计原则更新

- **P6 修订**："Don't implement hypothetical future requirements, but reserve extension interfaces for foreseeable needs"（不实现假设性需求，但为可预见需求预留扩展接口）。

### 测试

- 公共测试：新增 `tests/test_v2_features.py`（15 个流程测试）
- 内部测试：新增 `tests_internal/test_v2_features_mock.py`（29 个 mock 测试）
- 总计：566 tests（67 public + 485 internal mock + 14 real）

---

## v1.9.9 — Config-Driven Mode Initialization + Continuous Default Scope

### 设计动机

授权模式只能通过代码 `set_mode()` 切换，不能从 config 直接设置。Continuous 模式无 seed_scopes 时无法写任何文件。

### 核心改动

- `agent.__init__`：始终以 interactive 启动，读取 `config.authorization_mode` 后自动调用 `set_mode()`
- `agent.__init__`：无效模式值 warning + 回退 interactive
- `_switch_policy("continuous")`：无 seed_scopes 时默认创建 project 读写 scope（source="default"）
- `config.py`：新增 `AUTHORIZATION_MODE` 环境变量支持
- `llamagent.yaml.example`：seed_scopes 注释更新（task/continuous 都支持）

### 三种模式行为

| 模式 | 有 seed_scopes | 无 seed_scopes |
|------|---------------|---------------|
| interactive | - | 逐次确认 |
| task | 加载，auto_execute=True | session_authorize 询问用户 |
| continuous | 加载 | 默认 project 读写 |

### 测试重构

公共测试从 204 个单元测试合并为 52 个流程测试，所有原始断言保留。

- 公共测试：204 → 52（流程测试风格）
- 总计：522 tests（52 public + 456 internal mock + 14 real）

### 修改文件

- `llamagent/core/agent.py`：__init__ config-driven mode init + 无效模式校验
- `llamagent/core/authorization.py`：continuous 默认 project scope
- `llamagent/core/config.py`：AUTHORIZATION_MODE env var
- `llamagent.yaml.example`：seed_scopes 注释更新
- `tests/` 全部 14 个测试文件：合并为流程测试

---

## v1.0 — 初始架构（2026-03-12）

### Core 层

- **ReAct 循环移入 core**：将 ReAct（Thought → Action → Observation）从 `modules/reasoning/` 移入 `core/agent.py`，作为统一的底层执行引擎。原因：工具调用是 Agent 核心能力，不应依赖可选模块。
- **工具调用移入 core**：工具调用逻辑（function calling 循环）从 `modules/tools/` 移入 core 的 ReAct Action 阶段。原因：与 ReAct 统一，模块只负责注册工具，不负责调用。
- **废弃 `on_execute` callback**：用可插拔执行策略（Strategy Pattern）替代 `on_execute`。原因：`on_execute` 模块互斥竞争（第一个非 None 拦截），无法优雅组合多种执行方式。
- **`call_tool()` + `pre_call_check` 回调**：core 提供统一的工具调用入口（查找 → 权限检查 → 执行），safety 模块通过回调槽位注入权限检查逻辑。原因：分离调用机制与安全策略，core 只含最小兜底（`safety_level >= 2` 拒绝）。
- **执行策略模式（SimpleReAct / PlanReAct）**：策略负责组装 `tools_schema` + `tool_dispatch` 并调用 `run_react()`，ReAct 引擎本身不感知工具来源。原因：支持多种执行方式（直接 ReAct、先规划再 ReAct），策略可注入额外内部工具（如 `replan`）。
- **循环保护**：内置 `max_react_steps`、重复检测、`react_timeout`、`max_observation_tokens` 截断、`ContextWindowExceededError` 兜底。原因：防止 Agent 陷入无限循环或 token 溢出。
- **`run_react()` 参数化接口**：`run_react(messages, tools_schema, tool_dispatch)` 三参数由调用方提供，引擎无状态。原因：让任何策略都能自由组合工具集，不修改 core。

### 模块层

- **四层工具体系**：`default` / `common` / `admin` / `agent` 四层权限矩阵，通过 System Prompt + function calling schema 按角色过滤可见性。原因：区分核心工具、通用工具、管理员工具和用户自建工具的不同权限需求。
- **Memory 三档模式**：`off` / `autonomous` / `hybrid`，工具化接入（`save_memory` / `recall_memory`）。原因：让模型自主决定何时读写记忆，`hybrid` 增加写入兜底避免遗漏。
- **RAG 工具化（Agentic RAG）**：`on_context` 改为只注入知识库使用指南，模型通过 `search_knowledge` 工具自主检索。原因：统一 RAG 和 Memory 的接入模式，避免无关查询浪费 token。
- **PlanReAct 执行策略**：复杂度判断 → 任务分解 → 逐步 ReAct → 结果汇总 → 可选质量评估。原因：复杂任务需要先规划再执行，简单任务直接走 ReAct 节省开销。
- **`replan` 策略内部工具**：三条 replan 路径（模型主动、失败自动、质量驱动），共享 `max_plan_adjustments` 计数器。原因：让 Agent 面对变化能动态调整计划，同时防止无限调整。
- **Reflection 模块**：评估 + 教训管理（不做重试），教训通过 `on_context` 注入后续任务。原因：让 Agent 从失败中学习，避免重复犯错。
- **Multi-Agent 工具化**：`list_agents` / `create_agent` / `delegate` 注册为工具，`delegate` 用 `llm.ask()` 轻量执行。原因：模型自主判断是否需要协作，不依赖关键词检测。
- **MCP 桥接**：外部 MCP 工具包装为本地函数，通过 `agent.register_tool()` 注册。原因：统一内外部工具的调用方式。
- **Safety 三层机制**：输入过滤（`on_input`）+ 工具权限（`pre_call_check`）+ 输出脱敏（`on_output`），`permission_level` 与 `safety_level` 对比控制。原因：安全策略集中管理，编程调用时可选关闭。
- **Persona 角色系统**：`name` + `role_description` → LLM 自动扩展生成 `system_prompt`，携带 `permission_level`。原因：一句话创建角色，权限随角色走。

---

## v1.1 — 协议精化与结构化（2026-03-15）

### Core 层接口改进

- **`call_tool` 签名统一为 `call_tool(name: str, args: dict) -> str`**：原签名 `call_tool(name, **kwargs)` 改为显式 dict 参数。原因：与 LLM function calling 返回的 `arguments` dict 格式一致，消除调用协议不匹配。
- **`run_react()` 返回结构化 `ReactResult`**：替代原始 `str` 返回，包含 `status`（completed / max_steps / timeout / error / interrupted / context_overflow）、`error`、`steps_used`、`reason`。原因：执行策略需要根据循环终止原因做分支处理（如 PlanReAct 区分正常完成和中断）。
- **新增 `build_messages()` 方法**：统一消息构建逻辑（system prompt + summary + history + context + query），支持 `include_history`（PlanReAct 每步不带历史）和 `extra_system`（注入步骤指令）参数。原因：执行策略不应重复实现消息拼装，避免 SimpleReAct 和 PlanReAct 各写一套。
- **`run_react()` 新增 `should_continue` 回调参数**：每次工具调用后检查，返回 `str` 则中止循环（字符串作为 `ReactResult.reason`），返回 `None` 继续。原因：提供策略无关的循环中断机制，core 不再包含 PlanReAct 特有概念（如 `replanned` 状态），中断原因由策略自行定义。

### 数据模型改进

- **`summary` 与 `history` 分离**：压缩摘要独立存储在 `self.summary`，不混入 `self.history`，`build_messages()` 构建时将 summary 作为额外 system message 拼入。原因：避免摘要被 trim 误删，也便于独立管理摘要生命周期。
- **历史记录写入 `processed` 而非原始 `user_input`**：`self.history.append({"role": "user", "content": processed})`。原因：确保安全过滤后的输入不被绕过，history 中不残留被 on_input 拦截的内容。

### Planning 改进

- **Step 引入 `step_id` + `order` 双标识**：`step_id` 为唯一标识（如 "s1"），`order` 仅用于显示顺序。`depends_on` 改为引用 `step_id` 而非编号。原因：replan 重排步骤时编号变化会导致依赖关系断裂，`step_id` 稳定不变。
- **新增计划合法性校验（DAG 验证）**：`plan()` / `replan()` 返回后校验无环、无自依赖、引用存在、至少一个入口步骤。原因：防止 LLM 生成不合法的依赖关系导致死锁或无法执行。
- **移除 `replanned` 步骤状态**：被中断的步骤统一标记为 `failed`（附 result 说明原因），不再使用独立的 `replanned` 状态。原因：`replanned` 是 PlanReAct 特有概念，不应出现在通用 Step 数据结构中。
- **PlanReAct 利用 `should_continue` 实现 replan 中断**：replan 闭包设置 `interrupt_flag`，`should_continue` 回调检测后返回 `"replanned"` 中断循环，PlanReAct 自行决定是否刷新上下文。原因：将 replan 中断逻辑从 core 移到策略层，core 只提供通用中断机制。

### 上下文管理

- **部分压缩实现**：当 token 超过 `max_context_tokens × context_compress_threshold` 时，保留最近 `compress_keep_turns` 轮原文，将更早的对话 + 旧 summary 压缩为新摘要。原因：避免对话过长导致 token 溢出，同时保留近期上下文。
- **对话轮数裁剪**：每轮对话后检查，超过 `context_window_size` 则丢弃最早轮次。原因：简单有效的 history 长度控制，与 token 压缩互补。

### 接口层

- **三种交互方式**：CLI（Rich 美化）、Web UI（Gradio）、API Server（FastAPI + WebSocket）。原因：覆盖终端、浏览器、程序化调用三种场景。
- **接口层自动加载 Safety 模块**：所有接口在创建 Agent 时自动加载 safety。原因：确保外部入口始终受安全保护。

### 实现状态更新

- **大量 v1.0 设计落地实现**：`count_tokens`、`max_context_tokens`、API 重试、上下文压缩、`memory_mode` 三档、Safety `permission_level` / `pre_call_check`、RAG 工具化、Multi-Agent 工具化、MCP 直接注册、Persona JSON 持久化 + `_slugify()` ID 生成等。原因：v1.0 定义了目标架构，v1.1 将多项"待开发"标记推进为"已实现"。

---

## v1.2 — 受控执行面（Sandbox + Child Agent）

### 设计原则

- **核心思路：定义协议，不定义平台**：core 层只增加最小扩展点（`tool_executor` 属性 + `call_tool` 3 行分发），所有新能力通过模块和协议抽象实现。原因：保持 v1.1 的执行主链路不变，不推翻现有 `call_tool / pre_call_check / delegate` 设计。
- **可选性保证**：SandboxModule 不加载 → 行为与 v1.1 完全一致；ChildAgentModule 可加载但不调用 → 对 core 零影响；不安装沙箱依赖 → 优雅降级。原因：新能力不应强制所有用户买单。

### 安全沙箱（Sandbox Execution）

- **新增 `ExecutionPolicy` 数据类**：描述工具执行需要的运行时、隔离级别、文件系统访问、网络访问、资源限制等，不绑定具体平台。原因：用策略描述"需要什么"而非"用什么实现"，让同一份策略在不同后端（subprocess / Docker / gVisor）间透明切换。
- **`ToolInfo` 新增 `execution_policy` 可选字段**：默认 `None` = host 直调（v1.1 行为），非 `None` 则路由到沙箱执行。原因：向后兼容，只有显式标记的高风险工具才走沙箱。
- **`ExecutionBackend` / `ExecutionSession` 协议**：后端自描述能力（`capabilities()`），`BackendResolver` 按策略匹配最优后端。原因：新增后端只需注册，不改执行主链路。
- **`ToolExecutor` 分发层**：挂接在 `call_tool()` 和后端之间，根据 `execution_policy` 决定 host 直调还是沙箱执行，支持 `task_session` 模式复用会话。原因：透明分发，PlanReAct 等策略不感知底层沙箱实现。
- **`ExecutionResult` 结构化结果**：包含 `stdout` / `stderr` / `exit_code` / `artifacts` / `duration_ms` / `timed_out` 等，提供 `to_observation()` 转换为 ReAct 循环可用的字符串。原因：比纯 stdout 更丰富，便于父 Agent 判断步骤状态和汇总结果。
- **Core 层仅增加约 10 行代码**：`__init__` 新增 `self.tool_executor = None`，`call_tool()` 在权限检查后增加 `tool_executor` 分发（3 行）。原因：最小侵入，SandboxModule `on_attach` 时注入 executor，不加载则为 None 走原路径。
- **Phase 1 用 `LocalProcessBackend`（subprocess）**：macOS / Linux / CI 都能跑，无真正隔离但完整实现协议接口。原因：先跑通链路验证协议正确，后续替换为 gVisor / Docker 等真正隔离后端。
- **预置 policy 常量**：`POLICY_HOST` / `POLICY_READONLY` / `POLICY_UNTRUSTED_CODE` / `POLICY_SHELL_LIMITED`，用户也可自行组合。原因：降低使用门槛，常见场景开箱即用。

### 多子 Agent 控制（Child Agent Controller）

- **新增 `ChildAgentController`**：`spawn_child` / `wait_child` / `cancel_child` / `list_children` / `collect_results`，管理子 Agent 完整生命周期。原因：v1.1 的 `delegate()` 是轻量 `llm.ask()` 风格，不具备预算控制、取消、结果收集等能力。
- **`AgentExecutionPolicy` 能力边界描述**：`tool_allowlist` / `tool_denylist` / `execution_policy` / `budget` / `can_spawn_children` / `max_delegation_depth` / `history_mode` / `result_mode`。原因：子 Agent 不是父 Agent 的简单复制，而是带预算、带能力边界的受控执行单元。
- **`Budget` / `BudgetTracker` / `BudgetedLLM` 预算体系**：`max_tokens` / `max_time_seconds` / `max_steps` / `max_llm_calls`，`BudgetedLLM` 用 wrapper 模式包装 `LLMClient`，不侵入 LLMClient 核心。原因：防止子 Agent 资源失控，超预算直接抛 `BudgetExceededError`。
- **默认禁止子 Agent 再生子 Agent**：`max_delegation_depth = 1`，`can_spawn_children = False`。原因：防止 Agent 数量膨胀、预算失控、链路难追踪。
- **子 Agent 不继承父 Agent 完整上下文**：只接收 `task` / `context` / `tool_allowlist` / `budget`，不继承全量 history / summary / toolset。原因：更可控、更好 debug，避免子 Agent 被无关上下文干扰。
- **`TaskBoard` 轻量任务板**：记录 `task_id` / `parent_id` / `status` / `result` / `artifacts` / `metrics`，父 Agent 通过任务板做汇总和调度。原因：替代共享全局 history，实现结构化的任务追踪。
- **预设角色 policy 绑定**：`researcher` / `writer` / `analyst` / `coder` 各有默认的 `tool_allowlist` + `budget` + `execution_policy`，coder 强制走 sandbox。原因：角色不再只靠 prompt 区分，形成"角色 + 工具白名单 + 执行边界"的完整约束。
- **`delegate()` 保留为简化快捷方式**：简单任务继续用 `delegate()`，复杂多子 Agent 任务走 `ChildAgentController`。原因：向后兼容，不强制所有协作场景走重量级流程。
- **Phase 1 用 `InlineRunnerBackend`**：当前线程串行执行子 Agent，最简单无并发，但完整实现协议。原因：先验证协议正确，后续可替换为线程池 / 进程池等并发后端。

### 安全修复

- **LocalProcessBackend 环境隔离（S-A2）**：子进程不再继承宿主的完整环境变量，改用最小环境变量集（`PATH`、`HOME`、`LANG`、`TERM`），防止 API Key、凭据等敏感信息泄漏到沙箱进程。额外环境变量可通过 `ExecutionSpec.env_vars` 显式传递。
- **ToolExecutor session key 包含工具名（S-D1）**：session key 格式从 `"task_id:backend_name"` 改为 `"task_id:backend_name:tool_name"`，防止不同工具在同一 task 内共享 session 导致状态串扰。
- **子 Agent 工具深拷贝（C-A1）**：子 Agent 创建时通过 `copy.deepcopy(parent._tools)` 深拷贝父 Agent 工具，防止子 Agent 修改工具元数据（如注入 `execution_policy`）影响父 Agent。
- **工具 dict 包含 name 字段（Core-E5）**：`register_tool()` 现在在 tool dict 中存储 `"name"` 键，便于 `ToolExecutor`、`pre_call_check` 等接收 tool dict 的组件直接获取工具名称。
- **Runner 结果清理（C-B5/B6）**：`ChildAgentController.spawn_child()` 在将结果同步到 TaskBoard 后，自动清理 runner 内部的 `_results` 缓存，防止长时间运行时内存无限增长。
- **max_children 子 Agent 上限（C-B8）**：`ChildAgentController.__init__` 新增 `max_children: int = 20` 参数，`spawn_child()` 在达到上限时抛出 `RuntimeError`，防止无限制创建子 Agent。
- **SafetyModule.on_shutdown 关闭 FileHandler（Core-E4）**：`SafetyModule.on_shutdown()` 现在正确关闭审计日志的 `FileHandler`，防止文件句柄泄漏。
- **子 Agent 使用 copy.copy 构建 Config（C-C11/C12）**：子 Agent 创建流程从 `Config.__new__()` 改为 `copy.copy(parent.config)` + `LlamAgent(config)`，避免在 Config 新增属性时遗漏字段导致 AttributeError。
- **新增 POLICY_LOCAL_SUBPROCESS 预置策略常量**：`SandboxModule.auto_assign` 改为使用 `POLICY_LOCAL_SUBPROCESS`（而非 `POLICY_SHELL_LIMITED`），因为 `POLICY_SHELL_LIMITED` 要求 `isolation="process"` 而 `LocalProcessBackend` 仅支持 `isolation="none"`。
- **LocalProcessBackend supported_isolation 修正**：`capabilities()` 返回的 `supported_isolation` 从 `["none", "process"]` 修正为 `["none"]`，准确反映实际隔离能力。

### 明确的非目标

- 不做所有工具默认进沙箱、不做多层级子 Agent 递归、不做子 Agent 间自由协作、不做分布式 TaskBoard、不做 Firecracker / Wasmtime 后端。原因：v1.2 做"可控"，不做"大而全"，为 v1.3 预留扩展空间。

---

## v1.2.1 — 安全审计修复

### 工具系统修复

- **ToolsModule 桥接修复**：内部 registry（global_registry / agent_registry）的工具通过 `_bridge_to_core()` 同步到 `agent._tools`，LLM 才能看到和调用 builtin 工具和元工具。原因：v1.2 的 ToolsModule 注册工具到内部 registry 但从未桥接到 core，导致所有 builtin 工具不可达。
- **动态工具同步**：`create_tool` / `delete_tool` / `create_common_tool` / `promote_tool` 操作时同步更新 `agent._tools`。
- **agent 层工具 creator_id 过滤**：`register_tool()` 新增 `creator_id` 参数，`get_all_tool_schemas()` 按 `creator_id` 过滤 agent 层工具（仅创建者可见）。
- **桥接覆盖保护**：`_bridge_to_core()` 跳过 `agent._tools` 中已存在的工具，防止覆盖其他模块注册的工具。

### 安全架构简化

- **`pre_call_check` → `safety_loaded`**：将 Callable 回调替换为 bool 标记。SafetyModule 设 `safety_loaded=True`，core 兜底仅在 `safety_loaded=False` 时拒绝 sl>=2 工具。
- **SafetyModule 精简**：删除 `_check_permission()` 和 `_get_permission_level()`，SafetyModule 仅负责 `on_input`（输入过滤）+ `on_output`（输出脱敏）。
- **tier 控制可见性，safety_level 控制 core 兜底**：明确两个维度独立运作。

### 安全修复（C1-C3）

- **C1 execute_command**：`shell=True` → `shell=False` + `shlex.split()`，tier 改为 admin。
- **C2 create_tool**：新增 SandboxModule 前置检查；AST 白名单 + 受限命名空间 exec()（可配置 `allowed_modules`）。
- **C3 read_file**：`os.path.abspath` → `os.path.realpath`（防 symlink），路径限制在 cwd 内。

### 其他修复

- **H1**：child agent shutdown 移至 finally 块。
- **H2**：replan_closure 中 validate_plan() 加 try/except ValueError。
- **死代码清理**：删除 `reasoning/react.py`（ReAct 已在 core）、`guard.py` 的 `check_permission()` 方法。
- **SandboxModule auto_assign 阈值**：`safety_level >= 3` → `safety_level >= 2`。

---

## v1.3 — 区域安全体系

### 设计原则

- **核心思路：sandbox the environment, not the operations**。从"限制操作"转向"限制范围"。不限制工具能做什么，限制工具能在哪做。
- **威胁模型**：防止 LLM 幻觉导致的误操作（误删文件、误写配置等），不防恶意用户。

### 三层区域系统

- **新增三层安全区域**：以 `cwd` 为基准，划分 Playground（`{cwd}/llama_playground/`）、项目目录（`{cwd}/` 内）、外部（`{cwd}/` 以外）。原因：不同区域需要不同级别的保护，playground 内完全自由，项目内谨慎操作，外部基本锁死。
- **safety_level 新定义**：sl=1（只读/无副作用）和 sl=2（有副作用）不再是"能不能用"的门槛，而是与区域联动决定执行/确认/拒绝。原因：v1.2 的 safety_level 只用于 core 兜底，浪费了这个维度的表达能力。
- **Playground 自动创建**：框架启动时检测并自动创建 `llama_playground/`，退出时不清理。原因：给 agent 一个安全的自由操作空间。
- **`safety_loaded` 标记移除**：区域系统取代其职责。原因：v1.2 的 safety_loaded 是二元开关（加载/未加载），区域系统提供更细粒度的控制。

### path_extractor 机制

- **`register_tool()` 新增 `path_extractor` 参数**：工具注册时声明如何从参数中提取路径，`call_tool()` 调用它获取路径列表并做区域检查。原因：不同工具的路径在不同参数中（read_file 是 filename，execute_command 是命令字符串），不能用统一的 path_args。
- **`auto_path_extractor` 兜底**：未注册 `path_extractor` 的工具（如 LLM 创建的工具），自动检查参数名中包含 path/file/filepath 的参数值。原因：创建的工具不知道 path_extractor 概念，需要自动推断。
- **提取为空则跳过**：无路径工具（delegate、list_agents 等）不受区域系统影响。

### 确认机制

- **`confirm_handler` 回调**：`LlamAgent` 新增可选回调，区域检查需要确认时调用。CLI 用 input() 实现，Web/API 后续实现。原因：agent 层只负责发出确认信号，接口层决定怎么展示。
- **确认等待不计入 react_timeout**：防止用户确认时间导致 ReAct 超时。

### 工具变更

- **execute_command 恢复 shell=True**：管道、重定向、变量展开恢复可用，安全靠区域系统 + 命令路径扫描。原因：shell=False 严重限制命令能力，等于废了这个工具。
- **create_tool 大幅简化**：移除 AST 白名单、模块白名单、`allowed_modules` 参数、SandboxModule 前置要求。仅保留 builtins 黑名单（exec/eval/\_\_import\_\_）防套娃 + 字符串字面量路径扫描 + auto_path_extractor。原因：v1.2 的限制过于严格，创建的工具几乎无法使用。
- **read_file / write_file**：边界从单一目录扩展为三层区域。

### SafetyModule 变更

- **变为可选增强**：不再控制工具执行（无 safety_loaded），仅提供 on_input + on_output。原因：区域系统独立运行，不依赖任何模块。

### 三层安全防线

| 层次 | 机制 | 状态 |
|------|------|------|
| 区域系统 | path_extractor + confirm_handler | v1.3 实现 |
| 字符串扫描 | create_tool 字面量路径扫描 + builtins 黑名单 | v1.3 实现 |
| 沙箱隔离 | Docker/gVisor 进程隔离 | Phase 2（用户自行开启） |

### 明确的非目标

- 不做 Docker/gVisor 后端（Phase 2）
- 不新增功能模块
- 不改 callback pipeline 和 ReAct/PlanReAct 执行逻辑
- 不防恶意用户

---

## v1.4 — Skill 模块（任务级 Playbook 层）

### 设计原则

- **核心思路：Tool 负责"能做什么"，Skill 负责"这件事该怎么做"**。Skill 是任务级、可延迟加载的项目流程提示层（playbook layer），不是工具，不进 `call_tool()`。
- **插入点**：`SkillModule.on_context()` 动态注入，与 RAG/Memory 的 `on_context` 注入模式一致。
- **渐进加载**：启动只读 `config.yaml` 元数据，激活时才加载 `SKILL.md` 正文。

### Skill 文件格式

- **config.yaml + SKILL.md 分离**：`config.yaml` 存结构化元数据（name、description、tags、aliases、invocation），`SKILL.md` 保持纯自然语言 playbook。配置和内容分离，SKILL.md 干净无 frontmatter。
- **兼容 Agent Skills 开放格式**：目录结构兼容 OpenAI Codex / Claude Code / Agent Skills 开放规范，可选 `references/` 和 `assets/`（v1.4 不自动加载）。
- **四级扫描路径**：`.llamagent/skills/`（项目级）→ `.agents/skills/`（兼容）→ `~/.llamagent/skills/`（用户级）→ `~/.agents/skills/`（兼容），高优先级覆盖低优先级。

### 三级激活匹配

- **A 级：`/skill` 命令**：`on_input` 拦截 `/skill <name>`，精确 lookup name/alias，确定性激活。一轮有效，`_forced_skill` 在 on_context 后清除。不检查 invocation。
- **B 级：Tag 匹配 + LLM 消歧**：对 query 分词归一化（单复数、时态变化、双辅音还原、静默 e 恢复），生成变体集，与 tag 的变体集做精确交集匹配。0 命中进 C 级或不激活，1 命中直接激活，2+ 命中 `ask_json()` LLM 消歧。含连字符的 tag 先分词再逐个归一化匹配。
- **C 级：LLM 全量 metadata 扫描**（可选，默认关闭）：B 级 0 候选时，把全量 metadata 发给 LLM 语义匹配。`config.skill_llm_fallback = True` 开启。

### 注入格式

- **独立 skill block**：`[Active Skill: name]...[End Skill]` 追加到 context 末尾。不写入 history，不进入 summary 压缩。每轮重新判断激活。

### 语义收口

- **激活一轮有效**：`/skill X` 和 `activate()` API 仅作用于当前轮，下一轮重新匹配。
- **invocation 只约束 B/C 级**：`user-invocable` 的 skill 不会被自动触发，但 `/skill` 命令和 `activate()` API 可以无条件激活。
- **LLM 消歧输出契约**：B/C 级均使用 `ask_json()` 强制 JSON 格式 `{"selected": [...]}`，未知 skill name 静默忽略，解析失败 B 级回退第一个候选、C 级返回空。
- **信任边界**：正式 skill 扫描目录是受保护的人类维护目录，LLM 产物只能写入 draft/。
- **冲突解决**：同名 skill 高优先级先得；alias 不要求全局唯一，先注册先得；name 直接匹配优先于 alias。

### 边界约束

- **Skill 不进 Memory**：skill 是版本管理的流程知识，memory 是零散经验，不混用。
- **子 Agent 不继承父 skill**：child agent 独立做 skill 检索，显式声明才预加载。
- **不做 scripts 自动执行**：首版 instruction-only，脚本执行留到后续版本。
- **不自动加载 references/assets**：v1.4 仅加载 config.yaml + SKILL.md 两层。

### Config 新增

- `skill_dirs: list[str]`（默认 `[]`，环境变量 `SKILL_DIRS`）：额外 skill 目录路径。
- `skill_max_active: int`（默认 `2`）：每轮最多激活 skill 数量。
- `skill_llm_fallback: bool`（默认 `False`）：C 级 LLM 兜底开关。

### 推荐注册顺序更新

- `safety → tools → rag → memory → skill → reflection → planning → mcp → multi_agent`

---

## v1.5 — 工具系统升级（Workspace-First Workflow）

### 设计原则

- **核心思路：workspace-first workflow**。Agent 的文件操作以 workspace（会话级隔离目录）为中心，旧的 read_file/write_file/execute_command 全部删除，由面向 workspace 的新工具替代。
- **Core 零变更**：所有新能力通过新模块和新工具实现，core 的 call_tool / run_react / callback pipeline 无改动。

### 工具系统变更

- **删除旧工具**：`read_file`、`write_file`、`execute_command` 全部删除。原因：旧工具是单文件粒度、路径语义模糊，不适合 workspace 隔离模型。
- **新增 JobModule（6 tools）**：`start_job`（`wait=True/False`，替代 execute_command）等任务执行工具。`start_job` 支持同步/异步两种模式，覆盖原 execute_command 的所有场景。
- **新增 WorkspaceService（11 tools）**：`read_files`、`write_files`、`apply_patch`、`glob_files`、`search_text` 等面向 workspace 的文件操作工具集。所有路径相对于 workspace 根目录解析。
- **新增 ProjectSyncService（5 tools）**：workspace 与项目目录之间的同步工具（拉取、推送、diff 等），隔离 Agent 操作空间与用户项目目录。

### Workspace 隔离

- **会话级 workspace 目录**：`sessions/<workspace_id>/shared/`（共享区）+ `tasks/<task_id>/`（任务级隔离区），属于 Zone 1（完全自由操作）。
- **Zone 1 语义**：workspace 目录内的操作不触发确认，Agent 可自由读写。项目目录操作通过 ProjectSyncService 显式同步，受区域系统保护。

### apply_patch

- **结构化 search/replace**：`apply_patch` 采用结构化的搜索替换语义，定位目标代码块后原子替换。非纯文本追加，支持精确定位。
- **原子性 + changeset stack**：每次 apply_patch 生成一个 changeset 记录，支持回滚和审查。多次 patch 形成有序变更栈。

### PlanReAct 集成

- **task_id workspace 隔离**：PlanReAct 每个 step 使用独立的 `tasks/<task_id>/` 目录，步骤之间通过 shared/ 目录交换数据。原因：防止并发步骤文件冲突，同时支持步骤级产物追踪。

### LLM 引导

- **on_context 工具使用指南注入**：新增模块通过 `on_context` callback 注入工具使用指南（如 apply_patch 的格式说明、workspace 路径约定），引导 LLM 正确使用新工具。原因：新工具语义与旧工具差异大，纯 function schema 不足以让 LLM 正确使用。

### 明确的非目标

- Core 不改动（call_tool / run_react / callback pipeline 保持不变）
- 不做远程 workspace（当前仅本地文件系统）
- 不做工具自动迁移（旧 skill/prompt 中引用旧工具名需手动更新）

---

## v1.6 — 工具系统收口（Pack-Based Conditional Exposure）

### 设计原则

- **核心思路：让 tools 帮助 LLM 而不是阻碍 LLM**。在不降低可用性的前提下，适配不同任务场景和不同能力级别的模型。
- **区分"已安装能力"和"公开工具面"**：`agent._tools` 存储全部工具（installed capabilities），`get_all_tool_schemas()` 只返回当前应暴露的子集（public tool surface）。
- **默认暴露小而稳**：默认公开面 12 个工具，其余通过条件 Pack 按需暴露。
- **Core 零变更**：pack 过滤只在 `get_all_tool_schemas()` 层面，不改 `call_tool` / `run_react` / callback pipeline。

### Pack 机制

- **5 个条件 Pack**：`job-followup`（状态驱动）、`web`（skill 驱动）、`toolsmith`（skill 驱动）、`workspace-maintenance`（skill 驱动）、`multi-agent`（skill 驱动）。
- **触发时序**：`ToolsModule.on_input()` 重置 → `ToolsModule.on_context()` 状态评估 → `SkillModule.on_context()` skill 驱动追加。
- **Capability Hint Block**：每轮注入固定文本提示 LLM 存在哪些隐藏 pack（约 60-80 tokens）。
- **内置 Pack-Trigger Skills**：4 个内置 skill（toolsmith、web-access、workspace-ops、lightweight-collab），存放在 `llamagent/modules/skill/builtin_skills/`，优先级 builtin < user < project。
- **SkillMeta 新增**：`required_tool_packs: list[str]`，skill 激活时自动打开对应 pack。

### 工具合并

- `read_ranges` → 合并进 `read_files`（新增 `ranges` 参数）
- `preview_patch` + `replace_block` → 合并进 `apply_patch`（新增 `preview` 参数）
- `job_status` + `tail_job` + `collect_artifacts` → 合并为 `inspect_job`（非阻塞统一查询）
- JobModule 从 6 个工具精简为 4 个：`start_job`（默认面）+ `inspect_job`/`wait_job`/`cancel_job`（job-followup pack）

### project: 前缀限制

- 写操作工具（`write_files`、`move_path`、`copy_path`、`delete_path`）统一使用 `resolve_path_workspace_only()`，拒绝 project: 前缀和 workspace 外路径。project 修改只能通过 sync 通道（`apply_patch`/`sync_workspace_to_project`/`revert_changes`）。

### 文件类型支持

- `read_files` 新增文件类型自动检测（`_TEXT_EXTENSIONS` 白名单 + 内容探测），`mode` 参数（auto/text/binary）。二进制文件 auto 模式返回元信息，binary 模式返回 base64（50MB 上限）。
- `write_files` 新增 `mode` 参数（text/binary），binary 模式接受 base64 编码内容。

### 其他变更

- `web_search` 默认不注册（无真实搜索后端时返回错误）。`web_fetch` 进入 `web` pack。
- `list_agents`/`create_agent`/`delegate` 进入 `multi-agent` pack，`spawn_child` 保留默认面。
- `register_tool()` 新增 `pack` 参数。`ToolInfo` / `ToolRegistry` / `@tool` 装饰器同步支持。

### 明确的非目标

- 不改 core 的 call_tool / run_react / callback pipeline
- 不做 LLM 自主 pack 选择（只做规则驱动 + skill 驱动）
- 不做 Docker / container backend
- 不做远程 workspace
- 不做跨 session 的 pack 持久化（每轮重新评估）

---

## v1.7 — Memory & RAG 升级（共享检索层）

### 设计原则

- **核心思路：Memory 和 RAG 从 Level 1 升级到 Level 2**。Memory 做结构化事实记忆，RAG 做 Hybrid Search + Rerank。
- **共享底层，不合并上层**：Memory/RAG/Reflection 共享 `modules/retrieval/` 检索基础设施（EmbeddingProvider + VectorBackend + LexicalBackend + RetrievalPipeline），但上层语义独立。
- **模块通过工厂访问，不直接引用具体实现**：`create_pipeline()` 是唯一入口，换 backend 只改 factory.py。
- **EmbeddingProvider 与 VectorBackend 正交**：换 embedding 不动 backend，换 backend 不动 embedding。
- **Core 零变更**。

### 共享检索层（modules/retrieval/）

- **新增服务包**（非 Module，不注册到 agent）：EmbeddingProvider、VectorBackend、LexicalBackend、RetrievalPipeline、Reranker、Factory
- **默认实现**：ChromaDefaultEmbedding（all-MiniLM-L6-v2）、ChromaVectorBackend（cosine similarity）、SQLiteFTSBackend（FTS5，零外部依赖）、LLMReranker（用当前 LLM，失败退回 RRF）
- **Config**：`retrieval_persist_dir`（接口层名称，`chroma_dir` 为 backward compat 别名）、`embedding_provider`、`embedding_model`

### Memory 升级

- **结构化事实记忆**：MemoryFact 数据模型（kind/subject/attribute/value + strength/status），替代纯文本存储
- **FactCompiler**：best-effort LLM 事实提取，失败退回纯文本（`memory_fact_fallback` 配置）
- **FactMerger**：归一化精确去重（`(kind, subject, attribute)` 主键），insert/update/skip
- **读写解耦**：`memory_mode`（写入）和 `memory_recall_mode`（读取）正交。save_memory 和 recall_memory 独立注册
- **轻量自动 recall**（`memory_recall_mode=auto`）：前置跳过闲聊 → 向量检索 → 相似度阈值门控 → 注入高相关事实
- **hybrid on_output**：合并为单次 LLM 调用（should_store + facts + summary）
- **遗忘/衰减**：strength +0.1/次（上限 2.0），superseded 不参与检索
- **backend.py 已删除**：ChromaMemoryBackend 被共享层完全替代

### RAG 升级

- **结构感知分块**：DocumentChunker 按扩展名分发（MarkdownChunker/CodeChunker/PlainTextChunker），替代固定 500 字符
- **Hybrid Retrieval**：向量搜索 + SQLite FTS5 关键词搜索 + RRF 合并
- **可选 LLM Reranking**：`rag_rerank_enabled` 配置
- **P0 格式**：.txt/.md/.py/.js/.ts/.java/.go/.rs
- **search_knowledge 接口不变**，内部升级为 hybrid + rerank

### Reflection 迁移

- **LessonStore** 迁移到共享检索层：通过 factory 创建 pipeline，不再直接 import chromadb

### 明确的非目标

- Letta 式分层 Memory / A-Mem 图谱记忆 / GraphRAG / 完整 Agentic RAG
- 多 agent 共享 memory / RL 驱动 memory policy
- embedding 级语义去重（只做归一化精确去重）

---

## v1.7.1 — YAML 层次化配置

### 配置系统

- **YAML 配置支持**：Config 新增 `config_path` 构造参数，支持从 YAML 文件加载层次化配置
- **优先级链**：环境变量 > YAML 文件 > 代码默认值
- **自动发现**：`llamagent.yaml` / `.llamagent/config.yaml` / `~/.llamagent/config.yaml`（单文件生效，不做 merge）
- **显式路径失败报错**：`Config(config_path="xxx.yaml")` 文件不存在或解析失败时直接 raise，不静默降级
- **未知键告警**：YAML 中不在映射表的键产生 `logger.warning`
- **Env alias 冲突规则**：新名优先（`RETRIEVAL_PERSIST_DIR` 优先于 `CHROMA_DIR`）
- **复杂类型**：dict/list 字段（`job_profiles`、`skill_dirs`）只能通过 YAML 或代码设置，不支持环境变量覆盖子字段
- **模块零改动**：模块仍通过 `config.xxx` 扁平属性访问，YAML 层次结构只在 Config 内部解析
- **CLI 支持**：`--config` 参数指定 YAML 文件
- **pyyaml 降级**：无 pyyaml 时自动发现的 YAML 静默跳过，显式指定的 raise ImportError

### 新增文件

- `llamagent.yaml.example`：完整 YAML 配置模板

### 测试补全

- 新增 6 个模块专属测试（Memory/RAG/Retrieval/Safety/Multi-Agent/Config-YAML），内部测试从 262 增至 401

---

## v1.8 — 事件 Hook 系统

### 设计原则

- **Pipeline Callback 管轮级，事件 Hook 管工具级**：两层共存互不替代
- **只有 PRE_TOOL_USE 支持 SKIP**：其他事件均为观察性
- **Hook 容错优先**：Hook 自身异常不阻止主流程
- **Core 零模块依赖**：Hook 系统在 core 层实现，不依赖任何模块

### 事件 Hook

- **工具级事件**：`PRE_TOOL_USE`（可 SKIP/修改参数）、`POST_TOOL_USE`（含 result/duration_ms）、`TOOL_ERROR`（含 error）
- **轮级事件**：`PRE_CHAT`（观察性）、`POST_CHAT`（always-fire，含 blocked/completed 字段）
- **生命周期事件**：`SESSION_START`（首次 chat 触发一次）、`SESSION_END`（shutdown 时触发）
- **预留事件**：`PLAN_CREATED` / `STEP_START` / `STEP_END` / `REPLAN`（v1.8 不 emit）
- **call_tool() 统一包裹**：两条执行路径（直调 + executor/sandbox）均在同一层 hook 下

### 处理器

- **CallableHandler**：Python callable，代码注册，默认 priority=100
- **ShellHandler**：Shell command，YAML 注册，默认 priority=200
  - 通过 `$HOOK_*` 环境变量传递数据（JSON 序列化，排除 tool_info）
  - 退出码语义：0=CONTINUE，非0=SKIP（仅 PRE_TOOL_USE 生效）
- **HookHandler ABC**：预留 HttpHandler / AgentHandler 扩展接口

### Matcher 过滤

- `HookMatcher`：tool_name / tool_names / pack / safety_level，AND 逻辑
- `pack` 和 `safety_level` 依赖 `tool_info`，仅对 PRE_TOOL_USE 有效

### YAML 配置

- `hooks:` 段独立解析到 `config.hooks_config`（不走 `_YAML_MAP`）
- `_check_unknown_keys()` 跳过 hooks 子树
- LlamAgent 初始化时调用 `_register_yaml_hooks()` 注册

### 新增文件

- `llamagent/core/hooks.py`：HookEvent / HookContext / HookResult / HookMatcher / HookHandler / CallableHandler / ShellHandler / HookRegistration
- `tests/test_hooks.py`：14 个公开流程测试
- `tests_internal/test_hooks_mock.py`：19 个内部单元测试
- `docs/modules/hooks/overview.md`、`docs/modules/hooks/api.md`

### 修改文件

- `llamagent/core/agent.py`：`_hooks` / `_session_started` / `_in_hook` / `register_hook()` / `emit_hook()` / `_register_yaml_hooks()` / call_tool 集成 / chat 集成 / shutdown 集成
- `llamagent/core/config.py`：`hooks_config` / YAML hooks 段解析 / unknown-key 跳过
- `llamagent/core/__init__.py`：导出 HookEvent / HookContext / HookResult / HookMatcher / HookCallback / HookHandler
- `llamagent.yaml.example`：hooks 配置示例

### 测试

- 公开测试：125 → 139（+14）
- 内部测试：401 → 420（+19）
- 总计：540 → 573

### 明确的非目标

- 不做 HttpHandler / AgentHandler（接口预留）
- 不 emit 规划级事件（枚举预留）
- Hook 不复用 zone / sandbox 安全模型（受信开发者能力）
- 子 Agent 隔离（不继承父 Agent hooks）

---

## v1.8.1 — 真实 Web 搜索

### 搜索后端

- **替换 LLM 模拟搜索**：`web_search` 从 LLM 假搜索改为真实搜索 API 调用
- **可切换后端**：SearchBackend ABC + 三个实现（DuckDuckGo / SerpAPI / Tavily）
- **自动检测**：无配置时按 SERPAPI_KEY → TAVILY_API_KEY → DuckDuckGo 顺序选择后端
- **DuckDuckGo 兜底**：免费无需 API key，安装 `pip install ddgs` 即可使用
- **工厂模式**：`create_search_backend(config)` 统一入口，跟 retrieval 的 factory 一致

### 工具注册

- `web_search` 加 `@tool` 装饰器 + `pack="web"`，与 `web_fetch` 同组
- web-access builtin skill 更新 tags（新增 search/query/find/lookup）

### Config

- 新增 `web_search_provider`（`""` = 自动检测）和 `web_search_num_results`（默认 5）
- 环境变量覆盖：`WEB_SEARCH_PROVIDER` / `WEB_SEARCH_NUM_RESULTS`

### 新增文件

- `llamagent/modules/tools/web.py`：SearchBackend 抽象 + 三个实现 + 工厂函数
- `tests/test_web_search.py`：3 个真实搜索测试
- `tests_internal/test_web_search_mock.py`：15 个内部单元测试

---

## v1.8.2 — ask_user 结构化提问工具

### 设计原则

- **完全解耦**：ask_user 工具只认 `UserInteractionHandler.ask()` 接口，不绑定任何 I/O 实现
- **外部注入**：交互处理器由调用方（接口层/库用户）在 `register_module` 前设置 `agent.interaction_handler`
- **接口预留**：ABC 设计允许后续扩展异步/超时/通知模式（持续工作模式用）

### ask_user 工具

- `tier="default"`（始终可见），`pack=None`，`safety_level=1`
- 通过 `_handler` 属性注入 `UserInteractionHandler` 实例
- 无 handler 时返回错误提示（不 crash）
- 支持自由文本提问和选项选择

### 交互处理器

- `UserInteractionHandler` ABC：`ask(question, choices) -> str`
- `CallbackInteractionHandler`：委托到用户提供的 callable（主要集成入口）
- `CLIInteractionHandler`：便利示例，blocking `input()`

### 新增文件

- `llamagent/modules/tools/interaction.py`：UserInteractionHandler ABC + CallbackInteractionHandler + CLIInteractionHandler
- `tests/test_ask_user.py`：7 个公开测试
- `tests_internal/test_interaction_mock.py`：9 个内部测试

### 修改文件

- `llamagent/modules/tools/builtin.py`：新增 ask_user 工具
- `llamagent/modules/tools/module.py`：on_attach 注入 interaction handler
- `llamagent/core/agent.py`：`__init__` 新增 `interaction_handler` 属性
- `llamagent/modules/child_agent/module.py`：子 agent 继承 `interaction_handler`

---

## v1.9.0 — 统一授权底座重构

### 设计原则

- **Zone 不可绕过**：zone 判定是基础安全边界，HARD_DENY 永远不可覆盖
- **行为不变**：interactive 模式下用户体验与 v1.8.x 完全一致
- **结构化接口**：confirm_handler 从 `Callable[[str], bool]` 升级为 `Callable[[ConfirmRequest], ConfirmResponse]`
- **底座先行**：只做授权引擎基础，不做任何放权

### 授权引擎

- **AuthorizationEngine**：封装路径提取 + zone 评估 + policy 决策，从 LlamAgent 中独立出来
- **ZoneEvaluation**：逐路径判定结果（`list[ZoneDecisionItem]` + `overall_verdict`），支持多路径场景
- **InteractivePolicy**：v1.9.0 唯一策略，逐 item 确认 CONFIRMABLE，行为与 v1.8.x 等价
- **action 推导**：新增 `action` 字段（read/write/execute），从 safety_level 自动推导，不破坏现有工具

### Breaking change

- `confirm_handler` 签名变更：`Callable[[str], bool]` → `Callable[[ConfirmRequest], ConfirmResponse]`（有 bool backward compat）
- LlamAgent 删除 `_check_zone()` 和 `_extract_paths()`（搬入 AuthorizationEngine）

### 新增文件

- `llamagent/core/zone.py`：ZoneVerdict、ZoneDecisionItem、ZoneEvaluation、ConfirmRequest、ConfirmResponse
- `llamagent/core/authorization.py`：AuthorizationEngine、AuthorizationPolicy、InteractivePolicy、AuthorizationState、infer_action
- `tests/test_authorization.py`：20 个公开测试（zone 等价性 + 多路径 + 确认结构 + 全链路 + action 推导）
- `tests_internal/test_zone_mock.py`：7 个内部测试
- `tests_internal/test_authorization_mock.py`：18 个内部测试

### 修改文件

- `llamagent/core/agent.py`：call_tool 接入 engine.evaluate()；删除 _check_zone/_extract_paths；confirm_handler 签名变更；新增 _authorization_engine + mode 属性
- `llamagent/core/config.py`：新增 authorization_mode 字段
- `llamagent/core/__init__.py`：导出 zone/authorization 类型
- `llamagent/modules/tools/registry.py`：ToolInfo 新增 action 字段
- `llamagent/modules/tools/module.py`：_bridge_to_core 传 action
- `llamagent/modules/child_agent/module.py`：子 agent 复制 mode

---

## v1.9.1 — Task Mode Prepare / Contract

### 设计原则

- **Controlled dry-run**：Prepare 阶段读操作真实执行，写/执行操作只记录不执行
- **状态机驱动**：TaskModeState 管理 idle→preparing→awaiting_confirmation→executing→idle 状态流
- **chat() 单入口**：task mode 通过内部 `_handle_task_mode_turn()` 驱动，不新增外部 API
- **不做放权执行**：execute 阶段暂退回 InteractivePolicy（1.9.2 加 scope 匹配）

### Task Mode

- **TaskPolicy**：prepare 阶段 action=read 允许执行，action=write/execute 拦截并记录 pending scope
- **TaskModeState**：独立状态对象，存储 phase / original_query / pending_scopes / contract
- **TaskContract**：聚合 pending scopes 后生成的合同，包含 task_summary / planned_operations / requested_scopes / risk_summary
- **Scope 聚合**：`normalize_scopes()` 将逐文件路径归并为目录前缀，合同更可读

### 新增类型

- `RequestedScope`（core/zone.py）：zone + actions + path_prefixes + tool_names
- `TaskContract` + `TaskModeState`（core/contract.py）
- `ConfirmRequest.requested_scopes` 可选字段：合同确认时携带结构化范围

### 新增文件

- `llamagent/core/contract.py`
- `tests/test_task_mode.py`：11 个公开测试

### 修改文件

- `llamagent/core/zone.py`：RequestedScope + ConfirmRequest 扩展
- `llamagent/core/authorization.py`：TaskPolicy + normalize_scopes + AuthorizationEngine.set_mode()
- `llamagent/core/agent.py`：set_mode() + _task_mode_state + _handle_task_mode_turn() + chat() task mode 分流
- `llamagent/core/__init__.py`：导出新类型

---

## v1.9.2 — Task Scope Authorization

### 设计原则

- **开始前一次确认，执行中少打扰**：合同范围内的 CONFIRMABLE 操作自动通过
- **task_id 由 TaskModeState 持有**：不依赖 PlanReAct 私有的 `_current_task_id`
- **统一 helper**：`get_active_task_id()` 保证 scope 写入和查找同源
- **缩小范围**：合同确认支持 approved_scopes 缩窄授权（不可扩大）

### Scope 匹配

- `ApprovalScope`：zone + actions + path_prefixes + tool_names（AND 逻辑）
- `_matches_any_scope()`：CONFIRMABLE item 逐个匹配已批准 scope，命中则自动通过
- 未命中 → 退回逐次确认
- HARD_DENY 不受 scope 影响

### task_id 管理

- `TaskModeState.task_id`：idle→preparing 时生成 UUID，贯穿整个 task 生命周期
- `get_active_task_id()`：task mode state > _current_task_id（PlanReAct）
- PlanReAct 在 task mode 下复用已有 `_current_task_id`，不生成新 UUID

### Scope 生命周期

- 合同确认 → ApprovalScope 写入 `task_scopes[task_id]`
- 执行完成 / 取消 → `task_scopes[task_id]` 清空
- 切回 interactive → 全部 task_scopes 清空

### 新增类型

- `ApprovalScope`（core/zone.py）
- `ConfirmResponse.approved_scopes`（可选字段）

### 修改文件

- `llamagent/core/zone.py`：ApprovalScope + ConfirmResponse 扩展
- `llamagent/core/contract.py`：TaskModeState.task_id
- `llamagent/core/authorization.py`：AuthorizationState.task_scopes + TaskPolicy._decide_execute + _matches_any_scope + set_mode 清理 scope
- `llamagent/core/agent.py`：get_active_task_id() + _write_task_scopes + _run_execute 设/清 _current_task_id + scope 清理
- `llamagent/modules/reasoning/module.py`：PlanReAct 复用已有 _current_task_id
- `tests/test_task_scope.py`：16 个公开测试

---

## v1.9.3 — Continuous Mode + Seed Scopes

### 设计原则

- **无交互，基于预授权运行**：continuous mode 不调 confirm_handler
- **未命中直接拒绝**：CONFIRMABLE 操作不暂停等待，匹配 seed scope 则执行，否则拒绝
- **不自动越权**：无 seed scopes = 所有 CONFIRMABLE 操作被拒绝

### Continuous Mode

- `ContinuousPolicy`：ALLOW→执行，HARD_DENY→拒绝，CONFIRMABLE→匹配 session_scopes 或拒绝
- 复用 `_matches_any_scope()` 函数（v1.9.2）
- `AuthorizationState.session_scopes`：seed scopes 加载后存储

### Seed Scopes 配置

```yaml
authorization:
  mode: continuous
  seed_scopes:
    - scope: session
      zone: project
      actions: [write]
      path_prefixes: ["docs/", "src/"]
```

- `config.seed_scopes`：YAML 独立解析（类似 hooks_config）
- `_load_seed_scopes()`：set_mode("continuous") 时转换为 ApprovalScope

### 修改文件

- `llamagent/core/authorization.py`：ContinuousPolicy + session_scopes + set_mode("continuous") + _load_seed_scopes
- `llamagent/core/config.py`：seed_scopes 字段 + YAML 解析 + unknown-key 跳过
- `llamagent.yaml.example`：seed_scopes 配置示例
- `tests/test_continuous_mode.py`：13 个公开测试

---

## v1.9.4 — 治理、审计与可观测性

### 设计原则

- **纯匹配函数**：`_find_matching_scope` 保持纯逻辑，不做 uses 计数和事件发送
- **两阶段消费**：先收集 matched scopes，全部通过后才统一 uses += 1 + emit
- **事件通过 AuthorizationResult 返回**：engine 不直接调 hook 系统，agent 负责投递
- **scope 生命周期闭环可观测**：SCOPE_ISSUED → SCOPE_USED → SCOPE_REVOKED

### Scope 治理

- `ApprovalScope` 新增 created_at / expires_at / max_uses / uses / source 字段
- `_find_matching_scope` 自动跳过过期和超限的 scope
- 路径匹配修复：`_path_in_prefixes` 使用 `os.path.normpath` + sep 判断（修复 startswith 误匹配）

### 审计事件

- `SCOPE_ISSUED`：scope 创建时（合同确认 / seed scopes 加载）
- `SCOPE_USED`：scope 匹配命中时（两阶段消费后统一发出）
- `SCOPE_DENIED`：CONFIRMABLE 操作未匹配任何 scope 时
- `SCOPE_REVOKED`：scope 清理时（任务完成 / 取消 / 模式切换）

### AuthorizationResult

- `evaluate()` 返回 `AuthorizationResult(decision, events)` 而非 `str | None`
- engine 不 import HookEvent，只返回事件名字符串
- agent.call_tool 负责从 events 投递到 hook 系统

### 可观测性

- `agent.authorization_status()` 返回 mode + task_scopes + session_scopes 快照

### 修改文件

- `llamagent/core/zone.py`：ApprovalScope 6 个新字段
- `llamagent/core/hooks.py`：4 个新 HookEvent
- `llamagent/core/authorization.py`：AuthorizationResult + _find_matching_scope + _path_in_prefixes + 三个 Policy 返回 AuthorizationResult + 两阶段消费
- `llamagent/core/agent.py`：call_tool 适配 AuthorizationResult + 事件投递 + authorization_status + SCOPE_ISSUED/REVOKED emit + source/created_at
- `llamagent/core/__init__.py`：导出 AuthorizationResult

---

## v1.9.5 — SmartAgent → LlamAgent 重命名

纯机械替换，35 个文件 123 处。不保留 SmartAgent 向后兼容别名。

- `class SmartAgent` → `class LlamAgent`
- `SmartAgentCLI` → `LlamAgentCLI`
- 所有 import、type hint、docstring、example、README 同步更新

---

## v1.9.8 — Task Mode Session Scopes (Shared Authorization)

### 设计动机

Task 模式的 prepare 阶段依赖 LLM 调用工具来发现所需 scope，不可靠。用户希望预授权 project 读写权限后直接执行，而非每个任务都走 prepare → contract → confirm 流程。

### 核心改动

- `_switch_policy("task")` 复用 `_load_seed_scopes()`（与 continuous 模式同一代码路径）
- 无 seed_scopes 时，engine 通过 `ConfirmRequest(kind="session_authorize")` 询问用户是否开放 project 访问
- 用户同意 → engine 构建 project 读写的 session scope（source="session_authorize"）
- `TaskPolicy._decide_execute` 合并查 `task_scopes[task_id] + session_scopes`（task-first 优先级）
- `TaskModeController` 新增 `auto_execute` 属性：有 session_scopes 时从 idle 直接跳到 run_execute，跳过 prepare/contract/confirm
- `AuthorizationUpdateResult` 新增 `has_session_scopes` 字段
- agent.set_mode() 在 `_switch_policy` 返回后事后设置 controller.auto_execute

### 行为变化

| 场景 | 行为 |
|------|------|
| 有 seed_scopes | auto_execute=True，跳过 prepare/contract |
| 无 seed_scopes + 用户同意 | 同上 |
| 无 seed_scopes + 用户拒绝 | auto_execute=False，当前流程不变 |
| 无 seed_scopes + confirm_handler 为 None | 同上（默认拒绝） |
| session_scopes + 任务取消 | session_scopes 保留（session 级别） |
| 模式切换 | _clear_all_scopes 清理所有 |

### 测试新增

- `TestTaskModeSessionScopes`：11 个测试覆盖 seed scopes、用户确认、fallback、scope 生命周期、优先级匹配
- 公共测试：191 → 200

### 修改文件

- `llamagent/core/authorization.py`：_switch_policy + TaskPolicy._decide_execute + AuthorizationUpdateResult
- `llamagent/core/controller.py`：auto_execute 属性 + handle_turn idle 分支
- `llamagent/core/agent.py`：set_mode 事后设置 auto_execute
- `tests/test_task_mode.py`：11 新测试 + 2 个既有测试调整（confirm_handler 设置顺序）

---

## v1.9.7 — Principle-Validated Improvements

### 代码改进

- `PipelineOutcome` 新增 `blocked: bool = False` 字段，`chat()` 通过 `outcome.blocked` 检测安全拦截，消除对响应文本的脆弱字符串比较
- `confirm_handler` 类型注解从 `Callable[..., Any]` 收窄为 `Callable[[ConfirmRequest], ConfirmResponse | bool]`
- `_run_task_mode_turn` 重命名为 `_run_controller_turn`
- `chat()` dispatch 条件从 `self.mode == "task"` 改为 `self._controller is not None`（为 ContinuousController 复用驱动循环铺路）
- `get_active_task_id()` 保留 `self.mode == "task"` 守卫（task_id 是 task-mode 专属概念）

### 测试新增

- `test_task_mode.py::TestMaxModeStepsExhaustion` — MAX_MODE_STEPS 耗尽边界测试
- `test_pipeline.py::test_blocked_pipeline_sets_outcome_blocked` — blocked 字段 + POST_CHAT hook 数据验证
- 公共测试：189 → 191

### 原则验证后跳过的项

以下 review 建议经 P1-P6 验证后删除或延期：

- ~~ModeAction terminal 字段~~（P6：引入双重真相源，agent 仍需在 3 处 dispatch on kind）
- ~~Policy 协议方法替代 isinstance~~（P6：泄漏 task-mode 概念到通用 ABC）
- ~~set_controller() 公共 API~~（P6：无外部消费者）
- ~~确认关键词可配置化~~（P6：稳定集合）
- AuthorizationContext 接口（延期：等待独立 engine 测试需求）
- contract.py 拆分（延期：118 行，低于 200 行阈值）
- event 字符串改枚举（跳过：当前模式正确且有错误处理）
- pipeline mode 枚举（跳过：仅 2 个值，稳定）

### 流程改进

- CLAUDE.md 新增 Design Principles 和 Principle Validation Rules 节
- 建立"review 建议必须逐条过 P1-P6 再入清单"的流程规范

### 修改文件

- `llamagent/core/contract.py`：PipelineOutcome 加 blocked 字段
- `llamagent/core/agent.py`：blocked 检测 + confirm_handler 注解 + 重命名 + dispatch 改动
- `tests/test_task_mode.py`：MAX_MODE_STEPS 耗尽测试
- `tests/test_pipeline.py`：blocked 字段测试

---

## v1.9.6 — Technical Debt Cleanup (Pipeline / Controller / Engine Decoupling)

### 设计原则

- **统一 pipeline**：消除 `chat()` 内联 pipeline + `_run_prepare()` + `_run_normal_pipeline()` 三重复制，归一为 `_run_pipeline()`
- **Controller 是纯状态机**：`TaskModeController` 不持有 agent/engine 引用，通过两步协议与 agent 交互
- **Agent 是协调者**：agent 只做授权数据的透明搬运，不理解 controller 内部逻辑
- **显式数据流**：controller 和 policy 之间无共享可变状态，所有数据通过返回值传递

### Pipeline 统一

- `_run_pipeline()` 替代 `chat()` 内联 pipeline + `_run_prepare()` + `_run_normal_pipeline()`
- 返回 `PipelineOutcome`（定义在 contract.py），包含 response / task_id / metadata（prepare 模式下 pending_scopes 通过 metadata 传递）

### TaskModeController

- 新增 `ModeController` ABC + `TaskModeController`（controller.py）
- 两步协议：`handle_turn()` → `ModeAction` → agent 执行 pipeline → `on_pipeline_done()` → `ModeAction`
- 纯状态机，不持有 agent / engine 引用
- 合同生成逻辑从 agent 搬入 controller

### 授权引擎变更

- `set_mode()` → `_switch_policy()`（内部方法）
- 新增：`apply_update()` / `_clear_all_scopes()` / `drain_prepare_data()` / `clear_pending_buffer()` / `authorization_status()`（scope 格式化从 agent 下沉到 engine）
- `_switch_policy()` 返回 `AuthorizationUpdateResult`（包含 seed scope 的 SCOPE_ISSUED 事件），agent 不再直接读 engine.state
- TaskPolicy：`_pending_buffer` 替代共享的 `TaskModeState.pending_scopes`
- `_decide_execute` 直接读 `self.state.task_id`（不再通过 engine.agent 间接获取）

### set_mode 重写

- agent 侧单一公开入口
- 执行序列：check idle → prepare controller → clear scopes → switch policy → commit → emit events
- 新增 `AuthorizationUpdateResult` 封装结构化事件返回

### 类型迁移

- `ApprovalScope`：zone.py → authorization.py
- `normalize_scopes`：authorization.py → contract.py
- `AuthorizationUpdate`：新增，定义在 contract.py
- `PipelineOutcome`：新增，定义在 contract.py
- `_task_mode_state` → `_controller`

### 历史管理

- 所有 task mode pipeline 使用 `record_history=False`
- agent 在驱动循环退出后手动写入 history

### 新增文件

- `llamagent/core/controller.py`

### 修改文件

- `llamagent/core/agent.py`：_run_pipeline + set_mode 重写 + _controller 替代 _task_mode_state + history 手动写入
- `llamagent/core/authorization.py`：_switch_policy + apply_update + drain_prepare_data + clear_pending_buffer + TaskPolicy._pending_buffer
- `llamagent/core/contract.py`：PipelineOutcome + AuthorizationUpdate + normalize_scopes 迁入
- `llamagent/core/zone.py`：ApprovalScope 迁出
- `llamagent/core/__init__.py`：导出更新

### 测试更新

- ApprovalScope + normalize_scopes import 路径迁移
- conftest fixture 适配 _controller 替代 _task_mode_state
