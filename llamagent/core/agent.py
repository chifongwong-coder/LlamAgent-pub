"""
SmartAgent core class: a complete, standalone AI Agent.

Without loading any modules, SmartAgent is a conversational AI assistant.
After loading modules via register_module(), the Agent gains tool calling, RAG, memory, and other capabilities.

Core components:
- SmartAgent:         Main Agent class containing the chat() entry point and run_react() engine
- Module:             Pluggable module base class that interacts with the Agent via pipeline callbacks
- ExecutionStrategy:  Pluggable execution strategy interface, replacing the deprecated on_execute callback
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from llamagent.core.authorization import AuthorizationEngine
from llamagent.core.config import Config
from llamagent.core.hooks import (
    CallableHandler,
    HookCallback,
    HookContext,
    HookEvent,
    HookHandler,
    HookMatcher,
    HookRegistration,
    HookResult,
    ShellHandler,
    _SKIPPABLE_EVENTS,
)
from llamagent.core.llm import LLMClient

logger = logging.getLogger(__name__)


# ======================================================================
# ReactResult (structured return from ReAct loop)
# ======================================================================


@dataclass
class ReactResult:
    """Structured return result from the ReAct loop."""
    text: str
    status: Literal["completed", "max_steps", "timeout", "error", "interrupted", "context_overflow"]
    error: str | None = None
    steps_used: int = 0
    reason: str | None = None


# ======================================================================
# Execution Strategy (Strategy Pattern)
# ======================================================================


class ExecutionStrategy:
    """
    Pluggable execution strategy interface.

    The execution strategy is responsible for assembling tools_schema and tool_dispatch,
    then passing them to run_react(). The ReAct engine itself is unaware of tool origins
    and only handles loop control.

    Built-in strategies:
    - SimpleReAct (default): Directly runs the ReAct loop
    - PlanReAct: Plans first, then executes step by step (injected by the Planning module)
    """

    def execute(self, query: str, context: str, agent: SmartAgent) -> str:
        """
        Execute the user request and return response text.

        Args:
            query: User input after on_input processing
            context: Context enhanced by on_context
            agent: Agent instance; the strategy accesses run_react(), call_tool(), etc. through it

        Returns:
            Final response text
        """
        raise NotImplementedError("Subclasses must implement the execute() method")


class SimpleReAct(ExecutionStrategy):
    """
    Default execution strategy: directly runs the ReAct loop.

    Returns text response directly when there are no tool_calls; otherwise loops through tool calls.

    Backward compatible: if a module overrides on_execute() (deprecated), its result is used first.
    """

    def execute(self, query: str, context: str, agent: SmartAgent) -> str:
        # Backward compatibility: check if any module intercepts via on_execute callback
        for mod in agent.modules.values():
            if type(mod).on_execute is not Module.on_execute:
                result = mod.on_execute(query, context)
                if result is not None:
                    return result

        # Get tool schemas
        tools_schema = agent.get_all_tool_schemas()

        # Tool dispatch function
        tool_dispatch = agent.call_tool

        # Build messages
        messages = agent.build_messages(query, context)

        react_result = agent.run_react(messages, tools_schema, tool_dispatch)
        return react_result.text


# ======================================================================
# Module base class
# ======================================================================


class Module:
    """
    Module base class; all pluggable modules inherit from this class.

    Modules interact with the Agent pipeline by overriding callback methods.
    All callback base implementations are no-ops; modules override as needed.

    Lifecycle Callbacks (called once each):
    - on_attach(agent)    Called on registration, used for initialization
    - on_shutdown()       Called on Agent exit, used to release resources

    Pipeline Callbacks (called each conversation turn):
    - on_input(user_input) -> str        Input preprocessing
    - on_context(query, context) -> str  Context enhancement
    - on_output(response) -> str         Output post-processing

    Full lifecycle:
    on_attach -> [on_input -> on_context -> execution strategy -> on_output] x N -> on_shutdown
    """

    name: str = "base"
    description: str = ""

    # --- Lifecycle Callbacks ---

    def on_attach(self, agent: SmartAgent) -> None:
        """
        Called when the module is registered via register_module().

        Used for initializing storage, registering tools, injecting execution strategies, injecting safety callbacks, etc.
        Base class default saves the agent reference.
        """
        self.agent = agent

    def attach(self, agent: SmartAgent) -> None:
        """
        Backward compatibility: legacy modules use attach(), new modules should use on_attach().

        If a subclass only overrides attach(), register_module() will automatically call attach().
        New modules should directly override on_attach().
        """
        self.agent = agent

    def on_shutdown(self) -> None:
        """
        Called on Agent exit.

        Used to close connections and release resources (MCP disconnect, ChromaDB close, etc.).
        """
        pass

    # --- Pipeline Callbacks ---

    def on_input(self, user_input: str) -> str:
        """
        Input preprocessing callback.

        Called in module registration order (forward). Returning an empty string is treated as a safety interception.
        """
        return user_input

    def on_context(self, query: str, context: str) -> str:
        """
        Context enhancement callback.

        Called in module registration order (forward). Each module appends context in sequence.
        """
        return context

    def on_output(self, response: str) -> str:
        """
        Output post-processing callback.

        Called in **reverse** module registration order (onion model).
        """
        return response

    # --- Deprecated Callbacks (backward compatible, will be removed in future versions) ---

    def on_execute(self, query: str, context: str) -> str | None:
        """
        [Deprecated] Execution interception callback; returning non-None skips default execution.

        In the target architecture, this is replaced by ExecutionStrategy. This method is retained
        only for backward compatibility with modules that have not yet migrated (e.g., reasoning module).
        """
        return None


# ======================================================================
# SmartAgent main class
# ======================================================================


class SmartAgent:
    """
    LlamAgent: modular AI Agent core engine.

    Core capabilities (no modules required):
    - Multi-turn conversation
    - System Prompt customization
    - Conversation history management
    - ReAct loop engine (tool calling)

    Enhanced via modules:
    - tools:        Tool registration management + meta-tools
    - rag:          RAG knowledge retrieval
    - memory:       Long-term memory
    - planning:     Task planning (PlanReAct strategy)
    - reflection:   Self-reflection and error correction
    - multi_agent:  Multi-Agent collaboration
    - mcp:          MCP external system integration
    - safety:       Safety guardrails
    """

    def __init__(
        self,
        config: Config | None = None,
        persona: "Persona | None" = None,
    ):
        """
        Initialize SmartAgent.

        Args:
            config: Configuration object; auto-creates default Config() when None
            persona: Persona object; uses config.system_prompt as default identity when None
        """
        self.config = config or Config()
        self.persona = persona
        self.llm = LLMClient(
            model=self.config.model,
            api_retry_count=self.config.api_retry_count,
        )
        self.modules: dict[str, Module] = {}
        self.history: list[dict] = []
        self.summary: str | None = None

        # Backward compatibility: conversation as an alias for history
        self.conversation = self.history

        # Execution strategy, default SimpleReAct
        self._execution_strategy: ExecutionStrategy = SimpleReAct()

        # v1.3/v1.9: zone-based safety system (v1.9: structured ConfirmRequest/ConfirmResponse)
        self.confirm_handler: Callable | None = None  # Callable[[ConfirmRequest], ConfirmResponse]
        self.interaction_handler = None  # v1.8.2: injected by caller for ask_user tool
        self._confirm_wait_time: float = 0.0  # Accumulated confirmation wait, excluded from react_timeout
        self.project_dir: str = os.path.realpath(os.getcwd())
        self.playground_dir: str = os.path.realpath(os.path.join(self.project_dir, "llama_playground"))
        try:
            os.makedirs(self.playground_dir, exist_ok=True)
        except OSError:
            pass  # Playground creation failed (read-only fs, permissions, etc.); zone system still works
        self.tool_executor = None  # v1.2: injected by SandboxModule for sandbox execution dispatch

        # v1.9: authorization engine (encapsulates zone evaluation + policy decision)
        self.mode: str = getattr(self.config, "authorization_mode", "interactive")
        self._authorization_engine = AuthorizationEngine(self)
        self._task_mode_state = None  # v1.9.1: TaskModeState, set by set_mode("task")

        # Tool registry (simple implementation, later enhanced by tools module)
        # Format: {name: {"func": callable, "description": str, "parameters": dict,
        #               "tier": str, "safety_level": int}}
        self._tools: dict[str, dict[str, Any]] = {}

        # v1.6: pack-based conditional tool exposure
        self._active_packs: set[str] = set()

        # Tool registry version number, incremented on each register_tool/remove_tool, used for cache invalidation
        self._tools_version: int = 0

        # v1.8: event hook system
        self._hooks: dict[HookEvent, list[HookRegistration]] = {}
        self._session_started: bool = False
        self._in_hook: bool = False  # Reentry protection

        # Register YAML-configured hooks
        self._register_yaml_hooks()

    # ============================================================
    # Module management
    # ============================================================

    def register_module(self, module: Module) -> None:
        """
        Register a capability module.

        Compatible with both legacy and new modules:
        - If the module overrides on_attach(), call on_attach()
        - If the module only overrides attach() (legacy), call attach()
        - If both are overridden, on_attach() takes priority

        Args:
            module: The module instance to register
        """
        try:
            # Check if the module overrides on_attach (new API)
            has_custom_on_attach = type(module).on_attach is not Module.on_attach
            has_custom_attach = type(module).attach is not Module.attach

            if has_custom_on_attach:
                module.on_attach(self)
            elif has_custom_attach:
                # Legacy module only overrides attach()
                module.attach(self)
            else:
                # Neither overridden, call base on_attach
                module.on_attach(self)

            self.modules[module.name] = module
            logger.info("Module registered: %s (%s)", module.name, module.description)
        except Exception as e:
            logger.error("Module '%s' registration failed: %s", module.name, e)
            raise

    def get_module(self, name: str) -> Module | None:
        """Get a registered module by name. Returns None if not found."""
        return self.modules.get(name)

    def has_module(self, name: str) -> bool:
        """Check if a module is registered."""
        return name in self.modules

    def list_modules(self) -> list[str]:
        """List the names of all registered modules."""
        return list(self.modules.keys())

    # ============================================================
    # Execution strategy
    # ============================================================

    def authorization_status(self) -> dict:
        """Return current authorization state snapshot."""
        import time as _time
        now = _time.time()
        state = self._authorization_engine.state
        return {
            "mode": self.mode,
            "task_scopes": {
                tid: [
                    {"zone": s.zone, "actions": s.actions, "path_prefixes": s.path_prefixes,
                     "uses": s.uses, "max_uses": s.max_uses, "source": s.source,
                     "expired": s.expires_at is not None and now > s.expires_at}
                    for s in scopes
                ]
                for tid, scopes in state.task_scopes.items()
            },
            "session_scopes": [
                {"zone": s.zone, "actions": s.actions, "path_prefixes": s.path_prefixes,
                 "uses": s.uses, "max_uses": s.max_uses, "source": s.source,
                 "expired": s.expires_at is not None and now > s.expires_at}
                for s in state.session_scopes
            ],
        }

    def get_active_task_id(self) -> str | None:
        """
        Get the active task ID for scope storage/lookup.

        Priority: TaskModeState.task_id > _current_task_id (PlanReAct).
        """
        if self.mode == "task" and self._task_mode_state and self._task_mode_state.task_id:
            return self._task_mode_state.task_id
        return getattr(self, "_current_task_id", None)

    def set_mode(self, mode: str) -> None:
        """
        Switch authorization mode.

        Args:
            mode: "interactive" (default) | "task" | "continuous"
        """
        # Collect scopes to revoke before mode switch clears them
        old_scopes = []
        for scopes in self._authorization_engine.state.task_scopes.values():
            old_scopes.extend(scopes)
        old_scopes.extend(self._authorization_engine.state.session_scopes)

        self.mode = mode
        self._authorization_engine.set_mode(mode)

        # Emit SCOPE_REVOKED for cleared scopes
        for s in old_scopes:
            self.emit_hook(HookEvent("scope_revoked"), {"scope": s, "reason": "mode_switch"})

        # Emit SCOPE_ISSUED for newly loaded seed scopes (continuous mode)
        for s in self._authorization_engine.state.session_scopes:
            self.emit_hook(HookEvent("scope_issued"), {"scope": s, "task_id": None})

        logger.info("Authorization mode switched to: %s", mode)

    def set_execution_strategy(self, strategy: ExecutionStrategy) -> None:
        """
        Replace the current execution strategy. Default is SimpleReAct.

        Typically called by the Planning module during on_attach() to inject the PlanReAct strategy.

        Args:
            strategy: The new execution strategy instance
        """
        old_name = type(self._execution_strategy).__name__
        new_name = type(strategy).__name__
        self._execution_strategy = strategy
        logger.info("Execution strategy switched: %s -> %s", old_name, new_name)

    # ============================================================
    # Tool registration and management
    # ============================================================

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: dict | None = None,
        tier: str = "common",
        safety_level: int = 1,
        execution_policy=None,
        creator_id: str | None = None,
        path_extractor: Callable[[dict], list[str]] | None = None,
        pack: str | None = None,
        action: str | None = None,
    ) -> None:
        """
        Register a tool in the registry.

        Args:
            name: Tool name
            func: Tool execution function
            description: Tool description
            parameters: Parameter definition in JSON Schema format; inferred from function signature when None
            tier: Tool tier "default" / "common" / "admin" / "agent"
            safety_level: Safety level 1=read-only 2=has side effects 3=high risk
            execution_policy: ExecutionPolicy object from sandbox module, or None (default = host execution)
            creator_id: Creator persona_id (for agent-tier tools, used for visibility filtering)
            path_extractor: Optional function to extract file paths from tool arguments for zone checking
        """
        # Infer parameter definition from function signature when empty
        if parameters is None:
            parameters = self._infer_parameters(func)

        self._tools[name] = {
            "name": name,
            "func": func,
            "description": description,
            "parameters": parameters,
            "tier": tier,
            "safety_level": safety_level,
            "execution_policy": execution_policy,
            "creator_id": creator_id,
            "path_extractor": path_extractor,
            "pack": pack,
            "action": action,
        }
        self._tools_version += 1
        logger.debug("Tool registered: %s (tier=%s, safety_level=%d)", name, tier, safety_level)

    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name: Tool name

        Returns:
            True if successfully removed, False if tool does not exist
        """
        if name in self._tools:
            del self._tools[name]
            self._tools_version += 1
            logger.debug("Tool removed: %s", name)
            return True
        return False

    def call_tool(self, name: str, args: dict) -> str:
        """
        Registry tool call: lookup + hook + permission check + execution.

        Tool not found and permission denied are both returned as strings,
        serving as tool observations fed back to the model.

        Args:
            name: Tool name
            args: Tool arguments (dict, expanded internally for the call)

        Returns:
            Tool execution result string, or error description string
        """
        # 1. Lookup tool
        tool = self._tools.get(name)
        if tool is None:
            available = list(self._tools.keys())
            return f"Tool '{name}' does not exist. Available tools: {available}"

        # 2. PRE_TOOL_USE hook (can block or modify args)
        hook_data = {"tool_name": name, "args": dict(args), "tool_info": tool}
        hook_result = self.emit_hook(HookEvent.PRE_TOOL_USE, hook_data)
        if hook_result == HookResult.SKIP:
            return hook_data.get("skip_reason", f"Tool '{name}' blocked by hook.")
        args = hook_data.get("args", args)  # Hook may have modified args

        # 3. Authorization (path extraction + zone evaluation + policy decision)
        auth = self._authorization_engine.evaluate(tool, args)
        for event_name, event_data in auth.events:
            self.emit_hook(HookEvent(event_name), event_data)
        if auth.decision is not None:
            return auth.decision

        # 4. Execute tool (unified path for both direct and executor/sandbox)
        start_time = time.time()
        try:
            if self.tool_executor is not None:
                result_str = self.tool_executor.execute(tool, args)
            else:
                result = tool["func"](**args)
                result_str = str(result) if result is not None else ""

            # 5. POST_TOOL_USE hook
            duration_ms = (time.time() - start_time) * 1000
            self.emit_hook(HookEvent.POST_TOOL_USE, {
                "tool_name": name,
                "args": args,
                "result": result_str,
                "result_preview": result_str[:200] if result_str else "",
                "duration_ms": duration_ms,
            })
            return result_str

        except Exception as e:
            # 6. TOOL_ERROR hook
            logger.error("Tool '%s' execution error: %s", name, e)
            self.emit_hook(HookEvent.TOOL_ERROR, {
                "tool_name": name,
                "args": args,
                "error": str(e),
            })
            return f"Tool '{name}' execution error: {e}"

    def get_all_tool_schemas(self) -> list[dict]:
        """
        Merge tool schemas from the registry, filtered by tier + persona role.

        Filtering rules:
        - tier=default + tier=common: visible to all personas
        - tier=admin: included only when persona.role == "admin"
        - tier=agent: includes only the current persona's custom tools

        Returns:
            Tool schema list in OpenAI function calling format
        """
        schemas = []
        is_admin = self.persona and self.persona.is_admin

        for name, tool in self._tools.items():
            tier = tool.get("tier", "common")

            # Filter by tier (visibility = usability)
            if tier == "admin" and not is_admin:
                continue
            if tier == "agent":
                current_pid = self.persona.persona_id if self.persona else "default"
                if tool.get("creator_id") != current_pid:
                    continue

            # v1.6: pack-based filtering — tools with a pack are only visible when that pack is active
            pack = tool.get("pack")
            if pack is not None and pack not in self._active_packs:
                continue

            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            schemas.append(schema)

        return schemas

    @staticmethod
    def _infer_parameters(func: Callable) -> dict:
        """
        Infer JSON Schema parameter definitions from a function signature.

        Simple mapping: str -> string, int -> integer, float -> number, bool -> boolean.
        """
        import inspect

        sig = inspect.signature(func)
        properties = {}
        required = []

        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Type inference
            if param.annotation != inspect.Parameter.empty:
                json_type = type_mapping.get(param.annotation, "string")
            else:
                json_type = "string"

            properties[param_name] = {"type": json_type}

            # Whether required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        result = {"type": "object", "properties": properties}
        if required:
            result["required"] = required
        return result

    # ============================================================
    # Event Hook system (v1.8)
    # ============================================================

    def register_hook(
        self,
        event: HookEvent,
        handler: HookCallback | HookHandler,
        *,
        matcher: HookMatcher | None = None,
        priority: int = 100,
        source: str = "code",
    ) -> None:
        """
        Register an event hook.

        Args:
            event: Event type to listen for
            handler: Python callable or HookHandler instance
            matcher: Optional filter conditions (AND logic)
            priority: Execution order (lower = earlier). Code default=100, YAML default=200
            source: Registration source identifier ("code" / "yaml")
        """
        if isinstance(handler, HookHandler):
            hook_handler = handler
        else:
            hook_handler = CallableHandler(handler)

        reg = HookRegistration(
            event=event,
            handler=hook_handler,
            matcher=matcher,
            priority=priority,
            source=source,
        )

        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(reg)
        # Keep sorted by priority (stable sort preserves registration order for equal priority)
        self._hooks[event].sort(key=lambda r: r.priority)

        logger.debug("Hook registered: event=%s, source=%s, priority=%d", event.value, source, priority)

    def emit_hook(self, event: HookEvent, data: dict) -> HookResult:
        """
        Emit a hook event, executing all matching handlers.

        Reentry protection: if called from within a hook handler, returns CONTINUE immediately.

        Args:
            event: Event type
            data: Event data dict (mutable — handlers may modify it)

        Returns:
            SKIP if any handler blocked the operation (only for PRE_TOOL_USE),
            CONTINUE otherwise
        """
        if self._in_hook:
            return HookResult.CONTINUE

        registrations = self._hooks.get(event)
        if not registrations:
            return HookResult.CONTINUE

        self._in_hook = True
        try:
            for reg in registrations:
                # Matcher filtering
                if reg.matcher is not None and not reg.matcher.matches(data):
                    continue

                ctx = HookContext(
                    agent=self,
                    event=event,
                    data=data,
                    matcher=reg.matcher,
                )

                result = reg.handler.execute(ctx)

                # SKIP is only effective for skippable events (PRE_TOOL_USE)
                if result == HookResult.SKIP:
                    if event in _SKIPPABLE_EVENTS:
                        return HookResult.SKIP
                    else:
                        logger.debug(
                            "Hook returned SKIP for non-skippable event %s, treating as CONTINUE",
                            event.value,
                        )

            return HookResult.CONTINUE
        finally:
            self._in_hook = False

    def _register_yaml_hooks(self) -> None:
        """Register hooks declared in YAML config (config.hooks_config)."""
        hooks_config = getattr(self.config, "hooks_config", None)
        if not hooks_config or not isinstance(hooks_config, dict):
            return

        for event_name, hook_list in hooks_config.items():
            try:
                event = HookEvent(event_name)
            except ValueError:
                logger.warning("Unknown hook event in YAML config: '%s', skipping", event_name)
                continue

            if not isinstance(hook_list, list):
                logger.warning("Hook config for '%s' is not a list, skipping", event_name)
                continue

            for hook_def in hook_list:
                if not isinstance(hook_def, dict):
                    continue

                shell_cmd = hook_def.get("shell")
                if not shell_cmd:
                    logger.warning("Hook definition missing 'shell' key for event '%s', skipping", event_name)
                    continue

                # Parse matcher
                matcher = None
                matcher_def = hook_def.get("matcher")
                if isinstance(matcher_def, dict):
                    matcher = HookMatcher(
                        tool_name=matcher_def.get("tool_name"),
                        tool_names=matcher_def.get("tool_names"),
                        pack=matcher_def.get("pack"),
                        safety_level=matcher_def.get("safety_level"),
                    )

                priority = hook_def.get("priority", 200)
                timeout = hook_def.get("timeout", 30.0)

                handler = ShellHandler(command=shell_cmd, timeout=timeout)
                self.register_hook(
                    event=event,
                    handler=handler,
                    matcher=matcher,
                    priority=priority,
                    source="yaml",
                )

    # ============================================================
    # Task mode (v1.9.1)
    # ============================================================

    def _handle_task_mode_turn(self, user_input: str) -> str:
        """Handle one chat() turn in task mode based on current phase."""
        state = self._task_mode_state

        if state.phase == "idle":
            import uuid as _uuid
            state.task_id = _uuid.uuid4().hex
            state.phase = "preparing"
            state.original_query = user_input
            return self._run_prepare(user_input)

        if state.phase == "awaiting_confirmation":
            lower = user_input.strip().lower()
            if lower in ("yes", "y", "confirm", "ok", "approve"):
                # Write approved scopes before execution
                self._write_task_scopes()
                state.phase = "executing"
                state.confirmed = True
                return self._run_execute(state.original_query)
            elif lower in ("no", "n", "cancel", "abort"):
                task_id = state.task_id
                removed = self._authorization_engine.state.task_scopes.pop(task_id, [])
                for s in removed:
                    self.emit_hook(HookEvent("scope_revoked"), {"scope": s, "reason": "task_cancelled"})
                state.phase = "idle"
                state.task_id = ""
                state.pending_scopes.clear()
                state.contract = None
                state.confirmed = False
                return "Task cancelled."
            else:
                # Additional info — re-enter prepare
                state.phase = "preparing"
                return self._run_prepare(user_input)

        if state.phase == "executing":
            return self._run_normal_pipeline(user_input)

        return self._run_normal_pipeline(user_input)

    def _run_prepare(self, query: str) -> str:
        """Run one prepare turn: dry-run ReAct to collect pending scopes."""
        # on_input for safety filtering
        processed = query
        for mod in self.modules.values():
            try:
                processed = mod.on_input(processed)
            except Exception as e:
                logger.error("Module '%s' on_input error: %s", mod.name, e)

        if not processed or not processed.strip():
            return "Sorry, I cannot process this request."

        # on_context for knowledge/memory/skill injection
        context = ""
        for mod in self.modules.values():
            try:
                context = mod.on_context(processed, context)
            except Exception as e:
                logger.error("Module '%s' on_context error: %s", mod.name, e)

        tools_schema = self.get_all_tool_schemas()
        messages = self.build_messages(query, context)
        react_result = self.run_react(messages, tools_schema, self.call_tool)

        # After dry-run, generate contract from collected scopes
        from llamagent.core.authorization import normalize_scopes
        from llamagent.core.contract import TaskContract

        state = self._task_mode_state
        normalized = normalize_scopes(state.pending_scopes)

        contract = TaskContract(
            task_summary=query,
            planned_operations=[
                f"{s.actions[0]} in {s.zone}: {', '.join(s.path_prefixes)}"
                for s in normalized
            ],
            requested_scopes=normalized,
            open_questions=[],
            risk_summary=f"{len(normalized)} operations require authorization.",
        )
        state.contract = contract
        state.phase = "awaiting_confirmation"

        # Build contract summary for user
        if not normalized:
            # No CONFIRMABLE operations found — skip contract, go straight to execute
            state.phase = "executing"
            state.confirmed = True
            return self._run_execute(state.original_query)

        lines = [f"[Task Contract] {contract.task_summary}", ""]
        lines.append("Planned operations requiring authorization:")
        for op in contract.planned_operations:
            lines.append(f"  - {op}")
        lines.append(f"\nRisk: {contract.risk_summary}")
        lines.append("\nReply 'yes' to confirm, 'no' to cancel, or provide more details.")

        return "\n".join(lines)

    def _write_task_scopes(self):
        """Convert confirmed contract scopes into ApprovalScopes in AuthorizationState."""
        from llamagent.core.zone import ApprovalScope

        state = self._task_mode_state
        task_id = self.get_active_task_id()
        if not task_id or not state.contract:
            return

        import time as _time
        scopes_to_approve = state.contract.requested_scopes
        for rs in scopes_to_approve:
            scope = ApprovalScope(
                scope="task", zone=rs.zone, actions=rs.actions,
                path_prefixes=rs.path_prefixes, tool_names=rs.tool_names,
                created_at=_time.time(), source="contract",
            )
            self._authorization_engine.state.task_scopes.setdefault(task_id, []).append(scope)
            self.emit_hook(HookEvent("scope_issued"), {"scope": scope, "task_id": task_id})

    def _run_execute(self, query: str) -> str:
        """Run the normal pipeline for actual execution after contract confirmation."""
        state = self._task_mode_state
        state.phase = "executing"

        # Set _current_task_id so PlanReAct and workspace can use it
        self._current_task_id = state.task_id

        try:
            result = self._run_normal_pipeline(query)
        finally:
            # Clean up + emit SCOPE_REVOKED for each removed scope
            task_id = state.task_id
            self._current_task_id = None
            removed = self._authorization_engine.state.task_scopes.pop(task_id, [])
            for s in removed:
                self.emit_hook(HookEvent("scope_revoked"), {"scope": s, "reason": "task_completed"})
            state.phase = "idle"
            state.task_id = ""
            state.pending_scopes.clear()
            state.contract = None
            state.confirmed = False

        return result

    def _run_normal_pipeline(self, user_input: str) -> str:
        """Run the standard pipeline (on_input → on_context → strategy → on_output)."""
        processed = user_input
        for mod in self.modules.values():
            try:
                processed = mod.on_input(processed)
            except Exception as e:
                logger.error("Module '%s' on_input error: %s", mod.name, e)

        if not processed or not processed.strip():
            return "Sorry, I cannot process this request."

        context = ""
        for mod in self.modules.values():
            try:
                context = mod.on_context(processed, context)
            except Exception as e:
                logger.error("Module '%s' on_context error: %s", mod.name, e)

        try:
            response = self._execution_strategy.execute(processed, context, self)
        except Exception as e:
            logger.error("Execution strategy error: %s", e)
            response = f"Error processing request: {e}"

        for mod in reversed(list(self.modules.values())):
            try:
                response = mod.on_output(response)
            except Exception as e:
                logger.error("Module '%s' on_output error: %s", mod.name, e)

        self.history.append({"role": "user", "content": processed})
        self.history.append({"role": "assistant", "content": response})
        self._trim_history()

        return response

    # ============================================================
    # Conversation (core capability)
    # ============================================================

    def _ask_confirmation(self, request) -> "ConfirmResponse":
        """
        Call confirm_handler with structured request. Track wait time.

        Args:
            request: ConfirmRequest from the authorization engine

        Returns:
            ConfirmResponse (allow=False if no handler or exception)
        """
        from llamagent.core.zone import ConfirmResponse

        if self.confirm_handler is None:
            return ConfirmResponse(allow=False)
        t0 = time.time()
        try:
            response = self.confirm_handler(request)
            if not isinstance(response, ConfirmResponse):
                # Backward compat: if handler returns bool, wrap it
                response = ConfirmResponse(allow=bool(response))
        except Exception:
            response = ConfirmResponse(allow=False)
        self._confirm_wait_time += time.time() - t0
        return response

    def chat(self, user_input: str) -> str:
        """
        Agent main entry point: receives user input and returns a response.

        Pipeline: on_input -> on_context -> execution strategy -> on_output

        Context management before and after conversation:
        - Before: (reserved) token compression check
        - After: turn window check, discarding oldest turns

        Args:
            user_input: User input, should not be an empty string

        Returns:
            The Agent's response text
        """
        # --- SESSION_START (fires once on first chat) ---
        if not self._session_started:
            self.emit_hook(HookEvent.SESSION_START, {"modules": list(self.modules.keys())})
            self._session_started = True

        # --- PRE_CHAT (observational, SKIP not supported) ---
        self.emit_hook(HookEvent.PRE_CHAT, {"user_input": user_input})

        blocked = False
        blocked_by = None
        response = ""

        # --- Task mode branch (v1.9.1) ---
        if self.mode == "task" and self._task_mode_state is not None:
            response = self._handle_task_mode_turn(user_input)
            self.emit_hook(HookEvent.POST_CHAT, {
                "user_input": user_input, "response": response,
                "blocked": False, "blocked_by": None, "completed": True,
            })
            return response

        # --- Before conversation: context management (reserved for token compression) ---
        self._check_context_compression()

        # --- 1. on_input: modules preprocess input (forward registration order) ---
        processed = user_input
        for mod in self.modules.values():
            try:
                processed = mod.on_input(processed)
            except Exception as e:
                logger.error("Module '%s' on_input error: %s", mod.name, e)

        # Safety interception: if input is cleared (e.g., blocked by safety module)
        if not processed or not processed.strip():
            blocked = True
            blocked_by = "safety"
            response = "Sorry, I cannot process this request."
        else:
            # --- 2. on_context: modules enhance context (forward registration order) ---
            context = ""
            for mod in self.modules.values():
                try:
                    context = mod.on_context(processed, context)
                except Exception as e:
                    logger.error("Module '%s' on_context error: %s", mod.name, e)

            # --- 3. Execution strategy ---
            try:
                response = self._execution_strategy.execute(processed, context, self)
            except Exception as e:
                logger.error("Execution strategy error: %s", e)
                response = f"Error processing request: {e}"

            # --- 4. on_output: modules post-process output (reverse registration order, onion model) ---
            for mod in reversed(list(self.modules.values())):
                try:
                    response = mod.on_output(response)
                except Exception as e:
                    logger.error("Module '%s' on_output error: %s", mod.name, e)

            # --- 5. Update conversation history (write processed input, not raw user_input) ---
            self.history.append({"role": "user", "content": processed})
            self.history.append({"role": "assistant", "content": response})

            # --- 6. After conversation: turn window check ---
            self._trim_history()

        # --- POST_CHAT (always-fire, all exit paths) ---
        self.emit_hook(HookEvent.POST_CHAT, {
            "user_input": user_input,
            "response": response,
            "blocked": blocked,
            "blocked_by": blocked_by,
            "completed": not blocked,
        })

        return response

    # ============================================================
    # ReAct loop engine
    # ============================================================

    def run_react(
        self,
        messages: list[dict],
        tools_schema: list[dict],
        tool_dispatch: Callable[[str, dict], str],
        *,
        should_continue: Callable[[], str | None] | None = None,
    ) -> ReactResult:
        """
        ReAct loop engine. Stateless, unaware of tool origins, called by execution strategies.

        Loop protections:
        - Maximum steps (max_react_steps)
        - Duplicate detection (aborts on consecutive identical tool name + arguments)
        - Timeout protection (react_timeout, each step timed independently)
        - Observation truncation (max_observation_tokens)
        - ContextWindowExceededError -> abort loop

        Args:
            messages: Message list sent to LLM (including system prompt + conversation history + current query)
            tools_schema: Tool schema list in OpenAI format, passed to function calling
            tool_dispatch: Tool dispatch function with signature (name, args) -> result_str
            should_continue: Optional callback checked after each tool call; returns str to abort loop
                            (that string becomes the interruption reason), returns None to continue

        Returns:
            ReactResult structured return result
        """
        # No tools: direct LLM call returning text
        if not tools_schema:
            return self._simple_llm_call(messages)

        steps = 0
        last_action: tuple | None = None  # Previous (name, args_json)
        duplicate_count = 0
        last_text_response = ""  # Record the last successful text response

        while steps < self.config.max_react_steps:
            # --- Timeout protection ---
            step_start = time.time()
            self._confirm_wait_time = 0.0  # Reset per step

            try:
                resp = self.llm.chat(messages, tools=tools_schema,
                                     timeout=self.config.react_timeout)
            except Exception as e:
                # ContextWindowExceededError -> abort loop
                error_name = type(e).__name__
                if "ContextWindow" in error_name:
                    logger.warning("Context window overflow during ReAct loop, aborting")
                    text = last_text_response or "The task is too complex. Consider enabling the Planning module for step-by-step execution, or use /clear to clear conversation history."
                    return ReactResult(text=text, status="context_overflow",
                                       error=str(e), steps_used=steps)
                logger.error("ReAct LLM call failed: %s", e)
                return ReactResult(text=f"ReAct execution error: {e}", status="error",
                                   error=str(e), steps_used=steps)

            message = resp.choices[0].message

            # --- No tool_calls -> return text directly ---
            if not message.tool_calls:
                return ReactResult(text=message.content or "", status="completed",
                                   steps_used=steps)

            # Record text response (if any)
            if message.content:
                last_text_response = message.content

            # Append assistant message (with tool_calls) to messages
            messages.append(message.model_dump() if hasattr(message, 'model_dump') else dict(message))

            # --- Has tool_calls -> process in loop ---
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}
                    logger.warning("Tool argument JSON parsing failed: %s", tool_call.function.arguments)

                # Duplicate detection
                current_action = (tool_name, json.dumps(tool_args, sort_keys=True))
                if current_action == last_action:
                    duplicate_count += 1
                    if duplicate_count >= self.config.max_duplicate_actions:
                        logger.warning(
                            "Duplicate action detected (%s), %d consecutive times, aborting",
                            tool_name,
                            duplicate_count,
                        )
                        return ReactResult(
                            text=f"Duplicate action detected ({tool_name}), execution aborted.",
                            status="error", steps_used=steps)
                else:
                    last_action = current_action
                    duplicate_count = 1

                # --- Action: dispatch execution ---
                try:
                    result = tool_dispatch(tool_name, tool_args)
                    result = str(result) if result is not None else ""
                except Exception as e:
                    logger.error("Tool '%s' dispatch execution error: %s", tool_name, e)
                    result = f"Tool execution error: {e}"

                # --- Observation: truncate overly long results ---
                result = self._truncate_observation(result)

                # Append tool message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

                # --- should_continue check ---
                if should_continue is not None:
                    stop_reason = should_continue()
                    if stop_reason is not None:
                        return ReactResult(text=result, status="interrupted",
                                           reason=stop_reason, steps_used=steps)

            # Timeout check
            elapsed = time.time() - step_start - self._confirm_wait_time
            if elapsed > self.config.react_timeout:
                logger.warning("ReAct single step timeout (%.1fs > %.1fs)", elapsed, self.config.react_timeout)
                text = last_text_response or "Execution timed out. Please simplify the request and try again."
                return ReactResult(text=text, status="timeout",
                                   error="Execution timed out", steps_used=steps)

            steps += 1

        # Reached maximum steps
        logger.warning("ReAct loop reached maximum steps (%d)", self.config.max_react_steps)
        return ReactResult(text="Maximum execution steps reached. Please simplify the request or ask in smaller steps.",
                           status="max_steps", steps_used=steps)

    def _simple_llm_call(self, messages: list[dict]) -> ReactResult:
        """Simple LLM call when no tools are available."""
        try:
            resp = self.llm.chat(messages)
            return ReactResult(text=resp.choices[0].message.content or "",
                               status="completed")
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return ReactResult(text=f"Conversation error: {e}", status="error", error=str(e))

    def _truncate_observation(self, text: str) -> str:
        """
        Truncate overly long tool return results.

        Counts by tokens (max_observation_tokens); truncates and appends a notice when exceeded.
        """
        max_tokens = self.config.max_observation_tokens
        token_count = self.llm.count_tokens(text)
        if token_count <= max_tokens:
            return text

        # Rough proportional truncation (tokens and character count are roughly proportional)
        ratio = max_tokens / max(token_count, 1)
        cut_pos = max(int(len(text) * ratio), 100)
        truncated = text[:cut_pos]
        logger.debug("Observation truncated: %d tokens -> %d tokens", token_count, max_tokens)
        return truncated + "\n...(content truncated)"

    # ============================================================
    # Message building
    # ============================================================

    def build_messages(
        self,
        query: str,
        context: str,
        *,
        include_history: bool = True,
        extra_system: str = "",
    ) -> list[dict]:
        """
        Unified message building method. Both SimpleReAct and PlanReAct build LLM message lists through this method.

        Args:
            query: User query
            context: Context string from on_context output
            include_history: Whether to include conversation history (SimpleReAct=True, PlanReAct per-step=False)
            extra_system: Additional system prompt (PlanReAct step instructions, etc.)

        Returns:
            Message list
        """
        # 1. Build system prompt
        system = self._build_system_prompt(context)
        if extra_system:
            system += "\n" + extra_system
        messages = [{"role": "system", "content": system}]

        # 2. Summary (if available)
        if self.summary:
            messages.append({"role": "system", "content": f"[Conversation History Summary] {self.summary}"})

        # 3. Conversation history (optional)
        if include_history:
            messages.extend(self.history)

        # 4. Current query
        messages.append({"role": "user", "content": query})

        return messages

    # ============================================================
    # Context management
    # ============================================================

    def _build_system_prompt(self, context: str = "") -> str:
        """
        Build system prompt: persona identity + module info + dynamic context.

        Uses the persona's system_prompt when a Persona is set; otherwise uses the config default prompt.
        """
        # Persona identity
        base_prompt = (
            self.persona.to_system_prompt()
            if self.persona
            else self.config.system_prompt
        )
        parts = [base_prompt]

        # Behavioral guidelines
        parts.append(
            "\n[Behavioral Guidelines]\n"
            "- Results returned by tools are real data; do not fabricate or modify them\n"
            "- When a tool call fails, try an alternative approach instead of repeating the same call\n"
            "- When uncertain, proactively ask the user rather than guessing\n"
            "- Respect permission boundaries; do not exceed authorized operations"
        )

        # Module capability descriptions
        module_info = []
        for mod in self.modules.values():
            if mod.description:
                module_info.append(f"- {mod.name}: {mod.description}")
        if module_info:
            parts.append("\n[Capability Modules]\n" + "\n".join(module_info))

        # Dynamic context (output from on_context)
        if context:
            parts.append(f"\n{context}")

        return "\n\n".join(parts)

    def _check_context_compression(self) -> None:
        """
        Pre-conversation context compression check.

        Triggers compression when conversation tokens exceed the threshold.
        Phase 1: When tokens exceed the threshold, call _compress_history() to compress old messages into a summary.
        """
        if not self.history:
            return

        try:
            token_count = self.llm.count_tokens(self.history)
            threshold = int(self.config.max_context_tokens * self.config.context_compress_threshold)
            if token_count >= threshold:
                logger.info(
                    "Conversation tokens (%d) reached compression threshold (%d), triggering context compression",
                    token_count,
                    threshold,
                )
                self._compress_history()
        except Exception as e:
            logger.debug("Context compression check error (non-fatal): %s", e)

    def _compress_history(self) -> None:
        """
        Compress conversation history.

        Phase 1: Keep the most recent N turns uncompressed, compress older conversations
        (including old summary) into a new summary.
        Summary is stored independently in self.summary, not mixed into self.history.
        """
        keep_turns = self.config.compress_keep_turns
        keep_messages = keep_turns * 2  # 2 messages per turn

        if len(self.history) <= keep_messages:
            return

        # Old messages to compress
        old_messages = self.history[:-keep_messages]
        recent_messages = self.history[-keep_messages:]

        # Build compression input (including old summary)
        compress_parts = []
        if self.summary:
            compress_parts.append(f"Previous summary: {self.summary}")
        old_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in old_messages
        )
        compress_parts.append(old_text)

        # Call LLM for compression
        try:
            new_summary = self.llm.ask(
                f"Please compress the following conversation history into a concise summary, "
                f"retaining key information (user preferences, important decisions, task progress, key facts):\n\n"
                + "\n".join(compress_parts),
                temperature=0.3,
            )
            if new_summary and not new_summary.startswith("[LLM"):
                self.summary = new_summary
                self.history[:] = recent_messages
                logger.info("Conversation history compressed: %d messages -> %d messages (summary stored separately)",
                            len(old_messages) + len(recent_messages), len(self.history))
            else:
                logger.warning("Conversation compression failed, skipping compression")
        except Exception as e:
            logger.warning("Conversation compression error: %s", e)

    def _trim_history(self) -> None:
        """
        Turn window management: discard oldest turns when context_window_size is exceeded.

        Each conversation turn consists of 2 messages (user + assistant), so the message limit is context_window_size * 2.
        Summary is preserved independently and unaffected.
        """
        max_messages = self.config.context_window_size * 2
        if len(self.history) > max_messages:
            overflow = len(self.history) - max_messages
            self.history[:] = self.history[overflow:]
            logger.debug("Conversation history trimmed, discarded %d oldest messages", overflow)

    def clear_conversation(self) -> None:
        """Clear conversation history and summary."""
        self.history.clear()
        self.summary = None

    # ============================================================
    # Lifecycle
    # ============================================================

    def shutdown(self) -> None:
        """
        Agent shutdown: emits SESSION_END, then calls on_shutdown() on all modules
        in reverse order to release resources (onion model).
        """
        self.emit_hook(HookEvent.SESSION_END, {"modules": list(self.modules.keys())})

        for mod in reversed(list(self.modules.values())):
            try:
                mod.on_shutdown()
            except Exception as e:
                logger.error("Module '%s' on_shutdown error: %s", mod.name, e)
        logger.info("Agent shut down")

    # ============================================================
    # Status query
    # ============================================================

    def status(self) -> dict:
        """
        Return the Agent's current status.

        Returns:
            Dictionary containing model, persona, modules, conversation turns, and other information
        """
        return {
            "model": self.config.model,
            "persona": self.persona.name if self.persona else None,
            "modules": {
                name: mod.description for name, mod in self.modules.items()
            },
            "conversation_turns": len(self.history) // 2,
        }
