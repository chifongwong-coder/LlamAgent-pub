"""
LlamAgent core class: a complete, standalone AI Agent.

Without loading any modules, LlamAgent is a conversational AI assistant.
After loading modules via register_module(), the Agent gains tool calling, RAG, memory, and other capabilities.

Core components:
- LlamAgent:         Main Agent class containing the chat() entry point and run_react() engine
- Module:             Pluggable module base class that interacts with the Agent via pipeline callbacks
- ExecutionStrategy:  Pluggable execution strategy interface, replacing the deprecated on_execute callback
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator, Literal

from llamagent.core.authorization import AuthorizationEngine
from llamagent.core.config import Config
from llamagent.core.contract import PipelineOutcome
from llamagent.core.zone import ConfirmRequest, ConfirmResponse
from llamagent.core.controller import ModeAction, TaskModeController
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
    status: Literal["completed", "max_steps", "timeout", "error", "interrupted", "context_overflow", "aborted"]
    error: str | None = None
    steps_used: int = 0
    reason: str | None = None
    terminal: bool = False  # v2.0: non-recoverable result, caller should stop (not retry/replan)


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

    def execute(self, query: str, context: str, agent: LlamAgent) -> str:
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

    def execute_stream(self, query: str, context: str, agent: LlamAgent):
        """
        Streaming execution. Return a generator of text chunks, or None if not supported.

        Strategies that don't support streaming return None (the default),
        and the caller falls back to execute() + yield the full result.
        """
        return None


class SimpleReAct(ExecutionStrategy):
    """
    Default execution strategy: directly runs the ReAct loop.

    Returns text response directly when there are no tool_calls; otherwise loops through tool calls.

    Backward compatible: if a module overrides on_execute() (deprecated), its result is used first.
    """

    def execute(self, query: str, context: str, agent: LlamAgent) -> str:
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

    def execute_stream(self, query: str, context: str, agent: LlamAgent):
        """Streaming execution via run_react_stream."""
        tools_schema = agent.get_all_tool_schemas()
        tool_dispatch = agent.call_tool
        messages = agent.build_messages(query, context)
        return agent.run_react_stream(messages, tools_schema, tool_dispatch)


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

    def on_attach(self, agent: LlamAgent) -> None:
        """
        Called when the module is registered via register_module().

        Used for initializing storage, registering tools, injecting execution strategies, injecting safety callbacks, etc.
        Base class default saves the agent reference.
        """
        self.agent = agent

    def attach(self, agent: LlamAgent) -> None:
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
# LlamAgent main class
# ======================================================================


class LlamAgent:
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
        Initialize LlamAgent.

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
        self._llm_cache: dict[str, LLMClient] = {self.config.model: self.llm}
        self.modules: dict[str, Module] = {}
        self.history: list[dict] = []
        self.summary: str | None = None

        # Backward compatibility: conversation as an alias for history
        self.conversation = self.history

        # Execution strategy, default SimpleReAct
        self._execution_strategy: ExecutionStrategy = SimpleReAct()

        # v1.3/v1.9: zone-based safety system (v1.9: structured ConfirmRequest/ConfirmResponse)
        self.confirm_handler: Callable[[ConfirmRequest], ConfirmResponse | bool] | None = None
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
        self.mode: str = "interactive"  # v1.9.9: always start interactive; set_mode() handles actual switch
        self._authorization_engine = AuthorizationEngine(self)
        self._controller = None  # v1.9.6: ModeController instance, set by set_mode("task")
        self._current_task_id = None  # Legacy fallback for PlanReAct / Workspace

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

        # v2.0: abort mechanism
        self._abort = False
        # v2.0: open_questions buffer for prepare phase
        self._open_questions_buffer: list[str] = []

        # v2.0: snapshot interactive config values before any mode switch
        self._interactive_config = {k: getattr(self.config, k) for k in self._MODE_KEYS}

        # v1.9.9: config-driven mode initialization
        config_mode = getattr(self.config, "authorization_mode", "interactive")
        if config_mode != "interactive":
            if config_mode not in ("task", "continuous"):
                logger.warning("Invalid authorization_mode '%s' in config, falling back to interactive", config_mode)
            else:
                try:
                    self.set_mode(config_mode)
                except Exception as e:
                    logger.warning("Failed to apply authorization_mode '%s' from config: %s", config_mode, e)

    # ============================================================
    # LLM management
    # ============================================================

    def _get_llm(self, model: str) -> LLMClient:
        """Get or create LLMClient for the given model."""
        if model not in self._llm_cache:
            self._llm_cache[model] = LLMClient(model, self.config.api_retry_count)
        return self._llm_cache[model]

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

        Sets module.llm before on_attach so modules can use self.llm during initialization.

        Args:
            module: The module instance to register
        """
        try:
            # Set module-specific LLM before on_attach (module.name is a class attribute, available here)
            model_name = getattr(self.config, 'module_models', {}).get(module.name)
            module.llm = self._get_llm(model_name) if model_name else self.llm

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
        """Return current authorization state snapshot. Agent adds mode, engine formats scope details."""
        result = self._authorization_engine.authorization_status()
        result["mode"] = self.mode
        return result

    def get_active_task_id(self) -> str | None:
        """
        Get the active task ID for scope storage/lookup.

        Priority: controller.state.task_id > _current_task_id (PlanReAct legacy).
        """
        if self.mode == "task" and self._controller:
            tid = self._controller.state.task_id
            if tid:
                return tid
        return getattr(self, "_current_task_id", None)

    def set_mode(self, mode: str) -> None:
        """
        Switch authorization mode. Single public entry point.

        Sequence: check idle → prepare new controller → clear old scopes →
                  switch policy (loads seed scopes, may ask user) →
                  configure controller auto_execute → commit controller →
                  update mode → emit events.

        Args:
            mode: "interactive" (default) | "task" | "continuous"
        """
        # 0. Validate mode
        valid_modes = ("interactive", "task", "continuous")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # 1. Check if switch is allowed
        if self._controller and not self._controller.is_idle():
            raise RuntimeError("Cannot switch mode while task is active")

        # 2. Prepare new controller / state
        new_controller = None
        new_state = None
        if mode == "task":
            new_controller = TaskModeController()
            new_state = new_controller.state

        # 3-7 wrapped for exception safety: if _switch_policy fails, fall back to interactive
        try:
            # 3. Clear old scopes, collect SCOPE_REVOKED events (before switching policy)
            clear_result = self._authorization_engine._clear_all_scopes(reason="mode_switch")

            # 4. Switch policy (loads seed scopes, may ask user for project access in task mode)
            switch_result = self._authorization_engine._switch_policy(mode, state=new_state)

            # 5. Configure controller based on policy result (before commit)
            if new_controller is not None:
                new_controller.auto_execute = switch_result.has_session_scopes

            # 6. Commit controller reference
            self._controller = new_controller

            # 7. Update mode
            self.mode = mode

            # 8. Apply mode-aware config defaults (v2.0)
            if mode == "interactive":
                defaults = self._interactive_config
            else:
                defaults = self._MODE_DEFAULTS.get(mode, {})
            for k, v in defaults.items():
                setattr(self.config, k, v)
        except Exception as e:
            # Roll back to consistent interactive state
            logger.error("set_mode('%s') failed, falling back to interactive: %s", mode, e)
            try:
                self._authorization_engine._switch_policy("interactive")
            except Exception as fallback_e:
                logger.error("Fallback to interactive also failed: %s", fallback_e)
            self._controller = None
            self.mode = "interactive"
            for k, v in self._interactive_config.items():
                setattr(self.config, k, v)
            raise

        # 9. Emit events from clearing + switching (agent doesn't inspect, just forwards)
        for event_name, data in clear_result.events + switch_result.events:
            try:
                self.emit_hook(HookEvent(event_name), data)
            except ValueError:
                logger.warning("Unknown hook event: %s", event_name)

        logger.info("Authorization mode switched to: %s", mode)

    def abort(self) -> None:
        """Signal the agent to stop after the current atomic operation.

        The flag is checked at two points in run_react (loop top + after each tool call).
        chat() resets the flag at entry, so stale abort signals don't affect new tasks.
        """
        self._abort = True

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
    # Unified pipeline (v1.9.6)
    # ============================================================

    # Maximum iterations for the task mode driving loop (defensive guard)
    _MAX_MODE_STEPS = 10

    # v2.0: mode-aware config defaults. -1 = unlimited/disabled.
    _MODE_KEYS = {"max_react_steps", "max_duplicate_actions", "react_timeout",
                  "max_observation_tokens"}
    _MODE_DEFAULTS = {
        "task":       {"max_react_steps": 50, "react_timeout": 600,
                       "max_duplicate_actions": 5, "max_observation_tokens": 5000},
        "continuous": {"max_react_steps": -1, "react_timeout": 600,
                       "max_duplicate_actions": -1, "max_observation_tokens": 10000},
    }

    _PREPARE_TOOL_NAME = "_report_question"

    def _register_prepare_tools(self) -> None:
        """Register prepare-only tools (available during dry-run, removed after)."""
        if self._PREPARE_TOOL_NAME not in self._tools:
            self.register_tool(
                name=self._PREPARE_TOOL_NAME,
                func=lambda question: self._open_questions_buffer.append(question) or f"Question recorded: {question}",
                description="Report an open question or uncertainty during task planning. Use this when you need clarification from the user before execution.",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question or uncertainty to report"},
                    },
                    "required": ["question"],
                },
                tier="default",
                safety_level=0,
            )

    def _unregister_prepare_tools(self) -> None:
        """Remove prepare-only tools after dry-run completes."""
        self._tools.pop(self._PREPARE_TOOL_NAME, None)

    def _run_pipeline(
        self,
        query: str,
        *,
        mode: str = "normal",
        extra_system: str = "",
        skip_output: bool = False,
        task_id: str | None = None,
        record_history: bool = True,
    ) -> PipelineOutcome:
        """
        Unified pipeline: on_input → on_context → strategy → on_output → history.

        mode="normal": full pipeline
        mode="prepare": dry-run (clear pending buffer, skip on_output, no history,
                        drain prepare data into outcome.metadata at the end)
        """
        is_prepare = mode == "prepare"

        # Prepare mode: clear stale buffers and inject report_question tool
        if is_prepare:
            self._authorization_engine.clear_pending_buffer()
            self._open_questions_buffer.clear()
            self._register_prepare_tools()

        # 1. on_input
        processed = query
        for mod in self.modules.values():
            try:
                processed = mod.on_input(processed)
            except Exception as e:
                logger.error("Module '%s' on_input error: %s", mod.name, e)

        if not processed or not processed.strip():
            if is_prepare:
                self._unregister_prepare_tools()
            return PipelineOutcome(response="Sorry, I cannot process this request.", task_id=task_id, blocked=True)

        # 2. on_context
        context = ""
        for mod in self.modules.values():
            try:
                context = mod.on_context(processed, context)
            except Exception as e:
                logger.error("Module '%s' on_context error: %s", mod.name, e)

        # 3. Execution strategy
        try:
            if is_prepare:
                # Prepare mode: inject extra_system, run react directly
                messages = self.build_messages(processed, context, extra_system=extra_system)
                tools_schema = self.get_all_tool_schemas()
                react_result = self.run_react(messages, tools_schema, self.call_tool)
                response = react_result.text
            else:
                # Normal mode: use execution strategy (SimpleReAct or PlanReAct)
                if extra_system:
                    response = self._execution_strategy.execute(processed, context + "\n" + extra_system, self)
                else:
                    response = self._execution_strategy.execute(processed, context, self)
        except Exception as e:
            logger.error("Execution error: %s", e)
            response = f"Error processing request: {e}"

        # 4. on_output (skip for prepare mode)
        if not is_prepare and not skip_output:
            for mod in reversed(list(self.modules.values())):
                try:
                    response = mod.on_output(response)
                except Exception as e:
                    logger.error("Module '%s' on_output error: %s", mod.name, e)

        # 5. Build outcome
        outcome = PipelineOutcome(response=response, task_id=task_id)

        # Prepare mode: drain accumulated data from engine into outcome.metadata
        if is_prepare:
            outcome.metadata.update(self._authorization_engine.drain_prepare_data())
            if self._open_questions_buffer:
                outcome.metadata["open_questions"] = list(self._open_questions_buffer)
                self._open_questions_buffer.clear()
            self._unregister_prepare_tools()

        # 6. History (skip for prepare mode and when record_history=False)
        if record_history and not is_prepare:
            self.history.append({"role": "user", "content": processed})
            self.history.append({"role": "assistant", "content": response})
            self._trim_history()

        return outcome

    # ============================================================
    # Task mode driving loop (v1.9.6)
    # ============================================================

    def _run_controller_turn(self, user_input: str) -> str:
        """
        Run one chat() turn through the task mode controller.

        Drives the two-step protocol: handle_turn → pipeline → on_pipeline_done,
        in a loop until a terminal action (reply/await_user/cancel) is reached.
        """
        action = self._controller.handle_turn(user_input)

        for _ in range(self._MAX_MODE_STEPS):
            # 1. Process authorization_update BEFORE break (cleanup must not be skipped)
            if action.authorization_update is not None:
                result = self._authorization_engine.apply_update(action.authorization_update)
                for event_name, data in result.events:
                    try:
                        self.emit_hook(HookEvent(event_name), data)
                    except ValueError:
                        logger.warning("Unknown hook event from apply_update: %s", event_name)

            # 2. Terminal actions exit loop
            if action.kind in ("reply", "await_user", "cancel"):
                break

            # 3. Execute pipeline
            outcome = None  # Prevent UnboundLocalError on BaseException

            if action.kind == "run_prepare":
                try:
                    outcome = self._run_pipeline(
                        action.query, mode="prepare",
                        extra_system=action.extra_system,
                        task_id=action.task_id,
                        record_history=False,
                    )
                except Exception as e:
                    outcome = PipelineOutcome(response=f"Error: {e}", task_id=action.task_id)
                action = self._controller.on_pipeline_done(action, outcome)

            elif action.kind == "run_execute":
                self._current_task_id = action.task_id
                try:
                    try:
                        outcome = self._run_pipeline(
                            action.query, mode="normal",
                            task_id=action.task_id,
                            record_history=False,
                        )
                    except Exception as e:
                        outcome = PipelineOutcome(response=f"Error: {e}", task_id=action.task_id)
                    action = self._controller.on_pipeline_done(action, outcome)
                finally:
                    self._current_task_id = None
        else:
            # MAX_MODE_STEPS exhausted
            action = ModeAction(kind="reply", response="Task mode exceeded maximum steps.")

        # 4. Post-loop: hand-write history based on final action
        self._write_task_mode_history(user_input, action)

        return action.response or ""

    def _write_task_mode_history(self, user_input: str, action: ModeAction) -> None:
        """
        Write history after task mode driving loop exits.

        Agent does NOT read controller.state — all needed info comes from ModeAction fields:
        - action.query: user message for history (set by controller)
        - action.response: assistant message for history
        """
        if action.kind == "await_user":
            # Prepare complete — action.query carries the user message
            # (original_query for first prepare, supplementary input for re-prepare)
            self.history.append({"role": "user", "content": action.query or user_input})
            if action.response:
                self.history.append({"role": "assistant", "content": action.response})
        elif action.kind == "reply":
            # Execute complete — write confirmation + result
            self.history.append({"role": "user", "content": user_input})
            if action.response:
                self.history.append({"role": "assistant", "content": action.response})
        elif action.kind == "cancel":
            # Cancel — write cancel input + cancel message
            self.history.append({"role": "user", "content": user_input})
            if action.response:
                self.history.append({"role": "assistant", "content": action.response})
        else:
            logger.warning("Unknown action kind '%s' in task mode history, skipping history write", action.kind)

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
        if self.confirm_handler is None:
            return ConfirmResponse(allow=False)
        t0 = time.time()
        try:
            response = self.confirm_handler(request)
            if not isinstance(response, ConfirmResponse):
                # Backward compat: if handler returns bool, wrap it
                response = ConfirmResponse(allow=bool(response))
        except Exception as e:
            logger.warning("confirm_handler raised exception, defaulting to deny: %s", e)
            response = ConfirmResponse(allow=False)
        self._confirm_wait_time += time.time() - t0
        return response

    def chat(self, user_input: str) -> str:
        """
        Agent main entry point: receives user input and returns a response.

        Two execution paths:
        - Controller mode (_controller is not None): delegates to _run_controller_turn(),
          which drives the controller's two-step protocol (handle_turn / on_pipeline_done).
        - Normal mode: on_input -> on_context -> execution strategy -> on_output.
          If on_input blocks the input, outcome.blocked is set and LLM is skipped.

        Args:
            user_input: User input, should not be an empty string

        Returns:
            The Agent's response text
        """
        # --- v2.0: reset abort flag (clear stale signal from previous task) ---
        self._abort = False

        # --- SESSION_START (fires once on first chat) ---
        if not self._session_started:
            self.emit_hook(HookEvent.SESSION_START, {"modules": list(self.modules.keys())})
            self._session_started = True

        # --- PRE_CHAT (observational, SKIP not supported) ---
        self.emit_hook(HookEvent.PRE_CHAT, {"user_input": user_input})

        blocked = False
        blocked_by = None
        response = ""

        # --- Controller mode branch (v1.9.7: controller-presence dispatch) ---
        if self._controller is not None:
            response = self._run_controller_turn(user_input)
            self.emit_hook(HookEvent.POST_CHAT, {
                "user_input": user_input, "response": response,
                "blocked": False, "blocked_by": None, "completed": True,
            })
            return response

        # --- Before conversation: context management (reserved for token compression) ---
        self._check_context_compression()

        # --- Normal pipeline via unified _run_pipeline ---
        outcome = self._run_pipeline(user_input, mode="normal", record_history=True)
        response = outcome.response
        if outcome.blocked:
            blocked = True
            blocked_by = "safety"

        # --- POST_CHAT (always-fire, all exit paths) ---
        self.emit_hook(HookEvent.POST_CHAT, {
            "user_input": user_input,
            "response": response,
            "blocked": blocked,
            "blocked_by": blocked_by,
            "completed": not blocked,
        })

        return response

    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        Streaming version of chat(). Yields text chunks.

        - Pure text: streams LLM content token by token
        - Tool calls: yields status messages
        - on_output applied post-hoc to accumulated text for history recording
        - Task mode / PlanReAct: fallback to non-streaming (full execute, yield at once)
        """
        # 1. Reset abort
        self._abort = False

        # 2. SESSION_START hook (once)
        if not self._session_started:
            self.emit_hook(HookEvent.SESSION_START, {"modules": list(self.modules.keys())})
            self._session_started = True

        # 3. PRE_CHAT hook
        self.emit_hook(HookEvent.PRE_CHAT, {"user_input": user_input})

        # 4. Controller guard: task mode doesn't support streaming, fallback
        if self._controller is not None:
            response = self._run_controller_turn(user_input)
            yield response
            self.emit_hook(HookEvent.POST_CHAT, {
                "user_input": user_input, "response": response,
                "blocked": False, "blocked_by": None, "completed": True,
            })
            return

        # 5. on_input pipeline (safety — runs before streaming)
        processed = user_input
        for mod in self.modules.values():
            try:
                processed = mod.on_input(processed)
            except Exception as e:
                logger.error("Module '%s' on_input error: %s", mod.name, e)
        if not processed or not processed.strip():
            yield "[Input blocked by safety module]"
            self.emit_hook(HookEvent.POST_CHAT, {
                "user_input": user_input, "response": "",
                "blocked": True, "blocked_by": "safety", "completed": False,
            })
            return

        # 6. Context compression
        self._check_context_compression()

        # 7. on_context pipeline
        context = ""
        for mod in self.modules.values():
            try:
                context = mod.on_context(processed, context)
            except Exception as e:
                logger.error("Module '%s' on_context error: %s", mod.name, e)

        # 8. Route through strategy interface (agent doesn't check strategy type)
        stream_gen = self._execution_strategy.execute_stream(processed, context, self)

        accumulated = ""
        try:
            if stream_gen is None:
                # Strategy doesn't support streaming → fallback to full execute
                response = self._execution_strategy.execute(processed, context, self)
                accumulated = response
                yield response
            else:
                # Strategy supports streaming → yield chunks
                for chunk in stream_gen:
                    accumulated += chunk
                    yield chunk
        except GeneratorExit:
            # Generator abandoned (caller broke early / Ctrl+C / GC)
            # Still record history with whatever was accumulated
            pass
        finally:
            # 9. on_output post-processing (on accumulated full text)
            final_text = accumulated
            for mod in reversed(list(self.modules.values())):
                try:
                    final_text = mod.on_output(final_text)
                except Exception as e:
                    logger.error("Module '%s' on_output error: %s", mod.name, e)

            # 10. Record history (on_output processed version)
            if accumulated:
                self.history.append({"role": "user", "content": processed})
                self.history.append({"role": "assistant", "content": final_text})

            # 11. POST_CHAT hook
            self.emit_hook(HookEvent.POST_CHAT, {
                "user_input": user_input, "response": final_text,
                "blocked": False, "blocked_by": None, "completed": True,
            })

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

        while self.config.max_react_steps == -1 or steps < self.config.max_react_steps:
            # --- v2.0: abort check (before LLM call) ---
            if self._abort:
                return ReactResult(text="Operation aborted.", status="aborted",
                                   terminal=True, steps_used=steps)

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
                                       error=str(e), steps_used=steps, terminal=True)
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
                    if self.config.max_duplicate_actions != -1 and duplicate_count >= self.config.max_duplicate_actions:
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

                # --- v2.0: abort check (after tool call) ---
                if self._abort:
                    return ReactResult(text="Operation aborted.", status="aborted",
                                       terminal=True, steps_used=steps)

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

    # ============================================================
    # Streaming ReAct loop (v2.0.2)
    # ============================================================

    def run_react_stream(
        self,
        messages: list[dict],
        tools_schema: list[dict],
        tool_dispatch: Callable[[str, dict], str],
    ) -> Generator[str, None, None]:
        """
        Streaming version of run_react(). Yields text chunks.

        - Pure text: streams LLM content token by token
        - Tool calls: yields status messages ("[Calling tool...]", "[tool done]")
        - Includes full loop protections: abort, duplicate detection, timeout, max_steps
        """
        # No tools: pure stream
        if not tools_schema:
            try:
                for chunk in self.llm.chat_stream(messages):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content
            except Exception as e:
                logger.error("LLM stream call failed: %s", e)
                yield f"\n[Error: {e}]"
            return

        steps = 0
        last_action: tuple | None = None
        duplicate_count = 0

        while self.config.max_react_steps == -1 or steps < self.config.max_react_steps:
            # Abort check
            if self._abort:
                yield "\n[Operation aborted]"
                return

            step_start = time.time()
            self._confirm_wait_time = 0.0

            # Stream LLM call, yield content and accumulate tool_calls
            accumulated_content = ""
            accumulated_tool_calls: list[dict] = []

            try:
                for chunk in self.llm.chat_stream(messages, tools=tools_schema,
                                                   timeout=self.config.react_timeout):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        accumulated_content += delta.content
                        yield delta.content
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        self._merge_tool_call_deltas(accumulated_tool_calls, delta.tool_calls)
            except Exception as e:
                error_name = type(e).__name__
                if "ContextWindow" in error_name:
                    yield "\n[Context window overflow]"
                    return
                logger.error("ReAct stream LLM call failed: %s", e)
                yield f"\n[Error: {e}]"
                return

            # No tool calls → pure text (already yielded)
            if not accumulated_tool_calls:
                return

            # Build assistant message and append to messages
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": accumulated_content or ""}
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]}}
                for tc in accumulated_tool_calls
            ]
            messages.append(assistant_msg)

            # Dispatch each tool call
            for tc in accumulated_tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    tool_args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                except json.JSONDecodeError:
                    tool_args = {}

                # Duplicate detection
                current_action = (tool_name, json.dumps(tool_args, sort_keys=True))
                if current_action == last_action:
                    duplicate_count += 1
                    if self.config.max_duplicate_actions != -1 and duplicate_count >= self.config.max_duplicate_actions:
                        yield f"\n[Duplicate action detected ({tool_name}), aborting]"
                        return
                else:
                    last_action = current_action
                    duplicate_count = 1

                yield f"\n[Calling {tool_name}...]\n"
                try:
                    result = tool_dispatch(tool_name, tool_args)
                    result = str(result) if result is not None else ""
                except Exception as e:
                    logger.error("Tool '%s' dispatch error: %s", tool_name, e)
                    result = f"Tool execution error: {e}"
                result = self._truncate_observation(result)
                yield f"[{tool_name} done]\n"

                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

                # Abort check after each tool
                if self._abort:
                    yield "\n[Operation aborted]"
                    return

            # Timeout check
            elapsed = time.time() - step_start - self._confirm_wait_time
            if elapsed > self.config.react_timeout:
                yield "\n[Step timeout]"
                return

            steps += 1

        yield "\n[Maximum steps reached]"

    @staticmethod
    def _merge_tool_call_deltas(accumulated: list[dict], deltas: list) -> None:
        """Merge incremental tool_call deltas into accumulated list (by index)."""
        for delta in deltas:
            idx = delta.index if hasattr(delta, "index") else 0
            while len(accumulated) <= idx:
                accumulated.append({"id": "", "function": {"name": "", "arguments": ""}})
            tc = accumulated[idx]
            if hasattr(delta, "id") and delta.id:
                tc["id"] = delta.id
            if hasattr(delta, "function") and delta.function:
                if hasattr(delta.function, "name") and delta.function.name:
                    tc["function"]["name"] += delta.function.name
                if hasattr(delta.function, "arguments") and delta.function.arguments:
                    tc["function"]["arguments"] += delta.function.arguments

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

    def compress_conversation(self, new_summary: str, keep_turns: int) -> None:
        """Replace old history with summary, keeping recent turns.

        Called by compression module. Agent manages its own state.
        """
        keep_messages = keep_turns * 2
        if len(self.history) > keep_messages:
            self.history[:] = self.history[-keep_messages:]
        self.summary = new_summary

    def _check_context_compression(self) -> None:
        """Fallback: clear conversation when tokens hit hard limit."""
        if not self.history:
            return
        try:
            token_count = self.llm.count_tokens(self.history)
            if token_count >= self.config.max_context_tokens:
                logger.warning("Context tokens (%d) exceed limit (%d), clearing conversation",
                               token_count, self.config.max_context_tokens)
                self.clear_conversation()
        except Exception as e:
            logger.debug("Context check error (non-fatal): %s", e)

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
