"""
SmartAgent core class: a complete, standalone AI Agent.

Without loading any modules, SmartAgent is a conversational AI assistant.
After loading modules via register_module(), the Agent gains tool calling, RAG, memory, and other capabilities.

Core components:
- SmartAgent:         Main Agent class containing the chat() entry point and run_react() engine
- Module:             Pluggable module base class that interacts with the Agent via pipeline hooks
- ExecutionStrategy:  Pluggable execution strategy interface, replacing the deprecated on_execute hook
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from llamagent.core.config import Config
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
        # Backward compatibility: check if any module intercepts via on_execute hook
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

    Modules interact with the Agent pipeline by overriding hook methods.
    All hook base implementations are no-ops; modules override as needed.

    Lifecycle Hooks (called once each):
    - on_attach(agent)    Called on registration, used for initialization
    - on_shutdown()       Called on Agent exit, used to release resources

    Pipeline Hooks (called each conversation turn):
    - on_input(user_input) -> str        Input preprocessing
    - on_context(query, context) -> str  Context enhancement
    - on_output(response) -> str         Output post-processing

    Full lifecycle:
    on_attach -> [on_input -> on_context -> execution strategy -> on_output] x N -> on_shutdown
    """

    name: str = "base"
    description: str = ""

    # --- Lifecycle Hooks ---

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

    # --- Pipeline Hooks ---

    def on_input(self, user_input: str) -> str:
        """
        Input preprocessing hook.

        Called in module registration order (forward). Returning an empty string is treated as a safety interception.
        """
        return user_input

    def on_context(self, query: str, context: str) -> str:
        """
        Context enhancement hook.

        Called in module registration order (forward). Each module appends context in sequence.
        """
        return context

    def on_output(self, response: str) -> str:
        """
        Output post-processing hook.

        Called in **reverse** module registration order (onion model).
        """
        return response

    # --- Deprecated Hooks (backward compatible, will be removed in future versions) ---

    def on_execute(self, query: str, context: str) -> str | None:
        """
        [Deprecated] Execution interception hook; returning non-None skips default execution.

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

        # Safety module loaded flag: when True, core fallback (block sl>=2) is disabled
        # Set by SafetyModule.on_attach(); real safety is handled by on_input/on_output hooks
        self.safety_loaded: bool = False
        self.tool_executor = None  # v1.2: injected by SandboxModule for sandbox execution dispatch

        # Tool registry (simple implementation, later enhanced by tools module)
        # Format: {name: {"func": callable, "description": str, "parameters": dict,
        #               "tier": str, "safety_level": int}}
        self._tools: dict[str, dict[str, Any]] = {}

        # Tool registry version number, incremented on each register_tool/remove_tool, used for cache invalidation
        self._tools_version: int = 0

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
        Registry tool call: lookup + permission check + execution.

        Tool not found and permission denied are both returned as strings, serving as tool observations fed back to the model.

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

        # 2. Safety fallback: without safety module, block tools with safety_level >= 2
        if not self.safety_loaded and tool.get("safety_level", 1) >= 2:
            return f"Tool '{name}' requires safety module. Please load SafetyModule before using this tool."

        # v1.2: route through ToolExecutor if available
        if self.tool_executor is not None:
            try:
                return self.tool_executor.execute(tool, args)
            except Exception as e:
                logger.error("Tool '%s' sandbox execution error: %s", name, e)
                return f"Tool '{name}' execution error: {e}"

        # 3. Execute tool (expand dict internally)
        try:
            result = tool["func"](**args)
            return str(result) if result is not None else ""
        except Exception as e:
            logger.error("Tool '%s' execution error: %s", name, e)
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

            # Filter by tier (visibility control)
            if tier == "admin" and not is_admin:
                continue
            # Agent-tier tool filtering is handled by the module (no filtering here for now)

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
    # Conversation (core capability)
    # ============================================================

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
        # --- Before conversation: context management (reserved for token compression) ---
        self._check_context_compression()

        # --- 1. on_input: modules preprocess input (forward registration order) ---
        processed = user_input
        for mod in self.modules.values():
            try:
                processed = mod.on_input(processed)
            except Exception as e:
                logger.error("Module '%s' on_input error: %s", mod.name, e)

        # Safety interception: if input is cleared (e.g., blocked by safety module), return immediately
        if not processed or not processed.strip():
            return "Sorry, I cannot process this request."

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
            elapsed = time.time() - step_start
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
        Agent shutdown: calls on_shutdown() on all modules in reverse order to release resources (onion model).
        """
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
