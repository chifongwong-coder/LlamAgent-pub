"""
ChildAgentModule: spawn and control child agents with budget and capability boundaries.

Registers tools for the parent agent:
- spawn_child:      Spawn a child agent for a subtask with role-based constraints
- list_children:    List all spawned child agents and their status
- collect_results:  Collect results from all completed child agents
- wait_child:       (thread/process runner) Wait for a specific child and get its result

The module supports three runner backends:
- InlineRunnerBackend (default): synchronous, in-process execution
- ThreadRunnerBackend: concurrent execution in daemon threads
- ProcessRunnerBackend: isolated execution in child subprocesses

Each child agent is a constrained LlamAgent with filtered tools, budget limits,
and no ability to spawn its own children (unless explicitly allowed by policy).
"""

from __future__ import annotations

import copy
import logging
import time

from llamagent.core.agent import Module
from llamagent.modules.child_agent.budget import BudgetedLLM, BudgetTracker
from llamagent.modules.child_agent.policy import (
    AgentExecutionPolicy,
    ChildAgentSpec,
    ROLE_POLICIES,
)
from llamagent.modules.child_agent.runner import AgentRunnerBackend
from llamagent.modules.child_agent.runners.inline import InlineRunnerBackend
from llamagent.modules.child_agent.task_board import TaskBoard, TaskRecord

logger = logging.getLogger(__name__)


# LLM guide injected via on_context when runner is async (thread or process)
CHILD_AGENT_GUIDE_ASYNC = (
    "[Child Agent] You can spawn multiple child agents in parallel.\n"
    "- Use spawn_child to start a child agent (returns immediately with task_id).\n"
    "- Use wait_child to wait for a specific child and get its result.\n"
    "- Use collect_results to gather all completed results.\n"
    "- Use list_children to check status of all spawned children.\n"
    "Spawn multiple children first, then collect results "
    "-- this is faster than sequential execution."
)


# ======================================================================
# ChildAgentController
# ======================================================================


class ChildAgentController:
    """
    Controller that coordinates the runner backend and task board.

    Provides a clean API for spawning, waiting, listing, and collecting
    child agent results. The module delegates all orchestration to this class.

    The controller is runner-agnostic: it uses runner.spawn() + runner.status()
    to determine initial state, then syncs to the task board. For thread runner,
    it registers an on_complete callback for async updates.
    """

    def __init__(self, runner: AgentRunnerBackend, task_board: TaskBoard,
                 max_children: int = 20, module_on_complete=None):
        self.runner = runner
        self.task_board = task_board
        self.max_children = max_children
        self._module_on_complete = module_on_complete

        # Wire completion callback for async runners
        if hasattr(runner, '_on_complete'):
            runner._on_complete = self._on_child_complete

    def spawn_child(self, spec: ChildAgentSpec, agent_factory) -> str:
        """
        Spawn a child agent and register it on the task board.

        For inline runner: spawn blocks, status is completed/failed immediately.
        For thread runner: spawn returns immediately, status is running.

        Args:
            spec: Child agent specification.
            agent_factory: Callable(spec) -> LlamAgent.

        Returns:
            The unique task_id.

        Raises:
            RuntimeError: If max_children limit is reached.
        """
        # Guard: prevent unbounded child spawning
        if spec.parent_task_id:
            existing = len(self.task_board.children_of(spec.parent_task_id))
            if existing >= self.max_children:
                raise RuntimeError(
                    f"Max children limit reached ({self.max_children}). "
                    f"Cannot spawn more child agents."
                )

        # Pre-generate task_id so we can create the board entry BEFORE spawning.
        # This prevents a race where the callback fires before the board entry exists
        # (happens when the child completes instantly, e.g., mock LLM in tests).
        import uuid
        task_id = uuid.uuid4().hex[:12]

        # Set task_id on spec so the agent factory can use it (e.g., for workspace dir)
        spec.task_id = task_id

        # Build input snapshot for debugging/auditing
        input_snapshot = {
            "task": spec.task,
            "role": spec.role,
            "context": spec.context[:200] if spec.context else "",
            "artifact_refs": spec.artifact_refs,
        }
        if spec.policy:
            input_snapshot["tool_allowlist"] = spec.policy.tool_allowlist
            input_snapshot["budget"] = {
                "max_tokens": spec.policy.budget.max_tokens if spec.policy.budget else None,
                "max_time_seconds": spec.policy.budget.max_time_seconds if spec.policy.budget else None,
                "max_steps": spec.policy.budget.max_steps if spec.policy.budget else None,
            }
            # Record child_root path for isolated children (share_parent_project_dir=False)
            if not spec.policy.share_parent_project_dir:
                import os
                # child_root is derived from parent playground_dir (set by module)
                input_snapshot["child_root"] = os.path.join(
                    "children", task_id
                )

        # Create board entry FIRST (status=running), then spawn.
        # The callback or status check will update it when the child completes.
        self.task_board.create(
            task_id=task_id,
            parent_id=spec.parent_task_id,
            role=spec.role,
            task=spec.task,
            status="running",
            input_snapshot=input_snapshot,
            created_at=time.time(),
        )

        # Spawn with the pre-generated task_id
        self.runner.spawn(spec, agent_factory, task_id=task_id)

        # For sync runners (inline), the child is already done. Sync result immediately.
        runner_status = self.runner.status(task_id)
        if runner_status in ("completed", "failed"):
            record = self.runner.wait(task_id)
            self.task_board.update(
                task_id,
                status=record.status,
                result=record.result,
                history=record.history,
                metrics=record.metrics,
                logs=record.logs,
                completed_at=time.time(),
            )
            # Trigger module callback for inline completion (memorize entry point)
            if self._module_on_complete:
                board_record = self.task_board.get(task_id)
                if board_record:
                    self._module_on_complete(task_id, board_record)

        return task_id

    def _on_child_complete(self, task_id: str, record: TaskRecord):
        """
        Callback invoked by ThreadRunner when a child finishes.

        Idempotency guard: if the record is already completed/failed on the
        task board (e.g., inline fast completion), skip the update.
        """
        existing = self.task_board.get(task_id)
        if existing and existing.status in ("completed", "failed", "cancelled"):
            return
        try:
            self.task_board.update(
                task_id,
                status=record.status,
                result=record.result,
                history=record.history,
                metrics=record.metrics,
                logs=record.logs,
                completed_at=time.time(),
            )
        except KeyError:
            # Task board record not yet created (unlikely race); log and skip
            logger.warning("on_child_complete: task_id '%s' not on board", task_id)

        # Trigger module callback (memorize entry point for async runners)
        if self._module_on_complete:
            board_record = self.task_board.get(task_id)
            if board_record:
                self._module_on_complete(task_id, board_record)

    def wait_child(self, task_id: str, timeout: float | None = None) -> TaskRecord:
        """Wait for a child and return its task record from the board."""
        record = self.task_board.get(task_id)
        if record is not None and record.status in ("completed", "failed"):
            return record
        # Block on runner (thread mode: wait for event; inline: immediate)
        runner_record = self.runner.wait(task_id, timeout)
        # Proactively sync runner result to board (callback may not have fired yet)
        if runner_record.status in ("completed", "failed"):
            try:
                self.task_board.update(
                    task_id,
                    status=runner_record.status,
                    result=runner_record.result,
                    history=runner_record.history,
                    metrics=runner_record.metrics,
                    logs=runner_record.logs,
                    completed_at=runner_record.completed_at,
                )
            except KeyError:
                pass  # Board entry not yet created (shouldn't happen)
        return runner_record

    def list_children(self, parent_id: str) -> list[TaskRecord]:
        """List all children belonging to a parent."""
        return self.task_board.children_of(parent_id)

    def cancel_child(self, task_id: str) -> bool:
        """Cancel a running child agent."""
        success = self.runner.cancel(task_id)
        if success:
            # Only overwrite status if still running (child may have completed before abort took effect)
            existing = self.task_board.get(task_id)
            if existing and existing.status == "running":
                self.task_board.update(task_id, status="cancelled")
        return success

    def collect_results(self, parent_id: str) -> list[TaskRecord]:
        """Collect completed/failed results for a parent."""
        return self.task_board.collect_results(parent_id)


# ======================================================================
# ChildAgentModule
# ======================================================================


class ChildAgentModule(Module):
    """
    Child agent control module.

    Spawn and control child agents with budget and capability boundaries.
    Each child is a constrained LlamAgent instance that inherits the parent's
    LLM and selected tools, but operates under strict resource limits.

    Supports three runner backends:
    - inline (default): synchronous, blocking execution
    - thread: concurrent execution with async result collection
    - process: isolated execution in child subprocesses
    """

    name = "child_agent"
    description = "Spawn and control child agents with budget and capability boundaries"

    def __init__(self):
        self.controller: ChildAgentController | None = None
        self.task_board: TaskBoard | None = None
        self._parent_id: str | None = None  # Scope key for list/collect
        self._runner_name: str = "inline"
        self._role_model_overrides: dict[str, str] = {}
        self._auto_memorize: bool = True
        # Messaging infrastructure (initialized for continuous mode agents)
        self._channel = None
        self._registry = None
        self._agent_id: str | None = None

    def on_attach(self, agent):
        """Initialize controller, task board, and register tools."""
        super().on_attach(agent)
        self._parent_id = agent.agent_id
        self.task_board = TaskBoard()

        # Read config-level role model overrides
        self._role_model_overrides = getattr(agent.config, "child_agent_role_models", {}) or {}
        self._auto_memorize = getattr(agent.config, "child_agent_auto_memorize", True)

        # Select runner backend from config
        self._runner_name = getattr(agent.config, "child_agent_runner", "inline")
        max_children = getattr(agent.config, "child_agent_max_children", 20)

        if self._runner_name == "thread":
            from llamagent.modules.child_agent.runners.thread import ThreadRunnerBackend
            runner = ThreadRunnerBackend()
        elif self._runner_name == "process":
            from llamagent.modules.child_agent.runners.process import ProcessRunnerBackend
            runner = ProcessRunnerBackend(
                parent_config=agent.config,
                parent_has_sandbox=agent.tool_executor is not None,
                parent_agent=agent,
            )
        else:
            runner = InlineRunnerBackend()

        self.controller = ChildAgentController(
            runner, self.task_board, max_children=max_children,
            module_on_complete=self._on_module_child_complete,
        )

        # Register tools
        agent.register_tool(
            name="spawn_child",
            func=self._spawn_child,
            description="Spawn a child agent to handle a subtask with controlled budget and capabilities",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task description for the child agent",
                    },
                    "role": {
                        "type": "string",
                        "description": "Role: researcher/writer/analyst/coder/worker/delegate",
                        "enum": ["researcher", "writer", "analyst", "coder", "worker", "delegate"],
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context from parent agent",
                    },
                },
                "required": ["task"],
            },
            tier="default",
            safety_level=2,
        )

        agent.register_tool(
            name="list_children",
            func=self._list_children,
            description="List all spawned child agents and their status",
            tier="default",
            safety_level=1,
        )

        agent.register_tool(
            name="collect_results",
            func=self._collect_results,
            description="Collect results from all completed child agents",
            parameters={
                "type": "object",
                "properties": {
                    "wait": {
                        "type": "boolean",
                        "description": "If true, wait for all running children to complete before collecting",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Maximum seconds to wait when wait=true (default 300)",
                    },
                },
            },
            tier="default",
            safety_level=1,
        )

        # Register wait_child for async runners (thread and process)
        if self._runner_name in ("thread", "process"):
            agent.register_tool(
                name="wait_child",
                func=self._wait_child,
                description="Wait for a specific child agent to complete and return its result",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task_id of the child agent to wait for",
                        },
                        "include_history": {
                            "type": "boolean",
                            "description": "If true, include the full conversation history of the child agent",
                        },
                        "include_logs": {
                            "type": "boolean",
                            "description": "If true, include the child agent's execution logs (stderr for process, captured logs for thread)",
                        },
                    },
                    "required": ["task_id"],
                },
                tier="default",
                safety_level=1,
            )

        # Initialize messaging infrastructure for continuous mode agents
        if agent.mode == "continuous":
            self._init_messaging(agent)
            # Pass channel to thread runner for MessageTrigger creation
            if self._runner_name == "thread" and self._channel is not None:
                self.controller.runner._channel = self._channel

    def _init_messaging(self, agent):
        """Set up MessageChannel, AgentRegistry, and register messaging tools."""
        from llamagent.core.message_channel import AgentRegistry, MessageChannel

        self._channel = MessageChannel()
        self._registry = AgentRegistry(self._channel)
        self._agent_id = agent.agent_id
        self._registry.register(self._agent_id, role="parent", mode="continuous")

        agent.register_tool(
            name="send_message",
            func=self._tool_send_message,
            description="Send a message to another agent by agent_id",
            parameters={
                "type": "object",
                "properties": {
                    "to_id": {
                        "type": "string",
                        "description": "The agent_id of the recipient",
                    },
                    "content": {
                        "type": "string",
                        "description": "Message content to send",
                    },
                    "msg_type": {
                        "type": "string",
                        "description": "Message type: info, alert, request, or response",
                        "enum": ["info", "alert", "request", "response"],
                    },
                },
                "required": ["to_id", "content"],
            },
            tier="default",
            safety_level=1,
        )

        agent.register_tool(
            name="check_messages",
            func=self._tool_check_messages,
            description="Check and return all pending messages in your inbox",
            tier="default",
            safety_level=1,
        )

        agent.register_tool(
            name="list_agents",
            func=self._tool_list_agents,
            description="List all active agents in the registry",
            tier="default",
            safety_level=1,
        )

        agent.register_tool(
            name="spawn_continuous_child",
            func=self._spawn_continuous_child,
            description="Spawn a long-running continuous child agent driven by triggers",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Ongoing task description for the child agent",
                    },
                    "role": {
                        "type": "string",
                        "description": "Role: researcher/writer/analyst/coder/worker",
                        "enum": ["researcher", "writer", "analyst", "coder", "worker"],
                    },
                    "trigger_type": {
                        "type": "string",
                        "description": "Trigger type: timer (periodic) or file (watch directory)",
                        "enum": ["timer", "file"],
                    },
                    "trigger_interval": {
                        "type": "number",
                        "description": "Seconds between triggers (timer mode, default 60)",
                    },
                    "trigger_watch_dir": {
                        "type": "string",
                        "description": "Directory to watch for new files (file mode)",
                    },
                },
                "required": ["task", "trigger_type"],
            },
            tier="default",
            safety_level=2,
        )

    def _tool_send_message(self, to_id: str, content: str, msg_type: str = "info") -> str:
        """Send a message to another agent."""
        try:
            msg_id = self._channel.send(self._agent_id, to_id, content, msg_type)
            return f"Message sent (id: {msg_id})"
        except KeyError:
            return f"Agent '{to_id}' not found."

    def _tool_check_messages(self) -> str:
        """Check and return pending messages."""
        msgs = self._channel.receive(self._agent_id)
        if not msgs:
            return "No pending messages."
        lines = [f"[{m.from_id} -> {m.msg_type}]: {m.content}" for m in msgs]
        return "\n".join(lines)

    def _tool_list_agents(self) -> str:
        """List all registered agents."""
        agents = self._registry.list_agents()
        if not agents:
            return "No other agents registered."
        lines = [f"- {a['agent_id']} ({a['role']}, {a['mode']})" for a in agents]
        return "\n".join(lines)

    def _spawn_continuous_child(self, task: str, role: str = "worker",
                               trigger_type: str = "timer",
                               trigger_interval: float = 60,
                               trigger_watch_dir: str | None = None) -> str:
        """Spawn a long-running continuous child agent driven by triggers."""
        # Runtime guard: only continuous mode parents
        if self.agent.mode != "continuous":
            return "Only continuous mode agents can spawn continuous children."

        # Runner guard: continuous children need thread runner (inline would block forever)
        if self._runner_name == "inline":
            return "Continuous children require thread or process runner (set child_agent_runner='thread')."

        policy = copy.copy(ROLE_POLICIES.get(role, AgentExecutionPolicy()))

        # Apply config-level model override for this role
        model_override = self._role_model_overrides.get(role)
        if model_override:
            policy.model = model_override

        spec = ChildAgentSpec(
            task=task,
            role=role,
            context="",
            policy=policy,
            parent_task_id=self._parent_id,
            continuous=True,
            trigger_type=trigger_type,
            trigger_interval=trigger_interval,
            trigger_watch_dir=trigger_watch_dir,
        )
        try:
            task_id = self.controller.spawn_child(spec, self._create_child_agent)
        except RuntimeError as e:
            return f"Cannot spawn continuous child agent: {e}"

        return (
            f"Spawned continuous child agent [task_id: {task_id}] with role={role}, "
            f"trigger={trigger_type}. Use send_message to communicate with it."
        )

    def on_context(self, query: str, context: str) -> str:
        """Inject parallel child agent guide when runner is async (thread/process)."""
        if self._runner_name in ("thread", "process"):
            if context:
                return context + "\n\n" + CHILD_AGENT_GUIDE_ASYNC
            return CHILD_AGENT_GUIDE_ASYNC
        return context

    def on_shutdown(self):
        """Shutdown the runner backend, cleanup messaging, and isolated child roots."""
        if self.controller:
            self.controller.runner.shutdown()
        # Unregister from messaging infrastructure
        if self._registry and self._agent_id:
            self._registry.unregister(self._agent_id)
        self._cleanup_child_roots()

    def _cleanup_child_roots(self):
        """Remove isolated project_dirs for completed children (share_parent_project_dir=False)."""
        import os
        children_dir = os.path.join(self.agent.playground_dir, "children")
        if os.path.isdir(children_dir):
            import shutil
            try:
                shutil.rmtree(children_dir)
            except OSError as e:
                logger.warning("Failed to cleanup child root directories: %s", e)

    # ============================================================
    # Tool implementations
    # ============================================================

    def _spawn_child(self, task: str, role: str = "worker", context: str = "") -> str:
        """Spawn a child agent for the given task."""
        policy = copy.copy(ROLE_POLICIES.get(role, AgentExecutionPolicy()))

        # Apply config-level model override for this role
        model_override = self._role_model_overrides.get(role)
        if model_override:
            policy.model = model_override

        spec = ChildAgentSpec(
            task=task,
            role=role,
            context=context,
            policy=policy,
            parent_task_id=self._parent_id,
        )
        try:
            task_id = self.controller.spawn_child(spec, self._create_child_agent)
        except RuntimeError as e:
            return f"Cannot spawn child agent: {e}"

        # Check if already completed (inline runner, or very fast thread execution)
        record = self.controller.task_board.get(task_id)
        if record and record.status in ("completed", "failed"):
            return record.result or f"Child agent ({role}) completed with no output."
        else:
            # Thread runner: return task_id for async collection
            return (
                f"Spawned child agent [task_id: {task_id}] with role={role}. "
                f"Use wait_child(task_id=\"{task_id}\") or collect_results() to get results."
            )

    def _wait_child(self, task_id: str, include_history: bool = False,
                    include_logs: bool = False) -> str:
        """Wait for a specific child agent and return its result."""
        record = self.controller.task_board.get(task_id)
        if record is None:
            return f"No child agent found with task_id '{task_id}'."

        if record.status == "running":
            # Use controller.wait_child which has fallback to runner
            record = self.controller.wait_child(task_id, timeout=300)

        if record is None:
            return f"Child agent '{task_id}' completed but result not yet available."

        result = record.result or "(no output)"
        if include_history and record.history:
            history_text = "\n".join(
                f"[{m.get('role', '?')}]: {(m.get('content') or '')[:2000]}"
                for m in record.history
            )
            result = f"Result: {result}\n\nFull history:\n{history_text}"
        if include_logs and record.logs:
            result += f"\n\nChild logs:\n{record.logs[:2000]}"
        # Include child_root path for isolated children (share_parent_project_dir=False)
        if record.input_snapshot and record.input_snapshot.get("child_root"):
            import os
            child_root = os.path.join(
                self.agent.playground_dir, record.input_snapshot["child_root"]
            )
            result += f"\n\nChild root: {child_root}"
        return result

    def _list_children(self) -> str:
        """List all spawned child agents and their status."""
        children = self.controller.list_children(self._parent_id)
        if not children:
            return "No child agents spawned yet."
        lines = []
        for c in children:
            task_preview = c.task[:50] + "..." if len(c.task) > 50 else c.task
            lines.append(f"[{c.status}] {c.role}: {task_preview}")
        return "\n".join(lines)

    def _collect_results(self, wait: bool = False, timeout: float = 300) -> str:
        """Collect results from all completed child agents."""
        timed_out = 0
        if wait:
            # Wait for all running children to complete (with timeout)
            deadline = time.time() + timeout
            children = self.controller.list_children(self._parent_id)
            for child in children:
                if child.status == "running":
                    remaining = max(0, deadline - time.time())
                    # Use controller.wait_child which syncs runner result to board
                    self.controller.wait_child(child.task_id, timeout=remaining)
                    # Check if it actually completed
                    refreshed = self.controller.task_board.get(child.task_id)
                    if refreshed and refreshed.status == "running":
                        timed_out += 1

        results = self.controller.collect_results(self._parent_id)
        if not results and timed_out == 0:
            return "No completed child agent results."
        lines = []
        if timed_out > 0:
            lines.append(f"({timed_out} child agent(s) timed out and are still running)")
        for r in results:
            task_preview = r.task[:50] + "..." if len(r.task) > 50 else r.task
            result_preview = r.result[:200] if r.result else "(no output)"
            cost_parts = []
            if r.metrics.get("llm_calls"):
                cost_parts.append(f"{r.metrics['llm_calls']} LLM calls")
            if r.metrics.get("tokens_used"):
                cost_parts.append(f"{r.metrics['tokens_used']} tokens")
            if r.metrics.get("elapsed_seconds"):
                cost_parts.append(f"{r.metrics['elapsed_seconds']}s")
            cost_line = f"\nCost: {', '.join(cost_parts)}" if cost_parts else ""
            lines.append(f"[{r.role}] {task_preview}\nResult: {result_preview}{cost_line}")
        return "\n\n".join(lines)

    # ============================================================
    # Child agent factory
    # ============================================================

    def _create_child_agent(self, spec: ChildAgentSpec):
        """
        Factory: create a constrained LlamAgent for child execution.

        For short-task children: inherits parent's LLM and selected tools,
        operates with minimal config.

        For continuous children: created in interactive mode first (clean state),
        then switched to continuous mode with proper scope setup.
        """
        if spec.continuous:
            return self._create_continuous_child_agent(spec)
        return self._create_short_child_agent(spec)

    def _create_short_child_agent(self, spec: ChildAgentSpec):
        """Factory for short-lived child agents (existing logic)."""
        from llamagent.core.agent import LlamAgent, SimpleReAct
        from llamagent.core.config import Config

        parent = self.agent

        # Build config by shallow-copying parent's, then overriding child-specific fields.
        # Avoids the fragile __new__ pattern that breaks when Config adds new attributes.
        config = copy.copy(parent.config)
        # Ensure api_retry_count exists (may be missing if parent config was manually constructed)
        if not hasattr(config, 'api_retry_count'):
            config.api_retry_count = 1
        config.system_prompt = (
            spec.system_prompt
            or f"You are a {spec.role}. Complete the assigned task concisely."
        )
        config.context_window_size = 10
        config.context_compress_threshold = 0.7
        config.compress_keep_turns = 2
        config.max_duplicate_actions = 2
        config.max_observation_tokens = 1500
        config.memory_mode = "off"
        config.reflection_write_mode = "off"
        config.reflection_read_mode = "off"
        config.max_plan_adjustments = 3
        config.permission_level = 1

        # Apply budget constraints
        if spec.policy and spec.policy.budget:
            config.max_react_steps = spec.policy.budget.max_steps or 5
            config.react_timeout = spec.policy.budget.max_time_seconds or 60
        else:
            config.max_react_steps = 5
            config.react_timeout = 60

        # Determine base LLM for child (model override or inherited)
        if spec.policy and spec.policy.model:
            base_llm = parent._get_llm(spec.policy.model)
        else:
            base_llm = self.llm  # module LLM (inherits parent or module_models)

        # Wrap with budget tracking if needed
        if spec.policy and spec.policy.budget:
            tracker = BudgetTracker(spec.policy.budget)
            child_llm = BudgetedLLM(base_llm, tracker)
        else:
            child_llm = base_llm

        # Create child agent via normal constructor, then replace internals
        child = LlamAgent(config)
        child.llm = child_llm
        child.persona = None
        child.modules = {}
        child.history = []
        child.summary = None
        child.conversation = child.history
        child._execution_strategy = SimpleReAct()
        # FS isolation: when share_parent_project_dir=False, child gets an
        # isolated project_dir under the parent's playground; True means
        # the child shares the parent's project_dir + scopes.
        # When policy is None (backward compat), default to True (share).
        import os
        import uuid as _uuid
        share_parent_project_dir = (
            spec.policy.share_parent_project_dir if spec.policy else True
        )
        if share_parent_project_dir:
            # Trusted role or no policy: full project access
            child.project_dir = parent.project_dir
            child.playground_dir = parent.playground_dir
        else:
            # Isolated: child gets its own project_dir under parent's playground
            child_task_id = spec.task_id or _uuid.uuid4().hex[:12]
            child_root = os.path.join(
                parent.playground_dir, "children", child_task_id
            )
            os.makedirs(child_root, exist_ok=True)
            child.project_dir = child_root
            child.playground_dir = os.path.join(child_root, "llama_playground")
            os.makedirs(child.playground_dir, exist_ok=True)
        child.confirm_handler = parent.confirm_handler
        child.interaction_handler = getattr(parent, "interaction_handler", None)
        child.mode = "interactive"  # Short-task children always use interactive mode

        # v2.7: scope inheritance — share_parent inherits parent scopes,
        # isolated mode gets empty scopes (project writes denied)
        if share_parent_project_dir:
            parent_scopes = parent._authorization_engine.export_scopes()
            if parent_scopes:
                child._authorization_engine.import_scopes(parent_scopes)
        child._tools = {}
        child._tools_version = 0
        child.tool_executor = getattr(parent, "tool_executor", None)  # Inherit sandbox

        # Filter tools by allowlist/denylist (deep copy to prevent parent mutation)
        if spec.policy and spec.policy.tool_allowlist is not None:
            for tool_name in spec.policy.tool_allowlist:
                if tool_name in parent._tools:
                    child._tools[tool_name] = copy.deepcopy(parent._tools[tool_name])
        else:
            child._tools = copy.deepcopy(parent._tools)

        if spec.policy and spec.policy.tool_denylist:
            for tool_name in spec.policy.tool_denylist:
                child._tools.pop(tool_name, None)

        # Don't give child the spawn_child tool (unless can_spawn_children)
        if spec.policy and not spec.policy.can_spawn_children:
            child._tools.pop("spawn_child", None)
            child._tools.pop("list_children", None)
            child._tools.pop("collect_results", None)
            child._tools.pop("wait_child", None)

        # Apply role-level execution_policy to child's tools (e.g., coder -> POLICY_SANDBOXED_CODER)
        if spec.policy and spec.policy.execution_policy is not None:
            for tool_name, tool in child._tools.items():
                if tool.get("execution_policy") is None:
                    tool["execution_policy"] = spec.policy.execution_policy

        return child

    def _create_continuous_child_agent(self, spec: ChildAgentSpec):
        """
        Factory for continuous child agents.

        Creation sequence:
        1. Build config with authorization_mode="interactive" (clean construction)
        2. Create LlamAgent (interactive mode, clean state)
        3. Set project_dir/playground_dir
        4. Set confirm_handler = None (no user interaction)
        5. Call set_mode("continuous") — creates default scope with correct project_dir
        6. Override config values that set_mode's _MODE_DEFAULTS may have set
        7. Import parent scopes (layered on top of default scope)
        8. Inject task into system_prompt
        9. Register to AgentRegistry + register messaging tools
        """
        from llamagent.core.agent import LlamAgent, SimpleReAct
        import os

        parent = self.agent

        # 1. Build config
        config = copy.copy(parent.config)
        if not hasattr(config, 'api_retry_count'):
            config.api_retry_count = 1
        config.authorization_mode = "interactive"  # Clean construction
        config.memory_mode = "off"
        config.reflection_write_mode = "off"
        config.reflection_read_mode = "off"
        config.max_plan_adjustments = 3
        config.permission_level = 1
        config.context_compress_threshold = 0.7
        config.compress_keep_turns = 2
        config.max_duplicate_actions = 2
        config.max_observation_tokens = 1500

        # Apply budget constraints for initial config
        if spec.policy and spec.policy.budget:
            config.max_react_steps = spec.policy.budget.max_steps or 10
            config.react_timeout = spec.policy.budget.max_time_seconds or 600
        else:
            config.max_react_steps = 10
            config.react_timeout = 600

        # Determine base LLM
        if spec.policy and spec.policy.model:
            base_llm = parent._get_llm(spec.policy.model)
        else:
            base_llm = self.llm

        # Wrap with budget tracking if needed
        if spec.policy and spec.policy.budget:
            tracker = BudgetTracker(spec.policy.budget)
            child_llm = BudgetedLLM(base_llm, tracker)
        else:
            child_llm = base_llm

        # 2. Create child agent in interactive mode (clean state)
        child = LlamAgent(config)
        child.llm = child_llm
        child.persona = None
        child.modules = {}
        child.history = []
        child.summary = None
        child.conversation = child.history
        child._execution_strategy = SimpleReAct()

        # 3. Set project_dir/playground_dir
        share_parent_project_dir = (
            spec.policy.share_parent_project_dir if spec.policy else True
        )
        if share_parent_project_dir:
            child.project_dir = parent.project_dir
            child.playground_dir = parent.playground_dir
        else:
            import uuid as _uuid
            child_task_id = spec.task_id or _uuid.uuid4().hex[:12]
            child_root = os.path.join(
                parent.playground_dir, "children", child_task_id
            )
            os.makedirs(child_root, exist_ok=True)
            child.project_dir = child_root
            child.playground_dir = os.path.join(child_root, "llama_playground")
            os.makedirs(child.playground_dir, exist_ok=True)

        # 4. No user interaction for continuous children
        child.confirm_handler = None
        child.interaction_handler = None

        # 5. Switch to continuous mode (creates default scope with correct project_dir)
        child.set_mode("continuous")

        # 6. Override _MODE_DEFAULTS side effects with reasonable child values
        # set_mode("continuous") applies _MODE_DEFAULTS which may set unlimited values.
        # Restore child-appropriate values (budget may have specified different values).
        budget_timeout = spec.policy.budget.max_time_seconds if spec.policy and spec.policy.budget else 600
        budget_steps = spec.policy.budget.max_steps if spec.policy and spec.policy.budget else 10
        config.max_react_steps = budget_steps
        config.react_timeout = budget_timeout
        config.max_duplicate_actions = 2
        config.max_observation_tokens = 1500
        config.context_window_size = 20

        # 7. Import parent scopes (only when sharing the parent's project,
        #    consistent with short child)
        if share_parent_project_dir:
            parent_scopes = parent._authorization_engine.export_scopes()
            if parent_scopes:
                child._authorization_engine.import_scopes(parent_scopes)

        # 8. Inject task into system_prompt
        config.system_prompt = (
            f"You are a {spec.role}. Your ongoing task: {spec.task}\n"
            f"You will receive trigger events. Process them according to your task."
        )

        # Set up tools
        child._tools = {}
        child._tools_version = 0
        child.tool_executor = getattr(parent, "tool_executor", None)

        # Filter tools by allowlist/denylist
        if spec.policy and spec.policy.tool_allowlist is not None:
            for tool_name in spec.policy.tool_allowlist:
                if tool_name in parent._tools:
                    child._tools[tool_name] = copy.deepcopy(parent._tools[tool_name])
        else:
            child._tools = copy.deepcopy(parent._tools)

        if spec.policy and spec.policy.tool_denylist:
            for tool_name in spec.policy.tool_denylist:
                child._tools.pop(tool_name, None)

        # Remove parent-only tools from child
        if spec.policy and not spec.policy.can_spawn_children:
            child._tools.pop("spawn_child", None)
            child._tools.pop("spawn_continuous_child", None)
            child._tools.pop("list_children", None)
            child._tools.pop("collect_results", None)
            child._tools.pop("wait_child", None)
        # Also remove parent's messaging tools (child gets its own below)
        child._tools.pop("send_message", None)
        child._tools.pop("check_messages", None)
        child._tools.pop("list_agents", None)

        # Apply role-level execution_policy
        if spec.policy and spec.policy.execution_policy is not None:
            for tool_name, tool in child._tools.items():
                if tool.get("execution_policy") is None:
                    tool["execution_policy"] = spec.policy.execution_policy

        # 9. Register to AgentRegistry and register messaging tools on child
        if self._registry:
            self._registry.register(child.agent_id, role=spec.role, mode="continuous")
        self._register_child_messaging_tools(child)

        return child

    def _register_child_messaging_tools(self, child):
        """Register messaging tools on a continuous child agent with its own agent_id."""
        child_id = child.agent_id
        channel = self._channel
        registry = self._registry

        def send(to_id: str, content: str, msg_type: str = "info") -> str:
            try:
                msg_id = channel.send(child_id, to_id, content, msg_type)
                return f"Message sent (id: {msg_id})"
            except KeyError:
                return f"Agent '{to_id}' not found."

        def check() -> str:
            msgs = channel.receive(child_id)
            if not msgs:
                return "No pending messages."
            return "\n".join(f"[{m.from_id} -> {m.msg_type}]: {m.content}" for m in msgs)

        def list_all() -> str:
            agents = registry.list_agents()
            if not agents:
                return "No agents registered."
            return "\n".join(f"- {a['agent_id']} ({a['role']}, {a['mode']})" for a in agents)

        child.register_tool(
            name="send_message", func=send,
            description="Send a message to another agent",
            parameters={
                "type": "object",
                "properties": {
                    "to_id": {"type": "string", "description": "Target agent ID"},
                    "content": {"type": "string", "description": "Message content"},
                    "msg_type": {"type": "string", "description": "Message type: info/alert/request/response"},
                },
                "required": ["to_id", "content"],
            },
            tier="default", safety_level=1,
        )

        child.register_tool(
            name="check_messages", func=check,
            description="Check for pending messages from other agents",
            parameters={"type": "object", "properties": {}, "required": []},
            tier="default", safety_level=1,
        )

        child.register_tool(
            name="list_agents", func=list_all,
            description="List all active agents",
            parameters={"type": "object", "properties": {}, "required": []},
            tier="default", safety_level=1,
        )

    # ============================================================
    # Memory callbacks
    # ============================================================

    def _on_module_child_complete(self, task_id: str, record: TaskRecord):
        """Module-level callback: auto-memorize child results, unregister continuous agents."""
        if self._auto_memorize and record.status == "completed" and record.result:
            self._save_child_result_to_memory(record.task, record.role, record.result)

        # Unregister continuous child agent from AgentRegistry
        agent_id = record.metrics.get("agent_id") if record.metrics else None
        if self._registry and agent_id:
            try:
                self._registry.unregister(agent_id)
            except Exception:
                pass

    def _save_child_result_to_memory(self, task: str, role: str, result: str):
        """If parent has memory module, save child result as a memory fact."""
        if not self.agent.has_module("memory"):
            return
        memory_mod = self.agent.get_module("memory")
        if not hasattr(memory_mod, 'remember'):
            return
        content = f"Child agent ({role}) completed task: {task[:100]}. Result: {result[:200]}"
        try:
            memory_mod.remember(content, category="child_agent_result")
        except Exception:
            pass  # Memory save failure should not affect main flow
