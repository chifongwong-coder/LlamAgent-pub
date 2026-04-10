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
                 max_children: int = 20):
        self.runner = runner
        self.task_board = task_board
        self.max_children = max_children

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

    def on_attach(self, agent):
        """Initialize controller, task board, and register tools."""
        super().on_attach(agent)
        self._parent_id = str(id(agent))
        self.task_board = TaskBoard()

        # Select runner backend from config
        self._runner_name = getattr(agent.config, "child_agent_runner", "inline")
        max_children = getattr(agent.config, "child_agent_max_children", 20)

        if self._runner_name == "thread":
            from llamagent.modules.child_agent.runners.thread import ThreadRunnerBackend
            runner = ThreadRunnerBackend()
        elif self._runner_name == "process":
            from llamagent.modules.child_agent.runners.process import ProcessRunnerBackend
            runner = ProcessRunnerBackend(parent_config=agent.config)
        else:
            runner = InlineRunnerBackend()

        self.controller = ChildAgentController(
            runner, self.task_board, max_children=max_children,
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

    def on_context(self, messages: list[dict], context: str) -> str:
        """Inject parallel child agent guide when runner is async (thread/process)."""
        if self._runner_name in ("thread", "process"):
            if context:
                return context + "\n\n" + CHILD_AGENT_GUIDE_ASYNC
            return CHILD_AGENT_GUIDE_ASYNC
        return context

    def on_shutdown(self):
        """Shutdown the runner backend (relevant for thread runner)."""
        if self.controller:
            self.controller.runner.shutdown()

    # ============================================================
    # Tool implementations
    # ============================================================

    def _spawn_child(self, task: str, role: str = "worker", context: str = "") -> str:
        """Spawn a child agent for the given task."""
        policy = ROLE_POLICIES.get(role, AgentExecutionPolicy())
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

        The child inherits the parent's LLM and selected tools, but operates
        with a minimal configuration: no memory, no reflection, limited steps,
        and tool access filtered by the role policy.
        """
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

        # Build child LLM (budget-wrapped or shared)
        if spec.policy and spec.policy.budget:
            tracker = BudgetTracker(spec.policy.budget)
            child_llm = BudgetedLLM(self.llm, tracker)
        else:
            child_llm = self.llm

        # Create child agent via normal constructor, then replace internals
        child = LlamAgent(config)
        child.llm = child_llm
        child.persona = None
        child.modules = {}
        child.history = []
        child.summary = None
        child.conversation = child.history
        child._execution_strategy = SimpleReAct()
        # Inherit zone system settings from parent (snapshotted paths, not os.getcwd())
        child.project_dir = parent.project_dir
        child.playground_dir = parent.playground_dir
        child.confirm_handler = parent.confirm_handler
        child.interaction_handler = getattr(parent, "interaction_handler", None)
        child.mode = getattr(parent, "mode", "interactive")
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
