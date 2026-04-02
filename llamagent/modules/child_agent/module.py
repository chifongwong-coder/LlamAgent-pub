"""
ChildAgentModule: spawn and control child agents with budget and capability boundaries.

Registers three tools for the parent agent:
- spawn_child:      Spawn a child agent for a subtask with role-based constraints
- list_children:    List all spawned child agents and their status
- collect_results:  Collect results from all completed child agents

The module uses an InlineRunnerBackend by default (synchronous, in-process).
Each child agent is a constrained SmartAgent with filtered tools, budget limits,
and no ability to spawn its own children (unless explicitly allowed by policy).
"""

from __future__ import annotations

import copy
import logging

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


# ======================================================================
# ChildAgentController
# ======================================================================


class ChildAgentController:
    """
    Controller that coordinates the runner backend and task board.

    Provides a clean API for spawning, waiting, listing, and collecting
    child agent results. The module delegates all orchestration to this class.
    """

    def __init__(self, runner: AgentRunnerBackend, task_board: TaskBoard,
                 max_children: int = 20):
        self.runner = runner
        self.task_board = task_board
        self.max_children = max_children

    def spawn_child(self, spec: ChildAgentSpec, agent_factory) -> str:
        """
        Spawn a child agent and register it on the task board.

        Args:
            spec: Child agent specification.
            agent_factory: Callable(spec) -> SmartAgent.

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

        task_id = self.runner.spawn(spec, agent_factory)

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

        # Sync the runner result to the task board, then clean runner to avoid duplication
        record = self.runner.wait(task_id)
        self.task_board.create(
            task_id=record.task_id,
            parent_id=record.parent_id,
            role=record.role,
            task=record.task,
            status=record.status,
            result=record.result,
            metrics=record.metrics,
            input_snapshot=input_snapshot,
            created_at=record.created_at,
            completed_at=record.completed_at,
        )
        # Clean up runner's copy to prevent unbounded memory growth
        if hasattr(self.runner, '_results'):
            self.runner._results.pop(task_id, None)
        return task_id

    def wait_child(self, task_id: str, timeout: float | None = None) -> TaskRecord:
        """Wait for a child and return its task record from the board."""
        record = self.task_board.get(task_id)
        if record is not None:
            return record
        # Fallback to runner (should not happen with inline backend)
        return self.runner.wait(task_id, timeout)

    def list_children(self, parent_id: str) -> list[TaskRecord]:
        """List all children belonging to a parent."""
        return self.task_board.children_of(parent_id)

    def cancel_child(self, task_id: str) -> bool:
        """Cancel a running child agent."""
        success = self.runner.cancel(task_id)
        if success:
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
    Each child is a constrained SmartAgent instance that inherits the parent's
    LLM and selected tools, but operates under strict resource limits.
    """

    name = "child_agent"
    description = "Spawn and control child agents with budget and capability boundaries"

    def __init__(self):
        self.controller: ChildAgentController | None = None
        self.task_board: TaskBoard | None = None
        self._parent_id: str | None = None  # Scope key for list/collect

    def on_attach(self, agent):
        """Initialize controller, task board, and register tools."""
        super().on_attach(agent)
        self._parent_id = str(id(agent))
        self.task_board = TaskBoard()
        runner = InlineRunnerBackend()
        self.controller = ChildAgentController(runner, self.task_board)

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
                        "description": "Role: researcher/writer/analyst/coder/worker",
                        "enum": ["researcher", "writer", "analyst", "coder", "worker"],
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
            tier="default",
            safety_level=1,
        )

    # ============================================================
    # Tool implementations
    # ============================================================

    def _spawn_child(self, task: str, role: str = "worker", context: str = "") -> str:
        """Spawn a child agent for the given task and return its result."""
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
        record = self.controller.wait_child(task_id)  # Inline: already done
        return record.result or f"Child agent ({role}) completed with no output."

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

    def _collect_results(self) -> str:
        """Collect results from all completed child agents."""
        results = self.controller.collect_results(self._parent_id)
        if not results:
            return "No completed child agent results."
        lines = []
        for r in results:
            task_preview = r.task[:50] + "..." if len(r.task) > 50 else r.task
            result_preview = r.result[:200] if r.result else "(no output)"
            lines.append(f"[{r.role}] {task_preview}\nResult: {result_preview}")
        return "\n\n".join(lines)

    # ============================================================
    # Child agent factory
    # ============================================================

    def _create_child_agent(self, spec: ChildAgentSpec):
        """
        Factory: create a constrained SmartAgent for child execution.

        The child inherits the parent's LLM and selected tools, but operates
        with a minimal configuration: no memory, no reflection, limited steps,
        and tool access filtered by the role policy.
        """
        from llamagent.core.agent import SmartAgent, SimpleReAct
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
        config.reflection_enabled = False
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
            child_llm = BudgetedLLM(parent.llm, tracker)
        else:
            child_llm = parent.llm

        # Create child agent via normal constructor, then replace internals
        child = SmartAgent(config)
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

        # Apply role-level execution_policy to child's tools (e.g., coder -> POLICY_SANDBOXED_CODER)
        if spec.policy and spec.policy.execution_policy is not None:
            for tool_name, tool in child._tools.items():
                if tool.get("execution_policy") is None:
                    tool["execution_policy"] = spec.policy.execution_policy

        return child
