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
import os
import time

from llamagent.core.agent import Module

# v3.5: Child completion report convention (template B — system prompt).
# Combined with the framework auto-prompt (template A, commit-6), the goal
# is for record.result to be a structured summary the parent's model can
# read consistently. Framework does NOT parse this output — record.result
# stays a free-form string.
CHILD_SYSTEM_PROMPT = (
    "You are a {role}. Complete the assigned task concisely.\n\n"
    "When your task is complete, your final reply MUST follow this format:\n"
    "Status: success | partial | failed\n"
    "Summary: <1-3 sentences describing what you did>\n"
    "Artifacts: <comma-separated absolute paths of files you created or modified, or \"none\">"
)
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


def _check_share_modules(parent, share_modules, runner_name=None):
    """v3.7: validate ``share_parent_modules`` against parent state.

    Returns a clean error string suitable for tool-result output, or
    ``None`` if validation passes. Used at two call sites:

    1. **Spawn-tool pre-check** (``_spawn_child`` /
       ``_spawn_continuous_child``) — before ``controller.spawn_child``,
       so the model receives a clean error string instead of the
       runner's exception-swallowing path producing a misleading
       "Spawned child agent.\\nResult: ...execution error: ValueError"
       (which falsely reads as success). Pass ``runner_name`` so the
       process-runner refusal (B2) fires here.
    2. **Factory-side defense** (``_apply_shared_modules``) — as a
       sanity check that callers raise on. Defends ``_create_child_agent``
       direct invocations (e.g. tests) that bypass the spawn tool. Pass
       ``runner_name=None`` to skip the runner check (the factory does
       not have direct access to the runner backend; the spawn-tool
       layer already validated for the production path).

    Args:
        parent: parent agent (provides ``modules`` registry).
        share_modules: ``policy.share_parent_modules`` value
            (``list[str] | None``). ``None`` and ``[]`` are both valid
            and return ``None`` (default no-share path).
        runner_name: when ``"process"`` and ``share_modules`` is non-
            empty, returns a clean error. The process runner cannot
            alias an in-process Python store handle; v3.7 plan §6.2
            explicitly leaves cross-process sharing out of scope.
            Pass ``None`` to skip the runner check.

    Returns:
        ``str | None``: error message or ``None`` on valid.
    """
    if not share_modules:
        # Default path is always valid.
        return None

    if runner_name == "process":
        return (
            "share_parent_modules is not supported on the process "
            "runner (the subprocess cannot alias an in-process store "
            "handle). Use child_agent_runner='inline' or 'thread', "
            "or remove share_parent_modules from the policy."
        )

    for mod_name in share_modules:
        parent_mod = parent.modules.get(mod_name)
        if parent_mod is None:
            return (
                f"share_parent_modules={mod_name!r}: parent has no "
                f"such module loaded. Load it on the parent before "
                f"spawning the child."
            )
        if not getattr(type(parent_mod), "shareable", False):
            return (
                f"Module {mod_name!r} (type {type(parent_mod).__name__}) "
                f"is not declared shareable; cannot pass via "
                f"share_parent_modules."
            )

    return None


def _apply_shared_modules(parent, child, share_modules):
    """v3.7: re-register shareable parent modules on the child and
    swap in the parent's storage handles.

    Called by ``_create_child_agent`` (both branches of ``spec.continuous``)
    AFTER ``child._tools`` has been rebuilt (deepcopy from parent + filter
    by allowlist/denylist) and BEFORE the child runs any tool. Mutates
    ``child.config`` and ``child.modules`` in place.

    The ordering matters: ``child.modules = {}`` earlier in the factory
    wiped any module __init__ created. Here we re-register a fresh
    module instance so ``on_attach`` runs against the child's agent /
    LLM binding (preserves child's BudgetedLLM / model selection),
    then swap in the parent's storage via ``inherit_storage_from`` so
    the child reads the parent's persistent data.

    Read-only contract: for each shareable module, this helper forces
    the child's write mode to "off" before re-registering. The
    module's ``on_attach`` checks the mode and skips write-tool
    registration entirely — there is no ``save_memory`` /
    ``consolidate_memory`` on the shared child. The child does inherit
    parent's read mode (``parent.config.memory_recall_mode``) so a
    parent that disabled reads keeps the child equally locked out.

    Default branch (``share_modules`` is None / empty): forces
    ``memory_mode = "off"`` on the child config AND strips any memory
    tool entries that landed in ``child._tools`` via the
    deepcopy(parent._tools) step earlier in the factory. Pre-v3.7,
    those deepcopied entries were closure-bound to parent's
    MemoryModule instance, so a child of a memory-loaded parent could
    silently invoke ``recall_memory`` / ``list_memories`` and reach
    back into parent's store -- bypassing the config-level lockout.
    v3.7 closes that leak: the no-share default is now genuinely
    isolated, not just config-disabled.

    Errors:
    - parent has no module named X -> ValueError (loud failure;
      typically a typo or wrong module load order).
    - parent module's class is not declared shareable -> ValueError.
    - parent module is loaded but disabled (e.g. memory_mode="off"
      and store=None) -> silent graceful: child also ends up disabled,
      matching parent's intent.
    """
    # Memory tools that may have landed in child._tools via the
    # deepcopy(parent._tools) step earlier in the factory. We
    # always start by clearing them, then conditionally let
    # MemoryModule's on_attach (called below) re-register the right
    # subset for the share case. Without this clear, parent's read
    # AND write tools leak into the child's tool table even when the
    # caller asked for no sharing — which would break the read-only
    # contract for the share case (parent's save_memory closure stays
    # registered) and let a no-share child indirectly read parent's
    # memory via the deepcopy'd tool entries.
    #
    # v3.7 commit-7: pull the tool-name list from MemoryModule._TOOL_NAMES
    # so helper + tests share a single source of truth — adding a new
    # memory tool only requires updating the class attribute, not two
    # parallel hardcoded tuples.
    from llamagent.modules.memory.module import MemoryModule
    for tool_name in MemoryModule._TOOL_NAMES:
        child._tools.pop(tool_name, None)

    if not share_modules:
        # Default path: child has no persistent modules + the tool-clearing
        # above genuinely isolates the child's tool table. Forces
        # ``memory_mode = "off"`` and ``memory_recall_mode = "off"`` so a
        # default child cannot reach into parent's persistent memory.
        child.config.memory_mode = "off"
        child.config.memory_recall_mode = "off"
        return

    # v3.7.1: validate via shared helper. Production callers go through
    # the spawn-tool pre-check (which produces a clean tool-result
    # string for the model); this raise here defends the
    # ``_create_child_agent`` direct path (used by tests).
    err = _check_share_modules(parent, share_modules)
    if err:
        raise ValueError(err)

    for mod_name in share_modules:
        parent_mod = parent.modules[mod_name]  # _check_share_modules confirmed presence

        # Module-specific config overrides + child module construction.
        # Each shareable module gets its own branch — keeps the read-
        # only forcing logic explicit instead of magic.
        if mod_name == "memory":
            # MemoryModule already imported at the top of this helper
            # for the unconditional tool-clear loop -- no need to
            # re-import here.
            #
            # Read-only contract: child never writes to shared memory.
            child.config.memory_mode = "off"
            # Inherit parent's read mode (parent off -> child off).
            child.config.memory_recall_mode = getattr(
                parent.config, "memory_recall_mode", "off"
            )
            child.register_module(MemoryModule())
        else:
            # Any future shareable module needs its own ``elif`` branch above.
            # Reaching this point means a module class was declared shareable
            # but the factory was not taught how to register it. Fail loudly.
            raise ValueError(
                f"Module {mod_name!r} is declared shareable but the "
                f"child_agent factory does not know how to register "
                f"it on the child. Add a branch in _apply_shared_modules."
            )

        # Swap in parent's storage handles. inherit_storage_from is
        # the contract that decides which fields are "data layer"
        # (copied) vs "LLM-bound" (kept per-agent).
        child.modules[mod_name].inherit_storage_from(parent_mod)


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
        # v3.5: respect pre-set spec.task_id (caller may have already allocated it
        # so it can compute downstream paths like runlog_path).
        if spec.task_id:
            task_id = spec.task_id
        else:
            import uuid
            task_id = uuid.uuid4().hex[:12]
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
        """Cancel a running child agent and all its descendants.

        v3.5: cancellation cascades — if the target child has spawned
        grandchildren, they are killed before the target itself. Mirrors
        Hermes' _active_children walk: depth-first so nothing escapes.
        """
        # Walk descendants depth-first
        descendants = self._collect_descendants(task_id)
        for desc_id in descendants:
            try:
                self.runner.cancel(desc_id)
                rec = self.task_board.get(desc_id)
                if rec and rec.status == "running":
                    self.task_board.update(desc_id, status="cancelled")
            except Exception:
                pass  # best-effort

        success = self.runner.cancel(task_id)
        if success:
            # Only overwrite status if still running (child may have completed before abort took effect)
            existing = self.task_board.get(task_id)
            if existing and existing.status == "running":
                self.task_board.update(task_id, status="cancelled")
        return success

    def _collect_descendants(self, task_id: str) -> list[str]:
        """Return [grandchild_id, ggc_id, ...] in depth-first order."""
        out = []
        stack = [task_id]
        while stack:
            current = stack.pop()
            for child in self.task_board.children_of(current):
                out.append(child.task_id)
                stack.append(child.task_id)
        return out

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
            description=(
                "Send a message to another agent. Pass the recipient's agent_id "
                "(from list_agents) or the task_id of a child you spawned."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "to_id": {
                        "type": "string",
                        "description": "Recipient agent_id, or task_id of a spawned child",
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
        """Send a message to another agent. v3.5: to_id may be an agent_id
        OR a task_id of a child this agent spawned."""
        # v3.5: if to_id matches a known task_id of a continuous child,
        # resolve to that child's agent_id.
        resolved_to_id = self._resolve_send_target(to_id)
        try:
            msg_id = self._channel.send(self._agent_id, resolved_to_id, content, msg_type)
            return f"Message sent (id: {msg_id})"
        except KeyError:
            return f"Agent '{to_id}' not found."

    def _resolve_send_target(self, target: str) -> str:
        """Resolve a send_message target. If target is a known task_id of a
        continuous child, return that child's agent_id; else return target
        unchanged (caller's KeyError handler will report on miss)."""
        try:
            record = self.controller.task_board.get(target)
        except Exception:
            return target
        if record is None:
            return target
        agent_id = (record.metrics or {}).get("agent_id")
        if isinstance(agent_id, str) and agent_id:
            return agent_id
        return target

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
        """Spawn a long-running continuous child agent driven by triggers.

        Continuous-mode-only entry point. Delegates the shared spawn body
        (policy build, depth check, share_parent_modules pre-check,
        spec construction, controller.spawn_child) to ``_spawn_impl``.
        """
        # Runtime guard: only continuous mode parents
        if self.agent.mode != "continuous":
            return "Only continuous mode agents can spawn continuous children."
        # Runner guard: continuous children need thread runner (inline would block forever)
        if self._runner_name == "inline":
            return "Continuous children require thread or process runner (set child_agent_runner='thread')."
        return self._spawn_impl(
            task=task,
            role=role,
            context="",
            continuous=True,
            trigger_type=trigger_type,
            trigger_interval=trigger_interval,
            trigger_watch_dir=trigger_watch_dir,
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
        """Spawn a short-lived child agent for the given task.

        Public tool entry point. Delegates the shared spawn body to
        ``_spawn_impl``.
        """
        return self._spawn_impl(
            task=task, role=role, context=context, continuous=False,
        )

    def _spawn_impl(self, *, task: str, role: str, context: str, continuous: bool,
                    trigger_type: str | None = None, trigger_interval: float = 60,
                    trigger_watch_dir: str | None = None) -> str:
        """Shared spawn body for ``_spawn_child`` / ``_spawn_continuous_child``.

        Owns: role-policy build + model override, max_delegation_depth check,
        ``_check_share_modules`` pre-check, ``ChildAgentSpec`` construction,
        ``_ensure_task_id`` + ``runlog_path`` pre-allocation, and the
        ``controller.spawn_child`` invocation with its RuntimeError catch.
        Mode-specific guards (``agent.mode``, runner name) live in the public
        wrapper; mode-specific return formatting is the tail of this method.

        v3.7.1: pre-validates share_parent_modules BEFORE controller.spawn_child.
        Without this, the runners' ``except Exception`` block wraps the
        helper's ValueError into TaskRecord(status="failed") and the model
        sees a misleading "Spawned child agent.\\n...Result: ...execution
        error: ValueError..." (false success header). The catch below is
        RuntimeError-only — v3.7 commit-8's (RuntimeError, ValueError)
        broaden was dead code for the share path and reverted in v3.7.1.
        """
        label = "continuous child agent" if continuous else "child agent"
        short_label = "continuous child" if continuous else "child"

        # v3.7.3: deepcopy (was shallow ``copy.copy``). Shallow copy shared
        # ``tool_allowlist`` / ``tool_denylist`` / ``budget`` / ``share_parent_modules``
        # references with the global preset; today only ``policy.model``
        # is mutated below (scalar) but any future ``policy.budget.max_steps = X``
        # or ``policy.tool_allowlist.append(...)`` would silently mutate
        # the global preset for all subsequent agents.
        policy = copy.deepcopy(ROLE_POLICIES.get(role, AgentExecutionPolicy()))
        model_override = self._role_model_overrides.get(role)
        if model_override:
            policy.model = model_override

        # v3.5: enforce max_delegation_depth before spawn. Default cap = 2
        # (Hermes-style). v3.5.1: continuous children also enforce this
        # symmetrically — currently moot because the factory prunes spawn
        # tools, but a future relaxation would otherwise bypass the cap.
        max_depth = getattr(self.agent.config, "child_agent_max_delegation_depth", 2)
        my_depth = getattr(self.agent, "_delegation_depth", 0)
        if my_depth + 1 > max_depth:
            return (
                f"Cannot spawn {short_label}: max_delegation_depth ({max_depth}) "
                f"exceeded. Current depth: {my_depth}. Restructure your task to be flatter."
            )

        err = _check_share_modules(
            self.agent, policy.share_parent_modules, self._runner_name
        )
        if err:
            return f"Cannot spawn {label}: {err}"

        spec = ChildAgentSpec(
            task=task,
            role=role,
            context=context,
            policy=policy,
            parent_task_id=self._parent_id,
            delegation_depth=my_depth + 1,
            continuous=continuous,
            trigger_type=trigger_type,
            trigger_interval=trigger_interval,
            trigger_watch_dir=trigger_watch_dir,
        )
        # v3.5 B2: pre-allocate task_id + runlog_path so the runner's finally
        # block can write the "end" record even after a crash.
        self._ensure_task_id(spec)
        spec.runlog_path = self._runlog_path_for(spec.task_id)
        try:
            task_id = self.controller.spawn_child(spec, self._create_child_agent)
        except RuntimeError as e:
            logger.exception("Cannot spawn %s", label)
            return f"Cannot spawn {label}: {e}"

        if continuous:
            return (
                f"Spawned continuous child agent [task_id: {task_id}] with role={role}, "
                f"trigger={trigger_type}. Use send_message to communicate with it."
            )

        # Short-child success path: emit structured return with child_dir for
        # cross-agent path resolution. share_parent_project_dir=True → child_dir
        # is parent.project_dir; False → parent.playground/children/<task_id>/.
        # ``policy`` is never None (constructed via ROLE_POLICIES.get default).
        if policy.share_parent_project_dir:
            child_dir = self.agent.project_dir
        else:
            child_dir = os.path.join(self.agent.playground_dir, "children", task_id)
        header = (
            f"Spawned child agent.\n"
            f"- task_id: {task_id}\n"
            f"- role: {role}\n"
            f"- child_dir: {child_dir}\n"
            f"- Note: if the child reports relative artifact paths, resolve them against child_dir.\n"
        )

        # Inline runner / very fast thread execution: child may already be done.
        record = self.controller.task_board.get(task_id)
        if record and record.status in ("completed", "failed"):
            body = record.result or f"Child agent ({role}) completed with no output."
            return f"{header}\nResult: {body}"
        return (
            f"{header}\n"
            f"Use wait_child(task_id=\"{task_id}\") or collect_results() to get results."
        )

    def _wait_child(self, task_id: str) -> str:
        """Wait for a specific child agent and return its result.

        v3.5: removed include_history / include_logs parameters (hard break,
        no shim). They were anti-patterns: leaking child internal state into
        parent's context. Parent gets the child's completion summary in
        record.result, plus the child_dir hint for resolving relative paths.
        For debugging, child runs are written to a runlog file (commit-4)
        which is for external observation only — not exposed to parent agent.
        """
        record = self.controller.task_board.get(task_id)
        if record is None:
            return f"No child agent found with task_id '{task_id}'."

        if record.status == "running":
            # Use controller.wait_child which has fallback to runner
            record = self.controller.wait_child(task_id, timeout=300)

        if record is None:
            return f"Child agent '{task_id}' completed but result not yet available."

        result = record.result or "(no output)"
        # Include child_dir hint so parent can resolve relative artifact paths.
        if record.input_snapshot and record.input_snapshot.get("child_root"):
            child_dir = os.path.join(
                self.agent.playground_dir, record.input_snapshot["child_root"]
            )
        else:
            child_dir = self.agent.project_dir
        result += (
            f"\n\n(Resolve relative artifact paths against {child_dir})"
        )
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
            # v3.5: no [:200] truncation — record.result is the child's summary,
            # bounded by design. ReAct's _truncate_observation handles overflow
            # at the framework layer if the aggregate gets too long.
            body = r.result if r.result else "(no output)"
            cost_parts = []
            if r.metrics.get("llm_calls"):
                cost_parts.append(f"{r.metrics['llm_calls']} LLM calls")
            if r.metrics.get("tokens_used"):
                cost_parts.append(f"{r.metrics['tokens_used']} tokens")
            if r.metrics.get("elapsed_seconds"):
                cost_parts.append(f"{r.metrics['elapsed_seconds']}s")
            cost_line = f"\nCost: {', '.join(cost_parts)}" if cost_parts else ""
            lines.append(f"[{r.role}] {task_preview}\nResult: {body}{cost_line}")
        return "\n\n".join(lines)

    # ============================================================
    # v3.5: Runlog (observability-only, not parent-visible)
    # ============================================================

    @staticmethod
    def _ensure_task_id(spec) -> str:
        """Ensure ``spec.task_id`` is set; allocate one when missing.

        The controller honours pre-set IDs (see ``ChildAgentController.spawn_child``),
        so callers that need to compute downstream paths (runlog, child_root)
        before invoking the controller can pre-bind via this helper. Returns
        the resulting task_id.
        """
        if not spec.task_id:
            import uuid as _uuid
            spec.task_id = _uuid.uuid4().hex[:12]
        return spec.task_id

    def _runlog_path_for(self, task_id: str) -> str:
        """Return the absolute runlog path for a child task_id.

        Lives under parent.playground_dir/child_runlogs/<task_id>.log so
        parent and child (even cross-process) can both physically reach it.
        """
        return os.path.join(
            self.agent.playground_dir, "child_runlogs", f"{task_id}.log"
        )

    def _attach_runlog(self, child, runlog_path: str) -> None:
        """Wrap child's LLM with LoggingLLM and register POST_TOOL_USE hook.

        Both writers are append-only to the same JSONL file. The hook closure
        captures runlog_path; LoggingLLM is a thin proxy.
        """
        from llamagent.core.logging_llm import LoggingLLM, append_runlog
        from llamagent.core.hooks import HookEvent, HookResult

        max_bytes = getattr(
            self.agent.config, "child_agent_runlog_max_bytes", 10 * 1024 * 1024
        )
        # Wrap the child's LLM (may already be BudgetedLLM-wrapped from earlier)
        child.llm = LoggingLLM(child.llm, runlog_path, max_bytes=max_bytes)

        def _runlog_tool_writer(ctx):
            data = ctx.data
            try:
                append_runlog(
                    runlog_path,
                    {
                        "ts": time.time(),
                        "kind": "tool",
                        "name": data.get("tool_name"),
                        "args_preview": str(data.get("args", ""))[:500],
                        "result_preview": str(data.get("result", ""))[:500],
                    },
                    max_bytes=max_bytes,
                )
            except Exception as e:
                logger.warning("runlog tool-side write failed: %s", e)
            return HookResult.CONTINUE

        try:
            child.register_hook(HookEvent.POST_TOOL_USE, _runlog_tool_writer)
        except Exception as e:
            logger.warning("failed to register runlog POST_TOOL_USE hook: %s", e)

    # ============================================================
    # Child agent factory
    # ============================================================

    def _create_child_agent(self, spec: ChildAgentSpec):
        """Factory: create a constrained LlamAgent for child execution.

        Branches on ``spec.continuous`` only at the points that genuinely
        differ. Short children are constructed directly in interactive
        mode; continuous children are constructed in interactive mode for
        a clean scope, then ``set_mode("continuous")`` re-establishes the
        continuous default scope (with the child's correct project_dir),
        followed by a re-application of budget / duplicate / observation
        / window overrides that ``_MODE_DEFAULTS`` would otherwise clobber.
        Continuous children additionally register to ``AgentRegistry``,
        publish their ``agent_id`` to the task_board, and bind their own
        messaging tools.
        """
        from llamagent.core.agent import LlamAgent, SimpleReAct
        import os

        parent = self.agent
        is_continuous = spec.continuous

        # Default budgets differ by mode (continuous gets a longer ceiling).
        default_steps = 10 if is_continuous else 5
        default_timeout = 600 if is_continuous else 60

        # ---- 1. Build config ----
        config = copy.copy(parent.config)
        if not hasattr(config, "api_retry_count"):
            config.api_retry_count = 1
        if is_continuous:
            # Force clean construction; set_mode below establishes the continuous scope.
            config.authorization_mode = "interactive"

        config.context_compress_threshold = 0.7
        config.compress_keep_turns = 2
        config.max_duplicate_actions = 2
        config.max_observation_tokens = 1500
        # v3.7: memory_mode / memory_recall_mode are set by _apply_shared_modules
        # below — default branch forces "off" for the no-share case, share branch
        # forces write-off + read-inherits-parent.
        config.reflection_write_mode = "off"
        config.reflection_read_mode = "off"
        config.max_plan_adjustments = 3
        config.permission_level = 1

        if not is_continuous:
            # Short-child window; continuous-child window is set after set_mode below.
            config.context_window_size = 10
            # v3.5: report_template gates the completion-report convention.
            #   "system_prompt" (default): inject CHILD_SYSTEM_PROMPT (template B)
            #   "auto"                  : same default + framework auto-prompt at end of run (template A)
            #   "off"                   : no completion-report shaping
            report_template = getattr(
                parent.config, "child_agent_report_template", "system_prompt"
            )
            if spec.system_prompt:
                config.system_prompt = spec.system_prompt
            elif report_template in ("system_prompt", "auto"):
                config.system_prompt = CHILD_SYSTEM_PROMPT.format(role=spec.role)
            else:
                config.system_prompt = (
                    f"You are a {spec.role}. Complete the assigned task concisely."
                )

        if spec.policy and spec.policy.budget:
            config.max_react_steps = spec.policy.budget.max_steps or default_steps
            config.react_timeout = spec.policy.budget.max_time_seconds or default_timeout
        else:
            config.max_react_steps = default_steps
            config.react_timeout = default_timeout

        # ---- 2. Determine base LLM (with optional budget tracking) ----
        if spec.policy and spec.policy.model:
            base_llm = parent._get_llm(spec.policy.model)
        else:
            base_llm = self.llm  # module LLM (inherits parent or module_models)

        if spec.policy and spec.policy.budget:
            tracker = BudgetTracker(spec.policy.budget)
            child_llm = BudgetedLLM(base_llm, tracker)
        else:
            child_llm = base_llm

        # ---- 3. Construct child + reset internals ----
        child = LlamAgent(config)
        child.llm = child_llm
        child.persona = None
        child.modules = {}
        child.history = []
        child.summary = None
        child.conversation = child.history
        child._execution_strategy = SimpleReAct()

        # ---- 4. project_dir / playground_dir ----
        # share_parent_project_dir: production paths always set policy via
        # ROLE_POLICIES; this fallback handles direct test/external callers.
        # Note: divergence with runners/process.py (which uses ``else False``
        # for cross-process safety) is intentional — both fallbacks are dead
        # in production, surfacing only in test/external paths.
        share_parent_project_dir = (
            spec.policy.share_parent_project_dir if spec.policy else True
        )
        if share_parent_project_dir:
            child.project_dir = parent.project_dir
            child.playground_dir = parent.playground_dir
        else:
            child_task_id = self._ensure_task_id(spec)
            child_root = os.path.join(parent.playground_dir, "children", child_task_id)
            os.makedirs(child_root, exist_ok=True)
            child.project_dir = child_root
            child.playground_dir = os.path.join(child_root, "llama_playground")
            os.makedirs(child.playground_dir, exist_ok=True)

        # ---- 5. Mode setup ----
        if is_continuous:
            child.confirm_handler = None
            child.interaction_handler = None
            # set_mode("continuous") applies _MODE_DEFAULTS which may overwrite
            # earlier budget / duplicate / observation values. Re-apply them after.
            # v3.5.3: must use ``or DEFAULT`` (not ``if cond else DEFAULT``) — when
            # spec.policy.budget exists but max_steps / max_time_seconds is None,
            # a bare conditional returns None, which makes run_react's
            # ``steps < self.config.max_react_steps`` raise TypeError.
            child.set_mode("continuous")
            if spec.policy and spec.policy.budget:
                budget_steps = spec.policy.budget.max_steps or default_steps
                budget_timeout = spec.policy.budget.max_time_seconds or default_timeout
            else:
                budget_steps = default_steps
                budget_timeout = default_timeout
            config.max_react_steps = budget_steps
            config.react_timeout = budget_timeout
            config.max_duplicate_actions = 2
            config.max_observation_tokens = 1500
            config.context_window_size = 20
            # Inject the ongoing-task system_prompt AFTER set_mode so the template is final.
            config.system_prompt = (
                f"You are a {spec.role}. Your ongoing task: {spec.task}\n"
                f"You will receive trigger events. Process them according to your task."
            )
        else:
            child.confirm_handler = parent.confirm_handler
            child.interaction_handler = getattr(parent, "interaction_handler", None)
            child.mode = "interactive"

        # ---- 6. Authorization scope ----
        # share_parent inherits parent scopes; isolated mode re-seeds the
        # auto_approve auto-scope against the child's NEW project_dir
        # (LlamAgent.__init__ ran on the pre-overwrite path). Gated by
        # auto_approve so the interactive contract is preserved.
        if share_parent_project_dir:
            parent_scopes = parent._authorization_engine.export_scopes()
            if parent_scopes:
                child._authorization_engine.import_scopes(parent_scopes)
        else:
            child._seed_auto_approve_scope()

        # ---- 7. Tools: deepcopy from parent, filter, prune ----
        child._tools = {}
        child._tools_version = 0
        child.tool_executor = getattr(parent, "tool_executor", None)  # Inherit sandbox

        if spec.policy and spec.policy.tool_allowlist is not None:
            for tool_name in spec.policy.tool_allowlist:
                if tool_name in parent._tools:
                    child._tools[tool_name] = copy.deepcopy(parent._tools[tool_name])
        else:
            child._tools = copy.deepcopy(parent._tools)

        if spec.policy and spec.policy.tool_denylist:
            for tool_name in spec.policy.tool_denylist:
                child._tools.pop(tool_name, None)

        if spec.policy and not spec.policy.can_spawn_children:
            child._tools.pop("spawn_child", None)
            child._tools.pop("list_children", None)
            child._tools.pop("collect_results", None)
            child._tools.pop("wait_child", None)
            if is_continuous:
                child._tools.pop("spawn_continuous_child", None)

        if is_continuous:
            # Strip parent's messaging tools — child gets its own bound to its agent_id.
            child._tools.pop("send_message", None)
            child._tools.pop("check_messages", None)
            child._tools.pop("list_agents", None)

        if spec.policy and spec.policy.execution_policy is not None:
            for tool in child._tools.values():
                if tool.get("execution_policy") is None:
                    tool["execution_policy"] = spec.policy.execution_policy

        # ---- 8. Re-register shareable parent modules ----
        # Default branch (no share_modules) forces memory_mode="off" AND strips
        # parent's deepcopied memory tool entries from child._tools (closes a
        # pre-v3.7 silent leak via parent-bound closures).
        share_modules = spec.policy.share_parent_modules if spec.policy else None
        _apply_shared_modules(parent, child, share_modules)

        # ---- 9. Delegation depth propagation ----
        # Allows a recursive ChildAgentModule to enforce max_delegation_depth on
        # grandchildren. For continuous children today this is moot (spawn tools
        # are pruned in step 7), but a future relaxation would still hit the cap.
        child._delegation_depth = spec.delegation_depth

        # ---- 10. Continuous-only registration + messaging ----
        if is_continuous:
            if self._registry:
                self._registry.register(
                    child.agent_id, role=spec.role, mode="continuous"
                )
            # v3.5 B1: write task_board agent_id metric IMMEDIATELY after
            # registry.register so a later setup-step failure (messaging /
            # runlog) still leaves the metric present — the on-complete
            # callback uses it to unregister the registry entry, avoiding
            # a registry leak on factory crash.
            if spec.task_id:
                try:
                    self.controller.task_board.update(
                        spec.task_id, metrics={"agent_id": child.agent_id}
                    )
                except KeyError:
                    pass  # Board entry not yet created (shouldn't happen)
            self._register_child_messaging_tools(child)

        # ---- 11. Runlog (observability-only, parent-invisible) ----
        if spec.runlog_path:
            self._attach_runlog(child, spec.runlog_path)

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

        # Unregister continuous child agent from AgentRegistry. Source the
        # agent_id from the task_board (single source of truth) rather than
        # the runner's record: the continuous branch of ``_create_child_agent``
        # writes the metric early, so it is present even when the child
        # crashes before the runner had a chance to populate
        # ``record.metrics`` with agent_id (which only happens on a clean
        # ContinuousRunner exit). The board's ``update`` merge semantics
        # preserve the early write through later runner-side metric writes.
        agent_id = record.metrics.get("agent_id") if record.metrics else None
        if not agent_id and self.controller is not None:
            try:
                board_record = self.controller.task_board.get(task_id)
            except Exception:
                board_record = None
            if board_record and board_record.metrics:
                agent_id = board_record.metrics.get("agent_id")
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
