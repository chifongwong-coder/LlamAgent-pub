"""
Execution policies and role specifications for child agents.

AgentExecutionPolicy:  Declares what a child agent is allowed to do (tools, sandbox, budget).
ChildAgentSpec:        Full specification for spawning a single child agent.
ROLE_POLICIES:         Preset policies for common roles (researcher, writer, analyst, coder).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llamagent.modules.child_agent.budget import Budget

# Optional dependency: sandbox module may not be installed
try:
    from llamagent.modules.sandbox.policy import ExecutionPolicy, POLICY_SANDBOXED_CODER
    _SANDBOX_AVAILABLE = True
except ImportError:
    ExecutionPolicy = None  # type: ignore[assignment,misc]
    POLICY_SANDBOXED_CODER = None
    _SANDBOX_AVAILABLE = False


@dataclass
class AgentExecutionPolicy:
    """
    Declares capability and resource boundaries for a child agent.

    Controls which tools the child can access, how it executes code,
    how much budget it can consume, and whether it can spawn its own children.
    """

    tool_allowlist: list[str] | None = None
    tool_denylist: list[str] | None = None
    execution_policy: object | None = None  # ExecutionPolicy when sandbox is available
    budget: Budget | None = None
    can_spawn_children: bool = False
    # v3.5: max_delegation_depth retired from policy — depth cap is now per-agent
    # via config.child_agent_max_delegation_depth (default 2). Per-policy override
    # was never enforced (dead code from v2.6); a future re-introduction should
    # consult both sources.
    history_mode: str = "none"
    result_mode: str = "text"
    # v3.4 R3: replaces the old workspace_mode str enum. Default False
    # means the child gets an isolated project_dir under the parent's
    # playground (was workspace_mode="sandbox"). True means the child
    # shares the parent's project_dir + scopes (was workspace_mode=
    # "project"). Avoids the old name's three concept-name collisions
    # ("sandbox" with SandboxModule, "project" with classify_write's
    # zone, "workspace" itself being retired in v3.4).
    share_parent_project_dir: bool = False
    model: str | None = None  # None = inherit parent's model


@dataclass
class ChildAgentSpec:
    """
    Full specification for spawning a child agent.

    Combines the task description, role identity, policy constraints,
    and optional context/artifacts from the parent agent.
    """

    task: str
    role: str = "worker"
    system_prompt: str = ""
    context: str = ""
    policy: AgentExecutionPolicy | None = None
    parent_task_id: str | None = None
    artifact_refs: list[dict] = field(default_factory=list)
    task_id: str | None = None  # Set by controller before factory call
    runlog_path: str = ""  # v3.5: absolute path of child's JSONL runlog (set by ChildAgentModule)
    delegation_depth: int = 0  # v3.5: 0 = direct child of root parent; +1 per level
    # Continuous child agent fields
    continuous: bool = False
    trigger_type: str | None = None        # "timer" | "file"
    trigger_interval: float = 60           # seconds (timer mode)
    trigger_watch_dir: str | None = None   # path (file mode)


# ---------------------------------------------------------------------------
# Preset role policies
# ---------------------------------------------------------------------------

ROLE_POLICIES: dict[str, AgentExecutionPolicy] = {
    "researcher": AgentExecutionPolicy(
        tool_allowlist=["web_search", "web_fetch", "search_knowledge", "search_text", "read_files"],
        budget=Budget(max_llm_calls=20, max_time_seconds=300),
        can_spawn_children=False,
        share_parent_project_dir=False,
    ),
    "writer": AgentExecutionPolicy(
        tool_allowlist=["read_files", "write_files", "apply_patch"],
        budget=Budget(max_llm_calls=15, max_time_seconds=300),
        can_spawn_children=False,
        share_parent_project_dir=False,
    ),
    "analyst": AgentExecutionPolicy(
        tool_allowlist=["read_files", "search_text", "web_search", "search_knowledge"],
        budget=Budget(max_llm_calls=15, max_time_seconds=300),
        can_spawn_children=False,
        share_parent_project_dir=False,
    ),
    "coder": AgentExecutionPolicy(
        tool_allowlist=["read_files", "write_files", "apply_patch", "start_job", "glob_files", "search_text"],
        execution_policy=POLICY_SANDBOXED_CODER if _SANDBOX_AVAILABLE else None,
        budget=Budget(max_llm_calls=30, max_time_seconds=600),
        can_spawn_children=False,
        share_parent_project_dir=True,
    ),
    "delegate": AgentExecutionPolicy(
        tool_allowlist=[],
        budget=Budget(max_llm_calls=1, max_time_seconds=30),
        can_spawn_children=False,
        share_parent_project_dir=False,
    ),
    "worker": AgentExecutionPolicy(
        share_parent_project_dir=True,
        budget=Budget(max_llm_calls=30, max_time_seconds=600),
        can_spawn_children=False,
    ),
}
