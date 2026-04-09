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
    max_delegation_depth: int = 1
    history_mode: str = "none"
    result_mode: str = "text"


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


# ---------------------------------------------------------------------------
# Preset role policies
# ---------------------------------------------------------------------------

ROLE_POLICIES: dict[str, AgentExecutionPolicy] = {
    "researcher": AgentExecutionPolicy(
        tool_allowlist=["web_search", "web_fetch", "search_knowledge", "search_text", "read_files"],
        budget=Budget(max_llm_calls=20, max_time_seconds=300),
        can_spawn_children=False,
    ),
    "writer": AgentExecutionPolicy(
        tool_allowlist=["read_files", "write_files", "apply_patch"],
        budget=Budget(max_llm_calls=15, max_time_seconds=300),
        can_spawn_children=False,
    ),
    "analyst": AgentExecutionPolicy(
        tool_allowlist=["read_files", "search_text", "web_search", "search_knowledge"],
        budget=Budget(max_llm_calls=15, max_time_seconds=300),
        can_spawn_children=False,
    ),
    "coder": AgentExecutionPolicy(
        tool_allowlist=["read_files", "write_files", "apply_patch", "start_job", "glob_files", "search_text"],
        execution_policy=POLICY_SANDBOXED_CODER if _SANDBOX_AVAILABLE else None,
        budget=Budget(max_llm_calls=30, max_time_seconds=600),
        can_spawn_children=False,
    ),
    "delegate": AgentExecutionPolicy(
        tool_allowlist=[],
        budget=Budget(max_llm_calls=1, max_time_seconds=30),
        can_spawn_children=False,
    ),
}
