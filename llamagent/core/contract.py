"""
Task mode contract, state, and shared data types.

TaskContract: pre-execution contract presented to user for confirmation.
TaskModeState: tracks task mode lifecycle across multiple chat() turns.
AuthorizationUpdate: structured authorization change request (controller creates, agent forwards, engine consumes).
PipelineOutcome: structured return from _run_pipeline().
normalize_scopes(): pure data function to aggregate RequestedScope lists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from llamagent.core.zone import RequestedScope


@dataclass
class TaskContract:
    """Pre-execution contract for task mode, presented to user for confirmation."""
    task_summary: str
    planned_operations: list[str]
    requested_scopes: list[RequestedScope]
    open_questions: list[str]
    risk_summary: str


@dataclass
class TaskModeState:
    """
    Tracks task mode lifecycle across multiple chat() turns.

    State flow:
    - idle → preparing (first task mode chat)
    - preparing → awaiting_confirmation (dry-run complete, contract generated)
    - awaiting_confirmation → executing (user confirms)
    - awaiting_confirmation → preparing (user provides more info)
    - awaiting_confirmation → idle (user cancels)
    - executing → idle (execution complete)

    Shared between TaskModeController (writes all fields) and TaskPolicy (reads phase/task_id only).
    pending_scopes are NOT stored here — they live in TaskPolicy._pending_buffer
    and flow explicitly through PipelineOutcome.metadata.
    """
    phase: str = "idle"
    task_id: str = ""              # Generated on idle→preparing transition
    original_query: str = ""
    contract: TaskContract | None = None
    confirmed: bool = False
    clarification_turns: int = 0   # v1.9.6: tracks re-prepare count


@dataclass
class AuthorizationUpdate:
    """
    Structured authorization change request.

    Created by TaskModeController, forwarded by agent (without understanding),
    consumed by AuthorizationEngine.apply_update().
    """
    task_id: str | None = None
    approved_scopes: list[RequestedScope] | None = None
    clear_task_scope: bool = False
    clear_session_scopes: bool = False


@dataclass
class PipelineOutcome:
    """
    Structured return from LlamAgent._run_pipeline().

    metadata is used as an opaque courier: engine deposits data (e.g., pending_scopes
    during prepare), agent forwards without inspecting, controller reads what it needs.
    """
    response: str
    task_id: str | None = None
    blocked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_scopes(scopes: list[RequestedScope]) -> list[RequestedScope]:
    """
    Aggregate pending scopes for a cleaner contract.

    Merges scopes with same zone + action + tool_name, deduplicates paths,
    and attempts to find common path prefixes.
    """
    if not scopes:
        return []

    # Group by (zone, action_tuple, tool_names_tuple)
    groups: dict[tuple, list[str]] = {}
    for s in scopes:
        key = (s.zone, tuple(sorted(s.actions)), tuple(sorted(s.tool_names or [])))
        if key not in groups:
            groups[key] = []
        groups[key].extend(s.path_prefixes)

    result = []
    for (zone, actions, tool_names), paths in groups.items():
        # Deduplicate paths
        unique_paths = sorted(set(paths))

        # Try to find common prefix
        if len(unique_paths) > 1:
            try:
                prefix = os.path.commonpath(unique_paths) if unique_paths else ""
            except ValueError:
                # Mixed absolute/relative paths — fall back to no common prefix
                prefix = ""
            if prefix and prefix != unique_paths[0]:
                unique_paths = [prefix + "/"]

        result.append(RequestedScope(
            zone=zone,
            actions=list(actions),
            path_prefixes=unique_paths,
            tool_names=list(tool_names) if tool_names else None,
        ))

    return result
