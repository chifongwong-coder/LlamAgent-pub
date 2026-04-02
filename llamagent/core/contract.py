"""
Task mode contract and state types.

TaskContract: pre-execution contract presented to user for confirmation.
TaskModeState: tracks task mode lifecycle across multiple chat() turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    """
    phase: str = "idle"
    original_query: str = ""
    pending_scopes: list = field(default_factory=list)  # list[RequestedScope]
    contract: TaskContract | None = None
    confirmed: bool = False
