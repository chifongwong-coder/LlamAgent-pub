"""
Zone evaluation and authorization request/response types.

ZoneVerdict / ZoneDecisionItem / ZoneEvaluation: pure zone judgment results.
ConfirmRequest / ConfirmResponse: structured confirmation interface.

These types are the foundation of the v1.9 authorization system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ZoneVerdict(str, Enum):
    """Zone judgment result for a single path."""
    ALLOW = "allow"
    CONFIRMABLE = "confirmable"
    HARD_DENY = "hard_deny"


@dataclass
class ZoneDecisionItem:
    """Single path's zone verdict."""
    path: str
    verdict: ZoneVerdict
    zone: str               # "playground" | "project" | "external"
    action: str             # "read" | "write" | "execute"
    message: str | None = None


@dataclass
class ZoneEvaluation:
    """
    Complete zone evaluation for one tool call (may involve multiple paths).

    overall_verdict is derived from items:
    - Any HARD_DENY → overall HARD_DENY
    - Else any CONFIRMABLE → overall CONFIRMABLE
    - Else → overall ALLOW
    """
    items: list[ZoneDecisionItem] = field(default_factory=list)
    overall_verdict: ZoneVerdict = ZoneVerdict.ALLOW

    @staticmethod
    def from_items(items: list[ZoneDecisionItem]) -> ZoneEvaluation:
        """Create a ZoneEvaluation with auto-derived overall_verdict."""
        if not items:
            return ZoneEvaluation(items=[], overall_verdict=ZoneVerdict.ALLOW)

        if any(i.verdict == ZoneVerdict.HARD_DENY for i in items):
            overall = ZoneVerdict.HARD_DENY
        elif any(i.verdict == ZoneVerdict.CONFIRMABLE for i in items):
            overall = ZoneVerdict.CONFIRMABLE
        else:
            overall = ZoneVerdict.ALLOW

        return ZoneEvaluation(items=items, overall_verdict=overall)


@dataclass
class RequestedScope:
    """One authorization scope requested in a task contract."""
    zone: str                     # "project" | "external"
    actions: list[str]            # ["write"] / ["read", "write"] / ["execute"]
    path_prefixes: list[str]      # ["project:src/", "project:docs/"]
    tool_names: list[str] | None = None


@dataclass
class ConfirmRequest:
    """Structured confirmation request passed to confirm_handler."""
    kind: str               # "operation_confirm" | "task_contract"
    tool_name: str
    action: str             # "read" | "write" | "execute"
    zone: str               # "playground" | "project" | "external"
    target_paths: list[str]
    message: str
    mode: str = "interactive"  # Informational field, not for decision logic
    requested_scopes: list[RequestedScope] | None = None  # v1.9.1: contract scopes


@dataclass
class ApprovalScope:
    """One approved authorization scope, stored after contract confirmation."""
    scope: str                    # "task" (v1.9.2)
    zone: str                     # "project" | "external"
    actions: list[str]            # ["write"] / ["execute"]
    path_prefixes: list[str]      # ["project:src/", "project:docs/"]
    tool_names: list[str] | None = None


@dataclass
class ConfirmResponse:
    """Structured confirmation response from confirm_handler."""
    allow: bool
    approved_scopes: list[RequestedScope] | None = None  # v1.9.2: user-narrowed scopes
