"""
Authorization engine: unified authorization decision layer for call_tool().

Encapsulates path extraction, zone evaluation, and policy-based decisions.
In v1.9.0, only InteractivePolicy is implemented (same behavior as v1.8.x).
Future versions add TaskPolicy (1.9.2) and ContinuousPolicy (1.9.3).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llamagent.core.contract import TaskModeState
from llamagent.core.zone import (
    ApprovalScope,
    ConfirmRequest,
    ConfirmResponse,
    RequestedScope,
    ZoneDecisionItem,
    ZoneEvaluation,
    ZoneVerdict,
)

if TYPE_CHECKING:
    from llamagent.core.agent import SmartAgent

logger = logging.getLogger(__name__)


# ======================================================================
# Action inference
# ======================================================================

# Tools known to be command execution (not just read/write)
_EXECUTE_TOOLS = frozenset({"start_job", "execute_command"})


def infer_action(tool: dict) -> str:
    """
    Infer the action type for a tool.

    Priority: explicit action field > tool name match > safety_level derivation.
    """
    explicit = tool.get("action")
    if explicit:
        return explicit

    if tool.get("name") in _EXECUTE_TOOLS:
        return "execute"

    sl = tool.get("safety_level", 1)
    return "write" if sl >= 2 else "read"


# ======================================================================
# Authorization state (placeholder for future scope accumulation)
# ======================================================================

@dataclass
class AuthorizationState:
    """Authorization state tracking approved scopes per task and session."""
    task_scopes: dict = field(default_factory=dict)    # dict[str, list[ApprovalScope]]
    session_scopes: list = field(default_factory=list)  # list[ApprovalScope]


# ======================================================================
# Authorization policies
# ======================================================================

class AuthorizationPolicy(ABC):
    """Abstract authorization policy. Different modes use different policies."""

    @abstractmethod
    def decide(
        self,
        evaluation: ZoneEvaluation,
        tool_name: str,
        engine: AuthorizationEngine,
    ) -> str | None:
        """
        Decide whether to allow, confirm, or deny based on zone evaluation.

        Returns None to allow, or rejection string to block.
        """
        ...


class InteractivePolicy(AuthorizationPolicy):
    """
    v1.9.0 interactive mode: same behavior as v1.8.x.

    Iterates ZoneEvaluation items, confirms CONFIRMABLE one by one.
    Any single denial → entire operation denied.
    """

    def decide(
        self,
        evaluation: ZoneEvaluation,
        tool_name: str,
        engine: AuthorizationEngine,
    ) -> str | None:
        # Fast path: all allowed
        if evaluation.overall_verdict == ZoneVerdict.ALLOW:
            return None

        for item in evaluation.items:
            if item.verdict == ZoneVerdict.ALLOW:
                continue

            if item.verdict == ZoneVerdict.HARD_DENY:
                return (
                    item.message
                    or f"Tool '{tool_name}' cannot operate on '{item.path}'."
                )

            # CONFIRMABLE — ask user for this specific path
            request = ConfirmRequest(
                kind="operation_confirm",
                tool_name=tool_name,
                action=item.action,
                zone=item.zone,
                target_paths=[item.path],
                message=(
                    item.message
                    or f"Tool '{tool_name}' wants to operate on '{item.path}'."
                ),
            )
            response = engine.confirm(request)
            if not response.allow:
                return f"Tool '{tool_name}' operation on '{item.path}' was denied."

        return None  # All items allowed


class TaskPolicy(AuthorizationPolicy):
    """
    v1.9.1 task mode: controlled dry-run during prepare, interactive during execute.

    Prepare phase: read operations execute normally, write/execute operations
    are blocked and recorded as pending scopes for the task contract.
    Execute phase: falls back to InteractivePolicy (1.9.2 will add scope matching).
    """

    def __init__(self, state: TaskModeState):
        self.state = state

    def decide(
        self,
        evaluation: ZoneEvaluation,
        tool_name: str,
        engine: AuthorizationEngine,
    ) -> str | None:
        if self.state.phase == "preparing":
            return self._decide_prepare(evaluation, tool_name, engine)
        if self.state.phase == "executing":
            return self._decide_execute(evaluation, tool_name, engine)
        return None  # idle / awaiting_confirmation: no tool calls expected

    def _decide_prepare(
        self, evaluation: ZoneEvaluation, tool_name: str, engine: AuthorizationEngine
    ) -> str | None:
        # ask_user always allowed during prepare (need user info)
        if tool_name == "ask_user":
            return None

        # First pass: check for HARD_DENY (must reject before recording anything)
        for item in evaluation.items:
            if item.verdict == ZoneVerdict.HARD_DENY:
                return item.message or f"Tool '{tool_name}' blocked."

        # Second pass: record write/execute as pending scopes
        recorded = []
        for item in evaluation.items:
            if item.action in ("write", "execute"):
                self._record_pending(item, tool_name)
                recorded.append(item.path)

        if recorded:
            paths_str = ", ".join(recorded)
            return (
                f"[Prepare] '{tool_name}' on {paths_str} recorded. "
                f"Will need authorization before execution."
            )

        # All read/allow — proceed
        return None

    def _decide_execute(
        self, evaluation: ZoneEvaluation, tool_name: str, engine: AuthorizationEngine
    ) -> str | None:
        """Execute phase: match against approved task scopes, fall back to confirm."""
        task_id = engine.agent.get_active_task_id()
        scopes = engine.state.task_scopes.get(task_id, []) if task_id else []

        for item in evaluation.items:
            if item.verdict == ZoneVerdict.HARD_DENY:
                return item.message or f"Tool '{tool_name}' blocked."
            if item.verdict == ZoneVerdict.ALLOW:
                continue
            # CONFIRMABLE — try scope match
            if _matches_any_scope(item, tool_name, scopes):
                continue  # Auto-approved by scope
            # No match — fall back to per-item confirm
            request = ConfirmRequest(
                kind="operation_confirm",
                tool_name=tool_name,
                action=item.action,
                zone=item.zone,
                target_paths=[item.path],
                message=f"Tool '{tool_name}' on '{item.path}' is outside approved scope.",
                mode="task",
            )
            response = engine.confirm(request)
            if not response.allow:
                return f"Tool '{tool_name}' operation on '{item.path}' was denied."
        return None

    def _record_pending(self, item: ZoneDecisionItem, tool_name: str):
        scope = RequestedScope(
            zone=item.zone,
            actions=[item.action],
            path_prefixes=[item.path],
            tool_names=[tool_name],
        )
        self.state.pending_scopes.append(scope)


class ContinuousPolicy(AuthorizationPolicy):
    """
    v1.9.3 continuous mode: no interaction, scope-based authorization only.

    ALLOW → execute. CONFIRMABLE → match against session scopes or deny.
    HARD_DENY → deny. Never calls confirm_handler.
    """

    def decide(
        self,
        evaluation: ZoneEvaluation,
        tool_name: str,
        engine: AuthorizationEngine,
    ) -> str | None:
        scopes = engine.state.session_scopes

        for item in evaluation.items:
            if item.verdict == ZoneVerdict.HARD_DENY:
                return item.message or f"Tool '{tool_name}' blocked."
            if item.verdict == ZoneVerdict.ALLOW:
                continue
            # CONFIRMABLE — match against session scopes
            if _matches_any_scope(item, tool_name, scopes):
                continue  # Auto-approved by seed scope
            # No match, no interaction — deny
            return (
                f"Tool '{tool_name}' on '{item.path}' requires authorization "
                f"not covered by seed scopes. Operation denied."
            )
        return None


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
            prefix = os.path.commonpath(unique_paths) if unique_paths else ""
            if prefix and prefix != unique_paths[0]:
                unique_paths = [prefix + "/"]

        result.append(RequestedScope(
            zone=zone,
            actions=list(actions),
            path_prefixes=unique_paths,
            tool_names=list(tool_names) if tool_names else None,
        ))

    return result


def _matches_any_scope(
    item: ZoneDecisionItem, tool_name: str, scopes: list[ApprovalScope]
) -> bool:
    """Check if a zone decision item is covered by any approved scope."""
    for scope in scopes:
        if item.zone != scope.zone:
            continue
        if item.action not in scope.actions:
            continue
        if scope.tool_names and tool_name not in scope.tool_names:
            continue
        if not any(item.path.startswith(prefix) for prefix in scope.path_prefixes):
            continue
        return True
    return False


# ======================================================================
# Authorization engine
# ======================================================================

class AuthorizationEngine:
    """
    Unified authorization engine for call_tool().

    Encapsulates:
    - Path extraction (moved from SmartAgent._extract_paths)
    - Zone evaluation (moved from SmartAgent._check_zone)
    - Policy-based authorization decision

    The engine accesses agent attributes (project_dir, playground_dir,
    confirm_handler) lazily via self.agent reference.
    """

    def __init__(self, agent: SmartAgent):
        self.agent = agent
        self.state = AuthorizationState()
        self.policy: AuthorizationPolicy = InteractivePolicy()

    def set_mode(self, mode: str):
        """Switch authorization mode and policy."""
        if mode == "task":
            state = TaskModeState()
            self.policy = TaskPolicy(state)
            self.agent._task_mode_state = state
        elif mode == "interactive":
            self.policy = InteractivePolicy()
            self.agent._task_mode_state = None
            self.state.task_scopes.clear()
            self.state.session_scopes.clear()
        elif mode == "continuous":
            self.policy = ContinuousPolicy()
            self.agent._task_mode_state = None
            self.state.task_scopes.clear()
            self._load_seed_scopes()

    def _load_seed_scopes(self):
        """Load seed scopes from config into session_scopes."""
        raw = getattr(self.agent.config, "seed_scopes", None)
        if not raw or not isinstance(raw, list):
            self.state.session_scopes = []
            return
        scopes = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            scopes.append(ApprovalScope(
                scope=item.get("scope", "session"),
                zone=item.get("zone", "project"),
                actions=item.get("actions", []),
                path_prefixes=item.get("path_prefixes", []),
                tool_names=item.get("tool_names"),
            ))
        self.state.session_scopes = scopes
        logger.info("Loaded %d seed scopes for continuous mode", len(scopes))

    def evaluate(self, tool: dict, args: dict) -> str | None:
        """
        Complete authorization flow: path extraction → zone evaluation → policy decision.

        Returns None to allow, or rejection string to block.
        """
        paths = self._extract_paths(tool, args)
        evaluation = self._evaluate_zone(tool, paths)
        return self.policy.decide(evaluation, tool.get("name", "unknown"), self)

    def confirm(self, request: ConfirmRequest) -> ConfirmResponse:
        """Delegate confirmation to agent._ask_confirmation."""
        return self.agent._ask_confirmation(request)

    # ------------------------------------------------------------------
    # Path extraction (moved from SmartAgent._extract_paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_paths(tool: dict, args: dict) -> list[str]:
        """Extract paths from tool arguments using path_extractor or auto-detection."""
        extractor = tool.get("path_extractor")
        if extractor is not None:
            try:
                return [p for p in extractor(args) if p]
            except Exception:
                return []
        # Auto path extractor fallback
        PATH_KEYWORDS = {"path", "file", "filepath"}
        return [
            v for k, v in args.items()
            if any(kw in k.lower() for kw in PATH_KEYWORDS)
            and isinstance(v, str)
        ]

    # ------------------------------------------------------------------
    # Zone evaluation (moved from SmartAgent._check_zone)
    # ------------------------------------------------------------------

    def _evaluate_zone(self, tool: dict, paths: list[str]) -> ZoneEvaluation:
        """
        Evaluate zone for each path. Pure judgment — no side effects.

        Produces ZoneDecisionItem per path, then derives overall_verdict.
        """
        if not paths:
            return ZoneEvaluation.from_items([])

        action = infer_action(tool)
        project = self.agent.project_dir
        playground = self.agent.playground_dir
        tool_name = tool.get("name", "unknown")

        items: list[ZoneDecisionItem] = []

        for raw_path in paths:
            resolved = (
                os.path.realpath(os.path.join(project, raw_path))
                if not os.path.isabs(raw_path)
                else os.path.realpath(raw_path)
            )

            # Zone 1: Playground — always allow
            if resolved.startswith(playground + os.sep) or resolved == playground:
                items.append(ZoneDecisionItem(
                    path=raw_path, verdict=ZoneVerdict.ALLOW,
                    zone="playground", action=action,
                ))
                continue

            # Zone 2: Project directory
            if resolved.startswith(project + os.sep) or resolved == project:
                if action == "read":
                    items.append(ZoneDecisionItem(
                        path=raw_path, verdict=ZoneVerdict.ALLOW,
                        zone="project", action=action,
                    ))
                else:
                    items.append(ZoneDecisionItem(
                        path=raw_path, verdict=ZoneVerdict.CONFIRMABLE,
                        zone="project", action=action,
                        message=f"Tool '{tool_name}' wants to operate on '{raw_path}' in project directory (has side effects)",
                    ))
                continue

            # Zone 3: External
            if action == "read":
                items.append(ZoneDecisionItem(
                    path=raw_path, verdict=ZoneVerdict.CONFIRMABLE,
                    zone="external", action=action,
                    message=f"Tool '{tool_name}' wants to read '{raw_path}' (outside project directory)",
                ))
            else:
                items.append(ZoneDecisionItem(
                    path=raw_path, verdict=ZoneVerdict.HARD_DENY,
                    zone="external", action=action,
                    message=f"Tool '{tool_name}' cannot operate on '{raw_path}'. Path is outside project directory, side-effect operations are not allowed.",
                ))

        return ZoneEvaluation.from_items(items)
