"""
Authorization engine: unified authorization decision layer for call_tool().

Encapsulates path extraction, zone evaluation, and policy-based decisions.
In v1.9.0, only InteractivePolicy is implemented (same behavior as v1.8.x).
Future versions add TaskPolicy (1.9.2) and ContinuousPolicy (1.9.3).
"""

from __future__ import annotations

import logging
import os
import time
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
    from llamagent.core.agent import LlamAgent

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


@dataclass
class AuthorizationResult:
    """Result from AuthorizationEngine.evaluate(). Carries decision + audit events."""
    decision: str | None = None  # None = allow, str = rejection message
    events: list = field(default_factory=list)  # list[tuple[str, dict]]


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
    ) -> AuthorizationResult:
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
    ) -> AuthorizationResult:
        if evaluation.overall_verdict == ZoneVerdict.ALLOW:
            return AuthorizationResult()

        for item in evaluation.items:
            if item.verdict == ZoneVerdict.ALLOW:
                continue
            if item.verdict == ZoneVerdict.HARD_DENY:
                return AuthorizationResult(
                    decision=item.message or f"Tool '{tool_name}' cannot operate on '{item.path}'."
                )
            request = ConfirmRequest(
                kind="operation_confirm", tool_name=tool_name,
                action=item.action, zone=item.zone,
                target_paths=[item.path],
                message=item.message or f"Tool '{tool_name}' wants to operate on '{item.path}'.",
            )
            response = engine.confirm(request)
            if not response.allow:
                return AuthorizationResult(
                    decision=f"Tool '{tool_name}' operation on '{item.path}' was denied."
                )

        return AuthorizationResult()


class TaskPolicy(AuthorizationPolicy):
    """
    v1.9.1/1.9.2 task mode: controlled dry-run during prepare, scope matching during execute.
    """

    def __init__(self, state: TaskModeState):
        self.state = state

    def decide(
        self,
        evaluation: ZoneEvaluation,
        tool_name: str,
        engine: AuthorizationEngine,
    ) -> AuthorizationResult:
        if self.state.phase == "preparing":
            return self._decide_prepare(evaluation, tool_name, engine)
        if self.state.phase == "executing":
            return self._decide_execute(evaluation, tool_name, engine)
        return AuthorizationResult()

    def _decide_prepare(
        self, evaluation: ZoneEvaluation, tool_name: str, engine: AuthorizationEngine
    ) -> AuthorizationResult:
        if tool_name == "ask_user":
            return AuthorizationResult()

        for item in evaluation.items:
            if item.verdict == ZoneVerdict.HARD_DENY:
                return AuthorizationResult(
                    decision=item.message or f"Tool '{tool_name}' blocked."
                )

        recorded = []
        for item in evaluation.items:
            if item.action in ("write", "execute"):
                self._record_pending(item, tool_name)
                recorded.append(item.path)

        if recorded:
            paths_str = ", ".join(recorded)
            return AuthorizationResult(
                decision=f"[Prepare] '{tool_name}' on {paths_str} recorded. "
                         f"Will need authorization before execution."
            )

        return AuthorizationResult()

    def _decide_execute(
        self, evaluation: ZoneEvaluation, tool_name: str, engine: AuthorizationEngine
    ) -> AuthorizationResult:
        """Execute phase: two-stage — match first, consume after all pass."""
        task_id = engine.agent.get_active_task_id()
        scopes = engine.state.task_scopes.get(task_id, []) if task_id else []
        events = []
        matched = []

        for item in evaluation.items:
            if item.verdict == ZoneVerdict.HARD_DENY:
                return AuthorizationResult(
                    decision=item.message or f"Tool '{tool_name}' blocked."
                )
            if item.verdict == ZoneVerdict.ALLOW:
                continue
            scope = _find_matching_scope(item, tool_name, scopes)
            if scope is not None:
                matched.append((item, scope))
            else:
                events.append(("scope_denied", {
                    "tool_name": tool_name, "path": item.path, "zone": item.zone,
                }))
                # Fall back to confirm
                request = ConfirmRequest(
                    kind="operation_confirm", tool_name=tool_name,
                    action=item.action, zone=item.zone,
                    target_paths=[item.path],
                    message=f"Tool '{tool_name}' on '{item.path}' is outside approved scope.",
                    mode="task",
                )
                response = engine.confirm(request)
                if not response.allow:
                    return AuthorizationResult(
                        decision=f"Tool '{tool_name}' operation on '{item.path}' was denied.",
                        events=events,
                    )

        # All passed — consume matched scopes
        for item, scope in matched:
            scope.uses += 1
            events.append(("scope_used", {
                "scope": scope, "tool_name": tool_name, "path": item.path,
            }))

        return AuthorizationResult(events=events)

    def _record_pending(self, item: ZoneDecisionItem, tool_name: str):
        scope = RequestedScope(
            zone=item.zone, actions=[item.action],
            path_prefixes=[item.path], tool_names=[tool_name],
        )
        self.state.pending_scopes.append(scope)


class ContinuousPolicy(AuthorizationPolicy):
    """
    v1.9.3 continuous mode: no interaction, scope-based authorization only.
    Two-stage: match first, consume after all pass.
    """

    def decide(
        self,
        evaluation: ZoneEvaluation,
        tool_name: str,
        engine: AuthorizationEngine,
    ) -> AuthorizationResult:
        scopes = engine.state.session_scopes
        events = []
        matched = []

        for item in evaluation.items:
            if item.verdict == ZoneVerdict.HARD_DENY:
                return AuthorizationResult(
                    decision=item.message or f"Tool '{tool_name}' blocked."
                )
            if item.verdict == ZoneVerdict.ALLOW:
                continue
            scope = _find_matching_scope(item, tool_name, scopes)
            if scope is not None:
                matched.append((item, scope))
            else:
                events.append(("scope_denied", {
                    "tool_name": tool_name, "path": item.path, "zone": item.zone,
                }))
                return AuthorizationResult(
                    decision=f"Tool '{tool_name}' on '{item.path}' requires authorization "
                             f"not covered by seed scopes. Operation denied.",
                    events=events,
                )

        # All passed — consume
        for item, scope in matched:
            scope.uses += 1
            events.append(("scope_used", {
                "scope": scope, "tool_name": tool_name, "path": item.path,
            }))

        return AuthorizationResult(events=events)


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


def _find_matching_scope(
    item: ZoneDecisionItem, tool_name: str, scopes: list[ApprovalScope]
) -> ApprovalScope | None:
    """
    Find the first approved scope that covers this item. Pure function — no side effects.

    Checks expiry, usage limits, zone, action, tool_names, and path (normalized).
    """
    now = time.time()
    for scope in scopes:
        # Expiry check
        if scope.expires_at is not None and now > scope.expires_at:
            continue
        # Usage limit check
        if scope.max_uses is not None and scope.uses >= scope.max_uses:
            continue
        # Zone + action + tool_names match
        if item.zone != scope.zone:
            continue
        if item.action not in scope.actions:
            continue
        if scope.tool_names and tool_name not in scope.tool_names:
            continue
        # Path match: normalized subtree check (not simple startswith)
        if not _path_in_prefixes(item.path, scope.path_prefixes):
            continue
        return scope
    return None


def _path_in_prefixes(path: str, prefixes: list[str]) -> bool:
    """Check if path is within any prefix subtree using normalized path comparison."""
    norm_path = os.path.normpath(path)
    for prefix in prefixes:
        norm_prefix = os.path.normpath(prefix)
        # Exact match or proper subtree (path starts with prefix + separator)
        if norm_path == norm_prefix or norm_path.startswith(norm_prefix + os.sep):
            return True
    return False


# ======================================================================
# Authorization engine
# ======================================================================

class AuthorizationEngine:
    """
    Unified authorization engine for call_tool().

    Encapsulates:
    - Path extraction (moved from LlamAgent._extract_paths)
    - Zone evaluation (moved from LlamAgent._check_zone)
    - Policy-based authorization decision

    The engine accesses agent attributes (project_dir, playground_dir,
    confirm_handler) lazily via self.agent reference.
    """

    def __init__(self, agent: LlamAgent):
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
                created_at=time.time(),
                source="seed",
            ))
        self.state.session_scopes = scopes
        logger.info("Loaded %d seed scopes for continuous mode", len(scopes))

    def evaluate(self, tool: dict, args: dict) -> AuthorizationResult:
        """
        Complete authorization flow: path extraction → zone evaluation → policy decision.

        Returns AuthorizationResult with decision (None=allow, str=rejection) and events.
        """
        paths = self._extract_paths(tool, args)
        evaluation = self._evaluate_zone(tool, paths)
        return self.policy.decide(evaluation, tool.get("name", "unknown"), self)

    def confirm(self, request: ConfirmRequest) -> ConfirmResponse:
        """Delegate confirmation to agent._ask_confirmation."""
        return self.agent._ask_confirmation(request)

    # ------------------------------------------------------------------
    # Path extraction (moved from LlamAgent._extract_paths)
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
    # Zone evaluation (moved from LlamAgent._check_zone)
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
