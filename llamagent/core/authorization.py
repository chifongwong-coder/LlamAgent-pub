"""
Authorization engine: unified authorization decision layer for call_tool().

Encapsulates path extraction, zone evaluation, and policy-based decisions.
In v1.9.0, only InteractivePolicy is implemented (same behavior as v1.8.x).
v1.9.2 adds TaskPolicy, v1.9.3 adds ContinuousPolicy.
v1.9.6: ApprovalScope moved here from zone.py; engine.set_mode() replaced by _switch_policy();
         new apply_update() / _clear_all_scopes() / drain_prepare_data() / clear_pending_buffer().
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llamagent.core.contract import AuthorizationUpdate, TaskModeState
from llamagent.core.zone import (
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
# ApprovalScope (moved from zone.py in v1.9.6)
# ======================================================================

@dataclass
class ApprovalScope:
    """One approved authorization scope, stored after contract confirmation."""
    scope: str                    # "task" | "session"
    zone: str                     # "project" | "external"
    actions: list[str]            # ["write"] / ["execute"]
    path_prefixes: list[str]      # ["project:src/", "project:docs/"]
    tool_names: list[str] | None = None
    # v1.9.4 governance fields
    created_at: float | None = None     # time.time() when created
    expires_at: float | None = None     # expiry time (None = no expiry)
    max_uses: int | None = None         # max usage count (None = unlimited)
    uses: int = 0                       # current usage count
    source: str = "contract"            # "contract" | "seed" | "api"


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
# Authorization state
# ======================================================================

@dataclass
class AuthorizationState:
    """Authorization state tracking approved scopes per task and session."""
    task_scopes: dict[str, list[ApprovalScope]] = field(default_factory=dict)
    session_scopes: list[ApprovalScope] = field(default_factory=list)


@dataclass
class AuthorizationResult:
    """Result from AuthorizationEngine.evaluate(). Carries decision + audit events."""
    decision: str | None = None  # None = allow, str = rejection message
    events: list[tuple[str, dict]] = field(default_factory=list)


@dataclass
class AuthorizationUpdateResult:
    """Result from AuthorizationEngine.apply_update() / _clear_all_scopes()."""
    events: list[tuple[str, dict]] = field(default_factory=list)
    changed: bool = False
    has_session_scopes: bool = False


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

    v1.9.6: pending scopes stored in internal _pending_buffer (not in shared TaskModeState).
    Shared state is read-only: policy reads phase/task_id, never writes to state.
    """

    def __init__(self, state: TaskModeState):
        self.state = state  # read-only: phase, task_id
        self._pending_buffer: list[RequestedScope] = []

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
        # v1.9.8: merge task scopes (specific) + session scopes (broad), task-first priority
        task_id = self.state.task_id
        task = engine.state.task_scopes.get(task_id, []) if task_id else []
        scopes = task + engine.state.session_scopes
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
        """Record a pending scope into the internal buffer (not into shared state)."""
        scope = RequestedScope(
            zone=item.zone, actions=[item.action],
            path_prefixes=[item.path], tool_names=[tool_name],
        )
        self._pending_buffer.append(scope)

    def take_pending_scopes(self) -> list[RequestedScope]:
        """Return and clear accumulated pending scopes. Called by engine.drain_prepare_data()."""
        result = list(self._pending_buffer)
        self._pending_buffer.clear()
        return result


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

    v1.9.6: set_mode() replaced by _switch_policy() (internal method).
    New methods: apply_update(), _clear_all_scopes(), drain_prepare_data(), clear_pending_buffer().
    """

    def __init__(self, agent: LlamAgent):
        self.agent = agent
        self.state = AuthorizationState()
        self.policy: AuthorizationPolicy = InteractivePolicy()

    def _switch_policy(self, mode: str, state: TaskModeState | None = None) -> AuthorizationUpdateResult:
        """
        Switch authorization policy. Internal method, called only by agent.set_mode().

        Does NOT clear scopes or write agent attributes — those are agent's responsibility.
        Returns events for newly loaded scopes (e.g., seed scopes in continuous mode).
        """
        events: list[tuple[str, dict]] = []
        has_session_scopes = False
        if mode == "task":
            if state is None:
                raise ValueError("TaskPolicy requires a TaskModeState instance")
            self.policy = TaskPolicy(state)
            self._load_seed_scopes()
            if not self.state.session_scopes:
                # No seed scopes configured — ask user whether to open project access
                request = ConfirmRequest(
                    kind="session_authorize",
                    tool_name="*",
                    action="read_write",
                    zone="project",
                    target_paths=[self.agent.project_dir],
                    message="Allow all tasks to read and write in the project directory?",
                )
                response = self.confirm(request)
                if response.allow:
                    self.state.session_scopes = [ApprovalScope(
                        scope="session", zone="project",
                        actions=["read", "write"],
                        path_prefixes=[self.agent.project_dir],
                        created_at=time.time(), source="session_authorize",
                    )]
            has_session_scopes = bool(self.state.session_scopes)
            for s in self.state.session_scopes:
                events.append(("scope_issued", {"scope": s, "task_id": None}))
        elif mode == "interactive":
            self.policy = InteractivePolicy()
        elif mode == "continuous":
            self.policy = ContinuousPolicy()
            self._load_seed_scopes()
            for s in self.state.session_scopes:
                events.append(("scope_issued", {"scope": s, "task_id": None}))
        return AuthorizationUpdateResult(
            events=events, changed=bool(events),
            has_session_scopes=has_session_scopes,
        )

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
        logger.info("Loaded %d seed scopes", len(scopes))

    def apply_update(self, update: AuthorizationUpdate) -> AuthorizationUpdateResult:
        """
        Apply an authorization state change. Called by agent as a transparent courier.

        - approved_scopes: converts RequestedScope → ApprovalScope and writes to task_scopes
        - clear_task_scope: removes scopes for a specific task_id
        - clear_session_scopes: removes all session scopes
        """
        events: list[tuple[str, dict]] = []
        changed = False

        # Write approved scopes
        if update.approved_scopes and update.task_id:
            for rs in update.approved_scopes:
                scope = ApprovalScope(
                    scope="task", zone=rs.zone, actions=rs.actions,
                    path_prefixes=rs.path_prefixes, tool_names=rs.tool_names,
                    created_at=time.time(), source="contract",
                )
                self.state.task_scopes.setdefault(update.task_id, []).append(scope)
                events.append(("scope_issued", {"scope": scope, "task_id": update.task_id}))
            changed = True

        # Clear task scopes
        if update.clear_task_scope and update.task_id:
            removed = self.state.task_scopes.pop(update.task_id, [])
            for s in removed:
                events.append(("scope_revoked", {"scope": s, "reason": "task_completed"}))
            if removed:
                changed = True

        # Clear session scopes
        if update.clear_session_scopes:
            for s in self.state.session_scopes:
                events.append(("scope_revoked", {"scope": s, "reason": "session_clear"}))
            if self.state.session_scopes:
                changed = True
            self.state.session_scopes.clear()

        return AuthorizationUpdateResult(events=events, changed=changed)

    def _clear_all_scopes(self, reason: str = "mode_switch") -> AuthorizationUpdateResult:
        """
        Clear all scopes (task + session). Internal method for set_mode() / shutdown.

        Returns revocation events for agent to emit via hook system.
        """
        events: list[tuple[str, dict]] = []
        for tid, scopes in self.state.task_scopes.items():
            for s in scopes:
                events.append(("scope_revoked", {"scope": s, "reason": reason}))
        for s in self.state.session_scopes:
            events.append(("scope_revoked", {"scope": s, "reason": reason}))
        changed = bool(self.state.task_scopes or self.state.session_scopes)
        self.state.task_scopes.clear()
        self.state.session_scopes.clear()
        return AuthorizationUpdateResult(events=events, changed=changed)

    def authorization_status(self) -> dict:
        """
        Return a formatted snapshot of current authorization state.

        Agent calls this for diagnostics without needing to understand scope internals.
        """
        now = time.time()
        return {
            "task_scopes": {
                tid: [
                    {"zone": s.zone, "actions": s.actions, "path_prefixes": s.path_prefixes,
                     "uses": s.uses, "max_uses": s.max_uses, "source": s.source,
                     "expired": s.expires_at is not None and now > s.expires_at}
                    for s in scopes
                ]
                for tid, scopes in self.state.task_scopes.items()
            },
            "session_scopes": [
                {"zone": s.zone, "actions": s.actions, "path_prefixes": s.path_prefixes,
                 "uses": s.uses, "max_uses": s.max_uses, "source": s.source,
                 "expired": s.expires_at is not None and now > s.expires_at}
                for s in self.state.session_scopes
            ],
        }

    def drain_prepare_data(self) -> dict:
        """
        Drain data accumulated during prepare pipeline. Returns opaque dict.

        Agent calls this after prepare pipeline and puts the result into
        PipelineOutcome.metadata. Agent does not inspect the contents.
        """
        if isinstance(self.policy, TaskPolicy):
            return {"pending_scopes": self.policy.take_pending_scopes()}
        return {}

    def clear_pending_buffer(self) -> None:
        """Clear any stale pending scopes from a previous failed prepare."""
        if isinstance(self.policy, TaskPolicy):
            self.policy._pending_buffer.clear()

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
            except Exception as e:
                logger.warning("path_extractor raised exception for tool '%s': %s",
                               tool.get("name", "unknown"), e)
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
