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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llamagent.core.zone import (
    ConfirmRequest,
    ConfirmResponse,
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
    """Placeholder for future scope accumulation. Empty in v1.9.0."""
    pass


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
