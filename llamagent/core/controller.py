"""
Mode controller: pure state machine for task mode (and future continuous mode).

ModeController: abstract base class for mode controllers.
TaskModeController: task mode state machine — handle_turn / on_pipeline_done protocol.
ModeAction: structured action returned by controller, interpreted by agent.

The controller does NOT hold references to agent or engine. All data flows
through method parameters (input) and return values (output).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from llamagent.core.contract import (
    AuthorizationUpdate,
    PipelineOutcome,
    TaskContract,
    TaskModeState,
    normalize_scopes,
)
from llamagent.core.zone import RequestedScope


# Prepare system prompt injected into the LLM during dry-run
PREPARE_SYSTEM_PROMPT = (
    "\n[Task Mode: Prepare Phase]\n"
    "You are in a dry-run preparation phase. Write and execute operations will be "
    "recorded but NOT actually performed. Plan the task thoroughly — the user will "
    "review and approve before real execution begins."
)


@dataclass
class ModeAction:
    """
    Structured action returned by a mode controller.

    The agent interprets kind to decide what to do next, but does not
    understand the semantic meaning of the fields — it just routes data.
    """
    kind: str                  # reply | run_prepare | run_execute | await_user | cancel
    query: str | None = None
    extra_system: str = ""
    task_id: str | None = None
    response: str | None = None
    authorization_update: AuthorizationUpdate | None = None


class ModeController(ABC):
    """
    Abstract base class for mode controllers.

    All mode controllers follow the two-step protocol:
    1. handle_turn(user_input) → ModeAction
    2. agent runs pipeline (if needed)
    3. on_pipeline_done(action, outcome) → ModeAction

    Controllers are pure state machines: no references to agent or engine.
    """

    @abstractmethod
    def handle_turn(self, user_input: str) -> ModeAction:
        """Receive user input, return next action."""
        ...

    @abstractmethod
    def on_pipeline_done(self, action: ModeAction, outcome: PipelineOutcome) -> ModeAction:
        """Receive pipeline result, return next action."""
        ...

    @abstractmethod
    def reset(self) -> list[tuple[str, dict]]:
        """Reset to idle state. Returns events to emit."""
        ...

    @abstractmethod
    def is_idle(self) -> bool:
        """Whether the controller is in idle state."""
        ...


class TaskModeController(ModeController):
    """
    Task mode state machine.

    States: idle → preparing → awaiting_confirmation → executing → idle
    Also: awaiting_confirmation → preparing (re-prepare)
          awaiting_confirmation → idle (cancel)
    """

    # v2.0: maximum re-prepare rounds before forcing a decision
    MAX_CLARIFICATION_TURNS = 3

    def __init__(self):
        self.state = TaskModeState()
        self.auto_execute = False

    def is_idle(self) -> bool:
        return self.state.phase == "idle"

    def reset(self) -> list[tuple[str, dict]]:
        """Reset to idle. Returns empty events (no scopes to revoke at controller level)."""
        self.state.phase = "idle"
        self.state.task_id = ""
        self.state.original_query = ""
        self.state.contract = None
        self.state.confirmed = False
        self.state.clarification_turns = 0
        return []

    def handle_turn(self, user_input: str) -> ModeAction:
        """
        Receive user input, return next action based on current state.

        Transitions:
        - idle + any input → preparing, run_prepare
        - awaiting_confirmation + confirm → executing, run_execute
        - awaiting_confirmation + cancel → idle, cancel
        - awaiting_confirmation + other → preparing, run_prepare (re-prepare)
        - preparing/executing + any → reply with error (should not happen in normal flow)
        """
        phase = self.state.phase

        if phase == "idle":
            self.state.task_id = uuid.uuid4().hex
            self.state.original_query = user_input
            if self.auto_execute:
                # Session scopes pre-authorized — skip prepare/contract
                self.state.phase = "executing"
                self.state.confirmed = True
                return ModeAction(
                    kind="run_execute",
                    query=user_input,
                    task_id=self.state.task_id,
                )
            self.state.phase = "preparing"
            return ModeAction(
                kind="run_prepare",
                query=user_input,
                extra_system=PREPARE_SYSTEM_PROMPT,
                task_id=self.state.task_id,
            )

        if phase == "awaiting_confirmation":
            lower = user_input.strip().lower()
            if lower in ("yes", "y", "confirm", "ok", "approve"):
                self.state.phase = "executing"
                self.state.confirmed = True
                # Build authorization_update from confirmed contract scopes
                auth_update = None
                if self.state.contract and self.state.contract.requested_scopes:
                    auth_update = AuthorizationUpdate(
                        task_id=self.state.task_id,
                        approved_scopes=self.state.contract.requested_scopes,
                    )
                return ModeAction(
                    kind="run_execute",
                    query=self.state.original_query,
                    task_id=self.state.task_id,
                    authorization_update=auth_update,
                )
            elif lower in ("no", "n", "cancel", "abort"):
                task_id = self.state.task_id
                self.reset()
                # Defensive: clear any task scopes (should be no-op during awaiting_confirmation)
                return ModeAction(
                    kind="cancel",
                    response="Task cancelled.",
                    authorization_update=AuthorizationUpdate(
                        task_id=task_id,
                        clear_task_scope=True,
                    ),
                )
            else:
                # Additional info — re-prepare
                self.state.clarification_turns += 1
                if self.state.clarification_turns > self.MAX_CLARIFICATION_TURNS:
                    return ModeAction(
                        kind="await_user",
                        query=user_input,
                        response=(
                            f"Maximum clarification rounds ({self.MAX_CLARIFICATION_TURNS}) reached. "
                            "Reply 'yes' to execute with the current plan, or 'no' to cancel."
                        ),
                    )
                self.state.phase = "preparing"
                return ModeAction(
                    kind="run_prepare",
                    query=user_input,
                    extra_system=PREPARE_SYSTEM_PROMPT,
                    task_id=self.state.task_id,
                )

        # preparing / executing — should not receive handle_turn in normal flow
        # (pipeline is running synchronously). Return error as safety guard.
        return ModeAction(
            kind="reply",
            response="Task is currently in progress. Please wait for it to complete.",
        )

    def on_pipeline_done(self, action: ModeAction, outcome: PipelineOutcome) -> ModeAction:
        """
        Receive pipeline result, return next action.

        For run_prepare: reads pending_scopes from outcome.metadata, builds contract.
        For run_execute: returns reply with cleanup authorization_update.
        """
        if action.kind == "run_prepare":
            return self._on_prepare_done(action, outcome)
        if action.kind == "run_execute":
            return self._on_execute_done(outcome)
        # Unknown action kind — should not happen
        return ModeAction(kind="reply", response=outcome.response)

    def _on_prepare_done(self, action: ModeAction, outcome: PipelineOutcome) -> ModeAction:
        """Process prepare pipeline result: build contract from pending scopes."""
        # Read pending_scopes from outcome metadata (explicit data flow, not shared state)
        raw = outcome.metadata.get("pending_scopes", [])
        if not isinstance(raw, list):
            raw = []
        scopes: list[RequestedScope] = raw

        normalized = normalize_scopes(scopes)

        # No CONFIRMABLE operations — skip contract, go straight to execute
        if not normalized:
            self.state.phase = "executing"
            self.state.confirmed = True
            auth_update = None
            return ModeAction(
                kind="run_execute",
                query=self.state.original_query,
                task_id=self.state.task_id,
                authorization_update=auth_update,
            )

        # v2.0: extract open_questions from outcome metadata
        open_questions = outcome.metadata.get("open_questions", [])
        if not isinstance(open_questions, list):
            open_questions = []

        # Build contract
        contract = TaskContract(
            task_summary=self.state.original_query,
            planned_operations=[
                f"{s.actions[0]} in {s.zone}: {', '.join(s.path_prefixes)}"
                for s in normalized
            ],
            requested_scopes=normalized,
            open_questions=open_questions,
            risk_summary=f"{len(normalized)} operations require authorization.",
        )
        self.state.contract = contract
        self.state.phase = "awaiting_confirmation"

        # Build contract display text
        lines = [f"[Task Contract] {contract.task_summary}", ""]
        lines.append("Planned operations requiring authorization:")
        for op in contract.planned_operations:
            lines.append(f"  - {op}")
        if contract.open_questions:
            lines.append("\nOpen questions:")
            for q in contract.open_questions:
                lines.append(f"  ? {q}")
        lines.append(f"\nRisk: {contract.risk_summary}")
        lines.append("\nReply 'yes' to confirm, 'no' to cancel, or provide more details.")

        # query carries the user message for history: from the run_prepare action that
        # triggered this pipeline. First prepare: original_query; re-prepare: supplementary input.
        # This way agent doesn't need to peek into controller.state.
        return ModeAction(
            kind="await_user",
            query=action.query,
            response="\n".join(lines),
        )

    def _on_execute_done(self, outcome: PipelineOutcome) -> ModeAction:
        """
        Process execute pipeline result: always return cleanup regardless of success/failure.

        The cleanup authorization_update clears the task's scopes.
        """
        task_id = self.state.task_id
        self.reset()

        return ModeAction(
            kind="reply",
            response=outcome.response,
            authorization_update=AuthorizationUpdate(
                task_id=task_id,
                clear_task_scope=True,
            ),
        )
