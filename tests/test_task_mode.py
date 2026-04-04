"""
Task mode flow tests: controller protocol, driving loop, data flow, history management,
controlled dry-run, state machine, scope normalization, scope matching, scope lifecycle.

Consolidated from ~37 unit tests into ~8 flow tests. Each test exercises a complete
scenario with multiple assertions covering all original test coverage.
"""

import os

import pytest

from llamagent.core.agent import LlamAgent, Module
from llamagent.core.zone import ConfirmRequest, ConfirmResponse, RequestedScope
from llamagent.core.authorization import ApprovalScope, TaskPolicy, InteractivePolicy
from llamagent.core.contract import (
    TaskContract, TaskModeState, PipelineOutcome, AuthorizationUpdate, normalize_scopes,
)
from llamagent.core.controller import TaskModeController, ModeAction, ModeController
from conftest import make_llm_response, make_tool_call


def _setup_zone(agent, tmp_path):
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


def _make_awaiting_ctrl():
    """Helper: create a controller in awaiting_confirmation with one scope."""
    ctrl = TaskModeController()
    ctrl.handle_turn("do task X")
    scopes = [RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"])]
    ctrl.on_pipeline_done(
        ModeAction(kind="run_prepare"),
        PipelineOutcome(response="ok", metadata={"pending_scopes": scopes}),
    )
    assert ctrl.state.phase == "awaiting_confirmation"
    return ctrl


# ============================================================
# 1. Controller state machine (pure, no agent)
# ============================================================

def test_controller_state_machine_flow():
    """Full controller lifecycle: idle -> prepare -> contract -> confirm -> execute -> cleanup,
    plus cancel, re-prepare, error guard, reset, and auto_execute."""

    # --- Initial state and ABC conformance ---
    ctrl = TaskModeController()
    assert isinstance(ctrl, ModeController)
    assert ctrl.is_idle()
    assert ctrl.state.phase == "idle"

    # --- idle -> preparing via handle_turn ---
    action = ctrl.handle_turn("do task X")
    assert action.kind == "run_prepare"
    assert action.query == "do task X"
    assert ctrl.state.phase == "preparing"
    assert ctrl.state.original_query == "do task X"
    assert ctrl.state.task_id  # non-empty UUID

    # --- Safety guard: handle_turn during preparing returns error ---
    guard_action = ctrl.handle_turn("unexpected")
    assert guard_action.kind == "reply"
    assert "progress" in guard_action.response.lower()

    # --- preparing -> awaiting_confirmation (scopes present) ---
    scopes = [RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"])]
    outcome = PipelineOutcome(response="planned", metadata={"pending_scopes": scopes})
    action = ctrl.on_pipeline_done(ModeAction(kind="run_prepare"), outcome)
    assert action.kind == "await_user"
    assert "[Task Contract]" in action.response
    assert ctrl.state.phase == "awaiting_confirmation"
    assert ctrl.state.contract is not None

    # --- Re-prepare: user provides more info while awaiting ---
    action = ctrl.handle_turn("also include tests/")
    assert action.kind == "run_prepare"
    assert action.query == "also include tests/"
    assert ctrl.state.phase == "preparing"
    assert ctrl.state.clarification_turns == 1

    # Return to awaiting for next sub-tests
    ctrl.on_pipeline_done(
        ModeAction(kind="run_prepare"),
        PipelineOutcome(response="re-planned", metadata={"pending_scopes": scopes}),
    )
    assert ctrl.state.phase == "awaiting_confirmation"

    # --- Cancel from awaiting ---
    cancel_action = ctrl.handle_turn("no")
    assert cancel_action.kind == "cancel"
    assert "cancelled" in cancel_action.response.lower()
    assert cancel_action.authorization_update.clear_task_scope is True
    assert ctrl.is_idle()

    # --- Fresh cycle: prepare with no scopes skips to execute ---
    ctrl2 = TaskModeController()
    ctrl2.handle_turn("simple task")
    outcome_empty = PipelineOutcome(response="no writes", metadata={"pending_scopes": []})
    action = ctrl2.on_pipeline_done(ModeAction(kind="run_prepare"), outcome_empty)
    assert action.kind == "run_execute"
    assert ctrl2.state.phase == "executing"

    # --- Confirm from awaiting -> run_execute with auth_update ---
    ctrl3 = _make_awaiting_ctrl()
    action = ctrl3.handle_turn("yes")
    assert action.kind == "run_execute"
    assert action.authorization_update is not None
    assert len(action.authorization_update.approved_scopes) == 1
    assert ctrl3.state.phase == "executing"

    # --- Execute done: cleanup returns reply with clear_task_scope ---
    ctrl4 = TaskModeController()
    ctrl4.handle_turn("do task X")
    ctrl4.state.phase = "executing"
    ctrl4.state.task_id = "T1"

    action = ctrl4.on_pipeline_done(ModeAction(kind="run_execute"), PipelineOutcome(response="Done."))
    assert action.kind == "reply"
    assert action.response == "Done."
    assert action.authorization_update.clear_task_scope is True
    assert action.authorization_update.task_id == "T1"
    assert ctrl4.is_idle()

    # Error case also triggers cleanup
    ctrl4.handle_turn("do again")
    ctrl4.state.phase = "executing"
    ctrl4.state.task_id = "T2"
    action = ctrl4.on_pipeline_done(ModeAction(kind="run_execute"), PipelineOutcome(response="Error: fail"))
    assert action.authorization_update.clear_task_scope is True
    assert ctrl4.is_idle()

    # --- Reset ---
    ctrl5 = TaskModeController()
    ctrl5.handle_turn("do task X")
    assert not ctrl5.is_idle()
    ctrl5.reset()
    assert ctrl5.is_idle()
    assert ctrl5.state.task_id == ""

    # --- auto_execute: skips prepare from idle ---
    ctrl6 = TaskModeController()
    ctrl6.auto_execute = True
    action = ctrl6.handle_turn("do something")
    assert action.kind == "run_execute"
    assert ctrl6.state.phase == "executing"
    assert ctrl6.state.confirmed is True


# ============================================================
# 2. Data flow: drain, clear, normalize_scopes
# ============================================================

def test_data_flow_drain_and_normalize(bare_agent, tmp_path):
    """Drain returns scopes and clears buffer; clear_pending_buffer works;
    drain in non-task mode returns empty; normalize_scopes merges and separates."""

    # --- drain returns empty when not task mode ---
    assert bare_agent._authorization_engine.drain_prepare_data() == {}

    # --- drain and clear buffer in task mode ---
    _setup_zone(bare_agent, tmp_path)
    bare_agent.set_mode("task")
    bare_agent._controller.state.phase = "preparing"
    bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                             path_extractor=lambda a: [a.get("path", "")])
    bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")})

    data = bare_agent._authorization_engine.drain_prepare_data()
    assert len(data["pending_scopes"]) == 1
    assert isinstance(data["pending_scopes"][0], RequestedScope)
    # Second drain returns empty (buffer cleared)
    assert len(bare_agent._authorization_engine.drain_prepare_data()["pending_scopes"]) == 0

    # --- clear_pending_buffer works independently ---
    bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "g.py")})
    assert len(bare_agent._authorization_engine.policy._pending_buffer) == 1
    bare_agent._authorization_engine.clear_pending_buffer()
    assert len(bare_agent._authorization_engine.policy._pending_buffer) == 0

    # --- normalize_scopes: merge and separate ---
    assert normalize_scopes([]) == []

    merged = normalize_scopes([
        RequestedScope(zone="project", actions=["write"], path_prefixes=["/a/b/c.py"], tool_names=["w"]),
        RequestedScope(zone="project", actions=["write"], path_prefixes=["/a/b/d.py"], tool_names=["w"]),
    ])
    assert len(merged) == 1

    separate = normalize_scopes([
        RequestedScope(zone="project", actions=["write"], path_prefixes=["a.py"]),
        RequestedScope(zone="external", actions=["write"], path_prefixes=["b.py"]),
    ])
    assert len(separate) == 2


# ============================================================
# 3. Controlled dry-run (call_tool level)
# ============================================================

def test_controlled_dry_run(bare_agent, tmp_path, mock_llm_client):
    """Prepare phase: read executes, write records (not executed), playground write records,
    HARD_DENY rejects without recording, ask_user always allowed, safety modules block."""

    _setup_zone(bare_agent, tmp_path)

    # --- Read executes, write records but does not execute ---
    call_log = []
    bare_agent.register_tool("reader", lambda path="": (call_log.append("r"), "content")[1],
                             "read", safety_level=1, path_extractor=lambda a: [a.get("path", "")])
    bare_agent.register_tool("writer", lambda path="": (call_log.append("w"), "written")[1],
                             "write", safety_level=2, path_extractor=lambda a: [a.get("path", "")])
    bare_agent.register_tool("ask_user", lambda question="": "yes", "ask")

    bare_agent.set_mode("task")
    bare_agent._controller.state.phase = "preparing"

    assert bare_agent.call_tool("reader", {"path": os.path.join(str(tmp_path), "f.py")}) == "content"
    result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")})
    assert "[Prepare]" in result
    assert call_log == ["r"]  # writer never actually called
    assert len(bare_agent._authorization_engine.policy._pending_buffer) >= 1

    # --- HARD_DENY rejects without recording ---
    result = bare_agent.call_tool("writer", {"path": "/etc/passwd"})
    assert "blocked" in result.lower() or "cannot" in result.lower()
    # pending_buffer should only have the one from before (project write), not the hard-denied one
    # We already had >= 1 from above; the hard-deny does NOT add more
    buffer_before = len(bare_agent._authorization_engine.policy._pending_buffer)

    # Call again to outside zone to verify no growth
    bare_agent.call_tool("writer", {"path": "/etc/shadow"})
    assert len(bare_agent._authorization_engine.policy._pending_buffer) == buffer_before

    # --- ask_user always allowed during prepare ---
    assert bare_agent.call_tool("ask_user", {"question": "ok?"}) == "yes"

    # --- on_input safety module blocks during prepare ---
    bare_agent2_setup = bare_agent  # reuse agent but switch to fresh mode cycle
    bare_agent._controller.state.phase = "idle"  # must be idle before switching
    bare_agent2_setup.set_mode("interactive")  # reset
    _setup_zone(bare_agent2_setup, tmp_path)

    class Blocker(Module):
        name = "blocker"
        description = ""
        def on_input(self, u): return ""

    bare_agent2_setup.register_module(Blocker())
    bare_agent2_setup.set_mode("task")
    result = bare_agent2_setup.chat("bad")
    assert "sorry" in result.lower() or "cannot" in result.lower()


# ============================================================
# 4. Happy path full lifecycle (prepare -> confirm -> execute -> idle)
# ============================================================

def test_happy_path_full_lifecycle(bare_agent, tmp_path, mock_llm_client):
    """Full flow: chat -> prepare -> contract -> confirm -> execute -> idle.
    Verifies scope cleanup after execution and history recording at each stage."""

    _setup_zone(bare_agent, tmp_path)
    bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                             path_extractor=lambda a: [a.get("path", "")])

    project_file = os.path.join(str(tmp_path), "main.py")
    mock_llm_client.set_responses([
        make_llm_response("", tool_calls=[make_tool_call("writer", {"path": project_file})]),
        make_llm_response("I need to write main.py"),
    ])

    bare_agent.set_mode("task")
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    result = bare_agent.chat("write main.py")
    assert "[Task Contract]" in result or "authorization" in result.lower()
    # History: user=query + assistant=contract
    assert len(bare_agent.history) == 2

    # --- Confirm and execute ---
    mock_llm_client.set_responses([make_llm_response("Done.")])
    bare_agent.chat("yes")
    assert bare_agent._controller.state.phase == "idle"
    # History: +user="yes" +assistant="Done."
    assert len(bare_agent.history) == 4
    assert bare_agent.history[-2]["content"] == "yes"
    assert bare_agent.history[-1]["content"] == "Done."

    # --- Execute cleans scopes after completion (separate setup) ---
    bare_agent.set_mode("interactive")  # reset
    bare_agent.set_mode("task")
    state = bare_agent._controller.state
    state.phase = "awaiting_confirmation"
    state.task_id = "T1"
    state.original_query = "do"
    state.contract = TaskContract(
        task_summary="do", planned_operations=["write"],
        requested_scopes=[RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"])],
        open_questions=[], risk_summary="1 op",
    )
    mock_llm_client.set_responses([make_llm_response("Done.")])
    before = len(bare_agent.history)
    bare_agent.chat("yes")
    assert "T1" not in bare_agent._authorization_engine.state.task_scopes
    assert len(bare_agent.history) == before + 2
    assert bare_agent.history[-1]["content"] == "Done."


# ============================================================
# 5. Cancel and mode switching
# ============================================================

def test_cancel_and_mode_switching(bare_agent, tmp_path, mock_llm_client):
    """Cancel resets state/scopes/history; mode switching shares state correctly;
    rejects non-idle switch; task_id priority; rejects invalid mode string."""

    _setup_zone(bare_agent, tmp_path)

    # --- Mode switching and shared state ---
    bare_agent.set_mode("task")
    assert bare_agent._controller is not None
    assert isinstance(bare_agent._authorization_engine.policy, TaskPolicy)
    assert bare_agent._authorization_engine.policy.state is bare_agent._controller.state

    bare_agent.set_mode("interactive")
    assert bare_agent._controller is None
    assert isinstance(bare_agent._authorization_engine.policy, InteractivePolicy)

    # --- Reject switch when not idle ---
    bare_agent.set_mode("task")
    bare_agent._controller.state.phase = "preparing"
    with pytest.raises(RuntimeError, match="Cannot switch mode"):
        bare_agent.set_mode("interactive")
    # Reset for next sub-test
    bare_agent._controller.state.phase = "idle"
    bare_agent.set_mode("interactive")

    # --- Invalid mode string ---
    with pytest.raises(ValueError, match="Invalid mode"):
        bare_agent.set_mode("invalid")

    # --- task_id priority: controller.state.task_id > _current_task_id ---
    bare_agent.set_mode("task")
    bare_agent._controller.state.task_id = "T123"
    assert bare_agent.get_active_task_id() == "T123"

    bare_agent._controller.state.phase = "idle"  # allow switch
    bare_agent.set_mode("interactive")
    assert bare_agent.get_active_task_id() is None
    bare_agent._current_task_id = "P456"
    assert bare_agent.get_active_task_id() == "P456"
    bare_agent._current_task_id = None  # cleanup

    # --- Cancel resets state, scopes, and history ---
    bare_agent.set_mode("task")
    state = bare_agent._controller.state
    state.phase = "awaiting_confirmation"
    state.task_id = "T1"
    state.original_query = "do"
    bare_agent._authorization_engine.state.task_scopes["T1"] = [
        ApprovalScope(scope="task", zone="project", actions=["write"], path_prefixes=["src/"])
    ]

    before = len(bare_agent.history)
    result = bare_agent.chat("no")
    assert "cancelled" in result.lower()
    assert state.phase == "idle"
    assert "T1" not in bare_agent._authorization_engine.state.task_scopes
    assert len(bare_agent.history) == before + 2
    assert bare_agent.history[-2]["content"] == "no"


# ============================================================
# 6. Scope matching and confirm handler edge cases
# ============================================================

def test_scope_matching_and_confirm_handler(bare_agent, tmp_path):
    """Tool within approved scope auto-approved; outside falls back to confirm;
    confirm_handler exception defaults to deny; legacy bool return is wrapped."""

    _setup_zone(bare_agent, tmp_path)
    bare_agent.set_mode("task")
    state = bare_agent._controller.state
    state.phase = "executing"
    state.task_id = "T1"
    bare_agent._current_task_id = "T1"

    bare_agent._authorization_engine.state.task_scopes["T1"] = [
        ApprovalScope(scope="task", zone="project", actions=["write"],
                      path_prefixes=[os.path.join(str(tmp_path), "src/")])
    ]

    call_count = [0]
    bare_agent.register_tool("writer",
                             lambda path="": (call_count.__setitem__(0, call_count[0]+1), "written")[1],
                             "write", safety_level=2, path_extractor=lambda a: [a.get("path", "")])

    # --- Auto-approve: within scope ---
    assert bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/main.py")}) == "written"
    assert call_count[0] == 1

    # --- Deny: outside scope, confirm_handler denies ---
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
    result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "docs/readme.md")})
    assert "denied" in result.lower()

    # --- confirm_handler exception defaults to deny ---
    bare_agent.confirm_handler = lambda req: (_ for _ in ()).throw(RuntimeError("broken"))
    result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")})
    assert "denied" in result.lower()

    # --- Legacy bool return is wrapped to ConfirmResponse ---
    bare_agent.confirm_handler = lambda req: True  # legacy bool return
    assert bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")}) == "written"


# ============================================================
# 7. Session scopes lifecycle
# ============================================================

def test_session_scopes_lifecycle(bare_agent, tmp_path, mock_llm_client):
    """Seed scopes enable auto_execute; user session_authorize grants scopes;
    no seed + deny/no handler falls back; session scopes persist across tasks;
    cleared on mode switch; execute uses session scopes for matching;
    task scopes take priority over session scopes."""

    _setup_zone(bare_agent, tmp_path)

    # --- seed_scopes -> session_scopes loaded -> auto_execute=True -> skip prepare ---
    bare_agent.config.seed_scopes = [
        {"scope": "session", "zone": "project", "actions": ["read", "write"],
         "path_prefixes": [str(tmp_path)]}
    ]
    bare_agent.register_tool("writer", lambda path="": "written", "write",
                             safety_level=2, path_extractor=lambda a: [a.get("path", "")])
    mock_llm_client.set_responses([make_llm_response("Done writing.")])

    bare_agent.set_mode("task")
    assert bare_agent._controller.auto_execute is True
    assert len(bare_agent._authorization_engine.state.session_scopes) == 1

    result = bare_agent.chat("write something")
    assert "Done writing" in result
    assert bare_agent._controller.state.phase == "idle"

    # --- Session scopes persist across tasks ---
    assert len(bare_agent._authorization_engine.state.session_scopes) == 1
    mock_llm_client.set_responses([make_llm_response("Done second.")])
    result = bare_agent.chat("second task")
    assert "Done second." in result

    # --- Session scopes cleared on mode switch ---
    bare_agent.set_mode("interactive")
    assert len(bare_agent._authorization_engine.state.session_scopes) == 0

    # --- User allows session_authorize -> auto_execute=True ---
    bare_agent.config.seed_scopes = None  # no seed scopes
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    mock_llm_client.set_responses([make_llm_response("Done.")])

    bare_agent.set_mode("task")
    assert bare_agent._controller.auto_execute is True
    scopes = bare_agent._authorization_engine.state.session_scopes
    assert len(scopes) == 1
    assert scopes[0].source == "session_authorize"
    assert scopes[0].zone == "project"

    result = bare_agent.chat("write file")
    assert "Done." in result
    bare_agent.set_mode("interactive")  # cleanup

    # --- No seed + user denies -> auto_execute=False ---
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
    bare_agent.set_mode("task")
    assert bare_agent._controller.auto_execute is False
    assert len(bare_agent._authorization_engine.state.session_scopes) == 0
    bare_agent.set_mode("interactive")

    # --- No seed + no confirm_handler -> auto_execute=False ---
    bare_agent.confirm_handler = None
    bare_agent.set_mode("task")
    assert bare_agent._controller.auto_execute is False
    assert len(bare_agent._authorization_engine.state.session_scopes) == 0
    bare_agent.set_mode("interactive")

    # --- Execute uses session_scope for matching ---
    bare_agent.config.seed_scopes = [
        {"scope": "session", "zone": "project", "actions": ["read", "write"],
         "path_prefixes": [str(tmp_path)]}
    ]
    bare_agent.set_mode("task")
    state = bare_agent._controller.state
    state.phase = "executing"
    state.task_id = "T_SESS"
    bare_agent._current_task_id = "T_SESS"

    call_count = [0]
    bare_agent.register_tool(
        "writer",
        lambda path="": (call_count.__setitem__(0, call_count[0]+1), "written")[1],
        "write", safety_level=2, path_extractor=lambda a: [a.get("path", "")]
    )

    result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/main.py")})
    assert result == "written"
    assert call_count[0] == 1

    # --- Task scopes take priority over session scopes ---
    state.task_id = "T_PRI"
    bare_agent._current_task_id = "T_PRI"

    # session scope from seed_scopes already covers project with tool_names unset (matches all)
    # Add a task scope with max_uses=1 and restricted path
    bare_agent._authorization_engine.state.task_scopes["T_PRI"] = [
        ApprovalScope(scope="task", zone="project", actions=["write"],
                      path_prefixes=[os.path.join(str(tmp_path), "src/")],
                      max_uses=1)
    ]

    # Re-register with tool_names on session scope for priority test
    session_scope = bare_agent._authorization_engine.state.session_scopes[0]
    session_scope.tool_names = ["writer"]
    session_uses_before = session_scope.uses

    call_count[0] = 0
    bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/main.py")})
    task_scope = bare_agent._authorization_engine.state.task_scopes["T_PRI"][0]
    assert task_scope.uses == 1

    # Second call: task_scope exhausted (max_uses=1), falls to session_scope
    bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/other.py")})
    assert session_scope.uses == session_uses_before + 1


# ============================================================
# 8. Re-prepare e2e, MAX_MODE_STEPS, interactive isolation
# ============================================================

def test_re_prepare_and_loop_protection(bare_agent, tmp_path, mock_llm_client):
    """Re-prepare end-to-end with supplementary info; MAX_MODE_STEPS exhaustion
    terminates the driving loop; interactive mode chat/call_tool unaffected."""

    # --- Interactive mode isolation: chat and call_tool work normally ---
    mock_llm_client.set_responses([make_llm_response("Hi")])
    assert bare_agent.mode == "interactive"
    assert bare_agent.chat("hi") == "Hi"

    _setup_zone(bare_agent, tmp_path)
    bare_agent.register_tool("t", lambda: "ok", "tool")
    assert bare_agent.call_tool("t", {}) == "ok"

    # --- Re-prepare end-to-end ---
    bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                             path_extractor=lambda a: [a.get("path", "")])

    project_file = os.path.join(str(tmp_path), "main.py")
    mock_llm_client.set_responses([
        make_llm_response("", tool_calls=[make_tool_call("writer", {"path": project_file})]),
        make_llm_response("I plan to write main.py"),
    ])

    bare_agent.set_mode("task")
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    result = bare_agent.chat("write main.py")
    assert "[Task Contract]" in result
    first_history_len = len(bare_agent.history)

    # Re-prepare with supplementary info
    test_file = os.path.join(str(tmp_path), "test.py")
    mock_llm_client.set_responses([
        make_llm_response("", tool_calls=[make_tool_call("writer", {"path": test_file})]),
        make_llm_response("I also plan to write test.py"),
    ])
    result2 = bare_agent.chat("also write test.py")
    assert "[Task Contract]" in result2
    assert len(bare_agent.history) == first_history_len + 2
    assert bare_agent.history[-2]["content"] == "also write test.py"

    # --- MAX_MODE_STEPS exhaustion ---
    bare_agent._controller.reset()  # reset controller to idle so we can switch
    bare_agent.set_mode("interactive")  # reset

    class InfiniteController(ModeController):
        """Always returns run_prepare, never terminates."""
        def __init__(self):
            self.state = TaskModeState()
            self.state.phase = "preparing"
            self.state.task_id = "T_INF"
            self.turn_count = 0

        def handle_turn(self, user_input):
            self.turn_count += 1
            return ModeAction(kind="run_prepare", query=user_input, task_id="T_INF")

        def on_pipeline_done(self, action, outcome):
            self.turn_count += 1
            return ModeAction(kind="run_prepare", query="again", task_id="T_INF")

        def reset(self):
            self.state.phase = "idle"
            return []

        def is_idle(self):
            return self.state.phase == "idle"

    ctrl = InfiniteController()
    bare_agent.mode = "task"
    bare_agent._controller = ctrl
    bare_agent._authorization_engine.policy = TaskPolicy(ctrl.state)

    result = bare_agent.chat("infinite task")
    assert "exceeded maximum steps" in result.lower()
    # handle_turn(1) + on_pipeline_done * MAX_MODE_STEPS
    assert ctrl.turn_count == 1 + bare_agent._MAX_MODE_STEPS
