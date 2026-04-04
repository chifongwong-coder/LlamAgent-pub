"""
Task mode tests: controller protocol, driving loop, data flow, history management,
controlled dry-run, state machine, scope normalization, scope matching, scope lifecycle.

Tests are written from the v1.9.6 plan document, NOT from implementation code.
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
# TaskModeController unit tests (pure state machine)
# ============================================================

class TestTaskModeController:
    """Controller as pure state machine: no agent/engine references, all data via params."""

    def test_initial_state_and_abc(self):
        ctrl = TaskModeController()
        assert isinstance(ctrl, ModeController)
        assert ctrl.is_idle()
        assert ctrl.state.phase == "idle"

    def test_handle_turn_idle_returns_run_prepare(self):
        ctrl = TaskModeController()
        action = ctrl.handle_turn("do task X")
        assert action.kind == "run_prepare"
        assert action.query == "do task X"
        assert ctrl.state.phase == "preparing"
        assert ctrl.state.original_query == "do task X"
        assert ctrl.state.task_id  # non-empty UUID

    def test_prepare_with_scopes_returns_await_user(self):
        ctrl = TaskModeController()
        ctrl.handle_turn("do task X")
        scopes = [RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"])]
        outcome = PipelineOutcome(response="planned", metadata={"pending_scopes": scopes})
        action = ctrl.on_pipeline_done(ModeAction(kind="run_prepare"), outcome)
        assert action.kind == "await_user"
        assert "[Task Contract]" in action.response
        assert ctrl.state.phase == "awaiting_confirmation"
        assert ctrl.state.contract is not None

    def test_prepare_no_scopes_skips_to_execute(self):
        ctrl = TaskModeController()
        ctrl.handle_turn("do task X")
        outcome = PipelineOutcome(response="no writes", metadata={"pending_scopes": []})
        action = ctrl.on_pipeline_done(ModeAction(kind="run_prepare"), outcome)
        assert action.kind == "run_execute"
        assert ctrl.state.phase == "executing"

    def test_confirm_returns_run_execute_with_auth_update(self):
        ctrl = _make_awaiting_ctrl()
        action = ctrl.handle_turn("yes")
        assert action.kind == "run_execute"
        assert action.authorization_update is not None
        assert len(action.authorization_update.approved_scopes) == 1
        assert ctrl.state.phase == "executing"

    def test_cancel_returns_cancel_with_cleanup(self):
        ctrl = _make_awaiting_ctrl()
        action = ctrl.handle_turn("no")
        assert action.kind == "cancel"
        assert "cancelled" in action.response.lower()
        assert action.authorization_update.clear_task_scope is True
        assert ctrl.is_idle()

    def test_re_prepare(self):
        ctrl = _make_awaiting_ctrl()
        action = ctrl.handle_turn("also include tests/")
        assert action.kind == "run_prepare"
        assert action.query == "also include tests/"
        assert ctrl.state.phase == "preparing"
        assert ctrl.state.clarification_turns == 1

    def test_execute_done_cleanup(self):
        """on_pipeline_done for execute always returns cleanup, success or error."""
        ctrl = TaskModeController()
        ctrl.handle_turn("do task X")
        ctrl.state.phase = "executing"
        ctrl.state.task_id = "T1"

        # Success
        action = ctrl.on_pipeline_done(ModeAction(kind="run_execute"), PipelineOutcome(response="Done."))
        assert action.kind == "reply"
        assert action.response == "Done."
        assert action.authorization_update.clear_task_scope is True
        assert action.authorization_update.task_id == "T1"
        assert ctrl.is_idle()

        # Error (re-enter executing to test)
        ctrl.handle_turn("do again")
        ctrl.state.phase = "executing"
        ctrl.state.task_id = "T2"
        action = ctrl.on_pipeline_done(ModeAction(kind="run_execute"), PipelineOutcome(response="Error: fail"))
        assert action.authorization_update.clear_task_scope is True
        assert ctrl.is_idle()

    def test_reset(self):
        ctrl = TaskModeController()
        ctrl.handle_turn("do task X")
        assert not ctrl.is_idle()
        ctrl.reset()
        assert ctrl.is_idle()
        assert ctrl.state.task_id == ""

    def test_handle_turn_in_preparing_returns_error(self):
        """Safety guard: handle_turn during pipeline execution returns error."""
        ctrl = TaskModeController()
        ctrl.handle_turn("do task X")
        action = ctrl.handle_turn("unexpected")
        assert action.kind == "reply"
        assert "progress" in action.response.lower()


# ============================================================
# Data flow tests (pending_scopes via metadata)
# ============================================================

class TestDataFlow:
    """Verify pending_scopes flow: TaskPolicy buffer -> drain -> metadata -> controller."""

    def test_drain_and_clear_buffer(self, bare_agent, tmp_path):
        """drain returns scopes, clears buffer; clear_pending_buffer also works."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.set_mode("task")
        bare_agent._controller.state.phase = "preparing"
        bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])
        bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")})

        # drain returns scopes and clears
        data = bare_agent._authorization_engine.drain_prepare_data()
        assert len(data["pending_scopes"]) == 1
        assert isinstance(data["pending_scopes"][0], RequestedScope)
        assert len(bare_agent._authorization_engine.drain_prepare_data()["pending_scopes"]) == 0

        # clear_pending_buffer works independently
        bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "g.py")})
        assert len(bare_agent._authorization_engine.policy._pending_buffer) == 1
        bare_agent._authorization_engine.clear_pending_buffer()
        assert len(bare_agent._authorization_engine.policy._pending_buffer) == 0

    def test_drain_returns_empty_when_not_task_mode(self, bare_agent):
        assert bare_agent._authorization_engine.drain_prepare_data() == {}


# ============================================================
# Controlled dry-run (call_tool level)
# ============================================================

class TestControlledDryRun:
    """Prepare phase: read executes, write/execute blocked, ask_user allowed, safety runs."""

    def test_read_write_and_playground_behavior(self, bare_agent, tmp_path):
        """Read executes, write records, playground write also records during prepare."""
        _setup_zone(bare_agent, tmp_path)
        call_log = []
        bare_agent.register_tool("reader", lambda path="": (call_log.append("r"), "content")[1],
                                 "read", safety_level=1, path_extractor=lambda a: [a.get("path", "")])
        bare_agent.register_tool("writer", lambda path="": (call_log.append("w"), "written")[1],
                                 "write", safety_level=2, path_extractor=lambda a: [a.get("path", "")])

        bare_agent.set_mode("task")
        bare_agent._controller.state.phase = "preparing"

        assert bare_agent.call_tool("reader", {"path": os.path.join(str(tmp_path), "f.py")}) == "content"
        result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")})
        assert "[Prepare]" in result

        assert call_log == ["r"]  # writer never called
        assert len(bare_agent._authorization_engine.policy._pending_buffer) >= 1

    def test_hard_deny_and_ask_user(self, bare_agent, tmp_path):
        """HARD_DENY rejects (not recorded). ask_user always allowed."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("writer", lambda path="": "no", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])
        bare_agent.register_tool("ask_user", lambda question="": "yes", "ask")

        bare_agent.set_mode("task")
        bare_agent._controller.state.phase = "preparing"

        result = bare_agent.call_tool("writer", {"path": "/etc/passwd"})
        assert "blocked" in result.lower() or "cannot" in result.lower()
        assert len(bare_agent._authorization_engine.policy._pending_buffer) == 0
        assert bare_agent.call_tool("ask_user", {"question": "ok?"}) == "yes"

    def test_on_input_safety_during_prepare(self, bare_agent, tmp_path, mock_llm_client):
        _setup_zone(bare_agent, tmp_path)
        class Blocker(Module):
            name = "blocker"
            description = ""
            def on_input(self, u): return ""
        bare_agent.register_module(Blocker())
        bare_agent.set_mode("task")
        result = bare_agent.chat("bad")
        assert "sorry" in result.lower() or "cannot" in result.lower()


# ============================================================
# Integration tests (state machine + scope + history combined)
# ============================================================

class TestIntegration:
    """Full flows through chat(): happy path, cancel, scope matching, lifecycle."""

    def test_mode_switching_and_shared_state(self, bare_agent):
        bare_agent.set_mode("task")
        assert bare_agent._controller is not None
        assert isinstance(bare_agent._authorization_engine.policy, TaskPolicy)
        assert bare_agent._authorization_engine.policy.state is bare_agent._controller.state

        bare_agent.set_mode("interactive")
        assert bare_agent._controller is None
        assert isinstance(bare_agent._authorization_engine.policy, InteractivePolicy)

    def test_set_mode_rejects_non_idle(self, bare_agent):
        bare_agent.set_mode("task")
        bare_agent._controller.state.phase = "preparing"
        with pytest.raises(RuntimeError, match="Cannot switch mode"):
            bare_agent.set_mode("interactive")

    def test_task_id_priority(self, bare_agent):
        """get_active_task_id: controller.state.task_id > _current_task_id."""
        bare_agent.set_mode("task")
        bare_agent._controller.state.task_id = "T123"
        assert bare_agent.get_active_task_id() == "T123"

        bare_agent.set_mode("interactive")
        assert bare_agent.get_active_task_id() is None
        bare_agent._current_task_id = "P456"
        assert bare_agent.get_active_task_id() == "P456"

    def test_happy_path_prepare_confirm_execute(self, bare_agent, tmp_path, mock_llm_client):
        """Full flow: chat → prepare → contract → confirm → execute → idle."""
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

        mock_llm_client.set_responses([make_llm_response("Done.")])
        bare_agent.chat("yes")
        assert bare_agent._controller.state.phase == "idle"
        # History: +user="yes" +assistant="Done."
        assert len(bare_agent.history) == 4
        assert bare_agent.history[-2]["content"] == "yes"
        assert bare_agent.history[-1]["content"] == "Done."

    def test_cancel_resets_state_scopes_and_history(self, bare_agent, tmp_path, mock_llm_client):
        """Cancel clears engine scopes, resets state, writes history."""
        _setup_zone(bare_agent, tmp_path)
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

    def test_execute_cleans_scopes_after_completion(self, bare_agent, tmp_path, mock_llm_client):
        """Scopes are cleaned after execute completes; history records confirm + result."""
        _setup_zone(bare_agent, tmp_path)
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

    def test_scope_matching_auto_approves(self, bare_agent, tmp_path):
        """Tool within approved scope auto-approved; outside falls back to confirm."""
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
        bare_agent.register_tool("writer", lambda path="": (call_count.__setitem__(0, call_count[0]+1), "written")[1],
                                 "write", safety_level=2, path_extractor=lambda a: [a.get("path", "")])

        assert bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/main.py")}) == "written"
        assert call_count[0] == 1

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
        result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "docs/readme.md")})
        assert "denied" in result.lower()

    def test_set_mode_rejects_invalid(self, bare_agent):
        """Invalid mode string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            bare_agent.set_mode("invalid")

    def test_re_prepare_end_to_end(self, bare_agent, tmp_path, mock_llm_client):
        """Re-prepare: user provides more info → new contract with fresh scopes."""
        _setup_zone(bare_agent, tmp_path)
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
        # History should have grown: user=supplementary + assistant=new contract
        assert len(bare_agent.history) == first_history_len + 2
        assert bare_agent.history[-2]["content"] == "also write test.py"

    def test_confirm_handler_exception_denies(self, bare_agent, tmp_path):
        """confirm_handler raising exception → default deny."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])
        bare_agent.confirm_handler = lambda req: (_ for _ in ()).throw(RuntimeError("broken"))
        result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")})
        assert "denied" in result.lower()

    def test_confirm_handler_bool_backward_compat(self, bare_agent, tmp_path):
        """confirm_handler returning bool (legacy) is wrapped to ConfirmResponse."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])
        bare_agent.confirm_handler = lambda req: True  # legacy bool return
        assert bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")}) == "ok"


# ============================================================
# Scope normalization (pure function)
# ============================================================

class TestScopeNormalization:
    def test_merge_and_separate(self):
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
# ============================================================
# v1.9.8: Task mode session scopes (shared authorization)
# ============================================================

class TestTaskModeSessionScopes:
    """Task mode with session scopes: seed_scopes, user confirmation, auto_execute."""

    def test_seed_scopes_enable_auto_execute(self, bare_agent, tmp_path, mock_llm_client):
        """Config seed_scopes → session_scopes loaded → auto_execute=True → skip prepare."""
        _setup_zone(bare_agent, tmp_path)
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

        # chat goes directly to execute, no prepare/contract
        result = bare_agent.chat("write something")
        assert "Done writing" in result
        assert bare_agent._controller.state.phase == "idle"

    def test_user_allows_session_authorize(self, bare_agent, tmp_path, mock_llm_client):
        """No seed_scopes + user allows session_authorize → auto_execute=True."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        bare_agent.register_tool("writer", lambda path="": "written", "write",
                                 safety_level=2, path_extractor=lambda a: [a.get("path", "")])
        mock_llm_client.set_responses([make_llm_response("Done.")])

        bare_agent.set_mode("task")
        assert bare_agent._controller.auto_execute is True
        scopes = bare_agent._authorization_engine.state.session_scopes
        assert len(scopes) == 1
        assert scopes[0].source == "session_authorize"
        assert scopes[0].zone == "project"

        result = bare_agent.chat("write file")
        assert "Done." in result

    def test_user_denies_session_authorize(self, bare_agent, tmp_path, mock_llm_client):
        """No seed_scopes + user denies → auto_execute=False → prepare/contract flow."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)

        bare_agent.set_mode("task")
        assert bare_agent._controller.auto_execute is False
        assert len(bare_agent._authorization_engine.state.session_scopes) == 0

    def test_no_confirm_handler_falls_back(self, bare_agent, tmp_path):
        """No seed_scopes + no confirm_handler → auto_execute=False."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.confirm_handler = None

        bare_agent.set_mode("task")
        assert bare_agent._controller.auto_execute is False

    def test_session_scopes_survive_task_cancel(self, bare_agent, tmp_path, mock_llm_client):
        """Session scopes persist after task cancellation."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.config.seed_scopes = [
            {"scope": "session", "zone": "project", "actions": ["read", "write"],
             "path_prefixes": [str(tmp_path)]}
        ]
        mock_llm_client.set_responses([make_llm_response("Done first.")])

        bare_agent.set_mode("task")
        # First task: execute and complete
        bare_agent.chat("first task")
        assert bare_agent._controller.state.phase == "idle"

        # Session scopes still present
        assert len(bare_agent._authorization_engine.state.session_scopes) == 1

        # Second task: also works with session scopes
        mock_llm_client.set_responses([make_llm_response("Done second.")])
        result = bare_agent.chat("second task")
        assert "Done second." in result

    def test_session_scopes_cleared_on_mode_switch(self, bare_agent, tmp_path):
        """Session scopes cleared when switching back to interactive."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.config.seed_scopes = [
            {"scope": "session", "zone": "project", "actions": ["read", "write"],
             "path_prefixes": [str(tmp_path)]}
        ]

        bare_agent.set_mode("task")
        assert len(bare_agent._authorization_engine.state.session_scopes) == 1

        bare_agent.set_mode("interactive")
        assert len(bare_agent._authorization_engine.state.session_scopes) == 0

    def test_execute_uses_session_scope_for_matching(self, bare_agent, tmp_path):
        """TaskPolicy._decide_execute matches against session_scopes."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.config.seed_scopes = [
            {"scope": "session", "zone": "project", "actions": ["read", "write"],
             "path_prefixes": [str(tmp_path)]}
        ]
        bare_agent.set_mode("task")

        # Manually set state to executing for direct call_tool test
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

        # No task_scopes for T_SESS, but session_scopes cover project
        result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/main.py")})
        assert result == "written"
        assert call_count[0] == 1

    def test_task_scopes_take_priority_over_session(self, bare_agent, tmp_path):
        """Task-specific scopes are checked before session scopes."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.config.seed_scopes = [
            {"scope": "session", "zone": "project", "actions": ["read", "write"],
             "path_prefixes": [str(tmp_path)], "tool_names": ["writer"]}
        ]
        bare_agent.set_mode("task")

        state = bare_agent._controller.state
        state.phase = "executing"
        state.task_id = "T_PRI"
        bare_agent._current_task_id = "T_PRI"

        # Add a task scope with max_uses=1
        bare_agent._authorization_engine.state.task_scopes["T_PRI"] = [
            ApprovalScope(scope="task", zone="project", actions=["write"],
                          path_prefixes=[os.path.join(str(tmp_path), "src/")],
                          max_uses=1)
        ]

        call_count = [0]
        bare_agent.register_tool(
            "writer",
            lambda path="": (call_count.__setitem__(0, call_count[0]+1), "written")[1],
            "write", safety_level=2, path_extractor=lambda a: [a.get("path", "")]
        )

        # First call: matches task_scope (more specific), consumes it
        bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/main.py")})
        task_scope = bare_agent._authorization_engine.state.task_scopes["T_PRI"][0]
        assert task_scope.uses == 1

        # Second call: task_scope exhausted (max_uses=1), falls to session_scope
        bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/other.py")})
        session_scope = bare_agent._authorization_engine.state.session_scopes[0]
        assert session_scope.uses == 1

    def test_auto_execute_controller_handle_turn(self):
        """Controller with auto_execute skips prepare from idle."""
        ctrl = TaskModeController()
        ctrl.auto_execute = True
        action = ctrl.handle_turn("do something")
        assert action.kind == "run_execute"
        assert ctrl.state.phase == "executing"
        assert ctrl.state.confirmed is True

    def test_auto_execute_false_preserves_prepare(self):
        """Controller without auto_execute goes through prepare."""
        ctrl = TaskModeController()
        ctrl.auto_execute = False
        action = ctrl.handle_turn("do something")
        assert action.kind == "run_prepare"
        assert ctrl.state.phase == "preparing"


# ============================================================
# Interactive isolation
# ============================================================

class TestMaxModeStepsExhaustion:
    """Driving loop terminates when MAX_MODE_STEPS is exhausted."""

    def test_max_mode_steps_exhaustion(self, bare_agent, mock_llm_client):
        """Controller that never returns terminal action → loop exhausts and returns error."""

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
        from llamagent.core.authorization import TaskPolicy
        bare_agent._authorization_engine.policy = TaskPolicy(ctrl.state)

        result = bare_agent.chat("infinite task")
        assert "exceeded maximum steps" in result.lower()
        # handle_turn(1) + on_pipeline_done * MAX_MODE_STEPS
        assert ctrl.turn_count == 1 + bare_agent._MAX_MODE_STEPS


class TestInteractiveIsolation:
    def test_chat_and_call_tool_unaffected(self, bare_agent, tmp_path, mock_llm_client):
        mock_llm_client.set_responses([make_llm_response("Hi")])
        assert bare_agent.mode == "interactive"
        assert bare_agent.chat("hi") == "Hi"

        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("t", lambda: "ok", "tool")
        assert bare_agent.call_tool("t", {}) == "ok"
