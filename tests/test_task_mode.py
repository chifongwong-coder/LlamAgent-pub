"""
Task mode tests: controlled dry-run, state machine, scope normalization,
scope matching, scope lifecycle, task_id management.
"""

import os

import pytest

from llamagent.core.agent import LlamAgent, Module
from llamagent.core.zone import ConfirmRequest, ConfirmResponse, RequestedScope, ApprovalScope
from llamagent.core.contract import TaskContract, TaskModeState
from llamagent.core.authorization import TaskPolicy, InteractivePolicy, normalize_scopes
from conftest import make_llm_response, make_tool_call


def _setup_zone(agent, tmp_path):
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


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
        state = bare_agent._task_mode_state
        state.phase = "preparing"

        assert bare_agent.call_tool("reader", {"path": os.path.join(str(tmp_path), "f.py")}) == "content"
        result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")})
        assert "[Prepare]" in result
        pg_result = bare_agent.call_tool("writer", {"path": os.path.join(bare_agent.playground_dir, "f.txt")})
        assert "[Prepare]" in pg_result

        assert call_log == ["r"]  # writer never called
        assert len(state.pending_scopes) == 2

    def test_hard_deny_and_ask_user(self, bare_agent, tmp_path):
        """HARD_DENY rejects (not recorded). ask_user always allowed."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("writer", lambda path="": "no", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])
        bare_agent.register_tool("ask_user", lambda question="": "yes", "ask")

        bare_agent.set_mode("task")
        bare_agent._task_mode_state.phase = "preparing"

        result = bare_agent.call_tool("writer", {"path": "/etc/passwd"})
        assert "blocked" in result.lower() or "cannot" in result.lower()
        assert len(bare_agent._task_mode_state.pending_scopes) == 0

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


class TestStateMachineAndScopes:
    """State transitions, scope matching, task_id, scope lifecycle."""

    def test_mode_switching(self, bare_agent):
        bare_agent.set_mode("task")
        assert isinstance(bare_agent._task_mode_state, TaskModeState)
        assert isinstance(bare_agent._authorization_engine.policy, TaskPolicy)
        assert bare_agent._authorization_engine.policy.state is bare_agent._task_mode_state

        bare_agent.set_mode("interactive")
        assert bare_agent._task_mode_state is None
        assert isinstance(bare_agent._authorization_engine.policy, InteractivePolicy)

    def test_task_id_and_get_active(self, bare_agent):
        bare_agent.set_mode("task")
        bare_agent._task_mode_state.task_id = "T123"
        assert bare_agent.get_active_task_id() == "T123"

        bare_agent.set_mode("interactive")
        assert bare_agent.get_active_task_id() is None
        bare_agent._current_task_id = "P456"
        assert bare_agent.get_active_task_id() == "P456"

    def test_cancel_resets_state_and_scopes(self, bare_agent, tmp_path, mock_llm_client):
        _setup_zone(bare_agent, tmp_path)
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "awaiting_confirmation"
        state.task_id = "T1"
        state.original_query = "do"
        state.pending_scopes.append(RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"]))
        bare_agent._authorization_engine.state.task_scopes["T1"] = [
            ApprovalScope(scope="task", zone="project", actions=["write"], path_prefixes=["src/"])
        ]

        result = bare_agent.chat("no")
        assert "cancelled" in result.lower()
        assert state.phase == "idle"
        assert "T1" not in bare_agent._authorization_engine.state.task_scopes

    def test_first_chat_triggers_prepare_and_confirm_executes(self, bare_agent, tmp_path, mock_llm_client):
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)

        project_file = os.path.join(str(tmp_path), "main.py")
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[make_tool_call("writer", {"path": project_file})]),
            make_llm_response("I need to write main.py"),
        ])

        bare_agent.set_mode("task")
        result = bare_agent.chat("write main.py")
        assert "[Task Contract]" in result or "authorization" in result.lower()

        mock_llm_client.set_responses([make_llm_response("Done.")])
        bare_agent.chat("yes")
        assert bare_agent._task_mode_state.phase == "idle"

    def test_scope_matching_auto_approves(self, bare_agent, tmp_path):
        """Tool within approved scope auto-approved; outside falls back to confirm."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
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

        # Within scope — auto-approved
        assert bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/main.py")}) == "written"
        assert call_count[0] == 1

        # Outside scope — falls back to confirm
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
        result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "docs/readme.md")})
        assert "denied" in result.lower()

    def test_scope_lifecycle_cleaned_after_execute(self, bare_agent, tmp_path, mock_llm_client):
        _setup_zone(bare_agent, tmp_path)
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "awaiting_confirmation"
        state.task_id = "T1"
        state.original_query = "do"
        state.contract = TaskContract(
            task_summary="do", planned_operations=["write"],
            requested_scopes=[RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"])],
            open_questions=[], risk_summary="1 op",
        )
        mock_llm_client.set_responses([make_llm_response("Done.")])
        bare_agent.chat("yes")
        assert "T1" not in bare_agent._authorization_engine.state.task_scopes


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


class TestInteractiveIsolation:
    def test_chat_and_call_tool_unaffected(self, bare_agent, tmp_path, mock_llm_client):
        mock_llm_client.set_responses([make_llm_response("Hi")])
        assert bare_agent.mode == "interactive"
        assert bare_agent.chat("hi") == "Hi"

        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("t", lambda: "ok", "tool")
        assert bare_agent.call_tool("t", {}) == "ok"
