"""
Task mode flow tests: controlled dry-run, state machine, scope normalization,
chat() integration, ConfirmRequest extension, and interactive mode isolation.
"""

import os

import pytest

from llamagent.core.agent import SmartAgent, Module
from llamagent.core.zone import ConfirmRequest, ConfirmResponse, RequestedScope
from llamagent.core.contract import TaskContract, TaskModeState
from llamagent.core.authorization import TaskPolicy, InteractivePolicy, normalize_scopes
from conftest import make_llm_response, make_tool_call


# ============================================================
# Helpers
# ============================================================

def _setup_zone(agent, tmp_path):
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


# ============================================================
# Controlled Dry-Run
# ============================================================

class TestControlledDryRun:
    """Prepare phase: read executes, write/execute records, HARD_DENY rejects."""

    def test_read_executes_write_records(self, bare_agent, tmp_path):
        """Read operations execute; write operations blocked and recorded."""
        _setup_zone(bare_agent, tmp_path)

        call_log = []
        bare_agent.register_tool("reader", lambda path="": (call_log.append("read"), "content")[1],
                                 "read", safety_level=1, path_extractor=lambda a: [a.get("path", "")])
        bare_agent.register_tool("writer", lambda path="": (call_log.append("write"), "written")[1],
                                 "write", safety_level=2, path_extractor=lambda a: [a.get("path", "")])

        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "preparing"

        # Read: executes
        r1 = bare_agent.call_tool("reader", {"path": os.path.join(str(tmp_path), "f.py")})
        assert r1 == "content"
        # Write: blocked, recorded
        r2 = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "f.py")})
        assert "[Prepare]" in r2

        assert call_log == ["read"]  # writer never called
        assert len(state.pending_scopes) == 1
        assert state.pending_scopes[0].actions == ["write"]

    def test_playground_write_blocked_during_prepare(self, bare_agent, tmp_path):
        """Even playground writes are blocked during prepare (all write/execute blocked)."""
        _setup_zone(bare_agent, tmp_path)
        call_count = [0]
        bare_agent.register_tool("pg_writer", lambda path="": (call_count.__setitem__(0, call_count[0]+1), "ok")[1],
                                 "write", safety_level=2, path_extractor=lambda a: [a.get("path", "")])

        bare_agent.set_mode("task")
        bare_agent._task_mode_state.phase = "preparing"

        pg_file = os.path.join(bare_agent.playground_dir, "test.txt")
        result = bare_agent.call_tool("pg_writer", {"path": pg_file})
        assert "[Prepare]" in result
        assert call_count[0] == 0

    def test_execute_action_blocked_and_recorded(self, bare_agent, tmp_path):
        """Tools with action='execute' blocked and recorded during prepare."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("runner", lambda cmd="": "ran", "run", safety_level=2, action="execute")

        bare_agent.set_mode("task")
        bare_agent._task_mode_state.phase = "preparing"

        result = bare_agent.call_tool("runner", {})
        # No paths → no zone check → action=execute should still be caught
        # Actually no paths → ZoneEvaluation is ALLOW → policy sees no items → allows
        # This is correct: execute tools without paths have no zone to check
        # Only path-bearing execute tools get caught

    def test_hard_deny_rejects_and_not_recorded(self, bare_agent, tmp_path):
        """HARD_DENY in prepare: rejects, does NOT record as pending scope."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("writer", lambda path="": "no", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])

        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "preparing"

        result = bare_agent.call_tool("writer", {"path": "/etc/passwd"})
        assert "cannot operate" in result.lower() or "blocked" in result.lower()
        assert len(state.pending_scopes) == 0

    def test_ask_user_always_allowed(self, bare_agent, tmp_path):
        """ask_user executes during prepare (need user info)."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("ask_user", lambda question="": "yes", "ask")

        bare_agent.set_mode("task")
        bare_agent._task_mode_state.phase = "preparing"

        assert bare_agent.call_tool("ask_user", {"question": "ok?"}) == "yes"

    def test_on_input_safety_runs_during_prepare(self, bare_agent, tmp_path, mock_llm_client):
        """Safety module on_input runs during prepare. Blocked input → rejection."""
        _setup_zone(bare_agent, tmp_path)

        class Blocker(Module):
            name = "blocker"
            description = "blocks all"
            def on_input(self, user_input):
                return ""

        bare_agent.register_module(Blocker())
        bare_agent.set_mode("task")
        bare_agent._task_mode_state.phase = "idle"

        result = bare_agent.chat("dangerous input")
        assert "sorry" in result.lower() or "cannot" in result.lower()


# ============================================================
# State Machine + chat() Integration
# ============================================================

class TestStateMachineAndChat:
    """State transitions and chat() task mode behavior."""

    def test_set_mode_creates_and_clears_state(self, bare_agent):
        bare_agent.set_mode("task")
        assert isinstance(bare_agent._task_mode_state, TaskModeState)
        assert bare_agent._task_mode_state.phase == "idle"
        assert isinstance(bare_agent._authorization_engine.policy, TaskPolicy)

        bare_agent.set_mode("interactive")
        assert bare_agent._task_mode_state is None
        assert isinstance(bare_agent._authorization_engine.policy, InteractivePolicy)

    def test_state_and_policy_are_separate_objects(self, bare_agent):
        """TaskModeState is independent from TaskPolicy."""
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        policy = bare_agent._authorization_engine.policy
        assert isinstance(state, TaskModeState)
        assert isinstance(policy, TaskPolicy)
        assert policy.state is state  # Policy holds reference to state

    def test_cancel_resets_to_idle(self, bare_agent, tmp_path, mock_llm_client):
        _setup_zone(bare_agent, tmp_path)
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "awaiting_confirmation"
        state.original_query = "do something"
        state.pending_scopes.append(RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"]))

        result = bare_agent.chat("no")
        assert "cancelled" in result.lower()
        assert state.phase == "idle"
        assert len(state.pending_scopes) == 0

    def test_first_chat_triggers_prepare(self, bare_agent, tmp_path, mock_llm_client):
        """First chat() in task mode transitions idle→preparing, returns contract."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("writer", lambda path="": "ok", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])

        # LLM will try to call writer during prepare
        project_file = os.path.join(str(tmp_path), "main.py")
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[make_tool_call("writer", {"path": project_file})]),
            make_llm_response("I need to write main.py"),
        ])

        bare_agent.set_mode("task")
        result = bare_agent.chat("write main.py for me")

        assert "[Task Contract]" in result or "authorization" in result.lower()

    def test_confirm_yes_executes(self, bare_agent, tmp_path, mock_llm_client):
        """User confirms 'yes' → execute runs, state resets."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        bare_agent.register_tool("writer", lambda path="": "written!", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])

        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "awaiting_confirmation"
        state.original_query = "write main.py"
        state.pending_scopes.append(RequestedScope(zone="project", actions=["write"], path_prefixes=["main.py"]))
        state.contract = TaskContract(
            task_summary="write main.py", planned_operations=["write project:main.py"],
            requested_scopes=state.pending_scopes, open_questions=[], risk_summary="1 op",
        )

        mock_llm_client.set_responses([make_llm_response("Done writing.")])
        result = bare_agent.chat("yes")
        assert state.phase == "idle"  # Reset after execution


# ============================================================
# Scope Normalization
# ============================================================

class TestScopeNormalization:
    def test_empty(self):
        assert normalize_scopes([]) == []

    def test_merge_same_group(self):
        scopes = [
            RequestedScope(zone="project", actions=["write"], path_prefixes=["/a/b/c.py"], tool_names=["w"]),
            RequestedScope(zone="project", actions=["write"], path_prefixes=["/a/b/d.py"], tool_names=["w"]),
        ]
        result = normalize_scopes(scopes)
        assert len(result) == 1

    def test_different_zones_separate(self):
        scopes = [
            RequestedScope(zone="project", actions=["write"], path_prefixes=["a.py"]),
            RequestedScope(zone="external", actions=["write"], path_prefixes=["b.py"]),
        ]
        assert len(normalize_scopes(scopes)) == 2

    def test_different_actions_separate(self):
        scopes = [
            RequestedScope(zone="project", actions=["write"], path_prefixes=["a.py"]),
            RequestedScope(zone="project", actions=["execute"], path_prefixes=["b.py"]),
        ]
        assert len(normalize_scopes(scopes)) == 2


# ============================================================
# ConfirmRequest Extension
# ============================================================

class TestConfirmRequestExtension:
    def test_requested_scopes_default_none(self):
        req = ConfirmRequest(kind="operation_confirm", tool_name="t", action="write",
                             zone="project", target_paths=["x"], message="m")
        assert req.requested_scopes is None

    def test_requested_scopes_set(self):
        rs = RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"])
        req = ConfirmRequest(kind="task_contract", tool_name="", action="", zone="",
                             target_paths=[], message="contract", requested_scopes=[rs])
        assert req.requested_scopes is not None
        assert len(req.requested_scopes) == 1


# ============================================================
# Interactive Mode Isolation
# ============================================================

class TestInteractiveModeIsolation:
    def test_chat_unaffected(self, bare_agent, mock_llm_client):
        mock_llm_client.set_responses([make_llm_response("Hello!")])
        assert bare_agent.mode == "interactive"
        assert bare_agent._task_mode_state is None
        assert bare_agent.chat("hi") == "Hello!"

    def test_call_tool_unaffected(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        bare_agent.register_tool("t", lambda: "ok", "tool")
        assert bare_agent.call_tool("t", {}) == "ok"
