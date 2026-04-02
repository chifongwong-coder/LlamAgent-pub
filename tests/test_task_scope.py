"""
Task Scope Authorization tests (v1.9.2).

Covers:
- Scope matching logic (_matches_any_scope via call_tool integration)
- task_id management (TaskModeState.task_id, get_active_task_id)
- Scope lifecycle (cleanup on complete, cancel, mode switch)
- ConfirmResponse.approved_scopes (backward compat, dataclass fields)
- Interactive mode isolation (no scope matching)
"""

import os
from uuid import uuid4

import pytest

from llamagent.core.agent import SmartAgent
from llamagent.core.zone import (
    ApprovalScope,
    ConfirmRequest,
    ConfirmResponse,
    RequestedScope,
    ZoneDecisionItem,
    ZoneVerdict,
)
from llamagent.core.contract import TaskContract, TaskModeState
from llamagent.core.authorization import (
    AuthorizationEngine,
    AuthorizationState,
    InteractivePolicy,
    TaskPolicy,
    _matches_any_scope,
)
from conftest import make_llm_response, make_tool_call


# ============================================================
# Helpers
# ============================================================

def _setup_zone(agent, tmp_path):
    """Set project_dir and playground_dir to real tmp_path directories."""
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


def _setup_task_execute(agent, tmp_path, scopes):
    """
    Put agent into task mode executing phase with given ApprovalScopes
    registered under the active task_id.
    """
    _setup_zone(agent, tmp_path)
    agent.set_mode("task")
    state = agent._task_mode_state
    state.phase = "executing"
    state.task_id = uuid4().hex

    task_id = agent.get_active_task_id()
    engine = agent._authorization_engine
    engine.state.task_scopes[task_id] = list(scopes)
    return task_id


# ============================================================
# Scope Matching (plan section V)
# ============================================================

class TestScopeMatching:
    """Scope matching during execute phase of task mode."""

    def test_tool_within_scope_auto_approved(self, bare_agent, tmp_path):
        """Tool call whose zone+action+path_prefix all match -> auto-approved, no confirm."""
        project_dir = str(tmp_path)
        target = os.path.join(project_dir, "src", "main.py")
        os.makedirs(os.path.join(project_dir, "src"), exist_ok=True)

        scope = ApprovalScope(
            scope="task",
            zone="project",
            actions=["write"],
            path_prefixes=[os.path.join(project_dir, "src/")],
        )
        task_id = _setup_task_execute(bare_agent, tmp_path, [scope])

        executed = []
        bare_agent.register_tool(
            "writer",
            lambda path="": (executed.append(path), "written")[1],
            "write file",
            safety_level=2,
            path_extractor=lambda a: [a.get("path", "")],
        )

        # confirm_handler should NOT be called (scope auto-approves)
        def _no_confirm(req):
            raise AssertionError("confirm_handler should not be called for in-scope tool")

        bare_agent.confirm_handler = _no_confirm

        result = bare_agent.call_tool("writer", {"path": target})
        assert result == "written"
        assert target in executed

    def test_tool_outside_scope_path_falls_back_to_confirm(self, bare_agent, tmp_path):
        """Tool call where path does not match any scope prefix -> falls back to confirm."""
        project_dir = str(tmp_path)
        target = os.path.join(project_dir, "docs", "readme.md")
        os.makedirs(os.path.join(project_dir, "docs"), exist_ok=True)

        # Scope only covers src/
        scope = ApprovalScope(
            scope="task",
            zone="project",
            actions=["write"],
            path_prefixes=[os.path.join(project_dir, "src/")],
        )
        _setup_task_execute(bare_agent, tmp_path, [scope])

        confirm_called = []
        bare_agent.register_tool(
            "writer",
            lambda path="": "written",
            "write file",
            safety_level=2,
            path_extractor=lambda a: [a.get("path", "")],
        )

        def _confirm(req):
            confirm_called.append(req)
            return ConfirmResponse(allow=True)

        bare_agent.confirm_handler = _confirm

        result = bare_agent.call_tool("writer", {"path": target})
        # Confirm was invoked because path is outside scope
        assert len(confirm_called) == 1
        assert result == "written"

    def test_tool_outside_scope_action_falls_back_to_confirm(self, bare_agent, tmp_path):
        """Tool call with different action than scope -> falls back to confirm."""
        project_dir = str(tmp_path)
        target = os.path.join(project_dir, "src", "run.sh")
        os.makedirs(os.path.join(project_dir, "src"), exist_ok=True)

        # Scope only covers 'write', not 'execute'
        scope = ApprovalScope(
            scope="task",
            zone="project",
            actions=["write"],
            path_prefixes=[os.path.join(project_dir, "src/")],
        )
        _setup_task_execute(bare_agent, tmp_path, [scope])

        confirm_called = []
        bare_agent.register_tool(
            "executor",
            lambda path="": "executed",
            "run script",
            safety_level=2,
            action="execute",
            path_extractor=lambda a: [a.get("path", "")],
        )

        def _confirm(req):
            confirm_called.append(req)
            return ConfirmResponse(allow=True)

        bare_agent.confirm_handler = _confirm

        result = bare_agent.call_tool("executor", {"path": target})
        assert len(confirm_called) == 1

    def test_hard_deny_blocked_regardless_of_scope(self, bare_agent, tmp_path):
        """HARD_DENY is always blocked even if a scope would otherwise match."""
        scope = ApprovalScope(
            scope="task",
            zone="external",
            actions=["write"],
            path_prefixes=["/etc/"],
        )
        _setup_task_execute(bare_agent, tmp_path, [scope])

        bare_agent.register_tool(
            "writer",
            lambda path="": "written",
            "write file",
            safety_level=2,
            path_extractor=lambda a: [a.get("path", "")],
        )
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)

        result = bare_agent.call_tool("writer", {"path": "/etc/passwd"})
        assert "blocked" in result.lower() or "cannot" in result.lower()

    def test_multiple_scopes_one_match_sufficient(self, bare_agent, tmp_path):
        """With multiple scopes, matching any one is enough to auto-approve."""
        project_dir = str(tmp_path)
        target = os.path.join(project_dir, "docs", "guide.md")
        os.makedirs(os.path.join(project_dir, "docs"), exist_ok=True)

        scope_src = ApprovalScope(
            scope="task",
            zone="project",
            actions=["write"],
            path_prefixes=[os.path.join(project_dir, "src/")],
        )
        scope_docs = ApprovalScope(
            scope="task",
            zone="project",
            actions=["write"],
            path_prefixes=[os.path.join(project_dir, "docs/")],
        )
        _setup_task_execute(bare_agent, tmp_path, [scope_src, scope_docs])

        bare_agent.register_tool(
            "writer",
            lambda path="": "written",
            "write file",
            safety_level=2,
            path_extractor=lambda a: [a.get("path", "")],
        )

        def _no_confirm(req):
            raise AssertionError("Should be auto-approved by scope_docs")

        bare_agent.confirm_handler = _no_confirm

        result = bare_agent.call_tool("writer", {"path": target})
        assert result == "written"


# ============================================================
# task_id Management (decisions section II-IV)
# ============================================================

class TestTaskIdManagement:
    """task_id generation and get_active_task_id() behavior."""

    def test_task_id_generated_on_idle_to_preparing(self, bare_agent, tmp_path, mock_llm_client):
        """TaskModeState.task_id is generated when phase transitions idle -> preparing."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        assert state.task_id == ""  # Empty in idle

        # Register a write tool so prepare phase records a pending scope
        # and the state stays at awaiting_confirmation (not resetting to idle)
        project_file = os.path.join(str(tmp_path), "main.py")
        bare_agent.register_tool(
            "writer", lambda path="": "ok", "write file",
            safety_level=2, path_extractor=lambda a: [a.get("path", "")],
        )

        # LLM calls writer during prepare -> scope recorded -> contract generated
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[make_tool_call("writer", {"path": project_file})]),
            make_llm_response("I need to write main.py"),
        ])
        bare_agent.chat("write main.py")

        # State should be at awaiting_confirmation with task_id set
        assert state.phase == "awaiting_confirmation"
        assert state.task_id != ""
        assert len(state.task_id) > 0

    def test_get_active_task_id_returns_task_mode_state_id(self, bare_agent):
        """In task mode with task_id set, get_active_task_id returns TaskModeState.task_id."""
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.task_id = "task-abc-123"

        assert bare_agent.get_active_task_id() == "task-abc-123"

    def test_get_active_task_id_returns_current_task_id_outside_task_mode(self, bare_agent):
        """Outside task mode, get_active_task_id falls back to _current_task_id."""
        assert bare_agent.mode == "interactive"
        bare_agent._current_task_id = "planreact-xyz"

        assert bare_agent.get_active_task_id() == "planreact-xyz"

    def test_scope_written_with_task_id_from_get_active_task_id(self, bare_agent, tmp_path):
        """Scopes are keyed by get_active_task_id() in task_scopes."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.task_id = "my-task-42"

        engine = bare_agent._authorization_engine
        task_id = bare_agent.get_active_task_id()
        assert task_id == "my-task-42"

        scope = ApprovalScope(
            scope="task", zone="project", actions=["write"],
            path_prefixes=["src/"],
        )
        engine.state.task_scopes.setdefault(task_id, []).append(scope)

        assert "my-task-42" in engine.state.task_scopes
        assert len(engine.state.task_scopes["my-task-42"]) == 1


# ============================================================
# Scope Lifecycle (plan section VI)
# ============================================================

class TestScopeLifecycle:
    """Scope cleanup on completion, cancellation, and mode switch."""

    def test_scopes_cleaned_after_execution(self, bare_agent, tmp_path, mock_llm_client):
        """After execution completes, task_scopes[task_id] is cleaned up."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)

        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "awaiting_confirmation"
        state.task_id = uuid4().hex
        state.original_query = "write code"
        state.pending_scopes.append(
            RequestedScope(zone="project", actions=["write"],
                           path_prefixes=[str(tmp_path) + "/"])
        )
        state.contract = TaskContract(
            task_summary="write code",
            planned_operations=["write project:main.py"],
            requested_scopes=state.pending_scopes,
            open_questions=[],
            risk_summary="1 op",
        )

        task_id = state.task_id
        engine = bare_agent._authorization_engine
        # Pre-populate scopes (simulating what _write_task_scopes does)
        engine.state.task_scopes[task_id] = [
            ApprovalScope(scope="task", zone="project", actions=["write"],
                          path_prefixes=[str(tmp_path) + "/"])
        ]

        mock_llm_client.set_responses([make_llm_response("Done.")])
        bare_agent.chat("yes")

        # After execution completes, scopes should be cleaned
        assert task_id not in engine.state.task_scopes

    def test_scopes_cleaned_after_cancel(self, bare_agent, tmp_path, mock_llm_client):
        """After user cancels, task_scopes[task_id] is cleaned up."""
        _setup_zone(bare_agent, tmp_path)

        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "awaiting_confirmation"
        state.task_id = uuid4().hex
        state.original_query = "write code"
        state.pending_scopes.append(
            RequestedScope(zone="project", actions=["write"],
                           path_prefixes=[str(tmp_path) + "/"])
        )
        state.contract = TaskContract(
            task_summary="write code",
            planned_operations=["write project:main.py"],
            requested_scopes=state.pending_scopes,
            open_questions=[],
            risk_summary="1 op",
        )

        task_id = state.task_id
        engine = bare_agent._authorization_engine
        engine.state.task_scopes[task_id] = [
            ApprovalScope(scope="task", zone="project", actions=["write"],
                          path_prefixes=[str(tmp_path) + "/"])
        ]

        result = bare_agent.chat("no")
        assert "cancelled" in result.lower()
        assert task_id not in engine.state.task_scopes

    def test_set_mode_interactive_clears_all_task_scopes(self, bare_agent):
        """Switching to interactive mode clears all task_scopes.

        Spec (plan section VI): 'mode switch to interactive -> all task_scopes cleared'.
        """
        bare_agent.set_mode("task")
        engine = bare_agent._authorization_engine

        # Populate multiple task scopes
        engine.state.task_scopes["task-1"] = [
            ApprovalScope(scope="task", zone="project", actions=["write"],
                          path_prefixes=["a/"])
        ]
        engine.state.task_scopes["task-2"] = [
            ApprovalScope(scope="task", zone="project", actions=["execute"],
                          path_prefixes=["b/"])
        ]
        assert len(engine.state.task_scopes) == 2

        bare_agent.set_mode("interactive")
        assert len(engine.state.task_scopes) == 0


# ============================================================
# ConfirmResponse.approved_scopes (decisions section V-VI)
# ============================================================

class TestConfirmResponseApprovedScopes:
    """ConfirmResponse.approved_scopes backward compat and structure."""

    def test_default_is_none(self):
        """ConfirmResponse.approved_scopes defaults to None (backward compat)."""
        resp = ConfirmResponse(allow=True)
        assert resp.approved_scopes is None

    def test_approval_scope_has_required_fields(self):
        """ApprovalScope dataclass has all required fields per spec."""
        scope = ApprovalScope(
            scope="task",
            zone="project",
            actions=["write", "execute"],
            path_prefixes=["src/", "docs/"],
            tool_names=["writer", "runner"],
        )
        assert scope.scope == "task"
        assert scope.zone == "project"
        assert scope.actions == ["write", "execute"]
        assert scope.path_prefixes == ["src/", "docs/"]
        assert scope.tool_names == ["writer", "runner"]

    def test_approval_scope_tool_names_defaults_none(self):
        """ApprovalScope.tool_names defaults to None (optional field)."""
        scope = ApprovalScope(
            scope="task", zone="project", actions=["write"],
            path_prefixes=["src/"],
        )
        assert scope.tool_names is None


# ============================================================
# Interactive Mode Isolation
# ============================================================

class TestInteractiveIsolation:
    """Interactive mode call_tool never uses scope matching."""

    def test_interactive_call_tool_ignores_scopes(self, bare_agent, tmp_path):
        """In interactive mode, even if task_scopes exist, scope matching is not used."""
        _setup_zone(bare_agent, tmp_path)
        assert bare_agent.mode == "interactive"

        engine = bare_agent._authorization_engine
        # Artificially stuff scopes into state (should be ignored)
        engine.state.task_scopes["stale-task"] = [
            ApprovalScope(scope="task", zone="project", actions=["write"],
                          path_prefixes=[str(tmp_path) + "/"])
        ]

        confirm_called = []
        bare_agent.register_tool(
            "writer",
            lambda path="": "written",
            "write file",
            safety_level=2,
            path_extractor=lambda a: [a.get("path", "")],
        )

        def _confirm(req):
            confirm_called.append(req)
            return ConfirmResponse(allow=True)

        bare_agent.confirm_handler = _confirm

        target = os.path.join(str(tmp_path), "file.py")
        result = bare_agent.call_tool("writer", {"path": target})
        # InteractivePolicy uses per-item confirm, not scope matching
        assert len(confirm_called) == 1
        assert result == "written"
