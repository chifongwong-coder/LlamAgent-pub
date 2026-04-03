"""
Governance and audit tests for v1.9.4: scope lifecycle, path matching,
two-phase consume, audit events, authorization_status, and governance fields.
"""

import os
import time

import pytest

from llamagent.core.zone import (
    ApprovalScope,
    ConfirmResponse,
    ZoneDecisionItem,
    ZoneVerdict,
)
from llamagent.core.authorization import _find_matching_scope, AuthorizationResult
from llamagent.core.hooks import HookEvent


# ============================================================
# Helpers
# ============================================================

def _setup_zone(agent, tmp_path):
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


def _reg(agent, name, sl=1, result="ok", action=None):
    agent.register_tool(
        name, lambda **kw: result, f"tool {name}",
        safety_level=sl, action=action,
        path_extractor=lambda args: (
            [args["path"]] if isinstance(args.get("path"), str)
            else args.get("paths", [])
        ),
    )


def _make_item(path, zone="project", action="write", verdict=ZoneVerdict.CONFIRMABLE):
    return ZoneDecisionItem(
        path=path, verdict=verdict, zone=zone, action=action, message=None,
    )


def _make_scope(zone="project", actions=None, path_prefixes=None, tool_names=None,
                expires_at=None, max_uses=None, uses=0, source="contract"):
    return ApprovalScope(
        scope="task",
        zone=zone,
        actions=actions or ["write"],
        path_prefixes=path_prefixes or [],
        tool_names=tool_names,
        created_at=time.time(),
        expires_at=expires_at,
        max_uses=max_uses,
        uses=uses,
        source=source,
    )


# ============================================================
# _find_matching_scope pure function (plan section II.1)
# ============================================================

class TestFindMatchingScopePure:
    """Pure function tests: no side effects, correct match/skip logic."""

    def test_expired_scope_not_matched(self):
        """Scope with expires_at in the past is skipped."""
        scope = _make_scope(
            path_prefixes=["/proj/src/"],
            expires_at=time.time() - 100,  # expired 100s ago
        )
        item = _make_item("/proj/src/main.py")
        result = _find_matching_scope(item, "write_file", [scope])
        assert result is None

    def test_used_up_scope_not_matched(self):
        """Scope with uses >= max_uses is skipped."""
        scope = _make_scope(
            path_prefixes=["/proj/src/"],
            max_uses=3,
            uses=3,
        )
        item = _make_item("/proj/src/main.py")
        result = _find_matching_scope(item, "write_file", [scope])
        assert result is None

    def test_valid_scope_matched(self):
        """Scope within limits (not expired, uses < max_uses) matches."""
        scope = _make_scope(
            path_prefixes=["/proj/src/"],
            expires_at=time.time() + 3600,
            max_uses=5,
            uses=2,
        )
        item = _make_item("/proj/src/main.py")
        result = _find_matching_scope(item, "write_file", [scope])
        assert result is scope

    def test_pure_no_uses_increment(self):
        """Calling _find_matching_scope does NOT change scope.uses."""
        scope = _make_scope(
            path_prefixes=["/proj/src/"],
            max_uses=10,
            uses=3,
        )
        item = _make_item("/proj/src/main.py")
        _find_matching_scope(item, "write_file", [scope])
        assert scope.uses == 3  # unchanged


# ============================================================
# Path matching fix (plan section II.4)
# ============================================================

class TestPathMatchingFix:
    """Normalized subtree path matching, not naive startswith."""

    def test_prefix_does_not_match_sibling_directory(self):
        """'src/' must NOT match 'src_backup/file.py' (the old startswith bug)."""
        scope = _make_scope(path_prefixes=["/proj/src/"])
        item = _make_item("/proj/src_backup/file.py")
        result = _find_matching_scope(item, "write_file", [scope])
        assert result is None

    def test_prefix_matches_subtree(self):
        """'src/' DOES match 'src/main.py'."""
        scope = _make_scope(path_prefixes=["/proj/src/"])
        item = _make_item("/proj/src/main.py")
        result = _find_matching_scope(item, "write_file", [scope])
        assert result is scope

    def test_exact_path_match(self):
        """Exact path (no trailing slash) matches itself."""
        scope = _make_scope(path_prefixes=["/proj/src/main.py"])
        item = _make_item("/proj/src/main.py")
        result = _find_matching_scope(item, "write_file", [scope])
        assert result is scope


# ============================================================
# Two-phase consume (plan section II.2)
# ============================================================

class TestTwoPhaseConsume:
    """Scope uses incremented only after ALL items pass."""

    def test_task_execute_increments_after_all_pass(self, bare_agent, tmp_path):
        """When two items both match scopes, uses incremented only after both pass."""
        _setup_zone(bare_agent, tmp_path)
        _reg(bare_agent, "multi_write", sl=2)

        # Enter task mode with pre-loaded scopes
        from llamagent.core.contract import TaskModeState
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "executing"
        state.task_id = "test-task-1"
        bare_agent._current_task_id = "test-task-1"

        scope = _make_scope(
            path_prefixes=[str(tmp_path) + "/"],
            max_uses=10,
            uses=0,
        )
        bare_agent._authorization_engine.state.task_scopes["test-task-1"] = [scope]

        p1 = os.path.join(str(tmp_path), "a.py")
        p2 = os.path.join(str(tmp_path), "b.py")
        result = bare_agent.call_tool("multi_write", {"paths": [p1, p2]})
        assert result == "ok"
        # Both items consumed: uses = 2
        assert scope.uses == 2

    def test_second_item_denied_no_increment(self, bare_agent, tmp_path):
        """If second item is denied (no matching scope), first item's scope.uses NOT incremented."""
        _setup_zone(bare_agent, tmp_path)
        _reg(bare_agent, "multi_write", sl=2)

        from llamagent.core.contract import TaskModeState
        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "executing"
        state.task_id = "test-task-2"
        bare_agent._current_task_id = "test-task-2"

        # Scope only covers src/ subdirectory
        scope = _make_scope(
            path_prefixes=[os.path.join(str(tmp_path), "src") + "/"],
            max_uses=10,
            uses=0,
        )
        bare_agent._authorization_engine.state.task_scopes["test-task-2"] = [scope]

        p_in = os.path.join(str(tmp_path), "src", "a.py")     # covered
        p_out = os.path.join(str(tmp_path), "other", "b.py")  # NOT covered

        # No confirm handler -> deny on fallback
        bare_agent.confirm_handler = None
        result = bare_agent.call_tool("multi_write", {"paths": [p_in, p_out]})
        assert "denied" in result.lower()
        # First item's scope.uses must NOT have been incremented
        assert scope.uses == 0


# ============================================================
# Audit events via AuthorizationResult (plan section IV)
# ============================================================

class TestAuditEvents:
    """Hook events emitted through AuthorizationResult during call_tool."""

    def test_scope_used_event_on_match(self, bare_agent, tmp_path):
        """SCOPE_USED event emitted when scope matches during execute."""
        _setup_zone(bare_agent, tmp_path)
        _reg(bare_agent, "w", sl=2)

        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "executing"
        state.task_id = "task-audit-1"
        bare_agent._current_task_id = "task-audit-1"

        scope = _make_scope(path_prefixes=[str(tmp_path) + "/"])
        bare_agent._authorization_engine.state.task_scopes["task-audit-1"] = [scope]

        events = []
        bare_agent.register_hook(
            HookEvent.SCOPE_USED,
            lambda ctx: events.append(ctx),
        )

        path = os.path.join(str(tmp_path), "f.py")
        bare_agent.call_tool("w", {"path": path})
        assert len(events) == 1
        assert events[0].data["tool_name"] == "w"
        assert events[0].data["scope"] is scope

    def test_scope_denied_event_when_no_match(self, bare_agent, tmp_path):
        """SCOPE_DENIED event emitted when no scope matches."""
        _setup_zone(bare_agent, tmp_path)
        _reg(bare_agent, "w", sl=2)

        bare_agent.set_mode("task")
        state = bare_agent._task_mode_state
        state.phase = "executing"
        state.task_id = "task-audit-2"
        bare_agent._current_task_id = "task-audit-2"

        # No scopes loaded -> scope_denied
        bare_agent._authorization_engine.state.task_scopes["task-audit-2"] = []
        bare_agent.confirm_handler = None  # deny on fallback

        events = []
        bare_agent.register_hook(
            HookEvent.SCOPE_DENIED,
            lambda ctx: events.append(ctx),
        )

        path = os.path.join(str(tmp_path), "f.py")
        result = bare_agent.call_tool("w", {"path": path})
        assert "denied" in result.lower()
        assert len(events) == 1
        assert events[0].data["tool_name"] == "w"
        assert events[0].data["zone"] == "project"

    def test_scope_issued_on_continuous_mode(self, bare_agent, tmp_path):
        """SCOPE_ISSUED emitted when set_mode('continuous') loads seed scopes."""
        _setup_zone(bare_agent, tmp_path)

        bare_agent.config.seed_scopes = [
            {
                "scope": "session",
                "zone": "project",
                "actions": ["write"],
                "path_prefixes": [str(tmp_path) + "/src/"],
            }
        ]

        events = []
        bare_agent.register_hook(
            HookEvent.SCOPE_ISSUED,
            lambda ctx: events.append(ctx),
        )

        bare_agent.set_mode("continuous")
        assert len(events) == 1
        assert events[0].data["scope"].source == "seed"
        assert events[0].data["task_id"] is None

    def test_scope_revoked_on_mode_switch_to_interactive(self, bare_agent, tmp_path):
        """SCOPE_REVOKED emitted when set_mode('interactive') clears scopes."""
        _setup_zone(bare_agent, tmp_path)

        bare_agent.config.seed_scopes = [
            {
                "scope": "session",
                "zone": "project",
                "actions": ["write"],
                "path_prefixes": [str(tmp_path) + "/"],
            }
        ]

        bare_agent.set_mode("continuous")
        # Now session_scopes are populated; switch back to interactive
        revoked = []
        bare_agent.register_hook(
            HookEvent.SCOPE_REVOKED,
            lambda ctx: revoked.append(ctx),
        )

        bare_agent.set_mode("interactive")
        assert len(revoked) == 1
        assert revoked[0].data["reason"] == "mode_switch"
        assert revoked[0].data["scope"].source == "seed"


# ============================================================
# authorization_status (plan section V)
# ============================================================

class TestAuthorizationStatus:
    """authorization_status() returns current snapshot with mode, scopes, uses, source."""

    def test_status_returns_mode_and_scopes(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)

        bare_agent.config.seed_scopes = [
            {
                "scope": "session",
                "zone": "project",
                "actions": ["read", "write"],
                "path_prefixes": [str(tmp_path) + "/"],
            }
        ]

        bare_agent.set_mode("continuous")
        status = bare_agent.authorization_status()

        assert status["mode"] == "continuous"
        assert isinstance(status["task_scopes"], dict)
        assert isinstance(status["session_scopes"], list)
        assert len(status["session_scopes"]) == 1

        s = status["session_scopes"][0]
        assert s["source"] == "seed"
        assert s["uses"] == 0
        assert s["zone"] == "project"
        assert "read" in s["actions"]
        assert "write" in s["actions"]


# ============================================================
# ApprovalScope governance fields (plan section III)
# ============================================================

class TestApprovalScopeGovernanceFields:
    """Seed scopes carry source='seed' and created_at is set."""

    def test_seed_scope_source(self, bare_agent, tmp_path):
        """Seed scopes loaded via set_mode('continuous') have source='seed'."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.config.seed_scopes = [
            {
                "scope": "session",
                "zone": "project",
                "actions": ["write"],
                "path_prefixes": [str(tmp_path) + "/"],
            }
        ]

        bare_agent.set_mode("continuous")
        scopes = bare_agent._authorization_engine.state.session_scopes
        assert len(scopes) == 1
        assert scopes[0].source == "seed"

    def test_seed_scope_created_at(self, bare_agent, tmp_path):
        """Seed scopes have created_at set to a recent timestamp."""
        _setup_zone(bare_agent, tmp_path)
        before = time.time()
        bare_agent.config.seed_scopes = [
            {
                "scope": "session",
                "zone": "project",
                "actions": ["write"],
                "path_prefixes": [str(tmp_path) + "/"],
            }
        ]

        bare_agent.set_mode("continuous")
        scopes = bare_agent._authorization_engine.state.session_scopes
        after = time.time()
        assert scopes[0].created_at is not None
        assert before <= scopes[0].created_at <= after
