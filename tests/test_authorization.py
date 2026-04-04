"""
Authorization system tests: zone equivalence, multi-path, action inference,
confirm structure, scope governance, audit events, continuous mode, path fix.
"""

import os
import time

import pytest

from llamagent.core.agent import LlamAgent
from llamagent.core.zone import ConfirmRequest, ConfirmResponse, RequestedScope
from llamagent.core.authorization import ApprovalScope, infer_action, _find_matching_scope, _path_in_prefixes, AuthorizationResult
from llamagent.core.hooks import HookEvent, HookResult
from conftest import make_llm_response


def _setup_zone(agent, tmp_path):
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


def _reg(agent, name, sl=1, result="ok", action=None):
    agent.register_tool(name, lambda **kw: result, f"tool {name}", safety_level=sl, action=action,
                        path_extractor=lambda args: [args["path"]] if isinstance(args.get("path"), str) else args.get("paths", []))


# ============================================================
# Zone Equivalence
# ============================================================

class TestZoneEquivalence:
    def test_playground_no_paths_allow(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        _reg(bare_agent, "t", sl=2)
        assert bare_agent.call_tool("t", {"path": os.path.join(bare_agent.playground_dir, "f.txt")}) == "ok"
        bare_agent.register_tool("nop", lambda: "ok", "nop", safety_level=2)
        assert bare_agent.call_tool("nop", {}) == "ok"

    def test_project_and_external_matrix(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        _reg(bare_agent, "r", sl=1)
        _reg(bare_agent, "w", sl=2)
        p = os.path.join(str(tmp_path), "f.py")

        assert bare_agent.call_tool("r", {"path": p}) == "ok"
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        assert bare_agent.call_tool("w", {"path": p}) == "ok"
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
        assert "denied" in bare_agent.call_tool("w", {"path": p}).lower()
        bare_agent.confirm_handler = None
        assert "denied" in bare_agent.call_tool("w", {"path": p}).lower()

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        assert "cannot" in bare_agent.call_tool("w", {"path": "/etc/passwd"}).lower() or "not allowed" in bare_agent.call_tool("w", {"path": "/etc/passwd"}).lower()
        assert bare_agent.call_tool("r", {"path": "/tmp/x.txt"}) == "ok"
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
        assert "denied" in bare_agent.call_tool("r", {"path": "/tmp/x.txt"}).lower()


class TestMultiPathAndConfirm:
    def test_mixed_and_hard_deny(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        _reg(bare_agent, "t", sl=2)
        # Mixed: playground + project → only project confirmed
        captured = []
        bare_agent.confirm_handler = lambda req: (captured.append(1), ConfirmResponse(allow=True))[1]
        assert bare_agent.call_tool("t", {"paths": [os.path.join(bare_agent.playground_dir, "a"), os.path.join(str(tmp_path), "b")]}) == "ok"
        assert len(captured) == 1
        # Hard deny blocks
        result = bare_agent.call_tool("t", {"paths": [os.path.join(str(tmp_path), "ok.py"), "/etc/shadow"]})
        assert "cannot" in result.lower() or "not allowed" in result.lower()

    def test_no_accumulation_hard_deny_no_confirm(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        count = [0]
        bare_agent.confirm_handler = lambda req: (count.__setitem__(0, count[0]+1), ConfirmResponse(allow=True))[1]
        _reg(bare_agent, "w", sl=2)
        p = os.path.join(str(tmp_path), "f.py")
        bare_agent.call_tool("w", {"path": p})
        bare_agent.call_tool("w", {"path": p})
        assert count[0] == 2
        count[0] = 0
        bare_agent.call_tool("w", {"path": "/etc/passwd"})
        assert count[0] == 0

    def test_action_inference(self):
        assert infer_action({"action": "execute", "safety_level": 1}) == "execute"
        assert infer_action({"name": "start_job", "safety_level": 2}) == "execute"
        assert infer_action({"name": "x", "safety_level": 1}) == "read"
        assert infer_action({"name": "x", "safety_level": 2}) == "write"

    def test_hook_skip_and_lifecycle(self, bare_agent, tmp_path, mock_llm_client):
        _setup_zone(bare_agent, tmp_path)
        called = []
        bare_agent.confirm_handler = lambda req: (called.append(1), ConfirmResponse(allow=True))[1]
        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: HookResult.SKIP)
        _reg(bare_agent, "t", sl=2)
        assert "blocked" in bare_agent.call_tool("t", {"path": os.path.join(str(tmp_path), "f")}).lower()
        assert called == []

        mock_llm_client.set_responses([make_llm_response("Hi")])
        events = []
        bare_agent.register_hook(HookEvent.SESSION_START, lambda ctx: events.append("s"))
        bare_agent.register_hook(HookEvent.SESSION_END, lambda ctx: events.append("e"))
        bare_agent.chat("hello")
        bare_agent.shutdown()
        assert events == ["s", "e"]


# ============================================================
# Scope Governance (v1.9.4)
# ============================================================

class TestScopeGovernance:
    def test_find_matching_scope_expiry_and_uses(self):
        from llamagent.core.zone import ZoneDecisionItem, ZoneVerdict
        item = ZoneDecisionItem(path="src/main.py", verdict=ZoneVerdict.CONFIRMABLE, zone="project", action="write")

        expired = ApprovalScope(scope="task", zone="project", actions=["write"], path_prefixes=["src"], expires_at=time.time()-1)
        assert _find_matching_scope(item, "w", [expired]) is None

        used_up = ApprovalScope(scope="task", zone="project", actions=["write"], path_prefixes=["src"], max_uses=3, uses=3)
        assert _find_matching_scope(item, "w", [used_up]) is None

        valid = ApprovalScope(scope="task", zone="project", actions=["write"], path_prefixes=["src"], max_uses=10, uses=0)
        found = _find_matching_scope(item, "w", [valid])
        assert found is valid
        assert valid.uses == 0  # Pure: not incremented

    def test_path_fix(self):
        assert _path_in_prefixes("src/main.py", ["src"]) is True
        assert _path_in_prefixes("src_backup/x.py", ["src"]) is False
        assert _path_in_prefixes("src", ["src"]) is True

    def test_authorization_status(self, bare_agent):
        status = bare_agent.authorization_status()
        assert status["mode"] == "interactive"
        assert status["task_scopes"] == {}
        assert status["session_scopes"] == []


# ============================================================
# Audit Events
# ============================================================

class TestAuditEvents:
    def test_scope_issued_and_revoked_on_mode_switch(self, bare_agent):
        issued = []
        revoked = []
        bare_agent.register_hook(HookEvent("scope_issued"), lambda ctx: issued.append(ctx.data))
        bare_agent.register_hook(HookEvent("scope_revoked"), lambda ctx: revoked.append(ctx.data))

        bare_agent.config.seed_scopes = [{"zone": "project", "actions": ["write"], "path_prefixes": ["src/"]}]
        bare_agent.set_mode("continuous")
        assert len(issued) == 1
        assert issued[0]["scope"].source == "seed"

        bare_agent.set_mode("interactive")
        assert len(revoked) == 1
        assert revoked[0]["reason"] == "mode_switch"

    def test_scope_used_and_denied_events(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        bare_agent.set_mode("task")
        state = bare_agent._controller.state
        state.phase = "executing"
        state.task_id = "T1"
        bare_agent._current_task_id = "T1"

        bare_agent._authorization_engine.state.task_scopes["T1"] = [
            ApprovalScope(scope="task", zone="project", actions=["write"], path_prefixes=[os.path.join(str(tmp_path), "src/")])
        ]

        _reg(bare_agent, "w", sl=2)
        used_events = []
        denied_events = []
        bare_agent.register_hook(HookEvent("scope_used"), lambda ctx: used_events.append(ctx.data))
        bare_agent.register_hook(HookEvent("scope_denied"), lambda ctx: denied_events.append(ctx.data))

        bare_agent.call_tool("w", {"path": os.path.join(str(tmp_path), "src/main.py")})
        assert len(used_events) == 1

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
        bare_agent.call_tool("w", {"path": os.path.join(str(tmp_path), "docs/readme.md")})
        assert len(denied_events) == 1


# ============================================================
# Continuous Mode
# ============================================================

class TestContinuousMode:
    def test_seed_scope_match_deny_and_config(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        bare_agent.config.seed_scopes = [
            {"zone": "project", "actions": ["write"], "path_prefixes": [os.path.join(str(tmp_path), "src/")]},
        ]
        bare_agent.set_mode("continuous")
        _reg(bare_agent, "w", sl=2)

        assert bare_agent.call_tool("w", {"path": os.path.join(str(tmp_path), "src/main.py")}) == "ok"
        result = bare_agent.call_tool("w", {"path": os.path.join(str(tmp_path), "docs/r.md")})
        assert "denied" in result.lower() or "not covered" in result.lower()

    def test_default_project_scope_and_mode_clear(self, bare_agent):
        """v1.9.9: no seed_scopes → default project scope; seed_scopes → use those; interactive → clear all."""
        bare_agent.set_mode("continuous")
        scopes = bare_agent._authorization_engine.state.session_scopes
        assert len(scopes) == 1
        assert scopes[0].source == "default"
        assert scopes[0].zone == "project"
        assert scopes[0].actions == ["read", "write"]

        bare_agent.set_mode("interactive")
        bare_agent.config.seed_scopes = [{"zone": "project", "actions": ["write"], "path_prefixes": ["x"]}]
        bare_agent.set_mode("continuous")
        scopes = bare_agent._authorization_engine.state.session_scopes
        assert len(scopes) == 1
        assert scopes[0].source == "seed"

        bare_agent.set_mode("interactive")
        assert bare_agent._authorization_engine.state.session_scopes == []


# ============================================================
# Config-driven mode initialization (v1.9.9)
# ============================================================

class TestConfigDrivenMode:
    """v1.9.9: agent.__init__ applies authorization_mode from config."""

    def test_config_task_with_seed_scopes(self, mock_llm_client):
        """Config task + seed_scopes → starts in task mode with auto_execute."""
        from llamagent.core.config import Config
        config = Config()
        config.seed_scopes = [
            {"zone": "project", "actions": ["read", "write"], "path_prefixes": ["src/"]}
        ]
        config.authorization_mode = "task"
        agent = LlamAgent(config)
        assert agent.mode == "task"
        assert agent._controller is not None
        assert agent._controller.auto_execute is True
        assert len(agent._authorization_engine.state.session_scopes) == 1

    def test_config_continuous(self, mock_llm_client):
        """Config continuous → starts in continuous mode with default project scope."""
        from llamagent.core.config import Config
        config = Config()
        config.authorization_mode = "continuous"
        agent = LlamAgent(config)
        assert agent.mode == "continuous"
        assert agent._controller is None
        scopes = agent._authorization_engine.state.session_scopes
        assert len(scopes) == 1
        assert scopes[0].source == "default"

    def test_config_interactive_default(self, mock_llm_client):
        """Default config (interactive) → no set_mode called, standard behavior."""
        from llamagent.core.config import Config
        agent = LlamAgent(Config())
        assert agent.mode == "interactive"
        assert agent._controller is None

    def test_config_task_no_seed_scopes_no_handler(self, mock_llm_client):
        """Config task + no seed_scopes + no confirm_handler → prepare flow (auto_execute=False)."""
        from llamagent.core.config import Config
        config = Config()
        config.authorization_mode = "task"
        agent = LlamAgent(config)
        assert agent.mode == "task"
        assert agent._controller is not None
        assert agent._controller.auto_execute is False

    def test_config_invalid_mode_falls_back(self, mock_llm_client):
        """Invalid authorization_mode in config → warning + interactive."""
        from llamagent.core.config import Config
        config = Config()
        config.authorization_mode = "invalid_mode"
        agent = LlamAgent(config)
        assert agent.mode == "interactive"
        assert agent._controller is None


# ============================================================
# apply_update / _clear_all_scopes (v1.9.6)
# ============================================================

class TestApplyUpdateAndClearScopes:
    """engine.apply_update() and _clear_all_scopes(): v1.9.6 scope management."""

    def test_write_and_clear_task_scopes(self, bare_agent):
        """Write approved scopes, then clear them — full lifecycle."""
        engine = bare_agent._authorization_engine
        from llamagent.core.contract import AuthorizationUpdate

        # Write scopes
        result = engine.apply_update(AuthorizationUpdate(
            task_id="T1",
            approved_scopes=[
                RequestedScope(zone="project", actions=["write"], path_prefixes=["src/"]),
                RequestedScope(zone="project", actions=["execute"], path_prefixes=["scripts/"]),
            ],
        ))
        assert result.changed is True
        assert len(engine.state.task_scopes["T1"]) == 2
        scope = engine.state.task_scopes["T1"][0]
        assert isinstance(scope, ApprovalScope)
        assert scope.source == "contract"
        assert len([e for name, e in result.events if name == "scope_issued"]) == 2

        # Clear scopes
        result = engine.apply_update(AuthorizationUpdate(task_id="T1", clear_task_scope=True))
        assert result.changed is True
        assert "T1" not in engine.state.task_scopes
        assert len([e for name, e in result.events if name == "scope_revoked"]) == 2

    def test_clear_noop_and_session(self, bare_agent):
        """Clear nonexistent task is no-op; clear session works."""
        engine = bare_agent._authorization_engine
        from llamagent.core.contract import AuthorizationUpdate

        result = engine.apply_update(AuthorizationUpdate(task_id="NONE", clear_task_scope=True))
        assert result.changed is False

        engine.state.session_scopes = [
            ApprovalScope(scope="session", zone="project", actions=["write"], path_prefixes=["src/"])
        ]
        result = engine.apply_update(AuthorizationUpdate(clear_session_scopes=True))
        assert result.changed is True
        assert engine.state.session_scopes == []

    def test_clear_all_scopes(self, bare_agent):
        """_clear_all_scopes: clears everything, returns events; empty state is no-op."""
        engine = bare_agent._authorization_engine
        # Empty state
        assert engine._clear_all_scopes().changed is False

        # Populated state
        engine.state.task_scopes["T1"] = [
            ApprovalScope(scope="task", zone="project", actions=["write"], path_prefixes=["a"])
        ]
        engine.state.session_scopes = [
            ApprovalScope(scope="session", zone="project", actions=["read"], path_prefixes=["c"])
        ]
        result = engine._clear_all_scopes(reason="test")
        assert result.changed is True
        assert engine.state.task_scopes == {}
        assert engine.state.session_scopes == []
        revoked = [e for name, e in result.events if name == "scope_revoked"]
        assert len(revoked) == 2
        assert all(e["reason"] == "test" for e in revoked)
