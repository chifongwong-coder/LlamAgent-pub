"""
Authorization system flow tests: zone equivalence, confirm handling, scope
governance, audit events, continuous mode, config-driven init, apply_update.

Consolidated from ~21 unit tests into 5 flow tests. Every original assertion
is preserved.
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
# Flow 1: Zone evaluation + confirm structure
# ============================================================

def test_zone_evaluation_and_confirm_flow(bare_agent, tmp_path, mock_llm_client):
    """
    Full zone evaluation flow: playground auto-allow, project/external matrix,
    mixed-path confirm, hard_deny blocking, no-accumulation semantics,
    action inference, hook skip, and lifecycle events.
    """
    _setup_zone(bare_agent, tmp_path)

    # -- Playground: no-path + playground-path both auto-allow --
    _reg(bare_agent, "t", sl=2)
    assert bare_agent.call_tool("t", {"path": os.path.join(bare_agent.playground_dir, "f.txt")}) == "ok"
    bare_agent.register_tool("nop", lambda: "ok", "nop", safety_level=2)
    assert bare_agent.call_tool("nop", {}) == "ok"

    # -- Project / external matrix --
    _reg(bare_agent, "r", sl=1)
    _reg(bare_agent, "w", sl=2)
    p = os.path.join(str(tmp_path), "f.py")

    # sl=1 project read: auto-allow
    assert bare_agent.call_tool("r", {"path": p}) == "ok"
    # sl=2 project write: confirm allow
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    assert bare_agent.call_tool("w", {"path": p}) == "ok"
    # sl=2 project write: confirm deny
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
    assert "denied" in bare_agent.call_tool("w", {"path": p}).lower()
    # sl=2 project write: no handler => denied
    bare_agent.confirm_handler = None
    assert "denied" in bare_agent.call_tool("w", {"path": p}).lower()

    # External write: hard deny
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    assert "cannot" in bare_agent.call_tool("w", {"path": "/etc/passwd"}).lower() or "not allowed" in bare_agent.call_tool("w", {"path": "/etc/passwd"}).lower()
    # External read: confirm allow
    assert bare_agent.call_tool("r", {"path": "/tmp/x.txt"}) == "ok"
    # External read: confirm deny
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
    assert "denied" in bare_agent.call_tool("r", {"path": "/tmp/x.txt"}).lower()

    # -- Mixed paths: playground + project -> only project confirmed --
    bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
    captured = []
    bare_agent.confirm_handler = lambda req: (captured.append(1), ConfirmResponse(allow=True))[1]
    assert bare_agent.call_tool("t", {"paths": [os.path.join(bare_agent.playground_dir, "a"), os.path.join(str(tmp_path), "b")]}) == "ok"
    assert len(captured) == 1

    # Hard deny blocks entire call (mixed with external)
    result = bare_agent.call_tool("t", {"paths": [os.path.join(str(tmp_path), "ok.py"), "/etc/shadow"]})
    assert "cannot" in result.lower() or "not allowed" in result.lower()

    # -- No accumulation: confirm fires every time; hard deny skips confirm --
    count = [0]
    bare_agent.confirm_handler = lambda req: (count.__setitem__(0, count[0]+1), ConfirmResponse(allow=True))[1]
    bare_agent.call_tool("w", {"path": p})
    bare_agent.call_tool("w", {"path": p})
    assert count[0] == 2
    count[0] = 0
    bare_agent.call_tool("w", {"path": "/etc/passwd"})
    assert count[0] == 0

    # -- Action inference --
    assert infer_action({"action": "execute", "safety_level": 1}) == "execute"
    assert infer_action({"name": "start_job", "safety_level": 2}) == "execute"
    assert infer_action({"name": "x", "safety_level": 1}) == "read"
    assert infer_action({"name": "x", "safety_level": 2}) == "write"

    # -- Hook skip: PRE_TOOL_USE SKIP blocks tool, no confirm --
    called = []
    bare_agent.confirm_handler = lambda req: (called.append(1), ConfirmResponse(allow=True))[1]
    bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: HookResult.SKIP)
    assert "blocked" in bare_agent.call_tool("t", {"path": os.path.join(str(tmp_path), "f")}).lower()
    assert called == []

    # -- Lifecycle events --
    mock_llm_client.set_responses([make_llm_response("Hi")])
    events = []
    bare_agent.register_hook(HookEvent.SESSION_START, lambda ctx: events.append("s"))
    bare_agent.register_hook(HookEvent.SESSION_END, lambda ctx: events.append("e"))
    bare_agent.chat("hello")
    bare_agent.shutdown()
    assert events == ["s", "e"]


# ============================================================
# Flow 2: Scope governance + audit events
# ============================================================

def test_scope_governance_and_audit(bare_agent, tmp_path):
    """
    Scope matching (expiry, uses, path prefix), authorization status,
    audit events: scope_issued/revoked on mode switch, scope_used/denied.
    """
    # -- _find_matching_scope: expiry + uses --
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

    # -- _path_in_prefixes: boundary cases --
    assert _path_in_prefixes("src/main.py", ["src"]) is True
    assert _path_in_prefixes("src_backup/x.py", ["src"]) is False
    assert _path_in_prefixes("src", ["src"]) is True

    # -- authorization_status: default state --
    status = bare_agent.authorization_status()
    assert status["mode"] == "interactive"
    assert status["task_scopes"] == {}
    assert status["session_scopes"] == []

    # -- Audit: scope_issued + scope_revoked on mode switch --
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

    # -- Audit: scope_used + scope_denied during task execution --
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
# Flow 3: Continuous mode scopes
# ============================================================

def test_continuous_mode_scopes(bare_agent, tmp_path):
    """
    Continuous mode: seed scope match/deny, no-seed default,
    mode clear on switch.
    """
    _setup_zone(bare_agent, tmp_path)
    bare_agent.config.seed_scopes = [
        {"zone": "project", "actions": ["write"], "path_prefixes": [os.path.join(str(tmp_path), "src/")]},
    ]
    bare_agent.set_mode("continuous")
    _reg(bare_agent, "w", sl=2)

    # Seed scope covers src/ -> allow
    assert bare_agent.call_tool("w", {"path": os.path.join(str(tmp_path), "src/main.py")}) == "ok"
    # Seed scope does not cover docs/ -> deny
    result = bare_agent.call_tool("w", {"path": os.path.join(str(tmp_path), "docs/r.md")})
    assert "denied" in result.lower() or "not covered" in result.lower()

    # -- No seed scopes: v1.9.9 default project scope --
    bare_agent.set_mode("interactive")  # reset
    bare_agent.config.seed_scopes = None
    bare_agent.set_mode("continuous")
    scopes = bare_agent._authorization_engine.state.session_scopes
    assert len(scopes) == 1
    assert scopes[0].source == "default"
    assert scopes[0].zone == "project"
    assert scopes[0].actions == ["read", "write"]

    # -- Seed scopes override default --
    bare_agent.set_mode("interactive")
    bare_agent.config.seed_scopes = [{"zone": "project", "actions": ["write"], "path_prefixes": ["x"]}]
    bare_agent.set_mode("continuous")
    scopes = bare_agent._authorization_engine.state.session_scopes
    assert len(scopes) == 1
    assert scopes[0].source == "seed"
    bare_agent.set_mode("interactive")
    assert bare_agent._authorization_engine.state.session_scopes == []


# ============================================================
# Flow 4: apply_update + _clear_all_scopes
# ============================================================

def test_apply_update_and_clear_scopes(bare_agent):
    """
    apply_update: write + clear task scopes, clear noop, clear session.
    _clear_all_scopes: empty noop, populated clear with events.
    """
    engine = bare_agent._authorization_engine
    from llamagent.core.contract import AuthorizationUpdate

    # -- Write scopes via apply_update --
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

    # -- Clear task scopes via apply_update --
    result = engine.apply_update(AuthorizationUpdate(task_id="T1", clear_task_scope=True))
    assert result.changed is True
    assert "T1" not in engine.state.task_scopes
    assert len([e for name, e in result.events if name == "scope_revoked"]) == 2

    # -- Clear nonexistent task is no-op --
    result = engine.apply_update(AuthorizationUpdate(task_id="NONE", clear_task_scope=True))
    assert result.changed is False

    # -- Clear session scopes via apply_update --
    engine.state.session_scopes = [
        ApprovalScope(scope="session", zone="project", actions=["write"], path_prefixes=["src/"])
    ]
    result = engine.apply_update(AuthorizationUpdate(clear_session_scopes=True))
    assert result.changed is True
    assert engine.state.session_scopes == []

    # -- _clear_all_scopes: empty state is no-op --
    assert engine._clear_all_scopes().changed is False

    # -- _clear_all_scopes: populated state clears everything --
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


# ============================================================
# Flow 5: Config-driven mode initialization (v1.9.9)
# ============================================================

def test_config_driven_mode_initialization(mock_llm_client):
    """Config authorization_mode applied at agent init: task+seeds, continuous,
    interactive default, task no seeds/handler, invalid mode fallback."""
    from llamagent.core.config import Config

    # -- task + seed_scopes → auto_execute --
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

    # -- continuous → default project scope --
    config2 = Config()
    config2.authorization_mode = "continuous"
    agent2 = LlamAgent(config2)
    assert agent2.mode == "continuous"
    assert agent2._controller is None
    scopes = agent2._authorization_engine.state.session_scopes
    assert len(scopes) == 1
    assert scopes[0].source == "default"

    # -- interactive (default) → no set_mode called --
    agent3 = LlamAgent(Config())
    assert agent3.mode == "interactive"
    assert agent3._controller is None

    # -- task + no seed_scopes + no confirm_handler → prepare flow --
    config4 = Config()
    config4.authorization_mode = "task"
    agent4 = LlamAgent(config4)
    assert agent4.mode == "task"
    assert agent4._controller is not None
    assert agent4._controller.auto_execute is False

    # -- invalid mode → warning + interactive fallback --
    config5 = Config()
    config5.authorization_mode = "invalid_mode"
    agent5 = LlamAgent(config5)
    assert agent5.mode == "interactive"
    assert agent5._controller is None
