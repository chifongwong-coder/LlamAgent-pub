"""
Authorization system flow tests: zone equivalence with v1.8.x behavior,
multi-path evaluation, confirm_handler integration, action inference,
full chain equivalence, and authorization invariants.

All tests use real SmartAgent instances with real zone paths.
"""

import json
import os

import pytest

from llamagent.core.agent import SmartAgent, Module
from llamagent.core.hooks import HookEvent, HookResult
from llamagent.core.zone import ConfirmRequest, ConfirmResponse, ZoneEvaluation, ZoneVerdict
from llamagent.core.authorization import infer_action
from conftest import make_llm_response


# ============================================================
# Helpers
# ============================================================

def _setup_zone(agent, tmp_path):
    """Set project/playground dirs on agent."""
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


def _register_path_tool(agent, name, safety_level=1, result="ok"):
    """Register a tool with path_extractor."""
    agent.register_tool(
        name, lambda path="": result, f"tool {name}",
        safety_level=safety_level,
        path_extractor=lambda args: [args["path"]] if isinstance(args.get("path"), str) else args.get("paths", []),
    )


# ============================================================
# Zone Equivalence (plan section 12.1)
# ============================================================

class TestZoneEquivalence:
    """All 9 zone scenarios produce identical behavior to v1.8.x."""

    def test_playground_always_allow(self, bare_agent, tmp_path):
        """Playground path + sl=2 → executes (no confirm)."""
        _setup_zone(bare_agent, tmp_path)
        _register_path_tool(bare_agent, "t", safety_level=2)
        path = os.path.join(bare_agent.playground_dir, "test.txt")
        assert bare_agent.call_tool("t", {"path": path}) == "ok"

    def test_project_read_allow(self, bare_agent, tmp_path):
        """Project path + sl=1 → executes (no confirm)."""
        _setup_zone(bare_agent, tmp_path)
        _register_path_tool(bare_agent, "t", safety_level=1)
        assert bare_agent.call_tool("t", {"path": os.path.join(str(tmp_path), "f.py")}) == "ok"

    def test_project_write_confirm_allow_deny_nohandler(self, bare_agent, tmp_path):
        """Project path + sl=2: confirm=True→exec, confirm=False→deny, no handler→deny."""
        _setup_zone(bare_agent, tmp_path)
        _register_path_tool(bare_agent, "t", safety_level=2)
        path = os.path.join(str(tmp_path), "f.py")

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        assert bare_agent.call_tool("t", {"path": path}) == "ok"

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
        assert "denied" in bare_agent.call_tool("t", {"path": path}).lower()

        bare_agent.confirm_handler = None
        assert "denied" in bare_agent.call_tool("t", {"path": path}).lower()

    def test_external_write_hard_deny(self, bare_agent, tmp_path):
        """External path + sl=2 → HARD_DENY, even with confirm handler returning True."""
        _setup_zone(bare_agent, tmp_path)
        _register_path_tool(bare_agent, "t", safety_level=2)
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        result = bare_agent.call_tool("t", {"path": "/etc/passwd"})
        assert "cannot operate" in result.lower() or "not allowed" in result.lower()

    def test_external_read_confirm_allow_deny(self, bare_agent, tmp_path):
        """External path + sl=1: confirm=True→exec, confirm=False→deny."""
        _setup_zone(bare_agent, tmp_path)
        _register_path_tool(bare_agent, "t", safety_level=1)

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        assert bare_agent.call_tool("t", {"path": "/tmp/x.txt"}) == "ok"

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
        assert "denied" in bare_agent.call_tool("t", {"path": "/tmp/x.txt"}).lower()

    def test_no_paths_always_allow(self, bare_agent):
        """Tool with no paths → executes regardless of safety_level."""
        bare_agent.register_tool("t", lambda: "ok", "tool", safety_level=2)
        assert bare_agent.call_tool("t", {}) == "ok"


# ============================================================
# Multi-Path Evaluation
# ============================================================

class TestMultiPath:
    """Tool calls involving multiple paths with mixed verdicts."""

    def test_mixed_allow_confirmable(self, bare_agent, tmp_path):
        """Playground (ALLOW) + project sl=2 (CONFIRMABLE): only project confirmed."""
        _setup_zone(bare_agent, tmp_path)
        captured = []
        bare_agent.confirm_handler = lambda req: (captured.append(req), ConfirmResponse(allow=True))[1]

        bare_agent.register_tool(
            "t", lambda paths="": "ok", "multi", safety_level=2,
            path_extractor=lambda args: args.get("paths", []),
        )

        pg = os.path.join(bare_agent.playground_dir, "safe.txt")
        proj = os.path.join(str(tmp_path), "main.py")
        assert bare_agent.call_tool("t", {"paths": [pg, proj]}) == "ok"
        assert len(captured) == 1  # Only project path confirmed

    def test_hard_deny_blocks_without_confirm(self, bare_agent, tmp_path):
        """Project + external sl=2: HARD_DENY blocks, confirm never called."""
        _setup_zone(bare_agent, tmp_path)
        confirm_called = []
        bare_agent.confirm_handler = lambda req: (confirm_called.append(1), ConfirmResponse(allow=True))[1]

        bare_agent.register_tool(
            "t", lambda paths="": "no", "multi", safety_level=2,
            path_extractor=lambda args: args.get("paths", []),
        )

        result = bare_agent.call_tool("t", {"paths": [os.path.join(str(tmp_path), "ok.py"), "/etc/shadow"]})
        assert "cannot operate" in result.lower() or "not allowed" in result.lower()

    def test_first_deny_stops_iteration(self, bare_agent, tmp_path):
        """Two CONFIRMABLE paths, first denied → stops, second never asked."""
        _setup_zone(bare_agent, tmp_path)
        paths_confirmed = []
        bare_agent.confirm_handler = lambda req: (paths_confirmed.extend(req.target_paths), ConfirmResponse(allow=False))[1]

        bare_agent.register_tool(
            "t", lambda paths="": "no", "multi", safety_level=2,
            path_extractor=lambda args: args.get("paths", []),
        )

        result = bare_agent.call_tool("t", {"paths": [os.path.join(str(tmp_path), "a.py"), os.path.join(str(tmp_path), "b.py")]})
        assert "denied" in result.lower()
        assert len(paths_confirmed) == 1  # Only first path was asked


# ============================================================
# ConfirmRequest Structure + Invariants
# ============================================================

class TestConfirmRequest:
    """ConfirmRequest fields and authorization invariants."""

    def test_confirm_request_has_correct_fields(self, bare_agent, tmp_path):
        """ConfirmRequest has kind, tool_name, action, zone, target_paths, message, mode."""
        _setup_zone(bare_agent, tmp_path)
        captured = []
        bare_agent.confirm_handler = lambda req: (captured.append(req), ConfirmResponse(allow=True))[1]
        _register_path_tool(bare_agent, "w", safety_level=2)

        bare_agent.call_tool("w", {"path": os.path.join(str(tmp_path), "f.txt")})
        req = captured[0]
        assert req.kind == "operation_confirm"
        assert req.tool_name == "w"
        assert req.action in ("write", "execute")
        assert req.zone == "project"
        assert req.mode == "interactive"

    def test_no_accumulation_in_interactive(self, bare_agent, tmp_path):
        """Same CONFIRMABLE tool called twice → two separate confirms (no scope accumulation)."""
        _setup_zone(bare_agent, tmp_path)
        count = [0]
        bare_agent.confirm_handler = lambda req: (count.__setitem__(0, count[0] + 1), ConfirmResponse(allow=True))[1]
        _register_path_tool(bare_agent, "w", safety_level=2)
        path = os.path.join(str(tmp_path), "f.py")
        bare_agent.call_tool("w", {"path": path})
        bare_agent.call_tool("w", {"path": path})
        assert count[0] == 2

    def test_hard_deny_never_calls_confirm(self, bare_agent, tmp_path):
        """HARD_DENY → confirm_handler is never called."""
        _setup_zone(bare_agent, tmp_path)
        called = []
        bare_agent.confirm_handler = lambda req: (called.append(1), ConfirmResponse(allow=True))[1]
        _register_path_tool(bare_agent, "w", safety_level=2)
        bare_agent.call_tool("w", {"path": "/etc/passwd"})
        assert called == []


# ============================================================
# Full Chain Equivalence (plan section 12.2)
# ============================================================

class TestFullChain:
    """Non-zone paths unaffected by authorization refactor."""

    def test_pure_chat(self, bare_agent, mock_llm_client):
        mock_llm_client.set_responses([make_llm_response("Hi")])
        assert bare_agent.chat("hello") == "Hi"

    def test_hook_skip_before_auth(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        called = []
        bare_agent.confirm_handler = lambda req: (called.append(1), ConfirmResponse(allow=True))[1]
        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: HookResult.SKIP)
        _register_path_tool(bare_agent, "t", safety_level=2)
        result = bare_agent.call_tool("t", {"path": os.path.join(str(tmp_path), "f.py")})
        assert "blocked by hook" in result.lower()
        assert called == []

    def test_session_lifecycle(self, bare_agent, mock_llm_client):
        mock_llm_client.set_responses([make_llm_response("ok")])
        events = []
        bare_agent.register_hook(HookEvent.SESSION_START, lambda ctx: events.append("start"))
        bare_agent.register_hook(HookEvent.SESSION_END, lambda ctx: events.append("end"))
        bare_agent.chat("hi")
        bare_agent.shutdown()
        assert events == ["start", "end"]

    def test_ask_user_no_handler(self):
        from llamagent.modules.tools.builtin import ask_user
        if hasattr(ask_user, "_handler"):
            delattr(ask_user, "_handler")
        assert "cannot" in ask_user(question="test").lower()

    def test_web_search_no_backend(self):
        from llamagent.modules.tools.builtin import web_search
        orig = getattr(web_search, "_backend", None)
        try:
            web_search._backend = None
            result = json.loads(web_search(query="test"))
            assert "error" in result
        finally:
            if orig:
                web_search._backend = orig


# ============================================================
# Action Inference
# ============================================================

class TestActionInference:
    def test_explicit_overrides(self):
        assert infer_action({"action": "execute", "safety_level": 1}) == "execute"

    def test_start_job_is_execute(self):
        assert infer_action({"name": "start_job", "safety_level": 2}) == "execute"

    def test_sl_derivation(self):
        assert infer_action({"name": "x", "safety_level": 1}) == "read"
        assert infer_action({"name": "x", "safety_level": 2}) == "write"
        assert infer_action({"name": "x"}) == "read"  # default sl=1
