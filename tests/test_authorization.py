"""
Authorization system tests: zone equivalence, multi-path, action inference,
confirm request structure, and interactive mode invariants.
"""

import os
import json

import pytest

from llamagent.core.zone import ConfirmRequest, ConfirmResponse
from llamagent.core.authorization import infer_action
from llamagent.core.hooks import HookEvent, HookResult
from conftest import make_llm_response


def _setup_zone(agent, tmp_path):
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


def _reg(agent, name, sl=1, result="ok", action=None):
    agent.register_tool(
        name, lambda **kw: result, f"tool {name}",
        safety_level=sl, action=action,
        path_extractor=lambda args: [args["path"]] if isinstance(args.get("path"), str) else args.get("paths", []),
    )


class TestZoneEquivalence:
    """All zone scenarios produce identical behavior to v1.8.x."""

    def test_playground_and_no_paths_allow(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        _reg(bare_agent, "t", sl=2)
        pg = os.path.join(bare_agent.playground_dir, "f.txt")
        assert bare_agent.call_tool("t", {"path": pg}) == "ok"

        bare_agent.register_tool("nop", lambda: "ok", "nop", safety_level=2)
        assert bare_agent.call_tool("nop", {}) == "ok"

    def test_project_read_write_matrix(self, bare_agent, tmp_path):
        """Project: sl=1 allow, sl=2 confirm(allow/deny), sl=2 no handler deny."""
        _setup_zone(bare_agent, tmp_path)
        _reg(bare_agent, "r", sl=1)
        _reg(bare_agent, "w", sl=2)
        path = os.path.join(str(tmp_path), "f.py")

        assert bare_agent.call_tool("r", {"path": path}) == "ok"

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        assert bare_agent.call_tool("w", {"path": path}) == "ok"

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
        assert "denied" in bare_agent.call_tool("w", {"path": path}).lower()

        bare_agent.confirm_handler = None
        assert "denied" in bare_agent.call_tool("w", {"path": path}).lower()

    def test_external_read_write_matrix(self, bare_agent, tmp_path):
        """External: sl=2 hard deny, sl=1 confirm(allow/deny)."""
        _setup_zone(bare_agent, tmp_path)
        _reg(bare_agent, "r", sl=1)
        _reg(bare_agent, "w", sl=2)

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        assert "cannot operate" in bare_agent.call_tool("w", {"path": "/etc/passwd"}).lower() or \
               "not allowed" in bare_agent.call_tool("w", {"path": "/etc/passwd"}).lower()
        assert bare_agent.call_tool("r", {"path": "/tmp/x.txt"}) == "ok"

        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=False)
        assert "denied" in bare_agent.call_tool("r", {"path": "/tmp/x.txt"}).lower()


class TestMultiPath:
    def test_mixed_verdicts_and_first_deny_stops(self, bare_agent, tmp_path):
        """Playground+project: only project confirmed. Two project paths: second denied stops."""
        _setup_zone(bare_agent, tmp_path)
        captured = []
        bare_agent.confirm_handler = lambda req: (captured.append(req.target_paths[0]), ConfirmResponse(allow=True))[1]
        _reg(bare_agent, "t", sl=2)

        pg = os.path.join(bare_agent.playground_dir, "a.txt")
        proj = os.path.join(str(tmp_path), "b.py")
        assert bare_agent.call_tool("t", {"paths": [pg, proj]}) == "ok"
        assert len(captured) == 1  # only project confirmed

        # Two project paths, second denied
        captured.clear()
        count = [0]
        bare_agent.confirm_handler = lambda req: (count.__setitem__(0, count[0]+1), ConfirmResponse(allow=count[0]==1))[1]
        result = bare_agent.call_tool("t", {"paths": [os.path.join(str(tmp_path), "a.py"), os.path.join(str(tmp_path), "b.py")]})
        assert "denied" in result.lower()

    def test_hard_deny_blocks_overall(self, bare_agent, tmp_path):
        """Multi-path with HARD_DENY: overall operation blocked."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        _reg(bare_agent, "t", sl=2)
        result = bare_agent.call_tool("t", {"paths": [os.path.join(str(tmp_path), "ok.py"), "/etc/shadow"]})
        assert "cannot operate" in result.lower() or "not allowed" in result.lower()


class TestConfirmAndInvariants:
    def test_confirm_request_fields(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        captured = []
        bare_agent.confirm_handler = lambda req: (captured.append(req), ConfirmResponse(allow=True))[1]
        _reg(bare_agent, "w", sl=2)
        bare_agent.call_tool("w", {"path": os.path.join(str(tmp_path), "f.txt")})
        req = captured[0]
        assert req.kind == "operation_confirm"
        assert req.zone == "project"
        assert req.mode == "interactive"

    def test_no_accumulation_and_hard_deny_never_confirms(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        count = [0]
        bare_agent.confirm_handler = lambda req: (count.__setitem__(0, count[0]+1), ConfirmResponse(allow=True))[1]
        _reg(bare_agent, "w", sl=2)
        path = os.path.join(str(tmp_path), "f.py")
        bare_agent.call_tool("w", {"path": path})
        bare_agent.call_tool("w", {"path": path})
        assert count[0] == 2  # confirmed twice, no accumulation

        count[0] = 0
        bare_agent.call_tool("w", {"path": "/etc/passwd"})
        assert count[0] == 0  # hard deny never calls confirm

    def test_action_inference(self):
        assert infer_action({"action": "execute", "safety_level": 1}) == "execute"
        assert infer_action({"name": "start_job", "safety_level": 2}) == "execute"
        assert infer_action({"name": "x", "safety_level": 1}) == "read"
        assert infer_action({"name": "x", "safety_level": 2}) == "write"

    def test_hook_skip_before_auth(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        called = []
        bare_agent.confirm_handler = lambda req: (called.append(1), ConfirmResponse(allow=True))[1]
        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: HookResult.SKIP)
        _reg(bare_agent, "t", sl=2)
        result = bare_agent.call_tool("t", {"path": os.path.join(str(tmp_path), "f.py")})
        assert "blocked by hook" in result.lower()
        assert called == []

    def test_full_chain_chat_and_lifecycle(self, bare_agent, mock_llm_client):
        mock_llm_client.set_responses([make_llm_response("Hi")])
        events = []
        bare_agent.register_hook(HookEvent.SESSION_START, lambda ctx: events.append("s"))
        bare_agent.register_hook(HookEvent.SESSION_END, lambda ctx: events.append("e"))
        assert bare_agent.chat("hello") == "Hi"
        bare_agent.shutdown()
        assert events == ["s", "e"]
