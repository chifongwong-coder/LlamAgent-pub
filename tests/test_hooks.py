"""
Event Hook system flow tests: tool-level hooks, SKIP semantics, matcher filtering,
lifecycle events, reentry protection, and shell hooks.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from llamagent.core.agent import Module
from llamagent.core.hooks import HookEvent, HookResult, HookMatcher
from conftest import make_llm_response


# ============================================================
# Helpers
# ============================================================

def _register_echo_tool(agent, name="echo", pack="default", safety_level=0):
    """Register a simple tool that returns its input."""
    agent._tools[name] = {
        "name": name,
        "func": lambda **kwargs: f"echo: {kwargs}",
        "pack": pack,
        "safety_level": safety_level,
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
    }


# ============================================================
# Tool-Level Hook Lifecycle
# ============================================================

class TestToolLevelHooks:
    """PRE_TOOL_USE / POST_TOOL_USE / TOOL_ERROR fire correctly with correct data."""

    def test_pre_post_error_hooks_fire_with_correct_data(self, bare_agent):
        """Full tool-level hook lifecycle: PRE fires before, POST fires after with
        result/duration_ms, ERROR fires on exception with error string."""
        _register_echo_tool(bare_agent)
        agent = bare_agent

        # Failing tool
        agent._tools["fail"] = {
            "name": "fail", "func": lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
            "pack": "default", "safety_level": 0, "parameters": {"type": "object", "properties": {}},
        }

        pre_data, post_data, error_data = [], [], []
        agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: pre_data.append(dict(ctx.data)))
        agent.register_hook(HookEvent.POST_TOOL_USE, lambda ctx: post_data.append(dict(ctx.data)))
        agent.register_hook(HookEvent.TOOL_ERROR, lambda ctx: error_data.append(dict(ctx.data)))

        # Successful call
        result = agent.call_tool("echo", {"text": "hi"})
        assert "echo:" in result
        assert len(pre_data) == 1
        assert pre_data[0]["tool_name"] == "echo"
        assert "tool_info" in pre_data[0]
        assert len(post_data) == 1
        assert post_data[0]["tool_name"] == "echo"
        assert "result" in post_data[0]
        assert "result_preview" in post_data[0]
        assert isinstance(post_data[0]["duration_ms"], (int, float))

        # Failing call
        result2 = agent.call_tool("fail", {})
        assert "error" in result2.lower()
        assert len(error_data) == 1
        assert error_data[0]["tool_name"] == "fail"
        assert "boom" in error_data[0]["error"]

    def test_priority_ordering(self, bare_agent):
        """Hooks execute in ascending priority order; same priority preserves registration order."""
        _register_echo_tool(bare_agent)
        order = []

        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: order.append("B"), priority=200)
        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: order.append("A"), priority=50)
        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: order.append("A2"), priority=50)
        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: order.append("C"), priority=150)
        bare_agent.call_tool("echo", {"text": "test"})

        assert order == ["A", "A2", "C", "B"]


# ============================================================
# SKIP Semantics
# ============================================================

class TestSkipSemantics:
    """SKIP blocks execution for PRE_TOOL_USE, ignored for other events."""

    def test_pre_tool_use_skip_blocks_and_returns_reason(self, bare_agent):
        """PRE_TOOL_USE SKIP prevents tool from running; skip_reason appears in result."""
        call_count = [0]
        bare_agent._tools["guarded"] = {
            "name": "guarded", "func": lambda **kw: (call_count.__setitem__(0, call_count[0] + 1), "ran")[1],
            "pack": "default", "safety_level": 0, "parameters": {"type": "object", "properties": {}},
        }

        def blocker(ctx):
            ctx.data["skip_reason"] = "Policy violation"
            return HookResult.SKIP

        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, blocker)
        result = bare_agent.call_tool("guarded", {})

        assert call_count[0] == 0
        assert "Policy violation" in result

    def test_skip_with_matcher_only_blocks_matching_tool(self, bare_agent):
        """PRE_TOOL_USE SKIP with matcher blocks only the target, not others."""
        _register_echo_tool(bare_agent, name="blocked")
        _register_echo_tool(bare_agent, name="allowed")

        bare_agent.register_hook(
            HookEvent.PRE_TOOL_USE, lambda ctx: HookResult.SKIP,
            matcher=HookMatcher(tool_name="blocked"),
        )

        assert "echo:" in bare_agent.call_tool("allowed", {"text": "ok"})
        assert "echo:" not in bare_agent.call_tool("blocked", {"text": "no"})

    def test_skip_ignored_for_non_pre_tool_events(self, bare_agent, mock_llm_client):
        """SKIP is ignored for PRE_CHAT and POST_TOOL_USE — operations proceed normally."""
        _register_echo_tool(bare_agent)
        mock_llm_client.set_responses([make_llm_response("reply")])

        bare_agent.register_hook(HookEvent.PRE_CHAT, lambda ctx: HookResult.SKIP)
        bare_agent.register_hook(HookEvent.POST_TOOL_USE, lambda ctx: HookResult.SKIP)

        # Chat proceeds despite PRE_CHAT SKIP
        assert len(bare_agent.chat("hello")) > 0
        # Tool runs despite POST_TOOL_USE SKIP
        assert "echo:" in bare_agent.call_tool("echo", {"text": "test"})

    def test_skip_stops_remaining_hooks(self, bare_agent):
        """When PRE_TOOL_USE returns SKIP, subsequent hooks do NOT run."""
        _register_echo_tool(bare_agent)
        order = []

        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: (order.append("first"), HookResult.SKIP)[1], priority=50)
        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: order.append("second"), priority=200)
        bare_agent.call_tool("echo", {"text": "test"})

        assert order == ["first"]


# ============================================================
# Matcher Filtering
# ============================================================

class TestMatcherFiltering:
    """HookMatcher filtering with tool_name, tool_names, pack, safety_level."""

    def test_tool_name_and_tool_names_matchers(self, bare_agent):
        """tool_name matches one tool; tool_names matches a set; no matcher matches all."""
        for n in ("a", "b", "c"):
            _register_echo_tool(bare_agent, name=n)

        single_fired, multi_fired, all_fired = [], [], []
        bare_agent.register_hook(HookEvent.POST_TOOL_USE, lambda ctx: single_fired.append(ctx.data["tool_name"]),
                                 matcher=HookMatcher(tool_name="a"))
        bare_agent.register_hook(HookEvent.POST_TOOL_USE, lambda ctx: multi_fired.append(ctx.data["tool_name"]),
                                 matcher=HookMatcher(tool_names=["a", "b"]))
        bare_agent.register_hook(HookEvent.POST_TOOL_USE, lambda ctx: all_fired.append(ctx.data["tool_name"]))

        for n in ("a", "b", "c"):
            bare_agent.call_tool(n, {"text": "x"})

        assert single_fired == ["a"]
        assert sorted(multi_fired) == ["a", "b"]
        assert all_fired == ["a", "b", "c"]

    def test_pack_and_safety_level_matchers(self, bare_agent):
        """pack and safety_level matchers work on PRE_TOOL_USE (tool_info available)."""
        _register_echo_tool(bare_agent, name="admin_risky", pack="admin", safety_level=2)
        _register_echo_tool(bare_agent, name="admin_safe", pack="admin", safety_level=1)
        _register_echo_tool(bare_agent, name="common_risky", pack="common", safety_level=2)

        pack_fired, sl_fired = [], []
        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: pack_fired.append(ctx.data["tool_name"]),
                                 matcher=HookMatcher(pack="admin"))
        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, lambda ctx: sl_fired.append(ctx.data["tool_name"]),
                                 matcher=HookMatcher(safety_level=2))

        for n in ("admin_risky", "admin_safe", "common_risky"):
            bare_agent.call_tool(n, {"text": "x"})

        assert sorted(pack_fired) == ["admin_risky", "admin_safe"]
        assert sorted(sl_fired) == ["admin_risky", "common_risky"]


# ============================================================
# Lifecycle Events
# ============================================================

class TestLifecycleEvents:
    """SESSION_START, PRE_CHAT, POST_CHAT, SESSION_END lifecycle flow."""

    def test_chat_lifecycle_hook_flow(self, bare_agent, mock_llm_client):
        """SESSION_START fires once; PRE_CHAT/POST_CHAT fire every turn with correct data."""
        mock_llm_client.set_responses([make_llm_response("r1"), make_llm_response("r2")])
        events = []

        bare_agent.register_hook(HookEvent.SESSION_START, lambda ctx: events.append("session_start"))
        bare_agent.register_hook(HookEvent.PRE_CHAT, lambda ctx: events.append("pre_chat"))
        bare_agent.register_hook(HookEvent.POST_CHAT, lambda ctx: events.append(("post_chat", ctx.data.get("completed"))))

        bare_agent.chat("first")
        bare_agent.chat("second")

        assert events[0] == "session_start"
        assert events.count("session_start") == 1
        assert events.count("pre_chat") == 2
        assert ("post_chat", True) in events

    def test_post_chat_always_fires_on_safety_block(self, bare_agent, mock_llm_client):
        """POST_CHAT fires even when safety blocks input, with blocked=True."""
        class SafetyBlocker(Module):
            name = "safety_blocker"
            description = "blocks everything"
            def on_input(self, user_input):
                return ""

        bare_agent.register_module(SafetyBlocker())
        post_data = {}
        bare_agent.register_hook(HookEvent.POST_CHAT, lambda ctx: post_data.update(ctx.data))
        bare_agent.chat("blocked")

        assert post_data["blocked"] is True
        assert post_data["blocked_by"] is not None
        assert post_data["completed"] is False

    def test_session_end_fires_on_shutdown(self, bare_agent):
        """SESSION_END fires when shutdown() is called."""
        fired = []
        bare_agent.register_hook(HookEvent.SESSION_END, lambda ctx: fired.append(True))
        bare_agent.shutdown()
        assert fired == [True]


# ============================================================
# Reentry Protection
# ============================================================

class TestReentryProtection:
    """Hook callback calling call_tool does NOT trigger hooks recursively."""

    def test_reentry_blocked(self, bare_agent):
        _register_echo_tool(bare_agent, name="inner")
        _register_echo_tool(bare_agent, name="outer")
        count = [0]

        def reentrant(ctx):
            count[0] += 1
            ctx.agent.call_tool("inner", {"text": "nested"})

        bare_agent.register_hook(HookEvent.PRE_TOOL_USE, reentrant)
        bare_agent.call_tool("outer", {"text": "start"})
        assert count[0] == 1


# ============================================================
# Shell Hooks
# ============================================================

class TestShellHooks:
    """Shell hook execution: exit codes, env vars, YAML registration."""

    @patch("llamagent.core.hooks.subprocess.run")
    def test_shell_hook_exit_codes_and_env_vars(self, mock_run, bare_agent):
        """Exit 0 = CONTINUE (tool runs); exit 1 = SKIP (tool blocked).
        Env vars include HOOK_EVENT, HOOK_TOOL_NAME, HOOK_ARGS (JSON)."""
        _register_echo_tool(bare_agent)

        bare_agent.config.hooks_config = {
            "pre_tool_use": [{"shell": "echo check"}]
        }
        bare_agent._register_yaml_hooks()

        # Exit 0 → tool runs
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = bare_agent.call_tool("echo", {"text": "hello"})
        assert "echo:" in result

        # Verify env vars
        env = mock_run.call_args.kwargs.get("env") or mock_run.call_args[1].get("env")
        assert env["HOOK_TOOL_NAME"] == "echo"
        assert env["HOOK_EVENT"] == "pre_tool_use"
        parsed_args = json.loads(env["HOOK_ARGS"])
        assert parsed_args["text"] == "hello"

        # Exit 1 → tool blocked
        mock_run.return_value = MagicMock(returncode=1, stdout="denied", stderr="")
        result2 = bare_agent.call_tool("echo", {"text": "blocked"})
        assert "echo:" not in result2

    def test_yaml_hook_registration(self, bare_agent):
        """YAML hooks register with correct priority/source/matcher."""
        bare_agent.config.hooks_config = {
            "pre_tool_use": [
                {"shell": "echo default"},
                {"shell": "echo custom", "priority": 50, "matcher": {"tool_name": "start_job"}},
            ],
            "post_tool_use": [{"shell": "echo done"}],
        }
        bare_agent._register_yaml_hooks()

        pre = bare_agent._hooks.get(HookEvent.PRE_TOOL_USE, [])
        assert len(pre) == 2
        # Custom priority comes first (50 < 200)
        assert pre[0].priority == 50
        assert pre[0].matcher.tool_name == "start_job"
        assert pre[1].priority == 200
        assert pre[1].source == "yaml"

        post = bare_agent._hooks.get(HookEvent.POST_TOOL_USE, [])
        assert len(post) == 1
