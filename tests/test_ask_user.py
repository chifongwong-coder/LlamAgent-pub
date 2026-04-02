"""
ask_user tool public flow tests: registration, no-handler error,
callback handler end-to-end, choices support.
"""

import json

import pytest

from llamagent.modules.tools.interaction import (
    CallbackInteractionHandler,
    CLIInteractionHandler,
)


# ============================================================
# Tool registration and basic behavior
# ============================================================

class TestAskUserTool:
    """ask_user tool registration, no-handler fallback, handler integration."""

    def test_ask_user_registered_as_default_tool(self):
        """ask_user is in global_registry with tier=default, pack=None (always visible)."""
        import llamagent.modules.tools.builtin  # noqa: F401 — triggers @tool registration
        from llamagent.modules.tools.registry import global_registry

        info = global_registry.get("ask_user")
        assert info is not None
        assert info.tier == "default"
        assert info.pack is None

    def test_no_handler_returns_error(self):
        """Without handler, ask_user returns error string (not crash)."""
        from llamagent.modules.tools.builtin import ask_user

        # Ensure no handler
        if hasattr(ask_user, "_handler"):
            delattr(ask_user, "_handler")

        result = ask_user(question="What is your name?")
        assert "Cannot ask user" in result or "no interaction handler" in result

    def test_callback_handler_end_to_end(self):
        """CallbackInteractionHandler returns user's response via ask_user tool."""
        from llamagent.modules.tools.builtin import ask_user

        handler = CallbackInteractionHandler(lambda q, c: "Alice")
        ask_user._handler = handler

        result = ask_user(question="What is your name?")
        assert result == "Alice"

        # Cleanup
        delattr(ask_user, "_handler")

    def test_callback_handler_with_choices(self):
        """Choices are passed through to the callback."""
        from llamagent.modules.tools.builtin import ask_user

        captured = {}

        def my_callback(question, choices):
            captured["question"] = question
            captured["choices"] = choices
            return "Option B"

        ask_user._handler = CallbackInteractionHandler(my_callback)
        result = ask_user(question="Pick one", choices=["Option A", "Option B"])

        assert result == "Option B"
        assert captured["choices"] == ["Option A", "Option B"]

        delattr(ask_user, "_handler")

    def test_handler_exception_returns_error_string(self):
        """If handler raises, ask_user returns error string (not crash)."""
        from llamagent.modules.tools.builtin import ask_user

        def broken(q, c):
            raise RuntimeError("connection lost")

        ask_user._handler = CallbackInteractionHandler(broken)
        result = ask_user(question="test")

        assert "Failed to get user response" in result
        assert "connection lost" in result

        delattr(ask_user, "_handler")


# ============================================================
# Agent integration
# ============================================================

class TestAskUserAgentIntegration:
    """ask_user with real agent + ToolsModule."""

    def test_agent_with_interaction_handler(self, bare_agent):
        """Agent with interaction_handler set → ask_user works via call_tool."""
        from llamagent.modules.tools.interaction import CallbackInteractionHandler
        from llamagent.modules.tools.builtin import ask_user

        bare_agent.interaction_handler = CallbackInteractionHandler(lambda q, c: "42")

        # Simulate what ToolsModule.on_attach does
        ask_user._handler = bare_agent.interaction_handler

        # Register tool manually (bare_agent doesn't load ToolsModule)
        bare_agent.register_tool(
            "ask_user", ask_user, "Ask user",
            parameters={"type": "object", "properties": {"question": {"type": "string"}}},
        )

        result = bare_agent.call_tool("ask_user", {"question": "What is 6*7?"})
        assert result == "42"

        delattr(ask_user, "_handler")

    def test_ask_user_visible_without_pack(self, bare_agent):
        """ask_user has no pack — always visible in get_all_tool_schemas."""
        bare_agent.register_tool(
            "ask_user", lambda **kw: "", "Ask user", tier="default",
        )

        schemas = bare_agent.get_all_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "ask_user" in names
