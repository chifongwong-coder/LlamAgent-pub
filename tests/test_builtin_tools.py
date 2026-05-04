"""
Builtin tool tests: ask_user and web_search registration, error handling, integration.
"""

import json

import pytest

from llamagent.modules.tools.interaction import CallbackInteractionHandler
from llamagent.core.zone import ConfirmResponse


class TestBuiltinTools:
    """Consolidated builtin tool flow tests."""

    def test_ask_user_flow(self, bare_agent):
        """ask_user: registration, no-handler, exception, callback with choices, agent integration.

        v3.7.6: ask_user is ``takes_agent=True``; per-agent state lives on
        ``agent._tool_state["ask_user_handler"]``. The dispatcher injects
        ``agent`` as the first positional arg, so direct ``ask_user(...)``
        calls (without an agent) no longer work — exercise via
        ``bare_agent.call_tool`` instead.
        """
        # --- Registration as default tool with takes_agent=True ---
        import llamagent.modules.tools.builtin  # noqa: F401
        from llamagent.modules.tools.registry import global_registry
        from llamagent.modules.tools.builtin import ask_user
        info = global_registry.get("ask_user")
        assert info is not None
        assert info.tier == "default"
        assert info.pack is None
        assert info.takes_agent is True

        # Manually register on the bare agent (takes_agent must be passed through).
        bare_agent.register_tool(
            "ask_user", ask_user, "Ask user",
            parameters={"type": "object", "properties": {"question": {"type": "string"}}},
            takes_agent=True,
        )

        # --- No handler returns "cannot" ---
        bare_agent._tool_state.pop("ask_user_handler", None)
        assert "cannot" in bare_agent.call_tool("ask_user", {"question": "test"}).lower()

        # --- Handler that raises returns "failed" ---
        bare_agent._tool_state["ask_user_handler"] = CallbackInteractionHandler(
            lambda q, c: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        assert "failed" in bare_agent.call_tool("ask_user", {"question": "test"}).lower()

        # --- Callback with choices ---
        captured = {}
        bare_agent._tool_state["ask_user_handler"] = CallbackInteractionHandler(
            lambda q, c: (captured.update({"q": q, "c": c}), "Option B")[1]
        )
        assert bare_agent.call_tool(
            "ask_user", {"question": "Pick", "choices": ["A", "B"]}
        ) == "Option B"
        assert captured["c"] == ["A", "B"]

        # --- Full agent integration via interaction_handler attribute ---
        bare_agent.interaction_handler = CallbackInteractionHandler(lambda q, c: "42")
        bare_agent._tool_state["ask_user_handler"] = bare_agent.interaction_handler
        assert bare_agent.call_tool("ask_user", {"question": "6*7?"}) == "42"

    def test_web_search_flow(self):
        """web_search: backend creation, real search, and pack visibility."""
        # --- Backend and real search ---
        from llamagent.modules.tools.web import create_search_backend, DuckDuckGoBackend
        from llamagent.core.config import Config
        backend = create_search_backend(Config())
        if backend is not None:
            assert isinstance(backend, DuckDuckGoBackend)
            results = backend.search("Python programming", num_results=2)
            assert len(results) > 0
            assert all(k in results[0] for k in ("title", "url", "snippet"))

        # --- Pack visibility ---
        import llamagent.modules.tools.builtin  # noqa: F401
        from llamagent.modules.tools.registry import global_registry
        for name in ("web_search", "web_fetch"):
            info = global_registry.get(name)
            assert info is not None
            assert info.pack == "web"
