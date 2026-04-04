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
        """ask_user: registration, no-handler, exception, callback with choices, agent integration."""
        # --- Registration as default tool ---
        import llamagent.modules.tools.builtin  # noqa: F401
        from llamagent.modules.tools.registry import global_registry
        info = global_registry.get("ask_user")
        assert info is not None
        assert info.tier == "default"
        assert info.pack is None

        # --- No handler returns "cannot"; exception returns "failed" ---
        from llamagent.modules.tools.builtin import ask_user
        if hasattr(ask_user, "_handler"):
            delattr(ask_user, "_handler")
        assert "cannot" in ask_user(question="test").lower()

        ask_user._handler = CallbackInteractionHandler(lambda q, c: (_ for _ in ()).throw(RuntimeError("fail")))
        assert "failed" in ask_user(question="test").lower()
        delattr(ask_user, "_handler")

        # --- Callback with choices ---
        captured = {}
        ask_user._handler = CallbackInteractionHandler(
            lambda q, c: (captured.update({"q": q, "c": c}), "Option B")[1]
        )
        assert ask_user(question="Pick", choices=["A", "B"]) == "Option B"
        assert captured["c"] == ["A", "B"]
        delattr(ask_user, "_handler")

        # --- Agent integration ---
        bare_agent.interaction_handler = CallbackInteractionHandler(lambda q, c: "42")
        ask_user._handler = bare_agent.interaction_handler
        bare_agent.register_tool("ask_user", ask_user, "Ask user",
                                 parameters={"type": "object", "properties": {"question": {"type": "string"}}})
        assert bare_agent.call_tool("ask_user", {"question": "6*7?"}) == "42"
        delattr(ask_user, "_handler")

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
