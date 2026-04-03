"""
Builtin tool tests: ask_user and web_search registration, error handling, integration.
"""

import json

import pytest

from llamagent.modules.tools.interaction import CallbackInteractionHandler
from llamagent.core.zone import ConfirmResponse


class TestAskUser:
    """ask_user: registration, no-handler, callback, choices, exception."""

    def test_registered_as_default_tool(self):
        import llamagent.modules.tools.builtin  # noqa: F401
        from llamagent.modules.tools.registry import global_registry
        info = global_registry.get("ask_user")
        assert info is not None
        assert info.tier == "default"
        assert info.pack is None

    def test_no_handler_and_exception(self):
        from llamagent.modules.tools.builtin import ask_user
        if hasattr(ask_user, "_handler"):
            delattr(ask_user, "_handler")
        assert "cannot" in ask_user(question="test").lower()

        ask_user._handler = CallbackInteractionHandler(lambda q, c: (_ for _ in ()).throw(RuntimeError("fail")))
        assert "failed" in ask_user(question="test").lower()
        delattr(ask_user, "_handler")

    def test_callback_with_choices(self):
        from llamagent.modules.tools.builtin import ask_user
        captured = {}
        ask_user._handler = CallbackInteractionHandler(
            lambda q, c: (captured.update({"q": q, "c": c}), "Option B")[1]
        )
        assert ask_user(question="Pick", choices=["A", "B"]) == "Option B"
        assert captured["c"] == ["A", "B"]
        delattr(ask_user, "_handler")

    def test_agent_integration(self, bare_agent):
        from llamagent.modules.tools.builtin import ask_user
        bare_agent.interaction_handler = CallbackInteractionHandler(lambda q, c: "42")
        ask_user._handler = bare_agent.interaction_handler
        bare_agent.register_tool("ask_user", ask_user, "Ask user",
                                 parameters={"type": "object", "properties": {"question": {"type": "string"}}})
        assert bare_agent.call_tool("ask_user", {"question": "6*7?"}) == "42"
        delattr(ask_user, "_handler")


class TestWebSearch:
    """web_search: real search, tool registration, pack visibility."""

    def test_backend_and_real_search(self):
        from llamagent.modules.tools.web import create_search_backend, DuckDuckGoBackend
        from llamagent.core.config import Config
        backend = create_search_backend(Config())
        if backend is None:
            pytest.skip("No search backend available")
        assert isinstance(backend, DuckDuckGoBackend)
        results = backend.search("Python programming", num_results=2)
        assert len(results) > 0
        assert all(k in results[0] for k in ("title", "url", "snippet"))

    def test_pack_visibility(self, bare_agent):
        import llamagent.modules.tools.builtin  # noqa: F401
        from llamagent.modules.tools.registry import global_registry
        for name in ("web_search", "web_fetch"):
            info = global_registry.get(name)
            assert info is not None
            assert info.pack == "web"
