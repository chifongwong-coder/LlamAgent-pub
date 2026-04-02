"""
Web search public flow tests: real backend creation, real search calls,
tool integration with pack system.
"""

import json
import os

import pytest

from llamagent.core.config import Config
from llamagent.modules.tools.web import create_search_backend, DuckDuckGoBackend


# ============================================================
# Real backend + search
# ============================================================

class TestRealWebSearch:
    """Tests that hit real DuckDuckGo search. Requires ddgs or duckduckgo-search installed."""

    def test_backend_auto_detect_and_real_search(self):
        """Factory auto-detects DuckDuckGo; real search returns results with correct format."""
        backend = create_search_backend(Config())
        if backend is None:
            pytest.skip("No search backend available (ddgs not installed)")

        assert isinstance(backend, DuckDuckGoBackend)

        results = backend.search("Python programming language", num_results=3)
        assert len(results) > 0
        assert len(results) <= 3

        for r in results:
            assert "title" in r and isinstance(r["title"], str)
            assert "url" in r and isinstance(r["url"], str)
            assert "snippet" in r and isinstance(r["snippet"], str)
            assert r["url"].startswith("http")

    def test_web_search_tool_end_to_end(self):
        """web_search tool returns valid JSON with search results."""
        from llamagent.modules.tools.builtin import web_search
        from llamagent.modules.tools.web import create_search_backend

        backend = create_search_backend(Config())
        if backend is None:
            pytest.skip("No search backend available")

        web_search._backend = backend
        result = web_search(query="what is Python", num_results=2)
        parsed = json.loads(result)

        assert isinstance(parsed, list)
        assert len(parsed) > 0
        assert "title" in parsed[0]
        assert "url" in parsed[0]


# ============================================================
# Tool + pack integration
# ============================================================

class TestWebToolPackIntegration:
    """web_search and web_fetch pack visibility with real ToolsModule."""

    def test_web_tools_hidden_by_default_visible_with_pack(self, bare_agent):
        """web_search and web_fetch are hidden without pack, visible with pack='web'."""
        from llamagent.modules.tools.registry import global_registry

        # Bridge tools to agent (simulate what ToolsModule does)
        for name, info in global_registry._tools.items():
            bare_agent._tools[name] = {
                "name": name, "func": info.func,
                "description": info.description, "parameters": info.parameters,
                "tier": info.tier, "safety_level": info.safety_level, "pack": info.pack,
            }

        # Without pack: web tools hidden
        schemas = bare_agent.get_all_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "web_search" not in names
        assert "web_fetch" not in names

        # With pack: web tools visible
        bare_agent._active_packs.add("web")
        schemas2 = bare_agent.get_all_tool_schemas()
        names2 = [s["function"]["name"] for s in schemas2]
        assert "web_search" in names2
        assert "web_fetch" in names2
