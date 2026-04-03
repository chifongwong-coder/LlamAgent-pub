"""
Continuous mode tests: seed scope matching, no-interaction policy, config loading.
"""

import os

import pytest

from llamagent.core.zone import ConfirmResponse, ApprovalScope
from llamagent.core.authorization import ContinuousPolicy
from conftest import make_llm_response


def _setup_zone(agent, tmp_path):
    agent.project_dir = str(tmp_path)
    agent.playground_dir = str(tmp_path / "llama_playground")
    os.makedirs(agent.playground_dir, exist_ok=True)


class TestContinuousPolicy:
    """Seed scope matching, no-interaction denial, hard deny."""

    def test_seed_scope_match_and_no_match(self, bare_agent, tmp_path):
        """Matching seed scope auto-approves; no match directly denies (no confirm)."""
        _setup_zone(bare_agent, tmp_path)
        bare_agent.config.seed_scopes = [
            {"scope": "session", "zone": "project", "actions": ["write"],
             "path_prefixes": [os.path.join(str(tmp_path), "src/")]},
        ]
        called = []
        bare_agent.confirm_handler = lambda req: (called.append(1), ConfirmResponse(allow=True))[1]

        bare_agent.register_tool("writer", lambda path="": "written", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])

        bare_agent.set_mode("continuous")

        # Within seed scope — auto-approved
        assert bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "src/main.py")}) == "written"
        assert called == []  # confirm never called

        # Outside seed scope — denied
        result = bare_agent.call_tool("writer", {"path": os.path.join(str(tmp_path), "docs/readme.md")})
        assert "denied" in result.lower() or "not covered" in result.lower()
        assert called == []  # still no confirm

    def test_hard_deny_and_allow(self, bare_agent, tmp_path):
        _setup_zone(bare_agent, tmp_path)
        bare_agent.config.seed_scopes = [
            {"scope": "session", "zone": "project", "actions": ["write"], "path_prefixes": [str(tmp_path)]},
        ]
        bare_agent.register_tool("r", lambda: "ok", "read", safety_level=1)
        bare_agent.register_tool("w", lambda path="": "no", "write", safety_level=2,
                                 path_extractor=lambda a: [a.get("path", "")])

        bare_agent.set_mode("continuous")
        assert bare_agent.call_tool("r", {}) == "ok"  # ALLOW (no paths)
        result = bare_agent.call_tool("w", {"path": "/etc/passwd"})
        assert "cannot operate" in result.lower() or "blocked" in result.lower()


class TestSeedScopeConfig:
    """Seed scope loading from config."""

    def test_load_and_clear(self, bare_agent):
        bare_agent.config.seed_scopes = [
            {"scope": "session", "zone": "project", "actions": ["write"], "path_prefixes": ["src/"]},
            {"scope": "session", "zone": "project", "actions": ["execute"], "path_prefixes": ["scripts/"]},
        ]
        bare_agent.set_mode("continuous")
        assert len(bare_agent._authorization_engine.state.session_scopes) == 2

        bare_agent.set_mode("interactive")
        assert bare_agent._authorization_engine.state.session_scopes == []

    def test_no_or_invalid_config(self, bare_agent):
        bare_agent.set_mode("continuous")
        assert bare_agent._authorization_engine.state.session_scopes == []

        bare_agent.config.seed_scopes = "not a list"
        bare_agent.set_mode("continuous")
        assert bare_agent._authorization_engine.state.session_scopes == []
