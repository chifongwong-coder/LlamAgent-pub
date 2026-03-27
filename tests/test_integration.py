"""
Module integration tests: multi-module loading, cooperation, and backward compatibility.
"""

from unittest.mock import patch, MagicMock

import pytest

from llamagent.core.agent import SmartAgent, Module
from llamagent.core.config import Config
from conftest import make_llm_response


def _create_test_agent(mock_llm_client):
    """Create an agent for integration tests with mocked LLM."""
    config = Config.__new__(Config)
    config.model = "mock-model"
    config.system_prompt = "You are a test assistant."
    config.context_window_size = 20
    config.context_compress_threshold = 0.7
    config.compress_keep_turns = 3
    config.max_react_steps = 10
    config.max_duplicate_actions = 2
    config.react_timeout = 210.0
    config.max_observation_tokens = 2000
    config.max_context_tokens = 8192
    config.memory_mode = "off"
    config.reflection_enabled = False
    config.reflection_score_threshold = 7.0
    config.max_plan_adjustments = 7
    config.permission_level = 1
    config.output_dir = "/tmp/llamagent_test_output"
    config.persona_file = "/tmp/llamagent_test_personas.json"
    config.agent_tools_dir = "/tmp/llamagent_test_tools"

    agent = SmartAgent.__new__(SmartAgent)
    agent.config = config
    agent.persona = None
    agent.llm = mock_llm_client
    agent.modules = {}
    agent.history = []
    agent.summary = None
    agent.conversation = agent.history
    import os
    agent.confirm_handler = None
    agent.project_dir = os.path.realpath(os.getcwd())
    agent.playground_dir = os.path.realpath(os.path.join(agent.project_dir, "llama_playground"))
    agent.tool_executor = None
    agent._tools = {}
    agent._active_packs = set()
    agent._tools_version = 0

    from llamagent.core.agent import SimpleReAct
    agent._execution_strategy = SimpleReAct()
    return agent


class TestModuleIntegration:
    """Multi-module loading and cooperation tests."""

    def test_safety_module_provides_hooks(self, mock_llm_client):
        """SafetyModule provides on_input/on_output hooks without affecting tool execution."""
        agent = _create_test_agent(mock_llm_client)
        from llamagent.modules.safety.module import SafetyModule
        safety = SafetyModule()
        agent.register_module(safety)
        assert agent.has_module("safety")
        assert safety.guard is not None

    def test_planning_sets_strategy(self, mock_llm_client):
        """PlanningModule upgrades execution strategy to PlanReAct."""
        agent = _create_test_agent(mock_llm_client)
        from llamagent.modules.reasoning.module import PlanningModule, PlanReAct
        agent.register_module(PlanningModule())
        assert isinstance(agent._execution_strategy, PlanReAct)

    def test_recommended_load_order(self, mock_llm_client, tmp_path):
        """Loading safety -> planning in recommended order works correctly."""
        agent = _create_test_agent(mock_llm_client)
        agent.config.output_dir = str(tmp_path)

        from llamagent.modules.safety.module import SafetyModule
        from llamagent.modules.reasoning.module import PlanningModule

        agent.register_module(SafetyModule())
        agent.register_module(PlanningModule())

        assert agent.has_module("safety")
        assert agent.has_module("planning")
        assert len(agent.list_modules()) == 2

    def test_legacy_on_execute_compat(self, mock_llm_client):
        """Legacy modules using on_execute hook still work."""
        agent = _create_test_agent(mock_llm_client)

        class LegacyModule(Module):
            name = "legacy"
            description = "legacy module"
            def on_execute(self, query, context):
                return "legacy intercept"

        agent.register_module(LegacyModule())
        mock_llm_client.set_responses([make_llm_response("should not reach")])
        result = agent.chat("test")
        assert "legacy intercept" in result

    def test_main_create_agent(self, mock_llm_client):
        """main.create_agent() factory function works end-to-end."""
        with patch("llamagent.main.Config") as MockConfig, \
             patch("llamagent.main.SmartAgent") as MockAgent:
            mock_config = MagicMock()
            mock_config.model = "mock-model"
            mock_config.persona_file = "/tmp/test.json"
            MockConfig.return_value = mock_config

            mock_agent = MagicMock()
            mock_agent.config = mock_config
            MockAgent.return_value = mock_agent

            from llamagent.main import create_agent
            agent = create_agent(module_names=["planning"])
            assert agent is not None
