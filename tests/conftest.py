"""
Shared fixtures for public tests.

Mock strategy: only mock litellm.completion(); all framework-internal methods run real logic.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llamagent.core.config import Config
from llamagent.core.llm import LLMClient
from llamagent.core.agent import SmartAgent


# ============================================================
# Mock LLM Response Builders
# ============================================================

def make_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    """Build a mock tool_call object."""
    import json
    tc = SimpleNamespace()
    tc.id = call_id
    tc.type = "function"
    tc.function = SimpleNamespace()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def make_llm_response(content: str = "", tool_calls: list | None = None):
    """Build a mock LLM response simulating litellm.completion() output."""
    message = SimpleNamespace()
    message.content = content
    message.tool_calls = tool_calls

    def _model_dump():
        d = {"role": "assistant", "content": content}
        if tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return d

    message.model_dump = _model_dump
    choice = SimpleNamespace()
    choice.message = message
    resp = SimpleNamespace()
    resp.choices = [choice]
    return resp


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_llm_client():
    """LLMClient with chat() mocked. Use client.set_responses([...]) to preset responses."""
    with patch("llamagent.core.llm._LITELLM_AVAILABLE", True), \
         patch("llamagent.core.llm.completion") as mock_completion:

        client = LLMClient.__new__(LLMClient)
        client.model = "mock-model"
        client.api_retry_count = 0
        client.max_context_tokens = 8192

        _responses = []
        _call_index = [0]

        def _side_effect(**kwargs):
            idx = _call_index[0]
            if idx < len(_responses):
                _call_index[0] += 1
                r = _responses[idx]
                if isinstance(r, Exception):
                    raise r
                return r
            return make_llm_response("default response")

        mock_completion.side_effect = _side_effect

        def set_responses(responses: list):
            _responses.clear()
            _responses.extend(responses)
            _call_index[0] = 0

        client.set_responses = set_responses
        client._mock_completion = mock_completion
        yield client


@pytest.fixture
def bare_agent(mock_llm_client):
    """SmartAgent with no modules loaded; LLM is mocked."""
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
    config.skill_dirs = []
    config.skill_max_active = 2
    config.skill_llm_fallback = False
    config.job_default_timeout = 300.0
    config.job_max_active = 10
    config.job_profiles = {}
    config.workspace_id = None

    agent = SmartAgent.__new__(SmartAgent)
    agent.config = config
    agent.persona = None
    agent.llm = mock_llm_client
    agent.modules = {}
    agent.history = []
    agent.summary = None
    agent.conversation = agent.history
    agent._execution_strategy = None
    import os
    agent.confirm_handler = None
    agent.project_dir = os.path.realpath(os.getcwd())
    agent.playground_dir = os.path.realpath(os.path.join(agent.project_dir, "llama_playground"))
    agent.tool_executor = None
    agent._tools = {}
    agent._tools_version = 0

    from llamagent.core.agent import SimpleReAct
    agent._execution_strategy = SimpleReAct()
    return agent
