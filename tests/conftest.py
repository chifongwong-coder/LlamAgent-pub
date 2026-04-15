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
from llamagent.core.agent import LlamAgent


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


def make_stream_chunks(content: str, chunk_size: int = 5):
    """Build mock streaming chunks for a text-only response (no tool calls)."""
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = SimpleNamespace()
        chunk.choices = [SimpleNamespace(delta=SimpleNamespace(
            content=content[i:i+chunk_size], tool_calls=None
        ), finish_reason=None)]
        chunks.append(chunk)
    # Final chunk with finish_reason
    final = SimpleNamespace()
    final.choices = [SimpleNamespace(delta=SimpleNamespace(
        content=None, tool_calls=None
    ), finish_reason="stop")]
    chunks.append(final)
    return chunks


def make_stream_tool_call_chunks(tool_name: str, arguments: dict, call_id: str = "call_1"):
    """Build mock streaming chunks for a tool call response."""
    import json
    args_str = json.dumps(arguments)
    chunks = []
    # First chunk: tool call id + name
    tc_delta = SimpleNamespace()
    tc_delta.index = 0
    tc_delta.id = call_id
    tc_delta.function = SimpleNamespace(name=tool_name, arguments="")
    chunk = SimpleNamespace()
    chunk.choices = [SimpleNamespace(delta=SimpleNamespace(
        content=None, tool_calls=[tc_delta]
    ), finish_reason=None)]
    chunks.append(chunk)
    # Arguments in fragments
    for i in range(0, len(args_str), 10):
        tc_delta = SimpleNamespace()
        tc_delta.index = 0
        tc_delta.id = None
        tc_delta.function = SimpleNamespace(name=None, arguments=args_str[i:i+10])
        chunk = SimpleNamespace()
        chunk.choices = [SimpleNamespace(delta=SimpleNamespace(
            content=None, tool_calls=[tc_delta]
        ), finish_reason=None)]
        chunks.append(chunk)
    # Final chunk
    final = SimpleNamespace()
    final.choices = [SimpleNamespace(delta=SimpleNamespace(
        content=None, tool_calls=None
    ), finish_reason="tool_calls")]
    chunks.append(final)
    return chunks


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
        _stream_responses = []
        _call_index = [0]
        _stream_index = [0]

        def _side_effect(**kwargs):
            if kwargs.get("stream"):
                idx = _stream_index[0]
                if idx < len(_stream_responses):
                    _stream_index[0] += 1
                    r = _stream_responses[idx]
                    if isinstance(r, Exception):
                        raise r
                    return iter(r)  # return iterator over chunks
                return iter(make_stream_chunks("default stream response"))
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

        def set_stream_responses(responses: list):
            """Set streaming responses. Each item is a list of chunks."""
            _stream_responses.clear()
            _stream_responses.extend(responses)
            _stream_index[0] = 0

        client.set_responses = set_responses
        client.set_stream_responses = set_stream_responses
        client._mock_completion = mock_completion
        yield client


@pytest.fixture
def bare_agent(mock_llm_client):
    """LlamAgent with no modules loaded; LLM is mocked."""
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
    config.tool_result_persist_threshold = 0
    config.max_context_tokens = 8192
    config.memory_mode = "off"
    config.memory_recall_mode = "tool"
    config.memory_fact_fallback = "text"
    config.memory_recall_top_k = 5
    config.memory_auto_recall_max_inject = 3
    config.memory_auto_recall_threshold = 0.35
    config.memory_consolidation_interval = 24
    config.memory_consolidation_min_count = 20
    config.embedding_provider = "chromadb"
    config.embedding_model = ""
    config.retrieval_persist_dir = "/tmp/llamagent_test_retrieval"
    config.rag_top_k = 3
    config.chunk_size = 500
    config.rag_retrieval_mode = "hybrid"
    config.rag_rerank_enabled = False
    config.chroma_dir = "/tmp/llamagent_test_retrieval"
    config.api_retry_count = 0
    config.reflection_write_mode = "off"
    config.reflection_read_mode = "off"
    config.reflection_backend = "rag"
    config.reflection_fs_dir = None
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
    config.hooks_config = None
    config.module_models = {}
    config.persistence_enabled = False
    config.persistence_auto_restore = True
    config.persistence_dir = None
    config.tool_result_strategy = "none"
    config.tool_result_max_chars = 2000
    config.tool_result_head_lines = 10
    config.strip_thinking = False
    config.child_agent_runner = "inline"
    config.child_agent_max_children = 20
    config.child_agent_role_models = {}
    config.child_agent_auto_memorize = True
    config.approval_mode = "persistent"
    config.auto_approve = False
    config.authorization_scopes = []
    config.fallback_model = None
    config.resilience_max_retries = 3
    config.routing_simple_model = None

    import uuid
    agent = LlamAgent.__new__(LlamAgent)
    agent.config = config
    agent.agent_id = uuid.uuid4().hex[:12]
    agent.persona = None
    agent.llm = mock_llm_client
    agent._llm_cache = {config.model: mock_llm_client}
    agent.modules = {}
    agent.history = []
    agent.summary = None
    agent.conversation = agent.history
    agent._execution_strategy = None
    import os
    agent.confirm_handler = None
    agent.interaction_handler = None
    agent._confirm_wait_time = 0.0
    agent.project_dir = os.path.realpath(os.getcwd())
    agent.playground_dir = os.path.realpath(os.path.join(agent.project_dir, "llama_playground"))
    agent.tool_executor = None
    agent._tools = {}
    agent._active_packs = set()
    agent._tools_version = 0
    agent._hooks = {}
    agent._session_started = False
    agent._in_hook = False
    agent.mode = "interactive"
    agent._controller = None
    agent._current_task_id = None
    agent._abort = False
    agent._open_questions_buffer = []
    agent._interactive_config = {k: getattr(config, k) for k in LlamAgent._MODE_KEYS}

    from llamagent.core.agent import SimpleReAct
    from llamagent.core.authorization import AuthorizationEngine
    agent._execution_strategy = SimpleReAct()
    agent._authorization_engine = AuthorizationEngine(agent)
    return agent
