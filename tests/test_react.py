"""
ReAct loop tests: tool calling, multi-step execution, edge cases, and weak model degradation.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from conftest import make_llm_response, make_tool_call


class TestReActLoop:
    """End-to-end ReAct loop flow tests."""

    def test_single_tool_call_flow(self, bare_agent, mock_llm_client):
        """Full flow: user query -> tool call -> observation -> final answer."""
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[
                make_tool_call("search", {"q": "weather"}, "call_1"),
            ]),
            make_llm_response("The weather is sunny today"),
        ])
        tools = [{"type": "function", "function": {"name": "search"}}]
        dispatch = lambda name, args: "Search result: sunny, 22 degrees"
        msgs = [{"role": "user", "content": "How is the weather?"}]

        result = bare_agent.run_react(msgs, tools, dispatch)
        assert result.status == "completed"
        assert "sunny" in result.text

        # Verify message sequence: user -> assistant(tool_call) -> tool(obs) -> assistant(answer)
        roles = [m["role"] if isinstance(m, dict) else m.get("role") for m in msgs]
        assert "tool" in roles

    def test_multi_step_tool_calls(self, bare_agent, mock_llm_client):
        """Multi-round: search -> analyze -> final answer, tracks steps_used."""
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[
                make_tool_call("search", {"q": "data"}, "c1"),
            ]),
            make_llm_response("", tool_calls=[
                make_tool_call("analyze", {"data": "raw"}, "c2"),
            ]),
            make_llm_response("Analysis complete: the trend is upward"),
        ])
        tools = [
            {"type": "function", "function": {"name": "search"}},
            {"type": "function", "function": {"name": "analyze"}},
        ]
        dispatch = lambda name, args: f"{name} completed successfully"
        msgs = [{"role": "user", "content": "analyze the data"}]

        result = bare_agent.run_react(msgs, tools, dispatch)
        assert result.status == "completed"
        assert result.steps_used == 2

    def test_parallel_tool_calls(self, bare_agent, mock_llm_client):
        """Single LLM response with multiple tool_calls executed in parallel."""
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[
                make_tool_call("search", {"q": "A"}, "c1"),
                make_tool_call("search", {"q": "B"}, "c2"),
            ]),
            make_llm_response("Combined results from both searches"),
        ])
        tools = [{"type": "function", "function": {"name": "search"}}]
        calls = []
        dispatch = lambda name, args: (calls.append(args), f"result_{args['q']}")[1]
        msgs = [{"role": "user", "content": "search A and B"}]

        result = bare_agent.run_react(msgs, tools, dispatch)
        assert result.status == "completed"
        assert len(calls) == 2
        tool_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) >= 2
        ids = {m["tool_call_id"] for m in tool_msgs}
        assert "c1" in ids and "c2" in ids

    def test_max_steps_protection(self, bare_agent, mock_llm_client):
        """Loop terminates after max_react_steps even if LLM keeps calling tools."""
        bare_agent.config.max_react_steps = 3
        responses = [
            make_llm_response("", tool_calls=[make_tool_call("loop", {"i": str(i)}, f"c{i}")])
            for i in range(10)
        ]
        mock_llm_client.set_responses(responses)

        tools = [{"type": "function", "function": {"name": "loop"}}]
        msgs = [{"role": "user", "content": "keep going"}]
        result = bare_agent.run_react(msgs, tools, lambda n, a: "looping")
        assert result.status == "max_steps"

    def test_duplicate_action_detection(self, bare_agent, mock_llm_client):
        """Identical (tool_name, args) repeated beyond threshold triggers abort."""
        bare_agent.config.max_duplicate_actions = 2
        bare_agent.config.max_react_steps = 10
        responses = [
            make_llm_response("", tool_calls=[make_tool_call("same", {"x": "1"}, f"c{i}")])
            for i in range(5)
        ]
        mock_llm_client.set_responses(responses)

        tools = [{"type": "function", "function": {"name": "same"}}]
        msgs = [{"role": "user", "content": "do"}]
        result = bare_agent.run_react(msgs, tools, lambda n, a: "same result")
        assert result.status == "error"
        assert "Duplicate" in result.text

    def test_observation_truncation(self, bare_agent, mock_llm_client):
        """Tool returning oversized output gets truncated."""
        bare_agent.config.max_observation_tokens = 10
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[make_tool_call("big", {}, "c1")]),
            make_llm_response("ok"),
        ])
        tools = [{"type": "function", "function": {"name": "big"}}]
        msgs = [{"role": "user", "content": "go"}]

        with patch.object(bare_agent.llm, "count_tokens",
                          side_effect=lambda x: len(x) if isinstance(x, str) else 100):
            bare_agent.run_react(msgs, tools, lambda n, a: "A" * 10000)

        tool_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
        assert tool_msgs and "truncated" in tool_msgs[0]["content"]

    def test_context_window_overflow(self, bare_agent, mock_llm_client):
        """ContextWindowExceededError returns context_overflow status."""
        class ContextWindowExceededError(Exception):
            pass

        mock_llm_client.set_responses([ContextWindowExceededError("too big")])
        tools = [{"type": "function", "function": {"name": "t"}}]
        msgs = [{"role": "user", "content": "huge query"}]
        result = bare_agent.run_react(msgs, tools, lambda n, a: "")
        assert result.status == "context_overflow"

    def test_should_continue_interrupt(self, bare_agent, mock_llm_client):
        """External interrupt via should_continue callback."""
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[make_tool_call("act", {}, "c1")]),
        ])
        tools = [{"type": "function", "function": {"name": "act"}}]
        msgs = [{"role": "user", "content": "go"}]

        result = bare_agent.run_react(
            msgs, tools, lambda n, a: "ok",
            should_continue=lambda: "replanned",
        )
        assert result.status == "interrupted"
        assert result.reason == "replanned"


class TestWeakModelDegradation:
    """Framework fault tolerance for typical weak model errors."""

    def test_unknown_tool_name(self, bare_agent, mock_llm_client):
        """Model misspells tool name -> observation shows error + available tools."""
        bare_agent.register_tool("search", lambda q="": f"found: {q}", "Search")
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[make_tool_call("serach", {"q": "x"}, "c1")]),
            make_llm_response("Sorry, let me answer directly"),
        ])
        msgs = [{"role": "user", "content": "find something"}]
        result = bare_agent.run_react(
            msgs, [{"type": "function", "function": {"name": "search"}}],
            tool_dispatch=bare_agent.call_tool,
        )
        assert result.status == "completed"
        tool_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
        assert "does not exist" in tool_msgs[0]["content"]

    def test_text_and_tool_calls_coexist(self, bare_agent, mock_llm_client):
        """Model returns both text and tool_calls -> framework executes tools."""
        mock_llm_client.set_responses([
            make_llm_response("I'm unsure", tool_calls=[
                make_tool_call("search", {"q": "answer"}, "c1"),
            ]),
            make_llm_response("Found it: 42"),
        ])
        called = [False]
        def dispatch(n, a):
            called[0] = True
            return "42"
        msgs = [{"role": "user", "content": "what is the answer"}]
        result = bare_agent.run_react(
            msgs, [{"type": "function", "function": {"name": "search"}}], dispatch,
        )
        assert called[0] is True
        assert result.status == "completed"

    def test_arguments_plain_text_fallback(self, bare_agent, mock_llm_client, caplog):
        """Model sends natural language instead of JSON in arguments -> fallback to {}."""
        tc = make_tool_call("search", {}, "c1")
        tc.function.arguments = "please search for weather"
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[tc]),
            make_llm_response("ok"),
        ])
        msgs = [{"role": "user", "content": "weather"}]
        with caplog.at_level(logging.WARNING):
            result = bare_agent.run_react(
                msgs, [{"type": "function", "function": {"name": "search"}}],
                lambda n, a: f"args={a}",
            )
        assert result.status == "completed"
        assert any("JSON parsing failed" in r.message for r in caplog.records)
