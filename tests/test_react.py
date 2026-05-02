"""
ReAct loop tests: tool calling, multi-step execution, edge cases, and weak model degradation.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from conftest import make_llm_response, make_tool_call


class TestReActLoopFlow:
    """End-to-end ReAct loop flow: single tool, multi-step, parallel, max steps,
    duplicate detection, observation truncation, context overflow, interrupt."""

    def test_react_loop_flow(self, bare_agent, mock_llm_client):
        """Complete ReAct loop coverage: single tool call, multi-step, parallel calls,
        max_steps protection, duplicate detection, observation truncation,
        context window overflow, and should_continue interrupt."""
        # --- Single tool call flow ---
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
        roles = [m["role"] if isinstance(m, dict) else m.get("role") for m in msgs]
        assert "tool" in roles

        # --- Multi-step tool calls ---
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[
                make_tool_call("search", {"q": "data"}, "c1"),
            ]),
            make_llm_response("", tool_calls=[
                make_tool_call("analyze", {"data": "raw"}, "c2"),
            ]),
            make_llm_response("Analysis complete: the trend is upward"),
        ])
        tools2 = [
            {"type": "function", "function": {"name": "search"}},
            {"type": "function", "function": {"name": "analyze"}},
        ]
        dispatch2 = lambda name, args: f"{name} completed successfully"
        msgs2 = [{"role": "user", "content": "analyze the data"}]

        result2 = bare_agent.run_react(msgs2, tools2, dispatch2)
        assert result2.status == "completed"
        assert result2.steps_used == 2

        # --- Parallel tool calls ---
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[
                make_tool_call("search", {"q": "A"}, "c1"),
                make_tool_call("search", {"q": "B"}, "c2"),
            ]),
            make_llm_response("Combined results from both searches"),
        ])
        tools3 = [{"type": "function", "function": {"name": "search"}}]
        calls = []
        dispatch3 = lambda name, args: (calls.append(args), f"result_{args['q']}")[1]
        msgs3 = [{"role": "user", "content": "search A and B"}]

        result3 = bare_agent.run_react(msgs3, tools3, dispatch3)
        assert result3.status == "completed"
        assert len(calls) == 2
        tool_msgs3 = [m for m in msgs3 if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs3) >= 2
        ids = {m["tool_call_id"] for m in tool_msgs3}
        assert "c1" in ids and "c2" in ids

        # --- Max steps protection ---
        bare_agent.config.max_react_steps = 3
        responses = [
            make_llm_response("", tool_calls=[make_tool_call("loop", {"i": str(i)}, f"c{i}")])
            for i in range(10)
        ]
        mock_llm_client.set_responses(responses)
        tools4 = [{"type": "function", "function": {"name": "loop"}}]
        msgs4 = [{"role": "user", "content": "keep going"}]
        result4 = bare_agent.run_react(msgs4, tools4, lambda n, a: "looping")
        assert result4.status == "max_steps"

        # --- Duplicate action detection ---
        bare_agent.config.max_duplicate_actions = 2
        bare_agent.config.max_react_steps = 10
        dup_responses = [
            make_llm_response("", tool_calls=[make_tool_call("same", {"x": "1"}, f"c{i}")])
            for i in range(5)
        ]
        mock_llm_client.set_responses(dup_responses)
        tools5 = [{"type": "function", "function": {"name": "same"}}]
        msgs5 = [{"role": "user", "content": "do"}]
        result5 = bare_agent.run_react(msgs5, tools5, lambda n, a: "same result")
        assert result5.status == "error"
        assert "Duplicate" in result5.text

        # --- Observation truncation ---
        bare_agent.config.max_observation_tokens = 10
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[make_tool_call("big", {}, "c1")]),
            make_llm_response("ok"),
        ])
        tools6 = [{"type": "function", "function": {"name": "big"}}]
        msgs6 = [{"role": "user", "content": "go"}]
        with patch.object(bare_agent.llm, "count_tokens",
                          side_effect=lambda x: len(x) if isinstance(x, str) else 100):
            bare_agent.run_react(msgs6, tools6, lambda n, a: "A" * 10000)
        tool_msgs6 = [m for m in msgs6 if isinstance(m, dict) and m.get("role") == "tool"]
        assert tool_msgs6 and "truncated" in tool_msgs6[0]["content"]

        # --- Context window overflow ---
        # v3.5.2: agent.py only catches LiteLLM's real exception class via
        # isinstance(_CONTEXT_WINDOW_ERRORS); a locally-defined class with
        # the same name no longer matches. Use the real one.
        import litellm
        mock_llm_client.set_responses([
            litellm.ContextWindowExceededError(
                message="too big", model="mock-model", llm_provider="mock",
            )
        ])
        tools7 = [{"type": "function", "function": {"name": "t"}}]
        msgs7 = [{"role": "user", "content": "huge query"}]
        result7 = bare_agent.run_react(msgs7, tools7, lambda n, a: "")
        assert result7.status == "context_overflow"

        # --- should_continue interrupt ---
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[make_tool_call("act", {}, "c1")]),
        ])
        tools8 = [{"type": "function", "function": {"name": "act"}}]
        msgs8 = [{"role": "user", "content": "go"}]
        result8 = bare_agent.run_react(
            msgs8, tools8, lambda n, a: "ok",
            should_continue=lambda: "replanned",
        )
        assert result8.status == "interrupted"
        assert result8.reason == "replanned"


class TestWeakModelDegradation:
    """Framework fault tolerance for typical weak model errors."""

    def test_weak_model_degradation(self, bare_agent, mock_llm_client, caplog):
        """Unknown tool name shows error + available tools; text and tool_calls
        coexist; plain text arguments fall back to {}."""
        # --- Unknown tool name ---
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

        # --- Text and tool_calls coexist ---
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
        msgs2 = [{"role": "user", "content": "what is the answer"}]
        result2 = bare_agent.run_react(
            msgs2, [{"type": "function", "function": {"name": "search"}}], dispatch,
        )
        assert called[0] is True
        assert result2.status == "completed"

        # --- Arguments plain text fallback ---
        tc = make_tool_call("search", {}, "c1")
        tc.function.arguments = "please search for weather"
        mock_llm_client.set_responses([
            make_llm_response("", tool_calls=[tc]),
            make_llm_response("ok"),
        ])
        msgs3 = [{"role": "user", "content": "weather"}]
        with caplog.at_level(logging.WARNING):
            result3 = bare_agent.run_react(
                msgs3, [{"type": "function", "function": {"name": "search"}}],
                lambda n, a: f"args={a}",
            )
        assert result3.status == "completed"
        assert any("JSON parsing failed" in r.message for r in caplog.records)
