"""
Chat pipeline tests: hook execution order, safety interception, history management, and context compression.
"""

import logging
from unittest.mock import patch, MagicMock

import pytest

from llamagent.core.agent import Module, ExecutionStrategy
from conftest import make_llm_response


# ============================================================
# Helper modules
# ============================================================

class TrackingModule(Module):
    """Records hook call order for verification."""

    def __init__(self, name: str, call_log: list):
        self.name = name
        self.description = f"tracking {name}"
        self._log = call_log

    def on_input(self, user_input):
        self._log.append(f"{self.name}.on_input")
        return user_input

    def on_context(self, query, context):
        self._log.append(f"{self.name}.on_context")
        return context

    def on_output(self, response):
        self._log.append(f"{self.name}.on_output")
        return response


class BlockerModule(Module):
    """Blocks dangerous input by returning empty string."""
    name = "blocker"
    description = "safety blocker"

    def on_input(self, user_input):
        if "danger" in user_input:
            return ""
        return user_input


class InputFilterModule(Module):
    """Replaces profanity in input."""
    name = "filter"
    description = "input filter"

    def on_input(self, user_input):
        return user_input.replace("profanity", "***")


# ============================================================
# Tests
# ============================================================

class TestChatPipeline:
    """Consolidated chat pipeline flow tests."""

    def test_chat_pipeline_flow(self, bare_agent, mock_llm_client):
        """Full pipeline: hook order, context injection, input filtering, history, and strategy."""
        # --- Hook pipeline order with 2 modules ---
        mock_llm_client.set_responses([make_llm_response("reply")])
        log = []
        bare_agent.register_module(TrackingModule("A", log))
        bare_agent.register_module(TrackingModule("B", log))
        bare_agent.chat("test")

        # on_input/on_context: forward order (A, B)
        # on_output: reverse order (B, A) -- onion model
        assert log == [
            "A.on_input", "B.on_input",
            "A.on_context", "B.on_context",
            "B.on_output", "A.on_output",
        ]

        # --- Input filter writes processed text to history ---
        mock_llm_client.set_responses([make_llm_response("ok")])
        bare_agent.modules.clear()
        bare_agent.history.clear()
        bare_agent.register_module(InputFilterModule())
        bare_agent.chat("hello profanity world")
        assert bare_agent.history[0]["content"] == "hello *** world"

        # --- History trimming ---
        bare_agent.modules.clear()
        bare_agent.history.clear()
        bare_agent.config.context_window_size = 2

        for i in range(4):
            mock_llm_client.set_responses([make_llm_response(f"reply{i}")])
            bare_agent.chat(f"msg{i}")

        assert len(bare_agent.history) <= 4
        contents = [m["content"] for m in bare_agent.history]
        assert "msg0" not in contents  # oldest trimmed

        # --- Custom ExecutionStrategy is invoked ---
        class SpyStrategy(ExecutionStrategy):
            called = False
            def execute(self, query, context, agent):
                SpyStrategy.called = True
                return "spy result"

        bare_agent.set_execution_strategy(SpyStrategy())
        result = bare_agent.chat("test")
        assert SpyStrategy.called
        assert result == "spy result"

        # --- Context overflow recovery ---
        # v3.5.2: agent.py's scoped exception policy matches the real
        # litellm.ContextWindowExceededError class (isinstance check), not
        # any locally-defined Exception subclass with the same name.
        import litellm
        bare_agent.set_execution_strategy(None)
        from llamagent.core.agent import SimpleReAct
        bare_agent._execution_strategy = SimpleReAct()
        mock_llm_client.set_responses([
            litellm.ContextWindowExceededError(
                message="too long", model="mock-model", llm_provider="mock",
            )
        ])
        result = bare_agent.chat("very long query")
        assert result is not None
        assert isinstance(result, str)

    def test_safety_and_blocked(self, bare_agent, mock_llm_client):
        """Safety interception skips LLM and sets PipelineOutcome blocked state."""
        from llamagent.core.hooks import HookEvent

        bare_agent.register_module(BlockerModule())

        # --- on_input returns empty -> LLM never called ---
        result = bare_agent.chat("danger operation")
        assert "cannot process" in result
        assert mock_llm_client._mock_completion.call_count == 0

        # --- POST_CHAT hook receives blocked=True ---
        post_chat_data = {}
        bare_agent.register_hook(HookEvent.POST_CHAT, lambda ctx: post_chat_data.update(ctx.data))
        result = bare_agent.chat("danger operation")

        assert "cannot process" in result
        assert post_chat_data.get("blocked") is True
        assert post_chat_data.get("blocked_by") == "safety"
        assert post_chat_data.get("completed") is False
        assert mock_llm_client._mock_completion.call_count == 0
