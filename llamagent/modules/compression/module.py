"""
CompressionModule: context compression via LLM summarization.

When loaded, monitors conversation token count on each turn (via on_input).
If tokens exceed the configured threshold (default 70%), compresses old messages
into a summary via LLM call and updates the agent's history through the public
compress_conversation() method.

Without this module, the agent relies on a hard-limit fallback (clear conversation
when tokens hit 100% of max_context_tokens).
"""

import logging

from llamagent.core.agent import Module

logger = logging.getLogger(__name__)


class CompressionModule(Module):
    """Context compression: summarizes old conversation history to save tokens."""

    name = "compression"
    description = "Context compression via LLM summarization"

    def on_attach(self, agent):
        super().on_attach(agent)
        self._threshold_ratio = agent.config.context_compress_threshold  # 0.7
        self._keep_turns = agent.config.compress_keep_turns  # 3

    def on_input(self, user_input: str) -> str:
        """Check context size and compress if needed (side-effect in on_input)."""
        if not self.agent.history:
            return user_input

        try:
            token_count = self.llm.count_tokens(self.agent.history)
            threshold = int(self.agent.config.max_context_tokens * self._threshold_ratio)

            if token_count >= threshold:
                self._compress()
        except Exception as e:
            logger.warning("Compression check error: %s", e)

        return user_input  # input unchanged

    def _compress(self):
        """Generate summary and call agent.compress_conversation()."""
        keep_messages = self._keep_turns * 2
        if len(self.agent.history) <= keep_messages:
            return

        old_messages = self.agent.history[:-keep_messages]

        compress_parts = []
        if self.agent.summary:
            compress_parts.append(f"Previous summary: {self.agent.summary}")
        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)
        compress_parts.append(old_text)

        try:
            new_summary = self.llm.ask(
                "Please compress the following conversation history into a concise summary, "
                "retaining key information (user preferences, important decisions, task progress, key facts):\n\n"
                + "\n".join(compress_parts),
                temperature=0.3,
            )
        except Exception as e:
            logger.warning("Compression LLM call failed: %s", e)
            return

        if new_summary and not new_summary.startswith("[LLM"):
            self.agent.compress_conversation(new_summary, self._keep_turns)
            logger.info("Conversation compressed via CompressionModule")
