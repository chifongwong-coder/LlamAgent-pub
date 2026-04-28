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
    """Context compression: summarizes old conversation history to save tokens.

    Note: The Tools module's _truncate_observation may add a persistence
    hint of the form
      "Output truncated. Full result saved to <rel_path> (<size> bytes, <lines>
       lines). Use read_files(['<rel_path>']) to read it."
    to long tool results.

    Two known interactions with this module:

    1. _compress_tool_result strategies 'head' / 'placeholder' / 'llm_summary'
       may strip the trailing hint line when compressing tool messages.
       Mitigation: the path also lives in the subsequent assistant message's
       tool_calls[].arguments, so multi-turn access is usually preserved
       without coupling Compression to the Tools hint format.

    2. compress_conversation (full-history summarization) replaces history
       prefix with a summary, dropping the assistant tool_call message that
       carried the path argument. After such compression the model can no
       longer discover the persisted-file path from history. The in-memory
       _persisted_files set is unaffected, but the model has no way to ask.
       This is an accepted limitation; users with long-context workflows
       should raise compression_threshold or rely on same-turn read_files
       follow-up before history compression triggers.
    """

    name = "compression"
    description = "Context compression via LLM summarization"

    def on_attach(self, agent):
        super().on_attach(agent)
        self._threshold_ratio = agent.config.context_compress_threshold  # 0.7
        self._keep_turns = agent.config.compress_keep_turns  # 3
        # v2.9.4: tool result compression settings
        self._tool_result_strategy = getattr(agent.config, "tool_result_strategy", "none")
        self._tool_result_max_chars = getattr(agent.config, "tool_result_max_chars", 2000)
        self._tool_result_head_lines = getattr(agent.config, "tool_result_head_lines", 10)
        self._strip_thinking = getattr(agent.config, "strip_thinking", False)

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
        # Find messages to compress (all except recent keep_turns)
        user_indices = [i for i, m in enumerate(self.agent.history) if m.get("role") == "user"]
        if len(user_indices) <= self._keep_turns:
            return

        cut_at = user_indices[-self._keep_turns]
        old_messages = self.agent.history[:cut_at]

        compress_parts = []
        if self.agent.summary:
            compress_parts.append(f"Previous summary: {self.agent.summary}")
        old_text = "\n".join(
            f"{m['role']}: {m.get('content') or ''}" for m in old_messages
        )
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

    # ----------------------------------------------------------
    # v2.9.4: trace message pre-processing
    # ----------------------------------------------------------

    def prepare_trace_message(self, msg: dict) -> dict:
        """Pre-process a trace message before writing to history.

        Applies configured compression strategies:
        - Strip thinking from assistant messages (if strip_thinking=True)
        - Compress tool results (if strategy != 'none' and content > max_chars)
        """
        role = msg.get("role")

        # Strip thinking from assistant messages
        if role == "assistant" and self._strip_thinking:
            msg.pop("reasoning_content", None)
            msg.pop("thinking_blocks", None)

        # Compress tool results
        if role == "tool":
            msg = self._compress_tool_result(msg)

        return msg

    def _compress_tool_result(self, msg: dict) -> dict:
        """Apply configured compression strategy to a tool result message."""
        content = msg.get("content") or ""
        if len(content) <= self._tool_result_max_chars:
            return msg  # Under threshold, keep as-is

        strategy = self._tool_result_strategy

        if strategy == "none":
            return msg
        elif strategy == "placeholder":
            msg["content"] = f"[Tool result ({len(content)} chars, trimmed)]"
        elif strategy == "head":
            lines = content.split("\n")
            head = "\n".join(lines[:self._tool_result_head_lines])
            msg["content"] = f"{head}\n...[trimmed, original {len(content)} chars]"
        elif strategy == "llm_summary":
            try:
                summary = self.llm.ask(
                    f"Summarize this tool output in 2-3 sentences, preserving key data:\n\n"
                    f"{content[:3000]}",
                    temperature=0.2,
                )
                if summary and not summary.startswith("[LLM"):
                    msg["content"] = f"[Summary] {summary}"
                else:
                    # Fallback to head
                    lines = content.split("\n")
                    msg["content"] = "\n".join(lines[:self._tool_result_head_lines]) + \
                        f"\n...[trimmed, original {len(content)} chars]"
            except Exception as e:
                logger.debug("Tool result summarization failed: %s", e)
                lines = content.split("\n")
                msg["content"] = "\n".join(lines[:self._tool_result_head_lines]) + \
                    f"\n...[trimmed, original {len(content)} chars]"

        return msg
