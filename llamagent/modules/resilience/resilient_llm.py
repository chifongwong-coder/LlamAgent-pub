"""
ResilientLLM: LLMClient subclass with error classification, smart retry, and model failover.

Inherits from LLMClient and only overrides chat(). ask() and ask_json() are inherited
and automatically get resilience because they call self.chat() internally. chat_stream()
and count_tokens() are also inherited but do NOT get resilience (they call litellm
directly, not self.chat()). Behavior fully consistent with vanilla LLMClient.
"""

from __future__ import annotations

import logging
import time

from llamagent.core.llm import LLMClient
from llamagent.modules.resilience.classifier import ClassifiedError, classify

logger = logging.getLogger(__name__)


class ResilientLLM(LLMClient):
    """
    LLMClient subclass that adds error classification, smart retry, and model failover.

    Only chat() is overridden. ask() and ask_json() get resilience through inheritance
    (they call self.chat()). chat_stream() and count_tokens() are inherited but do NOT
    get resilience (they call litellm directly).
    """

    def __init__(
        self,
        model: str,
        fallback_llm: LLMClient | None = None,
        max_retries: int = 3,
        simple_llm: LLMClient | None = None,
    ):
        """
        Args:
            model: Model identifier string (same as LLMClient).
            fallback_llm: Independent LLMClient for failover (optional).
            max_retries: Maximum retry attempts before failover/give up.
            simple_llm: Cheap model for simple queries (optional, smart routing).
        """
        super().__init__(model, api_retry_count=0)
        self._fallback_llm = fallback_llm
        self._max_retries = max(0, max_retries)
        self._primary_cooldown_until: float = 0  # Turn-scoped: use fallback during cooldown
        self._simple_llm = simple_llm

    def chat(self, messages, **kwargs):
        """Chat with smart routing, error classification, retry, and failover."""
        # Smart routing: simple queries → cheap model (skip resilience overhead)
        if self._simple_llm and not kwargs.get("tools") and self._is_simple_query(messages):
            try:
                return self._simple_llm.chat(messages, **kwargs)
            except Exception:
                logger.debug("Simple model failed, falling through to primary")
        return self._call_with_resilience(messages, **kwargs)

    @staticmethod
    def _is_simple_query(messages: list[dict]) -> bool:
        """Check if the last user message is a simple query (short, no code, no URLs)."""
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if not last_user:
            return False
        content = last_user.get("content", "")
        if not isinstance(content, str):
            return False
        return len(content) < 200 and "```" not in content and "http" not in content

    def _call_with_resilience(self, messages, **kwargs):
        """
        Retry loop with error classification, failover, and turn-scoped recovery.

        Turn-scoped failover: after a successful failover, the primary model enters
        a cooldown period. During cooldown, fallback is used directly (avoiding
        repeated failures on primary). After cooldown, primary is tried again.

        Flow:
        1. If primary is in cooldown and fallback exists, use fallback directly
        2. Try super().chat() with retry loop
        3. After retries exhausted, try fallback model if should_failover
        4. On successful failover, set primary cooldown
        5. On successful primary call, clear cooldown
        """
        # Turn-scoped: during cooldown, use fallback directly
        if self._fallback_llm and time.time() < self._primary_cooldown_until:
            logger.info("Primary in cooldown, using fallback model: %s", self._fallback_llm.model)
            try:
                return self._fallback_llm.chat(messages, **kwargs)
            except Exception:
                logger.warning("Fallback failed during cooldown, trying primary")
                # Fall through to normal primary retry

        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                result = super().chat(messages, **kwargs)
                self._primary_cooldown_until = 0  # Primary recovered, clear cooldown
                return result
            except Exception as e:
                # Defensive: if classifier itself fails, treat as unknown/retryable
                try:
                    classified = classify(e)
                except Exception:
                    logger.error("Error classifier failed, treating as unknown retryable")
                    classified = ClassifiedError(
                        reason="unknown",
                        retryable=True,
                        should_failover=False,
                        retry_after=0,
                        original=e,
                    )
                last_error = classified
                logger.warning(
                    "LLM call failed (attempt %d/%d): [%s] %s",
                    attempt + 1,
                    self._max_retries + 1,
                    classified.reason,
                    e,
                )

                if not classified.retryable:
                    break  # Non-retryable, jump to failover check

                # Wait before next retry (skip sleep on last attempt)
                if attempt < self._max_retries:
                    wait = min(classified.retry_after, 30) if classified.retry_after > 0 else min(2 ** attempt, 30)
                    time.sleep(wait)

        # All retries exhausted (or non-retryable) → try failover
        if last_error and last_error.should_failover and self._fallback_llm:
            logger.warning("Failing over to fallback model: %s", self._fallback_llm.model)
            try:
                result = self._fallback_llm.chat(messages, **kwargs)
                # Successful failover → set primary cooldown
                cooldown = min(last_error.retry_after, 300) if last_error.retry_after > 0 else 60
                self._primary_cooldown_until = time.time() + cooldown
                logger.info("Primary cooldown set for %.0fs", cooldown)
                return result
            except Exception as fallback_error:
                logger.error("Failover also failed: %s", fallback_error)
                raise fallback_error

        # Give up
        raise last_error.original
