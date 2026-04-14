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

    def __init__(self, model: str, fallback_llm: LLMClient | None = None, max_retries: int = 3):
        """
        Args:
            model: Model identifier string (same as LLMClient).
            fallback_llm: Independent LLMClient for failover (optional).
            max_retries: Maximum retry attempts before failover/give up.
        """
        super().__init__(model, api_retry_count=0)
        self._fallback_llm = fallback_llm
        self._max_retries = max_retries

    def chat(self, messages, **kwargs):
        """Chat with error classification, smart retry, and model failover."""
        return self._call_with_resilience(messages, **kwargs)

    def _call_with_resilience(self, messages, **kwargs):
        """
        Retry loop with error classification and failover.

        1. Try super().chat() (single attempt, api_retry_count=0)
        2. Classify failure → decide retry / failover / give up
        3. After retries exhausted, try fallback model if should_failover
        4. If all fails, raise the original exception
        """
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                return super().chat(messages, **kwargs)
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

                # Wait before retry (cap at 30s to prevent excessive blocking)
                wait = min(classified.retry_after, 30) if classified.retry_after > 0 else min(2 ** attempt, 30)
                time.sleep(wait)

        # All retries exhausted (or non-retryable) → try failover
        if last_error and last_error.should_failover and self._fallback_llm:
            logger.warning("Failing over to fallback model: %s", self._fallback_llm.model)
            try:
                return self._fallback_llm.chat(messages, **kwargs)
            except Exception as fallback_error:
                logger.error("Failover also failed: %s", fallback_error)
                raise fallback_error

        # Give up
        raise last_error.original
