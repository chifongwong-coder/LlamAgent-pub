"""
Budget enforcement for child agents.

Budget:             Declarative resource limits (tokens, time, steps, LLM calls, artifacts).
BudgetTracker:      Runtime counter that checks limits and records usage.
BudgetedLLM:        Wrapper around LLMClient that enforces budget on every call.
BudgetExceededError: Raised when any budget limit is exceeded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Budget:
    """Declarative resource limits for a child agent execution."""

    max_tokens: int | None = None
    max_time_seconds: float | None = None
    max_steps: int | None = None
    max_llm_calls: int | None = None
    max_artifact_bytes: int | None = None


class BudgetExceededError(Exception):
    """Raised when a child agent exceeds its allocated budget."""
    pass


class BudgetTracker:
    """
    Runtime budget enforcement.

    Tracks cumulative usage across tokens, steps, LLM calls, artifact bytes,
    and wall-clock time. Call check() before each expensive operation.
    """

    def __init__(self, budget: Budget):
        self.budget = budget
        self.tokens_used: int = 0
        self.steps_used: int = 0
        self.llm_calls: int = 0
        self.artifact_bytes_used: int = 0
        self.time_started: float = time.time()

    def check(self) -> str | None:
        """
        Check all budget limits.

        Returns:
            None if all limits are within bounds, or a reason string describing
            which limit was exceeded.
        """
        b = self.budget

        if b.max_tokens is not None and self.tokens_used >= b.max_tokens:
            return f"Token budget exceeded: {self.tokens_used}/{b.max_tokens}"

        if b.max_llm_calls is not None and self.llm_calls >= b.max_llm_calls:
            return f"LLM call budget exceeded: {self.llm_calls}/{b.max_llm_calls}"

        if b.max_steps is not None and self.steps_used >= b.max_steps:
            return f"Step budget exceeded: {self.steps_used}/{b.max_steps}"

        if b.max_artifact_bytes is not None and self.artifact_bytes_used >= b.max_artifact_bytes:
            return f"Artifact size budget exceeded: {self.artifact_bytes_used}/{b.max_artifact_bytes} bytes"

        if b.max_time_seconds is not None:
            elapsed = time.time() - self.time_started
            if elapsed >= b.max_time_seconds:
                return f"Time budget exceeded: {elapsed:.1f}s/{b.max_time_seconds}s"

        return None

    def record_llm_call(self, tokens: int = 0) -> None:
        """Record one LLM call and its token usage."""
        self.llm_calls += 1
        self.tokens_used += tokens

    def record_step(self) -> None:
        """Record one execution step."""
        self.steps_used += 1

    def record_artifact(self, size_bytes: int) -> None:
        """Record artifact bytes produced."""
        self.artifact_bytes_used += size_bytes


class BudgetedLLM:
    """
    Wrapper around LLMClient that enforces budget constraints on every call.

    Delegates all LLM operations to the underlying client while checking the
    budget before each call and recording usage after each call.
    """

    def __init__(self, llm, tracker: BudgetTracker):
        """
        Args:
            llm: The underlying LLMClient instance.
            tracker: BudgetTracker that enforces limits.
        """
        self._llm = llm
        self.tracker = tracker
        # Expose model attribute for compatibility
        self.model = llm.model

    def chat(self, messages, **kwargs):
        """
        Budget-checked chat call.

        Raises BudgetExceededError if the budget is exhausted before the call.
        Records token usage after a successful call.
        """
        reason = self.tracker.check()
        if reason:
            raise BudgetExceededError(reason)

        resp = self._llm.chat(messages, **kwargs)
        tokens = self._estimate_tokens(resp)
        self.tracker.record_llm_call(tokens)
        return resp

    def ask(self, prompt: str, **kwargs) -> str:
        """
        Budget-checked single-turn Q&A returning plain text.

        Raises BudgetExceededError if the budget is exhausted before the call.
        """
        reason = self.tracker.check()
        if reason:
            raise BudgetExceededError(reason)

        result = self._llm.ask(prompt, **kwargs)
        tokens = len(result) // 4 + 1  # Rough estimate
        self.tracker.record_llm_call(tokens)
        return result

    def ask_json(self, prompt: str, **kwargs) -> dict:
        """
        Budget-checked single-turn Q&A returning parsed JSON.

        Raises BudgetExceededError if the budget is exhausted before the call.
        """
        reason = self.tracker.check()
        if reason:
            raise BudgetExceededError(reason)

        result = self._llm.ask_json(prompt, **kwargs)
        tokens = len(str(result)) // 4 + 1  # Rough estimate
        self.tracker.record_llm_call(tokens)
        return result

    def count_tokens(self, messages) -> int:
        """Delegate token counting to the underlying LLM (no budget check)."""
        return self._llm.count_tokens(messages)

    @staticmethod
    def _estimate_tokens(response) -> int:
        """
        Rough token estimate from an LLM response object.

        Tries to read usage.total_tokens from the response; falls back to
        estimating from content length.
        """
        # Prefer usage info from the response
        try:
            usage = response.usage
            if usage and hasattr(usage, "total_tokens") and usage.total_tokens:
                return usage.total_tokens
        except (AttributeError, TypeError):
            pass

        # Fallback: estimate from content length
        try:
            content = response.choices[0].message.content or ""
            return len(content) // 4 + 1
        except (AttributeError, IndexError, TypeError):
            return 0
