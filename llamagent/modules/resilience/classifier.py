"""
Error classification for LLM call failures.

Classifies raw exceptions into structured ClassifiedError with reason,
retryability, failover hint, and retry-after delay. Five levels of
classification, from most reliable to least:

1. Known exception types (ContextWindowExceededError)
2. HTTP status codes (429, 401, 500, etc.)
3. Error message pattern matching
4. Standard exception types (ConnectionError, TimeoutError)
5. Fallback (unknown, retryable)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedError:
    """Structured classification of an LLM call failure."""

    reason: str  # rate_limit / auth / context_overflow / billing / server_error / network / unknown
    retryable: bool
    should_failover: bool
    retry_after: float  # Suggested wait in seconds (0 = use default backoff)
    original: Exception


def classify(error: Exception) -> ClassifiedError:
    """
    Classify an LLM call exception into a structured error.

    Five-level classification (most reliable first):
    1. Known exception types (ContextWindowExceededError by class name)
    2. HTTP status code
    3. Error message pattern matching
    4. Standard exception types (ConnectionError, TimeoutError)
    5. Fallback
    """
    # Level 1: Known exception types (most reliable, no string dependency)
    if "ContextWindow" in type(error).__name__:
        return ClassifiedError(
            reason="context_overflow",
            retryable=False,
            should_failover=True,
            retry_after=0,
            original=error,
        )

    # Level 2: HTTP status code
    status = _extract_status(error)
    if status in (401, 403):
        return ClassifiedError(
            reason="auth",
            retryable=False,
            should_failover=False,
            retry_after=0,
            original=error,
        )
    if status == 429:
        retry_after = _extract_retry_after(error)
        return ClassifiedError(
            reason="rate_limit",
            retryable=True,
            should_failover=False,
            retry_after=retry_after,
            original=error,
        )
    if status in (500, 502, 503):
        return ClassifiedError(
            reason="server_error",
            retryable=True,
            should_failover=True,
            retry_after=0,
            original=error,
        )

    # Level 3: Error message pattern matching
    msg = str(error).lower()
    if "context length" in msg or "too many tokens" in msg:
        return ClassifiedError(
            reason="context_overflow",
            retryable=False,
            should_failover=True,
            retry_after=0,
            original=error,
        )
    if "billing" in msg or "quota" in msg or "insufficient" in msg:
        return ClassifiedError(
            reason="billing",
            retryable=False,
            should_failover=False,
            retry_after=0,
            original=error,
        )

    # Level 4: Standard exception types
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return ClassifiedError(
            reason="network",
            retryable=True,
            should_failover=False,
            retry_after=0,
            original=error,
        )

    # Level 5: Fallback
    return ClassifiedError(
        reason="unknown",
        retryable=True,
        should_failover=False,
        retry_after=0,
        original=error,
    )


def _extract_status(error: Exception) -> int | None:
    """Extract HTTP status code from exception attributes."""
    # litellm exceptions have status_code attribute
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    # httpx / requests response attribute
    response = getattr(error, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _extract_retry_after(error: Exception) -> float:
    """
    Extract retry-after delay from exception.

    Priority:
    1. HTTP header retry-after / retry-after-ms
    2. Error message text matching ("retry after 30s", "try again in 2 minutes")
    3. Default 0 (caller uses exponential backoff)
    """
    # Try HTTP response headers
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        if headers:
            # retry-after-ms (milliseconds)
            ms = headers.get("retry-after-ms")
            if ms is not None:
                try:
                    return float(ms) / 1000.0
                except (ValueError, TypeError):
                    pass
            # retry-after (seconds)
            sec = headers.get("retry-after")
            if sec is not None:
                try:
                    return float(sec)
                except (ValueError, TypeError):
                    pass

    # Try error message text
    msg = str(error)
    match = re.search(r"retry\s+after\s+(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"try\s+again\s+in\s+(\d+(?:\.\d+)?)\s*(?:second|minute)", msg, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        if "minute" in match.group(0).lower():
            value *= 60
        return value

    return 0
