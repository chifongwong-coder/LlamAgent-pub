"""
LLM Client: a thin wrapper around litellm providing a unified calling interface.

Functions:
- chat()          Low-level LLM call with function calling support
- ask()           Single-turn Q&A returning plain text
- ask_json()      Single-turn Q&A returning parsed JSON
- count_tokens()  Estimate token count for messages

Graceful degradation when litellm is not installed: all calls return error messages without crashing.
"""

import json
import logging
import re
import time

logger = logging.getLogger(__name__)

# litellm is an optional dependency; graceful degradation when not installed
try:
    import litellm
    from litellm import completion

    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False
    logger.warning(
        "litellm is not installed, LLM calling functionality is unavailable. Please run: pip install litellm"
    )


class LLMClient:
    """
    Unified LLM calling interface.

    Built on LiteLLM, supporting multiple model backends (OpenAI, Anthropic, DeepSeek, Ollama, etc.).
    When litellm is not installed, all calling methods return friendly error messages without raising exceptions.
    """

    def __init__(self, model: str, api_retry_count: int = 1):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier, e.g. "openai/gpt-4o-mini"
            api_retry_count: Number of retries on API call failure (retries N times, N+1 total calls)
        """
        self.model = model
        self.api_retry_count = api_retry_count

        # Context window size (attempt auto-detection)
        self.max_context_tokens: int = self._detect_max_context_tokens()

    def _detect_max_context_tokens(self) -> int:
        """Auto-detect the model's context window size; uses a conservative default on failure."""
        if not _LITELLM_AVAILABLE:
            return 8192
        try:
            max_tokens = litellm.get_max_tokens(self.model)
            if max_tokens and max_tokens > 0:
                return max_tokens
        except Exception:
            logger.debug("Unable to get context window size for model %s, using default 8192", self.model)
        return 8192

    # ------------------------------------------------------------------
    # Core calling methods
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        timeout: float | None = None,
    ):
        """
        Low-level LLM call returning the full response object, with function calling support.

        Built-in retry logic: retries on failure according to api_retry_count with exponential backoff.

        Args:
            messages: Message list [{"role": "system/user/assistant", "content": "..."}]
            temperature: Creativity parameter
            max_tokens: Maximum output tokens
            response_format: Force output format, e.g. {"type": "json_object"}
            tools: Tool schema list in OpenAI format
            tool_choice: Tool selection strategy ("auto", "none", "required")
            timeout: Request timeout in seconds; None means no timeout

        Returns:
            The raw litellm response object

        Raises:
            RuntimeError: Raised when litellm is not installed
        """
        if not _LITELLM_AVAILABLE:
            raise RuntimeError(
                "litellm is not installed, cannot call LLM. Please run: pip install litellm"
            )

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if timeout is not None:
            kwargs["timeout"] = timeout

        # Call with retry
        last_error = None
        for attempt in range(self.api_retry_count + 1):
            try:
                return completion(**kwargs)
            except Exception as e:
                last_error = e
                # ContextWindowExceededError should not be retried; raise immediately
                if _LITELLM_AVAILABLE and isinstance(
                    e, litellm.ContextWindowExceededError
                ):
                    raise
                if attempt < self.api_retry_count:
                    wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, ...
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        self.api_retry_count + 1,
                        wait,
                        e,
                    )
                    time.sleep(wait)
                else:
                    logger.error("LLM call failed, all retries exhausted: %s", e)
        raise last_error  # type: ignore[misc]

    def ask(self, prompt: str, system: str = "", **kwargs) -> str:
        """
        Single-turn Q&A convenience method returning plain text.

        Args:
            prompt: User question content
            system: System prompt (optional)
            **kwargs: Additional arguments passed to chat() (temperature, max_tokens, timeout, etc.)

        Returns:
            The model's reply text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.chat(messages, **kwargs)
            return resp.choices[0].message.content or ""
        except RuntimeError as e:
            # litellm not installed
            return f"[LLM unavailable] {e}"
        except Exception as e:
            logger.error("ask() call failed: %s", e)
            return f"[LLM call error] {e}"

    def ask_json(self, prompt: str, system: str = "", **kwargs) -> dict:
        """
        Single-turn Q&A returning parsed JSON.

        Automatically sets response_format={"type": "json_object"}.

        JSON parsing strategy:
        1. Direct json.loads() parsing
        2. On failure, attempt extraction from Markdown code block ```json ... ```
        3. If still failing, return {"raw_response": str, "error": "JSON parsing failed"}

        Args:
            prompt: User question content
            system: System prompt (optional)
            **kwargs: Additional arguments passed to chat() (temperature, max_tokens, timeout, etc.)

        Returns:
            Parsed dictionary
        """
        text = self.ask(
            prompt,
            system,
            response_format={"type": "json_object"},
            **kwargs,
        )

        # Strategy 1: Direct parsing
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Strategy 2: Extract from Markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        # Strategy 3: Fallback return
        logger.warning("JSON parsing failed, returning raw text. Response content: %s", text[:200])
        return {"raw_response": text, "error": "JSON parsing failed"}

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, messages: list[dict] | str) -> int:
        """
        Estimate token count for a message list or text.

        Prefers litellm.token_counter(); when unavailable, uses a rough character-based estimate
        (Chinese ~1 char = 1.5 tokens, English ~4 chars = 1 token, compromise: 1 char = 1 token).

        Args:
            messages: Message list [{"role": ..., "content": ...}] or plain text string

        Returns:
            Estimated token count
        """
        # If input is a string, wrap it as a message list
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if _LITELLM_AVAILABLE:
            try:
                return litellm.token_counter(model=self.model, messages=messages)
            except Exception:
                logger.debug("litellm.token_counter() failed, using character-based estimate")

        # Fallback: rough character-based estimate
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars  # Rough estimate: 1 char ~= 1 token
