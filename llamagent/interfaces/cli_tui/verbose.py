"""LLM thinking capture for the VerbosePane (plan v15 §4 C5).

install_verbose / uninstall_verbose monkey-patch the agent's LLM chain
so each completion response is scanned for thinking content from three
sources (plan §11 Q5) and emitted as ``ThinkingMessage`` to the
VerbosePane widget:

1. ``reasoning_content`` — LiteLLM-OpenAI normalized field
2. ``<think>…</think>`` — qwen3 / qwen3.5 / qwen3.6 inline tags
3. ``thinking_blocks`` — LiteLLM-Anthropic structured blocks

Per-turn SHA-256 dedup (16-char prefix) prevents duplicates from
resilience retry / fallback paths (plan §2.4 — see _patch_all_llm_chains).
The chain walker recurses through ``_wrapped`` (LoggingLLM) and
``_fallback_llm`` (ResilientLLMClient) so a model fallback still shows
its thinking instead of dropping it on the floor (round-8 H2).

Streaming detection: ``<think>…</think>`` is also extracted from
chat_stream chunks by tracking the open/close tags across chunk
boundaries. Non-streaming response scanning covers the remaining
two sources.

C5 first-iteration limitations (registered in plan §12):
- ``uninstall_verbose`` only clears the idempotency flag; it does not
  restore the original ``chat`` / ``chat_stream`` methods. agent rebuild
  on a new agent works fine (new agent has untouched methods); rebuild
  on the *same* agent is idempotent via the flag. Restoring originals
  for full uninstall is a polish task.
- Streaming thinking via ``reasoning_content`` per-chunk isn't extracted
  because LiteLLM doesn't surface it during streaming; only the
  ``<think>`` tag path catches mid-stream thinking. ``thinking_blocks``
  also only available on the final response object, not chunks.
"""
import hashlib
import logging
from typing import TYPE_CHECKING

from llamagent.interfaces.cli_tui.messages import ThinkingMessage

if TYPE_CHECKING:
    from textual.widget import Widget

logger = logging.getLogger(__name__)


def install_verbose(agent, target: "Widget") -> None:
    """Patch ``agent.llm`` chain to emit ThinkingMessage to ``target``.

    Idempotent via ``agent._verbose_patched`` — same agent install runs
    just once. The walker tolerates cycles + depth > 4 defensively.
    """
    if getattr(agent, "_verbose_patched", False):
        return

    # Per-agent dedup set so retry paths don't double-emit. Module-level
    # would leak across agent rebuilds. Per-instance keeps it scoped.
    seen_hashes: set[str] = set()

    def _emit(source: str, content: str) -> None:
        if not content:
            return
        content = content.strip()
        if not content:
            return
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        if digest in seen_hashes:
            return
        seen_hashes.add(digest)
        try:
            target.post_message(
                ThinkingMessage(source=source, content=content, dedup_hash=digest)
            )
        except Exception as e:
            logger.debug("post ThinkingMessage failed: %s", e)

    def _scan_response_message(msg) -> None:
        """Extract three thinking sources from a single response message."""
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            _emit("reasoning_content", str(reasoning))

        content = getattr(msg, "content", "") or ""
        if "<think>" in content and "</think>" in content:
            inner = content.split("<think>", 1)[1].split("</think>", 1)[0]
            _emit("<think>", inner)

        blocks = getattr(msg, "thinking_blocks", None) or []
        for b in blocks:
            if isinstance(b, dict):
                text = b.get("thinking") or ""
            else:
                text = getattr(b, "thinking", "") or ""
            if text:
                _emit("thinking_blocks", str(text))

    def _wrap_chat(client) -> None:
        if not hasattr(client, "chat"):
            return
        if getattr(client, "_verbose_chat_wrapped", False):
            return
        original_chat = client.chat

        def _traced_chat(messages, **kwargs):
            resp = original_chat(messages, **kwargs)
            try:
                msg = resp.choices[0].message
                _scan_response_message(msg)
            except Exception as e:
                logger.debug("_scan_response_message failed: %s", e)
            return resp

        client.chat = _traced_chat
        client._verbose_chat_wrapped = True

    def _wrap_chat_stream(client) -> None:
        if not hasattr(client, "chat_stream"):
            return
        if getattr(client, "_verbose_chat_stream_wrapped", False):
            return
        original_chat_stream = client.chat_stream

        def _traced_chat_stream(messages, **kwargs):
            in_think = False
            buf = ""
            for chunk in original_chat_stream(messages, **kwargs):
                yield chunk
                # Try to extract delta.content (LiteLLM ModelResponseStream shape)
                try:
                    delta_content = chunk.choices[0].delta.content or ""
                except Exception:
                    continue
                if not in_think:
                    if "<think>" in delta_content:
                        in_think = True
                        buf = delta_content.split("<think>", 1)[1]
                        if "</think>" in buf:
                            inner = buf.split("</think>", 1)[0]
                            _emit("<think>", inner)
                            in_think = False
                            buf = ""
                else:
                    buf += delta_content
                    if "</think>" in buf:
                        inner = buf.split("</think>", 1)[0]
                        _emit("<think>", inner)
                        in_think = False
                        buf = ""

        client.chat_stream = _traced_chat_stream
        client._verbose_chat_stream_wrapped = True

    def _walk(client, visited: set, depth: int) -> None:
        if id(client) in visited or depth > 4:
            return
        visited.add(id(client))
        _wrap_chat(client)
        _wrap_chat_stream(client)
        # Recurse through LoggingLLM wrappers + ResilientLLMClient fallbacks.
        for attr in ("_wrapped", "_fallback_llm"):
            inner = getattr(client, attr, None)
            if inner is not None:
                _walk(inner, visited, depth + 1)

    _walk(agent.llm, set(), 0)
    agent._verbose_patched = True
    agent._verbose_seen_hashes = seen_hashes


def uninstall_verbose(agent) -> None:
    """Clear the verbose-installed flag so a future install on the same
    agent re-runs the walker.

    C5 limitation: the chat / chat_stream methods themselves stay
    wrapped. For C5 this is acceptable because set_agent always rebinds
    to a NEW agent instance with untouched LLM methods. Restoring
    originals for full cleanup is a follow-up polish item.
    """
    if not getattr(agent, "_verbose_patched", False):
        return
    try:
        delattr(agent, "_verbose_patched")
    except AttributeError:
        pass
    try:
        delattr(agent, "_verbose_seen_hashes")
    except AttributeError:
        pass
