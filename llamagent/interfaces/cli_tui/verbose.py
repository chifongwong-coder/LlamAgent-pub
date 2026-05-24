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
The dedup set is cleared at every PRE_CHAT (turn boundary) — round-12
H3 fix — so identical short reasoning ("Let me check…") from two turns
both render instead of the second one being silently swallowed.

Round-18 A-6: install_verbose now patches only the outermost LLM
layer (``agent.llm``). LoggingLLM / ResilientLLMClient wrappers
delegate down via ``_wrapped`` / ``_fallback_llm`` and the response
flows back up through the outer wrapper, so a fallback's final
response still reaches the scanner. Pre-fix the walker recursed into
each inner layer and re-wrapped its ``chat`` / ``chat_stream`` — every
call paid for two ``_scan_response_message`` passes (one at outer
wrapper, one at inner wrapper invoked via ``self._wrapped.chat(...)``);
per-turn dedup hashes masked the duplicate emit but the scan cost
doubled.

Streaming detection: ``<think>…</think>`` is extracted from chat_stream
chunks while tracking open/close tags across chunk boundaries. A small
rolling pre-buffer (round-12 H2) handles the rare case where LiteLLM
splits the literal ``<think>`` characters across two chunks. A
``while`` loop (round-12 H1) consumes every ``<think>…</think>`` pair
in a single chunk so multi-block reasoning in one chunk isn't truncated.
Non-streaming response scanning covers the remaining two sources.

C5 first-iteration limitations (registered in plan §12):
- ``uninstall_verbose`` only clears the idempotency flag; it does not
  restore the original ``chat`` / ``chat_stream`` methods. agent rebuild
  on a new agent works fine (new agent has untouched methods); rebuild
  on the *same* agent is idempotent via the flag. Restoring originals
  for full uninstall is a polish task (plan §12 K8.M2).
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


# Tail size of the cross-chunk pre-tag rolling buffer. 16 chars is
# enough to span "<think>" (7) plus a margin for whitespace around it.
# Per-chunk cost is one string slice — negligible (round-12 H2).
_PREBUF_TAIL = 16


def install_verbose(agent, target: "Widget") -> None:
    """Patch ``agent.llm`` chain to emit ThinkingMessage to ``target``.

    Idempotent via ``agent._verbose_patched`` — same agent install runs
    just once. The walker tolerates cycles + depth > 4 defensively.

    Round-12 H3: also registers a PRE_CHAT hook that clears the per-agent
    dedup set on every new turn. Without this, identical short reasoning
    output in two turns dedups the second one to silence.
    """
    if getattr(agent, "_verbose_patched", False):
        return

    # Per-agent dedup set so retry paths don't double-emit within a turn.
    # Module-level would leak across agent rebuilds. Per-instance keeps it
    # scoped. PRE_CHAT hook (below) clears at turn boundaries so cross-turn
    # repeats render instead of getting swallowed (round-12 H3).
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
        # Multi-block: qwen can emit several <think>…</think> pairs in
        # one response. Loop until exhausted instead of grabbing only
        # the first pair (round-12 H1 mirror for the non-streaming path).
        remaining = content
        while "<think>" in remaining and "</think>" in remaining:
            _, _, after_open = remaining.partition("<think>")
            inner, _, after_close = after_open.partition("</think>")
            _emit("<think>", inner)
            remaining = after_close

        blocks = getattr(msg, "thinking_blocks", None) or []
        for b in blocks:
            if isinstance(b, dict):
                text = b.get("thinking") or ""
            else:
                text = getattr(b, "thinking", "") or ""
            if text:
                _emit("thinking_blocks", str(text))

    def _has_real_method(client, name: str) -> bool:
        """True if ``client.<name>`` is a real attribute (instance dict
        or somewhere in the class MRO), not just a ``__getattr__`` proxy.

        Round-18 A-6 + round-18-3 docstring fix: install_verbose now
        wraps only the outermost layer (``agent.llm``). The guard is
        defensive against a future config where the outermost LLM
        exposes ``chat`` / ``chat_stream`` only through ``__getattr__``
        forwarding (e.g., a tracing decorator that doesn't redefine
        these names). Wrapping such an attribute via instance-dict
        assignment IS technically possible — the assignment shadows
        ``__getattr__`` lookup on subsequent reads — but the wrapped
        callable then proxies through to the inner client whose
        original method has not been touched, so internal flows that
        bypass the outer attribute (e.g., inner methods invoking each
        other directly) skip our scan. Defaulting to "skip if not a
        real owner" matches the production stack today: ResilientLLM /
        LoggingLLM define ``chat`` directly and inherit ``chat_stream``
        from LLMClient — both pass the MRO walk.
        """
        if name in vars(client):
            return True
        for klass in type(client).__mro__:
            if name in klass.__dict__:
                return True
        return False

    def _wrap_chat(client) -> None:
        if not _has_real_method(client, "chat"):
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
        # M1 guard (round-12): only wrap chat_stream where it's a real
        # attribute (instance __dict__ or class MRO), not a __getattr__
        # proxy. Without the guard LoggingLLM (proxies to inner
        # ResilientLLM.chat_stream) gets wrapped, then the walker
        # descends to ResilientLLM and wraps again — two layers of
        # <think> scan per chunk. Dedup hides the duplicate emit but
        # wastes one substring pass.
        if not _has_real_method(client, "chat_stream"):
            return
        if getattr(client, "_verbose_chat_stream_wrapped", False):
            return
        original_chat_stream = client.chat_stream

        def _traced_chat_stream(messages, **kwargs):
            in_think = False
            buf = ""          # body accumulator once <think> seen
            prebuf = ""       # rolling tail before <think> (H2)
            for chunk in original_chat_stream(messages, **kwargs):
                yield chunk
                # Try to extract delta.content (LiteLLM ModelResponseStream shape)
                try:
                    delta_content = chunk.choices[0].delta.content or ""
                except Exception:
                    continue
                # Round-12 H1 + H2: consume open/close pairs in a loop
                # so multiple <think>…</think> in one chunk all fire; use
                # a rolling pre-buffer (last _PREBUF_TAIL chars) so the
                # literal "<think>" string is detected even when LiteLLM
                # splits its characters across two chunks.
                while delta_content:
                    if not in_think:
                        prebuf += delta_content
                        idx = prebuf.find("<think>")
                        if idx < 0:
                            # Keep only the tail (in case the next chunk
                            # completes a tag like "<thi" + "nk>"). Reset
                            # delta_content so the while loop exits.
                            if len(prebuf) > _PREBUF_TAIL:
                                prebuf = prebuf[-_PREBUF_TAIL:]
                            delta_content = ""
                        else:
                            # Open tag found. Anything after it begins
                            # the body. prebuf resets; delta_content
                            # becomes the rest so a `</think>` in the
                            # same chunk closes immediately.
                            after = prebuf[idx + len("<think>"):]
                            in_think = True
                            buf = ""
                            prebuf = ""
                            delta_content = after
                    else:
                        buf += delta_content
                        idx = buf.find("</think>")
                        if idx < 0:
                            delta_content = ""
                        else:
                            inner = buf[:idx]
                            _emit("<think>", inner)
                            # Anything after </think> is post-think text
                            # — may itself contain another <think>.
                            after = buf[idx + len("</think>"):]
                            in_think = False
                            buf = ""
                            delta_content = after

        client.chat_stream = _traced_chat_stream
        client._verbose_chat_stream_wrapped = True

    # Round-18 A-6: wrap only the outermost LLM layer. The previous
    # recursion descended through `_wrapped` (LoggingLLM → ResilientLLM)
    # and wrapped every layer with a real `chat` / `chat_stream` method.
    # That meant every LLM call paid for two _scan_response_message
    # passes — LoggingLLM.chat → original LoggingLLM.chat → which
    # internally calls self._wrapped.chat (= wrapped ResilientLLM.chat
    # → second scan) → returns up to outer wrapper → first scan. The
    # per-turn dedup hashes hid the duplicate emit but the substring
    # scans + thinking_blocks iteration ran twice for every chunk.
    # The outermost wrapper sees the same final response — wrapping
    # inner layers adds CPU cost for zero new information.
    _wrap_chat(agent.llm)
    _wrap_chat_stream(agent.llm)

    # Round-12 H3: clear the dedup set at every turn boundary so a turn
    # that repeats the same short reasoning as a previous turn isn't
    # silently swallowed. PRE_CHAT fires once per chat() call (core/hooks.py).
    from llamagent.core.hooks import HookEvent

    def _on_pre_chat(_ctx) -> None:
        seen_hashes.clear()

    try:
        agent.register_hook(HookEvent.PRE_CHAT, _on_pre_chat)
        agent._verbose_pre_chat_handler = _on_pre_chat
    except Exception as e:
        # Hook system unavailable / register failed — non-fatal; dedup
        # still works within a turn, just stays sticky across turns.
        logger.debug("register PRE_CHAT for verbose dedup failed: %s", e)

    agent._verbose_patched = True
    agent._verbose_seen_hashes = seen_hashes


def uninstall_verbose(agent) -> None:
    """Clear the verbose-installed flag so a future install on the same
    agent re-runs the walker.

    C5 limitation: the chat / chat_stream methods themselves stay
    wrapped. For C5 this is acceptable because set_agent always rebinds
    to a NEW agent instance with untouched LLM methods. Restoring
    originals for full cleanup is a follow-up polish item (plan §12).

    Round-12: also unregisters the PRE_CHAT dedup-clear hook via
    identity-match through HookRegistration.handler.func (mirrors
    bridge.uninstall_hooks pattern) so we don't leak stale handlers
    if a future agent rebuild re-installs verbose on a new instance.
    """
    if not getattr(agent, "_verbose_patched", False):
        return

    # Identity-match unregister of the PRE_CHAT dedup-clear hook (mirrors
    # bridge.uninstall_hooks). Safe no-op if registration earlier failed.
    handler_fn = getattr(agent, "_verbose_pre_chat_handler", None)
    if handler_fn is not None:
        from llamagent.core.hooks import HookEvent

        hooks_dict = getattr(agent, "_hooks", None)
        if hooks_dict is not None:
            bucket = hooks_dict.get(HookEvent.PRE_CHAT, [])
            try:
                kept = []
                for reg in bucket:
                    inner = getattr(reg, "handler", None)
                    inner_fn = getattr(inner, "func", None) if inner is not None else None
                    if inner_fn is handler_fn:
                        continue
                    kept.append(reg)
                hooks_dict[HookEvent.PRE_CHAT] = kept
            except Exception as e:
                logger.debug("uninstall_verbose PRE_CHAT unregister: %s", e)

    for attr in ("_verbose_patched", "_verbose_seen_hashes", "_verbose_pre_chat_handler"):
        try:
            delattr(agent, attr)
        except AttributeError:
            pass
