"""Transitional adapter: parse the current string-based chat_stream into
structured events (v3.0.3).

The framework's `agent.chat_stream()` yields strings that mix LLM content
with control markers (`[Calling foo...]`, `[foo done]`, `[Error: ...]`, ...).
This module converts that stream into the typed events defined in
`stream_protocol`. When the framework natively emits structured events
(v3.1+), callers can swap the data source and keep the same schema.

Scanning algorithm:
- Accumulate chunks into a buffer.
- Find the earliest `[` in the buffer. Text before it is plain content.
- Find the matching `]`. If absent and the buffer is short, wait for the
  next chunk (the marker may be split across chunks). If absent and the
  buffer has grown beyond `_FLUSH_THRESHOLD`, flush the leading `[` as
  content to avoid infinite buffering on pathological input.
- Match the `[...]` against known patterns. On match, emit the structured
  event; on miss, emit the raw bracketed text as content (preserves output).

Unterminated tool_call_start events (a `[Calling X...]` without a matching
`[X done]`) are closed with a synthetic `tool_call_end` carrying
`success=False, error="unterminated"` at stream end.

Known limitation: the scanner cuts each marker at the first `]`, so markers
whose payload contains `]` get truncated. In practice this only affects
`[Error: ...]` when an exception's string representation contains `]`;
the truncated tail is preserved as plain content (nothing is dropped).
Native event emission (v3.1+) will make this moot.
"""

import re
import time
from typing import Generator, Iterable


_FLUSH_THRESHOLD = 200  # max bytes to buffer waiting for a closing `]`


_CALL_RE = re.compile(r"^\[Calling (?P<name>[\w\-]+)\.\.\.\]$")
_DONE_RE = re.compile(r"^\[(?P<name>[\w\-]+) done\]$")
_ERROR_RE = re.compile(r"^\[Error: (?P<msg>.+)\]$")
_DUPLICATE_RE = re.compile(r"^\[Duplicate action detected \((?P<name>[\w\-]+)\), aborting\]$")

_STATIC_STATUS = {
    "[Operation aborted]":              ("info",    "aborted",          "Operation aborted"),
    "[Context window overflow]":        ("warning", "context_overflow", "Context window overflow"),
    "[Step timeout]":                   ("warning", "step_timeout",     "Step timeout"),
    "[Total timeout]":                  ("warning", "total_timeout",    "Total timeout"),
    "[Maximum steps reached]":          ("warning", "max_steps",        "Maximum steps reached"),
    "[Input blocked by safety module]": ("warning", "blocked",          "Input blocked by safety module"),
}


def adapt_stream(text_stream: Iterable[str]) -> Generator[dict, None, None]:
    """Transform a text-chunk stream into a structured event stream.

    Events conform to TypedDicts in `stream_protocol`. Always emits a final
    `done` event. Unterminated tool calls are closed with synthetic end events.
    """
    seq = 0
    buffer = ""
    open_calls: dict[str, tuple[str, float]] = {}  # tool name -> (call_id, started_at)
    call_counter = 0

    def next_seq() -> int:
        nonlocal seq
        seq += 1
        return seq

    def make_content(text: str):
        if not text:
            return None
        return {"type": "content", "seq": next_seq(), "text": text}

    def match_marker(marker: str):
        """Return a structured event for a `[...]` marker, or None if unknown."""
        nonlocal call_counter

        # Tool call start
        m = _CALL_RE.match(marker)
        if m:
            name = m.group("name")
            call_counter += 1
            call_id = f"tc_{call_counter}"
            started_at = time.time()
            open_calls[name] = (call_id, started_at)
            return {
                "type": "tool_call_start",
                "seq": next_seq(),
                "call_id": call_id,
                "name": name,
                "args": None,
                "started_at": started_at,
            }

        # Tool call end
        m = _DONE_RE.match(marker)
        if m:
            name = m.group("name")
            pair = open_calls.pop(name, None)
            if pair:
                call_id, started_at = pair
                duration_ms = int((time.time() - started_at) * 1000)
            else:
                # done without matching start — synthesize a call_id
                call_counter += 1
                call_id = f"tc_{call_counter}"
                duration_ms = None
            return {
                "type": "tool_call_end",
                "seq": next_seq(),
                "call_id": call_id,
                "success": True,
                "result_preview": None,
                "duration_ms": duration_ms,
                "error": None,
            }

        # Duplicate action (terminates the open call for that tool)
        m = _DUPLICATE_RE.match(marker)
        if m:
            name = m.group("name")
            pair = open_calls.pop(name, None)
            events = []
            if pair:
                call_id, started_at = pair
                events.append({
                    "type": "tool_call_end",
                    "seq": next_seq(),
                    "call_id": call_id,
                    "success": False,
                    "result_preview": None,
                    "duration_ms": int((time.time() - started_at) * 1000),
                    "error": "duplicate_action",
                })
            events.append({
                "type": "status",
                "seq": next_seq(),
                "level": "warning",
                "code": "duplicate_action",
                "message": f"Duplicate action detected ({name}), aborting",
            })
            return events

        # Error
        m = _ERROR_RE.match(marker)
        if m:
            return {
                "type": "error",
                "seq": next_seq(),
                "message": m.group("msg"),
            }

        # Static status markers
        info = _STATIC_STATUS.get(marker)
        if info:
            level, code, message = info
            return {
                "type": "status",
                "seq": next_seq(),
                "level": level,
                "code": code,
                "message": message,
            }

        return None

    # --- main scan loop ---
    for chunk in text_stream:
        if not chunk:
            continue
        buffer += chunk

        while buffer:
            idx = buffer.find("[")

            # No marker in buffer — flush all as content, wait for next chunk.
            if idx == -1:
                evt = make_content(buffer)
                if evt:
                    yield evt
                buffer = ""
                break

            # Flush any content before the marker.
            if idx > 0:
                evt = make_content(buffer[:idx])
                if evt:
                    yield evt
                buffer = buffer[idx:]
                idx = 0

            end = buffer.find("]")
            if end == -1:
                # Incomplete marker. If the buffer is growing too large, the
                # bracket is probably not a marker at all — flush as content.
                if len(buffer) > _FLUSH_THRESHOLD:
                    evt = make_content(buffer)
                    if evt:
                        yield evt
                    buffer = ""
                # Otherwise wait for more chunks.
                break

            marker = buffer[: end + 1]
            buffer = buffer[end + 1 :]

            result = match_marker(marker)
            if result is None:
                # Unknown marker — preserve as content.
                evt = make_content(marker)
                if evt:
                    yield evt
            elif isinstance(result, list):
                for e in result:
                    yield e
            else:
                yield result

    # End of stream: flush any remaining buffer as content.
    if buffer:
        evt = make_content(buffer)
        if evt:
            yield evt

    # Close any unterminated tool calls.
    for name, (call_id, started_at) in list(open_calls.items()):
        yield {
            "type": "tool_call_end",
            "seq": next_seq(),
            "call_id": call_id,
            "success": False,
            "result_preview": None,
            "duration_ms": int((time.time() - started_at) * 1000),
            "error": "unterminated",
        }
    open_calls.clear()

    yield {"type": "done", "seq": next_seq()}
