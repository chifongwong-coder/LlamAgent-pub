"""LoggingLLM: thin wrapper around an LLMClient that appends each chat()
reply to a JSONL runlog file.

Used by the child_agent module to record per-step model behavior of a
child agent for external observation. Pattern mirrors BudgetedLLM
(child_agent/budget.py) — proxy semantics, side effect on each call.

The runlog is **not exposed to the parent agent through the tool surface**.
It is intentionally a write-only observability sink for humans / external
monitors: when a child thread or subprocess dies, the file on disk still
records the last things the model did, even though no return path is left
to convey that to the parent agent.

JSONL line format:

    {"ts": <unix_ts>, "kind": "reply", "content_preview": "...", "tool_calls": [...]}

Each line is bounded to ~4 KiB to avoid PIPE_BUF-related write tearing
when the writer is a subprocess and a separate process reads concurrently.
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_MAX_LINE_BYTES = 4096  # PIPE_BUF on macOS / Linux is typically 4 KiB
_PREVIEW_CHARS = 500    # cap content_preview / args_preview / result_preview at this many chars


def append_runlog(runlog_path: str, record: dict, max_bytes: int = 10 * 1024 * 1024) -> None:
    """Append one JSONL record to the runlog, with rotation on size cap.

    The record is encoded, capped at _MAX_LINE_BYTES (truncating the largest
    string-valued field if needed), and appended. If the resulting file
    would exceed max_bytes, the existing file is renamed to ``.log.1``
    first and a new file is started.

    All errors are caught + logged; runlog write failure must never crash
    the child agent.
    """
    try:
        os.makedirs(os.path.dirname(runlog_path), exist_ok=True)
        encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        if len(encoded) > _MAX_LINE_BYTES:
            # Truncate the largest string field to fit. Keys we know are large.
            for key in ("content_preview", "result_preview", "args_preview"):
                if key in record and isinstance(record[key], str):
                    overshoot = len(encoded) - _MAX_LINE_BYTES + 32  # safety margin
                    record[key] = record[key][: max(50, len(record[key]) - overshoot)] + "...[truncated]"
                    encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                    if len(encoded) <= _MAX_LINE_BYTES:
                        break
            # Hard cap if still too long
            if len(encoded) > _MAX_LINE_BYTES:
                encoded = encoded[: _MAX_LINE_BYTES - 1] + b"\n"
        # Rotation: rename when file would exceed max_bytes
        try:
            existing = os.path.getsize(runlog_path) if os.path.exists(runlog_path) else 0
        except OSError:
            existing = 0
        if existing + len(encoded) > max_bytes:
            try:
                os.rename(runlog_path, runlog_path + ".1")
            except OSError as e:
                logger.warning("runlog rotation failed for %s: %s", runlog_path, e)
        with open(runlog_path, "ab") as f:
            f.write(encoded)
    except Exception as e:
        logger.warning("runlog append failed at %s: %s", runlog_path, e)


def _content_preview(text: str | None) -> str:
    if not text:
        return ""
    return text[:_PREVIEW_CHARS]


class LoggingLLM:
    """Wrap an LLMClient so each chat() reply is appended to a runlog file.

    Attribute proxy via __getattr__ keeps every other call (count_tokens,
    embeddings, etc.) routed to the wrapped client unchanged.

    Args:
        wrapped: The LLMClient to delegate to.
        runlog_path: Absolute path of the JSONL runlog file.
        max_bytes: Soft cap; rotation triggers a rename to ``.log.1`` once
            exceeded. Default 10 MiB.
    """

    def __init__(self, wrapped, runlog_path: str, max_bytes: int = 10 * 1024 * 1024):
        self._wrapped = wrapped
        self._runlog_path = runlog_path
        self._max_bytes = max_bytes

    def chat(self, messages, tools=None, **kwargs):
        result = self._wrapped.chat(messages, tools=tools, **kwargs)
        try:
            msg = result.choices[0].message
            content = getattr(msg, "content", None) or ""
            tool_calls = getattr(msg, "tool_calls", None) or []
            tc_summaries = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn:
                    name = getattr(fn, "name", "?")
                    args_str = getattr(fn, "arguments", "") or ""
                    tc_summaries.append({
                        "name": name,
                        "args_preview": args_str[:_PREVIEW_CHARS],
                    })
            append_runlog(
                self._runlog_path,
                {
                    "ts": time.time(),
                    "kind": "reply",
                    "content_preview": _content_preview(content),
                    "tool_calls": tc_summaries,
                },
                max_bytes=self._max_bytes,
            )
        except Exception as e:
            logger.warning("LoggingLLM chat-side runlog append failed: %s", e)
        return result

    def __getattr__(self, name):
        # Proxy everything else (count_tokens, model attribute, etc.).
        return getattr(self._wrapped, name)
