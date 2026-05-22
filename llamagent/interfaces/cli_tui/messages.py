"""Message bus dataclasses for cli_tui (plan v11 §2.5).

These Messages are posted via ``app.post_message(...)`` from worker
threads or hook callbacks, then dispatched to widgets by Textual's
event loop (post_message is thread-safe per Textual workers guide —
https://textual.textualize.io/guide/workers/#thread-workers).

Using typed Messages (not text markers parsed by stream_adapter)
sidesteps the round-1 Web UI userland inject B1 ship-blocker: any
bracketed text inside tool args / result could mis-parse as a marker.
TUI has its own typed channel.
"""
from typing import Optional

from textual.message import Message


class ChatChunkMessage(Message):
    """Streaming assistant text chunk from agent.chat_stream()."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class ToolStartMessage(Message):
    """PRE_TOOL_USE hook fired — tool execution started.

    call_id is TUI-generated (framework hook doesn't provide one;
    see plan §2.5 — thread_id + monotonic counter).
    """

    def __init__(self, name: str, args: dict, call_id: str) -> None:
        super().__init__()
        self.name = name
        self.args = args
        self.call_id = call_id


class ToolEndMessage(Message):
    """POST_TOOL_USE hook fired — tool execution completed successfully."""

    def __init__(
        self,
        call_id: str,
        duration_ms: float,
        result_preview: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.call_id = call_id
        self.duration_ms = duration_ms
        self.result_preview = result_preview


class ToolErrorMessage(Message):
    """TOOL_ERROR hook fired — tool raised an exception.

    Separate from ToolEndMessage because framework dispatches via two
    distinct hook events (plan §1.4 framework hard facts).
    """

    def __init__(self, call_id: str, name: str, error: str) -> None:
        super().__init__()
        self.call_id = call_id
        self.name = name
        self.error = error


class ThinkingMessage(Message):
    """LLM thinking content captured from response.

    source: 'reasoning_content' | '<think>' | 'thinking_blocks'
    dedup_hash: SHA-256 first 16 chars for per-turn dedup
                (resilience retry / fallback path may emit twice)
    """

    def __init__(self, source: str, content: str, dedup_hash: str) -> None:
        super().__init__()
        self.source = source
        self.content = content
        self.dedup_hash = dedup_hash


class StatusMessage(Message):
    """Generic status text shown in ChatLog as a muted line."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class ErrorMessage(Message):
    """Error to display in ChatLog as a highlighted line."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


class ChildSpawnMessage(Message):
    """Pushed from monkey-patched ``task_board.create`` (plan §2.4.3)."""

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id


class ChildCompleteMessage(Message):
    """Pushed from monkey-patched ``task_board.update`` when status enters
    a terminal state {completed, failed, cancelled} (plan §2.4.3)."""

    def __init__(self, task_id: str, status: str) -> None:
        super().__init__()
        self.task_id = task_id
        self.status = status
