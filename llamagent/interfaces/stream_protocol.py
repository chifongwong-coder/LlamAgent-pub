"""Streaming event protocol for interface layer (v3.0.3).

Version 1. All events carry a `type` discriminator and a `seq` monotonic
counter within a single stream. When the framework natively emits structured
events (planned v3.1+), the same schema is reused — the adapter in
`stream_adapter.py` becomes a passthrough.

Event types:
- content           : LLM text chunk
- tool_call_start   : Tool invocation started
- tool_call_end     : Tool invocation finished (success or failure)
- status            : Non-fatal signal (max steps, duplicate action, ...)
- error             : Fatal error during generation
- done              : End of stream marker (always last)

Fields that are `None` in the current adapter (`args`, `result_preview`) will
be populated when native event emission lands. UI code should treat `None`
as "not available" and degrade gracefully.
"""

from typing import TypedDict, Literal, Optional


PROTOCOL_VERSION = 1


class ContentEvent(TypedDict):
    type: Literal["content"]
    seq: int
    text: str


class ToolCallStartEvent(TypedDict):
    type: Literal["tool_call_start"]
    seq: int
    call_id: str
    name: str
    args: Optional[dict]
    started_at: float


class ToolCallEndEvent(TypedDict):
    type: Literal["tool_call_end"]
    seq: int
    call_id: str
    success: bool
    result_preview: Optional[str]
    duration_ms: Optional[int]
    error: Optional[str]


class StatusEvent(TypedDict):
    type: Literal["status"]
    seq: int
    level: Literal["info", "warning"]
    code: str
    message: str


class ErrorEvent(TypedDict):
    type: Literal["error"]
    seq: int
    message: str


class DoneEvent(TypedDict):
    type: Literal["done"]
    seq: int
