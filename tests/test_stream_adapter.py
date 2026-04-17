"""Unit tests for the interface-layer stream adapter (v3.0.3).

Covers the string-to-structured-event translation:
- Plain content
- Tool call pair (start + done)
- Multiple sequential tool calls
- Error markers
- Static status markers (abort, max_steps, timeouts, context_overflow, blocked)
- Duplicate-action marker closes open call and emits status
- Unterminated tool call produces synthetic end at stream close
- Unknown `[...]` tokens preserved as content
- Markers split across chunks
"""

from llamagent.interfaces.stream_adapter import adapt_stream
from llamagent.interfaces.stream_protocol import PROTOCOL_VERSION
from llamagent.interfaces.web_ui import _apply_event, _render_segments


def _collect(chunks):
    return list(adapt_stream(iter(chunks)))


def _types(events):
    return [e["type"] for e in events]


def test_protocol_version_exposed():
    assert PROTOCOL_VERSION == 1


def test_plain_content_only():
    events = _collect(["hello ", "world"])
    # One or more content events, then done
    assert _types(events) == ["content"] * (len(events) - 1) + ["done"]
    assert "".join(e["text"] for e in events if e["type"] == "content") == "hello world"


def test_empty_stream():
    events = _collect([])
    assert _types(events) == ["done"]
    assert events[0]["seq"] == 1


def test_tool_call_pair():
    chunks = ["hi\n[Calling web_search...]\nsearching\n[web_search done]\nresult is X"]
    events = _collect(chunks)
    types = _types(events)
    assert "tool_call_start" in types
    assert "tool_call_end" in types
    assert types[-1] == "done"

    start = next(e for e in events if e["type"] == "tool_call_start")
    end = next(e for e in events if e["type"] == "tool_call_end")
    assert start["name"] == "web_search"
    assert start["call_id"] == end["call_id"]
    assert end["success"] is True
    assert end["error"] is None
    assert end["duration_ms"] is not None and end["duration_ms"] >= 0


def test_multiple_sequential_tools():
    chunks = [
        "[Calling foo...]\n[foo done]\n",
        "[Calling bar...]\n[bar done]\n",
    ]
    events = _collect(chunks)
    starts = [e for e in events if e["type"] == "tool_call_start"]
    ends = [e for e in events if e["type"] == "tool_call_end"]
    assert [s["name"] for s in starts] == ["foo", "bar"]
    assert [e["success"] for e in ends] == [True, True]
    assert starts[0]["call_id"] != starts[1]["call_id"]


def test_error_marker():
    events = _collect(["\n[Error: model unreachable]\n"])
    err = next(e for e in events if e["type"] == "error")
    assert err["message"] == "model unreachable"


def test_static_status_markers():
    cases = {
        "[Operation aborted]":              ("aborted",          "info"),
        "[Context window overflow]":        ("context_overflow", "warning"),
        "[Step timeout]":                   ("step_timeout",     "warning"),
        "[Total timeout]":                  ("total_timeout",    "warning"),
        "[Maximum steps reached]":          ("max_steps",        "warning"),
        "[Input blocked by safety module]": ("blocked",          "warning"),
    }
    for marker, (code, level) in cases.items():
        events = _collect([marker])
        status = next(e for e in events if e["type"] == "status")
        assert status["code"] == code
        assert status["level"] == level


def test_duplicate_action_closes_open_call_and_emits_status():
    chunks = ["[Calling foo...]\n[Duplicate action detected (foo), aborting]\n"]
    events = _collect(chunks)
    types = _types(events)
    assert types.count("tool_call_start") == 1
    assert types.count("tool_call_end") == 1
    assert types.count("status") == 1

    end = next(e for e in events if e["type"] == "tool_call_end")
    status = next(e for e in events if e["type"] == "status")
    assert end["success"] is False
    assert end["error"] == "duplicate_action"
    assert status["code"] == "duplicate_action"


def test_unterminated_tool_call_synthesized_end():
    chunks = ["[Calling foo...]\nworking..."]
    events = _collect(chunks)
    end = next(e for e in events if e["type"] == "tool_call_end")
    assert end["success"] is False
    assert end["error"] == "unterminated"
    # stream should still terminate with done
    assert events[-1]["type"] == "done"


def test_unknown_marker_preserved_as_content():
    # A bracketed token that doesn't match any known pattern should appear
    # verbatim in the concatenated content — user text isn't lost.
    events = _collect(["before [TODO: review] after"])
    contents = "".join(e["text"] for e in events if e["type"] == "content")
    assert contents == "before [TODO: review] after"
    # No tool/status/error events emitted for unknown markers
    assert "tool_call_start" not in _types(events)
    assert "status" not in _types(events)


def test_marker_split_across_chunks():
    # `[Calling foo...]` is split into three chunks
    chunks = ["[Call", "ing foo...", "]\n[foo done]"]
    events = _collect(chunks)
    types = _types(events)
    assert "tool_call_start" in types
    assert "tool_call_end" in types
    start = next(e for e in events if e["type"] == "tool_call_start")
    assert start["name"] == "foo"


def test_seq_is_monotonic():
    events = _collect(["hi [Calling foo...]\n[foo done]\nbye"])
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)  # all unique


def test_done_is_always_last():
    events = _collect(["[Error: boom]"])
    assert events[-1]["type"] == "done"


# ============================================================
# Web UI render helpers (v3.0.3)
# ============================================================


def _build(events):
    segments: list[dict] = []
    index: dict[str, int] = {}
    for e in events:
        _apply_event(segments, index, e)
    return segments


def test_render_plain_content():
    events = _collect(["hello world"])
    rendered = _render_segments(_build(events))
    assert rendered == "hello world"


def test_render_completed_tool_call():
    events = _collect(["pre\n[Calling web_search...]\n[web_search done]\npost"])
    rendered = _render_segments(_build(events))
    assert "<details><summary>✅" in rendered
    assert "<code>web_search</code>" in rendered
    assert "pre" in rendered and "post" in rendered
    # Closed block should not be open by default
    assert "<details open>" not in rendered


def test_render_pending_tool_call_is_open():
    # No matching done -> still open during streaming
    events = list(adapt_stream(iter(["[Calling foo...]\npartial"])))
    # Drop the synthetic unterminated tool_call_end + done to simulate mid-stream
    mid_stream = [e for e in events if e["type"] in ("content", "tool_call_start")]
    rendered = _render_segments(_build(mid_stream))
    assert "<details open><summary>⏳ Calling <code>foo</code>" in rendered


def test_render_escapes_tool_name_and_error():
    events = [
        {"type": "tool_call_start", "seq": 1, "call_id": "tc_1",
         "name": "<script>evil</script>", "args": None, "started_at": 0.0},
        {"type": "error", "seq": 2, "message": "<img src=x onerror=y>"},
    ]
    rendered = _render_segments(_build(events))
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img" in rendered


def test_render_failed_tool_shows_error():
    events = [
        {"type": "tool_call_start", "seq": 1, "call_id": "tc_1",
         "name": "foo", "args": None, "started_at": 0.0},
        {"type": "tool_call_end", "seq": 2, "call_id": "tc_1",
         "success": False, "result_preview": None,
         "duration_ms": 12, "error": "boom"},
    ]
    rendered = _render_segments(_build(events))
    assert "❌" in rendered
    assert "boom" in rendered


def test_render_empty_segments_placeholder():
    assert _render_segments([]) == "(empty response)"
