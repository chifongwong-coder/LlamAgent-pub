"""Textual Pilot integration tests for the CLI TUI (plan §5.1 C9).

These tests run the real ``LlamAgentTUI`` Textual App inside Textual's
``app.run_test()`` async context, simulating keyboard events through
``Pilot.press`` and asserting against rendered widget state. They cover
the UI-path layer that the C1-C8 unit tests can't reach because the
unit tests don't have a Textual event loop available
(``LookupError: <ContextVar name='active_app' ...>`` on any reactive
write).

Scope (plan §5.1):
- C1  Input submit → ChatLog dispatch
- C2  worker thread → ChatChunkMessage round-trip
- C3  SetupScreen modal dismiss → agent build
- C4  ConfirmModal / AskUserModal sentinel-dismiss
- C5  VerbosePane thinking / tool routing
- C6  dispatch_slash + Footer BINDINGS + LlamAgentInput history
- C6.1 Tab accept suggestion + Up/Down history
- C7  ContinuousSetupModal validation + cmd_monitor / cmd_stop wiring
- C8  router argparse + ImportError fallback

Mock strategy: each test builds an in-memory MagicMock agent that
mimics the LlamAgent surface the TUI touches (mode, modules, _tools,
status, has_module, get_module, list_prompt_slots, chat, chat_stream,
abort, clear_conversation, set_mode, shutdown). No real LLM is called —
the chat path returns a fixed string when needed.

Why ``app.run_test()`` instead of full ``app.run()``: the test context
spins up a headless Textual event loop with size 80×24, captures
post_message routing, and exposes a ``Pilot`` for keyboard / click /
pause primitives. Tests stay deterministic without a real terminal.
"""
from __future__ import annotations

import pytest
from typing import Iterator
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_agent(
    *,
    mode: str = "interactive",
    tools: dict | None = None,
    modules: dict | None = None,
    chat_response: str = "mock reply",
) -> MagicMock:
    """Build a LlamAgent-shaped mock that the TUI can drive end-to-end.

    Covers exactly the surface the C1-C8 code touches; missing attributes
    on real LlamAgent are not stubbed (test will surface AttributeError
    if the production code grows a new dependency we haven't accounted
    for here).
    """
    agent = MagicMock()
    agent.mode = mode
    agent.modules = modules if modules is not None else {}
    agent._tools = tools if tools is not None else {}
    agent.config = MagicMock()
    agent.config.model = "mock/llama"
    agent.config.memory_mode = "off"
    agent.persona = MagicMock()
    agent.persona.name = "TestPersona"
    agent._tool_state = {}
    agent._hooks = {}
    agent.confirm_handler = None
    agent.interaction_handler = None
    agent.status.return_value = {
        "model": "mock/llama",
        "persona": "TestPersona",
        "modules": {},
        "conversation_turns": 0,
    }
    agent.has_module.return_value = False
    agent.get_module.return_value = None
    agent.list_prompt_slots.return_value = {"_agent": {}}
    agent.chat.return_value = chat_response

    # chat_stream returns a tiny generator so worker turn finalizes
    def _stream(_):
        yield chat_response

    agent.chat_stream.side_effect = _stream
    agent.abort.return_value = None
    agent.clear_conversation.return_value = None

    def _set_mode(m):
        agent.mode = m

    agent.set_mode.side_effect = _set_mode
    agent.shutdown.return_value = None
    # register_hook accepts (event, callable); test path doesn't fire hooks
    agent.register_hook.return_value = None
    return agent


@pytest.fixture
def mock_agent() -> MagicMock:
    """Default interactive-mode mock agent with no modules and no tools."""
    return _make_mock_agent()


@pytest.fixture
def mock_agent_with_tools() -> MagicMock:
    """Mock agent with a small registered-tools dict so /tools renders content."""
    return _make_mock_agent(
        tools={
            "read_files": {
                "name": "read_files",
                "description": "read files",
                "tier": "default",
                "safety_level": 1,
            },
            "shell_run": {
                "name": "shell_run",
                "description": "run shell",
                "tier": "admin",
                "safety_level": 3,
            },
        }
    )


# ---------------------------------------------------------------------------
# C1 / C6 — Input submit + slash dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_smoke_app_mounts(mock_agent):
    """Pilot framework sanity: App mounts, default screen has the expected
    widget ids. If this fails, every other Pilot test will fail too."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Default screen widgets all present
        assert app.query_one("#chat-log") is not None
        assert app.query_one("#verbose-pane") is not None
        assert app.query_one("#monitor-pane") is not None
        assert app.query_one("#input") is not None


@pytest.mark.asyncio
async def test_pilot_slash_help_renders(mock_agent):
    """C6 acceptance: typing /help into Input + Enter triggers dispatch_slash
    which mounts a Static with the help table inside ChatLog."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/help")
        await pilot.press("enter")
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        # The help table is a single Static — check its rendered content
        bodies = [
            str(getattr(c, "renderable", c)) for c in log.children
        ]
        joined = "\n".join(bodies)
        assert "Available Commands" in joined, f"help not rendered: {joined[:200]}"
        assert "/help" in joined
        assert "/verbose" in joined  # C5 + round-12 addition
        assert "/monitor" in joined  # C7 addition
        assert "/tools" in joined    # C6 new addition


@pytest.mark.asyncio
async def test_pilot_unknown_slash_renders_error(mock_agent):
    """C6: unrecognised slash command emits an ErrorMessage."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/nope")
        await pilot.press("enter")
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        bodies = [str(getattr(c, "renderable", c)) for c in log.children]
        joined = "\n".join(bodies)
        assert "Unknown command" in joined, f"missing error: {joined[:200]}"


# ---------------------------------------------------------------------------
# C5 / C7 — Right-pane mutex (verbose ↔ monitor)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_verbose_monitor_mutex(mock_agent):
    """C7: /verbose on shows VerbosePane + hides MonitorPane; /monitor on
    flips them (set_right_pane helper). Both stay mounted (display:none
    only) so each retains its history."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        verbose = app.query_one("#verbose-pane")
        monitor = app.query_one("#monitor-pane")
        # Both hidden at startup
        assert not verbose.display
        assert not monitor.display

        await pilot.press(*"/verbose on")
        await pilot.press("enter")
        await pilot.pause()
        assert verbose.display
        assert not monitor.display

        await pilot.press(*"/monitor on")
        await pilot.press("enter")
        await pilot.pause()
        assert monitor.display
        assert not verbose.display

        await pilot.press(*"/monitor off")
        await pilot.press("enter")
        await pilot.pause()
        assert not monitor.display
        assert not verbose.display


# ---------------------------------------------------------------------------
# C6.1 — Tab accept + Up/Down history through LlamAgentInput
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_history_up_down(mock_agent):
    """C6.1: after two submissions, Up walks back through history, Down
    walks forward and restores the scratch buffer."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import LlamAgentInput

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Submit two slash commands so history populates
        await pilot.press(*"/help")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press(*"/status")
        await pilot.press("enter")
        await pilot.pause()

        inp = app.query_one("#input", LlamAgentInput)
        # Round-18 A-12: _history is now a deque(maxlen=200); compare
        # as list for clean equality.
        assert list(inp._history) == ["/help", "/status"]
        # Now Input is empty, scratch is ""
        # First Up: scratch saved (""), cursor = 1 → "/status"
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "/status"
        # Second Up: cursor = 0 → "/help"
        await pilot.press("up")
        await pilot.pause()
        assert inp.value == "/help"
        # Down: back to "/status"
        await pilot.press("down")
        await pilot.pause()
        assert inp.value == "/status"
        # Down: past newest → restore scratch ""
        await pilot.press("down")
        await pilot.pause()
        assert inp.value == ""


@pytest.mark.asyncio
async def test_pilot_tab_accepts_suggestion(mock_agent):
    """C6.1: typing /v shows a Suggester ghost text for /verbose, Tab
    promotes it into the actual Input value."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import LlamAgentInput

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/v")
        # Suggester is async — wait a tick so it computes the suggestion
        await pilot.pause(0.1)
        inp = app.query_one("#input", LlamAgentInput)
        # The suggester should have populated _suggestion; Tab accepts it
        await pilot.press("tab")
        await pilot.pause()
        # value now starts with /v (definitely) and is at least /verbose
        # length — be tolerant of whether Suggester picks /verbose, /verbose on, etc.
        assert inp.value.startswith("/verbose"), f"Tab did not accept suggestion, value={inp.value!r}"


# ---------------------------------------------------------------------------
# C8 — Routing logic (unit-level via _route — no Pilot needed but kept here
# for §5.1 coverage symmetry)
# ---------------------------------------------------------------------------


def test_route_default_picks_tui():
    """C8 acceptance: default args (no flags) route to TUI when TTY check
    passes — patched here so the test doesn't depend on whether pytest is
    invoked from a real terminal."""
    from llamagent.interfaces import cli as cli_mod

    class _NS:
        legacy = False
        command = None
        modules = None
        no_modules = False

    orig = cli_mod._terminal_supports_tui
    cli_mod._terminal_supports_tui = lambda: True
    try:
        assert cli_mod._route(_NS()) == "tui"
    finally:
        cli_mod._terminal_supports_tui = orig


def test_route_legacy_flag_picks_legacy():
    from llamagent.interfaces import cli as cli_mod

    class _NS:
        legacy = True
        command = None
        modules = None
        no_modules = False

    assert cli_mod._route(_NS()) == "legacy"


def test_route_ask_subcommand_picks_legacy():
    from llamagent.interfaces import cli as cli_mod

    class _NS:
        legacy = False
        command = "ask"
        question = "x"
        modules = None
        no_modules = False

    assert cli_mod._route(_NS()) == "legacy"


def test_route_modules_flag_picks_legacy():
    from llamagent.interfaces import cli as cli_mod

    class _NS:
        legacy = False
        command = None
        modules = "tools"
        no_modules = False

    assert cli_mod._route(_NS()) == "legacy"


def test_route_no_tty_picks_legacy():
    from llamagent.interfaces import cli as cli_mod

    class _NS:
        legacy = False
        command = None
        modules = None
        no_modules = False

    orig = cli_mod._terminal_supports_tui
    cli_mod._terminal_supports_tui = lambda: False
    try:
        assert cli_mod._route(_NS()) == "legacy"
    finally:
        cli_mod._terminal_supports_tui = orig


# ---------------------------------------------------------------------------
# C6 — Each legacy-ported slash command renders something to ChatLog
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_slash_status_renders(mock_agent):
    """C6 /status: produces a block containing model + mode."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/status")
        await pilot.press("enter")
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        joined = "\n".join(str(getattr(c, "renderable", c)) for c in log.children)
        assert "mock/llama" in joined  # model name from fixture
        assert "interactive" in joined  # mode from fixture


@pytest.mark.asyncio
async def test_pilot_slash_modules_no_modules(mock_agent):
    """C6 /modules with empty modules dict: prints 'No modules ... pure chat'."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/modules")
        await pilot.press("enter")
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        joined = "\n".join(str(getattr(c, "renderable", c)) for c in log.children)
        assert "No modules" in joined or "pure chat mode" in joined


@pytest.mark.asyncio
async def test_pilot_slash_tools_groups_by_tier(mock_agent_with_tools):
    """C6 /tools with fixture-registered tools: renders both tools grouped
    by tier (default / admin)."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent_with_tools)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/tools")
        await pilot.press("enter")
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        joined = "\n".join(str(getattr(c, "renderable", c)) for c in log.children)
        assert "read_files" in joined
        assert "shell_run" in joined
        assert "default" in joined  # tier header
        assert "admin" in joined


@pytest.mark.asyncio
async def test_pilot_slash_clear_resets_chatlog(mock_agent):
    """C6 /clear: empties chat-log widgets + clears agent.clear_conversation."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Add some content first
        await pilot.press(*"/help")
        await pilot.press("enter")
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        assert len(log.children) >= 1  # at least the help table

        # /clear
        await pilot.press(*"/clear")
        await pilot.press("enter")
        await pilot.pause()

        # After clear the only remaining child should be the "Conversation cleared"
        # status line; help table is gone.
        bodies = [str(getattr(c, "renderable", c)) for c in log.children]
        joined = "\n".join(bodies)
        assert "Conversation cleared" in joined, f"missing status: {joined[:200]}"
        # Available Commands table from /help should have been wiped
        assert "Available Commands" not in joined
        # Agent's clear_conversation invoked
        assert mock_agent.clear_conversation.called


@pytest.mark.asyncio
async def test_pilot_slash_abort_calls_agent_abort(mock_agent):
    """C6 /abort: invokes agent.abort()."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/abort")
        await pilot.press("enter")
        await pilot.pause()
        assert mock_agent.abort.called


@pytest.mark.asyncio
async def test_pilot_slash_mode_invalid_keeps_mode(mock_agent):
    """C6 /mode bogus: agent.mode unchanged + error rendered."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        # mode starts interactive (fixture default)
        assert mock_agent.mode == "interactive"
        await pilot.press(*"/mode bogus")
        await pilot.press("enter")
        await pilot.pause()
        assert mock_agent.mode == "interactive", "mode must not change"
        log = app.query_one("#chat-log", ChatLog)
        joined = "\n".join(str(getattr(c, "renderable", c)) for c in log.children)
        assert "Invalid mode" in joined


@pytest.mark.asyncio
async def test_pilot_slash_mode_task_switches(mock_agent):
    """C6 /mode task: agent.set_mode called + agent.mode flips."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/mode task")
        await pilot.press("enter")
        await pilot.pause()
        assert mock_agent.mode == "task"
        mock_agent.set_mode.assert_called_with("task")


# ---------------------------------------------------------------------------
# C6 — Footer BINDINGS shortcuts route through dispatch_slash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_ctrl_l_runs_clear(mock_agent):
    """C6 BINDINGS: Ctrl+L invokes the same path as /clear."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Add a help table so we can verify clear wipes it
        await pilot.press(*"/help")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("ctrl+l")
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        joined = "\n".join(str(getattr(c, "renderable", c)) for c in log.children)
        assert "Conversation cleared" in joined
        assert "Available Commands" not in joined


@pytest.mark.asyncio
async def test_pilot_ctrl_g_runs_abort(mock_agent):
    """C6 BINDINGS: Ctrl+G invokes /abort path."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert mock_agent.abort.called


# ---------------------------------------------------------------------------
# C7 — /mode continuous opens setup modal; cancel keeps interactive mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_mode_continuous_pushes_setup_modal(mock_agent):
    """C7: /mode continuous pushes ContinuousSetupModal but does NOT
    set_mode yet — the modal's dismiss callback is responsible."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.screens import ContinuousSetupModal

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/mode continuous")
        await pilot.press("enter")
        await pilot.pause()
        # Modal screen on top of stack
        assert len(app.screen_stack) >= 2
        assert isinstance(app.screen_stack[-1], ContinuousSetupModal)
        # Agent mode untouched until modal commits
        assert mock_agent.mode == "interactive"


@pytest.mark.asyncio
async def test_pilot_continuous_setup_cancel_leaves_mode(mock_agent):
    """C7: cancelling ContinuousSetupModal via Esc keeps interactive mode."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/mode continuous")
        await pilot.press("enter")
        await pilot.pause()
        # Modal pushed; press Esc to cancel
        await pilot.press("escape")
        await pilot.pause()
        # Back to single-screen stack
        assert len(app.screen_stack) == 1
        # Mode unchanged, runner not built
        assert mock_agent.mode == "interactive"
        assert getattr(app, "_runner", None) is None


# ---------------------------------------------------------------------------
# C7 — /stop without active runner reports "Not in continuous mode"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_stop_without_runner(mock_agent):
    """C7 /stop guard: no runner → friendly message, no crash."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/stop")
        await pilot.press("enter")
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        joined = "\n".join(str(getattr(c, "renderable", c)) for c in log.children)
        assert "Not in continuous mode" in joined


# ---------------------------------------------------------------------------
# C8 — App._handle_exception override writes to log file (not stderr)
# ---------------------------------------------------------------------------


def test_handle_exception_writes_log_and_sets_return_code(tmp_path, monkeypatch, mock_agent):
    """C0 / C8 KPI #1: unhandled exceptions don't dump traceback to
    host scrollback — they go to ~/.llamagent/cli_tui.log + one-line
    notice. Verified statically by mocking Path.home() to a temp dir
    and feeding a synthetic exception into _handle_exception."""
    from pathlib import Path
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    app = LlamAgentTUI(agent=mock_agent)
    # Avoid calling .exit() which would tear down the un-mounted App
    monkeypatch.setattr(app, "exit", lambda: None)
    try:
        raise RuntimeError("synthetic crash for testing")
    except RuntimeError as exc:
        app._handle_exception(exc)
    log = tmp_path / ".llamagent" / "cli_tui.log"
    assert log.exists()
    contents = log.read_text()
    assert "synthetic crash for testing" in contents
    assert app._return_code == 1
    assert app._crash_notice
    assert str(log) in app._crash_notice


# ---------------------------------------------------------------------------
# Modal Enter event bubble fix (2026-05-24 user crash report)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_modal_input_submit_does_not_crash_app(mock_agent):
    """2026-05-24 crash fix: Enter inside a modal Input must NOT bubble
    up to App.on_input_submitted and crash with NoMatches('#chat-log').

    We trigger this by opening ContinuousSetupModal and pressing Enter
    inside one of its inputs — the modal should consume the event via
    its own on_input_submitted (which calls event.stop() now)."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Push the continuous setup modal
        await pilot.press(*"/mode continuous")
        await pilot.press("enter")
        await pilot.pause()
        # Press Enter inside the modal — if the bubble fix is missing
        # this crashes with NoMatches('#chat-log'). After fix, modal's
        # _do_build runs with empty inputs → renders #cs-error.
        await pilot.press("enter")
        await pilot.pause()
        # App still alive (no crash), still in interactive mode (validation
        # failed, modal didn't dismiss).
        assert mock_agent.mode == "interactive"
        # Modal still on screen (Build failed validation, no dismiss)
        assert len(app.screen_stack) >= 2


# ---------------------------------------------------------------------------
# History dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_history_dedup_consecutive(mock_agent):
    """C6.1: submitting the same command twice records only one entry."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import LlamAgentInput

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/help")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press(*"/help")
        await pilot.press("enter")
        await pilot.pause()
        inp = app.query_one("#input", LlamAgentInput)
        assert list(inp._history) == ["/help"], f"dedup failed: {list(inp._history)}"


# ---------------------------------------------------------------------------
# C5 — VerbosePane Thinking message handler routes to the pane
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_thinking_message_routes_to_verbose_pane(mock_agent):
    """C5 routing: posting a ThinkingMessage directly to VerbosePane mounts
    a Static carrying the source + content. Verifies the handler chain
    independently of verbose.py's chain-walker patching."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.messages import ThinkingMessage
    from llamagent.interfaces.cli_tui.widgets import VerbosePane

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        pane = app.query_one("#verbose-pane", VerbosePane)
        pane.post_message(
            ThinkingMessage(source="reasoning_content", content="step 1: hello", dedup_hash="x")
        )
        await pilot.pause()
        bodies = [str(getattr(c, "renderable", c)) for c in pane.children]
        joined = "\n".join(bodies)
        assert "step 1: hello" in joined
        assert "reasoning_content" in joined


# ---------------------------------------------------------------------------
# C2 / C5 — ToolStartMessage routes to BOTH ChatLog (folded card) and VerbosePane
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_tool_message_fans_out_to_both(mock_agent):
    """C5 round-12 B1: install_hooks(target, verbose_target) wiring fans
    ToolStartMessage out to both ChatLog (folded card) and VerbosePane
    (full args repr). Verify by posting to both widgets directly so the
    test doesn't depend on a real ReAct turn."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.messages import ToolStartMessage
    from llamagent.interfaces.cli_tui.widgets import ChatLog, VerbosePane

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        pane = app.query_one("#verbose-pane", VerbosePane)
        msg_to_log = ToolStartMessage(name="read_files", args={"paths": ["a.py"]}, call_id="t1-0")
        msg_to_pane = ToolStartMessage(name="read_files", args={"paths": ["a.py"]}, call_id="t1-0")
        log.post_message(msg_to_log)
        pane.post_message(msg_to_pane)
        await pilot.pause()
        log_text = "\n".join(str(getattr(c, "renderable", c)) for c in log.children)
        pane_text = "\n".join(str(getattr(c, "renderable", c)) for c in pane.children)
        assert "read_files" in log_text
        assert "read_files" in pane_text
        # ChatLog renders as folded card with "tool:" prefix
        assert "tool:" in log_text
        # VerbosePane renders full args repr
        assert "args:" in pane_text


# ---------------------------------------------------------------------------
# C7 — MonitorPane renders idle text when no runner attached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_monitor_pane_idle_render(mock_agent):
    """C7 V1+: MonitorPane shows "Continuous mode not active" until a
    runner is attached. We don't attach a runner here so the idle path
    is the one that renders."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import MonitorPane

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        pane = app.query_one("#monitor-pane", MonitorPane)
        # Open monitor so it actually renders
        await pilot.press(*"/monitor on")
        await pilot.press("enter")
        # set_interval fires every 1.0s — wait long enough to catch one tick
        await pilot.pause(1.2)
        body = "\n".join(str(getattr(c, "renderable", c)) for c in pane.children)
        assert "Continuous mode not active" in body


# ---------------------------------------------------------------------------
# C8 — ImportError fallback path: confirm cli.main calls legacy_main when
# cli_tui import fails
# ---------------------------------------------------------------------------


def test_cli_main_falls_back_when_cli_tui_import_fails(monkeypatch, capsys, mock_agent):
    """C8 round-16 fallback: simulate ``from cli_tui import run`` raising
    ImportError → main() prints a stderr notice and calls legacy_main."""
    from llamagent.interfaces import cli as cli_mod

    legacy_called = []

    def fake_legacy_main(args):
        legacy_called.append(args)

    monkeypatch.setattr(cli_mod, "legacy_main", fake_legacy_main)
    monkeypatch.setattr(cli_mod, "_terminal_supports_tui", lambda: True)

    # Replace the cli_tui submodule in sys.modules so the deferred
    # ``from llamagent.interfaces.cli_tui import run`` raises.
    import sys

    class _Broken:
        def __getattr__(self, name):
            raise ImportError(f"simulated ({name})")

    monkeypatch.setitem(sys.modules, "llamagent.interfaces.cli_tui", _Broken())

    # Call main() with default args → routes to TUI → ImportError → fallback
    monkeypatch.setattr(sys, "argv", ["llamagent-cli"])
    cli_mod.main()
    captured = capsys.readouterr()
    assert "Textual not available" in captured.err
    assert legacy_called  # legacy_main was invoked


# ---------------------------------------------------------------------------
# Round-18 A-7 — Test-gap closure for high-risk new paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_chatlog_clear_public_resets_state(mock_agent):
    """Round-18 A-9: ChatLog.clear() is the single source of truth for
    "blank the chat log". After calling it, _current_assistant /
    _accum / _pending_tool_cards should all be reset."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        # Seed some streaming state and tool card state.
        log.append_user("hello")
        log.append_assistant_chunk("partial reply ")
        log._pending_tool_cards["call-1"] = (object(), "running")
        assert len(list(log.children)) >= 1
        assert log._current_assistant is not None
        assert log._accum
        assert log._pending_tool_cards

        log.clear()
        # child.remove() in Textual is async — let the event loop process
        # the scheduled removals before asserting child count.
        await pilot.pause()

        assert len(list(log.children)) == 0
        assert log._current_assistant is None
        assert log._accum == []
        assert log._pending_tool_cards == {}


@pytest.mark.asyncio
async def test_pilot_inject_reply_renders_standalone_bubble(mock_agent):
    """Round-18 A-4: InjectReplyMessage mounts a self-contained bubble
    instead of accumulating into _current_assistant. Two concurrent
    inject replies should produce two separate bubbles."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.messages import InjectReplyMessage
    from llamagent.interfaces.cli_tui.widgets import ChatLog

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        log = app.query_one("#chat-log", ChatLog)
        initial_count = len(list(log.children))

        # Open a streaming bubble first so we can verify inject reply
        # finalizes it before mounting its own block.
        log.append_assistant_chunk("streaming partial ")
        assert log._current_assistant is not None

        log.post_message(InjectReplyMessage(text="reply A"))
        log.post_message(InjectReplyMessage(text="reply B"))
        await pilot.pause()
        await pilot.pause()

        # Open streaming bubble was finalized + 2 standalone inject bubbles
        # were mounted. Total children grew by at least 3 (1 finalized +
        # 2 standalone).
        assert len(list(log.children)) - initial_count >= 3
        # No lingering streaming target / accumulator.
        assert log._current_assistant is None
        assert log._accum == []
        # Round-18-2: assert the two inject replies actually rendered in
        # the right order and aren't merged into one bubble. Each bubble
        # is a Static wrapping a rich Group of (Text + Markdown); pull
        # the Markdown source text out of each Group so we can compare
        # against the posted strings.
        from rich.console import Group
        from rich.markdown import Markdown

        def _bubble_text(child) -> str:
            # Round-18-3 LOW: ``Markdown.markup`` is an undocumented Rich
            # attribute (stable across 12.x..14.x today). Add ``str()``
            # fallback so a rename in a future Rich keeps the test
            # informative rather than silently returning "" for every
            # bubble (which would surface as a confusing "bubble not
            # found" rather than a clear extractor-broke message).
            r = getattr(child, "renderable", None)
            if isinstance(r, Group):
                for sub in getattr(r, "renderables", []):
                    if isinstance(sub, Markdown):
                        return getattr(sub, "markup", None) or str(sub)
            return ""

        bubble_texts = [_bubble_text(c) for c in log.children]
        idx_a = next((i for i, t in enumerate(bubble_texts) if "reply A" in t), -1)
        idx_b = next((i for i, t in enumerate(bubble_texts) if "reply B" in t), -1)
        assert idx_a >= 0, f"reply A bubble not found: {bubble_texts}"
        assert idx_b >= 0, f"reply B bubble not found: {bubble_texts}"
        assert idx_a < idx_b, "inject reply order should match post order"


@pytest.mark.asyncio
async def test_pilot_modal_response_timeout_returns_sentinel(mock_agent):
    """Round-18 A-7: bridge.install_handlers wires a 60s timeout sentinel
    so an undismissed ConfirmModal eventually returns ConfirmResponse(
    allow=False) instead of hanging the worker forever. Shorten the
    timeout via monkeypatch and verify the sentinel fires."""
    import threading
    from llamagent.interfaces.cli_tui import bridge as bridge_mod
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.core.zone import ConfirmRequest

    # Cut the modal-response timeout from 60s to 0.2s for the test.
    monkeypatched = 0.2
    original_timeout = bridge_mod.MODAL_RESPONSE_TIMEOUT_S
    bridge_mod.MODAL_RESPONSE_TIMEOUT_S = monkeypatched

    try:
        app = LlamAgentTUI(agent=mock_agent)
        async with app.run_test() as pilot:
            await pilot.pause()
            # install_handlers wires confirm_handler in on_mount.
            handler = mock_agent.confirm_handler
            assert handler is not None, "install_handlers should wire confirm_handler"

            req = ConfirmRequest(
                kind="operation_confirm",
                tool_name="write_files",
                action="write",
                zone="project",
                target_paths=["x"],
                message="write file",
            )
            result_box: dict = {}

            def _call():
                # Don't dismiss the modal — the timeout should fire and
                # return the sentinel response. handler is a plain
                # callable (agent.confirm_handler is wired as a function
                # by install_handlers, not an object with .confirm).
                result_box["resp"] = handler(req)

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            # Modal pushes on App; let event loop process it and the
            # timer expire (timeout is 0.2s).
            for _ in range(40):
                await pilot.pause()
                if not t.is_alive():
                    break
            t.join(timeout=3.0)
            assert not t.is_alive(), "handler did not return within timeout window"
            resp = result_box.get("resp")
            assert resp is not None
            # Sentinel response: allow=False with no approved scopes.
            assert resp.allow is False
            assert resp.approved_scopes is None
    finally:
        bridge_mod.MODAL_RESPONSE_TIMEOUT_S = original_timeout


@pytest.mark.asyncio
async def test_pilot_continuous_setup_happy_path_starts_runner(mock_agent):
    """Round-18 A-3 / A-7 test gap: ContinuousSetupModal dismiss with a
    valid timer dict starts the runner thread, swaps mode, opens
    MonitorPane. /stop then joins the thread + restores interactive."""
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI

    app = LlamAgentTUI(agent=mock_agent)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Drive _on_continuous_setup_done directly with a valid dict so
        # we don't depend on ContinuousSetupModal's form fields/keystroke
        # mechanics (which are exercised by other tests).
        # Use a long interval so the runner doesn't actually fire before
        # we stop it.
        app._on_continuous_setup_done({
            "type": "timer",
            "interval": 999.0,
            "message": "tick",
            "fire_immediately": False,
        })
        await pilot.pause()
        await pilot.pause()

        assert app._runner is not None, "runner should be built"
        assert app._runner_thread is not None
        assert app._runner_thread.is_alive(), "runner thread should be alive"
        assert mock_agent.mode == "continuous", "mode should flip"

        # /stop should cleanly join the thread + restore interactive
        await pilot.press(*"/stop")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert app._runner is None
        assert app._runner_thread is None
        assert mock_agent.mode == "interactive"


def test_disabled_tools_inherits_into_child_agent_as_copy():
    """Round-18 B-1 / B-7 test gap: a child agent built via
    ChildAgentModule._create_child_agent gets a COPY of parent's
    disabled_tools list, so a runtime mutation on parent doesn't
    silently re-shape the child's tool surface.

    Round-18-2 fix: invoke the real production path
    (``ChildAgentModule._create_child_agent``) so a regression that
    swaps ``list(...)`` back to a shared reference is actually caught.
    The previous version re-asserted the contract inline and would
    pass even if module.py was reverted."""
    from llamagent.core import Config, LlamAgent
    from llamagent.modules.child_agent import ChildAgentModule
    from llamagent.modules.child_agent.policy import (
        AgentExecutionPolicy,
        ChildAgentSpec,
    )

    parent_config = Config()
    parent_config.disabled_tools = ["ask_user", "web_search"]
    parent = LlamAgent(parent_config)
    mod = ChildAgentModule()
    parent.register_module(mod)
    try:
        spec = ChildAgentSpec(
            task="probe",
            role="worker",
            policy=AgentExecutionPolicy(),
            parent_task_id=None,
            task_id="test-task-id",
        )
        child = mod._create_child_agent(spec)
        try:
            # Production path produced a value-equal but reference-distinct list.
            assert child.config.disabled_tools == ["ask_user", "web_search"]
            assert child.config.disabled_tools is not parent.config.disabled_tools
            # Mutating parent must NOT propagate.
            parent.config.disabled_tools.append("web_fetch")
            assert "web_fetch" not in child.config.disabled_tools
        finally:
            child.shutdown()
    finally:
        parent.shutdown()
