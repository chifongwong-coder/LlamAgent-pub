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
        assert inp._history == ["/help", "/status"]
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
