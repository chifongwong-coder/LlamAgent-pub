"""LlamAgent TUI App (cli_tui main class).

C0 Spike scaffold + C1.a widget extraction:
- ChatLog / StatusHeader live in widgets.py (production widgets)
- Message dataclasses in messages.py (used by C1.b+ Message handlers)
- This file owns App-level concerns: compose, on_mount, _handle_exception
  override (Q6 mitigation), and the smoke-mode mock-turn machinery used
  by C0 Spike to validate KPIs.

Q6 mitigation (plan v11 §2.2): override App._handle_exception to write
traceback to ~/.llamagent/cli_tui.log + stash one-line notice; smoke.py
emits the notice to stderr after app.run() returns. Real-terminal
verified 85 bytes scrollback (was 2415 baseline).

Input focus (plan v11 §2.11.5): ChatLog declares can_focus=False so it
doesn't steal focus from Input; on_mount uses call_after_refresh to
defer Input.focus() until all widget mount cycles complete.
"""
import traceback
from datetime import datetime
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Input

from llamagent.interfaces.cli_tui.messages import (
    ChatChunkMessage,
    ToolEndMessage,
    ToolStartMessage,
)
from llamagent.interfaces.cli_tui.widgets import (
    ChatLog,
    SlashCommandSuggester,
    StatusHeader,
    VerbosePane,
)


class LlamAgentTUI(App):
    """LlamAgent terminal UI.

    Smoke / spike mode: when ``n_mock_turns > 0`` auto-runs N mock chat
    turns and exits, used by ``cli_tui.smoke`` for C0 KPI validation.

    Interactive mode (``n_mock_turns == 0``): waits for user input via
    the Input widget. Real agent integration lands in C2 (worker thread
    consuming ``agent.chat_stream`` and posting ChatChunkMessage).
    """

    CSS = """
    Screen { layout: vertical; }
    #input { dock: bottom; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        # Esc is NOT priority — modal screens (ConfirmModal /
        # AskUserModal / SetupScreen) install their own escape
        # bindings to dismiss-with-sentinel. priority=True on the
        # App's escape would shadow those (round-11 BLOCKER B1).
        # When no modal is on the stack, Esc bubbles up to the App
        # and quits — matches the C3.f intent for the chat surface.
        Binding("escape", "quit", "Quit"),
        # C5 — VerbosePane toggle is exposed as the `/verbose` slash
        # command (see commands.py::cmd_verbose), NOT a keyboard binding.
        #
        # Round-14 user-test history: Ctrl+V silently exited the App on
        # macOS Terminal.app (termios VLNEXT collision); F3 was grabbed
        # by macOS Mission Control's Window Overview before Textual saw
        # it. Every other plausible binding — Ctrl+S/XOFF, Ctrl+A/
        # beginning-of-line, F-keys 1-12 (multimedia / brightness /
        # mission control by default) — has a similar reservation on at
        # least one common macOS terminal. A slash command sidesteps the
        # whole class: it travels through the Input widget as plain
        # text, so no terminal can intercept it.
        # C6 — Footer shortcuts for the three most-used slash commands
        # (round-7 LOW A-1). Each routes through dispatch_slash so behaviour
        # stays identical to typing the command. Modal screen guard mirrors
        # round-12 M3 — the actions short-circuit when a modal is on top.
        #
        # Round-14 binding choices (Rev B H1 + M1):
        # - Ctrl+L → /clear: Textual captures Ctrl+L before the PTY, no collision.
        # - Ctrl+G → /abort: GNU readline "abort current command"; zero collision
        #   with Input widget conventions (Ctrl+A would clash with readline
        #   "beginning-of-line" muscle memory).
        # - F2     → /stop : Function keys have no terminal-level meaning, so
        #   they side-step the Ctrl+S XOFF flow-control collision that would
        #   freeze the PTY in tmux / mosh / certain macOS Terminal profiles.
        Binding("ctrl+l", "slash_clear", "Clear"),
        Binding("ctrl+g", "slash_abort", "Abort"),
        Binding("f2", "slash_stop", "Stop"),
    ]

    def __init__(
        self,
        agent=None,
        n_mock_turns: int = 0,
        crash_after_turns: int | None = None,
    ) -> None:
        super().__init__()
        # C2: when ``agent`` is set, on_input_submitted dispatches into a
        # worker thread that iterates agent.chat_stream. When None, the
        # interactive path falls back to the spike echo (used by smoke 0
        # and IME tests). Smoke mode (n_mock_turns > 0) ignores ``agent``
        # entirely and runs the deterministic mock turn loop.
        self.agent = agent
        self.n_mock_turns = n_mock_turns
        self._turns_completed = 0
        self._crash_after_turns = crash_after_turns

    def compose(self) -> ComposeResult:
        yield StatusHeader()
        # C5 — split chat row into Horizontal so VerbosePane can sit
        # on the right column. ChatLog gets 1fr (flex), VerbosePane
        # default-hidden width 40. Together they fill the row.
        with Horizontal(id="chat-row"):
            yield ChatLog(id="chat-log")
            yield VerbosePane(id="verbose-pane")
        yield Input(
            placeholder="Type a message and press Enter (slash for commands)…",
            suggester=SlashCommandSuggester(),
            id="input",
        )
        yield Footer()

    def action_toggle_verbose(self) -> None:
        """F3 toggle for VerbosePane (plan §2.3 / §C5).

        Toggles ``.display`` rather than mounting/unmounting so the
        widget retains its message history across hide/show cycles.
        Default is hidden (DEFAULT_CSS sets display:none).

        Round-12 M3: guard against modal screens — when SetupScreen /
        ConfirmModal / AskUserModal is active, F3 would still flip the
        VerbosePane below the modal overlay, leaving the user with a
        surprising visible-state change after dismiss. Only act on the
        default screen.

        Round-14 user-test follow-up: wrap the body in try/except so any
        unexpected error (Textual layout edge case, reactive surprise)
        renders as a chat-log error bubble instead of crashing the App
        through ``_handle_exception``. The original Ctrl+V crash report
        turned out to be a terminal/key collision (Ctrl+V is termios
        VLNEXT) — we changed the binding to F3 — but the defensive
        try/except is still worth keeping for any future regression.
        """
        try:
            if len(self.screen_stack) > 1:
                return
            try:
                pane = self.query_one("#verbose-pane", VerbosePane)
            except Exception:
                return
            pane.display = not pane.display
        except Exception as e:
            try:
                self.query_one("#chat-log", ChatLog).append_error(
                    f"toggle_verbose failed: {type(e).__name__}: {e}"
                )
            except Exception:
                pass

    def on_mount(self) -> None:
        self.refresh_status_header()

        self.title = (
            f"LlamAgent TUI — turns: 0 / {self.n_mock_turns or '∞'}"
        )

        # Install hooks on the pre-built agent (scripted / smoke path).
        # Interactive (SetupScreen) path defers hook install to set_agent.
        # Target the ChatLog widget directly — Textual on_<msg> handlers
        # only fire on the widget that received post_message; routing
        # via App leaves messages stranded in App's queue (5/23 bug).
        # C5 round-12 B1: also pass VerbosePane as verbose_target so
        # ToolStart/End/Error fan out to both widgets (full args /
        # result preview render on the right pane per plan §4 C5).
        # C4 also wires confirm/ask handlers so authorization prompts +
        # ask_user tool calls route through the TUI modal screens.
        if self.agent is not None:
            from llamagent.interfaces.cli_tui.bridge import (
                install_handlers,
                install_hooks,
            )
            from llamagent.interfaces.cli_tui.verbose import install_verbose
            chat_target = self.query_one("#chat-log", ChatLog)
            verbose_target = self.query_one("#verbose-pane", VerbosePane)
            install_hooks(self.agent, chat_target, verbose_target=verbose_target)
            install_handlers(self.agent, self)
            install_verbose(self.agent, verbose_target)

        if self.n_mock_turns > 0:
            # Mock smoke — no SetupScreen, focus Input directly.
            self.call_after_refresh(lambda: self.query_one("#input", Input).focus())
            self.set_timer(0.05, self._run_mock_turn)
            return

        if self.agent is None:
            # Interactive: SetupScreen pushed below owns focus while
            # mounted. After it dismisses, set_agent() re-focuses
            # App's Input. We MUST NOT schedule an Input.focus() here:
            # call_after_refresh runs after the next render tick — by
            # then SetupScreen is the active screen and App.query_one
            # routes through it, raising NoMatches for "#input" which
            # lives in the App's default screen (test B crash 5/23).
            #
            # Deferring push_screen via call_after_refresh: pushing
            # during on_mount can race with the default screen's
            # mount cycle, leaving the modal mounted but without a
            # focused child — keystrokes go nowhere (test B 5/23
            # re-run: "打字 / Esc / Tab 全无反应"). Deferring to the
            # next refresh tick lets the default screen settle first.
            def _push_setup() -> None:
                from llamagent.interfaces.cli_tui.screens import SetupScreen
                self.push_screen(SetupScreen(), self._on_setup_done)
            self.call_after_refresh(_push_setup)
            return

        # Scripted path — agent pre-built, no SetupScreen, focus Input.
        self.call_after_refresh(lambda: self.query_one("#input", Input).focus())

    # ------------------------------------------------------------------
    # C3 — SetupScreen integration
    # ------------------------------------------------------------------

    def _on_setup_done(self, result: dict | None) -> None:
        """Callback after SetupScreen dismisses.

        ``None`` means the user cancelled — quit the App. Otherwise
        build the LlamAgent and bind it via ``set_agent``. If build
        raises, surface the error in ChatLog AND re-push SetupScreen
        so the user can fix their selection and retry (round-10
        HIGH A-H2). Without the re-push the user is dead-ended in an
        empty ChatLog with no way to retry except killing the App.
        """
        if result is None:
            self.exit()
            return
        from llamagent.interfaces.cli_tui.__main__ import build_agent_from_setup
        try:
            agent = build_agent_from_setup(result)
        except Exception as exc:
            log = self.query_one("#chat-log", ChatLog)
            log.append_error(f"setup failed: {type(exc).__name__}: {exc}")
            from llamagent.interfaces.cli_tui.screens import SetupScreen
            self.push_screen(SetupScreen(), self._on_setup_done)
            return
        self.set_agent(agent)

    def set_agent(self, agent) -> None:
        """Bind ``agent`` to the App, refresh StatusHeader, install
        hooks + confirm/ask handlers. Handles agent-rebuild path by
        uninstalling the previous agent's hooks + handlers first
        (round-8 Rev A M1 / plan v13 §12 + C4 §4 verification).
        """
        from llamagent.interfaces.cli_tui.bridge import (
            install_handlers,
            install_hooks,
            uninstall_handlers,
            uninstall_hooks,
        )
        from llamagent.interfaces.cli_tui.verbose import (
            install_verbose,
            uninstall_verbose,
        )

        if self.agent is not None and self.agent is not agent:
            uninstall_hooks(self.agent)
            uninstall_handlers(self.agent)
            uninstall_verbose(self.agent)

        self.agent = agent
        self.refresh_status_header()
        # Hooks target the ChatLog widget directly (handlers live there).
        # C5 round-12 B1: verbose_target=VerbosePane so Tool*Message
        # fans out to the right pane too (plan §4 C5).
        chat_target = self.query_one("#chat-log", ChatLog)
        verbose_target = self.query_one("#verbose-pane", VerbosePane)
        install_hooks(agent, chat_target, verbose_target=verbose_target)
        # Confirm / ask handlers push modal screens via the App.
        install_handlers(agent, self)
        # C5 — thinking capture posts to the VerbosePane widget (which
        # is hidden by default; user toggles via Ctrl+V to see).
        install_verbose(agent, verbose_target)
        # Re-focus the Input so user can immediately type after build.
        self.call_after_refresh(lambda: self.query_one("#input", Input).focus())

    def refresh_status_header(self) -> None:
        """Re-read agent attributes into the StatusHeader reactives.

        Called from set_agent and (eventually) from slash-command
        handlers like ``/mode`` that mutate ``agent.mode`` after build
        — round-8 Rev C H1 / plan v13 §12. Without this the StatusHeader
        shows the stale value because reactive properties don't poll.
        """
        header = self.query_one(StatusHeader)
        if self.agent is not None:
            header.model = str(self.agent.config.model)
            header.persona = (
                self.agent.persona.name
                if getattr(self.agent, "persona", None)
                else "default"
            )
            header.mode = self.agent.mode
            header.modules_count = (
                len(self.agent.modules)
                if hasattr(self.agent, "modules")
                else 0
            )
        else:
            header.model = "(setup pending)"
            header.persona = "—"
            header.mode = "interactive"
            header.modules_count = 0

    # ------------------------------------------------------------------
    # Smoke mode — used by C0 Spike to validate alt-screen + Q6 KPIs
    # ------------------------------------------------------------------

    def _run_mock_turn(self) -> None:
        if self._turns_completed >= self.n_mock_turns:
            # NOTE (round-7 HIGH-2): the crash variant fires AFTER all
            # n_mock_turns complete, not mid-stream. This still validates
            # the same _handle_exception path — Textual's internal catch
            # + alt-screen exit + Rich Console traceback emission happens
            # identically regardless of WHEN the exception is raised. The
            # 85-byte KPI #12 result is therefore valid. If true mid-stream
            # crash semantics are needed for a future test, add a separate
            # --crash-at-turn N parameter rather than repurposing this one.
            if (
                self._crash_after_turns is not None
                and self._turns_completed >= self._crash_after_turns
            ):
                raise RuntimeError(
                    "C0 Spike intentional crash — verifying plan v11 §11 Q6 "
                    "(does exception leak to host scrollback after alt-screen exit?)"
                )
            self.exit()
            return

        log = self.query_one("#chat-log", ChatLog)
        n = self._turns_completed + 1

        # User turn — direct call (display-only, no Message round-trip needed).
        log.append_user(f"turn {n}")

        # Assistant chunks — go through ChatChunkMessage so the C1.b
        # Message-driven rendering path is exercised by every smoke run.
        # This is the end-to-end validation that messages.py + ChatLog
        # handlers actually wire up correctly inside the event loop.
        log.post_message(ChatChunkMessage(f"Response {n} "))
        log.post_message(ChatChunkMessage("with **markdown** "))
        log.post_message(ChatChunkMessage("and `code` "))

        # Tool call — exercises ToolStartMessage + ToolEndMessage pairing
        # via the call_id map in ChatLog.
        call_id = f"mock-call-{n}"
        log.post_message(
            ToolStartMessage(name="read_files", args={"paths": ["mock.py"]}, call_id=call_id)
        )
        log.post_message(
            ToolEndMessage(call_id=call_id, duration_ms=23.0, result_preview="(mock result)")
        )

        log.post_message(ChatChunkMessage("plus tool follow-up."))
        log.finalize_assistant_bubble()

        self._turns_completed = n
        self.title = f"LlamAgent TUI — turns: {n} / {self.n_mock_turns}"
        self.set_timer(0.02, self._run_mock_turn)

    # ------------------------------------------------------------------
    # Interactive mode — smoke 0 path used for IME validation
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        # C6 — slash command path. dispatch_slash returns True for any
        # input that starts with "/" (handler ran or unknown-command error
        # rendered); only non-slash text flows into the chat pipeline.
        # Round-14 Rev B M2: wrap the import + call so a top-level import
        # error in commands.py (typo, missing dep, refactor regression)
        # doesn't propagate to Textual's unhandled-exception path and kill
        # the session — render a chat-log error instead.
        if text.startswith("/"):
            try:
                from llamagent.interfaces.cli_tui.commands import dispatch_slash
                dispatch_slash(self, text)
            except Exception as e:
                try:
                    self.query_one("#chat-log", ChatLog).append_error(
                        f"slash dispatch crashed: {type(e).__name__}: {e}"
                    )
                except Exception:
                    pass
            return

        log = self.query_one("#chat-log", ChatLog)
        log.append_user(text)

        if self.agent is not None:
            # C2 — real agent path. Spawn worker thread that iterates
            # agent.chat_stream + posts ChatChunkMessage / Tool*Message
            # / TurnCompleteMessage. ``exclusive=True`` ensures only one
            # turn runs at a time (avoids interleaved hook callbacks).
            self._run_real_turn(text)
        else:
            # Spike/IME-test path — synchronous echo so smoke 0 still works.
            log.append_assistant_chunk(f"[Spike echo] {text}")
            log.finalize_assistant_bubble()

    # ------------------------------------------------------------------
    # C6 — Footer keyboard shortcuts (route through dispatch_slash)
    # ------------------------------------------------------------------

    def _shortcut(self, slash: str) -> None:
        """Helper for Ctrl+L / Ctrl+G / F2 — modal-screen guarded slash
        dispatch. Same import-error tolerance as ``on_input_submitted``
        (round-14 Rev B M2) so a broken commands.py can't kill the App
        from a keyboard shortcut either."""
        if len(self.screen_stack) > 1:
            return
        try:
            from llamagent.interfaces.cli_tui.commands import dispatch_slash
            dispatch_slash(self, slash)
        except Exception as e:
            try:
                self.query_one("#chat-log", ChatLog).append_error(
                    f"shortcut dispatch crashed: {type(e).__name__}: {e}"
                )
            except Exception:
                pass

    def action_slash_clear(self) -> None:
        self._shortcut("/clear")

    def action_slash_abort(self) -> None:
        self._shortcut("/abort")

    def action_slash_stop(self) -> None:
        self._shortcut("/stop")

    def _run_real_turn(self, user_input: str) -> None:
        """Spawn a worker thread to iterate ``agent.chat_stream``.

        Uses Textual's ``run_worker`` (thread mode) — @work decorator
        would attach the worker as an instance method which requires
        a class-level declaration. ``run_worker`` lets us pass a plain
        callable, which is cleaner for the C2 bridge.

        Targets the ChatLog widget for post_message so the
        ChatChunk/ToolStart/ToolEnd/TurnComplete handlers actually
        fire — App-level post_message doesn't dispatch to children
        (5/23 bug found via layered LLM diagnostic).
        """
        from llamagent.interfaces.cli_tui.bridge import run_turn
        agent = self.agent
        target = self.query_one("#chat-log", ChatLog)
        self.run_worker(
            lambda: run_turn(target, agent, user_input),
            thread=True,
            exclusive=True,
            group="agent-turn",
        )

    # ------------------------------------------------------------------
    # Q6 crash-path mitigation (plan v11 §2.2)
    # ------------------------------------------------------------------

    def _handle_exception(self, error: Exception) -> None:
        """Override Textual's default unhandled-exception handler.

        Default behavior writes a Rich-formatted traceback (~2.4 KB box-
        drawing chars + ANSI) to stderr after alt-screen exits — lands
        in host scrollback, same nanozone attack surface as the
        2026-05-21 Terminal.app incident.

        We log the full traceback to file and stash a one-line notice
        for the launcher (smoke.py / production main) to emit after
        ``app.run()`` returns. Real-terminal verified: 85 bytes vs 2415
        baseline (28x reduction).
        """
        log_path = Path.home() / ".llamagent" / "cli_tui.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(f"\n=== {datetime.now().isoformat()} unhandled in TUI ===\n")
                traceback.print_exception(type(error), error, error.__traceback__, file=f)
        except Exception:
            pass
        self._crash_notice = f"[llamagent TUI crash — full traceback in {log_path}]"
        self._return_code = 1
        self.exit()


