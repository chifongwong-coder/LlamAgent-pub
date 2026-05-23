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

    BINDINGS = [("ctrl+c", "quit", "Quit")]

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
        yield ChatLog(id="chat-log")
        yield Input(
            placeholder="Type a message and press Enter (slash for commands)…",
            suggester=SlashCommandSuggester(),
            id="input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_status_header()

        self.title = (
            f"LlamAgent TUI — turns: 0 / {self.n_mock_turns or '∞'}"
        )

        # Defer Input.focus() until all widget mount cycles complete
        # (plan v11 §2.11.5 — VerticalScroll race fix verified Step 3).
        self.call_after_refresh(lambda: self.query_one("#input", Input).focus())

        # C2: register hooks on real agent so PRE/POST/TOOL_ERROR
        # produce ToolStart/End/Error Messages. install_hooks is
        # idempotent so multiple App rebuilds on same agent are safe.
        if self.agent is not None:
            from llamagent.interfaces.cli_tui.bridge import install_hooks
            install_hooks(self.agent, self)

        if self.n_mock_turns > 0:
            self.set_timer(0.05, self._run_mock_turn)
            return

        # C3: launch the SetupScreen when no agent was pre-constructed
        # (mock-turn smoke mode bypasses this). SetupScreen returns a
        # dict that _on_setup_done feeds into _build_agent_from_setup.
        if self.agent is None:
            from llamagent.interfaces.cli_tui.screens import SetupScreen
            self.push_screen(SetupScreen(), self._on_setup_done)

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
        hooks. Handles agent-rebuild path by uninstalling the previous
        agent's hooks first (round-8 Rev A M1 / plan v13 §12).
        """
        from llamagent.interfaces.cli_tui.bridge import (
            install_hooks,
            uninstall_hooks,
        )

        if self.agent is not None and self.agent is not agent:
            uninstall_hooks(self.agent)

        self.agent = agent
        self.refresh_status_header()
        install_hooks(agent, self)
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
        if text in ("/quit", "/exit", "/q"):
            self.exit()
            return
        log = self.query_one("#chat-log", ChatLog)
        log.append_user(text)
        event.input.value = ""

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

    def _run_real_turn(self, user_input: str) -> None:
        """Spawn a worker thread to iterate ``agent.chat_stream``.

        Uses Textual's ``run_worker`` (thread mode) — @work decorator
        would attach the worker as an instance method which requires
        a class-level declaration. ``run_worker`` lets us pass a plain
        callable, which is cleaner for the C2 bridge.
        """
        from llamagent.interfaces.cli_tui.bridge import run_turn
        agent = self.agent
        self.run_worker(
            lambda: run_turn(self, agent, user_input),
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


