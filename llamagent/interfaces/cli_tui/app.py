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
        n_mock_turns: int = 0,
        crash_after_turns: int | None = None,
    ) -> None:
        super().__init__()
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
        # StatusHeader initial values
        header = self.query_one(StatusHeader)
        header.model = "mock (C0 Spike)"
        header.persona = "default"
        header.mode = "interactive"
        header.modules_count = 0

        self.title = (
            f"LlamAgent TUI — turns: 0 / {self.n_mock_turns or '∞'}"
        )

        # Defer Input.focus() until all widget mount cycles complete
        # (plan v11 §2.11.5 — VerticalScroll race fix verified Step 3).
        self.call_after_refresh(lambda: self.query_one("#input", Input).focus())

        if self.n_mock_turns > 0:
            self.set_timer(0.05, self._run_mock_turn)

    # ------------------------------------------------------------------
    # Smoke mode — used by C0 Spike to validate alt-screen + Q6 KPIs
    # ------------------------------------------------------------------

    def _run_mock_turn(self) -> None:
        if self._turns_completed >= self.n_mock_turns:
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
        log.append_assistant_chunk(f"[Spike echo] {text}")
        log.finalize_assistant_bubble()
        event.input.value = ""

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


