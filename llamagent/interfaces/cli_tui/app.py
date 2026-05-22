"""Minimal Textual TUI app for C0 Spike.

Layout:
- Header (Textual built-in)
- ChatLog: VerticalScroll containing one Static per message
- Input field (used in interactive mode; ignored in smoke mode)
- Footer

The ChatLog uses Static-per-message (not RichLog) per plan v9 §2.6 —
Static supports update() for in-place stream redraw (post-stream),
keeping the scrollback footprint bounded.

Q6 mitigation (plan v10/v11): Textual catches unhandled exceptions
internally in `_handle_exception` and prints a Rich-formatted traceback
to stderr after alt-screen exits — sys.excepthook is NEVER called for
this path. We override `_handle_exception` to redirect the traceback to
~/.llamagent/cli_tui.log and emit only a single short line to stderr.
"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static


class ChatLog(VerticalScroll):
    """Scrollable column of message bubbles. Each message is one Static.

    Ring buffer cap (max 200 messages per plan v9 §2.6) — evict the
    first child when exceeded. Streaming bubbles use Static.update()
    to redraw in place without appending new lines.

    `can_focus = False`: VerticalScroll defaults to focusable (so users
    can scroll with arrow keys when it has focus). For our chat UX the
    Input should always own focus; arrow-scroll the ChatLog with Page
    Up/Down or mouse wheel instead. Real-terminal Step-3 spike found
    that without this, Input never received keystrokes.
    """

    can_focus = False

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
    }
    """

    MAX_MESSAGES = 200

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_assistant: Static | None = None

    def append_user(self, text: str) -> None:
        self._current_assistant = None
        self._mount_capped(Static(f"[bold]You:[/bold] {text}"))

    def append_assistant_chunk(self, text: str) -> None:
        """Streaming: accumulate into the current assistant bubble."""
        if self._current_assistant is None:
            self._current_assistant = Static(f"[bold cyan]Assistant:[/bold cyan] {text}")
            self._mount_capped(self._current_assistant)
        else:
            existing = self._current_assistant.renderable
            self._current_assistant.update(f"{existing}{text}")

    def finalize_assistant_bubble(self) -> None:
        """Mark the current assistant bubble as complete."""
        self._current_assistant = None

    def append_tool_card(self, name: str, status: str = "running") -> None:
        marker = {"running": "[yellow]⏳[/]", "success": "[green]✓[/]", "error": "[red]✗[/]"}.get(status, "")
        self._mount_capped(Static(f"{marker} tool: [cyan]{name}[/cyan]"))

    def _mount_capped(self, widget: Static) -> None:
        self.mount(widget)
        if len(self.children) > self.MAX_MESSAGES:
            self.children[0].remove()
            if self._current_assistant is not None and self._current_assistant not in self.children:
                self._current_assistant = None


class LlamAgentTUISpike(App):
    """Minimal Textual app — C0 Spike scaffold.

    In smoke mode (n_mock_turns > 0), auto-runs the mock chat loop and
    exits when all turns are done. In interactive mode (n_mock_turns == 0),
    waits for user input.
    """

    CSS = """
    Screen { layout: vertical; }
    #input { dock: bottom; }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, n_mock_turns: int = 0, crash_after_turns: int | None = None):
        super().__init__()
        self.n_mock_turns = n_mock_turns
        self._turns_completed = 0
        self._crash_after_turns = crash_after_turns

    def compose(self) -> ComposeResult:
        yield Header()
        yield ChatLog(id="chat-log")
        yield Input(placeholder="Type a message and press Enter...", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"LlamAgent TUI Spike (C0) — turns: 0 / {self.n_mock_turns or '∞'}"
        # Real-terminal Step-3 found Input never received focus despite
        # an immediate `.focus()` call here. Root cause: VerticalScroll
        # (ChatLog) defaults to focusable and was claiming focus first.
        # Two-layer fix:
        #   1. ChatLog.can_focus = False  (above) — removes from tab order
        #   2. Defer the .focus() one tick so it runs after all widgets
        #      have completed their on_mount cycle.
        self.call_after_refresh(lambda: self.query_one("#input", Input).focus())
        if self.n_mock_turns > 0:
            self.set_timer(0.05, self._run_mock_turn)

    def _handle_exception(self, error: Exception) -> None:
        """Override Textual's default unhandled-exception handler.

        Default behavior writes a Rich-formatted traceback (~2.4 KB box-drawing
        chars + ANSI) to stderr after alt-screen exits — lands in host
        scrollback, same nanozone attack surface as the 2026-05-21 incident.

        We log the full traceback to file and emit only a one-line notice.
        """
        log_path = Path.home() / ".llamagent" / "cli_tui.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(f"\n=== {datetime.now().isoformat()} unhandled in TUI ===\n")
                traceback.print_exception(type(error), error, error.__traceback__, file=f)
        except Exception:
            pass
        # Stash for one-line stderr notice after exit
        self._crash_notice = f"[llamagent TUI crash — full traceback in {log_path}]"
        # Mark non-zero return code + exit alt-screen cleanly
        self._return_code = 1
        self.exit()

    def _run_mock_turn(self) -> None:
        if self._turns_completed >= self.n_mock_turns:
            if self._crash_after_turns is not None and self._turns_completed >= self._crash_after_turns:
                raise RuntimeError(
                    "C0 Spike intentional crash — verifying plan v9 §11 Q6 "
                    "(does exception leak to host scrollback after alt-screen exit?)"
                )
            self.exit()
            return

        log = self.query_one("#chat-log", ChatLog)
        n = self._turns_completed + 1
        log.append_user(f"turn {n}")
        log.append_assistant_chunk(f"Response {n} ")
        log.append_assistant_chunk("with **markdown** ")
        log.append_assistant_chunk("and `code` ")
        log.append_tool_card("read_files", "success")
        log.append_assistant_chunk("plus tool follow-up.")
        log.finalize_assistant_bubble()

        self._turns_completed = n
        self.title = f"LlamAgent TUI Spike (C0) — turns: {n} / {self.n_mock_turns}"
        self.set_timer(0.02, self._run_mock_turn)

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
