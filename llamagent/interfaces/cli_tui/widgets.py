"""Production widgets for cli_tui (plan v11 §3.3).

C1.a — initial extraction from spike app.py + new StatusHeader.
C1.b — Message handlers on ChatLog dispatch worker-thread Messages
       into the visible bubbles. ToolStart/End/Error use a call_id map
       so concurrent / out-of-order completions still pair correctly.
C5 will add VerbosePane (deferred until verbose.py + child counter
push wiring lands).
"""
from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape as markup_escape
from rich.text import Text
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.suggester import SuggestFromList
from textual.widgets import Static

from llamagent.interfaces.cli_tui.messages import (
    ChatChunkMessage,
    ErrorMessage,
    StatusMessage,
    ToolEndMessage,
    ToolErrorMessage,
    ToolStartMessage,
    TurnCompleteMessage,
)


# ---------------------------------------------------------------------------
# Slash command list + Suggester (plan §4 C1 / §4 C6)
# ---------------------------------------------------------------------------


# Plan §4 C6 — 13 slash commands + /mode subcommands. Production C6 will
# bind these to their handlers via commands.py; for C1 we only wire the
# autocomplete surface so the Input widget is feature-complete.
SLASH_COMMANDS: tuple[str, ...] = (
    "/help",
    "/modules",
    "/skills",
    "/memory",
    "/history",
    "/status",
    "/clear",
    "/abort",
    "/stop",
    "/prompts",
    "/quit",
    "/exit",
    "/q",
    "/mode",
    "/mode interactive",
    "/mode task",
    "/mode continuous",
)


class SlashCommandSuggester(SuggestFromList):
    """Slash command autocomplete Suggester for the Input widget.

    Wraps Textual SuggestFromList with the project's static slash list.
    Empty input shows nothing; partial input (e.g. "/m") suggests the
    first match (e.g. "/modules"). Case-insensitive so typing /Q maps
    to /quit. Production C6 will rebuild this list dynamically from the
    command registry; the static list keeps the C1 surface stable.
    """

    def __init__(self) -> None:
        super().__init__(SLASH_COMMANDS, case_sensitive=False)


# ---------------------------------------------------------------------------
# StatusHeader — top status bar (plan §3.1 layout)
# ---------------------------------------------------------------------------


class StatusHeader(Static):
    """Top status bar: model | persona | mode | modules | child counter.

    Replaces Textual's default Header (which is decorative-only). Reactive
    attributes update the rendered text via Textual's reactive system —
    e.g., switching mode via ``status_header.mode = "task"``.

    Fixed-width slot for child counter (right-justified) avoids the L2
    Header-jitter risk flagged in round-6 review.
    """

    DEFAULT_CSS = """
    StatusHeader {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    """

    model: reactive[str] = reactive("model: unknown")
    persona: reactive[str] = reactive("default")
    mode: reactive[str] = reactive("interactive")
    modules_count: reactive[int] = reactive(0)
    child_running: reactive[int] = reactive(0)
    child_failed: reactive[int] = reactive(0)

    def render(self) -> str:
        # NOTE (C1.e MED A-3): returns a markup string; relies on Static
        # `markup=True` default for [bold]/[cyan]/etc. to render. If a
        # future Textual changes the default, switch to returning
        # `Text.from_markup(...)` instead.
        left = (
            f"model: {self.model}  |  "
            f"persona: {self.persona}  |  "
            f"mode: {self.mode}  |  "
            f"modules: {self.modules_count}"
        )
        # Plan v11 §2.4.3 — three-state counter
        if self.child_running == 0 and self.child_failed == 0:
            child = "🤖 idle"
        elif self.child_failed == 0:
            child = f"🤖 {self.child_running} running"
        else:
            child = f"🤖 {self.child_running} running, {self.child_failed} failed"
        # Fixed 18-char right slot (L2 Header-jitter mitigation)
        return f"{left}  {child:>18}"


# ---------------------------------------------------------------------------
# ChatLog — scrollable message column (plan §2.6)
# ---------------------------------------------------------------------------


class ChatLog(VerticalScroll):
    """Scrollable column of message bubbles. Each message is one Static.

    Ring buffer cap (plan v11 §2.6 round-6 H3) — evict oldest message
    when count exceeds MAX_MESSAGES (200). Streaming bubbles use plain-
    text accumulation during stream + final Markdown reflow only at
    finalize_assistant_bubble — avoids the per-chunk reparse O(N²) CPU
    cliff identified in round-6 H1 / Q1.

    ``can_focus = False``: VerticalScroll defaults to focusable (so users
    can scroll with arrow keys when it has focus). For chat UX the Input
    should always own focus; scroll via Page Up/Down / mouse wheel
    instead. Real-terminal C0 Spike Step-3 verified this is necessary —
    without it Input never receives keystrokes. (plan v11 §2.11.5 / Q10)
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
        # Plan §2.6 two-stage streaming: chunks accumulate as plain
        # text in _accum during stream, then finalize_assistant_bubble
        # rerenders once as Markdown. Avoids per-chunk reparse O(N²)
        # CPU cliff (round-6 H1 / Q1).
        self._accum: list[str] = []
        # call_id -> (Static widget, tool name) — populated on ToolStartMessage,
        # popped on ToolEndMessage / ToolErrorMessage. Plan §2.5 spec: TUI
        # owns call_id because framework hook doesn't provide one.
        self._pending_tool_cards: dict[str, tuple[Static, str]] = {}

    # ----- streaming primitives (used by both spike mock and Message handlers in C1.b) -----

    def append_user(self, text: str) -> None:
        """Mount a user-turn bubble. Resets streaming target.

        ``text`` is escaped via rich.markup.escape so any ``[bold]``-style
        markup chars the user types render as literal text instead of
        Rich styling (plan §12 / round-7 NIT A-1).
        """
        self._current_assistant = None
        self._accum.clear()
        self._mount_capped(Static(f"[bold]You:[/bold] {markup_escape(text)}"))

    def append_assistant_chunk(self, text: str) -> None:
        """Streaming: accumulate chunk as plain text into the current
        assistant bubble (plan §2.6 stage 1).

        - ``self._accum`` stores RAW chunks (for the Markdown reflow at
          finalize — so ``**bold**`` / `` `code` `` still parse).
        - The DISPLAYED body during streaming is markup-escaped so any
          Rich-style ``[bold]`` literal the LLM emits doesn't accidentally
          re-style our prefix or the rest of the bubble (round-7 NIT A-1).
        """
        self._accum.append(text)
        raw_body = "".join(self._accum)
        display_body = markup_escape(raw_body)
        if self._current_assistant is None:
            self._current_assistant = Static(
                f"[bold cyan]Assistant:[/bold cyan] {display_body}"
            )
            self._mount_capped(self._current_assistant)
        else:
            self._current_assistant.update(
                f"[bold cyan]Assistant:[/bold cyan] {display_body}"
            )

    def finalize_assistant_bubble(self) -> None:
        """End of streaming: one-shot Markdown reflow (plan §2.6 stage 2).

        Renders the accumulated body once as Rich Markdown so **bold**,
        `code`, lists, etc. appear properly. Combines a Text prefix
        (``Assistant:``) with the Markdown body in a single Group so the
        Static still shows the speaker tag. Clears _accum afterwards so
        the next turn starts fresh.
        """
        if self._current_assistant is not None and self._accum:
            body = "".join(self._accum)
            self._current_assistant.update(
                Group(
                    Text.from_markup("[bold cyan]Assistant:[/bold cyan]"),
                    Markdown(body),
                )
            )
        self._current_assistant = None
        self._accum.clear()

    def append_tool_card(self, name: str, status: str = "running") -> None:
        """Append a tool-call card (collapsible to be wired in C1.b)."""
        marker = {
            "running": "[yellow]⏳[/]",
            "success": "[green]✓[/]",
            "error": "[red]✗[/]",
        }.get(status, "")
        self._mount_capped(Static(f"{marker} tool: [cyan]{name}[/cyan]"))

    def append_status(self, message: str) -> None:
        self._mount_capped(Static(f"[dim]{message}[/dim]"))

    def append_error(self, message: str) -> None:
        self._mount_capped(Static(f"[bold red]Error:[/bold red] {message}"))

    # ----- ring buffer eviction -----

    def _mount_capped(self, widget: Static) -> None:
        self.mount(widget)
        if len(self.children) > self.MAX_MESSAGES:
            evicted = self.children[0]
            evicted.remove()
            # Guard against dangling streaming-target reference
            # (plan v11 round-6 L-R3-2). Also clear _accum so the next
            # chunk doesn't extend a phantom body inherited from the
            # evicted bubble.
            if self._current_assistant is evicted:
                self._current_assistant = None
                self._accum.clear()
            # Same guard for tool card map — evict orphans whose widget
            # is the one being removed.
            for cid, (w, _name) in list(self._pending_tool_cards.items()):
                if w is evicted:
                    del self._pending_tool_cards[cid]

    # ----- Message handlers (C1.b — worker thread → app.post_message → here) -----

    def on_chat_chunk_message(self, message: ChatChunkMessage) -> None:
        self.append_assistant_chunk(message.text)

    def on_tool_start_message(self, message: ToolStartMessage) -> None:
        # Tool call interrupts the current assistant stream — finalize
        # the bubble (which now also triggers the Markdown reflow per
        # plan §2.6 stage 2) so the pre-tool assistant text is rendered
        # rich-text before the tool card mounts. Next ChatChunkMessage
        # will open a fresh bubble after the tool card.
        self.finalize_assistant_bubble()
        card = Static(
            f"[yellow]⏳[/] tool: [cyan]{message.name}[/cyan]  [dim](running…)[/dim]"
        )
        self._pending_tool_cards[message.call_id] = (card, message.name)
        self._mount_capped(card)

    def on_tool_end_message(self, message: ToolEndMessage) -> None:
        entry = self._pending_tool_cards.pop(message.call_id, None)
        if entry is None:
            # Orphan POST — plan v11 §2.5 defensive (PRE never fired or
            # was evicted by ring buffer). Render a standalone card so
            # the user sees the completion at least; include call_id so
            # the operator can trace which call was orphaned (round-7
            # MED A-1).
            self._mount_capped(
                Static(
                    f"[green]✓[/] tool: [dim](unmatched, id={message.call_id})[/dim]"
                    f"  {message.duration_ms:.0f}ms"
                )
            )
            return
        card, name = entry
        preview = (message.result_preview or "").rstrip()
        if len(preview) > 80:
            preview = preview[:79] + "…"
        suffix = f"\n  → {preview}" if preview else ""
        card.update(
            f"[green]✓[/] tool: [cyan]{name}[/cyan]  [dim]{message.duration_ms:.0f}ms[/dim]{suffix}"
        )

    def on_tool_error_message(self, message: ToolErrorMessage) -> None:
        entry = self._pending_tool_cards.pop(message.call_id, None)
        name = entry[1] if entry else message.name
        text = (
            f"[red]✗[/] tool: [cyan]{name}[/cyan]  "
            f"[red]error:[/red] {message.error}"
        )
        if entry is not None:
            entry[0].update(text)
        else:
            self._mount_capped(Static(text))

    def on_status_message(self, message: StatusMessage) -> None:
        self.append_status(message.message)

    def on_error_message(self, message: ErrorMessage) -> None:
        self.append_error(message.message)

    def on_turn_complete_message(self, message: TurnCompleteMessage) -> None:
        """C2 — turn ended. Finalize the assistant bubble (triggers the
        Markdown reflow per plan §2.6 stage 2) and surface any error
        from agent.chat_stream as an error bubble."""
        self.finalize_assistant_bubble()
        if not message.success and message.error:
            self.append_error(f"agent error: {message.error}")
