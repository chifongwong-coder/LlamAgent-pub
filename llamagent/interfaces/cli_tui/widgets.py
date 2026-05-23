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
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.suggester import SuggestFromList
from textual.widgets import Input, Static

from llamagent.interfaces.cli_tui.messages import (
    ChatChunkMessage,
    ErrorMessage,
    StatusMessage,
    ThinkingMessage,
    ToolEndMessage,
    ToolErrorMessage,
    ToolStartMessage,
    TurnCompleteMessage,
)


# ---------------------------------------------------------------------------
# Slash command list + Suggester (plan §4 C1 / §4 C6)
# ---------------------------------------------------------------------------


# Plan §4 C6 — 13 legacy slash commands (11 handlers + 3 /quit alias) +
# /mode (3 subcommands) + /tools + /verbose (both v16-new, non-legacy).
# Autocomplete surface; actual dispatch lives in commands.py::dispatch_slash.
# /verbose replaces the original Ctrl+V keyboard binding (round-14 user-test:
# Ctrl+V exits Terminal.app via termios VLNEXT; F3 hits macOS Mission
# Control; F-keys 1-12 are mostly multimedia/brightness by default).
SLASH_COMMANDS: tuple[str, ...] = (
    "/help",
    "/modules",
    "/tools",
    "/skills",
    "/memory",
    "/history",
    "/status",
    "/clear",
    "/abort",
    "/stop",
    "/prompts",
    "/verbose",
    "/verbose on",
    "/verbose off",
    "/monitor",
    "/monitor on",
    "/monitor off",
    "/quit",
    "/exit",
    "/q",
    "/mode",
    "/mode interactive",
    "/mode task",
    "/mode continuous",
)


class LlamAgentInput(Input):
    """Input subclass with Tab accept + Up/Down history (plan §2.10 v16 K10).

    Three extra keystrokes on top of the stock Textual ``Input`` BINDINGS:

    - **Tab**: if the Suggester has a current suggestion that extends the
      typed prefix, accept it (replace value with the full suggestion +
      move cursor to end). No suggestion → no-op (Tab does not move focus
      because ChatLog / VerbosePane have ``can_focus = False``).
    - **Up**: navigate to the previous history entry. First press saves
      the current input value as scratch so the user can Down back to it.
    - **Down**: navigate to the next history entry; past the newest, restore
      scratch and exit history mode.

    History semantics mirror GNU readline / bash:
    - Submitted values (slash commands AND chat) are appended via
      :meth:`append_history` (caller wires this from ``on_input_submitted``).
    - Consecutive duplicates are deduplicated (Up Up does not waste a slot
      repeating the last command).
    - Editing a recalled history entry does NOT re-snapshot scratch — Up
      from an edited history entry just walks to the next-older entry,
      losing the in-place edits, same as bash. Submit (Enter) is the only
      way to keep an edited value.
    - Capacity 200 ring buffer (matches ChatLog.MAX_MESSAGES). When full,
      oldest entry is evicted. Not persisted across sessions (V1) —
      plan §2.10 V2 defers persistence to C9 polish.
    """

    MAX_HISTORY = 200

    BINDINGS = [
        Binding("up", "history_prev", "History prev", show=False),
        Binding("down", "history_next", "History next", show=False),
        Binding("tab", "accept_suggestion", "Accept suggestion", show=False),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_cursor: int | None = None
        self._scratch: str = ""

    def append_history(self, value: str) -> None:
        """Record a submitted value (called by App on submit before clear).

        - Empty / whitespace-only values are not recorded.
        - Consecutive duplicates are dropped.
        - Capacity 200: oldest entry is evicted FIFO.
        - Resets cursor / scratch so the next Up starts fresh.
        """
        value = value.strip()
        # Always reset navigation state after a submit, even if we don't
        # actually record (empty / dup) — the previous Up/Down navigation
        # is no longer relevant once the user has submitted something.
        self._history_cursor = None
        self._scratch = ""
        if not value:
            return
        if self._history and self._history[-1] == value:
            return
        self._history.append(value)
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)

    def action_history_prev(self) -> None:
        """Up — move to an older history entry (or enter history mode)."""
        if not self._history:
            return
        if self._history_cursor is None:
            # First Up: snapshot the in-progress edit so Down can restore it
            self._scratch = self.value
            self._history_cursor = len(self._history) - 1
        elif self._history_cursor > 0:
            self._history_cursor -= 1
        else:
            return  # already at the oldest entry
        self.value = self._history[self._history_cursor]
        self.cursor_position = len(self.value)

    def action_history_next(self) -> None:
        """Down — move to a newer history entry, or restore scratch and exit."""
        if self._history_cursor is None:
            return
        if self._history_cursor < len(self._history) - 1:
            self._history_cursor += 1
            self.value = self._history[self._history_cursor]
        else:
            # Past the newest entry: leave history mode, restore the
            # snapshot the user was typing when they first pressed Up.
            self._history_cursor = None
            self.value = self._scratch
            self._scratch = ""
        self.cursor_position = len(self.value)

    def action_accept_suggestion(self) -> None:
        """Tab — accept the current Suggester suggestion if it extends value.

        Textual stores the active suggestion (full text including the
        typed prefix) on ``self._suggestion``. When the suggestion is
        longer than the current value it means the Suggester offered a
        completion the user hasn't accepted yet — replacing value with
        the suggestion is the accept gesture (mirrors what
        ``action_cursor_right`` does at the end of the value at
        textual/widgets/_input.py:688).
        """
        suggestion = self._suggestion or ""
        if suggestion and suggestion != self.value:
            self.value = suggestion
            self.cursor_position = len(self.value)


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
        width: 1fr;
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
        - After every chunk, scroll to bottom so the user follows the
          stream — chat UIs auto-follow by convention.
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
            self.scroll_end(animate=False)

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
        # Auto-follow newest message so user sees streamed output
        # without manual scrolling. animate=False avoids the ease
        # delay that would lag behind fast token streams.
        self.scroll_end(animate=False)
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


# ---------------------------------------------------------------------------
# VerbosePane — diagnostic side panel for thinking / tool detail (plan §4 C5)
# ---------------------------------------------------------------------------


class VerbosePane(VerticalScroll):
    """Right-side diagnostic pane: thinking content + tool args/result.

    Hidden by default (App toggles ``display`` with Ctrl+V per plan §2.3
    / §11 Q10). Subscribes to ThinkingMessage / ToolStart / ToolEnd /
    ToolError so the user sees the same events that flow to ChatLog
    but presented as collapsible diagnostic cards.

    ``can_focus = False`` — same rationale as ChatLog (the Input must
    own focus for chat; pane scrolls via mouse / Page Up/Down).
    """

    can_focus = False

    DEFAULT_CSS = """
    VerbosePane {
        width: 40;
        height: 1fr;
        border: solid $primary-darken-1;
        padding: 0 1;
        display: none;
    }
    """

    MAX_EVENTS = 200

    def on_thinking_message(self, message: ThinkingMessage) -> None:
        """ThinkingMessage from verbose.py LLM-chat wrapper — render as
        a labelled, indented block. Dedup is the caller's responsibility
        (verbose.py applies SHA-256 per-turn before posting)."""
        # Truncate per plan §11 Q1: long thinking dumps degrade
        # perf if rendered as Markdown each turn. Plain text only.
        content = message.content or ""
        if len(content) > 800:
            content = content[:799] + "…"
        self._mount_capped(
            Static(
                f"[dim italic]thinking[/dim italic] [dim]({markup_escape(message.source)})[/dim]\n"
                f"  {markup_escape(content)}"
            )
        )

    def on_tool_start_message(self, message: ToolStartMessage) -> None:
        # Re-render in the verbose pane with the full args dict (the
        # ChatLog card abbreviates to just the tool name).
        args_repr = repr(message.args)
        if len(args_repr) > 400:
            args_repr = args_repr[:399] + "…"
        self._mount_capped(
            Static(
                f"[yellow]⏳ tool:[/yellow] [cyan]{markup_escape(message.name)}[/cyan] "
                f"[dim]({message.call_id})[/dim]\n"
                f"  args: {markup_escape(args_repr)}"
            )
        )

    def on_tool_end_message(self, message: ToolEndMessage) -> None:
        preview = (message.result_preview or "").rstrip()
        if len(preview) > 400:
            preview = preview[:399] + "…"
        self._mount_capped(
            Static(
                f"[green]✓ tool end[/green] [dim]({message.call_id})  "
                f"{message.duration_ms:.0f}ms[/dim]\n"
                f"  result: {markup_escape(preview) if preview else '[dim](empty)[/dim]'}"
            )
        )

    def on_tool_error_message(self, message: ToolErrorMessage) -> None:
        self._mount_capped(
            Static(
                f"[red]✗ tool error[/red] [cyan]{markup_escape(message.name)}[/cyan] "
                f"[dim]({message.call_id})[/dim]\n"
                f"  [red]{markup_escape(message.error)}[/red]"
            )
        )

    def _mount_capped(self, widget: Static) -> None:
        self.mount(widget)
        self.scroll_end(animate=False)
        if len(self.children) > self.MAX_EVENTS:
            self.children[0].remove()


# ---------------------------------------------------------------------------
# MonitorPane — Continuous mode status board (plan §4 C7)
# ---------------------------------------------------------------------------


class MonitorPane(VerticalScroll):
    """Right-side status board for continuous mode (plan §4 C7 V1+).

    Shares its layout slot with VerbosePane — at most one is visible at a
    time, switched via :func:`commands._set_right_pane`. Both stay mounted
    (display:none only) so each retains its history while hidden.

    The pane is a *status board*, not an event stream — it polls a small
    set of fields on the App's active ContinuousRunner every second and
    re-renders the same single Static. plan v16 K-line consciously kept
    the per-fire event stream (V2) out of scope:
    surfacing every fire requires either a framework callback in Runner
    or a monkey-patch on Runner.run — both larger commitments. V1+ does
    show ``fired Xs ago`` per ``TimerTrigger`` because that instance
    already stores ``_last_fire`` (`runner.py:309`), so the math is local.

    P5 local violation note: reads several ``_``-prefixed fields off
    ContinuousRunner and TimerTrigger. plan §4 C7 (b) registers this as
    deferred to v3.10+ — Runner should expose ``get_status() -> dict``
    and Trigger should expose ``last_fired_at``. Until then we keep all
    private-attr reads inside :func:`_runner_snapshot` so a future
    framework refactor only needs to touch one place.

    ``can_focus = False`` — same rationale as ChatLog / VerbosePane: the
    Input must own focus for chat; pane scrolls via mouse / Page Up/Down.
    """

    can_focus = False

    DEFAULT_CSS = """
    MonitorPane {
        width: 40;
        height: 1fr;
        border: solid $primary-darken-1;
        padding: 0 1;
        display: none;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._body: Static | None = None

    def on_mount(self) -> None:
        # Single Static that we re-render each tick; cheaper than mounting
        # and unmounting children, and the status board doesn't need
        # incremental update semantics.
        self._body = Static(self._render_idle())
        self.mount(self._body)
        # 1-second poll cadence matches plan §4 C7 (b). Cheap field reads
        # on a single thread; no Worker / async needed.
        self.set_interval(1.0, self._refresh)

    def _refresh(self) -> None:
        if self._body is None:
            return
        # ``self.app`` may not be ready during very early mount or after
        # quit; guard defensively. ``self.app.agent`` and ``self.app._runner``
        # are App-level fields, missing on smoke / pilot stubs.
        try:
            runner = getattr(self.app, "_runner", None)
        except Exception:
            runner = None
        try:
            self._body.update(self._render_idle() if runner is None else self._render_running(runner))
        except Exception:
            # Never let a redraw exception kill the pane — log only if needed.
            pass

    @staticmethod
    def _render_idle() -> str:
        return (
            "[bold]Continuous mode not active[/bold]\n\n"
            "[dim]Use /mode continuous to start a runner.[/dim]"
        )

    @staticmethod
    def _render_running(runner) -> str:
        snap = _runner_snapshot(runner)
        lines = ["[bold cyan]Continuous Runner[/bold cyan]"]
        lines.append(
            f"  state:    [bold]{'running' if snap['running'] else 'stopped'}[/bold]"
            f"  task: [{('yes' if snap['task_running'] else 'idle')}]"
        )
        q = snap["queue"]
        lines.append(
            f"  queue:    urgent={q['urgent']}  normal={q['normal']}  retry={q['retry']}"
        )
        if snap["triggers"]:
            lines.append("")
            lines.append(f"[bold cyan]Triggers ({len(snap['triggers'])})[/bold cyan]")
            for trig in snap["triggers"]:
                head = f"  [cyan]{trig['kind']}[/cyan]  {trig['summary']}"
                if trig["last_fire_ago"] is not None:
                    head += f"  [dim](fired {trig['last_fire_ago']:.0f}s ago)[/dim]"
                lines.append(head)
        else:
            lines.append("")
            lines.append("[dim]No triggers configured[/dim]")
        return "\n".join(lines)


def _runner_snapshot(runner) -> dict:
    """Read a small status dict off a ContinuousRunner, isolating every
    private-attr read in one place (plan §4 C7 H3 P5-note).

    Returned shape::

        {
            "running":      bool,
            "task_running": bool,
            "queue": {"urgent": int, "normal": int, "retry": int},
            "triggers": [
                {"kind": "Timer", "summary": "60s -- 'check'", "last_fire_ago": 12.3},
                {"kind": "File",  "summary": "/tmp watch",      "last_fire_ago": None},
                ...
            ],
        }

    Missing / changed fields degrade gracefully (best-effort getattr).
    """
    import time

    triggers_info: list[dict] = []
    for t in getattr(runner, "triggers", []) or []:
        kind = type(t).__name__.replace("Trigger", "")
        if kind == "Timer":
            interval = getattr(t, "interval", "?")
            message = getattr(t, "message", "")
            summary = f"every {interval}s -- {message!r}"
            lf = getattr(t, "_last_fire", None)
            ago: float | None = (time.time() - lf) if lf is not None else None
        elif kind == "File":
            watch = getattr(t, "watch_dir", "?")
            summary = f"watch {watch}"
            ago = None
        else:
            summary = repr(t)
            ago = None
        triggers_info.append({"kind": kind, "summary": summary, "last_fire_ago": ago})

    def _qsize(name: str) -> int:
        q = getattr(runner, name, None)
        try:
            return q.qsize() if q is not None else 0
        except Exception:
            return 0

    return {
        "running":      bool(getattr(runner, "_running", False)),
        "task_running": bool(getattr(runner, "_task_running", False)),
        "queue": {
            "urgent": _qsize("_urgent_queue"),
            "normal": _qsize("_normal_queue"),
            "retry":  _qsize("_retry_queue"),
        },
        "triggers": triggers_info,
    }
