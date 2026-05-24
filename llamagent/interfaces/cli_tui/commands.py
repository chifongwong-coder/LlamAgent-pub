"""Slash command dispatch for the LlamAgent TUI (plan §4 C6).

Entry: ``dispatch_slash(app, user_input) -> bool``.

Returns True when the input was a recognized slash command (handler ran or
unknown-command error was rendered). Returns False when the input does not
start with ``/`` so the caller can route it to chat.

Handler signature: ``(app, arg: str) -> None``. Each handler renders its
output through ``app.query_one("#chat-log", ChatLog)`` using the existing
append_* / Static.update primitives — no direct ``console.print``. The
function never raises on unknown commands; the dispatcher renders the
error itself.

The 13 legacy commands mirror ``cli.py::LlamAgentCLI._cmd_*`` behaviour
(11 handlers + /quit/exit/q sharing one); /mode is dispatched separately
because it takes an argument. /tools is new in v16 and lists
``agent._tools`` grouped by tier.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from rich.markup import escape as markup_escape
from textual.widgets import Static

if TYPE_CHECKING:
    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    from llamagent.interfaces.cli_tui.widgets import ChatLog


# ---------------------------------------------------------------------------
# Rendering helpers — every handler writes through ChatLog, never stdout
# ---------------------------------------------------------------------------


def _log(app: "LlamAgentTUI") -> "ChatLog":
    from llamagent.interfaces.cli_tui.widgets import ChatLog
    return app.query_one("#chat-log", ChatLog)


def _info(app: "LlamAgentTUI", line: str) -> None:
    _log(app).append_status(line)


def _error(app: "LlamAgentTUI", line: str) -> None:
    _log(app).append_error(line)


def _mount_block(app: "LlamAgentTUI", lines: list[str]) -> None:
    """Mount a multi-line block as a single Static so it stays grouped."""
    body = "\n".join(lines)
    log = _log(app)
    log._mount_capped(Static(body))  # uses the shared ring-buffer eviction


# ---------------------------------------------------------------------------
# Individual handlers (legacy parity)
# ---------------------------------------------------------------------------


_MODES = ("interactive", "task", "continuous")


def cmd_quit(app: "LlamAgentTUI", arg: str) -> None:
    app.exit()


def cmd_help(app: "LlamAgentTUI", arg: str) -> None:
    rows = [
        ("/help", "Show this help message"),
        ("/mode [name]", "Show / switch mode (interactive, task, continuous)"),
        ("/abort", "Cancel the current task (Ctrl+G)"),
        ("/stop", "Stop the background runner (continuous mode)"),
        ("/status", "View agent runtime status"),
        ("/modules", "View loaded modules"),
        ("/tools", "List tools registered on the agent"),
        ("/skills", "List loaded skills"),
        ("/prompts", "List customizable system prompt slots"),
        ("/memory", "Show stored facts and memory usage"),
        ("/history", "Browse saved conversation sessions"),
        ("/clear", "Start a fresh conversation (Ctrl+L)"),
        ("/verbose [on|off]", "Toggle the right diagnostic pane (thinking / tool detail)"),
        ("/monitor [on|off]", "Toggle the right monitor pane (continuous mode status)"),
        ("/quit", "Exit the TUI (also: /exit, /q)"),
        ("Ctrl+C / Esc", "Quit"),
    ]
    lines = ["[bold cyan]Available Commands[/bold cyan]"]
    for cmd, desc in rows:
        lines.append(f"  [cyan]{cmd:<16}[/cyan] {markup_escape(desc)}")
    # Round-18 C-8 — in-session pointers for the two things users most
    # often need when something goes wrong but can't grep source for.
    lines.append("")
    lines.append("[bold cyan]Troubleshooting[/bold cyan]")
    lines.append("  TUI crash → traceback at [cyan]~/.llamagent/cli_tui.log[/cyan]")
    lines.append("  Suppress noisy tools → YAML [cyan]tools.disabled: [name1, ...][/cyan]")
    _mount_block(app, lines)


def cmd_status(app: "LlamAgentTUI", arg: str) -> None:
    agent = app.agent
    if agent is None:
        _info(app, "Agent not initialised yet")
        return
    try:
        status = agent.status()
    except Exception as e:
        _error(app, f"status() failed: {type(e).__name__}: {e}")
        return
    persona = status.get("persona") or "Default"
    modules = ", ".join(status.get("modules", {}).keys()) or "None"
    lines = [
        "[bold cyan]Agent Status[/bold cyan]",
        f"  [cyan]Model:[/cyan]   {markup_escape(str(status.get('model', '?')))}",
        f"  [cyan]Mode:[/cyan]    {markup_escape(agent.mode)}",
        f"  [cyan]Persona:[/cyan] {markup_escape(str(persona))}",
        f"  [cyan]Modules:[/cyan] {markup_escape(modules)}",
        f"  [cyan]Turns:[/cyan]   {status.get('conversation_turns', 0)}",
    ]
    _mount_block(app, lines)


def cmd_modules(app: "LlamAgentTUI", arg: str) -> None:
    agent = app.agent
    if agent is None or not getattr(agent, "modules", None):
        _info(app, "No modules currently loaded (pure chat mode)")
        return
    lines = ["[bold cyan]Loaded Modules[/bold cyan]"]
    for i, (name, mod) in enumerate(agent.modules.items(), 1):
        desc = (getattr(mod, "description", "") or "").strip()
        lines.append(f"  [dim]{i:>2}.[/dim] [cyan]{markup_escape(name)}[/cyan]  {markup_escape(desc)}")
    lines.append(f"[dim]{len(agent.modules)} module(s) loaded[/dim]")
    _mount_block(app, lines)


def cmd_tools(app: "LlamAgentTUI", arg: str) -> None:
    """List the agent's tool registry, grouped by tier (v16 new)."""
    agent = app.agent
    tools = getattr(agent, "_tools", None) if agent is not None else None
    if not tools:
        _info(app, "No tools registered")
        return
    # Bucket by tier; preserve registration order within each tier.
    by_tier: dict[str, list[dict]] = {}
    for name, meta in tools.items():
        tier = str(meta.get("tier") or "common")
        by_tier.setdefault(tier, []).append({"name": name, "meta": meta})
    lines = [f"[bold cyan]Registered Tools ({len(tools)})[/bold cyan]"]
    for tier in ("default", "common", "admin", "agent"):
        bucket = by_tier.get(tier)
        if not bucket:
            continue
        lines.append(f"[bold]{tier}[/bold] ({len(bucket)})")
        for entry in bucket:
            name = entry["name"]
            meta = entry["meta"]
            safety = meta.get("safety_level", "?")
            desc = (meta.get("description") or "").strip()
            if len(desc) > 70:
                desc = desc[:69] + "…"
            lines.append(
                f"  [cyan]{markup_escape(name):<28}[/cyan] "
                f"[dim]L{safety}[/dim]  {markup_escape(desc)}"
            )
    # Any leftover tier names (e.g. custom) not in the canonical 4
    extras = [t for t in by_tier if t not in ("default", "common", "admin", "agent")]
    for tier in extras:
        bucket = by_tier[tier]
        lines.append(f"[bold]{markup_escape(tier)}[/bold] ({len(bucket)})")
        for entry in bucket:
            name = entry["name"]
            meta = entry["meta"]
            safety = meta.get("safety_level", "?")
            desc = (meta.get("description") or "").strip()
            if len(desc) > 70:
                desc = desc[:69] + "…"
            lines.append(
                f"  [cyan]{markup_escape(name):<28}[/cyan] "
                f"[dim]L{safety}[/dim]  {markup_escape(desc)}"
            )
    _mount_block(app, lines)


def cmd_clear(app: "LlamAgentTUI", arg: str) -> None:
    """Clear the conversation. Round-14 Rev A M1: clear the agent's
    history first; only reset the visible widget state if the agent
    clear succeeded. Reversed ordering (widget first) left the UI
    looking empty while the agent's internal context was still alive,
    causing the next turn to mysteriously see "deleted" messages."""
    log = _log(app)
    if app.agent is not None:
        try:
            app.agent.clear_conversation()
        except Exception as e:
            _error(app, f"clear_conversation() failed: {type(e).__name__}: {e}")
            return
    # Round-18 A-9: ChatLog owns its own private state; route through
    # the public clear() so future ChatLog refactors don't desync with
    # us.
    log.clear()
    _info(app, "Conversation cleared")


def cmd_abort(app: "LlamAgentTUI", arg: str) -> None:
    if app.agent is None:
        _info(app, "No active agent to abort")
        return
    try:
        app.agent.abort()
    except Exception as e:
        _error(app, f"abort() failed: {type(e).__name__}: {e}")
        return
    _info(app, "Abort requested — current task will stop shortly")


# ---------------------------------------------------------------------------
# Right-pane mutex helper (plan §4 C7 H4 fix)
# ---------------------------------------------------------------------------


# Mapping of right-pane key → (widget id, hide mapping). When set_right_pane
# is asked to show a pane, every entry whose key differs from the target gets
# its widget hidden. Adding a third pane is a one-line addition here, no
# touching of every cmd_* call site (P7 single-source-of-truth fix vs the
# round-1 H4 finding's earlier `(app, pane_id, hide_ids)` signature that
# leaked hide knowledge to callers).
_RIGHT_PANES = {
    "verbose": "#verbose-pane",
    "monitor": "#monitor-pane",
}


def _set_right_pane(app: "LlamAgentTUI", target: str | None) -> bool:
    """Show one right pane (and hide the others), or hide all.

    ``target`` is one of ``"verbose"``, ``"monitor"``, or ``None`` (hide
    all). Returns True if the target pane is now visible after the call,
    False if all panes are hidden (or the target lookup failed silently —
    caller can still report it).
    """
    visible = False
    for key, selector in _RIGHT_PANES.items():
        try:
            widget = app.query_one(selector)
        except Exception:
            continue  # pane not mounted yet — skip
        if key == target:
            try:
                widget.display = True
                visible = True
            except Exception:
                pass
        else:
            try:
                widget.display = False
            except Exception:
                pass
    return visible


def _parse_pane_arg(arg: str, *, current: bool) -> "bool | None":
    """Parse a `/verbose` / `/monitor` argument into a desired display state.

    Returns True / False for explicit on/off, the *toggled* current state
    for empty arg, or ``None`` when ``arg`` is unrecognised — caller renders
    the error so the message can name the offending command.
    """
    token = arg.strip().lower()
    if token in ("on", "show", "open", "true", "1"):
        return True
    if token in ("off", "hide", "close", "false", "0"):
        return False
    if token == "":
        return not current
    return None


def cmd_verbose(app: "LlamAgentTUI", arg: str) -> None:
    """Toggle / explicitly set the VerbosePane (round-14 user-test result).

    Replaces the original Ctrl+V keyboard binding from plan §2.3, which
    silently exited the App on macOS Terminal.app (termios VLNEXT collision).
    F3 was the obvious fallback but macOS Mission Control's Window Overview
    intercepts it. A slash command travels through the Input widget as plain
    text, so no terminal-level mapping can intercept it.

    Round-1 H4 refactor (plan §4 C7): the show/hide path goes through
    :func:`_set_right_pane` so it stays consistent with /monitor (the two
    panes occupy the same layout slot and are mutually exclusive).

    Usage:
        /verbose        — toggle (mirrors the old Ctrl+V)
        /verbose on     — force show (auto-hides MonitorPane)
        /verbose off    — force hide
    """
    from llamagent.interfaces.cli_tui.widgets import VerbosePane

    try:
        pane = app.query_one("#verbose-pane", VerbosePane)
    except Exception as e:
        _error(app, f"VerbosePane unavailable: {type(e).__name__}: {e}")
        return

    new_state = _parse_pane_arg(arg, current=bool(pane.display))
    if new_state is None:
        _error(app, f"Invalid /verbose arg: {markup_escape(arg.strip())}. Use: on, off, or no arg (toggle)")
        return
    try:
        _set_right_pane(app, "verbose" if new_state else None)
    except Exception as e:
        _error(app, f"toggle failed: {type(e).__name__}: {e}")
        return
    _info(app, f"VerbosePane {'shown' if new_state else 'hidden'}")


def cmd_monitor(app: "LlamAgentTUI", arg: str) -> None:
    """Toggle / explicitly set the MonitorPane (plan §4 C7).

    Mutually exclusive with VerbosePane via :func:`_set_right_pane` — showing
    one auto-hides the other. Both panes stay mounted (display:none only),
    so the hidden one keeps accumulating ThinkingMessage / status-poll state
    and the user can switch back and see backfill.

    The MonitorPane is meaningful only while a ContinuousRunner is active,
    but the pane itself handles ``runner is None`` gracefully — opening it
    in interactive mode just shows "Continuous mode not active".

    Usage:
        /monitor        — toggle
        /monitor on     — force show (auto-hides VerbosePane)
        /monitor off    — force hide
    """
    from llamagent.interfaces.cli_tui.widgets import MonitorPane

    try:
        pane = app.query_one("#monitor-pane", MonitorPane)
    except Exception as e:
        _error(app, f"MonitorPane unavailable: {type(e).__name__}: {e}")
        return

    new_state = _parse_pane_arg(arg, current=bool(pane.display))
    if new_state is None:
        _error(app, f"Invalid /monitor arg: {markup_escape(arg.strip())}. Use: on, off, or no arg (toggle)")
        return
    try:
        _set_right_pane(app, "monitor" if new_state else None)
    except Exception as e:
        _error(app, f"toggle failed: {type(e).__name__}: {e}")
        return
    _info(app, f"MonitorPane {'shown' if new_state else 'hidden'}")


def cmd_stop(app: "LlamAgentTUI", arg: str) -> None:
    """Stop the continuous runner (plan §4 C7 (e)).

    Replaces the C6 placeholder that only flipped mode. Real path:
    1. Signal the runner to stop (releases waiting inject callers).
    2. ``thread.join(timeout=10)`` — 10s covers most agent.chat turns;
       any runner thread still alive past the timeout gets a warning
       logged and the reference dropped. ``daemon=True`` at thread
       creation means the process-exit path will reap stragglers, so
       we don't try to force-kill (Python can't safely kill threads).
    3. Drop ``self._runner`` / ``self._runner_thread`` so the next
       /mode continuous starts from a clean slate.
    4. Flip agent.mode back to interactive + refresh StatusHeader.
    5. Hide MonitorPane so the now-stale state board doesn't linger.
    """
    agent = app.agent
    runner = getattr(app, "_runner", None)
    if agent is None or agent.mode != "continuous" or runner is None:
        _info(app, "Not in continuous mode")
        return

    try:
        runner.stop()
    except Exception as e:
        _error(app, f"runner.stop() failed: {type(e).__name__}: {e}")
        # Keep going — we still want to clear state so the user isn't
        # stuck in a half-stopped mode.

    thread = getattr(app, "_runner_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=10.0)
        if thread.is_alive():
            import logging
            logging.getLogger(__name__).warning(
                "continuous runner thread did not stop within 10s; "
                "dropping reference and relying on daemon-thread reap"
            )

    app._runner = None
    app._runner_thread = None

    try:
        agent.set_mode("interactive")
    except Exception as e:
        _error(app, f"set_mode(interactive) failed: {type(e).__name__}: {e}")
        return
    app.refresh_status_header()
    _set_right_pane(app, None)  # hide MonitorPane
    _info(app, "Continuous runner stopped — switched back to interactive")


def cmd_mode(app: "LlamAgentTUI", arg: str) -> None:
    agent = app.agent
    if agent is None:
        _info(app, "No active agent")
        return
    target = arg.strip().lower()
    if not target:
        lines = [
            f"[bold cyan]Current mode:[/bold cyan] {markup_escape(agent.mode)}",
            "[dim]Available:[/dim]",
            "  [cyan]/mode interactive[/cyan]  Normal back-and-forth chat",
            "  [cyan]/mode task[/cyan]         Plan first, confirm, then execute",
            "  [cyan]/mode continuous[/cyan]   Automated triggers (C7 will wire the trigger UI)",
        ]
        _mount_block(app, lines)
        return
    if target not in _MODES:
        _error(app, f"Invalid mode: {target}. Use: interactive, task, continuous")
        return
    if target == agent.mode:
        _info(app, f"Already in {target} mode")
        return

    # C7 — continuous mode goes through a setup modal. The modal is async
    # / callback-based (push_screen pattern, plan §4 C7 (d) round-1 B1):
    # cmd_mode returns immediately, and the real mode switch happens in
    # ``app._on_continuous_setup_done`` once the user fills the form. If
    # the user cancels, no state changes.
    if target == "continuous":
        # Round-15 Rev A M2 — pull the callback access OUT of the try so
        # a missing attribute on a future App subclass / smoke FakeApp
        # surfaces as a clean AttributeError instead of being silently
        # rewrapped into "could not open continuous setup".
        callback = app._on_continuous_setup_done
        from llamagent.interfaces.cli_tui.screens import ContinuousSetupModal
        try:
            app.push_screen(ContinuousSetupModal(), callback)
        except Exception as e:
            _error(app, f"could not open continuous setup: {type(e).__name__}: {e}")
        return

    try:
        agent.set_mode(target)
    except Exception as e:
        _error(app, f"set_mode({target}) failed: {type(e).__name__}: {e}")
        return
    app.refresh_status_header()
    _info(app, f"Switched to {target} mode")


def cmd_skills(app: "LlamAgentTUI", arg: str) -> None:
    agent = app.agent
    if agent is None or not agent.has_module("skill"):
        _info(app, "Skill module is not loaded")
        return
    mod = agent.get_module("skill")
    skills = []
    try:
        if mod is not None and hasattr(mod, "list_skills"):
            skills = mod.list_skills() or []
    except Exception as e:
        _error(app, f"list_skills() failed: {type(e).__name__}: {e}")
        return
    if not skills:
        _info(app, "No skills available")
        return
    lines = [f"[bold cyan]Available Skills ({len(skills)})[/bold cyan]"]
    for i, s in enumerate(skills, 1):
        if isinstance(s, dict):
            name = str(s.get("name", "unnamed"))
            desc = str(s.get("description", "") or "").strip()
        else:
            name = str(s)
            desc = ""
        if len(desc) > 70:
            desc = desc[:69] + "…"
        lines.append(f"  [dim]{i:>2}.[/dim] [cyan]{markup_escape(name)}[/cyan]  {markup_escape(desc)}")
    _mount_block(app, lines)


def cmd_memory(app: "LlamAgentTUI", arg: str) -> None:
    agent = app.agent
    if agent is None or not agent.has_module("memory"):
        _info(app, "Memory module is not loaded")
        return
    mod = agent.get_module("memory")
    if mod is None:
        _info(app, "Memory module is not available")
        return
    store = getattr(mod, "store", None)
    count = None
    if store is not None and hasattr(store, "count"):
        try:
            count = store.count()
        except Exception:
            count = None
    mode = getattr(agent.config, "memory_mode", "unknown")
    if count is not None:
        _mount_block(app, [
            "[bold cyan]Memory[/bold cyan]",
            f"  Stored facts: {count}",
            f"  Mode: {markup_escape(str(mode))}",
        ])
    else:
        _mount_block(app, [
            "[bold cyan]Memory[/bold cyan]",
            f"  Mode: {markup_escape(str(mode))}",
        ])


def cmd_history(app: "LlamAgentTUI", arg: str) -> None:
    """Browse saved sessions. C6 first-cut: list-only (no resume / delete
    UI yet — that needs an interactive modal which lands in a follow-up)."""
    agent = app.agent
    if agent is None:
        _info(app, "No active agent")
        return
    try:
        from llamagent.interfaces.sessions import list_sessions, format_time_ago
    except Exception as e:
        _error(app, f"sessions module import failed: {type(e).__name__}: {e}")
        return
    try:
        sessions = list_sessions(agent) or []
    except Exception as e:
        _error(app, f"list_sessions() failed: {type(e).__name__}: {e}")
        return
    if not sessions:
        _info(app, "No saved sessions found. Enable persistence module to save sessions.")
        return
    lines = [f"[bold cyan]Saved Sessions ({len(sessions)})[/bold cyan]"]
    for i, s in enumerate(sessions[:20], 1):
        sid = str(s.get("id", "?"))
        turns = s.get("turn_count", s.get("turns", "?"))
        updated = s.get("updated_at") or s.get("timestamp")
        ago = ""
        if updated:
            try:
                ago = format_time_ago(updated)
            except Exception:
                ago = str(updated)
        lines.append(f"  [dim]{i:>2}.[/dim] [cyan]{markup_escape(sid)}[/cyan]  turns={turns}  {markup_escape(ago)}")
    if len(sessions) > 20:
        lines.append(f"[dim](showing 20 of {len(sessions)})[/dim]")
    _mount_block(app, lines)


def cmd_prompts(app: "LlamAgentTUI", arg: str) -> None:
    """List customizable system prompt slots (v3.9.0)."""
    agent = app.agent
    if agent is None or not hasattr(agent, "list_prompt_slots"):
        _info(app, "Prompt slot inspection unavailable")
        return
    try:
        view = agent.list_prompt_slots()
    except Exception as e:
        _error(app, f"list_prompt_slots() failed: {type(e).__name__}: {e}")
        return
    agent_slots = view.get("_agent", {}) if isinstance(view, dict) else {}
    agent_overrides = getattr(agent.config, "agent_prompts", None) or {}
    module_prompts_cfg = getattr(agent.config, "module_prompts", None) or {}

    def status_for(slot, override_value):
        if getattr(slot, "locked", False):
            return "[locked]"
        return "overridden" if override_value is not None else "default"

    def truncate(text, limit=58):
        text = (text or "").replace("\n", " ").strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    lines = ["[bold cyan]Agent prompts[/bold cyan]"]
    for slot_name, slot in sorted(agent_slots.items()):
        override = agent_overrides.get(slot_name)
        lines.append(
            f"  [cyan]{markup_escape(slot_name):<24}[/cyan] "
            f"[dim]{markup_escape(status_for(slot, override)):<11}[/dim] "
            f"{markup_escape(truncate(slot.description))}"
        )
    module_targets = [k for k in view.keys() if k != "_agent"]
    any_module_slots = any(view.get(m) for m in module_targets)
    if not any_module_slots:
        lines.append("[dim]Module slots: not yet customizable[/dim]")
    else:
        lines.append("")
        lines.append("[bold cyan]Module prompts[/bold cyan]")
        for mod_name in sorted(module_targets):
            slots = view.get(mod_name, {}) or {}
            mod_overrides = module_prompts_cfg.get(mod_name, {})
            for slot_name, slot in sorted(slots.items()):
                override = mod_overrides.get(slot_name)
                lines.append(
                    f"  [cyan]{markup_escape(mod_name)}/{markup_escape(slot_name):<20}[/cyan] "
                    f"[dim]{markup_escape(status_for(slot, override)):<11}[/dim] "
                    f"{markup_escape(truncate(slot.description))}"
                )
    lines.append("[dim]Hint: use agent.list_prompt_slots() in Python to inspect default content.[/dim]")
    _mount_block(app, lines)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


# Static handler map: 11 legacy handlers + new /tools. /mode and /quit-family
# use the same handler with aliasing semantics.
_HANDLERS: dict[str, Callable] = {
    "/help": cmd_help,
    "/status": cmd_status,
    "/modules": cmd_modules,
    "/tools": cmd_tools,
    "/skills": cmd_skills,
    "/memory": cmd_memory,
    "/history": cmd_history,
    "/prompts": cmd_prompts,
    "/clear": cmd_clear,
    "/abort": cmd_abort,
    "/stop": cmd_stop,
    "/verbose": cmd_verbose,
    "/monitor": cmd_monitor,
    "/quit": cmd_quit,
    "/exit": cmd_quit,
    "/q": cmd_quit,
}


def dispatch_slash(app: "LlamAgentTUI", user_input: str) -> bool:
    """Route ``user_input`` to a slash handler. Returns True if the input
    was a slash command (handler ran or unknown-command error was emitted),
    False if not a slash command (caller routes to chat)."""
    if not user_input.startswith("/"):
        return False
    parts = user_input.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    if cmd == "/mode":
        cmd_mode(app, arg)
        return True
    handler = _HANDLERS.get(cmd)
    if handler is None:
        _error(app, f"Unknown command: {markup_escape(cmd)} — type /help for available commands")
        return True
    try:
        handler(app, arg)
    except Exception as e:
        # Defensive: a handler raising should not kill the input loop.
        _error(app, f"{markup_escape(cmd)} failed: {type(e).__name__}: {e}")
    return True
