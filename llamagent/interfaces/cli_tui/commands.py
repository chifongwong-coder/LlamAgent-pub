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
        ("/stop", "Stop the background runner (F2)"),
        ("/status", "View agent runtime status"),
        ("/modules", "View loaded modules"),
        ("/tools", "List tools registered on the agent"),
        ("/skills", "List loaded skills"),
        ("/prompts", "List customizable system prompt slots"),
        ("/memory", "Show stored facts and memory usage"),
        ("/history", "Browse saved conversation sessions"),
        ("/clear", "Start a fresh conversation (Ctrl+L)"),
        ("/verbose [on|off]", "Toggle the right diagnostic pane"),
        ("/quit", "Exit the TUI (also: /exit, /q)"),
        ("Ctrl+C / Esc", "Quit"),
    ]
    lines = ["[bold cyan]Available Commands[/bold cyan]"]
    for cmd, desc in rows:
        lines.append(f"  [cyan]{cmd:<16}[/cyan] {markup_escape(desc)}")
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
    # Agent state is now clean (or there was no agent) — drop every
    # mounted bubble and reset the streaming target / accumulator so
    # the next chunk opens a fresh Assistant bubble.
    for child in list(log.children):
        child.remove()
    log._current_assistant = None
    log._accum.clear()
    log._pending_tool_cards.clear()
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


def cmd_verbose(app: "LlamAgentTUI", arg: str) -> None:
    """Toggle / explicitly set the VerbosePane (round-14 user-test result).

    Replaces the original Ctrl+V keyboard binding from plan §2.3, which
    silently exited the App on macOS Terminal.app (termios VLNEXT collision).
    F3 was the obvious fallback but macOS Mission Control's Window Overview
    intercepts it. A slash command travels through the Input widget as plain
    text, so no terminal-level mapping can intercept it.

    Usage:
        /verbose        — toggle (current behaviour mirrors the old Ctrl+V)
        /verbose on     — force show
        /verbose off    — force hide
    """
    from llamagent.interfaces.cli_tui.widgets import VerbosePane

    try:
        pane = app.query_one("#verbose-pane", VerbosePane)
    except Exception as e:
        _error(app, f"VerbosePane unavailable: {type(e).__name__}: {e}")
        return

    target = arg.strip().lower()
    if target in ("on", "show", "open", "true", "1"):
        new_state = True
    elif target in ("off", "hide", "close", "false", "0"):
        new_state = False
    elif target == "":
        new_state = not pane.display
    else:
        _error(app, f"Invalid /verbose arg: {markup_escape(target)}. Use: on, off, or no arg (toggle)")
        return
    try:
        pane.display = new_state
    except Exception as e:
        _error(app, f"toggle failed: {type(e).__name__}: {e}")
        return
    _info(app, f"VerbosePane {'shown' if new_state else 'hidden'}")


def cmd_stop(app: "LlamAgentTUI", arg: str) -> None:
    """Stop continuous mode. C7 will wire ContinuousRunner; for C6 we
    just flip mode back so the user sees a deterministic state change.

    TODO(C7): when ContinuousRunner is integrated into the TUI, replace
    the ``agent.set_mode("interactive")`` call with the real runner stop
    sequence (mirrors legacy ``LlamAgentCLI._cmd_stop`` calling
    ``self._runner.stop()`` at cli.py:898).
    """
    agent = app.agent
    if agent is None or agent.mode != "continuous":
        _info(app, "Not in continuous mode")
        return
    try:
        agent.set_mode("interactive")
    except Exception as e:
        _error(app, f"set_mode(interactive) failed: {type(e).__name__}: {e}")
        return
    app.refresh_status_header()
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
