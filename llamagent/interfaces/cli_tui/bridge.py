"""Worker-thread ↔ Textual App bridge (plan v12 §2.5 / §3.2).

C2 wires real ``agent.chat_stream`` into the TUI:

- ``install_hooks(agent, app)`` registers PRE_TOOL_USE / POST_TOOL_USE /
  TOOL_ERROR callbacks that build typed Messages and post them to the
  App. The hook callbacks run on the same worker thread that's
  iterating chat_stream (plan §1.4 framework fact); thread-id is the
  natural per-thread pending stack key.
- ``run_turn(app, agent, user_input)`` is the body of the App's
  ``@work(thread=True, exclusive=True)`` worker: it iterates the sync
  chat_stream generator, posts ChatChunkMessage per chunk, and emits
  TurnCompleteMessage in a try/finally so the generator's lazy
  evaluation (plan v9 BLOCKER B3) can't leak pending tool cards
  across turns.

Key invariants:
- ``call_id`` is TUI-generated as ``t<thread_id>-<counter>``; framework
  hooks don't carry one (plan §1.4). The counter is read inside the
  ``_pending_lock`` to keep it atomic across free-threaded Python builds
  (plan §2.5 H-R3-1).
- Tool name + args + result_preview can contain Rich markup chars;
  callers / handlers must escape via ``rich.markup.escape`` before any
  Static.update with ``markup=True``. Bridge does the escape at the
  hook callback boundary so widget code stays simple.
- post_message is thread-safe per Textual docs
  (https://textual.textualize.io/guide/workers/#thread-workers); we
  do NOT use ``call_from_thread`` here. The exception is modal
  push_screen which IS NOT thread-safe — that lives in C4 bridge code.
"""
import itertools
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.markup import escape as markup_escape

from llamagent.interfaces.cli_tui.messages import (
    ChatChunkMessage,
    ToolEndMessage,
    ToolErrorMessage,
    ToolStartMessage,
    TurnCompleteMessage,
)

if TYPE_CHECKING:
    from textual.widget import Widget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# call_id generation + per-thread pending stack
# ---------------------------------------------------------------------------

_pending_lock = threading.Lock()
_call_id_counter = itertools.count()
# thread_id -> list[(tool_name, call_id, started_at)]   — LIFO stack
_pending_by_thread: dict[int, list[tuple[str, str, float]]] = {}


def _next_call_id() -> str:
    """Generate a fresh call_id.

    Must be called inside ``_pending_lock`` so the counter increment is
    atomic on free-threaded Python builds (plan v11 round-3 H-R3-1).
    """
    return f"t{threading.get_ident()}-{next(_call_id_counter)}"


def _pop_pending(name: str) -> Optional[tuple[str, str, float]]:
    """Pop the most-recent unpaired entry matching ``name`` on this thread.

    LIFO matching: nested same-name calls (rare but possible) pair the
    inner POST with the inner PRE. Returns None when nothing matches —
    handler renders an orphan card with the call_id (round-7 MED A-1).
    """
    tid = threading.get_ident()
    with _pending_lock:
        stack = _pending_by_thread.get(tid, [])
        for i in range(len(stack) - 1, -1, -1):
            if stack[i][0] == name:
                return stack.pop(i)
        return None


def _drain_pending_for_thread(target: "Widget", tid: Optional[int] = None) -> None:
    """Emit ToolErrorMessage for any PREs left unmatched at turn end.

    Plan v11 §2.5 H2 — without this, an aborted / killed tool leaves
    its card pending forever and the next same-name PRE may LIFO-match
    the wrong stale entry. Called from ``run_turn`` finally block.
    """
    if tid is None:
        tid = threading.get_ident()
    with _pending_lock:
        orphans = _pending_by_thread.pop(tid, [])
    now = time.monotonic()
    for name, call_id, started_at in orphans:
        try:
            target.post_message(
                ToolErrorMessage(
                    call_id=call_id,
                    name=name,
                    error=f"orphaned (turn ended without POST/ERROR; ran {(now - started_at) * 1000:.0f}ms)",
                )
            )
        except Exception as e:
            logger.debug("drain orphan post failed for %s: %s", call_id, e)


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------


def install_hooks(
    agent,
    target: "Widget",
    verbose_target: "Optional[Widget]" = None,
) -> None:
    """Register PRE/POST/TOOL_ERROR hooks on ``agent`` that post typed
    Messages to ``target`` (chat surface) and optionally ``verbose_target``
    (VerbosePane).

    Round-12 B1: VerbosePane has its own Tool*Message handlers but
    Textual on_<msg> only fires on the widget that received post_message
    — messages don't fan out to siblings. So when verbose_target is set
    we post each tool message to both widgets. ChatLog renders the
    folded card; VerbosePane renders the full args / result_preview /
    error (plan §4 C5 verification: "tool args / result 显示在右栏").

    Idempotent via the ``_bridge_hooks_installed`` attribute on the
    agent — calling twice is a no-op. Each hook callback escapes any
    Rich markup chars in user-controlled fields (tool name, args,
    result_preview, error) before placing them into Message dataclass
    fields — keeps the round-7 NIT A-1 markup-injection concern from
    biting C2 the moment a real LLM produces ``[bold]`` in a tool
    arg or result.
    """
    if getattr(agent, "_bridge_hooks_installed", False):
        return

    from llamagent.core.hooks import HookEvent

    def _post_both(msg) -> None:
        """Post ``msg`` to chat target + verbose target (if set).

        Each post is wrapped individually so a destroyed widget on one
        side doesn't suppress the other side. post_message failures
        downgrade to logger.debug (round-8 widget-gone tolerance).
        """
        try:
            target.post_message(msg)
        except Exception as e:
            logger.debug("post %s to chat target failed: %s", type(msg).__name__, e)
        if verbose_target is not None:
            try:
                verbose_target.post_message(msg)
            except Exception as e:
                logger.debug(
                    "post %s to verbose target failed: %s", type(msg).__name__, e
                )

    def _on_pre(ctx) -> None:
        data = getattr(ctx, "data", {}) or {}
        name = str(data.get("tool_name", "?"))
        args = data.get("args", {}) or {}
        # Generate call_id + record pending
        with _pending_lock:
            call_id = _next_call_id()
            tid = threading.get_ident()
            _pending_by_thread.setdefault(tid, []).append((name, call_id, time.monotonic()))
        # Escape markup in name (args dict goes verbatim — widget handler
        # is responsible for repr() + escape if it embeds in markup)
        _post_both(
            ToolStartMessage(name=markup_escape(name), args=args, call_id=call_id)
        )

    def _on_post(ctx) -> None:
        data = getattr(ctx, "data", {}) or {}
        name = str(data.get("tool_name", "?"))
        matched = _pop_pending(name)
        if matched is not None:
            call_id = matched[1]
        else:
            # Orphan id increment must be under the lock — free-threaded
            # Python doesn't guarantee `next(itertools.count)` atomicity
            # (round-8 LOW-1; matches the H-R3-1 invariant from round-3).
            with _pending_lock:
                call_id = f"orphan-{next(_call_id_counter)}"
        preview = data.get("result_preview") or data.get("result") or ""
        preview = markup_escape(str(preview))
        _post_both(
            ToolEndMessage(
                call_id=call_id,
                duration_ms=float(data.get("duration_ms", 0.0)),
                result_preview=preview,
            )
        )

    def _on_error(ctx) -> None:
        data = getattr(ctx, "data", {}) or {}
        name = str(data.get("tool_name", "?"))
        matched = _pop_pending(name)
        if matched is not None:
            call_id = matched[1]
        else:
            with _pending_lock:
                call_id = f"orphan-{next(_call_id_counter)}"
        err = markup_escape(str(data.get("error", "(unspecified)")))
        _post_both(
            ToolErrorMessage(call_id=call_id, name=markup_escape(name), error=err)
        )

    # Atomic install (round-10 HIGH B-H1): record the handlers map
    # BEFORE calling register_hook so uninstall_hooks can find any
    # partially-registered handlers if a register_hook call raises.
    # Only flip the idempotency flag after all three registrations
    # succeed — otherwise a future install_hooks would skip while
    # half-registered, leaking dead closures.
    agent._bridge_hook_handlers = {
        HookEvent.PRE_TOOL_USE: _on_pre,
        HookEvent.POST_TOOL_USE: _on_post,
        HookEvent.TOOL_ERROR: _on_error,
    }
    try:
        agent.register_hook(HookEvent.PRE_TOOL_USE, _on_pre)
        agent.register_hook(HookEvent.POST_TOOL_USE, _on_post)
        agent.register_hook(HookEvent.TOOL_ERROR, _on_error)
    except Exception:
        # Roll back any partial registration via uninstall_hooks
        # (identity-matches by CallableHandler.func, so only the
        # handlers we actually registered get removed).
        uninstall_hooks(agent)
        raise
    agent._bridge_hooks_installed = True


def uninstall_hooks(agent) -> None:
    """Remove bridge-registered hooks from ``agent``.

    Called by the App before rebinding to a new agent (or when the
    same agent is rebuilt via SetupScreen) so dead closures don't
    keep firing into a stale App reference. Matching round-8 Rev A M1
    + plan v13 §12 defer entry.

    No-op if ``agent`` was never installed.
    """
    handlers = getattr(agent, "_bridge_hook_handlers", None)
    if not handlers:
        return
    # agent._hooks is a per-instance dict[HookEvent, list[HookRegistration]]
    # (agent.py:826). Each HookRegistration has .handler → CallableHandler,
    # and CallableHandler.func is the original callback we registered
    # (core/hooks.py:158). Identity-match through both layers.
    hooks_dict = getattr(agent, "_hooks", None)
    if hooks_dict is not None:
        for event, fn in handlers.items():
            bucket = hooks_dict.get(event, [])
            try:
                kept = []
                for reg in bucket:
                    inner = getattr(reg, "handler", None)
                    inner_fn = getattr(inner, "func", None) if inner is not None else None
                    if inner_fn is fn:
                        continue
                    kept.append(reg)
                hooks_dict[event] = kept
            except Exception as e:
                logger.debug("uninstall_hooks event %s: %s", event, e)
    try:
        delattr(agent, "_bridge_hook_handlers")
    except AttributeError:
        pass
    try:
        delattr(agent, "_bridge_hooks_installed")
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Worker function — body of App's @work(thread=True, exclusive=True)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Confirm / Ask handler adapters (plan §4 C4)
# ---------------------------------------------------------------------------


# Modal response wait timeout — plan §2.4 / §4 C4 (60s sentinel-dismiss).
MODAL_RESPONSE_TIMEOUT_S = 60.0


def install_handlers(agent, app) -> None:
    """Wire ``agent.confirm_handler`` + ``agent.interaction_handler`` so
    framework authorization prompts and ``ask_user`` calls route through
    the TUI's ConfirmModal / AskUserModal (plan §4 C4).

    Both handlers are called synchronously from the agent worker thread.
    Each one:
      1. Pushes its modal via ``app.call_from_thread(app.push_screen,
         ...)`` — push_screen is NOT thread-safe (Textual docs), only
         post_message is; call_from_thread is the documented bridge.
      2. Blocks on a ``threading.Event`` until the modal dismisses
         (UI thread runs the modal, sets the event in the dismiss
         callback) OR ``MODAL_RESPONSE_TIMEOUT_S`` elapses → sentinel
         response (deny / empty), matching plan §4 C4 "modal 60s
         无应答自动 sentinel-dismiss".

    Sentinel semantics:
      - confirm_handler timeout / Esc / cancel → ConfirmResponse(
        allow=False, approved_scopes=None)
      - interaction_handler timeout / Esc / cancel → empty string
        (the ``ask_user`` tool treats empty answer as cancel)

    Setting these attributes is idempotent — last installer wins, which
    is the right semantics for set_agent rebuild.
    """
    from llamagent.core.zone import ConfirmResponse
    from llamagent.modules.tools.interaction import UserInteractionHandler

    def _confirm_handler(req):
        from llamagent.interfaces.cli_tui.screens import ConfirmModal

        result_holder: list = []
        event = threading.Event()
        modal = ConfirmModal(req)

        def _on_dismiss(value):
            result_holder.append(value)
            event.set()

        try:
            app.call_from_thread(app.push_screen, modal, _on_dismiss)
        except Exception as e:
            logger.debug("confirm modal push_screen failed: %s", e)
            return ConfirmResponse(allow=False, approved_scopes=None)

        if not event.wait(timeout=MODAL_RESPONSE_TIMEOUT_S):
            logger.warning(
                "confirm modal 60s timeout — sentinel deny for tool=%s",
                getattr(req, "tool_name", "?"),
            )
            # Round-11 B3 — actively dismiss the modal so the user
            # doesn't keep staring at a dead dialog. dismiss(None)
            # re-fires _on_dismiss(None) (harmless — event already
            # used, result_holder no longer read).
            try:
                app.call_from_thread(modal.dismiss, None)
            except Exception:
                pass
            return ConfirmResponse(allow=False, approved_scopes=None)

        value = result_holder[0] if result_holder else None
        if value is True:
            return ConfirmResponse(allow=True, approved_scopes=None)
        return ConfirmResponse(allow=False, approved_scopes=None)

    class _TUIInteractionHandler(UserInteractionHandler):
        def ask(self, question, choices=None):
            from llamagent.interfaces.cli_tui.screens import AskUserModal

            result_holder: list = []
            event = threading.Event()
            modal = AskUserModal(question, choices)

            def _on_dismiss(value):
                result_holder.append(value)
                event.set()

            try:
                app.call_from_thread(app.push_screen, modal, _on_dismiss)
            except Exception as e:
                logger.debug("ask modal push_screen failed: %s", e)
                return ""

            if not event.wait(timeout=MODAL_RESPONSE_TIMEOUT_S):
                logger.warning("ask modal 60s timeout — sentinel empty")
                # Round-11 B3 — same modal-dismiss pattern as confirm.
                try:
                    app.call_from_thread(modal.dismiss, None)
                except Exception:
                    pass
                return ""

            value = result_holder[0] if result_holder else None
            return value if isinstance(value, str) else ""

    # Round-11 H-2 — atomic install. Mirror install_hooks pattern:
    # build the handler closures first, swap onto the agent in one
    # block, and only set the idempotency flag after both
    # assignments succeed so a partial install can be cleaned up
    # by a subsequent uninstall_handlers call.
    interaction_handler_instance = _TUIInteractionHandler()
    try:
        agent.confirm_handler = _confirm_handler
        agent.interaction_handler = interaction_handler_instance
        # Round-11 BLOCKER B-1 — ToolsModule snapshots the handler
        # into agent._tool_state["ask_user_handler"] at attach time
        # (modules/tools/module.py:234); the builtin ask_user tool
        # reads from that snapshot, not agent.interaction_handler.
        # Without this rewrite the SetupScreen → set_agent path
        # leaves the snapshot at None and ask_user always returns
        # "no interaction handler configured" — primary user flow
        # broken silently. Mirror tools/module.py snapshot key.
        state = getattr(agent, "_tool_state", None)
        if isinstance(state, dict):
            state["ask_user_handler"] = interaction_handler_instance
    except Exception:
        # Rollback — leave nothing partially attached.
        agent.confirm_handler = None
        agent.interaction_handler = None
        state = getattr(agent, "_tool_state", None)
        if isinstance(state, dict):
            state["ask_user_handler"] = None
        raise
    agent._tui_handlers_installed = True


def uninstall_handlers(agent) -> None:
    """Remove TUI confirm / interaction handlers from ``agent``.

    Idempotent. Mirrors uninstall_hooks contract so set_agent can swap
    cleanly between agents.
    """
    if not getattr(agent, "_tui_handlers_installed", False):
        return
    # Restore framework defaults — agent.py assigns ``None`` by default
    # in __init__ (chat falls back to permissive defaults when these
    # are None). Setting back to None lets the agent decide instead of
    # holding stale closures pointing at a dead app reference.
    agent.confirm_handler = None
    agent.interaction_handler = None
    # Round-11 B-1 mirror — clear the ToolsModule snapshot so the
    # built-in ask_user tool falls back to "no handler" instead of
    # firing into a dead closure after agent rebuild.
    state = getattr(agent, "_tool_state", None)
    if isinstance(state, dict):
        state["ask_user_handler"] = None
    try:
        delattr(agent, "_tui_handlers_installed")
    except AttributeError:
        pass


def run_turn(target: "Widget", agent, user_input: str) -> None:
    """Iterate ``agent.chat_stream(user_input)`` and post ChatChunkMessage
    per chunk. Emits TurnCompleteMessage in finally so ChatLog can
    finalize / reflow Markdown even if the generator raises.

    Plan v11 B3 (round-3): the try/finally MUST wrap the generator body
    inline (not via ``yield from`` on a separate generator function) so
    that ``_drain_pending_for_thread`` runs even when the worker is
    abandoned mid-stream (e.g. App quit during a turn).

    Cancellation (round-8 HIGH-1): ``run_worker(exclusive=True)`` in
    thread mode only sets ``worker.is_cancelled``; it does NOT kill the
    OS thread (Textual ``worker.py`` source verified — Python can't
    safely kill threads). Without an explicit cancel check, the old
    worker keeps posting chunks into the App while the user's 2nd turn
    is already streaming → ChatLog appends to the wrong bubble. We
    poll ``worker.is_cancelled`` at the top of each chunk and call
    ``agent.abort()`` to let the framework's inner loops bail at their
    next checkpoint (agent.py reads ``self._abort`` at multiple sites
    in chat_stream / ReAct).
    """
    from textual.worker import get_current_worker

    tid = threading.get_ident()
    error: Optional[str] = None
    try:
        worker = get_current_worker()
    except Exception:
        worker = None  # Called outside a worker (e.g. unit test)

    try:
        for chunk in agent.chat_stream(user_input):
            if worker is not None and worker.is_cancelled:
                # Signal the framework to stop its inner loops. agent.abort()
                # sets self._abort=True which chat_stream / ReAct check.
                # If the user spammed Enter, this is what stops turn N-1
                # so turn N's ChatLog isn't polluted with stale chunks.
                try:
                    agent.abort()
                except Exception:
                    pass
                break
            try:
                target.post_message(ChatChunkMessage(chunk))
            except Exception as e:
                # Worker shouldn't crash if the app went away mid-stream;
                # log + bail. drain() in finally still cleans up.
                logger.debug("post ChatChunkMessage failed (app gone?): %s", e)
                break
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.exception("agent.chat_stream raised in worker")
    finally:
        _drain_pending_for_thread(target, tid)
        try:
            target.post_message(TurnCompleteMessage(success=error is None, error=error))
        except Exception:
            pass
