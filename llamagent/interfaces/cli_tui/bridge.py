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
import threading
import time
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
    from textual.app import App

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


def _drain_pending_for_thread(app: "App", tid: Optional[int] = None) -> None:
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
            app.post_message(
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


def install_hooks(agent, app: "App") -> None:
    """Register PRE/POST/TOOL_ERROR hooks on ``agent`` that post typed
    Messages to ``app``.

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
    agent._bridge_hooks_installed = True

    from llamagent.core.hooks import HookEvent

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
        try:
            app.post_message(
                ToolStartMessage(name=markup_escape(name), args=args, call_id=call_id)
            )
        except Exception as e:
            logger.debug("post ToolStartMessage failed: %s", e)

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
        try:
            app.post_message(
                ToolEndMessage(
                    call_id=call_id,
                    duration_ms=float(data.get("duration_ms", 0.0)),
                    result_preview=preview,
                )
            )
        except Exception as e:
            logger.debug("post ToolEndMessage failed: %s", e)

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
        try:
            app.post_message(
                ToolErrorMessage(call_id=call_id, name=markup_escape(name), error=err)
            )
        except Exception as e:
            logger.debug("post ToolErrorMessage failed: %s", e)

    agent.register_hook(HookEvent.PRE_TOOL_USE, _on_pre)
    agent.register_hook(HookEvent.POST_TOOL_USE, _on_post)
    agent.register_hook(HookEvent.TOOL_ERROR, _on_error)

    # Store handler refs so uninstall_hooks can find + remove them on
    # agent rebuild (round-8 Rev A M1 / plan v13 §12 defer entry).
    agent._bridge_hook_handlers = {
        HookEvent.PRE_TOOL_USE: _on_pre,
        HookEvent.POST_TOOL_USE: _on_post,
        HookEvent.TOOL_ERROR: _on_error,
    }


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


def run_turn(app: "App", agent, user_input: str) -> None:
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
                app.post_message(ChatChunkMessage(chunk))
            except Exception as e:
                # Worker shouldn't crash if the app went away mid-stream;
                # log + bail. drain() in finally still cleans up.
                logger.debug("post ChatChunkMessage failed (app gone?): %s", e)
                break
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.exception("agent.chat_stream raised in worker")
    finally:
        _drain_pending_for_thread(app, tid)
        try:
            app.post_message(TurnCompleteMessage(success=error is None, error=error))
        except Exception:
            pass
