"""Persistent background event loop for MCP coroutines (v3.8.1 R7-#2 fix).

Pre-fix: ``MCPModule._init_client`` and ``bridge_func`` each called
``asyncio.run(...)`` per invocation. Each call created a fresh loop and
tore it down on completion, leaving any registered ``ClientSession``
referencing a destroyed loop. Inside FastAPI / Gradio / Jupyter (where an
outer event loop is already running), the cross-loop reference produced
``RuntimeError: ... bound to a different event loop`` on the first MCP
tool invocation.

Fix: every MCPClient owns one persistent ``asyncio`` event loop running
on a daemon thread for the client's lifetime. All coroutines that touch
the same sessions (connect, call_tool, disconnect) run on this loop via
``run_coroutine_threadsafe``. The shutdown sequence is canonical:

    1. submit ``disconnect_all()`` onto the persistent loop and wait for
       the future (sessions cleanup runs on the loop they were opened on)
    2. ``call_soon_threadsafe(loop.stop)``
    3. ``thread.join(timeout=...)``

Doing disconnect on a fresh short-lived loop would re-introduce the very
"different event loop" RuntimeError this fix is closing — that path is
rejected (see v3.8.1 plan §1 R7-#2 round-3 review).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Coroutine, Optional

logger = logging.getLogger(__name__)


class PersistentEventLoop:
    """Background asyncio loop running on a daemon thread."""

    def __init__(self, name: str = "mcp-loop"):
        self._name = name
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the loop thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._loop = asyncio.new_event_loop()
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name=self._name
            )
            self._thread.start()
            # Wait until run_forever is actually pumping the loop —
            # otherwise the first submit() can race with loop startup
            # and asyncio.run_coroutine_threadsafe gets a non-running loop.
            if not self._ready.wait(timeout=5.0):
                logger.warning(
                    "[%s] loop thread did not signal ready within 5s",
                    self._name,
                )

    def _run(self) -> None:
        loop = self._loop
        assert loop is not None
        asyncio.set_event_loop(loop)
        try:
            loop.call_soon(self._ready.set)
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def submit(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        """Schedule a coroutine on the loop and block for its result.

        Raises:
            RuntimeError: if loop is not running (start() not called or
                already stopped).
            concurrent.futures.TimeoutError: if the coroutine doesn't
                complete within ``timeout``.
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(
                f"PersistentEventLoop[{self._name}] is not running; call start() first"
            )
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def stop(self, *, join_timeout: float = 5.0) -> None:
        """Stop the loop and join the thread (idempotent).

        Callers must drain in-flight work via ``submit()`` BEFORE calling
        ``stop()``; after stop, ``submit()`` raises RuntimeError.
        """
        with self._lock:
            loop = self._loop
            thread = self._thread
            if loop is None or thread is None:
                return
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            if thread.is_alive():
                thread.join(timeout=join_timeout)
                if thread.is_alive():
                    logger.warning(
                        "[%s] loop thread did not exit within %.1fs; "
                        "daemon thread continues, OS reaps on process exit.",
                        self._name,
                        join_timeout,
                    )
            self._loop = None
            self._thread = None
            self._ready.clear()

    def is_running(self) -> bool:
        return self._loop is not None and self._loop.is_running()
