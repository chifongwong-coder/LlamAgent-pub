"""
JobHandle: wraps a ToolExecutor execution running in a background thread.

No direct subprocess usage — execution is delegated to the sandbox backend
via agent.tool_executor. JobHandle manages the thread lifecycle and provides
poll/wait/read_output for the job tools.

Phase 1 limitation: tail_job cannot stream partial output because
ToolExecutor.execute() is synchronous (returns full result only on completion).
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class JobHandle:
    """
    Wraps a command execution running in a background thread via ToolExecutor.

    The actual execution is delegated to the sandbox backend (LocalProcessBackend,
    future DockerBackend, etc.) through ToolExecutor.execute(). This class only
    manages the thread lifecycle and stores the result.
    """

    def __init__(self, job_id: str, command: str, cwd: str, timeout: float):
        self.job_id = job_id
        self.command = command
        self.cwd = cwd
        self.timeout = timeout
        self.start_time = time.time()

        # Result from tool_executor.execute() — available after thread completes
        self._result: str | None = None
        self._error: str | None = None
        self._exit_code: int | None = None
        self._completed = threading.Event()
        self._thread: threading.Thread | None = None
        self._cancelled = False

    def start(self, executor_fn) -> None:
        """
        Start execution in a background thread.

        Args:
            executor_fn: Callable that performs the execution and returns
                         the result string. This is typically a closure over
                         tool_executor.execute().
        """
        def _run():
            try:
                self._result = executor_fn()
                self._exit_code = 0
            except Exception as e:
                self._error = str(e)
                self._exit_code = 1
                logger.warning("Job '%s' execution error: %s", self.job_id, e)
            finally:
                self._completed.set()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def poll(self) -> str:
        """
        Check the current status of the job.

        Returns:
            "running", "completed", "failed", "timeout", or "cancelled"
        """
        if self._cancelled:
            return "cancelled"

        if not self._completed.is_set():
            # Check timeout
            elapsed = time.time() - self.start_time
            if self.timeout > 0 and elapsed > self.timeout:
                return "timeout"
            return "running"

        if self._error:
            return "failed"
        return "completed"

    def read_output(self, lines: int = 200) -> str:
        """
        Read job output.

        Phase 1 limitation: only available after job completes (ToolExecutor.execute
        is synchronous, no streaming). Returns empty string while running.
        """
        if not self._completed.is_set():
            return ""
        result = self._result or self._error or ""
        if lines > 0:
            result_lines = result.splitlines(keepends=True)
            return "".join(result_lines[-lines:])
        return result

    def wait(self, timeout: float | None = None) -> int:
        """
        Wait for the job to complete.

        Args:
            timeout: Maximum seconds to wait. None uses remaining job timeout budget.

        Returns:
            Exit code (0=success, 1=error, -1=timeout).
        """
        if timeout is None:
            elapsed = time.time() - self.start_time
            remaining = max(self.timeout - elapsed, 0) if self.timeout > 0 else None
        else:
            remaining = timeout

        completed = self._completed.wait(timeout=remaining)
        if not completed:
            return -1

        return self._exit_code if self._exit_code is not None else -1

    def cancel(self) -> bool:
        """
        Cancel the job.

        Note: The background thread is running tool_executor.execute() which is
        blocking. We mark the job as cancelled but cannot forcefully interrupt the
        thread. The backend's timeout will eventually stop it.

        Returns:
            True if marked as cancelled, False if already completed.
        """
        if self._completed.is_set():
            return False
        self._cancelled = True
        self._completed.set()
        self._error = "Job cancelled by user"
        self._exit_code = -1
        return True
