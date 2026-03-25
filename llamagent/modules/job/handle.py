"""JobHandle: wraps subprocess.Popen with lifecycle management."""

from __future__ import annotations

import subprocess
import threading
import time
from typing import List


class JobHandle:
    """
    Wraps a subprocess.Popen instance with lifecycle management.

    Provides non-blocking stdout/stderr capture via background reader threads,
    status polling, waiting, and termination.
    """

    def __init__(
        self,
        job_id: str,
        process: subprocess.Popen,
        cwd: str,
        command: str,
        timeout: float,
    ):
        self.job_id = job_id
        self.process = process
        self.cwd = cwd
        self.command = command
        self.timeout = timeout
        self.start_time = time.time()
        self._stdout_lines: List[str] = []
        self._stderr_lines: List[str] = []
        self._stdout_lock = threading.Lock()
        self._stderr_lock = threading.Lock()

        # Start background reader threads for non-blocking I/O
        self._stdout_reader = threading.Thread(
            target=self._read_stream,
            args=(process.stdout, self._stdout_lines, self._stdout_lock),
            daemon=True,
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stream,
            args=(process.stderr, self._stderr_lines, self._stderr_lock),
            daemon=True,
        )
        self._stdout_reader.start()
        self._stderr_reader.start()

    @staticmethod
    def _read_stream(stream, buffer: List[str], lock: threading.Lock) -> None:
        """Continuously read lines from a stream into a buffer."""
        try:
            for line in stream:
                with lock:
                    buffer.append(line)
        except (ValueError, OSError):
            # Stream closed
            pass

    def poll(self) -> str:
        """
        Check the current status of the job.

        Returns:
            "running", "completed", "failed", or "timeout"
        """
        elapsed = time.time() - self.start_time
        if self.timeout > 0 and elapsed > self.timeout:
            # Timeout: terminate the process if still running
            if self.process.poll() is None:
                self.terminate()
            return "timeout"

        rc = self.process.poll()
        if rc is None:
            return "running"
        elif rc == 0:
            return "completed"
        else:
            return "failed"

    def read_output(self, lines: int = 200) -> str:
        """Return the last N lines from stdout."""
        with self._stdout_lock:
            tail = self._stdout_lines[-lines:] if lines > 0 else self._stdout_lines[:]
        return "".join(tail)

    def read_stderr(self, lines: int = 200) -> str:
        """Return the last N lines from stderr."""
        with self._stderr_lock:
            tail = self._stderr_lines[-lines:] if lines > 0 else self._stderr_lines[:]
        return "".join(tail)

    def wait(self, timeout: float | None = None) -> int:
        """
        Wait for the process to complete.

        Args:
            timeout: Maximum seconds to wait. None uses the job's configured timeout.
                     If the remaining budget from the configured timeout is less,
                     it takes precedence.

        Returns:
            Process exit code, or -1 on timeout.
        """
        if timeout is None:
            # Use remaining budget from the configured job timeout
            elapsed = time.time() - self.start_time
            remaining = max(self.timeout - elapsed, 0) if self.timeout > 0 else None
        else:
            remaining = timeout

        try:
            rc = self.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            self.terminate()
            self._stdout_reader.join(timeout=2)
            self._stderr_reader.join(timeout=2)
            return -1

        # Wait for reader threads to finish draining
        self._stdout_reader.join(timeout=2)
        self._stderr_reader.join(timeout=2)
        return rc

    def terminate(self) -> None:
        """Send SIGTERM to the process."""
        try:
            self.process.terminate()
        except OSError:
            pass  # Process already exited
