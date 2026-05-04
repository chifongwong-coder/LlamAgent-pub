"""
JobService: manages all active JobHandles.

No direct subprocess usage — execution is delegated to ToolExecutor
via executor_fn closures passed to JobHandle.start().
"""

from __future__ import annotations

import logging
import uuid

from llamagent.modules.job.handle import JobHandle

logger = logging.getLogger(__name__)


class JobService:
    """
    Manages the lifecycle of all active jobs.

    Creates JobHandle instances and starts them with executor functions
    that delegate to ToolExecutor. No subprocess management here.
    """

    def __init__(self, max_active: int = 10, *, cancel_join_timeout: float = 5.0):
        self.max_active = max_active
        # v3.7.4: forwarded to each JobHandle so cancel() can bound-join
        # the worker thread (see JobHandle.cancel for rationale).
        self._cancel_join_timeout = cancel_join_timeout
        self._jobs: dict[str, JobHandle] = {}

    def create_job(
        self, command: str, cwd: str, timeout: float, executor_fn
    ) -> JobHandle:
        """
        Create a new job and start execution in a background thread.

        Args:
            command: Shell command to execute.
            cwd: Working directory for execution.
            timeout: Maximum lifetime in seconds (0 = no timeout).
            executor_fn: Callable that performs the actual execution via ToolExecutor.
                         Must return a result string.

        Returns:
            The new JobHandle.

        Raises:
            RuntimeError: If the max_active job limit is reached.
        """
        active_count = sum(1 for h in self._jobs.values() if h.poll() == "running")
        if active_count >= self.max_active:
            raise RuntimeError(
                f"Maximum active job limit reached ({self.max_active}). "
                f"Cancel or wait for existing jobs before starting new ones."
            )

        job_id = uuid.uuid4().hex[:12]
        handle = JobHandle(
            job_id=job_id, command=command, cwd=cwd, timeout=timeout,
            cancel_join_timeout=self._cancel_join_timeout,
        )
        handle.start(executor_fn)
        self._jobs[job_id] = handle
        logger.info("Job created: %s (command=%r, cwd=%s)", job_id, command, cwd)
        return handle

    def get_job(self, job_id: str) -> JobHandle | None:
        """Get a JobHandle by its ID. Returns None if not found."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobHandle]:
        """Return a list of all managed JobHandles."""
        return list(self._jobs.values())

    def has_active_jobs(self) -> bool:
        """Return True if any job is currently running."""
        return any(h.poll() == "running" for h in self._jobs.values())

    def shutdown(self) -> None:
        """Cancel all running jobs and clear the registry.

        v3.7.4: two-pass cancel so total wait stays bounded by
        ``cancel_join_timeout`` rather than ``N * cancel_join_timeout``.
        Pass 1 signals every running handle (no join), pass 2 joins each
        worker thread with the configured timeout. Workers are daemon, so
        any thread that misses both passes is still reaped on process exit.
        """
        running = [h for h in self._jobs.values() if h.poll() == "running"]
        for handle in running:
            logger.info("Shutting down job: %s", handle.job_id)
            handle.cancel(join_timeout=0)  # signal only
        # Second pass: bounded join. Threads that finished during pass 1 are
        # cheap (is_alive() returns False, the join inside cancel becomes
        # a no-op via the timeout > 0 + is_alive() guard).
        for handle in running:
            thread = handle._thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=self._cancel_join_timeout)
                if thread.is_alive():
                    logger.warning(
                        "JobHandle %s worker did not exit within %.1fs during "
                        "shutdown; worker is daemon and will be reaped by the OS.",
                        handle.job_id,
                        self._cancel_join_timeout,
                    )
        self._jobs.clear()
