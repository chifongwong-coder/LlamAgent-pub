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

    def __init__(self, max_active: int = 10):
        self.max_active = max_active
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
        """Cancel all running jobs and clear the registry."""
        for handle in self._jobs.values():
            if handle.poll() == "running":
                logger.info("Shutting down job: %s", handle.job_id)
                handle.cancel()
        self._jobs.clear()
