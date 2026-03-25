"""JobService: manages all active JobHandles."""

from __future__ import annotations

import logging
import subprocess
import uuid

from llamagent.modules.job.handle import JobHandle

logger = logging.getLogger(__name__)


class JobService:
    """
    Manages the lifecycle of all active jobs.

    Spawns subprocesses, wraps them in JobHandle instances, and provides
    lookup, listing, and bulk shutdown capabilities.
    """

    def __init__(self, max_active: int = 10):
        self.max_active = max_active
        self._jobs: dict[str, JobHandle] = {}

    def create_job(self, command: str, cwd: str, timeout: float) -> JobHandle:
        """
        Spawn a new subprocess and create a managed JobHandle.

        Args:
            command: Shell command to execute.
            cwd: Working directory for the subprocess.
            timeout: Maximum lifetime in seconds (0 = no timeout).

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

        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        handle = JobHandle(
            job_id=job_id,
            process=process,
            cwd=cwd,
            command=command,
            timeout=timeout,
        )
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
        """Terminate all running jobs and clear the registry."""
        for handle in self._jobs.values():
            if handle.poll() == "running":
                logger.info("Shutting down job: %s", handle.job_id)
                handle.terminate()
                try:
                    handle.process.wait(timeout=5)
                except Exception:
                    pass
                handle.wait(timeout=5)
        self._jobs.clear()
