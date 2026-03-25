"""
JobModule: managed command execution with lifecycle control.

Registers six tools for the agent:
- start_job:          Start a command (synchronous or asynchronous)
- wait_job:           Wait for an async job to complete
- job_status:         Check the current status of a job
- tail_job:           Read recent stdout/stderr from a job
- cancel_job:         Terminate a running job
- collect_artifacts:  List files created/modified during a job

The module uses subprocess.Popen for execution. It soft-depends on
ToolsModule (for workspace cwd resolution) and SafetyModule (for
command blacklist checking).
"""

from __future__ import annotations

import json
import logging
import os
import time

from llamagent.core.agent import Module
from llamagent.modules.job.service import JobService

logger = logging.getLogger(__name__)

# Default job timeout in seconds (used when no profile override is set)
DEFAULT_JOB_TIMEOUT = 300


class JobModule(Module):
    """
    Job system module: managed command execution with lifecycle control.

    Spawns and manages subprocesses with timeout, output capture, and
    artifact collection. Supports both synchronous (wait=True) and
    asynchronous (wait=False) execution modes.
    """

    name = "job"
    description = "Job system: managed command execution with lifecycle control"

    def __init__(self):
        self.service: JobService | None = None

    def on_attach(self, agent) -> None:
        """Initialize JobService and register all six job tools."""
        super().on_attach(agent)
        self.service = JobService(max_active=getattr(agent.config, "job_max_active", 10))

        # --- start_job (sl=2, common) ---
        agent.register_tool(
            name="start_job",
            func=self._start_job,
            description=(
                "Execute a shell command as a managed job. "
                "wait=True (default) blocks until completion and returns stdout/stderr. "
                "wait=False starts the job asynchronously and returns a job_id for later polling. "
                "cwd controls the working directory: \"workspace\" (default) uses the workspace root, "
                "\"project\" uses the project directory, relative paths resolve from workspace, "
                "absolute paths are used as-is."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory: \"workspace\" (default), \"project\", relative path, or absolute path",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "True (default) for synchronous execution, False for async",
                    },
                    "profile": {
                        "type": "string",
                        "description": "Execution profile name (default: \"default\")",
                    },
                },
                "required": ["command"],
            },
            tier="common",
            safety_level=2,
        )

        # --- wait_job (sl=1, common) ---
        agent.register_tool(
            name="wait_job",
            func=self._wait_job,
            description="Wait for an asynchronous job to complete and return its output",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID returned by start_job",
                    },
                },
                "required": ["job_id"],
            },
            tier="common",
            safety_level=1,
        )

        # --- job_status (sl=1, common) ---
        agent.register_tool(
            name="job_status",
            func=self._job_status,
            description="Check the current status of a job (running/completed/failed/timeout)",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to check",
                    },
                },
                "required": ["job_id"],
            },
            tier="common",
            safety_level=1,
        )

        # --- tail_job (sl=1, common) ---
        agent.register_tool(
            name="tail_job",
            func=self._tail_job,
            description="Read the last N lines of stdout and stderr from a job",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to read output from",
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of recent lines to return (default: 200)",
                    },
                },
                "required": ["job_id"],
            },
            tier="common",
            safety_level=1,
        )

        # --- cancel_job (sl=2, common) ---
        agent.register_tool(
            name="cancel_job",
            func=self._cancel_job,
            description="Terminate a running job by sending SIGTERM",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to cancel",
                    },
                },
                "required": ["job_id"],
            },
            tier="common",
            safety_level=2,
        )

        # --- collect_artifacts (sl=1, common) ---
        agent.register_tool(
            name="collect_artifacts",
            func=self._collect_artifacts,
            description="List files in the job's working directory that were created or modified during job execution",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID whose artifacts to collect",
                    },
                },
                "required": ["job_id"],
            },
            tier="common",
            safety_level=1,
        )

    def on_shutdown(self) -> None:
        """Terminate all running jobs on agent shutdown."""
        if self.service is not None:
            self.service.shutdown()

    # ================================================================
    # CWD resolution
    # ================================================================

    def _resolve_cwd(self, cwd: str) -> str:
        """
        Resolve the cwd parameter to an absolute directory path.

        Resolution rules:
        - "workspace" -> workspace root from ToolsModule, or playground_dir fallback
        - "project"   -> agent.project_dir
        - relative    -> resolved relative to workspace root
        - absolute    -> used as-is
        """
        agent = self.agent

        # Get workspace root (soft dependency on ToolsModule)
        workspace_root = None
        tools_mod = agent.get_module("tools")
        if tools_mod is not None:
            ws = getattr(tools_mod, "workspace_service", None)
            if ws is not None:
                workspace_root = getattr(ws, "workspace_root", None)

        # Fallback to playground_dir if ToolsModule unavailable
        if workspace_root is None:
            workspace_root = getattr(agent, "playground_dir", os.getcwd())

        if cwd == "workspace":
            return os.path.realpath(workspace_root)
        elif cwd == "project":
            return os.path.realpath(agent.project_dir)
        elif os.path.isabs(cwd):
            return os.path.realpath(cwd)
        else:
            # Relative path -> resolve from workspace root
            return os.path.realpath(os.path.join(workspace_root, cwd))

    # ================================================================
    # Tool implementations
    # ================================================================

    def _start_job(
        self,
        command: str,
        cwd: str = "workspace",
        wait: bool = True,
        profile: str = "default",
    ) -> str:
        """Start a command as a managed job."""
        # Safety check: if safety module is loaded, check command against blacklist
        safety_mod = self.agent.get_module("safety")
        if safety_mod is not None:
            rejection = safety_mod.check_command(command)
            if rejection:
                return json.dumps({
                    "status": "rejected",
                    "command": command,
                    "reason": rejection,
                }, ensure_ascii=False)

        # Resolve working directory
        resolved_cwd = self._resolve_cwd(cwd)
        os.makedirs(resolved_cwd, exist_ok=True)

        # Determine timeout from profile
        timeout = self._get_profile_timeout(profile)

        # Create the job
        try:
            handle = self.service.create_job(command, resolved_cwd, timeout)
        except RuntimeError as e:
            return json.dumps({
                "status": "error",
                "command": command,
                "reason": str(e),
            }, ensure_ascii=False)

        if wait:
            # Synchronous mode: wait for completion and return results
            rc = handle.wait()
            stdout = handle.read_output()
            stderr = handle.read_stderr()
            status = handle.poll()

            # Truncate overly long output
            max_len = 5000
            orig_stdout_len = len(stdout)
            if orig_stdout_len > max_len:
                stdout = stdout[:max_len] + f"\n...(output truncated, total {orig_stdout_len} characters)"
            orig_stderr_len = len(stderr)
            if orig_stderr_len > max_len:
                stderr = stderr[:max_len] + f"\n...(error output truncated, total {orig_stderr_len} characters)"

            if status == "timeout":
                return json.dumps({
                    "status": "timeout",
                    "command": command,
                    "reason": f"Command execution timed out ({timeout}-second limit)",
                    "stdout": stdout,
                    "stderr": stderr,
                }, ensure_ascii=False)

            return json.dumps({
                "status": "success" if rc == 0 else "error",
                "command": command,
                "return_code": rc,
                "stdout": stdout,
                "stderr": stderr,
            }, ensure_ascii=False)
        else:
            # Asynchronous mode: return job_id immediately
            return json.dumps({
                "job_id": handle.job_id,
                "status": "started",
            }, ensure_ascii=False)

    def _wait_job(self, job_id: str) -> str:
        """Wait for an async job to complete and return its output."""
        handle = self.service.get_job(job_id)
        if handle is None:
            return json.dumps({
                "status": "error",
                "reason": f"Job '{job_id}' not found",
            }, ensure_ascii=False)

        rc = handle.wait()
        stdout = handle.read_output()
        stderr = handle.read_stderr()
        status = handle.poll()

        # Truncate overly long output
        max_len = 5000
        orig_stdout_len = len(stdout)
        if orig_stdout_len > max_len:
            stdout = stdout[:max_len] + f"\n...(output truncated, total {orig_stdout_len} characters)"
        orig_stderr_len = len(stderr)
        if orig_stderr_len > max_len:
            stderr = stderr[:max_len] + f"\n...(error output truncated, total {orig_stderr_len} characters)"

        if status == "timeout":
            return json.dumps({
                "status": "timeout",
                "command": handle.command,
                "reason": f"Command execution timed out ({handle.timeout}-second limit)",
                "stdout": stdout,
                "stderr": stderr,
            }, ensure_ascii=False)

        return json.dumps({
            "status": "success" if rc == 0 else "error",
            "command": handle.command,
            "return_code": rc,
            "stdout": stdout,
            "stderr": stderr,
        }, ensure_ascii=False)

    def _job_status(self, job_id: str) -> str:
        """Check the current status of a job."""
        handle = self.service.get_job(job_id)
        if handle is None:
            return json.dumps({
                "status": "error",
                "reason": f"Job '{job_id}' not found",
            }, ensure_ascii=False)

        elapsed = time.time() - handle.start_time
        return json.dumps({
            "job_id": handle.job_id,
            "status": handle.poll(),
            "elapsed_seconds": round(elapsed, 1),
        }, ensure_ascii=False)

    def _tail_job(self, job_id: str, lines: int = 200) -> str:
        """Read recent output from a job."""
        handle = self.service.get_job(job_id)
        if handle is None:
            return json.dumps({
                "status": "error",
                "reason": f"Job '{job_id}' not found",
            }, ensure_ascii=False)

        return json.dumps({
            "job_id": handle.job_id,
            "stdout": handle.read_output(lines),
            "stderr": handle.read_stderr(lines),
        }, ensure_ascii=False)

    def _cancel_job(self, job_id: str) -> str:
        """Terminate a running job."""
        handle = self.service.get_job(job_id)
        if handle is None:
            return json.dumps({
                "status": "error",
                "reason": f"Job '{job_id}' not found",
            }, ensure_ascii=False)

        current_status = handle.poll()
        if current_status != "running":
            return json.dumps({
                "job_id": job_id,
                "status": "already_completed",
            }, ensure_ascii=False)

        handle.terminate()
        return json.dumps({
            "job_id": job_id,
            "status": "cancelled",
        }, ensure_ascii=False)

    def _collect_artifacts(self, job_id: str) -> str:
        """List files created or modified in the job's cwd during execution."""
        handle = self.service.get_job(job_id)
        if handle is None:
            return json.dumps({
                "status": "error",
                "reason": f"Job '{job_id}' not found",
            }, ensure_ascii=False)

        cwd = handle.cwd
        start_time = handle.start_time
        artifacts = []

        try:
            for root, _dirs, files in os.walk(cwd):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        st = os.stat(fpath)
                        if st.st_mtime >= start_time:
                            artifacts.append({
                                "path": os.path.relpath(fpath, cwd),
                                "size": st.st_size,
                                "mtime": round(st.st_mtime, 2),
                            })
                    except OSError:
                        continue
        except OSError:
            pass

        # Sort by modification time (most recent first)
        artifacts.sort(key=lambda a: a["mtime"], reverse=True)

        return json.dumps({
            "job_id": handle.job_id,
            "artifacts": artifacts,
        }, ensure_ascii=False)

    # ================================================================
    # Profile resolution
    # ================================================================

    def _get_profile_timeout(self, profile: str) -> float:
        """
        Resolve a profile name to a timeout value.

        Reads from agent config: job_default_timeout for "default" profile,
        job_profiles dict for custom profiles. Falls back to DEFAULT_JOB_TIMEOUT
        if config values are not set.
        """
        config = self.agent.config
        if profile == "default":
            return getattr(config, "job_default_timeout", DEFAULT_JOB_TIMEOUT)
        profiles = getattr(config, "job_profiles", {})
        if profile in profiles:
            return profiles[profile].get("timeout", DEFAULT_JOB_TIMEOUT)
        return getattr(config, "job_default_timeout", DEFAULT_JOB_TIMEOUT)
