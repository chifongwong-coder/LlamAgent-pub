"""
LocalProcessBackend: sandbox backend using subprocess for isolation.

This is the default backend shipped with LlamAgent. It provides process-level
isolation via subprocess.run with timeout enforcement, working-directory
confinement, and captured output.

Limitations:
- No true filesystem or network isolation (relies on OS-level permissions).
- Suitable for development and low-risk sandboxing; production deployments
  should consider container or microVM backends.
"""

from __future__ import annotations

import subprocess
import tempfile
import time

from llamagent.modules.sandbox.backend import (
    ExecutionBackend,
    ExecutionResult,
    ExecutionSession,
    ExecutionSpec,
)
from llamagent.modules.sandbox.policy import ExecutionPolicy


class LocalProcessSession(ExecutionSession):
    """A session backed by subprocess execution in a temporary workspace."""

    def __init__(self, policy: ExecutionPolicy, workspace: str) -> None:
        self._policy = policy
        self._workspace = workspace

    def run(self, spec: ExecutionSpec) -> ExecutionResult:
        """
        Execute *spec* in a subprocess.

        Supported runtimes:
        - "python": runs ``python -c <code>`` where code comes from args["code"].
        - "shell": runs ``sh -c <command>`` where command comes from args["command"].
        """
        policy = spec.policy
        args = spec.args

        # Build the command based on runtime.
        if policy.runtime == "python":
            code = args.get("code", "")
            cmd = ["python", "-c", code]
        elif policy.runtime == "shell":
            command = args.get("command", "")
            cmd = ["sh", "-c", command]
        else:
            return ExecutionResult(
                stderr=f"Unsupported runtime: {policy.runtime!r}",
                exit_code=1,
            )

        # Determine timeout.
        timeout = policy.timeout_seconds

        # Prepare environment: minimal base + explicit env_vars only.
        # Never inherit full host environment (API keys, credentials, etc.)
        import os
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "TERM": os.environ.get("TERM", "xterm"),
        }
        if spec.env_vars:
            env.update(spec.env_vars)

        # Execute.
        start_time = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._workspace,
                env=env,
            )
            duration_ms = (time.monotonic() - start_time) * 1000

            return ExecutionResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration_ms=duration_ms,
            )

        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            return ExecutionResult(
                stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr or "" if isinstance(exc.stderr, str) else "",
                exit_code=-1,
                duration_ms=duration_ms,
                timed_out=True,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            return ExecutionResult(
                stderr=f"Subprocess execution failed: {exc}",
                exit_code=-1,
                duration_ms=duration_ms,
            )

    def close(self) -> None:
        """No-op; workspace cleanup is handled by ToolExecutor."""
        pass

    @property
    def workspace_path(self) -> str:
        """Return the temporary workspace directory for this session."""
        return self._workspace


class LocalProcessBackend(ExecutionBackend):
    """Backend that executes tools as subprocesses with a temporary workspace."""

    name = "local_process"

    def create_session(self, policy: ExecutionPolicy) -> ExecutionSession:
        """Create a new session with a fresh temporary workspace."""
        workspace = tempfile.mkdtemp(prefix="llamagent_sandbox_")
        return LocalProcessSession(policy, workspace)

    def capabilities(self) -> dict:
        """Advertise what this backend can do."""
        return {
            "supported_runtimes": ["python", "shell"],
            "supported_isolation": ["none"],  # No real process isolation — subprocess runs with host privileges
            "supports_network_isolation": False,
            "supports_persistent_session": True,
            "available": True,
        }
