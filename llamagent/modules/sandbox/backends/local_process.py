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

import tempfile

from llamagent.modules.command_runner import CommandRunner
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

        # Delegate execution to CommandRunner.
        env = CommandRunner.build_safe_env(spec.env_vars)
        result = CommandRunner.run(
            cmd=cmd,
            cwd=self._workspace,
            timeout=policy.timeout_seconds,
            env=env,
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            timed_out=result.timed_out,
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
