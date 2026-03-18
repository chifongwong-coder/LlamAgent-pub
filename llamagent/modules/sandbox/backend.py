"""
Core abstractions for sandbox execution.

ExecutionSpec   — what to run and under which policy.
ExecutionResult — structured result of an execution (stdout, stderr, timing, etc.).
ExecutionSession — a stateful execution context created by a backend.
ExecutionBackend — factory that creates sessions for a given policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llamagent.modules.sandbox.policy import ExecutionPolicy


# ---------------------------------------------------------------------------
# ExecutionSpec
# ---------------------------------------------------------------------------

@dataclass
class ExecutionSpec:
    """Fully-resolved specification for a single execution request."""

    command: str
    args: dict
    policy: ExecutionPolicy
    workspace_path: str | None = None
    env_vars: dict | None = None


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Structured result returned by an ExecutionSession."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    artifacts: list[str] = field(default_factory=list)
    duration_ms: float = 0
    peak_memory_mb: float = 0
    timed_out: bool = False
    truncated: bool = False

    @property
    def success(self) -> bool:
        """True if the execution completed normally with exit code 0."""
        return self.exit_code == 0 and not self.timed_out

    def to_observation(self) -> str:
        """Convert to a tool-observation string suitable for the ReAct loop."""
        parts: list[str] = []

        if self.timed_out:
            parts.append(f"[TIMEOUT after {self.duration_ms:.0f}ms]")

        if self.stdout:
            parts.append(self.stdout)

        if self.stderr:
            parts.append(f"[stderr] {self.stderr}")

        if self.artifacts:
            parts.append(f"[artifacts] {', '.join(self.artifacts)}")

        if not parts:
            if self.success:
                return "(no output)"
            return f"[exit_code={self.exit_code}]"

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# ExecutionSession (abstract)
# ---------------------------------------------------------------------------

class ExecutionSession:
    """
    A stateful execution context created by a backend.

    For one_shot policies the session is created and closed per invocation.
    For task_session policies the session persists across invocations within
    the same task and is closed when the task ends.
    """

    def run(self, spec: ExecutionSpec) -> ExecutionResult:
        """Execute a spec and return the result."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources held by this session."""
        raise NotImplementedError

    @property
    def workspace_path(self) -> str | None:
        """Return the workspace directory for this session, if any."""
        return None


# ---------------------------------------------------------------------------
# ExecutionBackend (abstract)
# ---------------------------------------------------------------------------

class ExecutionBackend:
    """
    Factory that creates ExecutionSessions for a given policy.

    Concrete backends (local_process, docker, firecracker, etc.) subclass
    this and advertise their capabilities so the BackendResolver can pick
    the best match.
    """

    name: str = "base"

    def create_session(self, policy: ExecutionPolicy) -> ExecutionSession:
        """Create a new session that satisfies the given policy."""
        raise NotImplementedError

    def capabilities(self) -> dict:
        """
        Return a capability descriptor used by BackendResolver.

        Expected keys:
            supported_runtimes: list[str]
            supported_isolation: list[str]
            supports_network_isolation: bool
            supports_persistent_session: bool
            available: bool
        """
        raise NotImplementedError
