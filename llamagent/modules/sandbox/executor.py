"""
ToolExecutor: the bridge between tool invocations and sandbox backends.

For tools without an execution_policy (or with isolation="none") the executor
calls the tool function directly on the host — same behaviour as v1.1.

For tools with a sandbox policy the executor resolves a backend, manages
sessions (one-shot or task-scoped), and returns structured observations.

v1.5.1: Added run_command() for direct shell command execution via backend,
used by JobModule. Separate from execute() which routes tool functions.
"""

from __future__ import annotations

import shutil

from llamagent.modules.sandbox.backend import ExecutionSession, ExecutionSpec
from llamagent.modules.sandbox.resolver import BackendResolver


class ToolExecutor:
    """Executes tool invocations, optionally routing through a sandbox backend."""

    def __init__(self, resolver: BackendResolver) -> None:
        self.resolver = resolver
        self.current_task_id: str | None = None  # Set by the execution strategy
        self._sessions: dict[str, ExecutionSession] = {}
        self._managed_workspaces: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, tool_info: dict, args: dict, agent=None) -> str:
        """
        Execute a tool described by *tool_info* with the given *args*.

        Args:
            tool_info: Tool descriptor dict. Expected keys:
                - func: callable (the tool function)
                - name: str (tool name)
                - execution_policy: ExecutionPolicy | None
                - takes_agent: bool (v3.6) — when True and ``agent`` is
                  provided, inject ``agent`` as first positional arg.
            args: Keyword arguments to pass to the tool.
            agent: The CALLING agent. Forwarded to the tool when
                ``tool_info["takes_agent"]`` is True. None during sandbox
                isolation paths (the subprocess / container has its own
                agent identity), or when the caller doesn't need to pass
                identity (legacy-compatible).

        Returns:
            A string observation suitable for the ReAct loop.
        """
        from llamagent.modules.sandbox.policy import ExecutionPolicy

        policy: ExecutionPolicy | None = tool_info.get("execution_policy")

        # No sandbox — direct host execution (v1.1 compatible path).
        if policy is None or policy.isolation == "none":
            func = tool_info["func"]
            # v3.6: takes_agent dispatch — when the tool opts in AND the
            # caller forwarded an agent reference, inject it.
            if tool_info.get("takes_agent") and agent is not None:
                result = func(agent, **args)
            else:
                result = func(**args)
            return str(result) if result is not None else ""

        # Sandbox execution.
        backend = self.resolver.resolve(policy)

        # Determine session key for task-scoped sessions.
        # Key includes task_id + backend + tool name to avoid cross-tool session sharing.
        session_key: str | None = None
        if self.current_task_id and policy.session_mode == "task_session":
            tool_name = tool_info.get("name", "_unknown")
            session_key = f"{self.current_task_id}:{backend.name}:{tool_name}"

        session = self._sessions.get(session_key) if session_key else None

        if session is None:
            session = backend.create_session(policy)
            if session.workspace_path:
                self._managed_workspaces.append(session.workspace_path)
            if session_key:
                self._sessions[session_key] = session

        spec = ExecutionSpec(
            command=tool_info.get("name", ""),
            args=args,
            policy=policy,
            workspace_path=session.workspace_path,
        )

        try:
            result = session.run(spec)
            return result.to_observation()
        finally:
            # One-shot sessions are closed immediately after execution.
            if policy.session_mode == "one_shot":
                session.close()

    def run_command(self, command: str, cwd: str, timeout: float = 300) -> str:
        """
        Execute a shell command directly via CommandRunner.

        Unlike execute(), this method is purpose-built for shell command execution.
        It bypasses the backend/session layer entirely, running the command with
        environment isolation in the given cwd.

        Used by JobModule for start_job.

        Args:
            command: Shell command to execute.
            cwd: Working directory (absolute path).
            timeout: Maximum execution time in seconds.

        Returns:
            A string observation (stdout + stderr) suitable for the ReAct loop.
        """
        from llamagent.modules.command_runner import CommandRunner
        from llamagent.modules.sandbox.backend import ExecutionResult

        result = CommandRunner.run(
            cmd=["sh", "-c", command],
            cwd=cwd,
            timeout=timeout,
            env=CommandRunner.build_safe_env(),
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            timed_out=result.timed_out,
        ).to_observation()

    def close_task_sessions(self, task_id: str) -> None:
        """Close all sessions associated with a specific task."""
        prefix = f"{task_id}:"
        keys_to_remove = [k for k in self._sessions if k.startswith(prefix)]
        for key in keys_to_remove:
            session = self._sessions.pop(key)
            session.close()

    def shutdown(self) -> None:
        """Close all sessions and clean up managed workspaces."""
        # Close all active sessions.
        for session in self._sessions.values():
            try:
                session.close()
            except Exception:
                pass
        self._sessions.clear()

        # Remove managed workspace directories.
        for workspace in self._managed_workspaces:
            shutil.rmtree(workspace, ignore_errors=True)
        self._managed_workspaces.clear()
