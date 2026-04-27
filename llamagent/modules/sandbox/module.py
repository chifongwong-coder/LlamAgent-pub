"""
SandboxModule: pluggable module that adds sandbox execution to the Agent.

On attach the module:
1. Creates a BackendResolver and registers available backends.
2. Creates a ToolExecutor and injects it into the agent as agent.tool_executor.
3. Optionally auto-assigns sandbox policies to tools with side effects (safety_level >= 2).

On shutdown the module cleans up all sessions and temporary workspaces.
"""

from __future__ import annotations

from llamagent.core.agent import Module
from llamagent.modules.sandbox.executor import ToolExecutor
from llamagent.modules.sandbox.policy import POLICY_LOCAL_SUBPROCESS
from llamagent.modules.sandbox.resolver import BackendResolver


class SandboxModule(Module):
    """Sandbox execution for high-risk tools."""

    name = "sandbox"
    description = "Sandbox execution for high-risk tools"

    def __init__(self, auto_assign: bool = True) -> None:
        """
        Args:
            auto_assign: When True, automatically assign POLICY_LOCAL_SUBPROCESS
                to any registered tool with safety_level >= 2 that does not
                already have an execution_policy.
        """
        self.auto_assign = auto_assign
        self.executor: ToolExecutor | None = None
        self.resolver: BackendResolver | None = None

    def on_attach(self, agent) -> None:
        """Initialize the sandbox subsystem and inject executor into the agent."""
        super().on_attach(agent)

        # Build resolver with available backends.
        self.resolver = BackendResolver()

        from llamagent.modules.sandbox.backends.local_process import (
            LocalProcessBackend,
        )

        self.resolver.register(LocalProcessBackend())

        # Create executor and inject into agent.
        self.executor = ToolExecutor(self.resolver)
        agent.tool_executor = self.executor

        # Auto-assign sandbox policies to high-risk tools.
        if self.auto_assign:
            for name, tool in agent._tools.items():
                if (
                    tool.get("safety_level", 1) >= 2
                    and tool.get("execution_policy") is None
                ):
                    tool["execution_policy"] = POLICY_LOCAL_SUBPROCESS

        # v3.3: register the `command` shell tool. Provides a generic
        # escape hatch for path operations (mv / cp / rm / find / grep /
        # stat) that v3.3 deliberately took out of the typed surface.
        # The tool's safety classification is owned by the authorization
        # engine's `_evaluate_command` (shlex tokenize + structural
        # pattern match against config.hooks_config["command_safety"]
        # rules). When that classifier returns ASK, the agent's
        # InteractivePolicy prompts the user; when it returns
        # HARD_DENY, the call is blocked outright.
        self._register_command_tool(agent)

    def _register_command_tool(self, agent) -> None:
        import json as _json
        import subprocess as _subprocess

        def _command(cmd: str, cwd: str | None = None, timeout: int = 30) -> str:
            """Run a shell command. Cwd defaults to write_root.

            v3.3 D7: snapshot is captured eagerly at agent init when
            enabled — no per-call trigger needed.
            """
            effective_cwd = cwd or getattr(agent, "write_root", None) or agent.project_dir
            try:
                proc = _subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=timeout, cwd=effective_cwd,
                )
                return _json.dumps({
                    "status": "success" if proc.returncode == 0 else "error",
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }, ensure_ascii=False)
            except _subprocess.TimeoutExpired as e:
                return _json.dumps({
                    "status": "error",
                    "error": f"Command timed out after {timeout}s",
                    "stdout": e.stdout or "", "stderr": e.stderr or "",
                }, ensure_ascii=False)
            except Exception as e:
                return _json.dumps({
                    "status": "error",
                    "error": str(e),
                }, ensure_ascii=False)

        agent.register_tool(
            name="command", func=_command,
            description=(
                "Run a shell command. Default cwd is the project's write "
                "boundary. Use this for path operations (mv / cp / rm / "
                "find / grep / stat) and other one-shot shell tasks. "
                "Destructive patterns (rm -rf /, dd if=/dev/*, "
                "git push --force, etc.) are auto-classified by the "
                "authorization engine and either blocked or sent to the "
                "user for confirmation."
            ),
            parameters={"type": "object", "properties": {
                "cmd": {"type": "string", "description": "Shell command line"},
                "cwd": {"type": "string", "description": "Working directory (defaults to project write boundary)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)", "default": 30},
            }, "required": ["cmd"]},
            tier="common",
            safety_level=2,
            action="execute",
        )

    def on_shutdown(self) -> None:
        """Release all sandbox sessions and clean up workspaces."""
        if self.executor:
            self.executor.shutdown()
