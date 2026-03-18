"""
SandboxModule: pluggable module that adds sandbox execution to the Agent.

On attach the module:
1. Creates a BackendResolver and registers available backends.
2. Creates a ToolExecutor and injects it into the agent as agent.tool_executor.
3. Optionally auto-assigns sandbox policies to high-risk tools (safety_level >= 3).

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
                to any registered tool with safety_level >= 3 that does not
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
                    tool.get("safety_level", 1) >= 3
                    and tool.get("execution_policy") is None
                ):
                    tool["execution_policy"] = POLICY_LOCAL_SUBPROCESS

    def on_shutdown(self) -> None:
        """Release all sandbox sessions and clean up workspaces."""
        if self.executor:
            self.executor.shutdown()
