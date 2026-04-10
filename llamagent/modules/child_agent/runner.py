"""
AgentRunnerBackend: abstract interface for child agent execution backends.

Concrete backends (e.g., InlineRunnerBackend) implement spawn/wait/cancel/status
to control how child agents are actually run. This abstraction allows future
backends such as threaded, async, or remote execution.
"""

from __future__ import annotations

from llamagent.modules.child_agent.policy import ChildAgentSpec
from llamagent.modules.child_agent.task_board import TaskRecord


class AgentRunnerBackend:
    """
    Abstract base class for child agent execution backends.

    Each backend defines how a child agent is spawned, waited on,
    cancelled, and queried for status.
    """

    name: str = "base"

    def spawn(self, spec: ChildAgentSpec, agent_factory, task_id: str | None = None) -> str:
        """
        Spawn a child agent to execute the given spec.

        Args:
            spec: The child agent specification (task, role, policy, etc.).
            agent_factory: Callable that takes a ChildAgentSpec and returns a LlamAgent.
            task_id: Optional pre-generated task_id. If None, the backend generates one.

        Returns:
            The task_id for the spawned child.
        """
        raise NotImplementedError("Subclasses must implement spawn()")

    def wait(self, child_id: str, timeout: float | None = None) -> TaskRecord:
        """
        Wait for a child agent to complete and return its task record.

        Args:
            child_id: The task_id returned by spawn().
            timeout: Maximum seconds to wait (None = wait indefinitely).

        Returns:
            The completed TaskRecord.
        """
        raise NotImplementedError("Subclasses must implement wait()")

    def cancel(self, child_id: str) -> bool:
        """
        Attempt to cancel a running child agent.

        Args:
            child_id: The task_id to cancel.

        Returns:
            True if cancellation was successful, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement cancel()")

    def status(self, child_id: str) -> str:
        """
        Query the current status of a child agent.

        Args:
            child_id: The task_id to query.

        Returns:
            Status string: "pending", "running", "completed", "failed", or "unknown".
        """
        raise NotImplementedError("Subclasses must implement status()")

    def shutdown(self, timeout: float = 30) -> None:
        """
        Graceful shutdown: stop all running children and clean up.

        Default is a no-op. ThreadRunnerBackend overrides to abort and join threads.
        """
        pass
