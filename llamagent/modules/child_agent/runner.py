"""
AgentRunnerBackend: abstract interface for child agent execution backends.

Concrete backends (e.g., InlineRunnerBackend) implement spawn/wait/cancel/status
to control how child agents are actually run. This abstraction allows future
backends such as threaded, async, or remote execution.
"""

from __future__ import annotations

from llamagent.modules.child_agent.policy import ChildAgentSpec
from llamagent.modules.child_agent.task_board import TaskRecord


# v3.5 template A: the prompt the framework appends when the child's
# final reply is a free-form sentence with no Status/Summary/Artifacts
# structure. Used by runners when child_agent_report_template="auto".
COMPLETION_REPORT_REQUEST = (
    "Please conclude with a completion report in this exact format:\n"
    "Status: success | partial | failed\n"
    "Summary: <1-3 sentences describing what you did>\n"
    "Artifacts: <comma-separated absolute paths of files you created or modified, or \"none\">"
)


def maybe_request_completion_report(child, result_text: str) -> str:
    """If the child's final reply lacks the v3.5 completion-report shape AND
    the parent's config opted into ``"auto"`` template, ask the child once
    more to emit the structured form. Single retry, never recursive.

    Returns the (possibly re-asked) result text. If template != "auto" or
    the format is already present, returns the original text unchanged.
    """
    template = "system_prompt"
    try:
        template = getattr(child.config, "child_agent_report_template", "system_prompt")
    except Exception:
        pass
    if template != "auto":
        return result_text
    if "Status:" in (result_text or "") and "Summary:" in (result_text or ""):
        return result_text
    try:
        return child.chat(COMPLETION_REPORT_REQUEST)
    except Exception:
        # If the child errors trying to produce the report, keep the
        # original free-form result rather than masking with an exception.
        return result_text


def format_fallback_report(
    reason_kind: str,
    reason_detail: str,
    runlog_path: str | None = None,
) -> str:
    """Format a v3.5 fallback completion report for crashed children.

    Used when the child agent does not produce a normal completion report
    (Budget exceeded, unhandled exception, SIGKILL, timeout, etc.). Format
    is aligned with the v3.5 success-path report so the parent's model
    reads both via the same convention.

    Args:
        reason_kind: short label, e.g. "budget exceeded", "execution error",
            "killed by timeout", "max_steps reached".
        reason_detail: exception class + message, or other diagnostic text.
        runlog_path: absolute path to the child's runlog file. Surfaced only
            for human debugging; the parent agent does not act on it.

    Returns:
        Multi-line report ending with optional runlog hint.
    """
    parts = [
        "Status: failed",
        f"Summary: child {reason_kind}: {reason_detail}",
        "Artifacts: none",
    ]
    body = "\n".join(parts)
    if runlog_path:
        body += f"\n\n(See child runlog at {runlog_path} for details.)"
    return body


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
