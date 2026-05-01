"""
InlineRunnerBackend: synchronous, in-process child agent execution.

The simplest backend — spawns the child agent in the current thread,
runs it to completion, and stores the result immediately. No concurrency.
"""

from __future__ import annotations

import logging
import time
import uuid

from llamagent.modules.child_agent.budget import BudgetExceededError
from llamagent.modules.child_agent.policy import ChildAgentSpec
from llamagent.modules.child_agent.runner import (
    AgentRunnerBackend,
    format_fallback_report,
    maybe_request_completion_report,
)
from llamagent.modules.child_agent.task_board import TaskRecord

logger = logging.getLogger(__name__)


def _build_metrics(elapsed: float, child=None) -> dict:
    """Build metrics dict with elapsed time and budget tracker stats if available."""
    metrics = {"elapsed_seconds": round(elapsed, 2)}
    if child is not None and hasattr(child, 'llm') and hasattr(child.llm, 'tracker'):
        t = child.llm.tracker
        metrics["tokens_used"] = t.tokens_used
        metrics["llm_calls"] = t.llm_calls
        metrics["steps_used"] = t.steps_used
    return metrics


class InlineRunnerBackend(AgentRunnerBackend):
    """
    Synchronous inline execution backend.

    Runs the child agent in the current thread. spawn() blocks until
    the child finishes, so wait() simply returns the stored result.
    """

    name = "inline"

    def __init__(self):
        self._results: dict[str, TaskRecord] = {}

    def spawn(self, spec: ChildAgentSpec, agent_factory, task_id: str | None = None) -> str:
        """
        Spawn and immediately execute a child agent inline.

        Args:
            spec: Child agent specification.
            agent_factory: Callable(spec) -> LlamAgent.
            task_id: Optional pre-generated task_id.

        Returns:
            Unique task_id for the completed execution.
        """
        task_id = task_id or uuid.uuid4().hex[:12]
        start_time = time.time()
        child = None

        try:
            child = agent_factory(spec)

            # Build the prompt: include context if provided
            prompt = spec.task
            if spec.context:
                prompt = f"Context:\n{spec.context}\n\nTask:\n{spec.task}"

            result_text = child.chat(prompt)
            # v3.5 template A: optional auto-prompt for completion report
            result_text = maybe_request_completion_report(child, result_text)
            elapsed = time.time() - start_time

            record = TaskRecord(
                task_id=task_id,
                parent_id=spec.parent_task_id,
                role=spec.role,
                task=spec.task,
                status="completed",
                result=result_text,
                history=list(child.history),
                metrics=_build_metrics(elapsed, child),
                created_at=start_time,
                completed_at=time.time(),
            )
            logger.info(
                "Child agent (%s) completed in %.1fs: %s",
                spec.role, elapsed, spec.task[:60],
            )

        except BudgetExceededError as e:
            elapsed = time.time() - start_time
            record = TaskRecord(
                task_id=task_id,
                parent_id=spec.parent_task_id,
                role=spec.role,
                task=spec.task,
                status="failed",
                result=format_fallback_report(
                    "budget exceeded", str(e), spec.runlog_path or None
                ),
                history=list(child.history) if child else [],
                metrics=_build_metrics(elapsed, child),
                created_at=start_time,
                completed_at=time.time(),
            )
            logger.warning(
                "Child agent (%s) budget exceeded after %.1fs: %s",
                spec.role, elapsed, e,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            record = TaskRecord(
                task_id=task_id,
                parent_id=spec.parent_task_id,
                role=spec.role,
                task=spec.task,
                status="failed",
                result=format_fallback_report(
                    "execution error", f"{type(e).__name__}: {e}",
                    spec.runlog_path or None,
                ),
                history=list(child.history) if child else [],
                metrics=_build_metrics(elapsed, child),
                created_at=start_time,
                completed_at=time.time(),
            )
            logger.error(
                "Child agent (%s) failed after %.1fs: %s",
                spec.role, elapsed, e,
            )

        finally:
            # v3.5: emit runlog "end" record so observers see a clean terminator
            if spec.runlog_path:
                from llamagent.core.logging_llm import append_runlog
                try:
                    append_runlog(spec.runlog_path, {
                        "ts": time.time(),
                        "kind": "end",
                        "status": record.status if record else "unknown",
                    })
                except Exception:
                    pass
            if child is not None:
                try:
                    child.shutdown()
                except Exception as shutdown_err:
                    logger.error("Child agent shutdown error: %s", shutdown_err)

        self._results[task_id] = record
        return task_id

    def wait(self, child_id: str, timeout: float | None = None) -> TaskRecord:
        """
        Return the stored result for a completed child agent.

        Inline execution is synchronous, so the result is always available
        immediately after spawn() returns.
        """
        record = self._results.get(child_id)
        if record is None:
            return TaskRecord(
                task_id=child_id,
                status="unknown",
                result="No result found for this child agent.",
            )
        return record

    def cancel(self, child_id: str) -> bool:
        """
        Attempt to cancel a child agent.

        Inline execution is synchronous — by the time we have a task_id,
        the child has already finished. Cancellation is not possible.
        """
        return False

    def status(self, child_id: str) -> str:
        """Return the status of a child agent execution."""
        record = self._results.get(child_id)
        return record.status if record else "unknown"
