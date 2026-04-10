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
from llamagent.modules.child_agent.runner import AgentRunnerBackend
from llamagent.modules.child_agent.task_board import TaskRecord

logger = logging.getLogger(__name__)


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
            elapsed = time.time() - start_time

            record = TaskRecord(
                task_id=task_id,
                parent_id=spec.parent_task_id,
                role=spec.role,
                task=spec.task,
                status="completed",
                result=result_text,
                history=list(child.history),
                metrics={"elapsed_seconds": round(elapsed, 2)},
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
                result=f"Budget exceeded: {e}",
                history=list(child.history) if child else [],
                metrics={"elapsed_seconds": round(elapsed, 2)},
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
                result=f"Execution error: {e}",
                history=list(child.history) if child else [],
                metrics={"elapsed_seconds": round(elapsed, 2)},
                created_at=start_time,
                completed_at=time.time(),
            )
            logger.error(
                "Child agent (%s) failed after %.1fs: %s",
                spec.role, elapsed, e,
            )

        finally:
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
