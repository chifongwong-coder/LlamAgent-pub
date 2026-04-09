"""
ThreadRunnerBackend: concurrent child agent execution using threads.

Spawns each child agent in a daemon thread, returns immediately with a task_id.
The parent agent can later wait for results, cancel running children, or collect
all completed results. Thread-safe via a single lock protecting internal dicts.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from llamagent.modules.child_agent.budget import BudgetExceededError
from llamagent.modules.child_agent.policy import ChildAgentSpec
from llamagent.modules.child_agent.runner import AgentRunnerBackend
from llamagent.modules.child_agent.task_board import TaskRecord

logger = logging.getLogger(__name__)


class ThreadRunnerBackend(AgentRunnerBackend):
    """
    Concurrent thread-based execution backend.

    Each child agent runs in its own daemon thread. spawn() returns immediately
    with a task_id. Use wait() to block until a child finishes, or status()
    to poll without blocking.
    """

    name = "thread"

    def __init__(self, on_complete=None):
        self._threads: dict[str, threading.Thread] = {}
        self._agents: dict[str, object] = {}  # task_id -> LlamAgent
        self._results: dict[str, TaskRecord] = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._on_complete = on_complete  # callback(task_id, record)

    def __getstate__(self):
        """Support deepcopy/pickle by excluding non-picklable threading objects."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_threads", None)
        state.pop("_events", None)
        return state

    def __setstate__(self, state):
        """Restore state with fresh threading objects."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._threads = {}
        self._events = {}

    def spawn(self, spec: ChildAgentSpec, agent_factory) -> str:
        """
        Spawn a child agent in a new thread, returning immediately.

        The Event is created before starting the thread to prevent a race
        where _run_child completes before the caller can access the event.

        Args:
            spec: Child agent specification.
            agent_factory: Callable(spec) -> LlamAgent.

        Returns:
            Unique task_id for the spawned child.
        """
        task_id = uuid.uuid4().hex[:12]
        event = threading.Event()
        with self._lock:
            self._events[task_id] = event

        thread = threading.Thread(
            target=self._run_child,
            args=(task_id, spec, agent_factory),
            daemon=True,
        )
        with self._lock:
            self._threads[task_id] = thread
        thread.start()
        return task_id

    def _run_child(self, task_id: str, spec: ChildAgentSpec, agent_factory):
        """Thread target: create agent, run chat(), store result, fire callback."""
        start_time = time.time()
        child = None
        record = None

        try:
            child = agent_factory(spec)
            with self._lock:
                self._agents[task_id] = child

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

            # Store result and signal completion under lock
            with self._lock:
                if record is not None:
                    self._results[task_id] = record
                self._events[task_id].set()
                self._agents.pop(task_id, None)

            # Invoke callback outside lock to avoid deadlocks
            if self._on_complete and record is not None:
                try:
                    self._on_complete(task_id, record)
                except Exception as cb_err:
                    logger.error("on_complete callback error: %s", cb_err)

    def wait(self, child_id: str, timeout: float | None = None) -> TaskRecord:
        """
        Block until the specified child agent completes.

        Args:
            child_id: The task_id to wait for.
            timeout: Maximum seconds to wait (None = indefinite).

        Returns:
            The completed TaskRecord, or an "unknown" record if not found.
        """
        with self._lock:
            event = self._events.get(child_id)
        if event is None:
            return TaskRecord(
                task_id=child_id,
                status="unknown",
                result="No child agent found.",
            )
        event.wait(timeout=timeout)
        with self._lock:
            return self._results.get(
                child_id,
                TaskRecord(task_id=child_id, status="unknown"),
            )

    def cancel(self, child_id: str) -> bool:
        """
        Cancel a running child agent by setting its _abort flag.

        Acquires lock to get the agent reference, then releases before
        waiting on the event to avoid deadlock with _run_child's finally block.

        Args:
            child_id: The task_id to cancel.

        Returns:
            True if a running agent was found and abort was signalled.
        """
        with self._lock:
            child = self._agents.get(child_id)
            event = self._events.get(child_id)
        if child is None:
            return False
        child._abort = True
        if event:
            event.wait(timeout=5.0)
        return True

    def status(self, child_id: str) -> str:
        """
        Query the current status of a child agent.

        Returns:
            "running" if the child is still executing,
            the record's status if completed/failed,
            or "unknown" if the task_id is not recognized.
        """
        with self._lock:
            record = self._results.get(child_id)
            if record:
                return record.status
            if child_id in self._events:
                return "running"
        return "unknown"

    def shutdown(self, timeout: float = 30):
        """
        Abort all running child agents and join their threads.

        Called during parent agent shutdown to ensure clean termination.

        Args:
            timeout: Total time budget for joining all threads.
        """
        with self._lock:
            running = list(self._agents.items())
        for task_id, child in running:
            child._abort = True
        deadline = time.time() + timeout
        for task_id, thread in list(self._threads.items()):
            remaining = max(0, deadline - time.time())
            thread.join(timeout=remaining)
