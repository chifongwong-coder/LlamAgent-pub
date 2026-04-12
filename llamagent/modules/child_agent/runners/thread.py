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


def _summarize_task_log(task_log) -> str:
    """Summarize a ContinuousRunner task log (first 5 + last 5 entries)."""
    if not task_log:
        return "No tasks executed."
    entries = []
    show = task_log[:5]
    if len(task_log) > 10:
        entries.append(f"--- showing first 5 of {len(task_log)} entries ---")
    for e in show:
        status_mark = "OK" if e.status == "completed" else "ERR"
        entries.append(f"[{status_mark}] {e.trigger_type}: {e.input[:80]}")
    if len(task_log) > 10:
        entries.append(f"--- ... {len(task_log) - 10} entries omitted ... ---")
        for e in task_log[-5:]:
            status_mark = "OK" if e.status == "completed" else "ERR"
            entries.append(f"[{status_mark}] {e.trigger_type}: {e.input[:80]}")
    elif len(task_log) > 5:
        for e in task_log[5:]:
            status_mark = "OK" if e.status == "completed" else "ERR"
            entries.append(f"[{status_mark}] {e.trigger_type}: {e.input[:80]}")
    return "\n".join(entries)


def _build_metrics(elapsed: float, child=None) -> dict:
    """Build metrics dict with elapsed time and budget tracker stats if available."""
    metrics = {"elapsed_seconds": round(elapsed, 2)}
    if child is not None and hasattr(child, 'llm') and hasattr(child.llm, 'tracker'):
        t = child.llm.tracker
        metrics["tokens_used"] = t.tokens_used
        metrics["llm_calls"] = t.llm_calls
        metrics["steps_used"] = t.steps_used
    return metrics


class _ThreadLogCapture(logging.Handler):
    """Captures log records emitted by a specific thread."""

    def __init__(self, thread_id):
        super().__init__()
        self._thread_id = thread_id
        self.records: list[str] = []
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record):
        if record.thread == self._thread_id:
            self.records.append(self.format(record))


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
        self._continuous_runners: dict = {}  # task_id -> ContinuousRunner
        self._lock = threading.Lock()
        self._on_complete = on_complete  # callback(task_id, record)
        self._channel = None  # Set by module for MessageTrigger creation

    def __getstate__(self):
        """Support deepcopy/pickle by excluding non-picklable threading objects."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_threads", None)
        state.pop("_events", None)
        state.pop("_continuous_runners", None)
        state.pop("_channel", None)
        return state

    def __setstate__(self, state):
        """Restore state with fresh threading objects."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._threads = {}
        self._events = {}
        self._continuous_runners = {}
        self._channel = None

    def spawn(self, spec: ChildAgentSpec, agent_factory, task_id: str | None = None) -> str:
        """
        Spawn a child agent in a new thread, returning immediately.

        The Event is created before starting the thread to prevent a race
        where _run_child completes before the caller can access the event.

        Args:
            spec: Child agent specification.
            agent_factory: Callable(spec) -> LlamAgent.
            task_id: Optional pre-generated task_id.

        Returns:
            Unique task_id for the spawned child.
        """
        task_id = task_id or uuid.uuid4().hex[:12]
        event = threading.Event()
        thread = threading.Thread(
            target=self._run_child,
            args=(task_id, spec, agent_factory),
            daemon=True,
        )
        with self._lock:
            self._events[task_id] = event
            self._threads[task_id] = thread
        thread.start()
        return task_id

    def _run_child(self, task_id: str, spec: ChildAgentSpec, agent_factory):
        """Thread target: create agent, run chat() or ContinuousRunner, store result."""
        # Set thread name for log disambiguation
        threading.current_thread().name = f"child-{task_id[:8]}"

        # Attach log capture handler filtered to this thread
        capture = _ThreadLogCapture(threading.current_thread().ident)
        capture.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(capture)

        start_time = time.time()
        child = None
        record = None

        try:
            child = agent_factory(spec)
            with self._lock:
                self._agents[task_id] = child

            if spec.continuous:
                record = self._run_continuous_child(
                    task_id, spec, child, start_time,
                )
            else:
                record = self._run_short_child(
                    task_id, spec, child, start_time,
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
                result=f"Execution error: {e}",
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
            # Remove log capture handler and store captured logs
            logging.getLogger().removeHandler(capture)
            captured_logs = "\n".join(capture.records[-100:])  # Last 100 entries
            if record is not None:
                record.logs = captured_logs

            # Clean up continuous runner reference
            with self._lock:
                self._continuous_runners.pop(task_id, None)

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

    def _run_short_child(self, task_id, spec, child, start_time):
        """Run a short-lived child agent via chat()."""
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
            metrics=_build_metrics(elapsed, child),
            created_at=start_time,
            completed_at=time.time(),
        )
        logger.info(
            "Child agent (%s) completed in %.1fs: %s",
            spec.role, elapsed, spec.task[:60],
        )
        return record

    def _run_continuous_child(self, task_id, spec, child, start_time):
        """Run a continuous child agent via ContinuousRunner."""
        runner = self._create_continuous_runner(child, spec)
        with self._lock:
            self._continuous_runners[task_id] = runner

        runner.run()  # Blocks until stop() is called

        # Build record after normal exit (exceptions propagate to _run_child's handlers)
        elapsed = time.time() - start_time
        task_log = runner.get_log()
        log_summary = _summarize_task_log(task_log)
        record = TaskRecord(
            task_id=task_id,
            parent_id=spec.parent_task_id,
            role=spec.role,
            task=spec.task,
            status="completed",
            result=f"Continuous agent stopped. {len(task_log)} tasks executed.\n{log_summary}",
            history=list(child.history[-100:]),
            metrics={
                "total_tasks": len(task_log),
                "elapsed_seconds": round(elapsed, 2),
                "agent_id": child.agent_id,
            },
            created_at=start_time,
            completed_at=time.time(),
        )
        logger.info(
            "Continuous child agent (%s) stopped after %.1fs, %d tasks: %s",
            spec.role, elapsed, len(task_log), spec.task[:60],
        )
        return record

    def _create_continuous_runner(self, child, spec):
        """Create a ContinuousRunner with user trigger + MessageTrigger."""
        from llamagent.core.runner import ContinuousRunner, TimerTrigger, FileTrigger
        from llamagent.core.message_channel import MessageTrigger

        triggers = []

        # User-specified trigger
        if spec.trigger_type == "timer":
            triggers.append(TimerTrigger(spec.trigger_interval, spec.task))
        elif spec.trigger_type == "file":
            triggers.append(FileTrigger(spec.trigger_watch_dir))

        # Automatically add MessageTrigger for receiving messages
        if self._channel is not None:
            triggers.append(MessageTrigger(self._channel, child.agent_id))

        return ContinuousRunner(child, triggers, poll_interval=1.0)

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
        Cancel a running child agent.

        For continuous children: calls runner.stop() which signals the runner
        loop to exit and aborts the current chat.
        For short-task children: sets _abort flag directly.

        Acquires lock to get references, then releases before waiting on
        the event to avoid deadlock with _run_child's finally block.

        Args:
            child_id: The task_id to cancel.

        Returns:
            True if a running agent was found and stop/abort was signalled.
        """
        with self._lock:
            child = self._agents.get(child_id)
            event = self._events.get(child_id)
            runner = self._continuous_runners.get(child_id)
        if child is None:
            return False
        if runner:
            runner.stop()  # Sets _stopped + aborts current chat
        else:
            child._abort = True
        if event:
            event.wait(timeout=10)
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
        Stop all continuous runners, abort all running child agents, and join threads.

        Called during parent agent shutdown to ensure clean termination.

        Args:
            timeout: Total time budget for joining all threads.
        """
        with self._lock:
            continuous_snapshot = list(self._continuous_runners.items())
            running = list(self._agents.items())
            threads_snapshot = list(self._threads.items())
        # Stop continuous runners first (signals loop exit + abort)
        for task_id, runner in continuous_snapshot:
            try:
                runner.stop()
            except Exception as e:
                logger.error("Error stopping continuous runner %s: %s", task_id, e)
        # Abort all remaining (short-task) agents
        for task_id, child in running:
            child._abort = True
        # Join all threads
        deadline = time.time() + timeout
        for task_id, thread in threads_snapshot:
            remaining = max(0, deadline - time.time())
            thread.join(timeout=remaining)
