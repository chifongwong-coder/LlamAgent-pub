"""
Continuous mode runner: external component that drives agent.chat() via triggers.

ContinuousRunner: manages trigger sources, polls for input, calls agent.chat().
Trigger: abstract base class for trigger sources (timer, file, queue, etc.).

The runner does NOT modify agent internals. It calls agent.chat() and agent.abort()
through the public API. Agent does not know about the runner.

Dependency direction: Runner -> Agent -> Engine. Trigger has no reference to Agent.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Empty
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from llamagent.core.agent import LlamAgent

logger = logging.getLogger(__name__)


@dataclass
class TaskLogEntry:
    """Single task execution record for ContinuousRunner."""
    trigger_type: str        # trigger class name (e.g. "TimerTrigger")
    input: str               # task input from trigger.poll()
    output: str              # agent.chat() response
    status: str              # "completed" | "error"
    error: str | None        # error message if failed
    started_at: float        # time.time() at start
    duration: float          # elapsed seconds


class Trigger(ABC):
    """Trigger source: produces input strings, unaware of the agent."""

    interruptible: bool = True       # Can high-priority inject abort this trigger's task?
    on_interrupt: str = "discard"    # "discard" or "retry" when interrupted

    @abstractmethod
    def poll(self) -> str | None:
        """Return input string if triggered, None otherwise."""
        ...


class ContinuousRunner:
    """
    Continuous mode runner: manages trigger sources, drives agent.chat().

    Features:
    - Sequential trigger polling in a loop
    - Optional application-level task timeout with watchdog
    - on_timeout: "abort" (default) calls agent.abort(); callable for custom behavior
    - Graceful stop via stop() from any thread

    Usage:
        runner = ContinuousRunner(agent, [TimerTrigger(60, "check health")])
        # In main thread:
        runner.run()  # blocks until stop() is called
        # From another thread:
        runner.stop()
    """

    def __init__(
        self,
        agent: LlamAgent,
        triggers: list[Trigger],
        *,
        poll_interval: float = 1.0,
        task_timeout: float = 0,
        on_timeout: str | Callable = "abort",
    ):
        """
        Args:
            agent: LlamAgent instance (must be in continuous mode)
            triggers: List of Trigger instances to poll
            poll_interval: Seconds between poll cycles (default 1.0)
            task_timeout: Max seconds per task; 0 = no timeout (default 0)
            on_timeout: "abort" to call agent.abort(), or a callable for custom behavior
        """
        self.agent = agent
        self.triggers = triggers
        self.poll_interval = poll_interval
        self.task_timeout = task_timeout
        self.on_timeout = on_timeout
        self._stopped = threading.Event()
        self.task_log: list[TaskLogEntry] = []
        # Priority scheduling queues (v2.9.5)
        self._urgent_queue: queue.Queue = queue.Queue()    # High priority inject
        self._normal_queue: queue.Queue = queue.Queue()    # Low priority inject
        self._retry_queue: queue.Queue = queue.Queue()     # Medium: interrupted tasks
        self._wakeup = threading.Event()                   # Wake runner from sleep
        self._running = False                              # True while run() is active
        self._task_running = False                         # True while _run_task executes
        self._current_interruptible = True                 # Current task interruptible?
        self._aborted_by_inject = False                    # Runner-internal abort tracking

    def inject(self, message: str, immediate: bool = False) -> str:
        """
        Inject a message and wait for response. Thread-safe.

        Args:
            message: Input message for the agent.
            immediate: If True, interrupt current task (unless non-interruptible).

        Returns:
            Agent's response string.

        Raises:
            RuntimeError: If runner is stopped or not running.
        """
        if self._stopped.is_set():
            raise RuntimeError("Runner is stopped")
        if not self._running:
            raise RuntimeError("Runner is not running")

        event = threading.Event()
        result: list = [None]
        item = (message, event, result)

        if immediate:
            self._urgent_queue.put(item)
            if self._task_running and self._current_interruptible:
                self._aborted_by_inject = True
                self.agent.abort()
        else:
            self._normal_queue.put(item)

        self._wakeup.set()  # Wake runner from sleep
        event.wait()
        return result[0]

    def run(self) -> None:
        """Main loop. Blocks until stop() is called."""
        self._running = True
        try:
            logger.info("ContinuousRunner started (%d triggers)", len(self.triggers))
            while not self._stopped.is_set():
                self._wakeup.clear()

                # 1. High: urgent inject
                self._drain_queue(self._urgent_queue)

                # 2. Medium: retry queue (interrupted tasks)
                self._drain_queue(self._retry_queue)

                # 3. Medium: poll triggers
                for trigger in self.triggers:
                    if self._stopped.is_set():
                        break
                    task_input = trigger.poll()
                    if task_input:
                        self._run_task(task_input, trigger)

                # 4. Low: normal inject (check urgent between items)
                self._drain_queue(self._normal_queue, check_urgent=True)

                # 5. Sleep (wakeup event can interrupt)
                self._wakeup.wait(self.poll_interval)

            logger.info("ContinuousRunner stopped")
        finally:
            self._running = False
            self._release_waiting_callers()

    def _drain_queue(self, q: queue.Queue, check_urgent: bool = False) -> None:
        """Process all messages in a queue."""
        while not self._stopped.is_set():
            # When processing low-priority, yield to urgent if it arrives
            if check_urgent and not self._urgent_queue.empty():
                break
            try:
                message, event, result = q.get_nowait()
            except Empty:
                break
            try:
                result[0] = self.agent.chat(message)
            except Exception as e:
                result[0] = f"Error: {e}"
            finally:
                if event is not None:
                    event.set()

    def _run_task(self, task_input: str, trigger: Trigger) -> None:
        """Execute one task with optional timeout, record to task_log."""
        self._current_interruptible = getattr(trigger, "interruptible", True)
        self._task_running = True

        entry = TaskLogEntry(
            trigger_type=type(trigger).__name__,
            input=task_input, output="",
            status="completed", error=None,
            started_at=time.time(), duration=0,
        )
        start = time.time()
        try:
            if self.task_timeout <= 0:
                entry.output = self.agent.chat(task_input)
            else:
                # Resolve timeout action
                if callable(self.on_timeout):
                    timeout_action = self.on_timeout
                else:
                    timeout_action = self.agent.abort
                # Watchdog timer
                timer = threading.Timer(self.task_timeout, timeout_action)
                timer.start()
                try:
                    entry.output = self.agent.chat(task_input)
                finally:
                    timer.cancel()
        except Exception as e:
            entry.status = "error"
            entry.error = str(e)
            logger.error("Task failed: %s", e)
        finally:
            self._task_running = False
            self._current_interruptible = True

        entry.duration = time.time() - start

        # Detect interrupt by inject (runner-internal tracking, no magic strings)
        was_interrupted = self._aborted_by_inject
        self._aborted_by_inject = False

        if was_interrupted:
            entry.status = "interrupted"
            on_interrupt = getattr(trigger, "on_interrupt", "discard")
            if on_interrupt == "retry":
                # Re-queue with None event (no external caller waiting)
                self._retry_queue.put((task_input, None, [None]))

        self.task_log.append(entry)

    def stop(self) -> None:
        """Signal the runner to stop. Wake up all waiting inject callers."""
        self._stopped.set()
        self.agent.abort()
        self._wakeup.set()  # Wake from sleep
        self._release_waiting_callers()

    def _release_waiting_callers(self) -> None:
        """Wake up all pending inject() callers with stop message."""
        for q in (self._urgent_queue, self._normal_queue, self._retry_queue):
            while True:
                try:
                    message, event, result = q.get_nowait()
                    if event is not None:
                        result[0] = "Runner stopped."
                        event.set()
                except Empty:
                    break

    def get_log(self) -> list[TaskLogEntry]:
        """Return a copy of the task log."""
        return list(self.task_log)

    def clear_log(self) -> None:
        """Clear the task log."""
        self.task_log.clear()


# ======================================================================
# Built-in Trigger implementations
# ======================================================================


class TimerTrigger(Trigger):
    """Fires a fixed input string at regular intervals.

    Usage:
        TimerTrigger(interval=60, message="check system health")
        # Every 60 seconds, poll() returns "check system health"
    """

    def __init__(self, interval: float, message: str, *,
                 interruptible: bool = True, on_interrupt: str = "discard"):
        self.interval = interval
        self.message = message
        self.interruptible = interruptible
        self.on_interrupt = on_interrupt
        self._last_fire: float | None = None  # None = first poll initializes

    def poll(self) -> str | None:
        import time
        now = time.time()
        if self._last_fire is None:
            self._last_fire = now
            return None
        if now - self._last_fire >= self.interval:
            self._last_fire = now
            return self.message
        return None


class FileTrigger(Trigger):
    """Fires when new files appear in a watched directory.

    Returns a message containing the new file names. Only detects additions,
    not modifications or deletions.

    Usage:
        FileTrigger("/path/to/inbox", message_template="Process files: {files}")
    """

    def __init__(self, watch_dir: str, *, message_template: str = "New files detected: {files}",
                 interruptible: bool = True, on_interrupt: str = "discard"):
        self.watch_dir = watch_dir
        self.message_template = message_template
        self.interruptible = interruptible
        self.on_interrupt = on_interrupt
        self._known_files: set[str] | None = None  # None = first poll initializes

    def poll(self) -> str | None:
        import os
        try:
            current = set(os.listdir(self.watch_dir))
        except OSError:
            return None

        if self._known_files is None:
            # First poll: snapshot current state, don't fire
            self._known_files = current
            return None

        new_files = current - self._known_files
        self._known_files = current  # Always update (track deletions too)
        if new_files:
            return self.message_template.format(files=", ".join(sorted(new_files)))
        return None
