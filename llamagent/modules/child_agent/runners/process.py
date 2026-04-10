"""
ProcessRunnerBackend: subprocess-based child agent execution.

Spawns each child agent as an isolated subprocess via CommandRunner.start(),
communicates the task specification through a temporary JSON file, and
collects results from the subprocess stdout. API keys are inherited via
environment variables (never written to the spec file).

Each subprocess runs ``python -m llamagent.agent_runner --spec <path>``.
A dedicated monitor thread per child handles timeout, result parsing,
spec file cleanup, and the on_complete callback.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from llamagent.modules.child_agent.policy import ChildAgentSpec
from llamagent.modules.child_agent.runner import AgentRunnerBackend
from llamagent.modules.child_agent.task_board import TaskRecord
from llamagent.modules.command_runner import CommandRunner

logger = logging.getLogger(__name__)


class ProcessRunnerBackend(AgentRunnerBackend):
    """
    Subprocess-based execution backend for child agents.

    Each child agent runs in its own Python subprocess. spawn() returns
    immediately with a task_id. A monitor thread waits for the process
    to complete, parses the JSON output, and stores the result.
    """

    name = "process"

    def __init__(self, parent_config=None, on_complete=None):
        self._parent_config = parent_config
        self._processes: dict[str, subprocess.Popen] = {}
        self._results: dict[str, TaskRecord] = {}
        self._events: dict[str, threading.Event] = {}
        self._monitor_threads: dict[str, threading.Thread] = {}
        self._spec_paths: dict[str, str] = {}
        self._lock = threading.Lock()
        self._on_complete = on_complete

    def __getstate__(self):
        """Support deepcopy/pickle by excluding non-picklable objects."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_monitor_threads", None)
        state.pop("_events", None)
        state.pop("_processes", None)
        return state

    def __setstate__(self, state):
        """Restore state with fresh threading/process objects."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._monitor_threads = {}
        self._events = {}
        self._processes = {}

    def spawn(self, spec: ChildAgentSpec, agent_factory, task_id: str | None = None) -> str:
        """
        Spawn a child agent in a new subprocess.

        The agent_factory parameter is accepted for interface compatibility
        but not used — the subprocess creates its own agent from the spec file.

        Args:
            spec: Child agent specification.
            agent_factory: Unused (interface compatibility with AgentRunnerBackend).
            task_id: Optional pre-generated task_id.

        Returns:
            Unique task_id for the spawned child.
        """
        task_id = task_id or uuid.uuid4().hex[:12]

        # 1. Serialize spec to a temporary JSON file
        spec_path = self._write_spec_file(task_id, spec)

        # 2. Start the subprocess (inherit_env=True to pass API keys)
        proc = CommandRunner.start(
            cmd=[sys.executable, "-m", "llamagent.agent_runner", "--spec", spec_path],
            inherit_env=True,
        )

        # 3. Create event before starting monitor thread (prevent race)
        event = threading.Event()
        monitor = threading.Thread(
            target=self._monitor,
            args=(task_id, proc, event, spec_path, self._get_timeout(spec)),
            daemon=True,
        )
        with self._lock:
            self._processes[task_id] = proc
            self._events[task_id] = event
            self._monitor_threads[task_id] = monitor
            self._spec_paths[task_id] = spec_path
        monitor.start()
        return task_id

    def _write_spec_file(self, task_id: str, spec: ChildAgentSpec) -> str:
        """Serialize spec + parent config to a temporary JSON file.

        API keys are NEVER written to the spec file — they are inherited
        via environment variables (inherit_env=True in CommandRunner.start).
        """
        spec_dir = tempfile.mkdtemp(prefix="llamagent_proc_")
        spec_path = os.path.join(spec_dir, f"{task_id}.json")

        config_data = {
            "model": self._parent_config.model if self._parent_config else "auto",
            "max_react_steps": (
                spec.policy.budget.max_steps
                if spec.policy and spec.policy.budget and spec.policy.budget.max_steps
                else 5
            ),
            "react_timeout": (
                spec.policy.budget.max_time_seconds
                if spec.policy and spec.policy.budget and spec.policy.budget.max_time_seconds
                else 60
            ),
            "system_prompt": spec.system_prompt or f"You are a {spec.role}.",
            "project_dir": getattr(self._parent_config, "project_dir", None) if self._parent_config else None,
            "playground_dir": getattr(self._parent_config, "playground_dir", None) if self._parent_config else None,
        }

        data = {
            "task": spec.task,
            "role": spec.role,
            "context": spec.context,
            "config": config_data,
            "tool_allowlist": spec.policy.tool_allowlist if spec.policy else None,
            "budget": {
                "max_llm_calls": spec.policy.budget.max_llm_calls,
                "max_time_seconds": spec.policy.budget.max_time_seconds,
            } if spec.policy and spec.policy.budget else None,
        }

        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return spec_path

    def _get_timeout(self, spec: ChildAgentSpec) -> float:
        """Derive monitor timeout from budget (budget + 30s grace period)."""
        if spec.policy and spec.policy.budget and spec.policy.budget.max_time_seconds:
            return spec.policy.budget.max_time_seconds + 30
        return 330  # Default: 300 + 30

    def _monitor(self, task_id: str, proc: subprocess.Popen,
                 event: threading.Event, spec_path: str, timeout: float):
        """Monitor thread: wait for process, parse result, cleanup, signal completion."""
        record = None
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            output = json.loads(stdout)
            record = TaskRecord(
                task_id=task_id,
                status=output.get("status", "failed"),
                result=output.get("result", ""),
                history=output.get("history", []),
                metrics=output.get("metrics", {}),
                logs=stderr[:5000] if stderr else "",
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()  # Drain pipes, reap process
            record = TaskRecord(
                task_id=task_id,
                status="failed",
                result="Process timed out",
            )
        except (json.JSONDecodeError, Exception) as e:
            # Include subprocess stderr in error for debugging
            stderr_text = ""
            try:
                stderr_text = stderr[:500] if stderr else ""
            except NameError:
                pass
            error_msg = f"Process error: {e}"
            if stderr_text:
                error_msg += f"\nstderr: {stderr_text}"
            record = TaskRecord(
                task_id=task_id,
                status="failed",
                result=error_msg,
            )
        finally:
            # Cleanup spec file
            try:
                os.unlink(spec_path)
                os.rmdir(os.path.dirname(spec_path))
            except OSError:
                pass

            with self._lock:
                if record is not None:
                    self._results[task_id] = record
                self._events[task_id].set()
                self._processes.pop(task_id, None)
                self._spec_paths.pop(task_id, None)

            # Invoke callback outside lock to avoid deadlocks
            if self._on_complete and record is not None:
                try:
                    self._on_complete(task_id, record)
                except Exception as cb_err:
                    logger.error("on_complete callback error: %s", cb_err)

    def wait(self, child_id: str, timeout: float | None = None) -> TaskRecord:
        """
        Block until the specified child agent process completes.

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
        Cancel a running child agent by terminating its process.

        Sends SIGTERM first and waits for the monitor to detect exit.
        If still alive after 10 seconds, sends SIGKILL.

        Args:
            child_id: The task_id to cancel.

        Returns:
            True if a running process was found and termination was signalled.
        """
        with self._lock:
            proc = self._processes.get(child_id)
            event = self._events.get(child_id)
        if proc is None:
            return False
        try:
            proc.terminate()  # SIGTERM
        except OSError:
            pass  # Process already exited
        if event:
            event.wait(timeout=10)
        # If still alive, force kill
        try:
            proc.kill()
        except OSError:
            pass
        return True

    def status(self, child_id: str) -> str:
        """
        Query the current status of a child agent.

        Returns:
            "running" if the process is still executing,
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
        Terminate all running child processes and join monitor threads.

        Called during parent agent shutdown to ensure clean termination.
        Also cleans up any remaining spec files.

        Args:
            timeout: Total time budget for joining all monitor threads.
        """
        # Terminate all running processes
        with self._lock:
            running_procs = list(self._processes.items())
        for task_id, proc in running_procs:
            try:
                proc.terminate()
            except OSError:
                pass

        # Join all monitor threads
        deadline = time.time() + timeout
        with self._lock:
            monitors_snapshot = list(self._monitor_threads.items())
        for task_id, thread in monitors_snapshot:
            remaining = max(0, deadline - time.time())
            thread.join(timeout=remaining)

        # Cleanup any remaining spec files
        with self._lock:
            remaining_specs = list(self._spec_paths.items())
        for task_id, spec_path in remaining_specs:
            try:
                os.unlink(spec_path)
                os.rmdir(os.path.dirname(spec_path))
            except OSError:
                pass
