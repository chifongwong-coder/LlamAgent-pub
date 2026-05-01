"""
TaskBoard: centralized task tracking for child agent executions.

TaskRecord:  Data structure for a single task's lifecycle (pending -> running -> completed/failed).
TaskBoard:   Registry that creates, updates, queries, and collects task records.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TaskRecord:
    """Tracks the full lifecycle of a child agent task."""

    task_id: str
    parent_id: str | None = None
    role: str = ""
    task: str = ""
    status: str = "pending"
    result: str = ""
    artifacts: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    input_snapshot: dict = field(default_factory=dict)
    logs: str = ""
    created_at: float = 0
    completed_at: float = 0


class TaskBoard:
    """
    Centralized task registry for tracking child agent executions.

    Provides CRUD operations on TaskRecords and parent-scoped queries
    for listing children and collecting completed results.
    """

    def __init__(self):
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def __getstate__(self):
        """Support pickling/deepcopy by excluding the non-picklable lock."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        """Restore state and recreate the lock."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def create(self, task_id: str, parent_id: str | None = None, **kwargs) -> TaskRecord:
        """
        Create a new task record.

        Args:
            task_id: Unique task identifier.
            parent_id: Parent task/agent identifier for scoping.
            **kwargs: Additional TaskRecord fields to set.

        Returns:
            The newly created TaskRecord.
        """
        if "created_at" not in kwargs:
            kwargs["created_at"] = time.time()
        record = TaskRecord(
            task_id=task_id,
            parent_id=parent_id,
            **kwargs,
        )
        with self._lock:
            self._tasks[task_id] = record
        return record

    def update(self, task_id: str, **kwargs) -> None:
        """
        Update fields on an existing task record.

        Args:
            task_id: The task to update.
            **kwargs: Fields to update (e.g., status="completed", result="...").
                ``metrics`` has merge semantics: the supplied dict is merged
                into the existing record's metrics rather than replacing it,
                so partial writes (e.g. early ``agent_id`` from the factory,
                later ``elapsed_seconds`` from the runner) compose without
                clobbering each other.

        Raises:
            KeyError: If the task_id does not exist.
        """
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise KeyError(f"Task '{task_id}' not found on the task board")
            from dataclasses import replace
            patch = {k: v for k, v in kwargs.items() if hasattr(record, k)}
            # Merge metrics rather than replace so partial writers can
            # compose. Other fields keep replace semantics.
            if "metrics" in patch and isinstance(patch["metrics"], dict):
                merged = dict(record.metrics or {})
                merged.update(patch["metrics"])
                patch["metrics"] = merged
            self._tasks[task_id] = replace(record, **patch)

    def get(self, task_id: str) -> TaskRecord | None:
        """Look up a task record by ID. Returns a snapshot (safe for concurrent reads)."""
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            from dataclasses import replace
            return replace(record)

    def children_of(self, parent_id: str) -> list[TaskRecord]:
        """Return snapshot copies of all task records belonging to a given parent."""
        from dataclasses import replace
        with self._lock:
            return [replace(r) for r in self._tasks.values() if r.parent_id == parent_id]

    def collect_results(self, parent_id: str) -> list[TaskRecord]:
        """
        Collect snapshot copies of completed/failed children of a parent.

        Returns:
            List of TaskRecords with status in ("completed", "failed").
        """
        from dataclasses import replace
        with self._lock:
            return [
                replace(r) for r in self._tasks.values()
                if r.parent_id == parent_id and r.status in ("completed", "failed")
            ]
