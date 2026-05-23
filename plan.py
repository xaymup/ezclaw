"""Pure data model for the agent's task queue.

This module is import-time only — no I/O, no rendering, no LLM calls.
It exports `Task` and `Plan` dataclasses plus a `TASK_STATUSES` tuple
documenting the valid status values. The `Plan` class is the canonical
mutator for task status; the rest of the codebase reads `Plan` snapshots
but does not edit task fields directly.
"""

from dataclasses import dataclass
from typing import Optional


TASK_STATUSES = ("pending", "in_progress", "done", "failed", "skipped")


@dataclass
class Task:
    id: int                # 1-based, stable across plan lifetime
    description: str       # short user-facing line
    status: str = "pending"


@dataclass
class Plan:
    title: str
    tasks: list             # list[Task]
    current_task_id: Optional[int] = None

    def get_task(self, task_id: int) -> Optional[Task]:
        return next((t for t in self.tasks if t.id == task_id), None)

    def advance(self, task_id: int, status: str) -> None:
        """Transition a task to a new status. Invalid status or unknown id is a no-op.
        Already-finished tasks (done/skipped) cannot be regressed to in_progress."""
        if status not in TASK_STATUSES:
            return
        task = self.get_task(task_id)
        if task is None:
            return
        # Don't regress completed work
        if task.status in ("done", "skipped") and status == "in_progress":
            return
        task.status = status
        if status == "in_progress":
            self.current_task_id = task_id
        elif status in ("done", "failed", "skipped") and self.current_task_id == task_id:
            self.current_task_id = None

    def insert(self, after_id: int, description: str) -> Task:
        """Insert a new task after the given task id. If after_id is unknown,
        appends to the end. Returns the new task. The new id is `max(existing) + 1`."""
        new_id = max((t.id for t in self.tasks), default=0) + 1
        new_task = Task(id=new_id, description=description)
        idx = next((i for i, t in enumerate(self.tasks) if t.id == after_id), None)
        if idx is None:
            self.tasks.append(new_task)
        else:
            self.tasks.insert(idx + 1, new_task)
        return new_task

    def is_complete(self) -> bool:
        return all(t.status in ("done", "skipped") for t in self.tasks)

    def progress(self):
        """Return (done_or_skipped_count, total_count)."""
        done = sum(1 for t in self.tasks if t.status in ("done", "skipped"))
        return done, len(self.tasks)
