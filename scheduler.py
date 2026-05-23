"""Persistent task scheduler backed by `heartbeat.md`.

heartbeat.md is the canonical source of truth for "what does ezclaw need
to do later?" The file lives at the project root, is human-readable
markdown, and survives across sessions. This module is the only thing
that should write to it from code — agents call into it via tools
(`schedule_task`, `unschedule_task`, `list_scheduled_tasks`); the CLI's
heartbeat monitor reads from it every 30s to find due tasks.

Schema (4 columns, ID-first):
    | ID | Scheduled Time | Task | Status |
    |---:|:--- |:--- |:--- |
    | 1 | 2026-06-01 09:00 | description | Pending |

Status lifecycle:
    Pending → Notified (when fired but agent didn't auto-run)
    Pending → Done | Failed (when auto-execute completes)
    Pending → Cancelled (via unschedule_task)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

# heartbeat.md lives at the project root, NOT inside ./workspace/.
# It's a top-level state file, not part of the agent's sandboxed work area.
HEARTBEAT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "heartbeat.md",
)

VALID_STATUSES = ("Pending", "Notified", "Done", "Failed", "Cancelled")
_TERMINAL_STATUSES = frozenset({"Done", "Failed", "Cancelled"})


@dataclass
class ScheduledTask:
    id: int
    time: datetime
    description: str
    status: str = "Pending"

    @property
    def time_str(self) -> str:
        return self.time.strftime("%Y-%m-%d %H:%M")

    def to_row(self) -> str:
        return f"| {self.id} | {self.time_str} | {self.description} | {self.status} |"


class Scheduler:
    """Read/write heartbeat.md as a list of ScheduledTask. All mutations
    go through `schedule`, `unschedule`, or `mark_status` — never edit the
    file directly from outside this class."""

    def __init__(self, path: str = HEARTBEAT_PATH):
        self.path = path

    # ── Load / save ─────────────────────────────────────────────────────────

    def load(self) -> List[ScheduledTask]:
        """Parse heartbeat.md. Tolerant of the old 3-column schema (no ID)
        — IDs are synthesized in-memory so the next save() rewrites the
        file in the new schema."""
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            lines = f.readlines()

        tasks: List[ScheduledTask] = []
        next_synth_id = 1
        max_seen_id = 0

        for raw in lines:
            line = raw.strip()
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if not parts:
                continue
            # Skip table header rows: column names or alignment markers.
            first = parts[0].lower()
            if first in ("id", "scheduled time", "task", "status"):
                continue
            if all(set(p) <= set("-: ") for p in parts):
                continue

            try:
                if len(parts) >= 4:
                    # New schema: ID | Time | Description | Status
                    task_id = int(parts[0])
                    time = datetime.strptime(parts[1], "%Y-%m-%d %H:%M")
                    desc = parts[2]
                    status = parts[3] if parts[3] in VALID_STATUSES else "Pending"
                elif len(parts) == 3:
                    # Old schema: Time | Description | Status — synthesize ID
                    task_id = next_synth_id
                    next_synth_id += 1
                    time = datetime.strptime(parts[0], "%Y-%m-%d %H:%M")
                    desc = parts[1]
                    status = parts[2] if parts[2] in VALID_STATUSES else "Pending"
                else:
                    continue
            except (ValueError, IndexError):
                continue

            tasks.append(ScheduledTask(id=task_id, time=time, description=desc, status=status))
            max_seen_id = max(max_seen_id, task_id)
            if next_synth_id <= max_seen_id:
                next_synth_id = max_seen_id + 1

        return tasks

    def save(self, tasks: List[ScheduledTask]) -> None:
        lines = [
            "# EzClaw Heartbeat (Scheduled Tasks)",
            "",
            "| ID | Scheduled Time | Task | Status |",
            "|---:|:---|:---|:---|",
        ]
        for t in sorted(tasks, key=lambda x: x.id):
            lines.append(t.to_row())
        with open(self.path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ── Mutations ───────────────────────────────────────────────────────────

    def schedule(self, time_str: str, description: str) -> ScheduledTask:
        """Add a new task. Raises ValueError if `time_str` doesn't parse."""
        time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        tasks = self.load()
        next_id = max((t.id for t in tasks), default=0) + 1
        new_task = ScheduledTask(id=next_id, time=time, description=description)
        tasks.append(new_task)
        self.save(tasks)
        return new_task

    def unschedule(self, task_id: int) -> Optional[ScheduledTask]:
        """Cancel a Pending task. Returns the task on success, None if the
        id is unknown or the task is already in a terminal state."""
        tasks = self.load()
        for t in tasks:
            if t.id == task_id and t.status == "Pending":
                t.status = "Cancelled"
                self.save(tasks)
                return t
        return None

    def mark_status(self, task_id: int, status: str) -> bool:
        if status not in VALID_STATUSES:
            return False
        tasks = self.load()
        for t in tasks:
            if t.id == task_id:
                t.status = status
                self.save(tasks)
                return True
        return False

    # ── Queries ─────────────────────────────────────────────────────────────

    def find_due(self, now: Optional[datetime] = None) -> List[ScheduledTask]:
        """Pending tasks whose scheduled time has arrived."""
        now = now or datetime.now()
        return [t for t in self.load() if t.status == "Pending" and t.time <= now]

    def list_pending(self) -> List[ScheduledTask]:
        """Tasks that haven't reached a terminal state — Pending or Notified."""
        return [t for t in self.load() if t.status not in _TERMINAL_STATUSES]
