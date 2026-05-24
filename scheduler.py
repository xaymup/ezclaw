"""Persistent task scheduler backed by `heartbeat.md`.

heartbeat.md is the canonical source of truth for "what does ezclaw need
to do later?" The file lives at the project root, is human-readable
markdown, and survives across sessions. This module is the only thing
that should write to it from code — agents call into it via tools
(`schedule_task`, `unschedule_task`, `list_scheduled_tasks`); the CLI's
heartbeat monitor reads from it every 30s to find due tasks.

Schema (5 columns, ID-first):
    | ID | Scheduled Time | Task | Status | Recurrence |
    |---:|:--- |:--- |:--- |:--- |
    | 1 | 2026-06-01 09:00 | description | Pending |            |
    | 2 | 2026-06-01 09:00 | check the inbox | Pending | hourly |

Recurrence (column 5) is empty for one-shot tasks. Supported formats:
    every Nm | every Nh | every Nd     — fixed interval (e.g. "every 5m")
    hourly                              — same MM every hour
    daily                               — same HH:MM every day
    weekly                              — same weekday + HH:MM each week
    weekdays                            — Mon–Fri at the same HH:MM

When a recurring task fires successfully, the row stays in the table —
its time is advanced to the next valid occurrence and status returns to
Pending. To stop a recurring task, `unschedule_task` it.

Status lifecycle (one-shot):
    Pending → Notified (when fired) → Done | Failed (after run)
    Pending → Cancelled (via unschedule_task)

Status lifecycle (recurring):
    Pending → Notified (when fired) → Pending (after run, re-armed for
    the next occurrence)
    Pending → Cancelled (via unschedule_task)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

# heartbeat.md lives at the project root, NOT inside ./workspace/.
# It's a top-level state file, not part of the agent's sandboxed work area.
HEARTBEAT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "heartbeat.md",
)

VALID_STATUSES = ("Pending", "Notified", "Done", "Failed", "Cancelled")
_TERMINAL_STATUSES = frozenset({"Done", "Failed", "Cancelled"})


# ── Recurrence ──────────────────────────────────────────────────────────────

_INTERVAL_RE = re.compile(r"^\s*every\s+(\d+)\s*([smhd])\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_NAMED_RECURRENCES = frozenset({"hourly", "daily", "weekly", "weekdays"})


def _normalize_recurrence(spec: Optional[str]) -> Optional[str]:
    """Return the spec in its canonical lowercase form, or None if empty/
    unrecognized. Raises ValueError if non-empty but unparseable so the
    tool layer can surface a clear error to the agent."""
    if not spec or not spec.strip():
        return None
    s = spec.strip().lower()
    if s in _NAMED_RECURRENCES:
        return s
    m = _INTERVAL_RE.match(s)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if n <= 0:
            raise ValueError(f"recurrence interval must be positive (got {spec!r})")
        return f"every {n}{unit}"
    raise ValueError(
        f"unrecognized recurrence {spec!r}. Use 'every Nm' / 'every Nh' / "
        "'every Nd' or one of: hourly, daily, weekly, weekdays."
    )


def _advance_once(spec: str, when: datetime) -> datetime:
    """Advance `when` by one increment of the recurrence rule."""
    if spec == "hourly":
        return when + timedelta(hours=1)
    if spec == "daily":
        return when + timedelta(days=1)
    if spec == "weekly":
        return when + timedelta(weeks=1)
    if spec == "weekdays":
        # Advance one day; skip Sat (5) and Sun (6).
        nxt = when + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        return nxt
    m = _INTERVAL_RE.match(spec)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        return when + timedelta(seconds=n * _UNIT_SECONDS[unit])
    raise ValueError(f"unknown recurrence {spec!r}")


def next_occurrence(spec: str, base: datetime, now: Optional[datetime] = None) -> datetime:
    """First occurrence strictly AFTER `now`, walking forward from `base`.

    If a recurring task missed firings (laptop was asleep, ezclaw was
    closed for hours), `next_occurrence` skips past the missed slots so
    the task fires once on resume and re-arms for the next future slot —
    same semantics as cron, deliberately not "catch up by firing N times".
    """
    now = now or datetime.now()
    candidate = base
    # Safety bound — a tight interval (e.g. every 1m) crossed by a long
    # outage shouldn't spin forever; advance until past `now`.
    for _ in range(100_000):
        if candidate > now:
            return candidate
        candidate = _advance_once(spec, candidate)
    # Fallback: refuse to spin further. Caller will see the task as due
    # at the current candidate which is at or before `now` — still fires
    # once, then re-armed on the next pass.
    return candidate


@dataclass
class ScheduledTask:
    id: int
    time: datetime
    description: str
    status: str = "Pending"
    # Empty / None for one-shot tasks. Stored as the canonical lowercase
    # form returned by _normalize_recurrence.
    recurrence: Optional[str] = None

    @property
    def time_str(self) -> str:
        return self.time.strftime("%Y-%m-%d %H:%M")

    @property
    def is_recurring(self) -> bool:
        return bool(self.recurrence)

    def to_row(self) -> str:
        rec = self.recurrence or ""
        return f"| {self.id} | {self.time_str} | {self.description} | {self.status} | {rec} |"


class Scheduler:
    """Read/write heartbeat.md as a list of ScheduledTask. All mutations
    go through `schedule`, `unschedule`, `mark_status`, or
    `complete_or_reschedule` — never edit the file directly from outside
    this class."""

    def __init__(self, path: str = HEARTBEAT_PATH):
        self.path = path

    # ── Load / save ─────────────────────────────────────────────────────────

    def load(self) -> List[ScheduledTask]:
        """Parse heartbeat.md.

        Tolerant of three historical schemas — synthesizes IDs and a
        blank recurrence when missing so the next save() rewrites the
        file in the latest form:
            3 cols:  Time | Description | Status
            4 cols:  ID | Time | Description | Status
            5 cols:  ID | Time | Description | Status | Recurrence
        """
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
            if first in ("id", "scheduled time", "task", "status", "recurrence"):
                continue
            if all(set(p) <= set("-: ") for p in parts):
                continue

            try:
                recurrence: Optional[str] = None
                if len(parts) >= 5:
                    # Latest schema with recurrence column.
                    task_id = int(parts[0])
                    time = datetime.strptime(parts[1], "%Y-%m-%d %H:%M")
                    desc = parts[2]
                    status = parts[3] if parts[3] in VALID_STATUSES else "Pending"
                    rec_raw = parts[4] or None
                    try:
                        recurrence = _normalize_recurrence(rec_raw)
                    except ValueError:
                        recurrence = None
                elif len(parts) == 4:
                    # 4-col legacy: ID | Time | Description | Status
                    task_id = int(parts[0])
                    time = datetime.strptime(parts[1], "%Y-%m-%d %H:%M")
                    desc = parts[2]
                    status = parts[3] if parts[3] in VALID_STATUSES else "Pending"
                elif len(parts) == 3:
                    # 3-col legacy: Time | Description | Status — synth ID.
                    task_id = next_synth_id
                    next_synth_id += 1
                    time = datetime.strptime(parts[0], "%Y-%m-%d %H:%M")
                    desc = parts[1]
                    status = parts[2] if parts[2] in VALID_STATUSES else "Pending"
                else:
                    continue
            except (ValueError, IndexError):
                continue

            tasks.append(ScheduledTask(
                id=task_id, time=time, description=desc, status=status,
                recurrence=recurrence,
            ))
            max_seen_id = max(max_seen_id, task_id)
            if next_synth_id <= max_seen_id:
                next_synth_id = max_seen_id + 1

        return tasks

    def save(self, tasks: List[ScheduledTask]) -> None:
        lines = [
            "# EzClaw Heartbeat (Scheduled Tasks)",
            "",
            "| ID | Scheduled Time | Task | Status | Recurrence |",
            "|---:|:---|:---|:---|:---|",
        ]
        for t in sorted(tasks, key=lambda x: x.id):
            lines.append(t.to_row())
        with open(self.path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ── Mutations ───────────────────────────────────────────────────────────

    def schedule(
        self,
        time_str: str,
        description: str,
        recurrence: Optional[str] = None,
    ) -> ScheduledTask:
        """Add a new task.

        Raises ValueError on a malformed `time_str` or an unrecognized
        `recurrence` (see _normalize_recurrence)."""
        time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        normalized = _normalize_recurrence(recurrence)
        tasks = self.load()
        next_id = max((t.id for t in tasks), default=0) + 1
        new_task = ScheduledTask(
            id=next_id, time=time, description=description,
            recurrence=normalized,
        )
        tasks.append(new_task)
        self.save(tasks)
        return new_task

    def unschedule(self, task_id: int) -> Optional[ScheduledTask]:
        """Cancel a Pending / Notified task. Returns the task on success,
        None if the id is unknown or the task is already in a terminal
        state. Cancelling a recurring task stops all future firings."""
        tasks = self.load()
        for t in tasks:
            if t.id == task_id and t.status not in _TERMINAL_STATUSES:
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

    def complete_or_reschedule(
        self,
        task_id: int,
        success: bool = True,
        now: Optional[datetime] = None,
    ) -> Optional[ScheduledTask]:
        """Called by the heartbeat monitor when a fired task finishes.

        For one-shot tasks: marks status Done or Failed (existing
        behavior, equivalent to `mark_status(task_id, "Done"/"Failed")`).

        For recurring tasks on SUCCESS: rolls the task forward to its
        next occurrence and resets status to Pending. The same row stays
        in heartbeat.md — recurring rules are one row, not one per fire.

        For recurring tasks on FAILURE: marks Failed (terminal). The user
        decides whether to re-arm. We deliberately do NOT auto-retry a
        failing recurring task — silent self-repair would hide real
        breakage (e.g. a webhook the user moved).
        """
        tasks = self.load()
        for t in tasks:
            if t.id != task_id:
                continue
            if not success:
                t.status = "Failed"
                self.save(tasks)
                return t
            if not t.is_recurring:
                t.status = "Done"
                self.save(tasks)
                return t
            # Recurring + success: roll forward.
            try:
                t.time = next_occurrence(t.recurrence, t.time, now=now)
                t.status = "Pending"
            except ValueError:
                # Rule somehow became invalid — fall back to Done so the
                # task doesn't hammer in a tight loop.
                t.status = "Done"
            self.save(tasks)
            return t
        return None

    # ── Queries ─────────────────────────────────────────────────────────────

    def find_due(self, now: Optional[datetime] = None) -> List[ScheduledTask]:
        """Pending tasks whose scheduled time has arrived."""
        now = now or datetime.now()
        return [t for t in self.load() if t.status == "Pending" and t.time <= now]

    def list_pending(self) -> List[ScheduledTask]:
        """Tasks that haven't reached a terminal state — Pending or Notified."""
        return [t for t in self.load() if t.status not in _TERMINAL_STATUSES]
