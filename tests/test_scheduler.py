"""Tests for the Scheduler — heartbeat.md persistence + state transitions."""

import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import Scheduler, ScheduledTask, VALID_STATUSES


@pytest.fixture
def sched(tmp_path):
    return Scheduler(path=str(tmp_path / "heartbeat.md"))


# ── Schema constants ────────────────────────────────────────────────────────

def test_valid_statuses_exact():
    assert set(VALID_STATUSES) == {"Pending", "Notified", "Done", "Failed", "Cancelled"}


# ── Empty file ──────────────────────────────────────────────────────────────

def test_load_returns_empty_when_file_missing(sched):
    assert sched.load() == []


def test_list_pending_empty_when_no_tasks(sched):
    assert sched.list_pending() == []


# ── Schedule + load round-trip ──────────────────────────────────────────────

def test_schedule_assigns_id_starting_from_1(sched):
    t = sched.schedule("2026-06-01 09:00", "morning ritual")
    assert t.id == 1
    assert t.description == "morning ritual"
    assert t.status == "Pending"


def test_schedule_increments_id(sched):
    a = sched.schedule("2026-06-01 09:00", "first")
    b = sched.schedule("2026-06-01 10:00", "second")
    c = sched.schedule("2026-06-01 11:00", "third")
    assert (a.id, b.id, c.id) == (1, 2, 3)


def test_schedule_persists_across_reload(sched):
    sched.schedule("2026-06-01 09:00", "persistent")
    fresh = Scheduler(path=sched.path)
    tasks = fresh.load()
    assert len(tasks) == 1
    assert tasks[0].description == "persistent"


def test_schedule_rejects_bad_time(sched):
    with pytest.raises(ValueError):
        sched.schedule("not-a-time", "x")


# ── Unschedule ──────────────────────────────────────────────────────────────

def test_unschedule_cancels_pending_task(sched):
    t = sched.schedule("2026-06-01 09:00", "to be cancelled")
    result = sched.unschedule(t.id)
    assert result is not None
    assert result.status == "Cancelled"
    tasks = sched.load()
    assert tasks[0].status == "Cancelled"


def test_unschedule_unknown_id_returns_none(sched):
    assert sched.unschedule(999) is None


def test_unschedule_already_done_task_returns_none(sched):
    t = sched.schedule("2026-06-01 09:00", "x")
    sched.mark_status(t.id, "Done")
    # Trying to cancel a Done task is a no-op
    assert sched.unschedule(t.id) is None
    assert sched.load()[0].status == "Done"


# ── find_due ────────────────────────────────────────────────────────────────

def test_find_due_returns_past_pending_tasks(sched):
    now = datetime.now()
    past = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    future = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    sched.schedule(past, "past task")
    sched.schedule(future, "future task")
    due = sched.find_due()
    assert len(due) == 1
    assert due[0].description == "past task"


def test_find_due_excludes_already_notified(sched):
    now = datetime.now()
    past = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    t = sched.schedule(past, "past")
    sched.mark_status(t.id, "Notified")
    assert sched.find_due() == []


def test_find_due_excludes_cancelled(sched):
    now = datetime.now()
    past = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    t = sched.schedule(past, "past")
    sched.unschedule(t.id)
    assert sched.find_due() == []


# ── mark_status ─────────────────────────────────────────────────────────────

def test_mark_status_validates(sched):
    t = sched.schedule("2026-06-01 09:00", "x")
    assert sched.mark_status(t.id, "Done") is True
    assert sched.mark_status(t.id, "Garbage") is False


def test_mark_status_unknown_id(sched):
    assert sched.mark_status(999, "Done") is False


# ── Legacy schema migration ─────────────────────────────────────────────────

def test_load_migrates_old_3column_format(sched):
    """heartbeat.md from before this change had no ID column. Loading must
    synthesize IDs so the file gets normalized on the next save."""
    with open(sched.path, "w") as f:
        f.write(
            "# EzClaw Heartbeat\n\n"
            "| Scheduled Time | Task | Status |\n"
            "| :--- | :--- | :--- |\n"
            "| 2026-06-01 09:00 | old task A | Pending |\n"
            "| 2026-06-01 10:00 | old task B | Notified |\n"
        )
    tasks = sched.load()
    assert len(tasks) == 2
    assert tasks[0].id == 1
    assert tasks[1].id == 2
    assert tasks[0].description == "old task A"
    assert tasks[1].status == "Notified"


def test_save_normalizes_to_new_schema(sched):
    """After save, the file should use the 4-column ID/Time/Task/Status schema."""
    sched.schedule("2026-06-01 09:00", "fresh")
    with open(sched.path) as f:
        content = f.read()
    assert "| ID |" in content
    assert "| 1 | 2026-06-01 09:00 | fresh | Pending |" in content


# ── list_pending ────────────────────────────────────────────────────────────

def test_list_pending_excludes_terminal_states(sched):
    sched.schedule("2026-06-01 09:00", "a")           # Pending
    b = sched.schedule("2026-06-01 10:00", "b")       # → Notified
    c = sched.schedule("2026-06-01 11:00", "c")       # → Done
    d = sched.schedule("2026-06-01 12:00", "d")       # → Cancelled
    sched.mark_status(b.id, "Notified")
    sched.mark_status(c.id, "Done")
    sched.unschedule(d.id)
    pending = sched.list_pending()
    descs = {t.description for t in pending}
    assert descs == {"a", "b"}  # Pending and Notified are both "active"
