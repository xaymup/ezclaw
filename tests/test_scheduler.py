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
    """After save, the file should use the 5-column
    ID/Time/Task/Status/Recurrence schema. The Recurrence cell is blank
    for a one-shot task."""
    sched.schedule("2026-06-01 09:00", "fresh")
    with open(sched.path) as f:
        content = f.read()
    assert "| ID |" in content
    assert "| Recurrence |" in content
    assert "| 1 | 2026-06-01 09:00 | fresh | Pending |  |" in content


def test_load_migrates_4column_to_5column(sched):
    """Heartbeat written before recurrence-support had 4 columns. Loading
    must accept the older shape and normalize on the next save."""
    with open(sched.path, "w") as f:
        f.write(
            "# EzClaw Heartbeat\n\n"
            "| ID | Scheduled Time | Task | Status |\n"
            "|---:|:---|:---|:---|\n"
            "| 1 | 2026-06-01 09:00 | legacy task | Pending |\n"
        )
    tasks = sched.load()
    assert len(tasks) == 1
    assert tasks[0].id == 1
    assert tasks[0].recurrence is None
    sched.save(tasks)
    with open(sched.path) as f:
        content = f.read()
    assert "| Recurrence |" in content


# ── Recurrence ──────────────────────────────────────────────────────────────

def test_recurrence_normalization_named():
    from scheduler import _normalize_recurrence
    for s in ("hourly", "DAILY", "  weekly  ", "Weekdays"):
        out = _normalize_recurrence(s)
        assert out in {"hourly", "daily", "weekly", "weekdays"}


def test_recurrence_normalization_intervals():
    from scheduler import _normalize_recurrence
    assert _normalize_recurrence("every 5m") == "every 5m"
    assert _normalize_recurrence("EVERY 30s") == "every 30s"
    assert _normalize_recurrence("every  2h") == "every 2h"
    assert _normalize_recurrence("every 1d") == "every 1d"


def test_recurrence_normalization_empty_returns_none():
    from scheduler import _normalize_recurrence
    assert _normalize_recurrence(None) is None
    assert _normalize_recurrence("") is None
    assert _normalize_recurrence("   ") is None


def test_recurrence_normalization_rejects_garbage():
    from scheduler import _normalize_recurrence
    with pytest.raises(ValueError):
        _normalize_recurrence("nonsense")
    with pytest.raises(ValueError):
        _normalize_recurrence("every 0m")  # non-positive
    with pytest.raises(ValueError):
        _normalize_recurrence("every -5m")


def test_next_occurrence_interval_skips_missed_slots():
    """If the scheduler missed several firings (laptop asleep), next
    advances past `now` rather than catching up by firing for each one."""
    from scheduler import next_occurrence
    base = datetime(2026, 6, 1, 9, 0)
    # 7 minutes past base — `every 5m` would have fired at 9:00 and 9:05;
    # next future slot is 9:10.
    now = datetime(2026, 6, 1, 9, 7)
    assert next_occurrence("every 5m", base, now=now) == datetime(2026, 6, 1, 9, 10)


def test_next_occurrence_daily():
    from scheduler import next_occurrence
    base = datetime(2026, 6, 1, 9, 0)
    now = datetime(2026, 6, 1, 10, 0)
    assert next_occurrence("daily", base, now=now) == datetime(2026, 6, 2, 9, 0)


def test_next_occurrence_weekly():
    from scheduler import next_occurrence
    base = datetime(2026, 6, 1, 9, 0)        # Monday
    now = datetime(2026, 6, 5, 9, 0)         # following Friday
    nxt = next_occurrence("weekly", base, now=now)
    assert nxt == datetime(2026, 6, 8, 9, 0)
    assert nxt.weekday() == base.weekday()


def test_next_occurrence_weekdays_skips_weekend():
    from scheduler import next_occurrence
    base = datetime(2026, 6, 5, 9, 0)        # Friday
    assert base.weekday() == 4
    now = datetime(2026, 6, 5, 9, 30)        # past Friday's slot
    nxt = next_occurrence("weekdays", base, now=now)
    assert nxt == datetime(2026, 6, 8, 9, 0) # next Monday
    assert nxt.weekday() == 0


def test_schedule_with_recurrence_persists(sched):
    t = sched.schedule("2026-06-01 09:00", "morning standup", recurrence="weekdays")
    assert t.recurrence == "weekdays"
    assert t.is_recurring
    reloaded = sched.load()
    assert reloaded[0].recurrence == "weekdays"


def test_schedule_rejects_bad_recurrence(sched):
    with pytest.raises(ValueError):
        sched.schedule("2026-06-01 09:00", "x", recurrence="every banana")


def test_complete_or_reschedule_oneshot_marks_done(sched):
    t = sched.schedule("2026-06-01 09:00", "one and done")
    result = sched.complete_or_reschedule(t.id, success=True)
    assert result.status == "Done"
    assert sched.list_pending() == []


def test_complete_or_reschedule_recurring_rearms_to_next_slot(sched):
    t = sched.schedule("2026-06-01 09:00", "ping inbox", recurrence="every 5m")
    now = datetime(2026, 6, 1, 9, 7)
    result = sched.complete_or_reschedule(t.id, success=True, now=now)
    assert result.status == "Pending"
    assert result.time == datetime(2026, 6, 1, 9, 10)
    pending = sched.list_pending()
    assert len(pending) == 1
    assert pending[0].id == t.id


def test_complete_or_reschedule_recurring_on_failure_does_not_rearm(sched):
    """Silent self-repair on failures would hide breakage. We deliberately
    mark Failed and let the user decide whether to re-arm."""
    t = sched.schedule("2026-06-01 09:00", "ping webhook", recurrence="hourly")
    result = sched.complete_or_reschedule(t.id, success=False)
    assert result.status == "Failed"
    assert sched.list_pending() == []


def test_unschedule_stops_recurring_task(sched):
    t = sched.schedule("2026-06-01 09:00", "every five", recurrence="every 5m")
    cancelled = sched.unschedule(t.id)
    assert cancelled.status == "Cancelled"
    assert sched.list_pending() == []


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
