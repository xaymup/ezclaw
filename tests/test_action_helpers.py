"""Unit tests for the pure action-tracking helpers."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_tracking import classify_outcome, extract_why, summarize_action


# ── classify_outcome ────────────────────────────────────────────────────────

def test_classify_outcome_error_prefix():
    outcome, excerpt = classify_outcome("Error: file not found")
    assert outcome == "failed"
    assert excerpt == "Error: file not found"


def test_classify_outcome_traceback():
    out = "Traceback (most recent call last):\n  File ...\nValueError: bad"
    outcome, excerpt = classify_outcome(out)
    assert outcome == "failed"
    assert excerpt.startswith("Traceback")


def test_classify_outcome_non_zero_exit():
    outcome, excerpt = classify_outcome("ran command\nexit code 1")
    assert outcome == "partial"
    assert excerpt is not None


def test_classify_outcome_success():
    outcome, excerpt = classify_outcome("Wrote 42 bytes to foo.txt")
    assert outcome == "succeeded"
    assert excerpt is None


def test_classify_outcome_error_excerpt_truncated_to_300():
    big = "Error: " + ("x" * 1000)
    _, excerpt = classify_outcome(big)
    assert len(excerpt) == 300


# ── extract_why ─────────────────────────────────────────────────────────────

def test_extract_why_returns_last_sentence():
    text = "Let me check the file. I'll fix the auth bug."
    assert extract_why(text) == "I'll fix the auth bug."


def test_extract_why_single_sentence():
    assert extract_why("Fixing the import.") == "Fixing the import."


def test_extract_why_empty_returns_none():
    assert extract_why("") is None
    assert extract_why("   \n  ") is None


def test_extract_why_truncated_to_200():
    long_sentence = "Because " + ("very " * 100) + "important."
    why = extract_why(long_sentence)
    assert why is not None
    assert len(why) <= 200


# ── summarize_action ────────────────────────────────────────────────────────

def test_summarize_apply_diff_uses_basename():
    out = summarize_action("apply_diff", {"path": "/abs/path/auth.py", "old": "x", "new": "y"})
    assert out == "edited auth.py"


def test_summarize_write_file_uses_basename():
    out = summarize_action("write_file", {"path": "src/foo/bar.py", "content": "..."})
    assert out == "wrote bar.py"


def test_summarize_run_shell_truncates_command():
    out = summarize_action("run_shell", {"command": "pytest -k auth_test --maxfail=2"})
    assert out.startswith("ran: pytest -k auth_test")
    assert len(out) <= 70  # "ran: " + 60 chars


def test_summarize_schedule_task():
    out = summarize_action("schedule_task", {"scheduled_time": "2026-01-01T10:00", "description": "deploy"})
    assert out == "scheduled: deploy"


def test_summarize_unschedule_task():
    out = summarize_action("unschedule_task", {"task_id": 7})
    assert out == "unscheduled task 7"


def test_summarize_unknown_tool_falls_back_to_name():
    out = summarize_action("mystery_tool", {"k": "v"})
    assert out == "called mystery_tool"
