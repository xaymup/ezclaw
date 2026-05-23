"""Tests for the current_datetime tool exposed to agents."""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import current_datetime


def test_current_datetime_returns_iso_and_schedule_lines():
    out = current_datetime()
    # The ISO line is what most consumers will parse
    assert "ISO format:" in out
    iso_match = re.search(r"ISO format:\s+(\S+)", out)
    assert iso_match, f"no ISO line in: {out!r}"
    iso = iso_match.group(1)
    # YYYY-MM-DDTHH:MM:SS
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", iso)


def test_current_datetime_includes_scheduling_format():
    """The schedule_task tool accepts 'YYYY-MM-DD HH:MM' format — the
    datetime tool must surface that exact format so the agent can compose
    `schedule_task(<output>, ...)` without parsing."""
    out = current_datetime()
    assert "For scheduling (schedule_task format):" in out
    m = re.search(r"For scheduling \(schedule_task format\):\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", out)
    assert m, f"schedule_task format line malformed in: {out!r}"


def test_current_datetime_includes_weekday():
    """Day-of-week is useful when the user says 'next Monday' or 'tomorrow'."""
    out = current_datetime()
    # English weekday names appear in parens after the date
    weekday_names = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    assert any(d in out for d in weekday_names), f"no weekday in: {out!r}"
