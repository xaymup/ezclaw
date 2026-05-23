"""Tests for the actions table and Database methods."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import Database


def test_actions_table_exists(tmp_db):
    """The actions table is created on Database init."""
    import sqlite3
    with sqlite3.connect(tmp_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='actions'"
        )
        row = cursor.fetchone()
    assert row is not None, "actions table was not created"


def test_actions_table_has_expected_columns(tmp_db):
    """The actions table has the columns from the spec."""
    import sqlite3
    with sqlite3.connect(tmp_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(actions)")
        cols = {row[1] for row in cursor.fetchall()}
    expected = {
        "id", "session_id", "tool", "args_json", "summary",
        "why", "outcome", "error_excerpt", "embedding", "created_at",
    }
    assert expected <= cols, f"missing columns: {expected - cols}"
