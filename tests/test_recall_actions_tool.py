"""Tests for the recall_actions tool and session contextvar plumbing."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools
from memory import Database
from tools import MUTATING_TOOLS, create_action_tracking_tools, registry, set_session_context


def test_mutating_tools_set_has_expected_members():
    assert "apply_diff" in MUTATING_TOOLS
    assert "write_file" in MUTATING_TOOLS
    assert "run_shell" in MUTATING_TOOLS
    assert "schedule_task" in MUTATING_TOOLS
    assert "unschedule_task" in MUTATING_TOOLS
    # Reads must be excluded.
    assert "read_file" not in MUTATING_TOOLS
    assert "web_search" not in MUTATING_TOOLS
    assert "recall" not in MUTATING_TOOLS


def test_create_action_tracking_tools_registers_recall_actions(tmp_db):
    create_action_tracking_tools(tmp_db)
    assert "recall_actions" in registry.tools


def test_recall_actions_returns_no_match_message_when_empty(tmp_db, monkeypatch):
    """With no actions in the session, recall_actions says so."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0])

    create_action_tracking_tools(tmp_db)
    sid = tmp_db.create_session("t")
    set_session_context(sid)

    out = registry.tools["recall_actions"]("auth")
    assert "No matching actions" in out


def test_recall_actions_formats_rows_with_summary_why_outcome(tmp_db, monkeypatch):
    """Output contains summary, why, and outcome for each row."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0, 0.0])

    create_action_tracking_tools(tmp_db)
    sid = tmp_db.create_session("t")
    set_session_context(sid)

    import pickle
    vec = pickle.dumps([1.0, 0.0, 0.0])
    tmp_db.add_action(
        session_id=sid, tool="apply_diff",
        args_json='{"path": "auth.py"}',
        summary="edited auth.py",
        why="fix the token storage compliance issue",
        outcome="succeeded",
        error_excerpt=None,
        embedding=vec,
    )

    out = registry.tools["recall_actions"]("auth", limit=5)
    assert "edited auth.py" in out
    assert "fix the token storage compliance issue" in out
    assert "succeeded" in out


def test_recall_actions_filters_by_current_session(tmp_db, monkeypatch):
    """recall_actions only reads rows for the session set via set_session_context."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0])

    create_action_tracking_tools(tmp_db)
    sid_a = tmp_db.create_session("a")
    sid_b = tmp_db.create_session("b")

    import pickle
    vec = pickle.dumps([1.0, 0.0])
    tmp_db.add_action(
        session_id=sid_a, tool="apply_diff", args_json="{}",
        summary="edited a-side.py", why=None, outcome="succeeded",
        error_excerpt=None, embedding=vec,
    )
    tmp_db.add_action(
        session_id=sid_b, tool="apply_diff", args_json="{}",
        summary="edited b-side.py", why=None, outcome="succeeded",
        error_excerpt=None, embedding=vec,
    )

    set_session_context(sid_a)
    out = registry.tools["recall_actions"]("edited")
    assert "a-side.py" in out
    assert "b-side.py" not in out


def test_recall_actions_returns_error_when_no_session_set(tmp_db):
    """If set_session_context was never called, recall_actions must not crash."""
    create_action_tracking_tools(tmp_db)
    set_session_context(None)
    out = registry.tools["recall_actions"]("anything")
    assert isinstance(out, str)
    assert len(out) > 0
