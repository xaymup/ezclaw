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


# ── add_action / search_actions ─────────────────────────────────────────────


def _make_session(db) -> int:
    return db.create_session("test")


def test_add_action_roundtrips_basic_fields(tmp_db):
    sid = _make_session(tmp_db)
    tmp_db.add_action(
        session_id=sid,
        tool="apply_diff",
        args_json='{"path": "foo.py"}',
        summary="edited foo.py",
        why="fixing auth bug",
        outcome="succeeded",
        error_excerpt=None,
        embedding=None,
    )
    import sqlite3
    with sqlite3.connect(tmp_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, tool, summary, why, outcome FROM actions WHERE session_id=?",
            (sid,),
        )
        row = cursor.fetchone()
    assert row == (sid, "apply_diff", "edited foo.py", "fixing auth bug", "succeeded")


def test_add_action_persists_error_excerpt_and_embedding(tmp_db):
    sid = _make_session(tmp_db)
    import pickle
    fake_vec = [0.1, 0.2, 0.3]
    tmp_db.add_action(
        session_id=sid,
        tool="run_shell",
        args_json='{"command": "pytest"}',
        summary="ran: pytest",
        why="verify fix",
        outcome="failed",
        error_excerpt="AssertionError: bad",
        embedding=pickle.dumps(fake_vec),
    )
    import sqlite3
    with sqlite3.connect(tmp_db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT error_excerpt, embedding FROM actions WHERE session_id=?", (sid,)
        )
        excerpt, blob = cursor.fetchone()
    assert excerpt == "AssertionError: bad"
    assert pickle.loads(blob) == fake_vec


def test_search_actions_filters_by_session(tmp_db, monkeypatch):
    """search_actions only returns rows for the requested session."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0, 0.0])

    sid_a = _make_session(tmp_db)
    sid_b = _make_session(tmp_db)
    import pickle
    vec = pickle.dumps([1.0, 0.0, 0.0])
    tmp_db.add_action(
        session_id=sid_a, tool="apply_diff", args_json="{}",
        summary="edited auth.py", why=None, outcome="succeeded",
        error_excerpt=None, embedding=vec,
    )
    tmp_db.add_action(
        session_id=sid_b, tool="apply_diff", args_json="{}",
        summary="edited other.py", why=None, outcome="succeeded",
        error_excerpt=None, embedding=vec,
    )
    rows = tmp_db.search_actions(session_id=sid_a, query="auth")
    assert len(rows) == 1
    assert rows[0]["summary"] == "edited auth.py"


def test_search_actions_returns_empty_when_no_rows(tmp_db, monkeypatch):
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0])
    sid = _make_session(tmp_db)
    assert tmp_db.search_actions(session_id=sid, query="anything") == []


def test_search_actions_ranks_by_similarity(tmp_db, monkeypatch):
    """search_actions returns higher-similarity rows first."""
    import embed as embed_mod

    vecs = {
        "auth": [1.0, 0.0, 0.0],
        "unrelated": [0.0, 1.0, 0.0],
        "partial": [0.7, 0.3, 0.0],
    }

    def fake_embed(text: str):
        if "auth" in text:
            return vecs["auth"]
        if "unrelated" in text:
            return vecs["unrelated"]
        if "partial" in text:
            return vecs["partial"]
        return vecs["auth"]

    monkeypatch.setattr(embed_mod, "embed", fake_embed)

    sid = _make_session(tmp_db)
    import pickle
    for summary, key in [
        ("edited auth.py", "auth"),
        ("edited unrelated.py", "unrelated"),
        ("edited partial.py", "partial"),
    ]:
        tmp_db.add_action(
            session_id=sid, tool="apply_diff", args_json="{}",
            summary=summary, why=None, outcome="succeeded",
            error_excerpt=None, embedding=pickle.dumps(vecs[key]),
        )
    rows = tmp_db.search_actions(session_id=sid, query="auth", limit=3, threshold=0.0)
    summaries = [r["summary"] for r in rows]
    assert summaries[0] == "edited auth.py"
    assert summaries[-1] == "edited unrelated.py"


def test_search_actions_default_threshold_filters_unrelated(tmp_db, monkeypatch):
    """With the default threshold (0.2), zero-similarity rows are filtered out."""
    import embed as embed_mod

    vecs = {
        "auth": [1.0, 0.0, 0.0],
        "unrelated": [0.0, 1.0, 0.0],
    }

    def fake_embed(text: str):
        return vecs["auth"] if "auth" in text else vecs["unrelated"]

    monkeypatch.setattr(embed_mod, "embed", fake_embed)

    sid = tmp_db.create_session("test")
    import pickle
    tmp_db.add_action(
        session_id=sid, tool="apply_diff", args_json="{}",
        summary="edited auth.py", why=None, outcome="succeeded",
        error_excerpt=None, embedding=pickle.dumps(vecs["auth"]),
    )
    tmp_db.add_action(
        session_id=sid, tool="apply_diff", args_json="{}",
        summary="edited unrelated.py", why=None, outcome="succeeded",
        error_excerpt=None, embedding=pickle.dumps(vecs["unrelated"]),
    )
    # Default threshold = 0.2 — only the auth row should come back.
    rows = tmp_db.search_actions(session_id=sid, query="auth")
    summaries = [r["summary"] for r in rows]
    assert summaries == ["edited auth.py"]
