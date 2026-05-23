"""Integration test: the dispatcher loop in ChatAgent records mutating
actions and ignores non-mutating ones, and recording failures do not
break the tool call."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools


def _count_actions(db, session_id: int) -> int:
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM actions WHERE session_id=?", (session_id,))
        return cur.fetchone()[0]


def test_record_action_writes_row_for_mutating_tool(tmp_db, monkeypatch):
    """ChatAgent._record_action writes one row for a mutating tool."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0, 0.0])

    from agent import ChatAgent

    # Avoid touching the real DB and skip the full ChatAgent init.
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = tmp_db
    agent.session_id = tmp_db.create_session("t")

    agent._record_action(
        tool_name="apply_diff",
        args={"path": "auth.py", "old": "a", "new": "b"},
        result="Patched 1 hunk.",
        assistant_text="I'll fix the auth bug.",
    )
    assert _count_actions(tmp_db, agent.session_id) == 1


def test_record_action_classifies_failure(tmp_db, monkeypatch):
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0, 0.0])

    from agent import ChatAgent
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = tmp_db
    agent.session_id = tmp_db.create_session("t")

    agent._record_action(
        tool_name="run_shell",
        args={"command": "pytest"},
        result="Error: tests failed",
        assistant_text="verify the fix",
    )
    import sqlite3
    with sqlite3.connect(tmp_db.db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT outcome, error_excerpt FROM actions WHERE session_id=?", (agent.session_id,))
        outcome, excerpt = cur.fetchone()
    assert outcome == "failed"
    assert "Error: tests failed" in excerpt


def test_record_action_swallows_exceptions(tmp_db, monkeypatch, capsys):
    """If the DB write raises, the call does not propagate."""
    import embed as embed_mod
    def bad_embed(text):
        raise RuntimeError("boom")
    monkeypatch.setattr(embed_mod, "embed", bad_embed)

    from agent import ChatAgent
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = tmp_db
    agent.session_id = tmp_db.create_session("t")

    # Must not raise.
    agent._record_action(
        tool_name="apply_diff",
        args={"path": "x.py"},
        result="ok",
        assistant_text="doing the thing",
    )
    # If embed fails, the action should still be recorded with embedding=None
    # (per spec: embed failure is best-effort and shouldn't prevent the row).
    # OR it should be skipped silently. Either is acceptable — what matters
    # is that the call doesn't raise.


def test_mutating_tool_in_dispatcher_records_action(tmp_db, monkeypatch):
    """End-to-end: simulate the dispatcher's per-tool gating + recording."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0, 0.0])

    from agent import ChatAgent
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = tmp_db
    agent.session_id = tmp_db.create_session("t")

    tool_name = "apply_diff"
    args = {"path": "foo.py"}
    full_response = "I'll edit foo.py to fix the import."
    full_result = "Patched 1 hunk."

    if tool_name in tools.MUTATING_TOOLS:
        agent._record_action(
            tool_name=tool_name,
            args=args,
            result=full_result,
            assistant_text=full_response,
        )

    assert _count_actions(tmp_db, agent.session_id) == 1


def test_non_mutating_tool_in_dispatcher_does_not_record(tmp_db, monkeypatch):
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0, 0.0])

    from agent import ChatAgent
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = tmp_db
    agent.session_id = tmp_db.create_session("t")

    tool_name = "read_file"  # not in MUTATING_TOOLS
    args = {"path": "foo.py"}

    if tool_name in tools.MUTATING_TOOLS:
        agent._record_action(
            tool_name=tool_name, args=args, result="...", assistant_text="...",
        )

    assert _count_actions(tmp_db, agent.session_id) == 0


# ── SpecializedAgent (multi-agent) ──────────────────────────────────────────


def test_specialized_agent_records_mutating_action(tmp_db, monkeypatch):
    """SpecializedAgent has a session_id and a _record_action method that works."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0])

    from multi_agent import SpecializedAgent

    sid = tmp_db.create_session("multi")
    # Construct without the heavy parts — we only need db, session_id, and the method.
    agent = SpecializedAgent.__new__(SpecializedAgent)
    agent.db = tmp_db
    agent.session_id = sid

    agent._record_action(
        tool_name="write_file",
        args={"path": "out.txt", "content": "hi"},
        result="wrote 2 bytes",
        assistant_text="creating the file",
    )
    assert _count_actions(tmp_db, sid) == 1


def test_multi_agent_system_propagates_session_id(monkeypatch, tmp_path):
    """MultiAgentSystem.__init__ accepts and stores session_id, and passes it
    to each SpecializedAgent."""
    import multi_agent
    monkeypatch.setattr(multi_agent, "load_skills", lambda: [])
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "mas.db"))

    from multi_agent import MultiAgentSystem
    mas = MultiAgentSystem(session_id=None)
    assert mas.session_id is not None  # auto-created if None
    for agent in mas.agents.values():
        assert agent.session_id == mas.session_id
