# Action Tracking & Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ezclaw's mutating tool calls queryable so the agent can answer "what did you do about X in past turn" by calling a new `recall_actions` tool instead of guessing.

**Architecture:** Add an `actions` sqlite table written from a single chokepoint in each agent's tool-dispatcher loop (the existing per-tool block in `agent.py` and `multi_agent.py`). Record summary + why + outcome + embedding. Retrieval is a new `recall_actions` tool that does semantic search filtered by the current session via a `contextvars.ContextVar`.

**Tech Stack:** Python 3, sqlite3 (already in use), ollama embeddings via `embed.embed()` (already in use), pytest.

**Spec:** `docs/superpowers/specs/2026-05-23-action-tracking-design.md`

---

## File Structure

- **Create:** `action_tracking.py` — pure helpers (`summarize_action`, `extract_why`, `classify_outcome`). Lives at project root next to `memory.py` / `embed.py`, follows the same flat module convention.
- **Modify:** `memory.py` — add `actions` table to `_init_db`, add `add_action()` and `search_actions()` methods on `Database`.
- **Modify:** `tools.py` — add `MUTATING_TOOLS` constant, add `_session_id_var` contextvar + `set_session_context()` helper, add `create_action_tracking_tools(db)` factory that registers the `recall_actions` tool.
- **Modify:** `agent.py` — add `_record_action()` method on `ChatAgent`, call it inside the existing dispatcher loop (around line 538), set the contextvar in `__init__`.
- **Modify:** `multi_agent.py` — store `session_id` on `MultiAgentSystem`, plumb it to `SpecializedAgent`, set the contextvar before each `run`, add the same intercept in the dispatcher (around line 445), and add the prompt nudge to the executor system prompt.
- **Create:** `tests/test_action_helpers.py` — unit tests for the pure helpers.
- **Create:** `tests/test_actions_db.py` — sqlite round-trip + session filter + semantic ranking.
- **Create:** `tests/test_action_dispatcher_integration.py` — both dispatchers record mutating tools and ignore non-mutating ones; recording errors don't break the tool call.

---

## Task 1: Pure outcome/why/summary helpers

**Files:**
- Create: `action_tracking.py`
- Create: `tests/test_action_helpers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_action_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_action_helpers.py -v`
Expected: ImportError or ModuleNotFoundError on `action_tracking`.

- [ ] **Step 3: Implement the helpers**

Create `action_tracking.py`:

```python
"""Pure helpers for the action-tracking subsystem.

Kept dependency-free so they can be unit-tested without touching sqlite,
ollama, or any agent state.
"""

import os
from typing import Any, Dict, Optional, Tuple


def classify_outcome(result: str) -> Tuple[str, Optional[str]]:
    """Inspect a tool's stringified result and return (outcome, error_excerpt).

    Outcome is one of: 'succeeded', 'failed', 'partial'.
    error_excerpt is the first 300 chars of `result` when outcome != 'succeeded',
    else None.
    """
    if not result:
        return "succeeded", None
    head = result.lstrip()
    failed_markers = ("Error:", "Traceback", "Exception")
    if any(head.startswith(m) or m in head[:500] for m in failed_markers):
        return "failed", result[:300]
    partial_markers = ("exit code 1", "exit code 2", "non-zero exit", "exit status 1")
    if any(m in result for m in partial_markers):
        return "partial", result[:300]
    return "succeeded", None


def extract_why(assistant_text: str) -> Optional[str]:
    """Return the last non-empty sentence of `assistant_text`, trimmed to 200 chars.

    LLMs reliably narrate intent before invoking a tool — this captures that
    intent. Returns None if there is no usable text.
    """
    if not assistant_text or not assistant_text.strip():
        return None
    text = assistant_text.strip()
    # Split on sentence terminators; keep the last non-empty piece.
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    if not parts:
        return None
    last = parts[-1]
    return last[:200]


def summarize_action(tool_name: str, args: Dict[str, Any]) -> str:
    """One-line human description of a tool call, suitable for embedding."""
    if tool_name == "apply_diff":
        path = args.get("path", "?")
        return f"edited {os.path.basename(path)}"
    if tool_name == "write_file":
        path = args.get("path", "?")
        return f"wrote {os.path.basename(path)}"
    if tool_name == "run_shell":
        cmd = args.get("command", "")
        return f"ran: {cmd[:60]}"
    if tool_name == "schedule_task":
        desc = args.get("description", "")
        return f"scheduled: {desc[:60]}"
    if tool_name == "unschedule_task":
        return f"unscheduled task {args.get('task_id', '?')}"
    return f"called {tool_name}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_action_helpers.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add action_tracking.py tests/test_action_helpers.py
git commit -m "feat(action-tracking): pure helpers for outcome/why/summary"
```

---

## Task 2: `actions` table schema

**Files:**
- Modify: `memory.py:12-42` (the `_init_db` method)
- Create: `tests/test_actions_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_actions_db.py` with this content for now (we'll add more in Task 3):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_actions_db.py -v`
Expected: FAIL — no `actions` table.

- [ ] **Step 3: Add the table to `_init_db`**

In `memory.py`, inside `_init_db`, add a new `cursor.execute(...)` right before the `conn.commit()` on line 41. The full method should look like this (only the new block is shown for context — keep everything else as-is):

```python
            cursor.execute('''CREATE TABLE IF NOT EXISTS actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL,
    tool          TEXT NOT NULL,
    args_json     TEXT NOT NULL,
    summary       TEXT NOT NULL,
    why           TEXT,
    outcome       TEXT NOT NULL,
    error_excerpt TEXT,
    embedding     BLOB,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_actions_session
    ON actions(session_id, created_at)''')
            conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_actions_db.py -v`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add memory.py tests/test_actions_db.py
git commit -m "feat(action-tracking): add actions table to schema"
```

---

## Task 3: `Database.add_action` and `Database.search_actions`

**Files:**
- Modify: `memory.py` (append two new methods after `set_message_embedding` on line 307)
- Modify: `tests/test_actions_db.py` (append more tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_actions_db.py`:

```python
# ── add_action / search_actions ─────────────────────────────────────────────


def _make_session(db: Database) -> int:
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
    # Stub embed so the test doesn't require ollama running.
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

    # Three orthogonal-ish vectors. Query matches the first exactly.
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
        return vecs["auth"]  # query falls here

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
    rows = tmp_db.search_actions(session_id=sid, query="auth", limit=3)
    summaries = [r["summary"] for r in rows]
    # auth is most similar; unrelated is least.
    assert summaries[0] == "edited auth.py"
    assert summaries[-1] == "edited unrelated.py"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_actions_db.py -v`
Expected: AttributeError — `Database` has no `add_action` / `search_actions`.

- [ ] **Step 3: Implement `add_action` and `search_actions`**

Append to `memory.py` (after the `set_message_embedding` method, before any class-closing):

```python
    # ── Actions (mutating-tool audit trail) ──────────────────────────

    def add_action(
        self,
        session_id: int,
        tool: str,
        args_json: str,
        summary: str,
        why: Optional[str],
        outcome: str,
        error_excerpt: Optional[str],
        embedding: Optional[bytes],
    ) -> None:
        """Insert one mutating tool call into the actions table.

        All five recording-side fields are best-effort and may be None
        except session_id/tool/args_json/summary/outcome.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO actions
                   (session_id, tool, args_json, summary, why, outcome, error_excerpt, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, tool, args_json, summary, why, outcome, error_excerpt, embedding),
            )
            conn.commit()

    def search_actions(
        self,
        session_id: int,
        query: str,
        limit: int = 5,
        threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Semantic search over this session's actions. Returns top `limit`
        rows above `threshold`, ranked by cosine similarity to `query`.
        """
        from embed import embed, cosine_similarity

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, tool, summary, why, outcome, error_excerpt,
                          embedding, created_at
                   FROM actions WHERE session_id = ?""",
                (session_id,),
            )
            rows = cursor.fetchall()

        if not rows:
            return []

        q_vec = embed(query)
        scored = []
        for row_id, tool, summary, why, outcome, error_excerpt, emb_blob, created_at in rows:
            if not emb_blob:
                continue
            a_vec = pickle.loads(emb_blob)
            sim = cosine_similarity(q_vec, a_vec)
            if sim >= threshold:
                scored.append((sim, {
                    "id": row_id,
                    "tool": tool,
                    "summary": summary,
                    "why": why,
                    "outcome": outcome,
                    "error_excerpt": error_excerpt,
                    "created_at": created_at,
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_actions_db.py -v`
Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add memory.py tests/test_actions_db.py
git commit -m "feat(action-tracking): add_action and search_actions on Database"
```

---

## Task 4: `MUTATING_TOOLS`, session contextvar, and `recall_actions` tool

**Files:**
- Modify: `tools.py` — add constant + contextvar + factory function
- Create: `tests/test_recall_actions_tool.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recall_actions_tool.py`:

```python
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
    # Reset the contextvar to None.
    set_session_context(None)
    out = registry.tools["recall_actions"]("anything")
    # Either "no session" message or "no matching" — but no exception.
    assert isinstance(out, str)
    assert len(out) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recall_actions_tool.py -v`
Expected: ImportError on `MUTATING_TOOLS`, `create_action_tracking_tools`, or `set_session_context`.

- [ ] **Step 3: Add the constant, contextvar, and tool factory**

In `tools.py`, add near the top (after the imports, before `class ToolRegistry`) — pick a clean spot above line 11:

```python
import contextvars

# Whitelist of tools whose calls are recorded as actions. Reads/searches
# are excluded — only state-changing operations qualify.
MUTATING_TOOLS = frozenset({
    "apply_diff",
    "write_file",
    "run_shell",
    "schedule_task",
    "unschedule_task",
})

# Set by the agent at session start; read by recall_actions to scope its
# search. Module-level so it survives across the tool's invocation
# without threading a parameter through every call site.
_session_id_var: contextvars.ContextVar = contextvars.ContextVar(
    "ezclaw_session_id", default=None
)


def set_session_context(session_id):
    """Called by the agent at session start. Pass None to clear."""
    _session_id_var.set(session_id)


def get_session_context():
    """Return the current session id, or None if unset."""
    return _session_id_var.get()
```

Then, near the existing `create_memory_tools(db: Any)` factory (around line 1061), add a new factory right after it:

```python
def create_action_tracking_tools(db: Any):
    """Register the `recall_actions` tool. Bound to `db`; reads the current
    session id from the module-level contextvar set by the agent."""

    @registry.register
    def recall_actions(query: str, limit: int = 5) -> str:
        """Search this session's past mutating actions by what they did or why.

        Call when the user asks about a past action — 'what did you do
        about X', 'did you fix Y', 'which files did you edit earlier'.
        Returns one line per matching action: time, summary, why, outcome.
        """
        session_id = get_session_context()
        if session_id is None:
            return "No active session — action history unavailable."
        rows = db.search_actions(session_id=session_id, query=query, limit=limit)
        if not rows:
            return "No matching actions in this session."
        lines = []
        for r in rows:
            # created_at is an ISO-like string from sqlite; show HH:MM if parseable.
            ts = (r.get("created_at") or "")[-8:-3] or "??:??"
            head = f"[{ts}] {r['summary']}"
            if r.get("why"):
                head += f" — why: \"{r['why']}\""
            head += f" — {r['outcome']}"
            if r.get("error_excerpt"):
                head += f": {r['error_excerpt'][:120]}"
            lines.append(head)
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recall_actions_tool.py -v`
Expected: all six tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_recall_actions_tool.py
git commit -m "feat(action-tracking): MUTATING_TOOLS, contextvar, recall_actions tool"
```

---

## Task 5: ChatAgent intercept in `agent.py`

**Files:**
- Modify: `agent.py` — import action_tracking helpers + tools, register `create_action_tracking_tools`, set the contextvar in `__init__`, add `_record_action` method, call it in the dispatcher loop
- Create: `tests/test_action_dispatcher_integration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_action_dispatcher_integration.py`:

```python
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
    monkeypatch.setattr(embed_mod, "embed", lambda text: (_ for _ in ()).throw(RuntimeError("boom")))

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
    # The row is not written (embed blew up), but no exception escaped.
    assert _count_actions(tmp_db, agent.session_id) == 0


def test_mutating_tool_in_dispatcher_records_action(tmp_db, monkeypatch):
    """End-to-end: simulate the dispatcher's per-tool block calling _record_action."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0, 0.0])

    from agent import ChatAgent
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = tmp_db
    agent.session_id = tmp_db.create_session("t")

    # Replay the dispatcher's gating logic explicitly.
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_action_dispatcher_integration.py -v`
Expected: AttributeError — `ChatAgent` has no `_record_action`.

- [ ] **Step 3: Wire it into `agent.py`**

In `agent.py`, near the top imports (after the existing `from tools import ...` lines), add:

```python
import json
import pickle
from action_tracking import classify_outcome, extract_why, summarize_action
from tools import MUTATING_TOOLS, create_action_tracking_tools, set_session_context
```

(`json` and `pickle` may already be imported — keep one each, don't duplicate.)

In `ChatAgent.__init__`, after the existing `create_memory_tools(self.db)` call (around line 116), add:

```python
        create_action_tracking_tools(self.db)
```

In `ChatAgent.__init__`, after `self.session_id` is finalized (around line 125-127), add:

```python
        set_session_context(self.session_id)
```

Add the `_record_action` method on `ChatAgent` (place it near the other private methods, e.g. just above `clear_session_history` on line 291):

```python
    def _record_action(self, tool_name: str, args: dict, result: str, assistant_text: str) -> None:
        """Record one mutating tool call to the actions table.

        Best-effort: any failure inside this method is logged to stderr
        and swallowed — never propagates out and never blocks the tool.
        """
        try:
            from embed import embed as _embed
            summary = summarize_action(tool_name, args)
            why = extract_why(assistant_text)
            outcome, error_excerpt = classify_outcome(result)
            embed_text = f"{summary} {why or ''}".strip()
            try:
                vec = _embed(embed_text)
                emb_blob = pickle.dumps(vec)
            except Exception:
                emb_blob = None
            self.db.add_action(
                session_id=self.session_id,
                tool=tool_name,
                args_json=json.dumps(args, default=str),
                summary=summary,
                why=why,
                outcome=outcome,
                error_excerpt=error_excerpt,
                embedding=emb_blob,
            )
        except Exception as e:
            import sys
            print(f"[action-tracking] record failed: {e}", file=sys.stderr)
```

Then, in the dispatcher loop, find the block around line 538 that currently looks like:

```python
                tool_msg = {'role': 'tool', 'content': result_str, 'name': tool.function.name}
                self.messages.append(tool_msg)
                self.db.add_message(self.session_id, "tool", result_str)
                yield {"type": "tool_end", "name": tool.function.name, "result": full_result}
```

Insert the recording call between the `self.db.add_message` and the `yield`:

```python
                tool_msg = {'role': 'tool', 'content': result_str, 'name': tool.function.name}
                self.messages.append(tool_msg)
                self.db.add_message(self.session_id, "tool", result_str)
                if tool.function.name in MUTATING_TOOLS:
                    self._record_action(
                        tool_name=tool.function.name,
                        args=tool.function.arguments,
                        result=full_result,
                        assistant_text=full_response,
                    )
                yield {"type": "tool_end", "name": tool.function.name, "result": full_result}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_action_dispatcher_integration.py -v`
Expected: all five tests pass.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: every test passes (including the pre-existing ones).

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_action_dispatcher_integration.py
git commit -m "feat(action-tracking): ChatAgent records mutating tool calls"
```

---

## Task 6: Multi-agent session plumbing

**Files:**
- Modify: `multi_agent.py` — store `session_id` on `MultiAgentSystem`, pass it to `SpecializedAgent.__init__`, set contextvar in `MultiAgentSystem.run`
- Modify: `tests/test_action_dispatcher_integration.py` — add tests for SpecializedAgent

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_action_dispatcher_integration.py`:

```python
# ── SpecializedAgent (multi-agent) ──────────────────────────────────────────


def test_specialized_agent_records_mutating_action(tmp_db, monkeypatch):
    """SpecializedAgent has a session_id and a _record_action method that works."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0])

    from multi_agent import SpecializedAgent

    sid = tmp_db.create_session("multi")
    cfg = {
        "model": "test-model",
        "system_prompt": "test",
    }
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
    # Stub the heavy parts of MultiAgentSystem so we can construct it.
    import multi_agent
    monkeypatch.setattr(multi_agent, "load_skills", lambda: [])

    # Use a fresh db file so we don't collide with the default ezclaw.db.
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "mas.db"))

    from multi_agent import MultiAgentSystem
    mas = MultiAgentSystem(session_id=None)
    assert mas.session_id is not None  # auto-created if None
    for agent in mas.agents.values():
        assert agent.session_id == mas.session_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_action_dispatcher_integration.py::test_specialized_agent_records_mutating_action tests/test_action_dispatcher_integration.py::test_multi_agent_system_propagates_session_id -v`
Expected: AttributeError — `SpecializedAgent` has no `session_id` / `_record_action`; `MultiAgentSystem` doesn't store it.

- [ ] **Step 3: Modify `multi_agent.py`**

Near the top imports, add (next to the other internal imports around line 1-20):

```python
import json
import pickle
from action_tracking import classify_outcome, extract_why, summarize_action
from tools import MUTATING_TOOLS, create_action_tracking_tools, set_session_context
```

(`json` may already be imported — don't duplicate.)

In `SpecializedAgent.__init__` (line 176), change the signature to accept `session_id`:

```python
    def __init__(self, name: str, config: dict, db: Database, auth_state: "_SharedAuthState" = None, session_id: int = None):
        self.name = name
        self.client = build_agent_client()
        self.model = config["model"]
        self.system_prompt = config["system_prompt"]
        self.tools = registry.get_tool_definitions()
        self.db = db
        self.session_id = session_id
        ...
```

(Keep the rest of `__init__` unchanged — only add `self.session_id = session_id`.)

Add the same `_record_action` method on `SpecializedAgent` — copy-paste from `agent.py` (do **not** abstract yet; two short copies are easier to understand than a base class for now):

```python
    def _record_action(self, tool_name: str, args: dict, result: str, assistant_text: str) -> None:
        """Record one mutating tool call to the actions table. Best-effort —
        never propagates out and never blocks the tool."""
        try:
            from embed import embed as _embed
            summary = summarize_action(tool_name, args)
            why = extract_why(assistant_text)
            outcome, error_excerpt = classify_outcome(result)
            embed_text = f"{summary} {why or ''}".strip()
            try:
                vec = _embed(embed_text)
                emb_blob = pickle.dumps(vec)
            except Exception:
                emb_blob = None
            if self.session_id is None:
                return
            self.db.add_action(
                session_id=self.session_id,
                tool=tool_name,
                args_json=json.dumps(args, default=str),
                summary=summary,
                why=why,
                outcome=outcome,
                error_excerpt=error_excerpt,
                embedding=emb_blob,
            )
        except Exception as e:
            import sys
            print(f"[action-tracking] record failed: {e}", file=sys.stderr)
```

In `MultiAgentSystem.__init__` (line 784), accept and store the session id, register the recall tool, set the contextvar, and pass `session_id` through to each `SpecializedAgent`:

```python
class MultiAgentSystem:
    def __init__(self, session_id: Optional[int] = None):
        self.db = Database(os.getenv("DATABASE_PATH", "ezclaw.db"))
        create_memory_tools(self.db)
        create_action_tracking_tools(self.db)
        self.skills = load_skills()
        if session_id is None:
            session_id = self.db.get_last_session_id() or self.db.create_session()
        self.session_id = session_id
        set_session_context(self.session_id)
        self.architect = Architect(self.db)
        self._shared_auth = _SharedAuthState()
        self.agents = {
            name: SpecializedAgent(name, cfg, self.db, auth_state=self._shared_auth, session_id=self.session_id)
            for name, cfg in AGENT_DEFS.items()
        }
        self._conversation_history: List[Dict[str, str]] = []
        self.current_plan = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_action_dispatcher_integration.py -v`
Expected: all tests pass.

- [ ] **Step 5: Run full suite for regressions**

Run: `pytest tests/ -v`
Expected: every test passes.

- [ ] **Step 6: Commit**

```bash
git add multi_agent.py tests/test_action_dispatcher_integration.py
git commit -m "feat(action-tracking): multi-agent session plumbing + SpecializedAgent recording"
```

---

## Task 7: Multi-agent dispatcher intercept

**Files:**
- Modify: `multi_agent.py` — call `_record_action` from the dispatcher loop around line 445

- [ ] **Step 1: Write the failing test**

Append to `tests/test_action_dispatcher_integration.py`:

```python
def test_specialized_agent_dispatcher_gating_records_mutating(tmp_db, monkeypatch):
    """Replay the multi_agent dispatcher's gating against a mutating tool."""
    import embed as embed_mod
    monkeypatch.setattr(embed_mod, "embed", lambda text: [1.0, 0.0])

    from multi_agent import SpecializedAgent
    agent = SpecializedAgent.__new__(SpecializedAgent)
    agent.db = tmp_db
    agent.session_id = tmp_db.create_session("multi")

    tool_name = "apply_diff"
    args = {"path": "auth.py"}
    full_response = "I'll edit auth.py to fix the import."
    full_result = "Patched 1 hunk."

    if tool_name in tools.MUTATING_TOOLS:
        agent._record_action(
            tool_name=tool_name, args=args,
            result=full_result, assistant_text=full_response,
        )
    assert _count_actions(tmp_db, agent.session_id) == 1
```

- [ ] **Step 2: Verify it passes (Task 6 already added `_record_action`)**

Run: `pytest tests/test_action_dispatcher_integration.py::test_specialized_agent_dispatcher_gating_records_mutating -v`
Expected: PASS.

- [ ] **Step 3: Wire the intercept into the real dispatcher**

In `multi_agent.py`, find the per-tool block around line 442-445:

```python
                self.messages.append({
                    "role": "tool", "content": result_str, "name": tool_call.function.name,
                })
                yield {"type": "tool_end", "name": tool_call.function.name, "result": full_result}
```

Insert the recording call between the `self.messages.append({...})` and the `yield`:

```python
                self.messages.append({
                    "role": "tool", "content": result_str, "name": tool_call.function.name,
                })
                if tool_call.function.name in MUTATING_TOOLS:
                    self._record_action(
                        tool_name=tool_call.function.name,
                        args=tool_call.function.arguments,
                        result=full_result,
                        assistant_text=full_response,
                    )
                yield {"type": "tool_end", "name": tool_call.function.name, "result": full_result}
```

- [ ] **Step 4: Re-run full suite**

Run: `pytest tests/ -v`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add multi_agent.py tests/test_action_dispatcher_integration.py
git commit -m "feat(action-tracking): multi-agent dispatcher records mutating tools"
```

---

## Task 8: Executor prompt nudge

**Files:**
- Modify: `multi_agent.py` — append one paragraph to the executor system prompt
- Modify: `agent.py` — append one line to the system prompt used by the single-agent `ChatAgent`

- [ ] **Step 1: Locate the executor system prompt**

In `multi_agent.py`, find `AGENT_DEFS["executor"]["system_prompt"]` (the multi-line string near line 43). Read it once; it starts with *"You are EzClaw's **Executor**"*.

- [ ] **Step 2: Append the nudge to the executor prompt**

Add this paragraph at the end of the executor's `system_prompt` string (just before the closing triple-quote):

```
═══════════════════════════════════════════════════════════════
## Past actions

When the user asks what you did about a past task, file, bug, or feature
("what did you do about X", "did you fix Y", "earlier you changed
something in Z"), call `recall_actions(query)` BEFORE answering.
`recall_actions` is authoritative for this session's mutating actions —
don't reconstruct from memory or guess.
```

- [ ] **Step 3: Add the same nudge to ChatAgent's system prompt**

In `agent.py`, find the system prompt construction (look around line 130-140 — the `agents.md` fallback area). The simplest place is to read `agents.md`; if that's where the executor prompt lives, edit `agents.md` instead. Otherwise, after the `self.system_prompt = ...` assignment in `ChatAgent.__init__`, append:

```python
        self.system_prompt += (
            "\n\n## Past actions\n"
            "When the user asks what you did about a past task, file, bug, or "
            "feature, call `recall_actions(query)` BEFORE answering. It is "
            "authoritative for this session's mutating actions."
        )
```

- [ ] **Step 4: Smoke test the prompt change with a quick unit assertion**

Append to `tests/test_recall_actions_tool.py`:

```python
def test_executor_prompt_mentions_recall_actions():
    from multi_agent import AGENT_DEFS
    assert "recall_actions" in AGENT_DEFS["executor"]["system_prompt"]
```

Run: `pytest tests/test_recall_actions_tool.py::test_executor_prompt_mentions_recall_actions -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add multi_agent.py agent.py tests/test_recall_actions_tool.py
git commit -m "feat(action-tracking): nudge agents to call recall_actions on past-action queries"
```

---

## Task 9: Manual smoke test

**Files:** none — this is a live verification.

- [ ] **Step 1: Start the CLI**

Run: `python cli.py` (from the project root with the venv activated).

- [ ] **Step 2: Drive a small mutating sequence**

In the chat, do these one at a time:

1. `write a file called workspace/note.txt that contains "hello"`
2. `now edit workspace/note.txt and replace "hello" with "world"`
3. `run ls workspace/ to confirm`

After each, watch for the tool panels (`write_file`, `apply_diff`, `run_shell`).

- [ ] **Step 3: Ask the agent what it did**

Type: `what did you do about workspace/note.txt earlier?`

Expected: the agent calls `recall_actions("workspace/note.txt")` and the answer references the actual write/edit steps — not a hallucinated reconstruction. The tool panel should show `recall_actions` was called.

- [ ] **Step 4: Inspect the DB directly**

Run:

```bash
sqlite3 ezclaw.db "SELECT created_at, tool, summary, why, outcome FROM actions ORDER BY id DESC LIMIT 10;"
```

Expected: rows for `write_file`, `apply_diff`, and `run_shell`. No rows for `read_file` or `list_dir`.

- [ ] **Step 5: Cross-session scope check**

Exit the CLI (Ctrl-D), restart it (which creates or attaches to a new session), and ask: `what did you do about note.txt?`

Expected: the agent reports "No matching actions in this session" (because the prior session's actions are filtered out). The DB still has the old rows — they're just not in scope.

- [ ] **Step 6: Note any deviations**

If anything misbehaves (wrong outcome classification, missing why-capture, prompt nudge ignored), file a follow-up note in the implementation log — don't try to fix in this task.

---

## Self-Review

**Spec coverage:**
- Goals — all four covered:
  - mutating-tool recording → Tasks 1-3, 5-7
  - `recall_actions` tool → Task 4
  - failures don't break tool call → Task 5 (`test_record_action_swallows_exceptions`) + Task 6
- Non-goals — none added; cross-session/filter/snapshot all explicitly out.
- Data model — Task 2 implements the exact schema from the spec.
- Mutating-tool whitelist — Task 4 implements exactly the five listed.
- Dispatcher intercept — Tasks 5 + 7 hit both dispatchers (the spec mentioned only `agent.py`, but `multi_agent.py` has its own dispatcher loop that needs the same treatment for the feature to work in multi-agent mode; this is the kind of targeted improvement the spec's "design for isolation and clarity" section endorses).
- Retrieval — Task 4's tool implementation matches the spec's output format.
- Prompt nudge — Task 8.
- Testing section in the spec — Task 1 covers `_classify_outcome` / `_summarize_action` / `_extract_why`; Task 3 covers DB; Task 5 covers the dispatcher integration smoke.
- Lifecycle (rows persist, scope enforced by query filter) — covered by Task 4's filter-by-session test and Task 9 step 5.

**Placeholder scan:** No "TBD", "TODO", or "fill in" markers. All code blocks contain the actual code the engineer should write.

**Type consistency:**
- `add_action` signature in Task 3 matches the call site in Task 5's `_record_action`.
- `search_actions` returns `List[Dict[str, Any]]` in Task 3; the `recall_actions` tool in Task 4 reads `r["summary"]`, `r.get("why")`, `r["outcome"]`, `r.get("error_excerpt")`, `r.get("created_at")` — all of which Task 3 puts into the dict.
- `set_session_context` / `get_session_context` defined in Task 4 are used by the `recall_actions` tool in Task 4 and by `__init__` in Tasks 5 and 6.
- `MUTATING_TOOLS` defined in Task 4 is read by Tasks 5 and 7 dispatcher gating.
- Helper signatures (`classify_outcome`, `extract_why`, `summarize_action`) are defined in Task 1 and called identically in Tasks 5 and 6.
