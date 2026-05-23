# Action Tracking & Recall — Design

**Date:** 2026-05-23
**Status:** Approved, pending implementation plan
**Scope:** ezclaw agent — make mutating tool calls queryable so the agent can answer "what did you do about X in a past turn."

## Problem

Conversation history today is a chronological dump. `messages.tool_calls` stores each call as raw JSON inside the row, and the architect prompt renders history turn-by-turn. There is no way to ask the agent *"what did you do about the auth bug"* or *"which files did you edit earlier"* and get a grounded answer — the agent has to scan or hallucinate.

## Goals

- Every mutating tool call (apply_diff, run_shell, write_file, schedule_task, unschedule_task) is recorded as a first-class row with: tool name, args, a one-line summary, the agent's stated reason, an outcome label, and an embedding.
- A new `recall_actions(query)` tool returns the top-K matching actions in the current session, formatted for the agent to quote back.
- Failures inside the recording path never break the underlying tool call.

## Non-goals

- Cross-session recall (current-session-only).
- Structured filter queries (semantic-only retrieval).
- Before/after state snapshots.
- Tracking non-mutating tools (reads, searches, web).
- Deletion or expiry — rows persist; the session-id filter enforces scope.

## Architecture

Single intercept point in the existing dispatcher loop in `agent.py` (around lines 511-538). After a tool returns, if its name is in the mutating-tools whitelist, record the action. Retrieval is a new tool that does an embedding search over the new table, filtered by `session_id`.

### Data model (sqlite, in `memory.py`)

```sql
CREATE TABLE actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL,
    tool          TEXT NOT NULL,
    args_json     TEXT NOT NULL,
    summary       TEXT NOT NULL,
    why           TEXT,
    outcome       TEXT NOT NULL,        -- 'succeeded' | 'failed' | 'partial'
    error_excerpt TEXT,                  -- first 300 chars of failure output, NULL on success
    embedding     BLOB,                  -- embed(summary + " " + (why or ""))
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE INDEX idx_actions_session ON actions(session_id, created_at);
```

Migration: add to `_init_db` (idempotent `CREATE TABLE IF NOT EXISTS`) — no separate migration step needed; existing dbs pick it up on next start.

### Mutating-tool whitelist

In `tools.py`, exported constant:

```python
MUTATING_TOOLS = {
    "apply_diff",
    "write_file",
    "run_shell",
    "schedule_task",
    "unschedule_task",
}
```

`run_shell` is mutating-by-default. We do not parse shell commands to classify them. Reads (`read_file`, `list_dir`, `grep_codebase`, `code_outline`, `git_diff`, `git_log`, `git_blame`, `web_search`, `web_fetch`, `recall`, `current_datetime`, `get_system_info`, `python_eval`, `run_tests`, `generate_codebase_map`) are excluded.

### Dispatcher change (`agent.py`)

In the existing per-tool block (currently lines 511-538), after `result = tool_func(**args)` and the result-truncation logic, before the `yield {"type": "tool_end", ...}`:

```python
if tool.function.name in MUTATING_TOOLS:
    try:
        self._record_action(
            tool_name=tool.function.name,
            args=tool.function.arguments,
            result=full_result,
            assistant_text=full_response,
        )
    except Exception as e:
        # Never let recording break the tool call.
        # (We log to stderr; no user-visible noise.)
        import sys
        print(f"[action-tracking] record failed: {e}", file=sys.stderr)
```

`_record_action` is a new method on the agent class:

```python
def _record_action(self, tool_name: str, args: dict, result: str, assistant_text: str) -> None:
    summary = _summarize_action(tool_name, args)
    why = _extract_why(assistant_text)
    outcome, error_excerpt = _classify_outcome(result)
    embed_text = f"{summary} {why or ''}".strip()
    emb = embed.embed_text(embed_text)
    self.db.add_action(
        session_id=self.session_id,
        tool=tool_name,
        args_json=json.dumps(args),
        summary=summary,
        why=why,
        outcome=outcome,
        error_excerpt=error_excerpt,
        embedding=emb,
    )
```

Helpers (module-level in `memory.py` or a new small module — implementation plan decides):

- `_summarize_action(tool, args)` — produces a short human description:
  - `apply_diff(path=X, ...)` → `"edited {basename(X)}"`
  - `write_file(path=X, ...)` → `"wrote {basename(X)}"`
  - `run_shell(command=C)` → `"ran: {first 60 chars of C}"`
  - `schedule_task(...)` → `"scheduled: {description[:60]}"`
  - `unschedule_task(task_id=N)` → `"unscheduled task {N}"`
- `_extract_why(assistant_text)` — returns the last non-empty sentence of `assistant_text`, trimmed to 200 chars. If `assistant_text` is empty, returns `None`. LLMs reliably narrate before calling, so this captures intent.
- `_classify_outcome(result)`:
  - starts with `"Error:"`, contains `"Traceback"`, or contains `"Exception"` → `("failed", result[:300])`
  - contains `"exit code"` with non-zero value, or contains `"non-zero exit"` → `("partial", result[:300])`
  - else → `("succeeded", None)`

### Retrieval — new `recall_actions` tool

Registered in `tools.py` alongside `recall`:

```python
@registry.register
def recall_actions(query: str, limit: int = 5) -> str:
    """Search this session's past mutating actions by what they did or why.

    Use when the user asks 'what did you do about X', 'did you fix Y',
    'which files did you edit', or similar past-action questions.
    """
    session_id = _current_session_id()  # injected via a contextvar at agent start
    rows = db.search_actions(session_id=session_id, query=query, limit=limit)
    if not rows:
        return "No matching actions in this session."
    return _format_actions(rows)
```

`db.search_actions(session_id, query, limit=5, threshold=0.2)` in `memory.py`:

- Embed `query`.
- `SELECT id, tool, summary, why, outcome, error_excerpt, created_at FROM actions WHERE session_id = ?` plus cosine ranking against `embedding`, threshold 0.2, ordered by similarity desc, limit `limit`.
- Returns list of dicts.

Output format (one line per action):

```
[14:02] edited auth.py — why: "fix the token storage compliance issue" — succeeded
[14:08] ran: pytest auth/ — why: "verify the fix" — failed: AssertionError on test_session_expiry...
[14:15] edited auth.py — why: "expiry was wrong direction" — succeeded
```

Session-id injection: the cleanest path is a module-level `contextvars.ContextVar` set in `agent.py` at session start, read by `recall_actions`. Alternative: pass `session_id` as a hidden arg the dispatcher fills in. Implementation plan picks one.

### Prompt nudge (`multi_agent.py`)

Add one paragraph to the executor system prompt:

> When the user asks what you did about a past task, file, bug, or feature ("what did you do about X", "did you fix Y", "earlier you changed something in Z"), call `recall_actions(query)` **before** answering. Don't reconstruct from memory — `recall_actions` is authoritative for this session's mutating actions.

This is the only behavioral change to the prompts.

## Lifecycle

Rows persist past session end. The session-id filter in `search_actions` enforces the current-session-only scope. No expiry, no deletion. (Re-enabling cross-session later is then a one-line change to the query.)

## Testing

Unit tests (in `tests/`, matching existing layout):

- **`test_actions_db.py`**
  - Insert two actions in one session, search semantically, assert ranking.
  - Insert actions across two sessions, assert `search_actions(session_id=A)` returns only A's rows.
  - Insert a failed action with `error_excerpt`, assert it round-trips.
- **`test_action_helpers.py`**
  - `_classify_outcome("Error: foo")` → `("failed", "Error: foo")`
  - `_classify_outcome("ok\nexit code 1")` → `("partial", ...)`
  - `_classify_outcome("done")` → `("succeeded", None)`
  - `_summarize_action("apply_diff", {"path": "/a/b/c.py", ...})` → `"edited c.py"`
  - `_extract_why("I'm going to fix the auth bug.")` → `"I'm going to fix the auth bug."`
  - `_extract_why("")` → `None`
- **`test_action_dispatcher_integration.py`**
  - Smoke: stub a mutating tool, run it through the dispatcher, assert one row in `actions` with the expected summary and outcome.
  - Smoke: stub a non-mutating tool, run it through the dispatcher, assert zero rows in `actions`.
  - Smoke: force `_record_action` to raise, assert the tool result still reaches the caller (recording errors are isolated).

## Risks & mitigations

- **Why-extraction misses intent when the LLM doesn't narrate.** Mitigation: `why` is nullable; downstream consumers handle `None`. Accept lossy capture; do not block the call.
- **Embedding write latency on every mutating call.** Mitigation: existing `embed.py` is already on the hot path for messages/memories; one more call per mutating tool is negligible. If it becomes hot, batch on flush.
- **`run_shell` over-broad classification.** A `git status` runs but doesn't mutate. Mitigation: accept the noise; `recall_actions` ranks by relevance, so a `git status` won't surface for "what did you edit." If noise proves real, add a secondary command-prefix filter later.
- **Session-id contextvar leakage in tests.** Mitigation: tests set/reset the contextvar in fixtures.

## Out of scope (explicit deferrals)

- Cross-session recall.
- Filter queries (`tool=`, `path~`).
- Before/after diffs.
- Pruning / expiry.
- Tracking non-mutating tools.

These remain re-enable-able later without schema change (the table already carries all the needed columns).
