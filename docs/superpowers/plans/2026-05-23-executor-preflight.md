# Executor Pre-Flight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a cheap LLM pre-flight after `_short_circuit_classify` returns `"executor"` so multi-step requests (≥3 distinct tool kinds, or pre-flight failure) escalate to the architect instead of fast-routing.

**Architecture:** A new pure helper `_parse_tool_lines(text)` and a new method `MultiAgentSystem._estimate_tool_kinds(user_input)`. The wire-in is a 5-line block between `_short_circuit_classify` and the existing fast-route branch in `MultiAgentSystem.run`. No new modules, no schema changes.

**Tech Stack:** Python 3, ollama client (already wired via `self.architect.client`), pytest with monkeypatch.

**Spec:** `docs/superpowers/specs/2026-05-23-executor-preflight-design.md`

---

## File Structure

- **Modify:** `multi_agent.py` — add `PREFLIGHT_KIND_THRESHOLD` constant, `_KNOWN_TOOL_NAMES` frozenset, free function `_parse_tool_lines(text) -> set`, method `MultiAgentSystem._estimate_tool_kinds(user_input) -> Optional[int]`, 5-line wire-in inside `run()`.
- **Create:** `tests/test_preflight_estimator.py` — unit tests for `_parse_tool_lines`.
- **Create:** `tests/test_preflight_routing.py` — integration tests that monkeypatch `_estimate_tool_kinds` to verify routing decisions.

---

## Task 1: `_parse_tool_lines` helper

**Files:**
- Modify: `multi_agent.py` (add the constant and the free function near the top)
- Create: `tests/test_preflight_estimator.py`

- [ ] **Step 1: Write the failing tests**

Create `/home/lulu/Projects/ezclaw/tests/test_preflight_estimator.py`:

```python
"""Unit tests for the pre-flight tool-line parser."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent import _parse_tool_lines


def test_empty_string_returns_empty_set():
    assert _parse_tool_lines("") == set()


def test_two_clean_tool_names():
    assert _parse_tool_lines("write_file\nrun_shell") == {"write_file", "run_shell"}


def test_dash_and_number_prefixes_stripped():
    text = "- write_file\n1. run_shell\n* read_file"
    assert _parse_tool_lines(text) == {"write_file", "run_shell", "read_file"}


def test_unknown_names_dropped_and_duplicates_collapsed():
    text = "write_file\nfoo_bar\nwrite_file"
    assert _parse_tool_lines(text) == {"write_file"}


def test_prose_lines_rejected():
    text = "first I'd read_file then write_file"
    assert _parse_tool_lines(text) == set()


def test_case_insensitive_recognition():
    text = "WRITE_FILE\nRun_Shell"
    assert _parse_tool_lines(text) == {"write_file", "run_shell"}


def test_trailing_whitespace_tolerated():
    text = "write_file   \n  run_shell\t"
    assert _parse_tool_lines(text) == {"write_file", "run_shell"}


def test_parenthesised_or_bracketed_names_stripped():
    text = "(write_file)\n[run_shell]"
    assert _parse_tool_lines(text) == {"write_file", "run_shell"}


def test_lines_with_inline_prose_after_name_rejected():
    """A line like 'write_file to make the script' is prose — not a clean
    tool listing. After normalization, the remainder is not in the known
    set, so the line is dropped."""
    text = "write_file to make the script"
    assert _parse_tool_lines(text) == set()
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_preflight_estimator.py -v`
Expected: ImportError on `_parse_tool_lines` (function doesn't exist yet).

- [ ] **Step 3: Add the constant and the helper to `multi_agent.py`**

Open `/home/lulu/Projects/ezclaw/multi_agent.py`. Find the top of the file (imports area). Below the existing imports but above the first class definition (`class _SharedAuthState` or whatever appears first), add:

```python
PREFLIGHT_KIND_THRESHOLD = 3  # >= this many distinct tool kinds → architect

_KNOWN_TOOL_NAMES = frozenset({
    "apply_diff", "write_file", "run_shell", "read_file", "list_dir",
    "grep_codebase", "web_search", "web_fetch", "recall", "code_outline",
    "git_diff", "git_log", "git_blame", "run_tests", "python_eval",
    "schedule_task", "unschedule_task", "ask_user", "current_datetime",
    "get_system_info",
})


def _parse_tool_lines(text: str) -> set:
    """Extract known tool names from a newline-separated LLM response.

    Lowercases, strips leading list markers (`-`, `*`, digits, `.`, `)`,
    `(`, `[`, `]`), strips whitespace, filters to _KNOWN_TOOL_NAMES,
    returns the unique set. Tolerates simple list formats. Rejects prose
    (lines whose normalized form is not exactly a known tool name).
    """
    result: set = set()
    if not text:
        return result
    for raw_line in text.splitlines():
        token = raw_line.strip().lower()
        if not token:
            continue
        # Strip leading list markers / punctuation.
        token = token.lstrip("-*().[] \t0123456789")
        # Strip trailing punctuation.
        token = token.rstrip(" \t.,;:)]")
        if token in _KNOWN_TOOL_NAMES:
            result.add(token)
    return result
```

- [ ] **Step 4: Verify tests pass**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_preflight_estimator.py -v`
Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add multi_agent.py tests/test_preflight_estimator.py && git commit -m "feat(preflight): _parse_tool_lines helper + tool-name whitelist

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `_estimate_tool_kinds` method

**Files:**
- Modify: `multi_agent.py` (add the method on `MultiAgentSystem`)
- Modify: `tests/test_preflight_estimator.py` (append integration-style tests that stub the LLM call)

- [ ] **Step 1: Append failing tests**

Append to `/home/lulu/Projects/ezclaw/tests/test_preflight_estimator.py`:

```python
# ── _estimate_tool_kinds (stub the LLM client) ──────────────────────────────


def _make_mas_with_stubbed_architect(monkeypatch, llm_response: str, tmp_path):
    """Build a MultiAgentSystem whose architect.client.chat returns a canned
    response. Avoids touching real ollama and skips load_skills."""
    import multi_agent

    monkeypatch.setattr(multi_agent, "load_skills", lambda: [])
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "preflight.db"))

    from multi_agent import MultiAgentSystem
    mas = MultiAgentSystem(session_id=None)

    class _StubClient:
        def chat(self, **kwargs):
            return {"message": {"content": llm_response}}

    mas.architect.client = _StubClient()
    return mas


def test_estimate_returns_kind_count(monkeypatch, tmp_path):
    mas = _make_mas_with_stubbed_architect(
        monkeypatch, "write_file\nrun_shell\nrun_shell", tmp_path
    )
    assert mas._estimate_tool_kinds("anything") == 2


def test_estimate_returns_one_for_single_kind(monkeypatch, tmp_path):
    mas = _make_mas_with_stubbed_architect(monkeypatch, "read_file", tmp_path)
    assert mas._estimate_tool_kinds("show me foo.py") == 1


def test_estimate_returns_none_when_response_is_pure_prose(monkeypatch, tmp_path):
    mas = _make_mas_with_stubbed_architect(
        monkeypatch, "I would first read the file then write it", tmp_path
    )
    assert mas._estimate_tool_kinds("anything") is None


def test_estimate_returns_none_when_chat_raises(monkeypatch, tmp_path):
    mas = _make_mas_with_stubbed_architect(monkeypatch, "ignored", tmp_path)

    class _BoomClient:
        def chat(self, **kwargs):
            raise RuntimeError("ollama down")

    mas.architect.client = _BoomClient()
    assert mas._estimate_tool_kinds("anything") is None


def test_estimate_sends_request_text_in_prompt(monkeypatch, tmp_path):
    captured = {}

    mas = _make_mas_with_stubbed_architect(monkeypatch, "write_file", tmp_path)

    class _CapturingClient:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"message": {"content": "write_file"}}

    mas.architect.client = _CapturingClient()
    mas._estimate_tool_kinds("build me a snake game")
    assert any(
        "build me a snake game" in m.get("content", "")
        for m in captured.get("messages", [])
    )
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_preflight_estimator.py -v`
Expected: AttributeError — `MultiAgentSystem` has no `_estimate_tool_kinds`.

- [ ] **Step 3: Add the method on `MultiAgentSystem`**

In `/home/lulu/Projects/ezclaw/multi_agent.py`, find the `MultiAgentSystem` class (around line 783). Add this method near the other private helpers (e.g., directly above `_short_circuit_classify` which lives at line 1083 — search for `def _short_circuit_classify` and insert ABOVE it):

```python
    def _estimate_tool_kinds(self, user_input: str) -> Optional[int]:
        """Single LLM call estimating the distinct tool kinds the executor
        would need to fulfill `user_input`. Returns the count, or None on
        any failure (timeout, parse failure, empty result).

        Used by run() to gate the fast-route to executor: if this returns
        a count >= PREFLIGHT_KIND_THRESHOLD, the architect plans instead.
        """
        prompt = (
            "You are pre-flighting a tool plan. List the EZCLAW tool names "
            "you would need to fulfill this request, ONE PER LINE, no prose, "
            "no numbering.\nUse ONLY these names:\n"
            "  apply_diff, write_file, run_shell, read_file, list_dir, "
            "grep_codebase, web_search, web_fetch, recall, code_outline, "
            "git_diff, git_log, git_blame, run_tests, python_eval, "
            "schedule_task, unschedule_task, ask_user, current_datetime, "
            "get_system_info\n\n"
            f"Request: {user_input}\n\nTools needed (one per line):"
        )
        try:
            resp = self.architect.client.chat(
                model=self.architect.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_ctx": 2048},
            )
            text = resp.get("message", {}).get("content", "")
            kinds = _parse_tool_lines(text)
            return len(kinds) if kinds else None
        except Exception:
            return None
```

- [ ] **Step 4: Verify tests pass**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_preflight_estimator.py -v`
Expected: all 14 tests pass (9 from Task 1 + 5 new).

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add multi_agent.py tests/test_preflight_estimator.py && git commit -m "feat(preflight): _estimate_tool_kinds method on MultiAgentSystem

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire pre-flight into `run()`

**Files:**
- Modify: `multi_agent.py` (insert a 5-line block in `MultiAgentSystem.run` between `_short_circuit_classify` and the existing fast-route branch)
- Create: `tests/test_preflight_routing.py`

- [ ] **Step 1: Write failing routing tests**

Create `/home/lulu/Projects/ezclaw/tests/test_preflight_routing.py`:

```python
"""Routing tests: verify that the pre-flight kind count gates the
fast-route branch in MultiAgentSystem.run."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_mas(monkeypatch, tmp_path):
    import multi_agent
    monkeypatch.setattr(multi_agent, "load_skills", lambda: [])
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "routing.db"))
    from multi_agent import MultiAgentSystem
    return MultiAgentSystem(session_id=None)


def _drain_until_status_or_end(gen, limit: int = 20):
    """Iterate the run() generator until either a status chunk arrives
    or the generator ends. Returns list of chunks yielded. Caller is
    responsible for closing the generator after."""
    chunks = []
    try:
        for _ in range(limit):
            chunk = next(gen)
            chunks.append(chunk)
            if chunk.get("type") == "status":
                # Stop after the first status — enough to verify routing.
                break
    except StopIteration:
        pass
    return chunks


def test_low_kind_count_fast_routes(monkeypatch, tmp_path):
    """When the pre-flight returns 1 kind, the fast-route status fires
    and the architect loop is NOT entered."""
    mas = _build_mas(monkeypatch, tmp_path)
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: "executor")
    monkeypatch.setattr(mas, "_estimate_tool_kinds", lambda u: 1)

    # Stub the executor's chat_stream so the run doesn't actually call
    # any LLM/tool — yield one content chunk and stop.
    executor = mas.agents["executor"]
    def fake_stream(_input):
        yield {"type": "content", "content": "ok"}
    monkeypatch.setattr(executor, "chat_stream", fake_stream)

    gen = mas.run("read foo.py")
    chunks = _drain_until_status_or_end(gen)
    gen.close()

    status_msgs = [c["content"] for c in chunks if c.get("type") == "status"]
    # The fast-route status should be present; the pre-flight status should NOT.
    assert any("fast-routed" in m for m in status_msgs)
    assert not any("pre-flight" in m for m in status_msgs)


def test_high_kind_count_skips_fast_route(monkeypatch, tmp_path):
    """When the pre-flight returns >= PREFLIGHT_KIND_THRESHOLD, the
    pre-flight status fires and the fast-route status does NOT."""
    from multi_agent import PREFLIGHT_KIND_THRESHOLD
    mas = _build_mas(monkeypatch, tmp_path)
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: "executor")
    monkeypatch.setattr(mas, "_estimate_tool_kinds", lambda u: PREFLIGHT_KIND_THRESHOLD)

    # The architect loop will try to call the architect's LLM; stub it
    # to yield an early termination so we don't run real LLM calls.
    # Simplest: monkeypatch the architect's reflect/plan to raise an
    # immediate stop. We catch the resulting exception in the iterator.
    def boom(*args, **kwargs):
        raise RuntimeError("architect short-circuit for test")
    monkeypatch.setattr(mas.architect, "reflect_and_plan", boom, raising=False)

    gen = mas.run("write me a snake game")
    chunks = _drain_until_status_or_end(gen, limit=5)
    try:
        gen.close()
    except Exception:
        pass

    status_msgs = [c["content"] for c in chunks if c.get("type") == "status"]
    assert any("pre-flight" in m and "planning" in m for m in status_msgs)
    assert not any("fast-routed" in m for m in status_msgs)


def test_preflight_none_skips_fast_route(monkeypatch, tmp_path):
    """When the pre-flight returns None (failure), the pre-flight status
    fires with '?' and the fast-route status does NOT."""
    mas = _build_mas(monkeypatch, tmp_path)
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: "executor")
    monkeypatch.setattr(mas, "_estimate_tool_kinds", lambda u: None)
    def boom(*args, **kwargs):
        raise RuntimeError("architect short-circuit for test")
    monkeypatch.setattr(mas.architect, "reflect_and_plan", boom, raising=False)

    gen = mas.run("do a thing")
    chunks = _drain_until_status_or_end(gen, limit=5)
    try:
        gen.close()
    except Exception:
        pass

    status_msgs = [c["content"] for c in chunks if c.get("type") == "status"]
    assert any("pre-flight: ?" in m for m in status_msgs)
    assert not any("fast-routed" in m for m in status_msgs)


def test_preflight_not_called_for_general_short_circuit(monkeypatch, tmp_path):
    """Pre-flight should ONLY run when _short_circuit_classify returned
    'executor'. For 'general' / 'researcher', it must NOT be called."""
    mas = _build_mas(monkeypatch, tmp_path)
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: "general")

    estimate_calls = {"count": 0}
    def counting_estimate(_input):
        estimate_calls["count"] += 1
        return 1
    monkeypatch.setattr(mas, "_estimate_tool_kinds", counting_estimate)

    general = mas.agents["general"]
    def fake_stream(_input):
        yield {"type": "content", "content": "hi"}
    monkeypatch.setattr(general, "chat_stream", fake_stream)

    gen = mas.run("hello")
    _drain_until_status_or_end(gen)
    gen.close()

    assert estimate_calls["count"] == 0
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_preflight_routing.py -v`
Expected: tests fail — the pre-flight wire-in doesn't exist yet, so high-kind-count and None cases will see a `fast-routed` status instead of a `pre-flight` status.

- [ ] **Step 3: Add the wire-in to `run()`**

In `/home/lulu/Projects/ezclaw/multi_agent.py`, find:

```python
        short_circuit_agent = self._short_circuit_classify(user_input)
        if short_circuit_agent and short_circuit_agent != "debugger":
```

(around line 1196). Insert this block BETWEEN those two lines:

```python
        short_circuit_agent = self._short_circuit_classify(user_input)
        if short_circuit_agent == "executor":
            kind_count = self._estimate_tool_kinds(user_input)
            if kind_count is None or kind_count >= PREFLIGHT_KIND_THRESHOLD:
                kind_label = "?" if kind_count is None else str(kind_count)
                yield {
                    "type": "status",
                    "content": f"[pre-flight: {kind_label} tool kinds — planning]\n",
                }
                short_circuit_agent = None
        if short_circuit_agent and short_circuit_agent != "debugger":
```

- [ ] **Step 4: Verify routing tests pass**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_preflight_routing.py -v`
Expected: all 4 tests pass.

Then run the full pre-flight suite for regression:

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_preflight_estimator.py tests/test_preflight_routing.py -v`
Expected: 18/18 pass (14 estimator + 4 routing).

- [ ] **Step 5: Run the existing action-tracking suite to confirm no collateral damage**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_action_helpers.py tests/test_actions_db.py tests/test_recall_actions_tool.py tests/test_action_dispatcher_integration.py -v`
Expected: every test passes (no regression from the routing change).

- [ ] **Step 6: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add multi_agent.py tests/test_preflight_routing.py && git commit -m "feat(preflight): gate executor fast-route on tool-kind count

When _short_circuit_classify returns 'executor', run a cheap LLM
pre-flight that estimates how many distinct tool kinds the request
needs. >= 3 kinds (or LLM failure) falls through to the architect
loop; < 3 fast-routes as before.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Manual smoke

**Files:** none — live verification.

- [ ] **Step 1: Start ezclaw with multi-agent on**

Run from another terminal:

```
cd /home/lulu/Projects/ezclaw && ENABLE_MULTI_AGENT=1 python cli.py
```

- [ ] **Step 2: Issue a single-tool-kind request**

Type: `read tools.py`

Expected: status line briefly shows `[executor] … (fast-routed)`. No `[pre-flight: ...]` status. Executor runs once, response comes back.

- [ ] **Step 3: Issue a multi-step request**

Type: `write a small flask app with one route`

Expected: status line shows `[pre-flight: N tool kinds — planning]` (where N is 3 or higher). Architect chip appears; plan panel appears; tasks get worked through one by one.

- [ ] **Step 4: Test the LLM-failure path**

Stop ollama (or set `OLLAMA_BASE_URL` to a bad address) and issue any executor-classified request like `write a function that prints hello`. Restart ollama after.

Expected: `[pre-flight: ? tool kinds — planning]` status appears. The architect loop then runs as normal.

- [ ] **Step 5: Note any anomalies**

If the threshold seems too low/high in practice, the `PREFLIGHT_KIND_THRESHOLD` constant near the top of `multi_agent.py` is the single tuning knob. Log observations for a follow-up tuning pass.

---

## Self-Review

**Spec coverage:**
- `_KNOWN_TOOL_NAMES` whitelist → Task 1 (constant placed before any consumer).
- `_parse_tool_lines` parser → Task 1.
- `_estimate_tool_kinds` method → Task 2.
- Pre-flight wire-in → Task 3 Step 3.
- Yields a `status` chunk visible to the UI → Task 3 Step 3 (`yield {"type": "status", "content": ...}`).
- LLM failure falls through to architect → Task 2 (`_estimate_tool_kinds` returns None on exception) + Task 3 (None triggers escalation).
- Pre-flight ONLY runs for `executor` short-circuit → Task 3 (`if short_circuit_agent == "executor"`).
- Unit + integration tests → Tasks 1 + 2 + 3.
- Threshold tuning knob → Task 1 (`PREFLIGHT_KIND_THRESHOLD` is a top-level constant).

**Placeholder scan:** No TBDs, no "implement appropriately" placeholders. Each step contains the complete code change.

**Type consistency:**
- `_parse_tool_lines` defined in Task 1 returns `set` → Task 2's `_estimate_tool_kinds` reads `len(kinds)` from it; matches.
- `_estimate_tool_kinds` returns `Optional[int]` → Task 3 checks `kind_count is None or kind_count >= PREFLIGHT_KIND_THRESHOLD`; matches.
- Status payload `{"type": "status", "content": "[pre-flight: ...]"}` matches the `chunk["type"] == "status"` handler already in `cli.py:2018`.
- `PREFLIGHT_KIND_THRESHOLD` used identically in Task 1 (definition) and Task 3 (consumption + test import).
- Whitelist `_KNOWN_TOOL_NAMES` used by Task 1's `_parse_tool_lines` and nowhere else; consistent.
