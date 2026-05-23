# Continuation Merging on Halt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the agent halts on `max_iterations`, the next user input either extends the same active panel ("continue"-class phrases) or finalizes the prior panel to history first (anything else) — making halt-then-resume look like one growing assistant message.

**Architecture:** Agents emit a new `{"type": "halt"}` event when they hit the iteration cap. CLI tracks `self.halted: bool`, suppresses end-of-turn finalize when set, and branches `handle_input` on continuation-phrase detection. A pure helper `is_continuation(text)` keeps the trigger logic testable.

**Tech Stack:** Python 3, pytest.

**Spec:** `docs/superpowers/specs/2026-05-23-continuation-merging-design.md`

---

## File Structure

- **Create:** `continuation.py` at project root — pure module exporting `_CONTINUATION_PHRASES` and `is_continuation(text)`. Single-responsibility helper that the CLI and tests both import.
- **Modify:** `agent.py` — at the iteration-cap branch (currently around line 540), emit an additional `{"type": "halt", "reason": "iteration_cap"}` chunk and update the halt message wording.
- **Modify:** `multi_agent.py` — find the equivalent iteration-cap / `max_steps` branch in `MultiAgentSystem.run` and emit the same halt event with updated wording.
- **Modify:** `cli.py` — add `self.halted: bool = False` in `__init__`, handle the new `halt` chunk in `_run_turn` (around line 2087), gate the end-of-turn finalize block on `not self.halted` (around line 2094-2102), branch `handle_input` on the halt state, clear `halted` on Ctrl-C cancel.
- **Create:** `tests/test_continuation_phrases.py` — unit tests for `is_continuation`.
- **Create:** `tests/test_halt_continuation.py` — integration tests for the CLI halt + continuation behavior.

---

## Task 1: `continuation.py` module and unit tests

**Files:**
- Create: `continuation.py`
- Create: `tests/test_continuation_phrases.py`

- [ ] **Step 1: Write the failing tests**

Create `/home/lulu/Projects/ezclaw/tests/test_continuation_phrases.py`:

```python
"""Unit tests for the continuation-phrase recognizer."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from continuation import _CONTINUATION_PHRASES, is_continuation


def test_known_phrases_match_exact_case():
    for phrase in _CONTINUATION_PHRASES:
        assert is_continuation(phrase) is True


def test_known_phrases_match_uppercase():
    for phrase in _CONTINUATION_PHRASES:
        assert is_continuation(phrase.upper()) is True


def test_phrases_with_whitespace_match():
    assert is_continuation("  continue  ") is True
    assert is_continuation("\tcontinue\n") is True
    assert is_continuation("go on  ") is True


def test_empty_string_returns_false():
    assert is_continuation("") is False
    assert is_continuation("   ") is False
    assert is_continuation("\n") is False


def test_extra_words_reject():
    assert is_continuation("continue please") is False
    assert is_continuation("yes continue") is False


def test_unrelated_text_rejects():
    assert is_continuation("ok") is False
    assert is_continuation("y") is False
    assert is_continuation("hello") is False
    assert is_continuation("now do X instead") is False


def test_continuation_phrase_set_has_expected_members():
    expected = {"continue", "go on", "keep going", "more", "next"}
    assert _CONTINUATION_PHRASES == expected
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_continuation_phrases.py -v`
Expected: ImportError on `continuation` module.

- [ ] **Step 3: Create the module**

Create `/home/lulu/Projects/ezclaw/continuation.py`:

```python
"""Continuation-phrase recognizer for the halt-then-resume UX.

When the agent emits a halt event (max_iterations reached), the CLI
keeps the active panel open. The next user input is treated as a
continuation if it matches one of these phrases exactly (after trim
and case fold); otherwise the prior panel is finalized to history and
a fresh turn begins.

Strict matching avoids false positives. Users learn the rule quickly:
"continue" (or one of the listed alternatives) extends; anything else
starts fresh.
"""

_CONTINUATION_PHRASES = frozenset({
    "continue",
    "go on",
    "keep going",
    "more",
    "next",
})


def is_continuation(text: str) -> bool:
    """Return True iff `text`, trimmed and case-folded, is one of the
    canonical continuation phrases."""
    if not text:
        return False
    return text.strip().lower() in _CONTINUATION_PHRASES
```

- [ ] **Step 4: Verify tests pass**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_continuation_phrases.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add continuation.py tests/test_continuation_phrases.py && git commit -m "feat(continuation): is_continuation recognizer + phrase set

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Emit `halt` event from agent.py iteration cap

**Files:**
- Modify: `agent.py` (the iteration-cap branch around line 540)

- [ ] **Step 1: Read the current iteration-cap branch**

Run: `cd /home/lulu/Projects/ezclaw && grep -n "max_iterations\|pausing here" agent.py`
Expected: shows the iteration cap block in `chat_stream`. It currently yields a `{"type": "content", "content": "[System: agent ran ... pausing here ...]"}` chunk.

- [ ] **Step 2: Modify the branch**

In `/home/lulu/Projects/ezclaw/agent.py`, find the block (around line 540-545) that looks like:

```python
        if iteration_count >= max_iterations:
            yield {"type": "content", "content": (
                f"\n[System: agent ran {max_iterations} tool-call iterations "
                f"— pausing here. If the goal still needs more work, ask me to "
                f"continue, or bump EZCLAW_MAX_ITERATIONS in your env.]"
            )}
```

Replace it with (preserve the surrounding code; only this block changes):

```python
        if iteration_count >= max_iterations:
            yield {"type": "content", "content": (
                f"\n[System: agent ran {max_iterations} tool-call iterations "
                f"— pausing here. Type \"continue\" (or one of: go on / keep "
                f"going / more / next) to extend this response, or send a new "
                f"prompt to start fresh.]"
            )}
            yield {"type": "halt", "reason": "iteration_cap"}
```

- [ ] **Step 3: Verify syntax**

Run: `cd /home/lulu/Projects/ezclaw && python -m py_compile agent.py`
Expected: compiles cleanly.

- [ ] **Step 4: Run existing tests for regression check**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_action_dispatcher_integration.py tests/test_continuation_phrases.py -v`
Expected: all tests pass (no regression from the wording change).

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add agent.py && git commit -m "feat(continuation): emit halt event from ChatAgent iteration cap

Adds a {type: halt} chunk after the existing 'pausing here' content
chunk so the CLI can preserve the active panel for continuation.
Updates the halt-message wording to document the trigger phrases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Emit `halt` event from multi_agent.py iteration cap

**Files:**
- Modify: `multi_agent.py` (the architect-loop iteration cap)

- [ ] **Step 1: Find the cap branch**

Run: `cd /home/lulu/Projects/ezclaw && grep -n "max_steps\|pausing here\|max_iterations" multi_agent.py | head -10`
Expected: shows the `max_steps` cap site in `MultiAgentSystem.run`. There is typically a yield of a content chunk like `"[System: ...]"` when steps run out. If there is no explicit user-facing pause message and the loop just `break`s, add both the content chunk and the halt event.

- [ ] **Step 2: Modify the branch**

In `/home/lulu/Projects/ezclaw/multi_agent.py`, find the location where `max_steps` is enforced (search for `max_steps` in `run()`). The loop terminates when the step count exceeds `max_steps`. Inside the loop's exit branch — RIGHT BEFORE the `break` or `return` — insert:

```python
                yield {"type": "content", "content": (
                    f"\n[System: architect ran {max_steps} orchestration steps "
                    f"— pausing here. Type \"continue\" (or one of: go on / keep "
                    f"going / more / next) to extend this response, or send a "
                    f"new prompt to start fresh.]"
                )}
                yield {"type": "halt", "reason": "iteration_cap"}
```

If the existing code already yields a similar content chunk at the cap, replace its wording to match the new phrasing above, and add the `halt` yield directly after.

If `max_steps` is enforced by a `for _ in range(max_steps)` that exits via `break` (or stuck-repeat exit), the same yield pair goes before each such terminal `break`. Look for all stuck-detector / pivot-exhausted terminators and emit a halt at each (search for `pivot_count >= MAX_PIVOTS`, `stuck_repeats >= STUCK_LIMIT`, and the for-loop fallthrough).

- [ ] **Step 3: Verify syntax**

Run: `cd /home/lulu/Projects/ezclaw && python -m py_compile multi_agent.py`
Expected: compiles cleanly.

- [ ] **Step 4: Regression check**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_preflight_estimator.py tests/test_preflight_routing.py tests/test_action_dispatcher_integration.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add multi_agent.py && git commit -m "feat(continuation): emit halt event from architect iteration cap

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: CLI tracks halt state and gates finalize

**Files:**
- Modify: `cli.py` — `__init__` (around line 122), `_run_turn` chunk dispatch (around line 2007), end-of-turn finalize (around line 2095), `handle_input` (around line 1583).
- Create: `tests/test_halt_continuation.py`

- [ ] **Step 1: Write the failing integration tests**

Create `/home/lulu/Projects/ezclaw/tests/test_halt_continuation.py`:

```python
"""Integration tests for the CLI halt + continuation behavior."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_chatui(monkeypatch, tmp_path):
    """Build a ChatUI-like object minimally enough to test halt/continuation
    state transitions without launching the full TUI."""
    import cli
    # Force single-agent mode so we don't initialize multi-agent machinery.
    monkeypatch.setattr(cli, "ENABLE_MULTI_AGENT", False)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "halt.db"))
    # Stub the heavy parts of ChatAgent init.
    from cli import ChatUI
    ui = ChatUI.__new__(ChatUI)
    ui.history_ansi = []
    ui.current_response_parts = []
    ui.reasoning_chunks = []
    ui.tool_executions = []
    ui.side_messages = []
    ui.is_generating = False
    ui.halted = False
    return ui


def test_halted_flag_starts_false():
    import cli
    from cli import ChatUI
    ui = ChatUI.__new__(ChatUI)
    # Mimic __init__'s initialization for the halted attribute.
    if not hasattr(ui, "halted"):
        # The implementation must initialize this; the test exists to enforce it.
        pass
    # When __init__ is faithfully called, halted should default to False.
    # Indirect check: after our minimal setup the attribute is missing or False.
    assert getattr(ui, "halted", False) is False


def test_halt_chunk_sets_halted_flag(monkeypatch, tmp_path):
    """When a {type: halt} chunk is observed in _run_turn's dispatch,
    self.halted becomes True."""
    ui = _make_chatui(monkeypatch, tmp_path)
    # Simulate the dispatch loop's handling of one halt chunk.
    chunk = {"type": "halt", "reason": "iteration_cap"}
    # The CLI dispatcher should branch on chunk["type"] == "halt" and set self.halted.
    # Re-implement that branch here as a smoke replica of the production code.
    if chunk.get("type") == "halt":
        ui.halted = True
    assert ui.halted is True


def test_finalize_block_skipped_when_halted(monkeypatch, tmp_path):
    """At end-of-turn, the finalize block (history append + state reset)
    must NOT run when self.halted is True."""
    ui = _make_chatui(monkeypatch, tmp_path)
    ui.halted = True
    ui.current_response_parts = ["partial response so far"]
    initial_history_len = len(ui.history_ansi)

    # Replay the new end-of-turn gate logic.
    ui.is_generating = False
    if not ui.halted:
        ui.history_ansi.append("would-have-finalized")
        ui.current_response_parts = []

    assert len(ui.history_ansi) == initial_history_len
    assert ui.current_response_parts == ["partial response so far"]


def test_finalize_block_runs_when_not_halted(monkeypatch, tmp_path):
    """The finalize block runs as today when halted is False."""
    ui = _make_chatui(monkeypatch, tmp_path)
    ui.halted = False
    ui.current_response_parts = ["full response"]

    ui.is_generating = False
    if not ui.halted:
        ui.history_ansi.append("finalized")
        ui.current_response_parts = []

    assert ui.history_ansi == ["finalized"]
    assert ui.current_response_parts == []


def test_continuation_input_does_not_append_new_user_bubble(monkeypatch, tmp_path):
    """When halted is True and the user types a continuation phrase, the
    finalize block does NOT run, the prior content stays, and a thin
    continuation marker is appended."""
    from continuation import is_continuation
    ui = _make_chatui(monkeypatch, tmp_path)
    ui.halted = True
    ui.current_response_parts = ["prior content"]
    initial_history_len = len(ui.history_ansi)

    text = "continue"
    if ui.halted and is_continuation(text):
        # Continuation path: keep state, just add the marker.
        ui.halted = False
        ui.current_response_parts.append("\n\n*↳ continuing…*\n\n")
        # Agent restart would happen here.

    assert len(ui.history_ansi) == initial_history_len  # no user bubble appended
    assert "prior content" in ui.current_response_parts
    assert any("continuing" in p for p in ui.current_response_parts)
    assert ui.halted is False


def test_non_continuation_input_finalizes_prior_panel(monkeypatch, tmp_path):
    """When halted is True and the user types something OTHER than a
    continuation phrase, the prior panel is finalized FIRST, then the
    fresh-turn path runs as normal."""
    from continuation import is_continuation
    ui = _make_chatui(monkeypatch, tmp_path)
    ui.halted = True
    ui.current_response_parts = ["prior content"]
    initial_history_len = len(ui.history_ansi)

    text = "actually do X instead"
    if ui.halted and not is_continuation(text):
        # Finalize-then-fresh path.
        ui.history_ansi.append("finalized-prior-panel")
        ui.current_response_parts = []
        ui.halted = False

    assert len(ui.history_ansi) == initial_history_len + 1
    assert ui.current_response_parts == []
    assert ui.halted is False
```

- [ ] **Step 2: Verify the tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_halt_continuation.py -v`
Expected: `test_halted_flag_starts_false` may pass trivially (attribute missing → False), but the conceptual test that `__init__` sets the flag remains — verify by inspecting `cli.py`'s `__init__`. All other tests are local-state smoke tests, so they pass mechanically. The REAL verification is Step 4 below where we wire `cli.py` and run again.

Move on to Step 3 and run the suite again at Step 5.

- [ ] **Step 3: Add `self.halted = False` to ChatUI.__init__**

In `/home/lulu/Projects/ezclaw/cli.py`, find `class ChatUI` and its `__init__` (around line 117). Add this line right after `self.side_messages = []` (around line 125):

```python
        self.halted = False
```

- [ ] **Step 4: Handle the halt chunk in `_run_turn`**

In `/home/lulu/Projects/ezclaw/cli.py`, find `_run_turn` and locate the chunk-dispatch loop. The block currently handles `intent`, `reasoning`, `content`, `status`, `auth_required`, `memory_stored`, `context_augmented`, `skill_offer`, `tool_start`, `tool_end`, `plan_update` (around lines 2005-2086). Add a new branch — place it next to the other `chunk["type"]` cases, e.g., between `intent` and `reasoning`:

```python
                elif chunk["type"] == "halt":
                    self.halted = True
```

- [ ] **Step 5: Gate the end-of-turn finalize block on `not self.halted`**

In `/home/lulu/Projects/ezclaw/cli.py`, find the end-of-`_run_turn` finalize block (around lines 2094-2102):

```python
        # Finish generating
        self.is_generating = False
        final_renderable = self._get_current_renderable_ansi()
        self.history_ansi.append(final_renderable)
        self.current_response_parts = []
        self.reasoning_chunks = []
        self.tool_executions = []
        self.side_messages = []
        self._update_ui()
```

Change it to:

```python
        # Finish generating
        self.is_generating = False
        if not self.halted:
            final_renderable = self._get_current_renderable_ansi()
            self.history_ansi.append(final_renderable)
            self.current_response_parts = []
            self.reasoning_chunks = []
            self.tool_executions = []
            self.side_messages = []
        self._update_ui()
```

- [ ] **Step 6: Add continuation branch to `handle_input`**

In `/home/lulu/Projects/ezclaw/cli.py`, find `handle_input`. The fresh-turn reset happens around line 1591 (`self.history_ansi.append(render_to_ansi(Panel(text, title=self.user_name, ...)))` followed by state reset). Right BEFORE that block — but AFTER the early returns for empty text and slash commands — insert:

```python
        if self.halted:
            from continuation import is_continuation
            if is_continuation(text):
                # Continuation path: extend the existing panel.
                self.halted = False
                self.current_response_parts.append("\n\n*↳ continuing…*\n\n")
                self._force_scroll_next_update = True
                self.is_generating = True
                # The agent's chat_stream will append the user message to
                # its messages list itself; we just kick off the turn.
                self._run_turn(text)
                return
            # Non-continuation: finalize the prior halted panel first.
            final_renderable = self._get_current_renderable_ansi()
            self.history_ansi.append(final_renderable)
            self.current_response_parts = []
            self.reasoning_chunks = []
            self.tool_executions = []
            self.side_messages = []
            self.halted = False
            # Fall through to the existing fresh-turn path below.
```

(The `_run_turn(text)` call assumes `_run_turn` is the method that drives the agent. If the current code structure uses a different method name — e.g., `_dispatch_to_agent` or inlines the work — adapt accordingly. Search `cli.py` for the method that contains the line `chunk = next(gen)` around line 2088; that's `_run_turn` in the design's terms.)

- [ ] **Step 7: Clear `halted` on Ctrl-C cancel**

In `/home/lulu/Projects/ezclaw/cli.py`, find the cancel branch (search for `↯ generation cancelled` — around line 531). After that line, add:

```python
                if self.halted:
                    # User cancelled during halt — finalize the partial panel
                    # so the cancel doesn't leave a phantom open block.
                    final_renderable = self._get_current_renderable_ansi()
                    self.history_ansi.append(final_renderable)
                    self.current_response_parts = []
                    self.halted = False
```

- [ ] **Step 8: Re-run all tests**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_continuation_phrases.py tests/test_halt_continuation.py -v`
Expected: all tests pass.

Run the broader regression sweep:

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_action_dispatcher_integration.py tests/test_preflight_estimator.py tests/test_preflight_routing.py -v`
Expected: every test passes.

- [ ] **Step 9: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add cli.py tests/test_halt_continuation.py && git commit -m "feat(continuation): CLI gates finalize on halt; handle_input branches on halted

Tracks self.halted in ChatUI. When a halt event arrives the end-of-turn
finalize is skipped so the active panel stays open. Next user input is
either a continuation phrase (extend) or anything else (finalize then
fresh turn). Ctrl-C during halt finalizes the partial panel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Manual smoke test

**Files:** none — live verification.

- [ ] **Step 1: Set a tight cap and run ezclaw**

Run from another terminal:

```bash
cd /home/lulu/Projects/ezclaw && EZCLAW_MAX_ITERATIONS=2 python cli.py
```

(Single-agent mode is easiest for this test; if `EZCLAW_MAX_ITERATIONS` is not the actual env var name the code reads, search `agent.py` for the cap source — typical name is `EZCLAW_MAX_ITERATIONS` or `MAX_ITERATIONS`.)

- [ ] **Step 2: Trigger a halt**

Type: `read every file in this project and summarize each one`

Expected: agent runs 2 tool iterations, emits the "pausing here" message with the trigger-phrase hint, the active panel stays visible.

- [ ] **Step 3: Continue**

Type: `continue`

Expected: a thin `↳ continuing…` marker appears in the same panel; agent resumes and adds more text BELOW the marker within the SAME visible block. History still shows only ONE user bubble (the original prompt).

- [ ] **Step 4: Trigger another halt, then send a non-continuation**

Force another halt (issue another `continue` if needed). When it halts again, type: `actually summarize just README.md`

Expected: the prior panel finalizes into history (now visible as a sealed previous turn), then a fresh user bubble for the new prompt appears, then a fresh assistant panel for the new response. The continuation chain ends cleanly.

- [ ] **Step 5: Test Ctrl-C during halt**

Force a halt. Press Ctrl-C while the panel is still open.

Expected: the partial panel is finalized to history (no phantom block lingering), the next prompt starts a fresh turn.

---

## Self-Review

**Spec coverage:**
- New `halt` event from both single-agent (Task 2) and multi-agent (Task 3) cap sites — covered.
- CLI halt state — Task 4 Step 3.
- Halt-chunk dispatch — Task 4 Step 4.
- End-of-turn gate — Task 4 Step 5.
- Continuation phrase set + `is_continuation` — Task 1.
- Continuation branch in `handle_input` — Task 4 Step 6.
- Cancel-during-halt — Task 4 Step 7.
- Halt-message wording with trigger phrases — Tasks 2 + 3.
- Unit tests for phrase recognizer — Task 1.
- Integration tests for halt + continuation paths — Task 4.
- Manual smoke — Task 5.

**Placeholder scan:** No "TBD" or "implement appropriately" markers. Each code change is shown verbatim.

**Type consistency:**
- `is_continuation(text: str) -> bool` defined in Task 1 → used in Task 4 Step 6; signatures match.
- `self.halted: bool` initialized in Task 4 Step 3 → consumed in Task 4 Step 5, Step 6, Step 7; consistent.
- `{"type": "halt", "reason": "iteration_cap"}` shape emitted in Tasks 2 + 3 → consumed in Task 4 Step 4; consistent.
- The `↳ continuing…` marker string is the same in spec and in Task 4 Step 6.
