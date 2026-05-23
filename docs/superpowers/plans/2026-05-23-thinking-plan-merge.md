# Architect Intent + Plan Panel Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a plan is active, the architect's most recent intent renders nested under the in_progress task in the plan panel. The standalone architect chip disappears. Single-step (no-plan) mode keeps the standalone chip.

**Architecture:** All changes confined to `cli.py`. A new helper method `_render_intent_lines(intent)` returns indented Text lines. The standalone-chip render is gated on `self.current_plan is None`. The plan-panel loop appends intent lines under the in_progress task before its tool rows. A `plan_update` handler clears the stale intent when in_progress changes.

**Tech Stack:** Python 3, rich (already in use), pytest.

**Spec:** `docs/superpowers/specs/2026-05-23-thinking-plan-merge-design.md`

---

## File Structure

- **Modify:** `cli.py` — four small changes:
  1. New helper method `_render_intent_lines(intent)` on `ChatUI`.
  2. Gate the standalone-chip path on `self.current_plan is None` (around line 881).
  3. Inject intent lines under the in_progress task in `_render_plan_panel` (around line 1019-1046).
  4. Clear `self.architect_intent` on in_progress transitions inside the `plan_update` handler (around line 1996-2005).
- **Create:** `tests/test_render_intent_lines.py` — unit tests for the new helper.

---

## Task 1: `_render_intent_lines` helper + unit tests

**Files:**
- Modify: `cli.py` — add the method on `ChatUI`
- Create: `tests/test_render_intent_lines.py`

- [ ] **Step 1: Write the failing tests**

Create `/home/lulu/Projects/ezclaw/tests/test_render_intent_lines.py`:

```python
"""Unit tests for ChatUI._render_intent_lines (Spec D)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_ui_stub():
    """ChatUI.__new__ without invoking __init__; sufficient for testing
    _render_intent_lines because it touches no instance state beyond
    self.<helper functions>."""
    from cli import ChatUI
    return ChatUI.__new__(ChatUI)


def test_full_intent_returns_five_lines():
    ui = _make_ui_stub()
    intent = {
        "agent": "executor",
        "goal": "write the game skeleton",
        "observation": "pygame installed cleanly",
        "critical_thinking": "using 600x400 to keep tests fast",
        "reasoning": "I'll write the module then run it once",
    }
    lines = ui._render_intent_lines(intent)
    assert len(lines) == 5


def test_missing_reasoning_drops_that_line():
    ui = _make_ui_stub()
    intent = {
        "agent": "executor",
        "goal": "g",
        "observation": "o",
        "critical_thinking": "c",
    }
    lines = ui._render_intent_lines(intent)
    assert len(lines) == 4


def test_missing_agent_drops_that_line():
    ui = _make_ui_stub()
    intent = {
        "goal": "g",
        "observation": "o",
        "critical_thinking": "c",
        "reasoning": "r",
    }
    lines = ui._render_intent_lines(intent)
    assert len(lines) == 4


def test_empty_string_fields_are_omitted():
    ui = _make_ui_stub()
    intent = {
        "agent": "executor",
        "goal": "",
        "observation": "o",
        "critical_thinking": "",
        "reasoning": "r",
    }
    lines = ui._render_intent_lines(intent)
    assert len(lines) == 3  # agent + observation + reasoning


def test_empty_intent_returns_empty_list():
    ui = _make_ui_stub()
    assert ui._render_intent_lines({}) == []


def test_each_line_starts_with_eight_space_indent():
    ui = _make_ui_stub()
    intent = {
        "agent": "executor",
        "goal": "g",
        "observation": "o",
        "critical_thinking": "c",
        "reasoning": "r",
    }
    lines = ui._render_intent_lines(intent)
    for line in lines:
        # rich.text.Text.plain gives the raw string without styling.
        assert line.plain.startswith("        ")
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_render_intent_lines.py -v`
Expected: AttributeError — `ChatUI` has no `_render_intent_lines`.

- [ ] **Step 3: Add the helper to `cli.py`**

In `/home/lulu/Projects/ezclaw/cli.py`, find `class ChatUI` and locate the existing `_render_architect_intent` method (around line 1110). Insert this new method DIRECTLY AFTER `_render_architect_intent` returns (i.e., after the closing of that method's body):

```python
    def _render_intent_lines(self, intent: dict) -> list:
        """Return a list of Text lines representing the architect's intent,
        indented to sit under the active plan task row.

        The format mirrors the standalone architect chip but as inline
        lines rather than a Panel. Empty fields are omitted.
        """
        lines = []
        if not intent:
            return lines
        agent = intent.get("agent")
        if agent:
            role_color = THEME.role(agent).color
            t = Text()
            t.append("        agent: ", style=f"dim {DIM}")
            t.append(agent, style=f"bold {role_color}")
            lines.append(t)
        for field, label, style, italic in (
            ("goal", "goal", PRIMARY, False),
            ("observation", "observation", WARN, False),
            ("critical_thinking", "critical", ACCENT, False),
            ("reasoning", "reasoning", DIM, True),
        ):
            value = intent.get(field)
            if value:
                t = Text()
                t.append(f"        {label}: ", style=f"bold {style}")
                t.append(value, style="italic" if italic else "")
                lines.append(t)
        return lines
```

Verify imports: `THEME`, `Text`, `DIM`, `PRIMARY`, `WARN`, `ACCENT` are all already imported at the top of `cli.py` (search to confirm; they're used elsewhere in the file). No new imports needed.

- [ ] **Step 4: Verify tests pass**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_render_intent_lines.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add cli.py tests/test_render_intent_lines.py && git commit -m "feat(intent-merge): _render_intent_lines helper

Returns inline Text lines for the architect's intent, indented to sit
under the in_progress plan task. Used by the plan panel renderer in
the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Suppress standalone chip when plan is active

**Files:**
- Modify: `cli.py` (around line 881)

- [ ] **Step 1: Read the current gate**

Run: `cd /home/lulu/Projects/ezclaw && grep -n "architect_intent" cli.py | head -10`
Expected: shows the render call site around line 881 — `if self.architect_intent:` followed by `parts.append(self._render_architect_intent(self.architect_intent))`.

- [ ] **Step 2: Change the gate**

In `/home/lulu/Projects/ezclaw/cli.py`, find:

```python
        if self.architect_intent:
            parts.append(self._render_architect_intent(self.architect_intent))
```

Change to:

```python
        if self.architect_intent and self.current_plan is None:
            parts.append(self._render_architect_intent(self.architect_intent))
```

(Only the condition changes; the body is unchanged.)

- [ ] **Step 3: Verify syntax**

Run: `cd /home/lulu/Projects/ezclaw && python -m py_compile cli.py`
Expected: compiles cleanly.

- [ ] **Step 4: Run existing tests for regression**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_render_intent_lines.py tests/test_action_dispatcher_integration.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add cli.py && git commit -m "feat(intent-merge): suppress standalone architect chip when plan active

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Nest intent under in_progress task in `_render_plan_panel`

**Files:**
- Modify: `cli.py` (the per-task loop in `_render_plan_panel`, around line 1019-1046)

- [ ] **Step 1: Locate the per-task loop**

Run: `cd /home/lulu/Projects/ezclaw && grep -n "for task in plan.tasks\|tools_by_task" cli.py | head -10`
Expected: shows the loop in `_render_plan_panel` and the tool-row append site.

- [ ] **Step 2: Insert the intent injection**

In `/home/lulu/Projects/ezclaw/cli.py`, find the block where tool rows are appended under each task (around line 1045-1046):

```python
            # Tool rows nested under this task. ...
            for tidx, tool in tools_by_task.get(task.id, []):
                body_lines.append(self._render_nested_tool_row(tidx, tool))
```

Just BEFORE the `for tidx, tool` loop, insert:

```python
            # If this task is in_progress, nest the architect's most recent
            # intent under it (Spec D — replaces the standalone chip).
            if task.status == "in_progress" and self.architect_intent:
                body_lines.extend(self._render_intent_lines(self.architect_intent))
```

The final structure should be:

```python
        for task in plan.tasks:
            ...
            body_lines.append(line_text)

            if task.status == "in_progress" and self.architect_intent:
                body_lines.extend(self._render_intent_lines(self.architect_intent))

            # Tool rows nested under this task. ...
            for tidx, tool in tools_by_task.get(task.id, []):
                body_lines.append(self._render_nested_tool_row(tidx, tool))
```

- [ ] **Step 3: Verify syntax**

Run: `cd /home/lulu/Projects/ezclaw && python -m py_compile cli.py`
Expected: compiles cleanly.

- [ ] **Step 4: Run regression sweep**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_render_intent_lines.py tests/test_action_dispatcher_integration.py tests/test_preflight_estimator.py tests/test_preflight_routing.py -v`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add cli.py && git commit -m "feat(intent-merge): nest architect intent under in_progress task

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Clear stale intent on in_progress transition

**Files:**
- Modify: `cli.py` (the `plan_update` handler, around line 1996-2005)

- [ ] **Step 1: Locate the plan_update handler**

Run: `cd /home/lulu/Projects/ezclaw && grep -n "plan_update\|_last_task_states" cli.py | head -10`
Expected: shows the handler where the existing flash-state tracking lives.

- [ ] **Step 2: Capture prev in_progress before plan replacement, clear intent on transition**

In `/home/lulu/Projects/ezclaw/cli.py`, find the existing `plan_update` block (around lines 1995-2005):

```python
                elif chunk["type"] == "plan_update":
                    new_plan = chunk["plan"]
                    now = time.time()
                    for task in new_plan.tasks:
                        prev = self._last_task_states.get(task.id)
                        if prev is not None and prev != task.status:
                            self._task_flash_until[task.id] = now + 0.15
                        self._last_task_states[task.id] = task.status
                    self.current_plan = new_plan
```

Change to:

```python
                elif chunk["type"] == "plan_update":
                    new_plan = chunk["plan"]
                    now = time.time()
                    prev_in_progress = (
                        {t.id for t in self.current_plan.tasks if t.status == "in_progress"}
                        if self.current_plan else set()
                    )
                    for task in new_plan.tasks:
                        prev = self._last_task_states.get(task.id)
                        if prev is not None and prev != task.status:
                            self._task_flash_until[task.id] = now + 0.15
                        self._last_task_states[task.id] = task.status
                    self.current_plan = new_plan
                    new_in_progress = {t.id for t in new_plan.tasks if t.status == "in_progress"}
                    if new_in_progress != prev_in_progress:
                        # The previously-active task moved; any held intent is
                        # about the prior task. Wait for the next intent chunk.
                        self.architect_intent = None
```

- [ ] **Step 3: Verify syntax**

Run: `cd /home/lulu/Projects/ezclaw && python -m py_compile cli.py`
Expected: compiles cleanly.

- [ ] **Step 4: Run all tests**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_render_intent_lines.py tests/test_action_dispatcher_integration.py tests/test_preflight_estimator.py tests/test_preflight_routing.py tests/test_continuation_phrases.py tests/test_halt_continuation.py -v`
Expected: every test passes (no regression).

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add cli.py && git commit -m "feat(intent-merge): clear architect intent on in_progress transition

When plan_update changes which task is in_progress, the stored intent
is stale (it described the previous task). Clear it; wait for the
architect's next intent chunk.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Manual smoke

**Files:** none.

- [ ] **Step 1: Run ezclaw in multi-agent mode**

```bash
cd /home/lulu/Projects/ezclaw && ENABLE_MULTI_AGENT=1 python cli.py
```

- [ ] **Step 2: Trigger a multi-step request**

Type: `write a small flask app with one route and a unit test`

Expected:
- Pre-flight engages the architect (per Spec C).
- Plan panel appears with several tasks.
- As the architect picks each task, its intent (goal/observation/critical/reasoning) renders nested UNDER that task row.
- The standalone architect chip is NOT visible anywhere in the layout.
- When the task moves to `done`, its row no longer carries the intent block; the next in_progress task gets it instead.

- [ ] **Step 3: Trigger a single-step (no-plan) flow**

Type a short conversational message that won't trigger a plan: `hi, what's the date?`

Expected: the architect handles this in single-step mode. The standalone architect chip DOES appear (no-plan branch). No regression.

---

## Self-Review

**Spec coverage:**
- Standalone chip suppressed when plan active — Task 2.
- Intent nested under in_progress task — Task 3.
- New `_render_intent_lines` helper — Task 1.
- Stale-intent clearing on in_progress transition — Task 4.
- Single-step (no-plan) mode unchanged — Task 2's gate uses `self.current_plan is None`; preserved.
- Tests for helper — Task 1.
- Manual smoke — Task 5.

**Placeholder scan:** No placeholders; every code change shown verbatim.

**Type consistency:**
- `_render_intent_lines` returns `list[Text]` (Task 1) → consumed via `extend` in Task 3.
- `self.architect_intent` is a dict (existing) → used by `_render_intent_lines` in Task 1 + cleared in Task 4.
- `self.current_plan` is the existing Plan object → consulted in Tasks 2 + 3 + 4 with consistent attribute access (`.tasks`, `.tasks[i].status`, `.tasks[i].id`).
- `THEME`, `Text`, `DIM`, `PRIMARY`, `WARN`, `ACCENT` are existing imports in `cli.py` (Task 1 note); no new imports anywhere.
