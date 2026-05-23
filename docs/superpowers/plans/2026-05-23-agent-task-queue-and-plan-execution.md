# Agent Task Queue & Plan Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the architect's prose `plan` string with a structured, stable task queue that's generated once per multi-step user request, executed sequentially against task-level status updates from the architect, and rendered in the TUI as a panel above the architect chip with role-aware icons and flash-on-transition animation.

**Architecture:** New `plan.py` module owns the `Task` and `Plan` dataclasses (pure data, no I/O). The `Architect` class in `multi_agent.py` is split into `plan()` and `execute()` methods sharing a rewritten system prompt covering both modes. `MultiAgentSystem.run` calls `plan()` once, then loops through `execute()` calls that produce structured intents with `task_updates`/`new_tasks`/`current_task_id`. The UI subscribes to two new chunk types (`plan_created`, `plan_update`) and renders a sticky plan panel above the chip.

**Tech Stack:** Python, `rich` (Panel, Text, Group), `prompt_toolkit`, `ollama` Python SDK, pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-23-agent-task-queue-and-plan-execution-design.md`

---

## File Structure

- **Create:** `plan.py` — `Task` and `Plan` dataclasses with helpers (`advance`, `insert`, `is_complete`, `get_task`, `progress`). Pure data, no I/O.
- **Create:** `tests/test_plan.py` — unit tests for `Plan` mutators and queries.
- **Create:** `tests/test_architect_plan.py` — smoke tests for `Architect.plan()` and `Architect.execute()` with a stubbed Ollama client.
- **Modify:** `multi_agent.py` — Architect system prompt rewrite; new `Architect.plan()` and `Architect.execute()` methods; `analyze()` removed; `MultiAgentSystem.run` restructured to call planning once then loop through execute. ~150 lines changed.
- **Modify:** `cli.py` — new instance state (`current_plan`, `_last_task_states`, `_task_flash_until`); new chunk handlers in `_agent_worker`; new `_render_plan_panel` method; insertion of plan panel above the chip in `_get_current_renderable_ansi`. ~60 lines changed.
- **Modify:** `theme.py` — add `TASK_STATE_STYLE` and `TASK_STATE_FLASH` lookup tables. ~15 lines.
- **Unchanged:** `agent.py`, `tools.py`, `memory.py`, `model_client.py`.

---

## Task 1: Create `plan.py` data model with TDD

**Files:**
- Create: `tests/test_plan.py`
- Create: `plan.py`

- [ ] **Step 1: Write the failing tests**

Create `/home/lulu/Projects/ezclaw/tests/test_plan.py` with this exact content:

```python
import pytest
from plan import Plan, Task, TASK_STATUSES


def make_plan(n=3, title="test plan"):
    return Plan(
        title=title,
        tasks=[Task(id=i, description=f"task {i}") for i in range(1, n + 1)],
    )


def test_task_defaults_to_pending():
    t = Task(id=1, description="hello")
    assert t.status == "pending"


def test_plan_get_task_returns_task_by_id():
    p = make_plan(3)
    assert p.get_task(2).description == "task 2"
    assert p.get_task(99) is None


def test_advance_to_in_progress_sets_current_task_id():
    p = make_plan(3)
    p.advance(2, "in_progress")
    assert p.get_task(2).status == "in_progress"
    assert p.current_task_id == 2


def test_advance_from_in_progress_to_done_clears_current_task_id():
    p = make_plan(3)
    p.advance(2, "in_progress")
    p.advance(2, "done")
    assert p.get_task(2).status == "done"
    assert p.current_task_id is None


def test_advance_to_failed_clears_current_task_id_if_matching():
    p = make_plan(3)
    p.advance(2, "in_progress")
    p.advance(2, "failed")
    assert p.current_task_id is None


def test_advance_to_skipped_clears_current_task_id_if_matching():
    p = make_plan(3)
    p.advance(2, "in_progress")
    p.advance(2, "skipped")
    assert p.current_task_id is None


def test_advance_with_invalid_status_is_noop():
    p = make_plan(3)
    p.advance(1, "bogus")
    assert p.get_task(1).status == "pending"


def test_advance_with_unknown_task_id_is_noop():
    p = make_plan(3)
    p.advance(99, "done")
    assert all(t.status == "pending" for t in p.tasks)


def test_insert_after_existing_task_assigns_new_id():
    p = make_plan(3)
    new_task = p.insert(2, "inserted")
    assert new_task.id == 4  # max existing id + 1
    assert new_task.description == "inserted"
    # Task should be inserted immediately after task 2
    ids_in_order = [t.id for t in p.tasks]
    assert ids_in_order == [1, 2, 4, 3]


def test_insert_after_unknown_id_appends_to_end():
    p = make_plan(3)
    new_task = p.insert(99, "appended")
    assert p.tasks[-1] is new_task


def test_is_complete_false_when_any_pending():
    p = make_plan(3)
    p.advance(1, "done")
    p.advance(2, "done")
    assert not p.is_complete()


def test_is_complete_true_when_all_done():
    p = make_plan(3)
    for i in range(1, 4):
        p.advance(i, "done")
    assert p.is_complete()


def test_is_complete_true_when_mix_of_done_and_skipped():
    p = make_plan(3)
    p.advance(1, "done")
    p.advance(2, "skipped")
    p.advance(3, "done")
    assert p.is_complete()


def test_is_complete_false_when_any_failed():
    p = make_plan(3)
    p.advance(1, "done")
    p.advance(2, "failed")
    p.advance(3, "done")
    assert not p.is_complete()


def test_progress_counts_done_and_skipped():
    p = make_plan(4)
    p.advance(1, "done")
    p.advance(2, "skipped")
    p.advance(3, "in_progress")
    done, total = p.progress()
    assert done == 2
    assert total == 4


def test_task_statuses_constant():
    assert "pending" in TASK_STATUSES
    assert "in_progress" in TASK_STATUSES
    assert "done" in TASK_STATUSES
    assert "failed" in TASK_STATUSES
    assert "skipped" in TASK_STATUSES
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/lulu/Projects/ezclaw
pytest tests/test_plan.py -v
```

Expected: All tests fail with `ModuleNotFoundError: No module named 'plan'` (the `plan.py` file doesn't exist yet).

- [ ] **Step 3: Create the `plan.py` module**

Create `/home/lulu/Projects/ezclaw/plan.py` with this exact content:

```python
"""Pure data model for the agent's task queue.

This module is import-time only — no I/O, no rendering, no LLM calls.
It exports `Task` and `Plan` dataclasses plus a `TASK_STATUSES` tuple
documenting the valid status values. The `Plan` class is the canonical
mutator for task status; the rest of the codebase reads `Plan` snapshots
but does not edit task fields directly.
"""

from dataclasses import dataclass, field
from typing import Optional


TASK_STATUSES = ("pending", "in_progress", "done", "failed", "skipped")


@dataclass
class Task:
    id: int                # 1-based, stable across plan lifetime
    description: str       # short user-facing line
    status: str = "pending"


@dataclass
class Plan:
    title: str
    tasks: list             # list[Task]
    current_task_id: Optional[int] = None

    def get_task(self, task_id: int) -> Optional[Task]:
        return next((t for t in self.tasks if t.id == task_id), None)

    def advance(self, task_id: int, status: str) -> None:
        """Transition a task to a new status. Invalid status or unknown id is a no-op."""
        if status not in TASK_STATUSES:
            return
        task = self.get_task(task_id)
        if task is None:
            return
        task.status = status
        if status == "in_progress":
            self.current_task_id = task_id
        elif status in ("done", "failed", "skipped") and self.current_task_id == task_id:
            self.current_task_id = None

    def insert(self, after_id: int, description: str) -> Task:
        """Insert a new task after the given task id. If after_id is unknown,
        appends to the end. Returns the new task. The new id is `max(existing) + 1`."""
        new_id = max((t.id for t in self.tasks), default=0) + 1
        new_task = Task(id=new_id, description=description)
        idx = next((i for i, t in enumerate(self.tasks) if t.id == after_id), len(self.tasks) - 1)
        self.tasks.insert(idx + 1, new_task)
        return new_task

    def is_complete(self) -> bool:
        return all(t.status in ("done", "skipped") for t in self.tasks)

    def progress(self):
        """Return (done_or_skipped_count, total_count)."""
        done = sum(1 for t in self.tasks if t.status in ("done", "skipped"))
        return done, len(self.tasks)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/lulu/Projects/ezclaw
pytest tests/test_plan.py -v
```

Expected: All 16 tests pass.

- [ ] **Step 5: Commit**

```bash
git add plan.py tests/test_plan.py
git commit -m "feat(plan): add Task and Plan data model with full unit test coverage"
```

---

## Task 2: Extend `theme.py` with task-state styling tables

**Files:**
- Modify: `theme.py` (append at the end, after the `THEME` singleton)

- [ ] **Step 1: Add task-state lookup tables to `theme.py`**

Open `/home/lulu/Projects/ezclaw/theme.py`. After the `THEME = Theme(...)` block at the bottom, append:

```python


# Task-state styling for the plan panel. The state key matches Plan task
# statuses (plan.TASK_STATUSES). Each entry is (icon_glyph, color_hex).
TASK_STATE_STYLE: dict = {
    "pending":     ("○", _PALETTE.dim),
    "in_progress": ("▸", "#5fafff"),     # executor blue
    "done":        ("●", "#5fd75f"),     # green
    "failed":      ("✗", _PALETTE.err),
    "skipped":     ("⊘", _PALETTE.dim),
}

# Brighter variant of each state's color, used for the one-tick flash
# when a task transitions between statuses.
TASK_STATE_FLASH: dict = {
    "pending":     _PALETTE.secondary,
    "in_progress": "#afd7ff",            # brighter executor blue
    "done":        "#afffaf",            # brighter green
    "failed":      "#ffafaf",            # brighter red
    "skipped":     _PALETTE.secondary,
}
```

- [ ] **Step 2: Verify the additions import cleanly**

Write to `/tmp/_verify_theme.py`:

```python
from theme import TASK_STATE_STYLE, TASK_STATE_FLASH
# Sanity checks
assert TASK_STATE_STYLE["pending"] == ("○", "#808080")
assert TASK_STATE_STYLE["in_progress"][1] == "#5fafff"
assert TASK_STATE_STYLE["done"][0] == "●"
assert TASK_STATE_FLASH["in_progress"] == "#afd7ff"
assert TASK_STATE_FLASH["done"] == "#afffaf"
# All Plan statuses must have entries in both maps
for status in ("pending", "in_progress", "done", "failed", "skipped"):
    assert status in TASK_STATE_STYLE
    assert status in TASK_STATE_FLASH
print("theme task-state tables OK")
```

Run:

```bash
cd /home/lulu/Projects/ezclaw
python /tmp/_verify_theme.py
```

Expected output: `theme task-state tables OK`

- [ ] **Step 3: Commit**

```bash
git add theme.py
git commit -m "feat(theme): add TASK_STATE_STYLE and TASK_STATE_FLASH tables"
```

---

## Task 3: Rewrite Architect system prompt + add `Architect.plan()` method

**Files:**
- Modify: `multi_agent.py` — replace the `Architect.__init__` system prompt; add new `plan()` method on `Architect`.
- Create: `tests/test_architect_plan.py` — smoke tests with stubbed Ollama client.

- [ ] **Step 1: Add the smoke test for `Architect.plan()`**

Create `/home/lulu/Projects/ezclaw/tests/test_architect_plan.py` with this exact content:

```python
import json
import pytest
from unittest.mock import MagicMock, patch

from plan import Plan, Task


@pytest.fixture
def fake_db():
    """Architect.__init__ takes a Database but only stores it; smoke tests
    don't exercise db code, so a MagicMock is fine."""
    return MagicMock()


def _make_architect_with_response(fake_db, response_text):
    """Construct an Architect with build_architect_client stubbed to return a
    mock client that yields `response_text` as the chat content."""
    from multi_agent import Architect

    with patch("multi_agent.build_architect_client") as build_client:
        mock_client = MagicMock()
        # Ollama-style response
        mock_client.chat.return_value = {"message": {"content": response_text}}
        build_client.return_value = (mock_client, "fake-model")
        arch = Architect(fake_db)
    return arch


def test_plan_returns_populated_plan_when_kind_is_plan(fake_db):
    response = json.dumps({
        "kind": "plan",
        "title": "fix the SSE memory leak",
        "tasks": [
            {"id": 1, "description": "Read sse_handler.py"},
            {"id": 2, "description": "Add cleanup in disconnect path"},
            {"id": 3, "description": "Add regression test"},
        ],
    })
    arch = _make_architect_with_response(fake_db, response)
    plan = arch.plan("fix the SSE memory leak")
    assert plan is not None
    assert plan.title == "fix the SSE memory leak"
    assert len(plan.tasks) == 3
    assert plan.tasks[0].description == "Read sse_handler.py"
    assert all(t.status == "pending" for t in plan.tasks)


def test_plan_returns_none_when_kind_is_single(fake_db):
    response = json.dumps({
        "kind": "single",
        "reason": "Conversational reply, no task list needed.",
    })
    arch = _make_architect_with_response(fake_db, response)
    plan = arch.plan("hello there")
    assert plan is None


def test_plan_returns_none_on_malformed_json_twice(fake_db):
    """Architect retries once; if both attempts fail to parse, return None."""
    from multi_agent import Architect

    with patch("multi_agent.build_architect_client") as build_client:
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "not json at all"}}
        build_client.return_value = (mock_client, "fake-model")
        arch = Architect(fake_db)

    plan = arch.plan("anything")
    assert plan is None
    # Confirm it retried at least once
    assert mock_client.chat.call_count >= 2
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/lulu/Projects/ezclaw
pytest tests/test_architect_plan.py -v
```

Expected: All 3 tests fail with `AttributeError: 'Architect' object has no attribute 'plan'` (or similar).

- [ ] **Step 3: Replace the Architect system prompt**

In `multi_agent.py`, find the `Architect.__init__` method (around line 380). The current system prompt content starts after `"content": """You are the **Architect** —` and ends with `"complete": false\n}"""`.

Replace the entire system prompt content (the `content` string in the system message) with:

```
You are the **Architect** — a senior systems designer orchestrating a multi-agent workflow.

You operate in TWO modes. Each user prompt will start with either `## PLANNING REQUEST` or `## EXECUTION REQUEST`. Read the header carefully and respond with the JSON shape required by that mode.

## Routing roles available
- **executor**: File edits, shell commands, code implementation, memory management, verification runs.
- **researcher**: Web searching, documentation gathering.
- **debugger**: Root-cause analysis. Only route here for UNEXPECTED errors.
- **general**: Conversational responses.

═══════════════════════════════════════════════════════════════
## Mode 1: PLANNING REQUEST
═══════════════════════════════════════════════════════════════

You receive the user's original request plus context. Classify the request and produce ONE of:

**Multi-step request** → return a structured plan:
```
{
  "kind": "plan",
  "title": "<one-line summary, ≤60 chars, no trailing period>",
  "tasks": [
    {"id": 1, "description": "<short user-facing line, ≤80 chars>"},
    {"id": 2, "description": "..."},
    ...
  ]
}
```

Rules for plans:
- Between 2 and 7 tasks. If you cannot decompose into ≥2 meaningful tasks, return `kind: single` instead.
- Tasks must be in execution order. No out-of-order dependencies.
- Descriptions are USER-FACING summaries, not implementation jargon. ("Add regression test", not "Write pytest test for handle_disconnect helper using fixtures.")
- IDs are 1-based, contiguous, ascending.

**Single-step / conversational request** → return:
```
{"kind": "single", "reason": "<one-line explanation>"}
```

Return `kind: single` when the request is:
- Conversational ("hi", "what does X do", "explain Y")
- A single file read / lookup
- Anything that genuinely doesn't decompose into 2+ meaningful steps

═══════════════════════════════════════════════════════════════
## Mode 2: EXECUTION REQUEST
═══════════════════════════════════════════════════════════════

You receive: the current plan (with task statuses), the task_context (recent step results), memory/skills/routing context. Decide the NEXT step.

Return:
```
{
  "kind": "execute",
  "current_task_id": <int>,
  "recommended_agent": "executor|general|researcher|debugger",
  "reasoning": "<one short line on the routing choice>",
  "task_updates": [{"id": <int>, "status": "done|failed|skipped"}],
  "new_tasks": [{"after_id": <int>, "description": "<short line>"}],
  "complete": false,
  "reflection": {
    "goal": "<noun phrase>",
    "observation": "<what happened in the last step>",
    "critical_thinking": "<one line on why this action>"
  }
}
```

Rules for execution:
- `current_task_id` must reference an existing task in the plan (not yet `done` / `failed` / `skipped`).
- `task_updates` is for tasks finishing in the current step. Only mark `done` after a successful verification (test passing, file confirmed, etc.). Mark `failed` only after retries are exhausted. Mark `skipped` only when the task is genuinely no longer needed (e.g., the user pivoted).
- `new_tasks` is for genuinely-new work discovered during execution (e.g., a sub-issue uncovered while debugging). Leave empty most of the time. Each entry's `after_id` must reference an existing task.
- Set `complete: true` ONLY when every task in the plan is `done` or `skipped` AND the user's full original intent is verifiably satisfied. Premature completion is forbidden — an `edit` is not a `fix`; an `install` is not a working `build`. Demand evidence (a passing test, a successful build, a confirming `cat`).
- `reflection.observation` is one short sentence describing what actually happened in the previous step. Skip if first step.

═══════════════════════════════════════════════════════════════
## When no plan is active (single-step path)
═══════════════════════════════════════════════════════════════

If the EXECUTION REQUEST says `Plan: (none — single-step request)`, treat each call as a one-shot routing decision. Set `current_task_id` to `0`, leave `task_updates` and `new_tasks` empty, and set `complete: true` as soon as the user's intent is satisfied.

═══════════════════════════════════════════════════════════════
## General style
═══════════════════════════════════════════════════════════════

- One short sentence per reflection field. Direct, no "Let me consider..." filler.
- Return ONLY the JSON object. No prose before or after. No markdown fences.
- Empty arrays and empty strings are fine where no new information applies.
```

In Python this looks like:

```python
"""You are the **Architect** — a senior systems designer orchestrating a multi-agent workflow.

You operate in TWO modes. Each user prompt will start with either `## PLANNING REQUEST` or `## EXECUTION REQUEST`. Read the header carefully and respond with the JSON shape required by that mode.

## Routing roles available
- **executor**: File edits, shell commands, code implementation, memory management, verification runs.
- **researcher**: Web searching, documentation gathering.
- **debugger**: Root-cause analysis. Only route here for UNEXPECTED errors.
- **general**: Conversational responses.

═══════════════════════════════════════════════════════════════
## Mode 1: PLANNING REQUEST
═══════════════════════════════════════════════════════════════

You receive the user's original request plus context. Classify the request and produce ONE of:

**Multi-step request** → return a structured plan:
{
  "kind": "plan",
  "title": "<one-line summary, ≤60 chars, no trailing period>",
  "tasks": [
    {"id": 1, "description": "<short user-facing line, ≤80 chars>"},
    {"id": 2, "description": "..."}
  ]
}

Rules for plans:
- Between 2 and 7 tasks. If you cannot decompose into ≥2 meaningful tasks, return `kind: single` instead.
- Tasks must be in execution order. No out-of-order dependencies.
- Descriptions are USER-FACING summaries, not implementation jargon.
- IDs are 1-based, contiguous, ascending.

**Single-step / conversational request** → return:
{"kind": "single", "reason": "<one-line explanation>"}

Return `kind: single` when the request is:
- Conversational ("hi", "what does X do", "explain Y")
- A single file read / lookup
- Anything that genuinely doesn't decompose into 2+ meaningful steps

═══════════════════════════════════════════════════════════════
## Mode 2: EXECUTION REQUEST
═══════════════════════════════════════════════════════════════

You receive: the current plan (with task statuses), the task_context (recent step results), memory/skills/routing context. Decide the NEXT step.

Return:
{
  "kind": "execute",
  "current_task_id": <int>,
  "recommended_agent": "executor|general|researcher|debugger",
  "reasoning": "<one short line on the routing choice>",
  "task_updates": [{"id": <int>, "status": "done|failed|skipped"}],
  "new_tasks": [{"after_id": <int>, "description": "<short line>"}],
  "complete": false,
  "reflection": {
    "goal": "<noun phrase>",
    "observation": "<what happened in the last step>",
    "critical_thinking": "<one line on why this action>"
  }
}

Rules for execution:
- `current_task_id` must reference an existing task in the plan (not yet `done`/`failed`/`skipped`).
- `task_updates` is for tasks finishing in the current step. Only mark `done` after a successful verification. Mark `failed` only after retries are exhausted. Mark `skipped` only when the task is genuinely no longer needed.
- `new_tasks` is for genuinely-new work discovered during execution. Leave empty most of the time. Each entry's `after_id` must reference an existing task.
- Set `complete: true` ONLY when every task in the plan is `done` or `skipped` AND the user's full original intent is verifiably satisfied. Premature completion is forbidden.
- `reflection.observation` is one short sentence describing what actually happened in the previous step. Skip if first step.

═══════════════════════════════════════════════════════════════
## When no plan is active (single-step path)
═══════════════════════════════════════════════════════════════

If the EXECUTION REQUEST says `Plan: (none — single-step request)`, treat each call as a one-shot routing decision. Set `current_task_id` to `0`, leave `task_updates` and `new_tasks` empty, and set `complete: true` as soon as the user's intent is satisfied.

═══════════════════════════════════════════════════════════════
## General style
═══════════════════════════════════════════════════════════════

- One short sentence per reflection field. Direct, no filler.
- Return ONLY the JSON object. No prose before or after. No markdown fences.
- Empty arrays and empty strings are fine where no new information applies."""
```

- [ ] **Step 4: Add the `Architect.plan()` method**

In `multi_agent.py`, find the `Architect.analyze` method. Insert the new `plan()` method BEFORE `analyze()`. The new method:

```python
    def plan(
        self,
        user_input: str,
        memory_block: str = "",
        skills_block: str = "",
        history_block: str = "",
        map_block: str = "",
    ):
        """Planning pass. Returns a Plan object if the request is multi-step,
        or None if it's single-step / conversational.

        Retries once on malformed JSON, then returns None.
        """
        from plan import Plan, Task

        prompt = f"""## PLANNING REQUEST

## User Request
{user_input}

## Conversation History
{history_block}

## Codebase Map
{map_block}

## Memory & Skills
{memory_block}{skills_block}

## Decision Required
Classify this request. If multi-step, return a `kind: plan` JSON with 2-7 tasks. If single-step or conversational, return `kind: single`.

Return ONLY the JSON object."""

        for attempt in range(2):
            try:
                content = self._chat(
                    prompt if attempt == 0 else prompt + "\n\nCRITICAL: Return ONLY valid JSON.",
                    temperature=0.0,
                )
                data = extract_json(content)
                kind = data.get("kind")
                if kind == "single":
                    return None
                if kind == "plan":
                    title = str(data.get("title", "")).strip()
                    raw_tasks = data.get("tasks", [])
                    if not isinstance(raw_tasks, list) or not raw_tasks:
                        continue
                    tasks = []
                    for i, t in enumerate(raw_tasks, start=1):
                        if not isinstance(t, dict):
                            continue
                        desc = str(t.get("description", "")).strip()
                        if not desc:
                            continue
                        # Force contiguous 1-based ids regardless of what the
                        # model returned; this is the canonical id space.
                        tasks.append(Task(id=i, description=desc[:120]))
                    if len(tasks) < 2:
                        # Spec says "at least 2 tasks" — fewer means single-step.
                        return None
                    return Plan(title=title or "Plan", tasks=tasks)
                # Unknown kind → treat as single-step
                return None
            except Exception:
                continue
        return None
```

- [ ] **Step 5: Run the smoke tests to verify they pass**

```bash
cd /home/lulu/Projects/ezclaw
pytest tests/test_architect_plan.py -v
```

Expected: All 3 tests pass.

- [ ] **Step 6: Confirm the existing test suite still passes**

```bash
cd /home/lulu/Projects/ezclaw
pytest tests/ -v
```

Expected: All previously-passing tests still pass; the only new tests are in `test_plan.py` and `test_architect_plan.py`.

- [ ] **Step 7: Commit**

```bash
git add multi_agent.py tests/test_architect_plan.py
git commit -m "feat(architect): rewrite system prompt for two-mode operation; add plan() method"
```

---

## Task 4: Add `Architect.execute()` method

**Files:**
- Modify: `multi_agent.py` — add `execute()` method after `plan()`.
- Modify: `tests/test_architect_plan.py` — add smoke tests for `execute()`.

- [ ] **Step 1: Add tests for `Architect.execute()`**

Append to `/home/lulu/Projects/ezclaw/tests/test_architect_plan.py`:

```python


def test_execute_returns_intent_with_plan(fake_db):
    response = json.dumps({
        "kind": "execute",
        "current_task_id": 2,
        "recommended_agent": "executor",
        "reasoning": "Apply the cleanup hook found in task 1.",
        "task_updates": [{"id": 1, "status": "done"}],
        "new_tasks": [],
        "complete": False,
        "reflection": {
            "goal": "Fix SSE leak",
            "observation": "Read confirmed the leak location.",
            "critical_thinking": "Move to task 2.",
        },
    })
    arch = _make_architect_with_response(fake_db, response)
    plan = Plan(
        title="fix leak",
        tasks=[Task(id=1, description="read"), Task(id=2, description="fix"), Task(id=3, description="test")],
    )
    intent = arch.execute(plan, task_context="Step 1 read complete.")
    assert intent["kind"] == "execute"
    assert intent["current_task_id"] == 2
    assert intent["recommended_agent"] == "executor"
    assert intent["task_updates"] == [{"id": 1, "status": "done"}]
    assert intent["complete"] is False


def test_execute_returns_intent_without_plan(fake_db):
    """When plan is None, execute() still works — single-step path."""
    response = json.dumps({
        "kind": "execute",
        "current_task_id": 0,
        "recommended_agent": "general",
        "reasoning": "Conversational reply.",
        "task_updates": [],
        "new_tasks": [],
        "complete": True,
        "reflection": {"goal": "Answer", "observation": "", "critical_thinking": "Done."},
    })
    arch = _make_architect_with_response(fake_db, response)
    intent = arch.execute(None, task_context="User asked: hi")
    assert intent["kind"] == "execute"
    assert intent["complete"] is True


def test_execute_fallback_on_malformed_json(fake_db):
    """Two failed parses → return a safe fallback intent (no plan changes)."""
    from multi_agent import Architect

    with patch("multi_agent.build_architect_client") as build_client:
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "not json"}}
        build_client.return_value = (mock_client, "fake-model")
        arch = Architect(fake_db)

    intent = arch.execute(None, task_context="anything")
    # Fallback intent should be safe defaults, not raise
    assert isinstance(intent, dict)
    assert intent.get("recommended_agent") in ("executor", "general", "researcher", "debugger")
    assert intent.get("task_updates", []) == []
    assert intent.get("new_tasks", []) == []
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
cd /home/lulu/Projects/ezclaw
pytest tests/test_architect_plan.py::test_execute_returns_intent_with_plan -v
```

Expected: Fails with `AttributeError: 'Architect' object has no attribute 'execute'`.

- [ ] **Step 3: Add `Architect.execute()` method**

In `multi_agent.py`, immediately after the `plan()` method added in Task 3, insert:

```python
    def execute(
        self,
        plan,
        task_context: str,
        memory_block: str = "",
        skills_block: str = "",
        experiences_block: str = "",
        routing_block: str = "",
        history_block: str = "",
        map_block: str = "",
        temperature: float = 0.0,
        pivot_hint: str = "",
    ) -> "Dict[str, Any]":
        """Execution-mode call. Returns the structured intent dict. When
        `plan` is None, the architect operates in single-step mode (no plan
        active); when present, the plan is rendered into the prompt so the
        architect knows which task to advance.

        On parse failure, returns a safe fallback intent that does not
        mutate the plan.
        """
        if plan is not None:
            plan_render_lines = [f"Plan: {plan.title}"]
            for t in plan.tasks:
                marker = {
                    "pending": " ", "in_progress": "▸", "done": "✓",
                    "failed": "✗", "skipped": "⊘",
                }.get(t.status, " ")
                plan_render_lines.append(f"  [{marker}] {t.id}. {t.description}  ({t.status})")
            plan_block = "\n".join(plan_render_lines)
        else:
            plan_block = "Plan: (none — single-step request)"

        pivot_block = ""
        if pivot_hint:
            pivot_block = (
                "## CRITICAL PIVOT REQUIRED\n"
                f"{pivot_hint}\n"
                "Do NOT re-issue the previous plan. Choose a fundamentally different "
                "approach: switch the recommended_agent, decompose differently, "
                "or use a different tool.\n\n"
            )

        prompt = f"""## EXECUTION REQUEST

{pivot_block}{plan_block}

## Task Context
{task_context}

## Conversation History
{history_block}

## Auxiliary Context
### Codebase Map
{map_block}
### Past Experiences
{experiences_block}
{memory_block}{skills_block}{routing_block}

## Decision Required
Decide the next routing step. Return the EXECUTION JSON object."""

        for attempt in range(2):
            try:
                content = self._chat(
                    prompt if attempt == 0 else prompt + "\n\nCRITICAL: Return ONLY valid JSON.",
                    temperature=temperature,
                )
                intent = extract_json(content)
                intent.setdefault("kind", "execute")
                intent.setdefault("current_task_id", 0)
                intent.setdefault("recommended_agent", "executor")
                intent.setdefault("reasoning", "")
                intent.setdefault("task_updates", [])
                intent.setdefault("new_tasks", [])
                intent.setdefault("complete", False)
                intent.setdefault("reflection", {})

                valid_agents = {"executor", "general", "researcher", "debugger"}
                if intent.get("recommended_agent") not in valid_agents:
                    intent["recommended_agent"] = "executor"
                if not isinstance(intent.get("task_updates"), list):
                    intent["task_updates"] = []
                if not isinstance(intent.get("new_tasks"), list):
                    intent["new_tasks"] = []
                return intent
            except Exception:
                continue

        # Safe fallback — keep the loop alive without mutating the plan.
        return {
            "kind": "execute",
            "current_task_id": 0,
            "recommended_agent": "executor",
            "reasoning": "Parse fallback",
            "task_updates": [],
            "new_tasks": [],
            "complete": False,
            "reflection": {"critical_thinking": "Parsing failed; routing to executor as a default."},
        }
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
cd /home/lulu/Projects/ezclaw
pytest tests/ -v
```

Expected: All tests pass, including the 3 new `execute` tests.

- [ ] **Step 5: Commit**

```bash
git add multi_agent.py tests/test_architect_plan.py
git commit -m "feat(architect): add execute() method for plan-aware execution intents"
```

---

## Task 5: Restructure `MultiAgentSystem.run` to use plan/execute and emit plan chunks

**Files:**
- Modify: `multi_agent.py` — replace `analyze` call site in `MultiAgentSystem.run`; emit `plan_created` and `plan_update` chunks; delete the now-unused `analyze` method.

- [ ] **Step 1: Identify the call site**

In `multi_agent.py`, find the `MultiAgentSystem.run` method. Locate the existing line that calls the architect (a line containing `self.architect.analyze(`). Note its surrounding context — particularly the stuck-loop / pivot block that calls `self.architect.analyze(...)` a second time at higher temperature.

There are TWO call sites of `self.architect.analyze`:
1. The main loop body (the per-step architect call).
2. The pivot-recovery branch (the higher-temperature retry when the stuck-counter trips).

Both need to be replaced with `self.architect.execute(plan, task_context, ...)`.

- [ ] **Step 2: Replace the main-loop architect call**

In `MultiAgentSystem.run`, find the line:

```python
            intent = self.architect.analyze(task_context, memory_block, skills_block, routing_block=routing_block, history_block=history_block, map_block=map_block, experiences_block=exp_block)
```

Replace with:

```python
            intent = self.architect.execute(
                self.current_plan, task_context, memory_block, skills_block,
                routing_block=routing_block, history_block=history_block,
                map_block=map_block, experiences_block=exp_block,
            )
```

- [ ] **Step 3: Replace the pivot-recovery architect call**

In the same method, find the line (inside the stuck-counter branch):

```python
                    intent = self.architect.analyze(
                        task_context, memory_block, skills_block,
                        routing_block=routing_block, history_block=history_block,
                        map_block=map_block, experiences_block=exp_block,
                        temperature=0.7, pivot_hint=hint,
                    )
```

Replace with:

```python
                    intent = self.architect.execute(
                        self.current_plan, task_context, memory_block, skills_block,
                        routing_block=routing_block, history_block=history_block,
                        map_block=map_block, experiences_block=exp_block,
                        temperature=0.7, pivot_hint=hint,
                    )
```

- [ ] **Step 4: Add planning pass + plan-state management in `MultiAgentSystem.__init__` and `run`**

In `MultiAgentSystem.__init__`, find any field initialization (after `self.architect = Architect(...)` and the agents dict). Add:

```python
        # Current plan for the in-flight user request. Reset on each run.
        self.current_plan = None
```

In `MultiAgentSystem.run`, find the place AFTER the short-circuit fast-route check and BEFORE the `for step in range(...)` loop. (You'll see lines that set up `memory_facts`, `routing_priors`, `experiences`, etc.) After those setups and immediately before the loop, insert:

```python
        # Planning pass: produce a structured task list, or None for single-step.
        self.current_plan = self.architect.plan(
            user_input,
            memory_block=memory_block,
            skills_block=skills_block,
            history_block=history_block,
            map_block=map_block,
        )
        if self.current_plan is not None:
            yield {"type": "plan_created", "plan": self.current_plan}
```

Also, BEFORE the planning pass, reset the field per-run:

Find the early part of `run` where `task_context` is initialized (`task_context = f"User Request: {user_input}"`). Right after that line, add:

```python
        self.current_plan = None
```

- [ ] **Step 5: Apply task_updates and new_tasks after each execute() call, emit plan_update**

In `MultiAgentSystem.run`, inside the `for step in range(1, max_steps + 1):` loop, find the line that yields the existing `intent` chunk:

```python
            yield {
                "type": "intent",
                "reflection": intent.get("reflection"),
                "reasoning": intent.get("reasoning"),
                "plan": intent.get("plan"),
                "agent": intent.get("recommended_agent"),
                "complete": intent.get("complete")
            }
```

IMMEDIATELY BEFORE that yield, add:

```python
            # Apply plan mutations from the execute intent
            if self.current_plan is not None:
                for upd in intent.get("task_updates", []) or []:
                    if isinstance(upd, dict) and "id" in upd and "status" in upd:
                        self.current_plan.advance(upd["id"], upd["status"])
                for new in intent.get("new_tasks", []) or []:
                    if isinstance(new, dict) and "after_id" in new and "description" in new:
                        self.current_plan.insert(new["after_id"], new["description"])
                current_id = intent.get("current_task_id")
                if current_id and self.current_plan.get_task(current_id):
                    self.current_plan.advance(current_id, "in_progress")
                yield {"type": "plan_update", "plan": self.current_plan}
```

- [ ] **Step 6: Finalize the plan on loop exit**

Find the line at the end of `run` (after the `for step in range(...)` loop, where the existing code yields a "Step limit reached" status). Wrap that area to also finalize the plan: any tasks still `in_progress` are marked `skipped`, and one final `plan_update` chunk is emitted.

Just before any final `break` or after-loop housekeeping (search for `if step >= max_steps:` near the end), add a helper section that runs whenever the loop exits (whether by `break` or completion):

Actually the cleanest place is RIGHT BEFORE every `break` statement and after the loop completes. To keep the implementation simple, refactor with a try/finally pattern:

Find the `for step in range(1, max_steps + 1):` loop. Wrap the loop body in nothing new (no try/finally needed). Instead, AFTER the loop (after the existing `if step >= max_steps:` block), add:

```python
        # Finalize plan: any lingering in_progress task → skipped, emit final update
        if self.current_plan is not None:
            for t in self.current_plan.tasks:
                if t.status == "in_progress":
                    self.current_plan.advance(t.id, "skipped")
            yield {"type": "plan_update", "plan": self.current_plan}
```

(Note: this finalization will also fire after the `break` inside the loop, because all the `break`s exit the loop and then control falls through to the finalize block.)

- [ ] **Step 7: Delete the now-unused `Architect.analyze` method**

In `multi_agent.py`, find the `Architect.analyze` method. Delete it entirely. Its responsibility is now split between `plan()` (Task 3) and `execute()` (Task 4).

- [ ] **Step 8: Run all tests to verify they pass**

```bash
cd /home/lulu/Projects/ezclaw
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 9: Syntax check and import check**

```bash
cd /home/lulu/Projects/ezclaw
python -c "import ast; ast.parse(open('multi_agent.py').read()); print('OK')"
python -c "from multi_agent import MultiAgentSystem; print('import OK')"
```

Expected: Both print `OK` / `import OK` respectively.

- [ ] **Step 10: Commit**

```bash
git add multi_agent.py
git commit -m "feat(orchestration): MultiAgentSystem.run uses plan()+execute(), emits plan chunks"
```

---

## Task 6: ChatUI state + chunk handlers for plan_created/plan_update

**Files:**
- Modify: `cli.py` — `__init__` adds plan-related state; `handle_input` resets it; `_agent_worker` handles the two new chunk types.

- [ ] **Step 1: Add plan-related state to `__init__`**

In `cli.py`, find the `ChatUI.__init__` method. Locate the existing `self._chip_flash_until: float = 0.0` line (added in the theme work). Immediately after it, add:

```python
        # Active plan from the most recent run. None means "no plan, hide panel".
        self.current_plan = None
        # Track previous statuses so we can detect transitions and flash the row.
        self._last_task_states: dict = {}
        # task_id → monotonic timestamp at which the flash window expires.
        self._task_flash_until: dict = {}
```

- [ ] **Step 2: Reset plan state on each new user turn in `handle_input`**

In `handle_input`, find the line `self._chip_flash_until = 0.0` (added in the theme work). Immediately after it, add:

```python
        self.current_plan = None
        self._last_task_states = {}
        self._task_flash_until = {}
```

- [ ] **Step 3: Handle the two new chunk types in `_agent_worker`**

In `_agent_worker`, find the chunk-dispatch loop. The structure is an `if/elif` cascade where each branch handles a chunk type, then the loop bottom does `chunk = next(gen)` to fetch the next one (with `continue` only used by branches that re-issue `gen.send(...)`, like the `auth_required` branch).

Locate the existing `if chunk["type"] == "intent":` branch. Immediately BEFORE that branch, insert these two new branches following the same `if`/`elif` pattern as the existing code (the first new branch becomes the new top-level `if`; the existing `if chunk["type"] == "intent":` becomes an `elif`):

```python
                if chunk["type"] == "plan_created":
                    self.current_plan = chunk["plan"]
                    self._last_task_states = {t.id: t.status for t in self.current_plan.tasks}
                elif chunk["type"] == "plan_update":
                    new_plan = chunk["plan"]
                    now = time.time()
                    for task in new_plan.tasks:
                        prev = self._last_task_states.get(task.id)
                        if prev is not None and prev != task.status:
                            self._task_flash_until[task.id] = now + 0.15
                        self._last_task_states[task.id] = task.status
                    self.current_plan = new_plan
                elif chunk["type"] == "intent":
                    # ← existing line; convert from `if` to `elif`
```

So the resulting structure is:

```python
                if chunk["type"] == "plan_created":
                    ...
                elif chunk["type"] == "plan_update":
                    ...
                elif chunk["type"] == "intent":
                    ...  # existing body, unchanged
                elif chunk["type"] == "reasoning":
                    ...  # existing
                # ... rest of existing elif chain unchanged
```

No `continue` needed — both new branches fall through to the bottom of the loop where the existing `chunk = next(gen)` advances the generator.

- [ ] **Step 4: Syntax check**

```bash
cd /home/lulu/Projects/ezclaw
python -c "import ast; ast.parse(open('cli.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 5: Smoke-test the state initialization**

Write to `/tmp/_verify_state.py`:

```python
import cli
ui = cli.ChatUI()
assert ui.current_plan is None
assert ui._last_task_states == {}
assert ui._task_flash_until == {}
print("plan state init OK")
```

Run:

```bash
cd /home/lulu/Projects/ezclaw
python /tmp/_verify_state.py
```

Expected: `plan state init OK`.

- [ ] **Step 6: Commit**

```bash
git add cli.py
git commit -m "feat(ui): ChatUI tracks current_plan and task-state transitions"
```

---

## Task 7: `_render_plan_panel` method + integration into render path

**Files:**
- Modify: `cli.py` — add `_render_plan_panel` method; insert the panel above the architect chip in `_get_current_renderable_ansi`.

- [ ] **Step 1: Add the `_render_plan_panel` method**

In `cli.py`, find the existing `_render_architect_intent` method on `ChatUI`. Immediately BEFORE it, add:

```python
    def _render_plan_panel(self):
        """Render the active plan as a sticky panel. Returns None when no plan."""
        if self.current_plan is None or not self.current_plan.tasks:
            return None
        from theme import TASK_STATE_STYLE, TASK_STATE_FLASH

        plan = self.current_plan
        done, total = plan.progress()
        now = time.time()

        body_lines = []
        for task in plan.tasks:
            icon, color = TASK_STATE_STYLE.get(task.status, ("•", DIM))
            flashing = now < self._task_flash_until.get(task.id, 0.0)
            line_color = TASK_STATE_FLASH.get(task.status, color) if flashing else color

            # Bold for in_progress so the eye finds it immediately.
            weight = "bold " if task.status == "in_progress" else ""
            line_text = Text()
            line_text.append(f"  {icon} ", style=f"{weight}{line_color}")
            line_text.append(f"{task.id}. ", style=f"dim {DIM}")
            line_text.append(task.description, style=f"{weight}{line_color}")
            body_lines.append(line_text)

        title = f"Plan: {plan.title}  ·  {done}/{total}"
        return Panel(
            Group(*body_lines),
            title=f"[bold {PRIMARY}]{title}[/bold {PRIMARY}]",
            border_style=f"dim {PRIMARY}",
            box=ROUNDED,
            padding=(0, 1),
        )
```

- [ ] **Step 2: Insert the plan panel into the render path**

In `cli.py`, find `_get_current_renderable_ansi`. Locate the block that handles `self.architect_intent` (it calls `self._render_architect_intent(...)`). Immediately BEFORE that block, add:

```python
        plan_panel = self._render_plan_panel()
        if plan_panel is not None:
            parts.append(plan_panel)
```

- [ ] **Step 3: Syntax check**

```bash
cd /home/lulu/Projects/ezclaw
python -c "import ast; ast.parse(open('cli.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Smoke-test the panel rendering**

Write to `/tmp/_verify_plan_render.py`:

```python
import cli
from plan import Plan, Task

ui = cli.ChatUI()
# No plan → no panel
assert ui._render_plan_panel() is None

# Build a plan and render
plan = Plan(
    title="fix the SSE memory leak",
    tasks=[
        Task(id=1, description="Read sse_handler.py", status="done"),
        Task(id=2, description="Add cleanup in disconnect path", status="in_progress"),
        Task(id=3, description="Add regression test"),
        Task(id=4, description="Run test suite and verify"),
    ],
    current_task_id=2,
)
ui.current_plan = plan
panel = ui._render_plan_panel()
assert panel is not None

from rich.console import Console
Console().print(panel)
print("plan render OK")
```

Run:

```bash
cd /home/lulu/Projects/ezclaw
python /tmp/_verify_plan_render.py
```

Expected: A bordered panel renders with the title `Plan: fix the SSE memory leak  ·  1/4`, four task lines with icons `● ▸ ○ ○`, task 2 in bold blue, then `plan render OK`. No tracebacks.

- [ ] **Step 5: Commit**

```bash
git add cli.py
git commit -m "feat(ui): render plan panel above architect chip with task-state icons"
```

---

## Task 8: Manual TUI smoke test + glyph fallback

**Files:**
- Possibly modify: `theme.py` — swap any glyphs that render as missing boxes (□) in the user's terminal.

- [ ] **Step 1: Launch the TUI in a separate terminal**

```bash
cd /home/lulu/Projects/ezclaw
python cli.py
```

- [ ] **Step 2: Send a multi-step request and verify the plan panel**

In the TUI, send a multi-step prompt such as:

```
Add a Python script that reads tools.py, lists every function name with its docstring summary, and writes the result to functions.md.
```

Expected:
- A plan panel appears above the architect chip with 3-5 tasks
- Title shows `Plan: <one-line summary>  ·  0/N` initially
- The first task transitions to `▸` (in_progress, bold blue) — briefly flashing brighter on transition
- As steps complete, tasks transition to `●` (done, green) with a flash
- Progress counter increments: `0/N → 1/N → 2/N → ...`
- When the plan is fully complete, all tasks show `●`, the counter shows `N/N`, and the architect chip below shows "task completed" or similar

- [ ] **Step 3: Send a conversational request and verify NO plan panel**

In the TUI, send a chat-style prompt:

```
hi, what's your name?
```

Expected:
- No plan panel appears
- The architect chip shows the routing decision as before
- Behavior is identical to pre-feature (the architect classified this as `kind: single`)

- [ ] **Step 4: Interrupt mid-execution and verify cleanup**

Send a multi-step prompt. While it's executing (plan panel visible with at least one task in_progress), send a NEW prompt:

```
nevermind, what's 2+2?
```

Expected:
- The old plan panel disappears immediately when the new prompt is submitted
- The new conversational request gets no plan panel
- Behavior matches "fresh start" — no stuck state from the previous in-flight plan

- [ ] **Step 5: If any glyph renders as a missing box (□)**

Edit `theme.py` and swap the offending glyph in `TASK_STATE_STYLE`. ASCII fallbacks:

```python
TASK_STATE_STYLE: dict = {
    "pending":     ("-", _PALETTE.dim),
    "in_progress": (">", "#5fafff"),
    "done":        ("+", "#5fd75f"),
    "failed":      ("x", _PALETTE.err),
    "skipped":     ("~", _PALETTE.dim),
}
```

Re-run the TUI smoke test. Repeat as needed.

- [ ] **Step 6: If no tweaks needed, no commit. If tweaks made, commit**

```bash
git add theme.py
git commit -m "fix(theme): swap task-state glyphs for terminal compatibility"
```

- [ ] **Step 7: Final verification — full diff**

```bash
git log --oneline edb1884..HEAD
```

Expected: 7 or 8 commits (depending on whether Task 8 needed a glyph tweak):

```
<hash> fix(theme): swap task-state glyphs for terminal compatibility   (only if needed)
<hash> feat(ui): render plan panel above architect chip with task-state icons
<hash> feat(ui): ChatUI tracks current_plan and task-state transitions
<hash> feat(orchestration): MultiAgentSystem.run uses plan()+execute(), emits plan chunks
<hash> feat(architect): add execute() method for plan-aware execution intents
<hash> feat(architect): rewrite system prompt for two-mode operation; add plan() method
<hash> feat(theme): add TASK_STATE_STYLE and TASK_STATE_FLASH tables
<hash> feat(plan): add Task and Plan data model with full unit test coverage
```

---

## Notes for the implementer

- **TDD is real for Tasks 1, 3, 4.** Write the test, watch it fail with the right error, then implement, then watch it pass. Don't skip the "watch it fail" step — that's how you confirm your test actually exercises the code path you think it does.
- **No TDD for Tasks 2, 5, 6, 7, 8.** These are visual / orchestration changes that don't have a meaningful headless assertion target. Each task uses a smoke import + a small Python verification script to confirm the code does what it claims.
- **Commit between every task.** If a later task introduces a regression, you can `git revert` the offending commit without losing the others.
- **Architect system prompt is large.** Task 3 Step 3 includes the full new prompt. Use the Python triple-quoted string form (Task 3 Step 3 shows it). Do not paraphrase — the JSON shapes are part of the contract with the model.
- **Glyph fallbacks are real.** Unicode glyphs in `theme.py` (especially `○ ▸ ● ✗ ⊘`) may render differently or be missing in some terminals. Task 8 has the swap procedure.
- **No new dependencies.** All required libs (`rich`, `ollama`, `pytest`, `prompt_toolkit`) are already in `requirements.txt`.
- **Don't fold `plan.py` into `multi_agent.py`.** The separation matters: `plan.py` is pure data with unit tests, `multi_agent.py` does I/O + LLM calls. Keep them separate.
