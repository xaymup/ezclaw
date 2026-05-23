# Agent Task Queue & Plan Execution

**Status:** Draft
**Date:** 2026-05-23
**Owner:** Mohamed Kotp

## Problem

The architect already produces a `plan` string with numbered steps every iteration of `MultiAgentSystem.run`, but:

1. The plan is **re-generated every step**, so its text drifts mid-flight even when no real strategy change happened. This makes the plan unstable as a user-facing artifact.
2. The plan is **prose, not structured**, so the UI can't show task-level state (which task is in progress, which are done, which are pending).
3. The UI surfaces only a one-line architect chip showing the currently routed agent. There is no high-level view of "what is this multi-step request actually doing."

Users tackling complex issues (bug fixes, multi-file refactors) can't see the shape of the work, can't track progress, and can't tell when the agent is on step 3 of 5 vs. step 5 of 5.

## Goal

Add a stable, structured **task queue** that:

1. Is generated once per multi-step user request (planning pass) and remains stable through execution.
2. Tracks task-level state — `pending`, `in_progress`, `done`, `failed`, `skipped`.
3. Renders as a TUI panel above the existing architect chip, with role-aware icons and frame-stepped animation on state transitions.
4. Does not appear for trivially short or conversational requests (one-shot reads, chitchat) — only when the architect classifies the request as multi-step.

## Non-goals

- Cross-turn plan persistence. Each new user message starts a fresh plan. In-flight plans are finalized (in-progress task marked `skipped`) when a new message arrives.
- User editing of tasks. The queue is read-only in v1 (the user explicitly chose this scope).
- Parallel execution of independent tasks. v1 executes tasks sequentially.
- Resume after Ctrl-C or interruption. v1 discards in-flight plans on interruption.
- Visual plan history. Completed plans do not accumulate in the scroll buffer — they roll up into the assistant's final response as today.

## Architecture

Three pieces:

1. **`plan.py` (new)** — pure data: `Task` and `Plan` dataclasses with helpers (`advance`, `insert`, `is_complete`, `get_task`). No I/O. Same pattern as `theme.py`.
2. **`multi_agent.py` (edit)** — `Architect` class gains explicit `plan(user_input)` and `execute(plan, task_context)` methods, replacing the single `analyze` call site. `MultiAgentSystem.run` is restructured to call planning once, then loop through execution intents that reference the plan.
3. **`cli.py` (edit)** — new `_render_plan_panel`, new `current_plan` instance state, new chunk-type handlers in `_agent_worker`. The panel renders above the chip when a plan is active.

The UI gets two new chunk types from `MultiAgentSystem.run`:
- `{"type": "plan_created", "plan": Plan}` — emitted after the planning pass succeeds.
- `{"type": "plan_update", "plan": Plan}` — emitted after each execution intent is applied to the plan.

Both carry the full current `Plan` (a small dataclass snapshot, cheap to copy). The UI maintains `self.current_plan` and re-renders.

## Components

### 1. `plan.py`

```python
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
    tasks: list[Task]
    current_task_id: Optional[int] = None

    def get_task(self, task_id: int) -> Optional[Task]:
        return next((t for t in self.tasks if t.id == task_id), None)

    def advance(self, task_id: int, status: str) -> None:
        task = self.get_task(task_id)
        if task is None or status not in TASK_STATUSES:
            return
        task.status = status
        if status == "in_progress":
            self.current_task_id = task_id
        elif status in ("done", "failed", "skipped") and self.current_task_id == task_id:
            self.current_task_id = None

    def insert(self, after_id: int, description: str) -> Task:
        """Insert a new task after the given task id. Returns the new task."""
        new_id = max((t.id for t in self.tasks), default=0) + 1
        new_task = Task(id=new_id, description=description)
        idx = next((i for i, t in enumerate(self.tasks) if t.id == after_id), len(self.tasks) - 1)
        self.tasks.insert(idx + 1, new_task)
        return new_task

    def is_complete(self) -> bool:
        return all(t.status in ("done", "skipped") for t in self.tasks)

    def progress(self) -> tuple[int, int]:
        """Returns (done_or_skipped_count, total)."""
        done = sum(1 for t in self.tasks if t.status in ("done", "skipped"))
        return done, len(self.tasks)
```

Mutable `Task.status` is intentional — `Plan.advance` is the canonical mutator.

### 2. Architect changes (`multi_agent.py`)

The `Architect` class gains two new methods replacing the bulk of `analyze`:

**`plan(user_input, memory_block, skills_block, ...) -> Optional[Plan]`**

Runs a planning prompt that returns this JSON shape:

```json
{
  "kind": "plan",
  "title": "fix the SSE memory leak",
  "tasks": [
    {"id": 1, "description": "Read sse_handler.py to find the leak"},
    {"id": 2, "description": "Add explicit cleanup in disconnect path"},
    {"id": 3, "description": "Add regression test"},
    {"id": 4, "description": "Run test suite and verify"}
  ]
}
```

If the architect decides the request is *not* multi-step (chat, single fact lookup), it returns:

```json
{"kind": "single", "reason": "Conversational reply, no task list needed."}
```

The Python side returns `None` in that case. The plan UI then never appears.

Constraints baked into the prompt:
- 2 ≤ `len(tasks)` ≤ 7. If the architect can't decompose into at least 2 meaningful tasks, return `kind: single`.
- Each `description` is one line, ≤ 80 chars, user-readable (not implementation jargon).
- Tasks must be in execution order (no out-of-order dependencies).

**`execute(plan, task_context, ...) -> ExecutionIntent`**

Replaces the existing `analyze` for steps 2+. The prompt includes a rendered text version of the current plan (each task's id, status, and description) so the architect can reason about which task to advance and what new tasks to insert. Returns:

```json
{
  "kind": "execute",
  "current_task_id": 2,
  "recommended_agent": "executor",
  "reasoning": "Apply the cleanup hook identified in task 1's read.",
  "task_updates": [{"id": 1, "status": "done"}],
  "new_tasks": [
    {"after_id": 2, "description": "Investigate connection-count regression seen in task 1"}
  ],
  "complete": false,
  "reflection": {"goal": "...", "observation": "...", "critical_thinking": "..."}
}
```

`task_updates` is applied before `new_tasks` to the plan. `complete: true` only when `plan.is_complete()` would return True after applying the updates — the architect's prompt explicitly says so.

The existing `reflection` field stays as-is to preserve the chip-expansion view. The free-form `plan` string is removed (the structured `tasks` array replaces it).

### 3. `MultiAgentSystem.run` restructuring

Sketch (omitting unchanged housekeeping):

```python
def run(self, user_input):
    # ... existing setup: memory_block, routing_block, history_block, codebase_map ...
    
    # Short-circuit unchanged — fast-route still kicks in for trivial requests.
    sc_agent = self._short_circuit_classify(user_input)
    if sc_agent and sc_agent != "debugger":
        # ... existing fast-route logic, no plan involved ...
        return
    
    # Planning pass.
    plan = self.architect.plan(user_input, memory_block, ...)
    if plan is not None:
        self.current_plan = plan
        yield {"type": "plan_created", "plan": plan}
    
    # Execution loop.
    for step in range(1, max_steps + 1):
        intent = self.architect.execute(plan, task_context, ...)
        if plan is not None:
            for upd in intent.get("task_updates", []):
                plan.advance(upd["id"], upd["status"])
            for new in intent.get("new_tasks", []):
                plan.insert(new["after_id"], new["description"])
            current_id = intent.get("current_task_id")
            if current_id is not None:
                plan.advance(current_id, "in_progress")
            yield {"type": "plan_update", "plan": plan}
        
        yield {"type": "intent", ...}  # existing chip-style chunk
        
        # ... existing dispatch + loop detection + recovery ...
        
        if intent.get("complete") and (plan is None or plan.is_complete()):
            break
    
    # Finalize: any tasks still in_progress get marked skipped.
    if plan is not None:
        for t in plan.tasks:
            if t.status == "in_progress":
                plan.advance(t.id, "skipped")
        yield {"type": "plan_update", "plan": plan}
```

The existing per-step `analyze` call is gone. The existing loop-detector and pivot-recovery (from earlier this session) keep working — they operate on `(agent_key, reasoning)` hashes which the new `execute` still produces.

### 4. UI changes (`cli.py`)

New instance state in `__init__`:

```python
self.current_plan = None  # plan.Plan | None
self._last_task_states: dict[int, str] = {}  # for flash-on-transition detection
self._task_flash_until: dict[int, float] = {}  # task_id -> monotonic timestamp
```

Reset in `handle_input` alongside the other per-turn fields.

New `_agent_worker` chunk handlers:

```python
elif chunk["type"] == "plan_created":
    self.current_plan = chunk["plan"]
elif chunk["type"] == "plan_update":
    self.current_plan = chunk["plan"]
    # Detect state transitions for flash animation.
    now = time.time()
    for task in self.current_plan.tasks:
        prev = self._last_task_states.get(task.id)
        if prev is not None and prev != task.status:
            self._task_flash_until[task.id] = now + 0.15
        self._last_task_states[task.id] = task.status
```

New render method `_render_plan_panel`:

```python
def _render_plan_panel(self):
    if self.current_plan is None or not self.current_plan.tasks:
        return None
    plan = self.current_plan
    done, total = plan.progress()
    body_lines = []
    for task in plan.tasks:
        icon, color = TASK_STATE_STYLE[task.status]
        flashing = time.time() < self._task_flash_until.get(task.id, 0.0)
        line_color = TASK_STATE_FLASH[task.status] if flashing else color
        body_lines.append(Text(f"  {icon} {task.id}. {task.description}", style=line_color))
    title = f"Plan: {plan.title}  · {done}/{total}"
    return Panel(
        Group(*body_lines),
        title=f"[bold {PRIMARY}]{title}[/bold {PRIMARY}]",
        border_style=f"dim {PRIMARY}",
        box=ROUNDED,
        padding=(0, 1),
    )
```

`TASK_STATE_STYLE` and `TASK_STATE_FLASH` are added to `theme.py`:

```python
TASK_STATE_STYLE: dict[str, tuple[str, str]] = {
    "pending":     ("○", _PALETTE.dim),
    "in_progress": ("▸", "#5fafff"),    # executor blue — could be role-aware later
    "done":        ("●", "#5fd75f"),    # green
    "failed":      ("✗", _PALETTE.err),
    "skipped":     ("⊘", _PALETTE.dim),
}

TASK_STATE_FLASH: dict[str, str] = {
    "pending":     _PALETTE.secondary,
    "in_progress": "#afd7ff",            # brighter executor blue
    "done":        "#afffaf",            # brighter green
    "failed":      "#ffafaf",            # brighter red
    "skipped":     _PALETTE.secondary,
}
```

The plan panel is inserted into `_get_current_renderable_ansi`'s `parts` list **above** the architect chip and **below** the tool panels (so it shadows the conversation as a sticky header for the in-flight plan).

## Data flow

```
[user message]
      │
      ▼
MultiAgentSystem.run
      │
      ├─ short-circuit classifier
      │     ├─ trivial → fast-route (no plan ever created)
      │     └─ otherwise: continue
      │
      ├─ architect.plan(user_input)
      │     ├─ kind=single → no Plan, skip to execution
      │     └─ kind=plan  → Plan(title, [tasks])
      │           │
      │           ▼
      │      yield {plan_created, plan}
      │
      ▼
[loop step]
      │
      ├─ architect.execute(plan, task_context)
      │     returns ExecutionIntent
      │
      ├─ apply intent.task_updates  → plan.advance(id, status)
      ├─ apply intent.new_tasks     → plan.insert(after_id, desc)
      ├─ apply intent.current_task_id (mark in_progress)
      │
      ├─ yield {plan_update, plan}
      ├─ yield {intent, ...}      ← existing
      │
      ├─ dispatch sub-agent (existing)
      ├─ collect output, errors, tool results
      ├─ apply existing loop-detection + pivot-recovery
      │
      └─ if complete and plan.is_complete(): break
      │
      ▼
[finalize]
      │
      ├─ mark any lingering in_progress task as skipped
      └─ yield final {plan_update, plan}
```

## Error handling

| Situation | Behavior |
|---|---|
| Architect fails to parse planning JSON twice | Fall back to `kind: single` (no plan UI), execution proceeds with the legacy `analyze`-style intent for that turn. |
| Sub-agent fails during a task (the executor errors out) | Architect's next `execute` call marks the task `failed`; can optionally `new_tasks` a "debug X" insertion after the failed task. |
| Loop detection trips (same plan+agent repeat with no progress) | Existing pivot-recovery still applies. If pivot also stalls, current task → `failed`, plan finalized. |
| User sends new message mid-execution | Existing `handle_input` flow already cancels in-flight work via `self.architect_intent = None` reset. Extend: also finalize `self.current_plan` (mark lingering in_progress → skipped) and clear. |
| Architect returns `complete: true` while pending tasks remain | Remaining pending tasks marked `skipped` during finalize. Logged as a soft anomaly (status line briefly says "plan closed early"). |
| `current_task_id` in intent doesn't match any task in plan | Treated as a no-op for that field; logged once at debug level. |

## Testing

This is a mix of pure logic (testable) and visual rendering (manual).

**Unit tests for `plan.py`** (new `tests/test_plan.py`):
- `Plan.advance` transitions: pending → in_progress, in_progress → done, etc.
- `Plan.advance` to `in_progress` updates `current_task_id`; transitioning out clears it.
- `Plan.insert` produces a new id and inserts at the right position.
- `Plan.is_complete()` returns True only when every task is `done` or `skipped`.
- `Plan.progress()` returns the right counts.
- Invalid status passed to `advance` is a no-op.

**Smoke test for the architect's planning prompt** (new `tests/test_architect_plan.py`):
- Stub out the Ollama client. Feed in a fixed JSON response. Confirm `Architect.plan(...)` returns a populated `Plan` with the expected tasks.
- Feed a `kind: single` response. Confirm `Architect.plan(...)` returns `None`.
- Feed malformed JSON. Confirm `Architect.plan(...)` retries once, then returns `None`.

**Manual TUI verification** (Task 8 of the implementation plan):
- Run a multi-step request ("fix the SSE memory leak" or similar), observe the plan panel appears with 3-7 tasks.
- Watch tasks transition: pending → in_progress → done, with visible flash on each transition.
- Confirm a trivial request ("what's 2 + 2") does not show the plan panel.
- Confirm sending a new message mid-execution discards the old plan cleanly.

## Files

- **NEW:** `plan.py` (~80 lines)
- **NEW:** `tests/test_plan.py` (~60 lines)
- **NEW:** `tests/test_architect_plan.py` (~50 lines)
- **EDIT:** `multi_agent.py` (~150 lines changed: Architect.plan, Architect.execute, prompt rewrites, MultiAgentSystem.run restructure)
- **EDIT:** `cli.py` (~60 lines: state init, chunk handlers, _render_plan_panel, panel insertion in render path)
- **EDIT:** `theme.py` (~15 lines: TASK_STATE_STYLE, TASK_STATE_FLASH)
- **UNCHANGED:** `agent.py`, `tools.py`, `memory.py`, `model_client.py`

## Rollout

Single feature, no flag. Reversible by `git revert`. The existing architect prompt is retired and replaced by the planning+execution prompts. There is no migration concern because the architect state is per-session (not persisted).

## Risks

| Risk | Mitigation |
|---|---|
| Architect prompt rewrite produces lower-quality plans than the current free-form approach | Keep `reflection` field in the execute intent so the chip still shows the architect's thinking. Add a smoke test that uses real Ollama and inspects plan quality before merging. |
| Architect routinely classifies multi-step requests as `single`, suppressing the queue | Lean the prompt heavily toward `kind: plan` when 2+ steps are plausible. Add an env var `EZCLAW_FORCE_PLAN=true` to disable the `single` escape hatch during early testing. |
| Plan rendering eats too much vertical screen real estate on long task lists | Cap displayed tasks at 7 in the panel; if more exist, render `… +N more` on the last line. The 2 ≤ N ≤ 7 prompt constraint should normally prevent this. |
| `current_task_id` and `task_updates` get out of sync when the architect lies about progress | The Python side is the source of truth — we only believe what the architect signals via `task_updates`. The architect can't unilaterally change a task's status by claiming it; the update has to pass through `Plan.advance` which validates the status name. |
| Unicode glyphs (○ ▸ ● ✗ ⊘) may not render in all terminals | Same Task-8-style fallback procedure as the theme work; swap to ASCII (`-` `>` `+` `x` `~`) in `theme.py` if a glyph misrenders. |
