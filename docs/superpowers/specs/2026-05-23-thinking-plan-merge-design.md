# Architect Intent + Plan Panel Merge — Design

**Date:** 2026-05-23
**Status:** Approved, pending implementation plan.

## Problem

The UI surfaces three competing "thinking-like" panels during a multi-step turn:

- A **reasoning panel** at the top (model's streaming chain-of-thought).
- A **standalone architect chip** mid-stack (`_render_architect_intent` in `cli.py:1110-1179`) carrying the architect's `goal`, `observation`, `critical_thinking`, `reasoning`, and `agent`.
- The **plan panel** at the bottom listing tasks + nested tool rows.

The architect's chip describes work being done on a specific plan task, but renders nowhere near it. The user's eye has to ping-pong between the chip and the task row to connect them.

## Goal

When a plan is active, the architect's most recent intent renders nested under the in_progress task in the plan panel. The standalone chip disappears. When the task moves to `done`, the intent block disappears with it — only the row remains. Single-step (no plan) mode is unchanged.

## Non-goals

- No history-of-thinking per task. Completed task rows lose their intent block.
- No expand/collapse affordance for completed tasks.
- No change to the model-CoT reasoning panel at the top.
- No change to the standalone chip in single-step (no-plan) mode.
- No truncation of long reasoning fields in v1 (defer if it gets noisy).

## Visual

```
Plan: Build snake game · 1/4
  ✓ 1. Install pygame
  ▸ 2. Scaffold snake_game.py                       ← in_progress, bold
        agent: executor
        goal:        write the game skeleton with imports + window setup
        observation: pygame installed cleanly on previous step
        critical:    using 600x400 to keep tests fast; food rendering deferred
        reasoning:   I'll write the module then run it once to verify display
        ├─ ⚡ write_file  snake_game.py  ✓
  · 3. Implement collisions
  · 4. Game over screen
```

The intent lines sit 8 spaces deep, matching the visual indent of nested tool rows. Thinking comes BEFORE tools in the row order — the architect's "thought-then-act" sequence reads naturally top-to-bottom.

## Architecture

All changes confined to `cli.py`. No new files; no schema/data-model changes.

### 1. Suppress the standalone chip when a plan is active

`cli.py:881` currently:

```python
if self.architect_intent:
    parts.append(self._render_architect_intent(self.architect_intent))
```

Change to:

```python
if self.architect_intent and self.current_plan is None:
    parts.append(self._render_architect_intent(self.architect_intent))
```

When a plan exists, the chip path is skipped entirely. `_render_architect_intent` itself is untouched (used by single-step mode).

### 2. Nest the intent under the in_progress task in `_render_plan_panel`

In the per-task loop (`cli.py:1019-1046`), after the task-row append and BEFORE the tool-row loop:

```python
if task.status == "in_progress" and self.architect_intent:
    body_lines.extend(self._render_intent_lines(self.architect_intent))
for tidx, tool in tools_by_task.get(task.id, []):
    body_lines.append(self._render_nested_tool_row(tidx, tool))
```

### 3. New helper `_render_intent_lines(intent)`

```python
def _render_intent_lines(self, intent: dict) -> list:
    """Return a list of Text lines representing the architect's intent,
    indented to sit under the active plan task row.

    The format mirrors the standalone architect chip but as inline lines
    rather than a Panel. Empty fields are omitted.
    """
    lines = []
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

The 8-space indent matches the existing indent used by nested tool rows (`  ├─ `: 2 + 2 + 4 columns). The `reasoning` field renders italic to distinguish it from the structured fields (goal/observation/critical), echoing the standalone chip's existing styling.

### 4. Stale-intent clearing on task transition

When a `plan_update` chunk arrives that changes which task is `in_progress`, the previously-stored architect intent is stale (it was about the previous task). Clear it; wait for the next intent chunk.

In the existing `plan_update` handler (around `cli.py:1996-2005`):

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
        self.architect_intent = None
```

The `architect_intent` is naturally repopulated by the next `intent` chunk from the architect (at `cli.py:2006-2013`).

### 5. Per-turn reset

Already in place (`cli.py:1600` and `cli.py:2297` set `self.architect_intent = None` at turn start and reset). No change required.

## Testing

### Unit — `tests/test_render_intent_lines.py`

Inline test of the new helper. Stub a minimal `EzClawCLI`-like host that exposes `_render_intent_lines`, or use `EzClawCLI.__new__` to instantiate without running `__init__`.

- Full intent (all four fields + agent) → returns 5 `Text` lines.
- Missing reasoning → returns 4 lines (agent + 3 fields).
- Missing agent → returns 4 lines (no agent line).
- Empty string fields treated as missing → omitted.
- The agent line's styling includes the role's color (assert on the Text's style string).
- The reasoning line is rendered italic; other lines are not.

### Integration — manual smoke (documented in spec; no automated test)

UI rendering correctness can't be cheaply asserted in CI. Manual verification:

1. Start ezclaw with `ENABLE_MULTI_AGENT=1`.
2. Send a multi-step request that triggers a plan (`write a small flask app with one route`).
3. Observe the in_progress task carries the architect's intent inline.
4. Observe the standalone architect chip is NOT rendered.
5. As the architect advances `in_progress`, the intent block moves to the new active task. The previously-active task's row no longer carries it.
6. After the turn ends, the plan panel rolls into history with its final state (no intent visible because no task is in_progress).

A short manual-test recipe is added to the README or CONTRIBUTING under "Smoke tests for UI changes."

## Risks & mitigations

- **Plan panel grows taller.** 4-5 added lines under one task per turn. Acceptable.
- **Long reasoning wraps awkwardly.** The Panel that hosts the plan handles wrapping fine; lines wrap inside the panel's content width. If specific fields blow up (>200 chars), truncate-with-ellipsis in a follow-up.
- **Brief gap between intent arriving and `current_task_id` being set.** Edge case: architect emits intent BEFORE the plan_update with `current_task_id`. In that window, no task is in_progress so the intent has nowhere to nest; the chip is suppressed. The user sees a plan with no intent for a few hundred ms. Self-resolves when the plan_update arrives.
- **Stale intent across task transitions.** Handled by clearing `architect_intent` in the `plan_update` branch when in_progress changes.

## Interaction with prior specs

- **Spec A (continuation merging):** independent. Spec A controls when the active area gets finalized to history; the merge here only affects what renders inside the active area.
- **Spec B (inline code saver):** independent. Different code paths.
- **Spec C (executor pre-flight):** independent. Pre-flight is a routing-time decision, before any plan exists.

All four specs can implement in any order.

## Out of scope (explicit deferrals)

- Showing past intents in a completed-task expand panel.
- Truncating reasoning fields.
- Color-coding architect agent transitions.
- Persisting per-task intent history for `recall_actions`-style retrieval. (The current `actions` table tracks tool calls; architect intents could be its own audit trail if useful, but not now.)
