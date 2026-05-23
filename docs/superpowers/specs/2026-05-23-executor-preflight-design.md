# Executor Pre-Flight Before Fast-Route — Design

**Date:** 2026-05-23
**Status:** Approved, pending implementation plan.

## Problem

`MultiAgentSystem.run` routes requests via `_short_circuit_classify` (`multi_agent.py:1083`). When that classifier returns `"executor"`, the system skips the architect entirely and runs the executor's chat loop directly (`multi_agent.py:1196-1241`). The classifier uses embedding similarity against `ROUTING_EXAMPLES` (line 1044) with a 0.6 threshold; that's tuned for cases like *"read foo.py"* or *"git status"* but also accepts multi-step requests like *"write simple snake game"* because they match `"write a python script to"` strongly.

Concrete fallout: a snake-game request installs pygame via shell, writes the file, and runs it — three distinct tool kinds executed without a plan, no architect chip, no plan UI for the user to follow or modify.

## Goal

After `_short_circuit_classify` returns `"executor"`, run a cheap LLM pre-flight that counts how many distinct tool kinds the request would likely need. ≥ 3 kinds (or pre-flight failure) → fall through to the architect loop. < 3 → fast-route to executor as today.

## Non-goals

- Don't run pre-flight on `general` / `researcher` / `debugger` short-circuits.
- Don't use the pre-flight result for anything other than gating fast-route.
- No cross-turn caching.
- No fallback to a smaller/faster model. Use the architect's model.

## Architecture

A new method on `MultiAgentSystem` and a new module-level constant, both in `multi_agent.py`. No new files; no schema changes.

### Constants

```python
PREFLIGHT_KIND_THRESHOLD = 3  # ≥ this many distinct tool kinds → architect

_KNOWN_TOOL_NAMES = frozenset({
    "apply_diff", "write_file", "run_shell", "read_file", "list_dir",
    "grep_codebase", "web_search", "web_fetch", "recall", "code_outline",
    "git_diff", "git_log", "git_blame", "run_tests", "python_eval",
    "schedule_task", "unschedule_task", "ask_user", "current_datetime",
    "get_system_info",
})
```

The whitelist is the current set of ezclaw tools (verified against `tools.py`). It is duplicated, not derived from `registry.tools`, to keep this check stable when new tools are added — adding a tool means deciding whether it affects routing.

### Method

```python
def _estimate_tool_kinds(self, user_input: str) -> Optional[int]:
    """Single LLM call estimating the distinct tool kinds the executor
    would need to fulfill `user_input`. Returns the count, or None on
    any failure (timeout, parse failure, empty result)."""
```

Body:

1. Build the prompt (see below).
2. Call `self.architect.client.chat(model=self.architect.model, messages=[...],
   options={"temperature": 0.0, "num_ctx": 2048})`. No streaming.
3. Read `resp["message"]["content"]`, split into lines.
4. For each line, lowercase + strip, filter to lines that exactly equal a member of `_KNOWN_TOOL_NAMES`. (No fuzzy matching.)
5. Deduplicate; return `len(set)`. If the set is empty, return `None`.
6. Wrap the whole thing in `try/except Exception`. On any exception, return `None`.

The pre-flight prompt:

```
You are pre-flighting a tool plan. List the EZCLAW tool names you would
need to fulfill this request, ONE PER LINE, no prose, no numbering.
Use ONLY these names:
  apply_diff, write_file, run_shell, read_file, list_dir, grep_codebase,
  web_search, web_fetch, recall, code_outline, git_diff, git_log,
  git_blame, run_tests, python_eval, schedule_task, unschedule_task,
  ask_user, current_datetime, get_system_info

Request: {user_input}

Tools needed (one per line):
```

### Helper for parsing

The parsing logic is small enough to inline in `_estimate_tool_kinds`, but for unit testability it's broken out:

```python
def _parse_tool_lines(text: str) -> set:
    """Extract known tool names from a newline-separated LLM response.
    Lowercases, strips, filters to _KNOWN_TOOL_NAMES, returns the unique
    set. Robust to surrounding prose/numbering/dashes as long as the
    token appears alone on a line after normalization."""
```

Normalization rule: for each line, strip whitespace, strip leading `-`, `*`, `0-9`, `.`, `)`, `(`, then strip again. If the result is in `_KNOWN_TOOL_NAMES`, keep it.

This tolerates outputs like:

```
- write_file
1. run_shell
* read_file
```

…while still rejecting prose like `"first I'd read_file then write_file"`.

### Wiring into `run()`

Right after `short_circuit_agent = self._short_circuit_classify(user_input)` (current `multi_agent.py:1196`), and before the existing `if short_circuit_agent and short_circuit_agent != "debugger":` branch:

```python
if short_circuit_agent == "executor":
    kind_count = self._estimate_tool_kinds(user_input)
    if kind_count is None or kind_count >= PREFLIGHT_KIND_THRESHOLD:
        kind_label = "?" if kind_count is None else str(kind_count)
        yield {"type": "status",
               "content": f"[pre-flight: {kind_label} tool kinds — planning]\n"}
        short_circuit_agent = None
```

When `short_circuit_agent` is None, the existing branch at line 1197 short-circuits (truthy check fails) and control continues to the architect loop below.

The status yield surfaces the pre-flight decision in the UI — the user sees *why* the architect engaged.

## Data flow

```
user_input
    ↓
_short_circuit_classify  ── returns "executor"?
    ↓ yes
_estimate_tool_kinds     ── single LLM call
    ↓
< 3 kinds?  ── yes → fast-route as today
    ↓ no
fall through to architect loop
```

## Testing

### Unit — `tests/test_preflight_estimator.py`

- `_parse_tool_lines("")` → `set()`
- `_parse_tool_lines("write_file\nrun_shell")` → `{"write_file", "run_shell"}`
- `_parse_tool_lines("- write_file\n1. run_shell\n* read_file")` → `{"write_file", "run_shell", "read_file"}`
- `_parse_tool_lines("write_file\nfoo_bar\nwrite_file")` → `{"write_file"}` (dedup + unknown drop)
- `_parse_tool_lines("first I'd read_file then write_file")` → `set()` (prose rejected)
- `_parse_tool_lines("WRITE_FILE\nRun_Shell")` → `{"write_file", "run_shell"}` (case folded)

### Integration — `tests/test_preflight_routing.py`

These tests monkey-patch `_estimate_tool_kinds` directly on a `MultiAgentSystem` instance so the LLM call is bypassed.

- Patch returns `1` → assert the existing fast-route status (`[executor] … (fast-routed)`) is yielded. Architect loop not entered.
- Patch returns `3` → assert the pre-flight status (`[pre-flight: 3 tool kinds — planning]`) is yielded AND the architect loop runs.
- Patch returns `None` (failure) → assert the pre-flight status is yielded with `?` AND the architect loop runs.
- Pre-flight is NOT called when classifier returns `"general"`. (Set up a `_short_circuit_classify` patch that returns `"general"`; assert `_estimate_tool_kinds` was never invoked.)

## Risks & mitigations

- **Latency.** Adds ~200-500ms to every fast-routed request. Acceptable for v1. If real-world traces show this is annoying, switch to a smaller model — the prompt is small enough that a 1-3B quantized model would handle it. One-line change to `model=...`.
- **Prompt drift.** If the model returns pure prose, `_parse_tool_lines` returns an empty set → `_estimate_tool_kinds` returns `None` → safe default kicks in (architect). Worst case is over-planning, which is the direction we want.
- **Over-routing simple requests.** If users find the architect engages too often for genuinely-simple requests (e.g., *"read foo.py and tell me what it does"* — read_file + recall + ask_user could be 3 kinds), bump `PREFLIGHT_KIND_THRESHOLD` to 4 or revisit the prompt.
- **The estimate is approximate.** The model's guess about needed tools doesn't have to match what gets used. The point isn't an audit; it's a coarse "is this multi-step" signal.

## Out of scope (deferred)

- Showing the predicted tool kinds in the UI beyond the status line.
- Smarter routing decisions based on tool kinds (e.g., "this needs web — route to researcher first").
- Tuning the threshold per-domain.
- Caching pre-flight results.
- Using a cheaper model for the pre-flight call.

## Interaction with prior specs

- **Spec A (continuation merging):** independent. Pre-flight runs once per turn; continuation merging is about how panels render. No interaction.
- **Spec B (inline code saver):** independent. Pre-flight gates routing; inline-save handles end-of-turn rendering. No interaction.

All three specs target different code paths in different files; their implementation plans can run in any order.
