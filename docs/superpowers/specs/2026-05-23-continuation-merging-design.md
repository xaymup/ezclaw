# Continuation Merging on Halt — Design

**Date:** 2026-05-23
**Status:** Approved, pending implementation plan.

## Problem

When the agent hits `max_iterations` (or any cap-driven pause), it emits a `[System: agent ran 50 tool-call iterations — pausing here. ... ask me to continue]` content chunk and the generator returns. The CLI then runs its end-of-turn finalize (`cli.py:2095-2102`) — pushes the rendered active panel into `history_ansi`, clears `current_response_parts`. When the user types "continue", a fresh turn begins with a new user bubble and a new assistant panel below the halt-message. Visually it looks like two unrelated messages instead of one task that got resumed.

## Goal

When the user resumes a halted turn with an explicit continuation phrase, the new assistant output extends the same UI panel. The history shows ONE assistant message that grew across two LLM cycles. Non-continuation inputs after a halt commit the previous turn to history and start fresh, as today.

## Non-goals

- Halts from sources other than the iteration cap (e.g., model-emitted explicit pause). The same mechanism would work but is out of scope.
- Code-block formatting / file-name annotations (handled in a separate spec).
- Pause-on-question detection (separate item on the triage list).

## Continuation trigger

A fixed phrase set, case-insensitive, exact-trimmed match only. Anything else starts a fresh turn.

```python
_CONTINUATION_PHRASES = frozenset({"continue", "go on", "keep going", "more", "next"})

def is_continuation(text: str) -> bool:
    return text.strip().lower() in _CONTINUATION_PHRASES
```

"continue please", "yes continue", "ok", "y" → NOT continuations. The strictness avoids false positives at the cost of one moment of "oh right, just type continue."

## Architecture

### New event from the agent

In both `agent.py` (around line 540, where the iteration cap is reached) and `multi_agent.py` (its equivalent cap-hit branch), in addition to the existing `{"type": "content", "content": "..."}` yield, emit a second event right before the loop terminates:

```python
yield {"type": "halt", "reason": "iteration_cap"}
```

Pure signal, no rendering payload. Multi-agent has its own cap site — both must emit.

### CLI halt-state tracking

`EzClawCLI.__init__` gains:

```python
self.halted: bool = False
```

In the dispatch loop in `_run_turn` (around line 2087), add a branch:

```python
elif chunk["type"] == "halt":
    self.halted = True
```

At end-of-turn (around lines 2094-2102), gate the finalize block on `not self.halted`:

```python
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

When halted, the active panel keeps its current content; the UI continues to render `current_response_parts` etc. exactly as it did during generation, just with `is_generating = False`.

### handle_input branching

In `handle_input` (around line 1583, before the existing per-turn reset at 1591), insert a halt branch:

```python
if self.halted:
    if is_continuation(text):
        # Continuation: extend the existing panel.
        self.halted = False
        self.current_response_parts.append("\n\n*↳ continuing…*\n\n")
        # The LLM needs to see "continue" in its context, but the UI
        # does not render a new user bubble for this input.
        # (DB write of the user message happens inside agent.chat_stream
        # as today.)
        self._force_scroll_next_update = True
        self.is_generating = True
        self._run_turn(text)
        return
    else:
        # Non-continuation: finalize the prior halted panel, then proceed
        # as a fresh turn.
        final_renderable = self._get_current_renderable_ansi()
        self.history_ansi.append(final_renderable)
        self.current_response_parts = []
        self.reasoning_chunks = []
        self.tool_executions = []
        self.side_messages = []
        self.halted = False
        # Fall through to the normal new-turn path below.
```

(Exact line/structure adapts to current code — the implementation plan resolves naming.)

### Cancel-during-halt

In the cancel branch (Ctrl+C / `/cancel`), if `self.halted` is True, treat the cancel as "user is moving on" — finalize the partial panel to history, clear `self.halted`, then proceed with normal cancel.

### Halt-message wording

Change the existing halt string in both `agent.py:540` and the multi-agent equivalent to:

```
[System: agent ran {N} tool-call iterations — pausing here. Type "continue"
(or: go on / keep going / more / next) to extend this response, or send a
new prompt to start fresh.]
```

This documents the trigger set inline.

## Data model

No schema changes. The halt state lives only in the CLI process memory; messages already committed to the DB during streaming are untouched. On session restart, history rebuilds normally; a partial halt-then-no-continuation conversation comes back as one continuous assistant turn (since the LLM's content was already in `messages`).

## Testing

- **Unit (`tests/test_continuation_phrases.py`):** `is_continuation` returns True for each phrase in `_CONTINUATION_PHRASES`, case-insensitive, with leading/trailing whitespace; False for empty, for "continue please", for "yes", for "y", for random text.
- **Integration (`tests/test_halt_continuation.py`):**
  - Simulate a halt → call the continuation handler with "continue" → assert `history_ansi` length unchanged, `current_response_parts` still contains the prior content + the `↳ continuing…` marker, `self.halted` is False after.
  - Simulate a halt → call with "actually do X instead" → assert `history_ansi` gained exactly one entry (the finalized prior panel), `current_response_parts` reset, `self.halted` is False, the new turn proceeded.
  - Multiple halts in a row: halt → continue → halt → continue → assert the panel grew, history still has zero entries from these turns until a non-continuation input.

UI-rendering correctness is verified manually (start ezclaw, force a halt via `EZCLAW_MAX_ITERATIONS=1`, observe the merge).

## Risks

- **False positives.** None expected with strict phrase match. If users complain that "continue please" doesn't work, widen the set deliberately rather than introducing fuzzy matching.
- **Stale panel if user walks away during halt.** The active panel renders indefinitely until the next input. Acceptable — the panel content was already streamed and the user can scroll up.
- **Memory: actions table dependency.** Unrelated — this change doesn't touch the action-tracking subsystem.

## Out of scope (deferred)

- Visual treatment of the `↳ continuing…` marker (it'll just render as italic via Markdown for now; if it looks bad, polish in a follow-up).
- Suppressing the halt message from the LLM's own context (it currently goes into `messages` via the content yield; the LLM sees its own halt notice on the next turn). Leaving it in keeps continuity intact.
- Halt detection from sources other than `iteration_cap`.
