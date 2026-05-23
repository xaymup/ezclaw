# Delegated Agents → Architect → User — Design

**Date:** 2026-05-23
**Status:** Approved, pending implementation plan.

## Problem

During architect-orchestrated turns, each delegated sub-agent's `content` chunks stream straight to the user (`multi_agent.py:1571`). The exception is `debugger`, where `suppress_user_visible = True` (line 1555) routes its content into `step_output_parts` for architect consumption only. The user never sees raw debugger analysis — the architect turns the Proposed Fix into new plan tasks instead.

Other sub-agents (executor, researcher, general) don't get this treatment. Their narration leaks straight into the chat, often duplicating what the plan panel and tool panels already show. The user reads two stories about the same work: one from the architect (via plan + intent + status) and one from each sub-agent.

## Goal

In architect-orchestrated turns, ALL delegated sub-agent content chunks route silently into the architect (same path as today's debugger). At the end of the architect loop, the architect synthesizes ONE user-facing reply from the accumulated step history. Tool visibility (start/end events, panels) is preserved.

## Non-goals

- No change to fast-routed turns (post-pre-flight: classified `executor`/`general`/`researcher` with < 3 tool kinds). Those stream as today.
- No change to single-step (no-plan) architect mode behavior — only the multi-step loop's per-step yields.
- No change to debugger handling. It was already suppressed; the rest are joining it.
- No change to the streaming chunk protocol between agent and CLI.

## Scope

Affects only the architect orchestration loop in `MultiAgentSystem.run` — specifically the block that calls `agent.chat_stream(agent_context)` per step (around `multi_agent.py:1555-1582`).

## Architecture

### 1. Universal suppression in the per-step block

Change:

```python
suppress_user_visible = (agent_key == "debugger")
```

…to:

```python
suppress_user_visible = True
```

The existing content-gate at lines 1561-1567 already routes content into `step_output_parts` and skips the `yield`. With `suppress_user_visible=True` for everyone, all sub-agent content is captured but not surfaced.

The `final_response = step_output.strip()` assignment at line 1582 currently runs only when `not suppress_user_visible` — i.e., only for non-debugger today. After the change, no `final_response` will be set per-step. We compute it from the synthesis call instead. Remove the conditional assignment entirely; replace with a step-history append.

### 2. End-of-loop synthesis

After the orchestration loop finishes (when the architect signals completion or `step_history` is non-empty), invoke a new method:

```python
synthesis_tokens = self._synthesize_user_reply(
    user_input=user_input,
    step_history=step_history,
    last_step_output=step_output,
)
for chunk in synthesis_tokens:
    yield chunk
```

`synthesis_tokens` is an iterator of `{"type": "content", "content": "..."}` chunks streamed token-by-token from the architect's LLM.

### 3. New method `_synthesize_user_reply`

```python
def _synthesize_user_reply(
    self,
    user_input: str,
    step_history: List[Dict],
    last_step_output: str,
) -> Iterator[Dict[str, Any]]:
    """Have the architect produce the final user-facing reply from the
    accumulated orchestration history. Streams content chunks compatible
    with the existing CLI's chunk handler. Falls back to yielding the
    last sub-agent's text verbatim on any failure."""
```

Prompt shape (sent as user message; architect's existing system prompt is reused):

```
You orchestrated a multi-step plan to answer the user's request. Now write
the FINAL user-facing reply.

Rules:
- Address the user directly. Do not use internal terms like "executor",
  "task ID", "architect", "step".
- Do not re-narrate every step — the user has already seen the plan
  panel update in real time. Focus on the OUTCOME.
- If the plan succeeded, confirm what was delivered. Keep it short.
- If anything failed, say so plainly and stop. Do not pretend work was
  done that wasn't.
- No JSON, no markdown headers, no code fences unless quoting actual code.

User request:
{user_input}

Steps taken (internal record):
{step_history_formatted}

Last sub-agent output:
{last_step_output}

Your reply to the user:
```

Where `step_history_formatted` is a compact rendering of `step_history` (already a list of dicts with `agent`, `tools`, and step content). One line per step:

```
1. executor — write_file, run_shell — "Patched 1 hunk; tests pass"
2. researcher — web_search — "Found pygame docs section on event handling"
```

Streaming: call `self.architect.client.chat(..., stream=True)` and translate each chunk to the CLI's `{"type": "content", "content": ...}` shape.

### 4. Fallback on synthesis failure

If `_synthesize_user_reply` raises or returns nothing, yield the contents of `last_step_output` verbatim as a single content chunk. The user gets the last sub-agent's text — exactly today's behavior for non-debugger agents. Preserves UX continuity if the synthesis call breaks.

### 5. Tool visibility unchanged

The `suppress_user_visible` gate at line 1567 only `continue`s for `chunk["type"] == "content"`. Other chunk types (`tool_start`, `tool_end`, `status`, `intent`, `plan_update`) fall through to the `yield chunk` at line 1571 unconditionally. They keep surfacing to the CLI exactly as today.

### 6. The `final_response` variable

Currently used by `_append_conversation_turn` for history. After this change, `final_response` should be set from the synthesis output (accumulate the synthesized text as it streams; the accumulated string is what the history records).

```python
synthesis_text_parts = []
for chunk in self._synthesize_user_reply(...):
    synthesis_text_parts.append(chunk["content"])
    yield chunk
final_response = "".join(synthesis_text_parts).strip()
```

## Single-step (no-plan) architect mode

The architect operates in single-step mode when no plan is active (see `multi_agent.py:739`). That path uses a different code branch and is OUT OF SCOPE for this spec. Sub-agent content in single-step mode continues to stream to the user as today.

A future spec can extend the synthesis pattern to single-step if useful; not now.

## Testing

### Unit — `tests/test_synthesis_method.py`

Stub the architect's `chat` client to return canned streamed responses.

- `_synthesize_user_reply` yields content chunks in order from a fake streamed response.
- On `RuntimeError` from `chat`, falls back to yielding `last_step_output` as a single content chunk.
- The synthesis prompt embeds the user_input and the step_history's `agent` names.

### Integration — `tests/test_architect_suppression.py`

Stub a `SpecializedAgent` that yields `[content, tool_start, tool_end, content]` in its `chat_stream`. Run a synthetic architect loop (or extract a smaller helper if needed for testability) and assert:

- No `{"type": "content"}` chunks yielded with the sub-agent's content text.
- `tool_start` and `tool_end` chunks DID yield (visibility preserved).
- After the loop, synthesis content arrives as `{"type": "content"}` chunks.

### Manual smoke

1. `ENABLE_MULTI_AGENT=1 python cli.py`.
2. Send a multi-step request: `write a small flask app with one route`.
3. Observe: plan panel + tool rows animate as work happens. The chat area shows status + tool panels but no inline executor narration mid-work.
4. At the end: a final reply streams in, summarizing what was built. The user sees ONE coherent answer rather than fragments from multiple agents.

## Risks & mitigations

- **Synthesis-latency:** one extra architect LLM call at end-of-turn. Comparable to today's per-step architect calls. Acceptable; the user is already used to architect-paced turns.
- **Synthesis hallucination:** the architect could mis-report outcomes. Mitigation: prompt explicitly forbids inventing results and instructs to surface failures. If hallucination shows up in practice, swap synthesis to a deterministic bullet-point template per step.
- **Loss of streaming feel:** today, the user sees text streaming as sub-agents work. Now they see plan + tool panels animate, then synthesis streams at the end. The intermediate gap is filled by tool/plan animation, which is already rich.
- **Test surface:** the architect loop is tightly coupled to the orchestration. Integration tests may need helper extraction (a smaller per-step function) to be testable. The plan should account for this; if needed, refactor the per-step block into a method first.

## Interaction with prior specs

- **Spec A (continuation merging):** independent. Spec A controls end-of-turn finalize lifecycle; this spec controls what content arrives.
- **Spec B (inline code saver):** independent. B parses end-of-turn `current_response_parts`; the synthesis output flows into the same buffer. If the synthesis itself emits a tagged code block, B picks it up correctly.
- **Spec C (executor pre-flight):** independent. C decides whether to enter the architect loop at all; once inside, this spec defines what surfaces to the user.
- **Spec D (intent merge):** complements this. Without sub-agent text streaming directly, the architect's intent block (per D) becomes more important as the "thinking surface" during work.

## Out of scope (explicit deferrals)

- Single-step (no-plan) architect mode synthesis.
- Per-step user-facing summaries (the user only sees the final synthesis).
- Smart filtering of which step outputs to include in the synthesis prompt.
- Multiple synthesis modes (formal/casual/terse).
- Re-streaming the synthesis to the architect's own conversation history (today the LLM call is one-shot; no follow-up).
