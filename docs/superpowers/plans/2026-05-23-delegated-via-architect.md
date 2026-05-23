# Delegated Agents → Architect → User Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In architect-orchestrated turns, all sub-agent `content` chunks are suppressed (like today's debugger). After the orchestration loop finishes, the architect synthesizes ONE user-facing reply from `step_history` and streams it as content. Tool/plan visibility is preserved.

**Architecture:** One-line change to the suppression gate in the per-step block. A new method `_synthesize_user_reply(user_input, step_history, last_step_output)` that calls the architect's LLM and yields content chunks. End-of-loop wiring invokes the synthesis and aggregates its output into `final_response`.

**Tech Stack:** Python 3, ollama (already wired), pytest.

**Spec:** `docs/superpowers/specs/2026-05-23-delegated-via-architect-design.md`

---

## File Structure

- **Modify:** `multi_agent.py` — three changes:
  1. The per-step `suppress_user_visible = (agent_key == "debugger")` line becomes `suppress_user_visible = True`.
  2. The `if step_output.strip() and not suppress_user_visible:` guard around `final_response = step_output.strip()` is replaced — `final_response` now comes from the synthesis.
  3. A new method `_synthesize_user_reply(user_input, step_history, last_step_output) -> Iterator[Dict[str, Any]]`.
  4. End-of-loop call to the synthesis, yielding its content chunks and accumulating them into `final_response`.
- **Create:** `tests/test_synthesis_method.py` — unit tests for `_synthesize_user_reply` (stub the architect's LLM client).
- **Create:** `tests/test_architect_suppression.py` — integration test verifying sub-agent content is suppressed and tool chunks pass through.

---

## Task 1: `_synthesize_user_reply` method + unit tests

**Files:**
- Modify: `multi_agent.py` (add the method)
- Create: `tests/test_synthesis_method.py`

- [ ] **Step 1: Write failing tests**

Create `/home/lulu/Projects/ezclaw/tests/test_synthesis_method.py`:

```python
"""Unit tests for MultiAgentSystem._synthesize_user_reply."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_mas(monkeypatch, tmp_path):
    import multi_agent
    monkeypatch.setattr(multi_agent, "load_skills", lambda: [])
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "synth.db"))
    from multi_agent import MultiAgentSystem
    return MultiAgentSystem(session_id=None)


def test_streamed_chunks_yielded_in_order(monkeypatch, tmp_path):
    """The synthesis method yields content chunks corresponding to each
    streamed token from the architect's LLM."""
    mas = _build_mas(monkeypatch, tmp_path)

    streamed_tokens = ["I ", "wrote ", "the ", "file."]

    class _StreamingClient:
        def chat(self, **kwargs):
            assert kwargs.get("stream", False) is True
            for tok in streamed_tokens:
                yield {"message": {"content": tok}}

    mas.architect.client = _StreamingClient()
    chunks = list(mas._synthesize_user_reply(
        user_input="do a thing",
        step_history=[{"agent": "executor", "tools": ["write_file"], "output": "ok"}],
        last_step_output="ok",
    ))
    assert [c["content"] for c in chunks] == streamed_tokens
    assert all(c["type"] == "content" for c in chunks)


def test_fallback_to_last_step_output_on_chat_exception(monkeypatch, tmp_path):
    """If the LLM call raises, the method falls back to yielding the
    last sub-agent's text as a single content chunk."""
    mas = _build_mas(monkeypatch, tmp_path)

    class _BoomClient:
        def chat(self, **kwargs):
            raise RuntimeError("ollama down")

    mas.architect.client = _BoomClient()
    chunks = list(mas._synthesize_user_reply(
        user_input="x",
        step_history=[],
        last_step_output="Here's what the executor said.",
    ))
    assert len(chunks) == 1
    assert chunks[0]["type"] == "content"
    assert "Here's what the executor said." in chunks[0]["content"]


def test_fallback_when_last_step_output_empty_yields_nothing(monkeypatch, tmp_path):
    """If both the LLM AND the fallback are empty, nothing is yielded."""
    mas = _build_mas(monkeypatch, tmp_path)

    class _BoomClient:
        def chat(self, **kwargs):
            raise RuntimeError("oops")

    mas.architect.client = _BoomClient()
    chunks = list(mas._synthesize_user_reply(
        user_input="x",
        step_history=[],
        last_step_output="",
    ))
    assert chunks == []


def test_user_input_embedded_in_prompt(monkeypatch, tmp_path):
    """The prompt sent to the architect includes the user's original request."""
    mas = _build_mas(monkeypatch, tmp_path)
    captured = {}

    class _CapturingClient:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return iter([{"message": {"content": "ok"}}])

    mas.architect.client = _CapturingClient()
    list(mas._synthesize_user_reply(
        user_input="build me a thing",
        step_history=[],
        last_step_output="done",
    ))
    sent_msgs = captured.get("messages", [])
    assert any("build me a thing" in m.get("content", "") for m in sent_msgs)


def test_step_history_summarized_in_prompt(monkeypatch, tmp_path):
    """Each step in step_history surfaces (at least its agent name) in the prompt."""
    mas = _build_mas(monkeypatch, tmp_path)
    captured = {}

    class _CapturingClient:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return iter([{"message": {"content": "ok"}}])

    mas.architect.client = _CapturingClient()
    list(mas._synthesize_user_reply(
        user_input="x",
        step_history=[
            {"agent": "executor", "tools": ["write_file"], "output": "wrote foo.py"},
            {"agent": "researcher", "tools": ["web_search"], "output": "found docs"},
        ],
        last_step_output="found docs",
    ))
    sent_msgs = captured.get("messages", [])
    combined = " ".join(m.get("content", "") for m in sent_msgs)
    assert "executor" in combined
    assert "researcher" in combined
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_synthesis_method.py -v`
Expected: AttributeError — `MultiAgentSystem` has no `_synthesize_user_reply`.

- [ ] **Step 3: Add the method to `MultiAgentSystem`**

In `/home/lulu/Projects/ezclaw/multi_agent.py`, find `class MultiAgentSystem`. Add this method near the other private helpers (e.g., near `_estimate_tool_kinds` from the pre-flight feature):

```python
    def _synthesize_user_reply(
        self,
        user_input: str,
        step_history: List[Dict[str, Any]],
        last_step_output: str,
    ) -> Iterator[Dict[str, Any]]:
        """Have the architect produce the final user-facing reply from the
        accumulated orchestration history. Streams content chunks
        compatible with the existing CLI's chunk handler. Falls back to
        yielding the last sub-agent's text verbatim on any failure."""

        # Compact one-line summary per step for the prompt.
        if step_history:
            lines = []
            for i, step in enumerate(step_history, 1):
                agent = step.get("agent", "?")
                tools = step.get("tools") or []
                output = (step.get("output") or "")[:200]
                tools_str = ", ".join(tools) if tools else "(no tools)"
                lines.append(f"{i}. {agent} — {tools_str} — {output!r}")
            history_block = "\n".join(lines)
        else:
            history_block = "(no steps recorded)"

        prompt = (
            "You orchestrated a multi-step plan to answer the user's "
            "request. Now write the FINAL user-facing reply.\n\n"
            "Rules:\n"
            "- Address the user directly. Do not use internal terms like "
            "\"executor\", \"task ID\", \"architect\", \"step\".\n"
            "- Do not re-narrate every step — the user has already seen "
            "the plan panel update in real time. Focus on the OUTCOME.\n"
            "- If the plan succeeded, confirm what was delivered. Keep it "
            "short.\n"
            "- If anything failed, say so plainly and stop. Do not pretend "
            "work was done that wasn't.\n"
            "- No JSON, no markdown headers, no code fences unless quoting "
            "actual code.\n\n"
            f"User request:\n{user_input}\n\n"
            f"Steps taken (internal record):\n{history_block}\n\n"
            f"Last sub-agent output:\n{last_step_output}\n\n"
            "Your reply to the user:"
        )

        try:
            stream = self.architect.client.chat(
                model=self.architect.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_ctx": 4096},
                stream=True,
            )
            any_yielded = False
            for chunk in stream:
                text = chunk.get("message", {}).get("content", "")
                if text:
                    any_yielded = True
                    yield {"type": "content", "content": text}
            if not any_yielded and last_step_output.strip():
                yield {"type": "content", "content": last_step_output}
        except Exception:
            if last_step_output.strip():
                yield {"type": "content", "content": last_step_output}
            # If last_step_output is also empty, yield nothing.
            return
```

Verify `Iterator` is already imported at the top of `multi_agent.py` (it's used elsewhere in the file). If not, add to imports.

- [ ] **Step 4: Verify tests pass**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_synthesis_method.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add multi_agent.py tests/test_synthesis_method.py && git commit -m "feat(synthesis): _synthesize_user_reply method on MultiAgentSystem

Streams an architect-generated user-facing reply from step_history.
Falls back to last sub-agent output on LLM failure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Universal suppression in per-step block

**Files:**
- Modify: `multi_agent.py` (the `suppress_user_visible` assignment around line 1555)

- [ ] **Step 1: Locate the gate**

Run: `cd /home/lulu/Projects/ezclaw && grep -n "suppress_user_visible" multi_agent.py | head -10`
Expected: shows the current `suppress_user_visible = (agent_key == "debugger")` line and several reads of the variable.

- [ ] **Step 2: Change the gate**

In `/home/lulu/Projects/ezclaw/multi_agent.py`, find:

```python
            suppress_user_visible = (agent_key == "debugger")
```

Change to:

```python
            # All delegated agents now route content through the architect's
            # end-of-turn synthesis (Spec E). Tool/status/plan chunks still
            # surface; only content is suppressed.
            suppress_user_visible = True
```

- [ ] **Step 3: Remove the per-step `final_response = step_output.strip()` assignment**

In the same file, find the existing block:

```python
            if step_output.strip() and not suppress_user_visible:
                # Only update final_response from agents whose output is
                # meant for the user. ...
                final_response = step_output.strip()
```

Replace with:

```python
            # final_response is now produced by _synthesize_user_reply at
            # end-of-loop. Capture step_output into step_history below so
            # the synthesis can summarize it.
```

(That is — delete those three lines; the comment block above documents why.)

- [ ] **Step 4: Verify syntax**

Run: `cd /home/lulu/Projects/ezclaw && python -m py_compile multi_agent.py`
Expected: compiles cleanly.

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add multi_agent.py && git commit -m "feat(synthesis): universal sub-agent content suppression

All delegated agents in the architect loop now have their content
chunks captured into step_output_parts instead of streamed to the
user. The end-of-loop synthesis (next commit) produces the user
reply.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire synthesis into end-of-loop

**Files:**
- Modify: `multi_agent.py` — invoke synthesis after orchestration loop terminates
- Create: `tests/test_architect_suppression.py`

- [ ] **Step 1: Write the failing integration test**

Create `/home/lulu/Projects/ezclaw/tests/test_architect_suppression.py`:

```python
"""Integration tests verifying that sub-agent content is suppressed and
tool chunks pass through; synthesis yields content at end of loop."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_mas(monkeypatch, tmp_path):
    import multi_agent
    monkeypatch.setattr(multi_agent, "load_skills", lambda: [])
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "supr.db"))
    from multi_agent import MultiAgentSystem
    return MultiAgentSystem(session_id=None)


def test_sub_agent_content_suppressed_in_run(monkeypatch, tmp_path):
    """When a sub-agent yields content during architect orchestration,
    that content does NOT reach the run() generator. Tool chunks do."""
    mas = _build_mas(monkeypatch, tmp_path)

    # Force the run() loop to enter the architect path.
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: None)

    # Stub the architect to yield ONE step pointing at the executor with
    # an intent that completes immediately.
    step_count = {"n": 0}
    def fake_reflect_and_plan(*args, **kwargs):
        step_count["n"] += 1
        if step_count["n"] == 1:
            return {
                "recommended_agent": "executor",
                "goal": "do it",
                "complete": False,
                "plan": "",
            }
        return {
            "recommended_agent": "executor",
            "goal": "done",
            "complete": True,
            "plan": "",
        }
    monkeypatch.setattr(mas.architect, "reflect_and_plan", fake_reflect_and_plan, raising=False)

    # Stub the executor's chat_stream to yield a content + tool pair.
    executor = mas.agents["executor"]
    def fake_stream(_input):
        yield {"type": "content", "content": "EXECUTOR_NARRATION_HIDDEN"}
        yield {"type": "tool_start", "name": "write_file", "arguments": {}, "interactive": False}
        yield {"type": "tool_end", "name": "write_file", "result": "ok"}
        yield {"type": "content", "content": "EXECUTOR_TRAILING_HIDDEN"}
    monkeypatch.setattr(executor, "chat_stream", fake_stream)

    # Stub synthesis to yield a known string so we can identify it.
    def fake_synth(user_input, step_history, last_step_output):
        yield {"type": "content", "content": "SYNTH_OUT"}
    monkeypatch.setattr(mas, "_synthesize_user_reply", fake_synth)

    chunks = []
    try:
        for chunk in mas.run("multi-step request"):
            chunks.append(chunk)
            if len(chunks) > 50:  # safety stop
                break
    except Exception:
        pass

    content_texts = [c["content"] for c in chunks if c.get("type") == "content"]
    tool_starts = [c for c in chunks if c.get("type") == "tool_start"]
    tool_ends = [c for c in chunks if c.get("type") == "tool_end"]

    # Sub-agent content must NOT appear.
    assert all("EXECUTOR_NARRATION_HIDDEN" not in t for t in content_texts)
    assert all("EXECUTOR_TRAILING_HIDDEN" not in t for t in content_texts)
    # Synthesis content MUST appear.
    assert any("SYNTH_OUT" in t for t in content_texts)
    # Tool chunks MUST still pass through.
    assert len(tool_starts) == 1
    assert len(tool_ends) == 1
```

- [ ] **Step 2: Verify it fails**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_architect_suppression.py -v`
Expected: FAIL — the synthesis call site doesn't exist yet, so no `SYNTH_OUT` content is yielded.

- [ ] **Step 3: Wire synthesis at end of orchestration loop**

In `/home/lulu/Projects/ezclaw/multi_agent.py`, find the end of `MultiAgentSystem.run`'s orchestration loop — where it exits the `for step in range(max_steps):` or breaks out. After the loop terminates and BEFORE the existing `self._append_conversation_turn(...)` call (search `_append_conversation_turn` to find the natural insertion point), insert:

```python
        # End of orchestration — synthesize the user-facing reply from
        # the accumulated step history. Streams content chunks.
        if step_history:
            last_step_output = step_history[-1].get("output", "") if step_history else ""
            synthesis_parts = []
            for chunk in self._synthesize_user_reply(
                user_input=user_input,
                step_history=step_history,
                last_step_output=last_step_output,
            ):
                synthesis_parts.append(chunk.get("content", ""))
                yield chunk
            final_response = "".join(synthesis_parts).strip()
```

If the existing code already sets `final_response` somewhere after the loop (e.g., a fallback), keep that fallback path active when `step_history` is empty (no orchestration happened).

The exact insertion point depends on where the loop ends. Look for the pattern: end of `for` loop → maybe a `final_response` use → call to `_append_conversation_turn` or similar. The synthesis block goes between the loop end and the `_append_conversation_turn` call.

- [ ] **Step 4: Ensure `step_history` includes `output`**

The synthesis prompt expects each `step_history` entry to have at least `agent` and `output`. Check where `step_history.append(...)` is called in `MultiAgentSystem.run` and confirm `output` (or `step_output`) is included. If not, add it:

```python
            step_history.append({
                "agent": agent_key,
                "tools": step_tool_names,
                "output": step_output[:1000],  # cap to avoid prompt bloat
            })
```

(The dict shape may vary; whatever it currently is, ensure an `output` or equivalent field is present so the synthesis can summarize.)

- [ ] **Step 5: Verify the failing test passes**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_architect_suppression.py -v`
Expected: PASS.

Run the full multi-agent test sweep for regression:

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_synthesis_method.py tests/test_architect_suppression.py tests/test_preflight_estimator.py tests/test_preflight_routing.py tests/test_action_dispatcher_integration.py -v`
Expected: every test passes.

- [ ] **Step 6: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add multi_agent.py tests/test_architect_suppression.py && git commit -m "feat(synthesis): wire end-of-loop synthesis call

After the architect orchestration loop terminates, call
_synthesize_user_reply with the step_history. Stream its content
chunks to the user as the final reply. Tool/plan chunks continue to
pass through during orchestration; only sub-agent content is held back.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Manual smoke

**Files:** none.

- [ ] **Step 1: Multi-step run**

```bash
cd /home/lulu/Projects/ezclaw && ENABLE_MULTI_AGENT=1 python cli.py
```

Type: `write a small flask app with a route and a basic test`

Expected:
- Plan panel + tool rows animate as work happens.
- No inline executor/researcher/general prose in the chat area while sub-agents are working.
- At end-of-turn, a synthesized reply streams in — one coherent paragraph or two summarizing what was built.

- [ ] **Step 2: Fast-routed turn (should NOT trigger synthesis)**

Type: `read tools.py`

Expected: classified as executor, pre-flight returns < 3 kinds, fast-route fires, executor's text streams directly to user as today. Synthesis is NOT invoked. (This verifies the change is scoped to the architect loop.)

- [ ] **Step 3: Failure path**

Stop ollama, send a multi-step request. The architect loop will hit `_synthesize_user_reply` after orchestration; the LLM call inside fails; fallback yields the last step's output.

Expected: a reply still appears (the fallback). No traceback in the chat.

---

## Self-Review

**Spec coverage:**
- Universal suppression — Task 2.
- Synthesis method — Task 1.
- End-of-loop synthesis call — Task 3.
- Tool visibility preserved (only content suppressed) — confirmed in Task 3 integration test.
- Fallback on synthesis failure — Task 1 fallback path; test in Task 1.
- Single-step (no-plan) mode unchanged — Task 2 only changes the per-step block inside the orchestration loop; single-step path is unaffected.
- Manual smoke — Task 4.

**Placeholder scan:** All code shown verbatim. Two "look for the natural insertion point" caveats remain in Task 3 — these are reasonable because the surrounding orchestration code has a specific shape that varies with prior commits, but the EXACT code to insert is given. A skilled engineer can place the block correctly.

**Type consistency:**
- `_synthesize_user_reply` (Task 1) returns `Iterator[Dict[str, Any]]` → consumed via `for chunk in ...` in Task 3.
- Each yielded chunk is `{"type": "content", "content": ...}` matching the existing CLI handler's branch.
- `step_history` is a list of dicts with `agent`, `tools`, `output` — Task 3 Step 4 ensures the `output` field is populated; Task 1's synthesis prompt assumes it.
- `final_response` becomes the joined synthesis content (Task 3) — consumed by `_append_conversation_turn` later for history.
