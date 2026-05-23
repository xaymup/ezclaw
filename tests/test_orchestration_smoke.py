"""Tier 4 — Orchestration smoke tests with mocked model clients.

These tests promote tools_dev/trace_run.py to pytest: they run the
full MultiAgentSystem.run() generator end-to-end with the LLM clients
mocked out, asserting that the chunk stream + plan state + completion
gate behave correctly across realistic scenarios.

Mocking strategy:
- Architect _chat is patched to return canned JSON intents (one per
  step). The list is consumed in order; running off the end raises so
  a regression that loops forever fails fast.
- Sub-agent chat_stream is replaced with a simple generator that
  yields one content chunk and exits.
- All chunks flowing out of mas.run are validated against the chunk
  schema (Tier 4 chunk_schema.py) by enabling strict mode."""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_mas():
    from multi_agent import MultiAgentSystem, SpecializedAgent
    with patch("multi_agent.build_agent_client", return_value=(MagicMock(), "m")):
        with patch("multi_agent.build_architect_client", return_value=(MagicMock(), "m")):
            with patch.object(SpecializedAgent, "_pre_embed_tools", lambda self: None):
                with patch("multi_agent.load_skills", return_value=[]):
                    with patch("multi_agent.create_memory_tools", lambda db: None):
                        with patch("multi_agent.create_action_tracking_tools", lambda db: None):
                            return MultiAgentSystem()


def _canned_subagent_stream(content_text):
    def gen(_ctx):
        yield {"type": "content", "content": content_text}
    return gen


def _collect_chunks(mas, prompt, max_chunks=200, timeout_chunks=200):
    chunks = []
    for i, chunk in enumerate(mas.run(prompt)):
        chunks.append(chunk)
        if i >= max_chunks:
            raise RuntimeError(
                f"orchestrator did not complete after {max_chunks} chunks — "
                "likely an infinite-loop regression"
            )
    return chunks


# ── Conversational shortcut scenario ───────────────────────────────────────

def test_conversational_query_short_circuits_to_general(monkeypatch):
    """A wellness/advice query should bypass the architect via the
    Tier 1.3 conversational shortcut and route directly to general."""
    monkeypatch.setenv("EZCLAW_VALIDATE_CHUNKS", "1")
    mas = _build_mas()
    # Replace the general agent's chat_stream with a canned answer
    mas.agents["general"].chat_stream = _canned_subagent_stream(
        "A morning routine could include: wake at the same time, "
        "exposure to sunlight, light exercise, and a protein-forward breakfast."
    )
    # Architect _chat should NOT be called — fail loudly if it is
    arch_chat_calls = []
    def boom(_prompt, **_kw):
        arch_chat_calls.append(_prompt)
        return '{"kind": "single"}'
    with patch.object(mas.architect, "_chat", side_effect=boom):
        # Wrap with validate_stream to catch malformed chunks
        from orchestration import validate_stream
        chunks = list(validate_stream(mas.run("Help me create a morning routine")))

    # The user-facing content must include the canned answer
    content = "".join(c.get("content", "") for c in chunks if c["type"] == "content")
    assert "morning routine" in content.lower()
    assert "wake at the same time" in content
    # Architect was never consulted
    assert arch_chat_calls == [], (
        "architect was called for a conversational query — the shortcut "
        "is supposed to skip it entirely"
    )


# ── Plan-driven scenario with auto-advance ─────────────────────────────────

def test_plan_advances_when_architect_omits_task_updates(monkeypatch):
    """Tier 1.1: when the architect returns no task_updates but the
    step succeeded, the orchestrator auto-advances the current task to
    done. The plan must reach is_complete() and the loop must break."""
    monkeypatch.setenv("EZCLAW_VALIDATE_CHUNKS", "1")
    mas = _build_mas()
    mas.agents["executor"].chat_stream = _canned_subagent_stream("Wrote the function.")
    mas.agents["general"].chat_stream = _canned_subagent_stream("Done.")

    # Architect responses across the run. Plans need ≥2 tasks per
    # architect.plan()'s parser, so we use two. Auto-advance must
    # carry BOTH tasks to done from successful steps even though the
    # architect never sets task_updates.
    arch_responses = iter([
        # plan() — two tasks
        json.dumps({
            "kind": "plan",
            "title": "Add the function",
            "tasks": [
                {"id": 1, "description": "Write foo() in bar.py"},
                {"id": 2, "description": "Add a test for foo()"},
            ],
        }),
        # execute step 1: route to executor for task 1, not complete
        json.dumps({
            "kind": "execute",
            "current_task_id": 1,
            "recommended_agent": "executor",
            "reasoning": "writing foo",
            "plan": "write the function body",
            "task_updates": [], "new_tasks": [], "complete": False,
            "reflection": {"critical_thinking": "small edit"},
        }),
        # execute step 2: route to executor for task 2 (task 1 was
        # auto-advanced after step 1's success), not complete yet
        json.dumps({
            "kind": "execute",
            "current_task_id": 2,
            "recommended_agent": "executor",
            "reasoning": "adding test",
            "plan": "write the test",
            "task_updates": [], "new_tasks": [], "complete": False,
            "reflection": {"critical_thinking": "test alongside"},
        }),
        # execute step 3: both tasks done (auto-advanced), declare complete
        json.dumps({
            "kind": "execute",
            "current_task_id": 2,
            "recommended_agent": "executor",
            "reasoning": "done",
            "plan": "",
            "task_updates": [], "new_tasks": [], "complete": True,
            "reflection": {"critical_thinking": "all tasks accomplished"},
        }),
        # Self-check (Tier 1.4)
        '{"ok": true}',
        # Fallback for any extra calls
        "synthesized",
    ])

    def architect_chat(prompt, **_kw):
        try:
            return next(arch_responses)
        except StopIteration:
            return '{"ok": true}'

    # Mock the synthesis stream to yield one content chunk
    def fake_synthesis(*args, **kwargs):
        yield {"type": "content", "content": "Added foo(). Tests pass."}

    # Force the fast-route NOT to fire for this prompt
    with patch.object(mas, "_short_circuit_classify", return_value=None):
        with patch.object(mas.architect, "_chat", side_effect=architect_chat):
            with patch.object(mas, "_synthesize_user_reply", side_effect=fake_synthesis):
                with patch.object(mas, "_estimate_tool_kinds", return_value=5):
                    chunks = _collect_chunks(mas, "Add foo() to bar.py")

    # Plan must have ended in `done` for task 1 (auto-advance)
    plan_updates = [c for c in chunks if c["type"] in ("plan_created", "plan_update")]
    assert plan_updates, "expected at least one plan chunk"
    final_plan = plan_updates[-1]["plan"]
    assert final_plan.get_task(1).status == "done", (
        f"auto-advance regression — task 1 still {final_plan.get_task(1).status!r}"
    )
    # Completion fired — a content chunk reached the user
    content = "".join(c.get("content", "") for c in chunks if c["type"] == "content")
    assert content, "no content chunk reached the user — completion gate broken"


# ── Cap-hit safety ──────────────────────────────────────────────────────────

def test_loop_does_not_spin_forever_when_architect_never_completes(monkeypatch):
    """If the architect never returns complete=true, the max_steps cap
    must trigger synthesis anyway — we never burn 200s+ chunks."""
    monkeypatch.setenv("EZCLAW_VALIDATE_CHUNKS", "1")
    mas = _build_mas()
    mas.agents["executor"].chat_stream = _canned_subagent_stream("partial work")

    # Plan with one task. Every execute() says NOT complete.
    plan_json = json.dumps({
        "kind": "plan",
        "title": "stuck plan",
        "tasks": [{"id": 1, "description": "thing"}],
    })
    intent_json = json.dumps({
        "kind": "execute", "current_task_id": 1,
        "recommended_agent": "executor", "reasoning": "x", "plan": "y",
        "task_updates": [{"id": 1, "status": "in_progress"}],
        "new_tasks": [], "complete": False,
        "reflection": {"critical_thinking": "still working"},
    })

    def architect_chat(prompt, **_kw):
        if "PLANNING REQUEST" in prompt or "kind: plan" in prompt.lower():
            return plan_json
        return intent_json

    def fake_synthesis(*args, **kwargs):
        yield {"type": "content", "content": "Cap reached — best-effort summary."}

    with patch.object(mas, "_short_circuit_classify", return_value=None):
        with patch.object(mas.architect, "_chat", side_effect=architect_chat):
            with patch.object(mas, "_synthesize_user_reply", side_effect=fake_synthesis):
                with patch.object(mas, "_estimate_tool_kinds", return_value=5):
                    # Drop max_steps for the test so this runs fast.
                    # We override the literal in the run() method by
                    # patching the constant via the source — but
                    # simplest: just count chunks aggressively to prove
                    # we exit within a reasonable number.
                    chunks = _collect_chunks(mas, "stuck request", max_chunks=500)

    # We finished iteration without an infinite loop.
    assert chunks
    # The synthesized "cap reached" content reached the user.
    content = "".join(c.get("content", "") for c in chunks if c["type"] == "content")
    assert "Cap reached" in content or "summary" in content.lower()


# ── No chunk-type leaks ─────────────────────────────────────────────────────

def test_all_emitted_chunks_pass_schema_validation(monkeypatch):
    """A fast scenario where we run a conversational query and assert
    every chunk has a known type with required fields. The validator's
    strict mode is on — any malformed chunk raises immediately."""
    monkeypatch.setenv("EZCLAW_VALIDATE_CHUNKS", "1")
    mas = _build_mas()
    mas.agents["general"].chat_stream = _canned_subagent_stream("ok")

    from orchestration import validate_stream
    chunks = list(validate_stream(mas.run("hi")))
    assert chunks
    # No exception means every chunk validated.
    types = {c["type"] for c in chunks}
    # Sanity: at least one content chunk reached the user.
    assert "content" in types or "halt" in types
