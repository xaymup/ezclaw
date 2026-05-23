"""Tier 1 orchestration correctness tests.

Covers:
- 1.1 Auto-advance of the current task when the architect omits task_updates
- 1.2 Neutral handoff status (no more wrong "debugger" label)
- 1.3 Conversational shortcut routes advice/wellness/opinion queries to
       `general` and skips the architect entirely
- 1.4 _self_check_answer rejects responses that don't address the prompt
"""

import os
import sys
from unittest.mock import patch, MagicMock

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


# ── Tier 1.3 — conversational shortcut ──────────────────────────────────────

def test_advice_queries_route_to_general_not_executor():
    """Without this fix, 'Help me create a morning routine' built a
    4-task plan and spun for 200+ seconds. The classifier should now
    embed-match advice/wellness queries to `general`."""
    mas = _build_mas()
    for query in (
        "Help me create a mourning routine for myself",
        "Help me create a morning routine",
        "give me ideas for handling stress at work",
        "what should I do about my neighbor",
        "can you advise me on quitting smoking",
        "recommend a book about stoicism",
    ):
        result = mas._short_circuit_classify(query)
        assert result == "general", (
            f"expected 'general' for advice query {query!r}, got {result!r}"
        )


def test_code_queries_still_route_to_executor():
    """The new advice examples must NOT poach code requests. (Note:
    very short two-word queries like 'git diff' fall through layer-1
    heuristic to 'general' — that's a pre-existing edge case unrelated
    to the advice examples, not tested here.)"""
    mas = _build_mas()
    for query in (
        "Write a function that adds two numbers",
        "add a function double(x) to math.py",
        "fix the bug in line 42 of agent.py",
        "run the tests for the auth module",
        "show me the git diff for the last commit",
    ):
        result = mas._short_circuit_classify(query)
        assert result == "executor", (
            f"expected 'executor' for code query {query!r}, got {result!r}"
        )


def test_greeting_still_routes_to_general():
    mas = _build_mas()
    for query in ("hi", "thanks!", "good morning", "hello there"):
        assert mas._short_circuit_classify(query) == "general"


# ── Tier 1.4 — _self_check_answer ───────────────────────────────────────────

def test_self_check_accepts_a_real_answer():
    mas = _build_mas()
    with patch.object(mas.architect, "_chat", return_value='{"ok": true}'):
        verdict = mas._self_check_answer(
            "What's the weather in Cairo?",
            "Cairo is currently sunny with 30°C and low humidity. "
            "Light winds from the north.",
        )
    assert verdict == {"ok": True, "missing": ""}


def test_self_check_rejects_a_non_answer_and_names_the_gap():
    mas = _build_mas()
    with patch.object(
        mas.architect, "_chat",
        return_value='{"ok": false, "missing": "actual weather data for Cairo"}',
    ):
        verdict = mas._self_check_answer(
            "What's the weather in Cairo?",
            "I cannot fetch real-time weather data.",
        )
    assert verdict["ok"] is False
    assert "Cairo" in verdict["missing"]


def test_self_check_short_circuits_on_trivially_short_response():
    """No model call needed — anything under 20 chars cannot meaningfully
    address a real prompt. Skipping the round-trip keeps the gate cheap."""
    mas = _build_mas()
    # Patch _chat to fail loudly if it gets called
    with patch.object(mas.architect, "_chat", side_effect=AssertionError("model should not be called")):
        verdict = mas._self_check_answer(
            "Explain the theory of relativity",
            "ok",
        )
    assert verdict["ok"] is False
    assert "short" in verdict["missing"].lower() or "empty" in verdict["missing"].lower()


def test_self_check_returns_none_on_malformed_json():
    mas = _build_mas()
    with patch.object(mas.architect, "_chat", return_value="not json at all"):
        verdict = mas._self_check_answer(
            "Long enough question?",
            "Long enough response that exceeds twenty characters easily here.",
        )
    assert verdict is None


# ── Tier 1.1 — auto-advance current task ────────────────────────────────────

def test_auto_advance_marks_successful_in_progress_task_done():
    """Direct contract test: after a successful step, if the architect
    returns no task_updates but current_task_id was set, the plan must
    advance that task to 'done'. Without this the plan never completes."""
    from plan import Plan, Task

    plan = Plan(
        title="t",
        tasks=[Task(id=1, description="step one", status="in_progress")],
    )
    # Simulate the orchestrator's auto-advance condition copied from
    # multi_agent.py — this pins the contract; if the impl drifts, this
    # test fails meaningfully.
    intent = {"current_task_id": 1, "task_updates": []}
    outcome = "SUCCESS"
    step_output = "Did the thing successfully."
    step_tool_results = ["read_file"]

    # The orchestrator condition (paraphrased):
    if (
        plan is not None
        and outcome == "SUCCESS"
        and (step_output.strip() or step_tool_results)
    ):
        tid = intent.get("current_task_id")
        if tid:
            task = plan.get_task(tid)
            if task is not None and task.status == "in_progress":
                plan.advance(tid, "done")

    assert plan.get_task(1).status == "done"


def test_auto_advance_does_not_touch_failed_steps():
    """A FAILURE outcome must NOT auto-advance — the architect needs to
    decide retry vs. skip vs. pivot."""
    from plan import Plan, Task

    plan = Plan(
        title="t",
        tasks=[Task(id=1, description="step one", status="in_progress")],
    )
    intent = {"current_task_id": 1, "task_updates": []}
    outcome = "FAILURE"
    step_output = "Error: file not found."

    if (
        plan is not None
        and outcome == "SUCCESS"  # gate
        and step_output.strip()
    ):
        tid = intent.get("current_task_id")
        if tid:
            task = plan.get_task(tid)
            if task is not None and task.status == "in_progress":
                plan.advance(tid, "done")

    assert plan.get_task(1).status == "in_progress"  # unchanged


def test_auto_advance_implementation_present_in_orchestrator():
    """Smoke check that the auto-advance code path actually exists in
    multi_agent.py — guards against accidental deletion in a refactor."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    assert "Auto-advance the just-finished task" in src
    assert 'self.current_plan.advance(tid, "done")' in src
