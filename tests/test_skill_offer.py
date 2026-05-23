"""Skill-learning offer: after a turn where the user clearly guided
the assistant into a working approach, ezclaw should propose saving a
skill instead of forgetting the lesson next time."""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent import _has_learning_signal, MultiAgentSystem, SpecializedAgent


# ── _has_learning_signal: the cheap precheck ────────────────────────────────

def test_corrective_language_triggers_signal():
    """Phrases the user uses when correcting/teaching the assistant
    must surface as a signal so we run the (slower) architect proposal."""
    for msg in (
        "no, use wttr.in instead",
        "actually the right URL is example.com",
        "you should try the API at /v2/foo",
        "that's incorrect — the endpoint is /weather",
        "use the wttr.in service for this",
        "doesn't work that way — try a GET to /forecast",
        "fix it: the proper command is `gh pr list`",
        "rather than scraping, use the JSON API",
    ):
        assert _has_learning_signal(msg), f"missed cue in: {msg!r}"


def test_normal_conversation_does_not_trigger_signal():
    """We must NOT spend an architect call on every "thanks" or
    "what's the weather?" — only when the user clearly educated us."""
    for msg in (
        "what's the weather like?",
        "thanks!",
        "please summarize this file",
        "show me the diff",
        "list the tests",
        "",
        "hello",
        "is the build green?",
    ):
        assert not _has_learning_signal(msg), f"false positive on: {msg!r}"


def test_empty_or_none_history_is_safe():
    assert _has_learning_signal("") is False
    assert _has_learning_signal(None or "") is False


# ── _propose_skill_from_run: architect-drafted skill ────────────────────────

def _build_mas():
    with patch("multi_agent.build_agent_client", return_value=(MagicMock(), "m")):
        with patch("multi_agent.build_architect_client", return_value=(MagicMock(), "m")):
            with patch.object(SpecializedAgent, "_pre_embed_tools", lambda self: None):
                with patch("multi_agent.load_skills", return_value=[]):
                    with patch("multi_agent.create_memory_tools", lambda db: None):
                        return MultiAgentSystem()


def test_proposal_returns_draft_when_architect_says_propose_true():
    mas = _build_mas()
    fake_json = (
        '{"propose": true, "reason": "user supplied the wttr.in URL",'
        ' "skill": {"name": "Weather Forecast Retrieval",'
        ' "description": "Fetch wttr.in for a location",'
        ' "procedure": "1. web_fetch https://wttr.in/<loc>?format=3\\n'
        '2. summarize"}}'
    )
    with patch.object(mas.architect, "_chat", return_value=fake_json):
        draft = mas._propose_skill_from_run(
            user_input="what's the weather?",
            final_response="Sunny, 30°C in Giza.",
            step_history=[{"agent": "executor", "outcome": "SUCCESS",
                           "tools": ["web_fetch"], "output": "Giza: Sunny 30°C"}],
            recent_history="user: no, use wttr.in\nassistant: ok",
        )
    assert draft is not None
    assert draft["name"] == "Weather Forecast Retrieval"
    assert "wttr.in" in draft["procedure"]
    assert draft.get("reason")


def test_proposal_returns_none_when_architect_declines():
    mas = _build_mas()
    fake_json = (
        '{"propose": false, "reason": "trivial single-shot answer"}'
    )
    with patch.object(mas.architect, "_chat", return_value=fake_json):
        draft = mas._propose_skill_from_run(
            user_input="what is 2+2?",
            final_response="4",
            step_history=[],
            recent_history="",
        )
    assert draft is None


def test_proposal_returns_none_on_malformed_architect_json():
    """Don't crash the run if the architect returns garbage."""
    mas = _build_mas()
    with patch.object(mas.architect, "_chat", return_value="not json at all"):
        draft = mas._propose_skill_from_run(
            user_input="x",
            final_response="y",
            step_history=[],
            recent_history="user: try this instead",
        )
    assert draft is None


def test_proposal_returns_none_when_required_fields_missing():
    """If the architect proposes but omits name or procedure, refuse —
    a half-formed skill would just clutter ~/.ezclaw/skills/."""
    mas = _build_mas()
    fake_json = (
        '{"propose": true, "reason": "ok",'
        ' "skill": {"name": "", "description": "x", "procedure": "y"}}'
    )
    with patch.object(mas.architect, "_chat", return_value=fake_json):
        draft = mas._propose_skill_from_run(
            user_input="x", final_response="y",
            step_history=[], recent_history="actually try this",
        )
    assert draft is None

    fake_json2 = (
        '{"propose": true, "reason": "ok",'
        ' "skill": {"name": "X", "description": "x", "procedure": ""}}'
    )
    with patch.object(mas.architect, "_chat", return_value=fake_json2):
        draft = mas._propose_skill_from_run(
            user_input="x", final_response="y",
            step_history=[], recent_history="actually try this",
        )
    assert draft is None


def test_proposal_truncates_oversized_fields():
    """An aggressive architect could return giant strings — clamp so
    a single bad turn can't write a 1MB skill file."""
    mas = _build_mas()
    huge = "x" * 10_000
    import json as _json
    payload = _json.dumps({
        "propose": True,
        "reason": huge,
        "skill": {
            "name": huge,
            "description": huge,
            "procedure": huge,
        },
    })
    with patch.object(mas.architect, "_chat", return_value=payload):
        draft = mas._propose_skill_from_run(
            user_input="x", final_response="y",
            step_history=[], recent_history="actually try this",
        )
    assert draft is not None
    assert len(draft["name"]) <= 80
    assert len(draft["description"]) <= 300
    assert len(draft["procedure"]) <= 4000
    assert len(draft["reason"]) <= 200
