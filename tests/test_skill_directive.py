"""Skills must be applied, not just shown.

The previous skills_block format read as background context the model
often ignored — a "What's the weather?" query with a matching wttr.in
skill still routed to web_search + 3 web_fetches. These tests pin the
directive format and the planning rule so that skill-matched requests
follow the saved procedure."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import format_skills_block, match_skills


# ── format_skills_block ─────────────────────────────────────────────────────

def test_empty_skills_returns_empty_string():
    assert format_skills_block([]) == ""


def test_directive_block_is_imperative_not_descriptive():
    """The block must read as instructions, not background context.
    A model scanning the prompt should see this as "you MUST do X",
    not as "for your information, X exists"."""
    skill = {
        "name": "Weather Forecast Retrieval",
        "content": (
            "# Skill: Weather Forecast Retrieval\n"
            "## Procedure\n1. web_fetch https://wttr.in/Location\n"
        ),
    }
    block = format_skills_block([skill])

    # The header is unambiguous about authority
    assert "MATCHED SKILLS" in block
    assert "APPLY" in block
    assert "do not re-derive" in block.lower()

    # The skill content survives intact (the model needs the actual procedure)
    assert "wttr.in" in block
    assert "Weather Forecast Retrieval" in block


def test_block_demands_a_named_skill_in_reasoning_when_skipping():
    """If the model chooses NOT to follow a matched skill, the prompt
    must require it to name the skill and explain why. Silent override
    is what we're trying to prevent."""
    skill = {"name": "Email Check", "content": "## Procedure\n…"}
    block = format_skills_block([skill])
    assert "name the skill" in block.lower()


def test_multiple_skills_all_appear_with_clear_separation():
    skills = [
        {"name": "A", "content": "step A1\nstep A2"},
        {"name": "B", "content": "step B1\nstep B2"},
    ]
    block = format_skills_block(skills)
    assert "Skill: A" in block
    assert "Skill: B" in block
    assert "step A1" in block
    assert "step B1" in block
    # Skill name should appear in a heading-ish way so the model parses
    # each as its own unit, not as one merged blob
    assert block.count("### Skill:") == 2


# ── End-to-end: weather query must surface the weather skill ────────────────

def test_match_skills_surfaces_weather_skill_for_weather_query():
    """The matcher must put the Weather Forecast Retrieval skill at the
    top for queries that mention weather. If this regresses, the
    architect can't possibly route correctly because the skill block
    will be empty or wrong."""
    skills = [
        {
            "name": "Weather Forecast Retrieval",
            "content": (
                "## Description\nFetch current/future weather using wttr.in.\n"
                "## Procedure\n1. web_fetch https://wttr.in/Location\n"
            ),
        },
        {
            "name": "Spotify Control",
            "content": "## Description\nControl spotify via CLI.",
        },
        {
            "name": "Image Generation",
            "content": "## Description\nGenerate an image with stable-diffusion.",
        },
    ]

    for query in (
        "What's the weather like?",
        "weather in Giza today",
        "is it raining?",
        "forecast for tomorrow",
    ):
        matched = match_skills(query, skills)
        names = [s["name"] for s in matched]
        assert names, f"no skill matched for {query!r}"
        assert names[0] == "Weather Forecast Retrieval", (
            f"weather query {query!r} matched {names[0]!r} instead of "
            f"Weather Forecast Retrieval — skill routing will fail"
        )


def test_unrelated_query_does_not_force_a_match():
    """Threshold gating: a query with no related skill must return an
    empty list rather than the nearest-but-wrong skill. Otherwise the
    architect sees a 'MATCHED SKILLS' block for an unrelated procedure
    and tries to apply it."""
    skills = [
        {
            "name": "Spotify Control",
            "content": "## Description\nControl spotify via CLI.",
        },
    ]
    matched = match_skills("write a python script to sum two numbers", skills)
    assert matched == [], (
        f"unrelated query matched {[s['name'] for s in matched]!r} — "
        "threshold isn't gating tightly enough"
    )


# ── Architect prompts reinforce skill use ───────────────────────────────────

def test_architect_execute_prompt_has_hard_skill_rule():
    """The execute() system prompt MUST contain language that makes
    skill application a hard rule, not background information. If this
    section gets removed or softened, the matched-skill block will be
    ignored again."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    # The key phrases that make skill use binding
    assert "MATCHED SKILLS" in src
    assert "take precedence over" in src.lower()
    assert "reference the skill by name" in src.lower()


def test_architect_plan_prefers_single_for_skill_covered_requests():
    """When a skill fully covers the request, the planner should
    classify as single rather than building a multi-step plan that
    re-derives the skill's procedure."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    # Confirms the planner is told to prefer `single` for skill-covered work
    assert "prefer `kind: single`" in src or "prefer kind: single" in src
