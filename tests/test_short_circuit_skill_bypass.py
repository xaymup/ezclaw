"""Regression tests for the short-circuit classifier's skill-aware bypass.

Background: short, verb-light prompts like "What's the newest emails" used
to land in Layer 1 of `_short_circuit_classify` and route straight to
`general` — which has no tools and no awareness of saved skills, so the
matched-skills block was never used. The fix consults `match_skills` at a
strict (0.55) threshold and falls through to the architect when a saved
skill genuinely covers the request. These tests pin that behavior.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_mas(monkeypatch, tmp_path, skills):
    """Build a MultiAgentSystem with `skills` injected as the loaded list."""
    import multi_agent

    monkeypatch.setattr(multi_agent, "load_skills", lambda: skills)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "scbypass.db"))
    from multi_agent import MultiAgentSystem
    return MultiAgentSystem(session_id=None)


def test_email_query_falls_through_when_skill_matches(monkeypatch, tmp_path):
    """A short email question must NOT land in `general` when a relevant
    skill is saved. It should fall through to the architect (return None)
    so the matched-skills block has a chance to surface."""
    skills = [{
        "name": "read_emails",
        "content": "# Skill: read_emails\n\n## Description\nRead and parse emails using the himalaya CLI.\n\n## Procedure\n1. Run `himalaya envelope list -p 1` via run_shell.\n",
        "filename": "read_emails.md",
    }]
    mas = _build_mas(monkeypatch, tmp_path, skills)
    # The matcher is what decides the bypass. Stub it deterministically
    # so the test doesn't depend on Ollama embeddings being available.
    monkeypatch.setattr(
        "multi_agent.match_skills",
        lambda user_input, sk, top_n=1, threshold=0.55: list(sk),
    )
    assert mas._short_circuit_classify("What's the newest emails") is None


def test_pure_greeting_still_routes_to_general(monkeypatch, tmp_path):
    """A plain greeting must NOT be dragged into the architect just because
    a skill is loaded — the strict threshold has to actually filter."""
    skills = [{
        "name": "read_emails",
        "content": "# Skill: read_emails\n\n## Description\nRead emails.\n",
        "filename": "read_emails.md",
    }]
    mas = _build_mas(monkeypatch, tmp_path, skills)
    # Simulate a non-match (low similarity for the greeting): match_skills
    # returns [] at threshold=0.55 even though the skill is loaded.
    monkeypatch.setattr(
        "multi_agent.match_skills",
        lambda user_input, sk, top_n=1, threshold=0.55: [],
    )
    assert mas._short_circuit_classify("hi") == "general"
    assert mas._short_circuit_classify("thanks") == "general"


def test_no_skills_loaded_preserves_original_routing(monkeypatch, tmp_path):
    """With no saved skills, the bypass is a no-op — short verb-light
    inputs still route to general exactly as before."""
    mas = _build_mas(monkeypatch, tmp_path, skills=[])
    # match_skills should not even be reached when self.skills is empty,
    # but stub it anyway so the test is robust to refactors.
    monkeypatch.setattr(
        "multi_agent.match_skills",
        lambda *a, **kw: pytest.fail("match_skills was called with no skills loaded"),
    )
    assert mas._short_circuit_classify("hi") == "general"
    assert mas._short_circuit_classify("yo") == "general"


def test_request_verb_inputs_bypass_layer1_regardless_of_skills(monkeypatch, tmp_path):
    """Inputs that contain a request verb were always sent to Layer 2,
    never to general from Layer 1. The skill check should not interfere
    with that path."""
    skills = [{
        "name": "read_emails",
        "content": "# Skill: read_emails\n## Description\nRead emails.\n",
        "filename": "read_emails.md",
    }]
    mas = _build_mas(monkeypatch, tmp_path, skills)
    # Force Layer 2 to return a deterministic agent so we know we reached it.
    monkeypatch.setattr("multi_agent.classify_by_similarity", lambda *a, **kw: "executor")
    # Input contains "show" which is in the request-verb set — Layer 1 skips.
    assert mas._short_circuit_classify("show me my inbox") == "executor"
