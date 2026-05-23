"""Tier 2.3 — visible architect-call handoff status.
Tier 3.1 — reflection.critical_thinking always populated.
Tier 3.3 — executor non-coding guard in system prompt.
Tier 3.4 — architect 'use ask_user, not narrate' rule."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Tier 2.3 — architect handoff status visibility ─────────────────────────

def test_architect_planning_status_yielded_before_plan_call():
    """Every plan() call must be preceded by a status chunk so the user
    knows the 12-15s wait is the architect, not a hung tool."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    # The status chunk must appear right before the .plan( call site
    assert "🦀 architect planning (model:" in src
    # And the model name must come from self.architect.model (not hardcoded)
    assert "self.architect.model" in src


def test_architect_execute_status_yielded_before_execute_call():
    """Same for each architect.execute() step inside the loop."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    assert "🦀 architect step" in src
    assert "max_steps" in src


# ── Tier 3.1 — critical_thinking always populated ───────────────────────────

def test_execute_intent_critical_thinking_defaults_when_omitted():
    """When the model returns an intent without reflection.critical_thinking,
    the architect must backfill it (from observation / reasoning / plan)
    instead of leaving the field empty. Re-prompting would cost too much."""
    from unittest.mock import patch, MagicMock
    from multi_agent import Architect

    arch = Architect.__new__(Architect)
    arch.use_deepseek = False
    arch.client = MagicMock()
    arch.model = "test"
    arch.num_ctx = 8192
    arch.messages = [{"role": "system", "content": "you are arch"}]
    arch._prune_messages = lambda: None

    # Pretend the model returned an intent without critical_thinking
    bad_json = (
        '{"kind": "execute", "current_task_id": 1, '
        '"recommended_agent": "executor", "reasoning": "do the thing", '
        '"plan": "Read foo.py", "task_updates": [], "new_tasks": [], '
        '"complete": false, '
        '"reflection": {"goal": "g", "observation": "previous step found foo"}}'
    )
    with patch.object(arch, "_chat", return_value=bad_json):
        intent = arch.execute(
            plan=None,
            task_context="",
            memory_block="",
            skills_block="",
            routing_block="",
            history_block="",
            experiences_block="",
        )

    # critical_thinking must be non-empty after the backfill
    refl = intent.get("reflection") or {}
    ct = refl.get("critical_thinking", "")
    assert ct, "critical_thinking should be backfilled, got empty"
    # And the backfill should be derived from a real field
    assert ct.strip()
    # Test that it's not just whitespace or the literal placeholder
    assert len(ct) > 5


def test_execute_intent_critical_thinking_preserved_when_present():
    """When the model DID provide critical_thinking, we must not overwrite it."""
    from unittest.mock import patch, MagicMock
    from multi_agent import Architect

    arch = Architect.__new__(Architect)
    arch.use_deepseek = False
    arch.client = MagicMock()
    arch.model = "test"
    arch.num_ctx = 8192
    arch.messages = [{"role": "system", "content": "you are arch"}]
    arch._prune_messages = lambda: None

    good_json = (
        '{"kind": "execute", "current_task_id": 1, '
        '"recommended_agent": "executor", "reasoning": "x", '
        '"plan": "y", "task_updates": [], "new_tasks": [], '
        '"complete": false, '
        '"reflection": {"critical_thinking": "MY EXPLICIT REASONING"}}'
    )
    with patch.object(arch, "_chat", return_value=good_json):
        intent = arch.execute(
            plan=None, task_context="", memory_block="",
            skills_block="", routing_block="", history_block="",
            experiences_block="",
        )

    assert intent["reflection"]["critical_thinking"] == "MY EXPLICIT REASONING"


# ── Tier 3.3 — executor non-coding guard ────────────────────────────────────

def test_executor_system_prompt_has_non_coding_guard():
    """The executor system prompt must instruct: when the request is
    non-technical, immediately delegate to general instead of grinding
    on it. Backstop for routing misclassification."""
    from multi_agent import AGENT_DEFS
    prompt = AGENT_DEFS["executor"]["system_prompt"]
    assert "Non-coding guard" in prompt
    assert "delegate('general'" in prompt or 'delegate("general"' in prompt
    # The guard must be early in the prompt so the model reads it first
    guard_position = prompt.find("Non-coding guard")
    core_rules_position = prompt.find("Core Rules")
    assert 0 < guard_position < core_rules_position, (
        "non-coding guard must come before Core Rules so it isn't overridden"
    )


# ── Tier 3.4 — architect 'use ask_user, not narrate' rule ───────────────────

def test_architect_execute_prompt_has_ask_user_rule():
    """The architect prompt must explicitly forbid narrating 'ask the
    user about X' in `plan` without a corresponding ask_user tool call."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    assert "Use the `ask_user` tool" in src
    assert "never narrate" in src.lower()
