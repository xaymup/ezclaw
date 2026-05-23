"""Architect must finalize cleanly when the plan is done.

Reported bug: after asking ezclaw to code a simple game, the
architect completed the plan but then started looking for OTHER
tasks from workspace/memory instead of stopping.

These tests pin the contract:
  - Once plan.is_complete() is True, the architect must declare
    `complete: true` and emit no new_tasks (no scope drift).
  - The synthesis must produce a finalization message with file
    paths, a "how to use" line, and 1-2 optional next-step offers.
  - The self-check must NOT reject a delivery summary just because
    the response doesn't contain the actual source code.
"""

import json
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


# ── Architect prompt has explicit finalization rules ───────────────────────

def test_architect_execute_prompt_has_finalization_section():
    """The execute() prompt must explicitly forbid scope drift after
    the plan is done. If this section is removed or softened, the
    'kept fishing for new tasks' bug returns."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    assert "Finalization (when the plan is done)" in src
    assert "STOP planning new work" in src
    # Specific anti-drift examples
    assert "write tests" in src.lower() and "add documentation" in src.lower()
    assert "commit and push" in src.lower()
    # Memory must explicitly not be a source of new tasks
    assert "Searching memory" in src or "recall_actions" in src


def test_architect_prompt_says_two_failures_are_equally_bad():
    """The 'keep iterating' line previously read as one-sided
    permission to do more work. It now names BOTH failure modes
    (premature completion AND scope drift) so the model treats them
    as equally bad."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    assert "partial result before the goal is met" in src
    assert "drifting past completion" in src


# ── Self-check tolerates delivery summaries ─────────────────────────────────

def test_self_check_accepts_a_delivery_summary_for_a_coding_request():
    """The self-check should NOT reject a summary like 'Wrote game.py.
    Run python game.py' for a request like 'code a game' — the actual
    code is in the file, not the chat."""
    mas = _build_mas()
    # Mock the architect _chat so the self-check returns {ok: true}
    # — this verifies the prompt is sent in a way the architect can
    # cleanly evaluate, not the LLM's actual judgment.
    fake_verdict = '{"ok": true}'
    with patch.object(mas.architect, "_chat", return_value=fake_verdict):
        verdict = mas._self_check_answer(
            user_input="code a simple snake game",
            candidate_response=(
                "**Built the snake game.**\n"
                "Wrote `snake.py` with the Snake/Apple/Game classes "
                "and a pygame loop.\n"
                "Run `python snake.py` to play.\n"
                "Want me to add a high-score file or a difficulty toggle?"
            ),
        )
    assert verdict["ok"] is True


def test_self_check_prompt_explicitly_handles_coding_summaries():
    """Pin the self-check prompt change so the 'files are the
    deliverable; the response confirms what was delivered' guidance
    can't get silently removed."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    assert "Files are the deliverable" in src
    assert "demand the source code inline" in src
    # Specific anti-pedantry rule
    assert '"more code"' in src or "more code" in src


# ── Synthesis prompt produces a delivery summary ───────────────────────────

def test_synthesis_prompt_lists_finalization_shape():
    """The _synthesize_user_reply prompt must instruct the architect
    to produce a finalization message in a specific shape:
      1. one-line outcome
      2. what's where (file paths)
      3. how to use it
      4. optional next steps (as OFFERS)
    """
    import multi_agent
    src = open(multi_agent.__file__).read()
    assert "One-line outcome" in src
    assert "What's where" in src or "Whats where" in src
    assert "How to use it" in src
    assert "Next steps" in src
    # Next-steps are offers, not auto-applied
    assert "OFFERS the user can decline" in src
    assert "NOT auto-applied work" in src


def test_synthesis_failure_branch_skips_finalization_shape():
    """On failure the synthesis should say what failed and stop —
    NOT produce 'how to use it' for work that wasn't done."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    # The rule is present and unambiguous
    assert "Skip the 'how to use' and 'next " in src


# ── Integration: architect doesn't drift after a finished plan ─────────────

def test_orchestrator_breaks_when_architect_declares_complete_after_plan_done(monkeypatch):
    """End-to-end: drive the orchestrator with canned architect
    responses where the plan completes naturally. The orchestrator
    MUST break the loop (call synthesis, exit) instead of letting
    the architect keep going."""
    monkeypatch.setenv("EZCLAW_VALIDATE_CHUNKS", "1")
    mas = _build_mas()

    def fake_stream(_ctx):
        yield {"type": "content", "content": "Wrote snake.py with the Snake class."}
    mas.agents["executor"].chat_stream = fake_stream

    arch_responses = iter([
        # plan() — 2 tasks
        json.dumps({
            "kind": "plan",
            "title": "Build snake game",
            "tasks": [
                {"id": 1, "description": "Write snake.py with Game class"},
                {"id": 2, "description": "Verify the game runs"},
            ],
        }),
        # execute step 1
        json.dumps({
            "kind": "execute", "current_task_id": 1,
            "recommended_agent": "executor", "reasoning": "writing",
            "plan": "write snake.py", "task_updates": [], "new_tasks": [],
            "complete": False,
            "reflection": {"critical_thinking": "minimal game loop"},
        }),
        # execute step 2
        json.dumps({
            "kind": "execute", "current_task_id": 2,
            "recommended_agent": "executor", "reasoning": "verify",
            "plan": "run the game", "task_updates": [], "new_tasks": [],
            "complete": False,
            "reflection": {"critical_thinking": "smoke test"},
        }),
        # execute step 3 — architect correctly declares complete (no
        # new_tasks, no scope drift to "also add tests" etc.)
        json.dumps({
            "kind": "execute", "current_task_id": 2,
            "recommended_agent": "executor", "reasoning": "done",
            "plan": "", "task_updates": [], "new_tasks": [],
            "complete": True,
            "reflection": {"critical_thinking": "delivered"},
        }),
        # Self-check verdict
        '{"ok": true}',
        # Defensive fallback
        '{"ok": true}',
    ])

    def architect_chat(_prompt, **_kw):
        try:
            return next(arch_responses)
        except StopIteration:
            return '{"ok": true}'

    def fake_synthesis(*_args, **_kwargs):
        yield {"type": "content", "content": (
            "**Built the snake game.** Wrote `snake.py`. "
            "Run `python snake.py` to play."
        )}

    chunks = []
    with patch.object(mas, "_short_circuit_classify", return_value=None):
        with patch.object(mas, "_estimate_tool_kinds", return_value=5):
            with patch.object(mas.architect, "_chat", side_effect=architect_chat):
                with patch.object(mas, "_synthesize_user_reply",
                                  side_effect=fake_synthesis):
                    for i, c in enumerate(mas.run("code a simple snake game")):
                        chunks.append(c)
                        # If the loop runs past 50 chunks, that's a
                        # drift regression — fail loudly.
                        assert i < 50, (
                            "orchestrator ran > 50 chunks for a 2-task "
                            "plan — drift regression"
                        )

    # Synthesis fired exactly once and the delivery summary reached
    # the user as content.
    content = "".join(c.get("content", "") for c in chunks if c["type"] == "content")
    assert "snake" in content.lower()
    assert "python snake.py" in content

    # No skill_offer fired (no learning signal in this turn)
    assert not any(c["type"] == "skill_offer" for c in chunks)


def test_orchestrator_refuses_to_advance_when_architect_adds_drift_tasks(monkeypatch):
    """If the architect IS still emitting new_tasks for scope-creep
    after the original plan would be complete, the cap-hit synthesis
    still saves us. The point: even with a misbehaving architect, the
    user gets an answer in bounded time."""
    monkeypatch.setenv("EZCLAW_VALIDATE_CHUNKS", "1")
    mas = _build_mas()

    def fake_stream(_ctx):
        yield {"type": "content", "content": "Wrote a file."}
    mas.agents["executor"].chat_stream = fake_stream

    # Architect always adds new "let me also..." tasks; never completes
    plan_json = json.dumps({
        "kind": "plan",
        "title": "Build game",
        "tasks": [{"id": 1, "description": "Write game.py"},
                  {"id": 2, "description": "Verify"}],
    })
    drift_intent = json.dumps({
        "kind": "execute", "current_task_id": 1,
        "recommended_agent": "executor", "reasoning": "drift",
        "plan": "drift", "task_updates": [],
        "new_tasks": [{"after_id": 1, "description": "now add tests"}],
        "complete": False,
        "reflection": {"critical_thinking": "let me also..."},
    })

    def architect_chat(prompt, **_kw):
        if "PLANNING REQUEST" in prompt or "kind: plan" in prompt.lower():
            return plan_json
        return drift_intent

    def fake_synthesis(*_args, **_kwargs):
        yield {"type": "content", "content": "Cap hit — best-effort summary."}

    chunks = []
    with patch.object(mas, "_short_circuit_classify", return_value=None):
        with patch.object(mas, "_estimate_tool_kinds", return_value=5):
            with patch.object(mas.architect, "_chat", side_effect=architect_chat):
                with patch.object(mas, "_synthesize_user_reply",
                                  side_effect=fake_synthesis):
                    for i, c in enumerate(mas.run("code a game")):
                        chunks.append(c)
                        # Even with a runaway architect, max_steps=40
                        # + cap-hit synthesis means we exit. Allow up
                        # to 500 chunks (status spam) before failing.
                        assert i < 500, "cap-hit synthesis didn't fire"

    content = "".join(c.get("content", "") for c in chunks if c["type"] == "content")
    assert content, "no synthesis fired even after the cap was hit"
