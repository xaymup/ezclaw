"""When a sub-agent emits tagged code blocks (```python:path.py) in
its content but doesn't call write_file as a tool, the orchestrator
must catch and save those blocks to disk before Spec E suppresses
the content. Without this, 'code a game' turns finish with a
delivery summary referencing files that were never written."""

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


# ── Auto-save catches tagged blocks emitted by the executor ─────────────────

def test_tagged_python_block_is_saved_to_workspace(tmp_path, monkeypatch):
    """When the executor emits ```python:guess.py with code, the
    orchestrator must save it to workspace/guess.py before Spec E
    suppresses the content. Files in workspace are the deliverable."""
    monkeypatch.chdir(tmp_path)
    # Point WORKSPACE_DIR at our temp tree
    import tools as _tools_mod
    monkeypatch.setattr(_tools_mod, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    os.makedirs(str(tmp_path / "workspace"))

    mas = _build_mas()

    # Executor emits a tagged block — no explicit tool call
    code_body = (
        "import random\n"
        "n = random.randint(1, 100)\n"
        "while True:\n"
        "    g = int(input('Guess: '))\n"
        "    if g == n: print('Right!'); break\n"
    )
    tagged_content = (
        "Here's the number guessing game:\n\n"
        f"```python:guess.py\n{code_body}\n```\n"
        "Run with `python guess.py`."
    )

    def fake_stream(_ctx):
        yield {"type": "content", "content": tagged_content}
    mas.agents["executor"].chat_stream = fake_stream

    arch_responses = iter([
        json.dumps({
            "kind": "plan",
            "title": "Build guessing game",
            "tasks": [
                {"id": 1, "description": "Write guess.py"},
                {"id": 2, "description": "Verify it runs"},
            ],
        }),
        json.dumps({
            "kind": "execute", "current_task_id": 1,
            "recommended_agent": "executor", "reasoning": "write",
            "plan": "write the game", "task_updates": [], "new_tasks": [],
            "complete": False,
            "reflection": {"critical_thinking": "minimal game"},
        }),
        json.dumps({
            "kind": "execute", "current_task_id": 2,
            "recommended_agent": "executor", "reasoning": "verify",
            "plan": "smoke test", "task_updates": [], "new_tasks": [],
            "complete": False,
            "reflection": {"critical_thinking": "quick check"},
        }),
        json.dumps({
            "kind": "execute", "current_task_id": 2,
            "recommended_agent": "executor", "reasoning": "done",
            "plan": "", "task_updates": [], "new_tasks": [],
            "complete": True,
            "reflection": {"critical_thinking": "delivered"},
        }),
        '{"ok": true}',
    ])

    def architect_chat(_p, **_kw):
        try:
            return next(arch_responses)
        except StopIteration:
            return '{"ok": true}'

    def fake_synthesis(*_a, **_kw):
        yield {"type": "content", "content": "Wrote guess.py. Run `python guess.py`."}

    chunks = []
    with patch.object(mas, "_short_circuit_classify", return_value=None):
        with patch.object(mas, "_estimate_tool_kinds", return_value=5):
            with patch.object(mas.architect, "_chat", side_effect=architect_chat):
                with patch.object(mas, "_synthesize_user_reply",
                                  side_effect=fake_synthesis):
                    for c in mas.run("Create a number guessing game"):
                        chunks.append(c)

    # File ACTUALLY exists on disk
    saved_path = tmp_path / "workspace" / "guess.py"
    assert saved_path.exists(), (
        f"guess.py was NOT saved — auto-save layer regression. "
        f"Workspace contents: {list((tmp_path / 'workspace').iterdir())}"
    )
    saved_content = saved_path.read_text()
    assert "random.randint(1, 100)" in saved_content
    assert "Guess:" in saved_content

    # A synthetic write_file tool_start/tool_end pair was emitted for
    # the UI so the user sees the save happen in real time.
    tool_starts = [c for c in chunks if c["type"] == "tool_start"
                   and c.get("name") == "write_file"]
    tool_ends = [c for c in chunks if c["type"] == "tool_end"
                 and c.get("name") == "write_file"]
    assert tool_starts, "no synthetic tool_start emitted for the save"
    assert tool_ends, "no synthetic tool_end emitted for the save"
    # The synthetic tool_end's result mentions the saved path
    assert "guess.py" in tool_ends[0]["result"]


def _two_task_responses(_extra_complete_step=True):
    """Build the standard 5-response canned architect script for a
    2-task plan: plan → step 1 → step 2 → finalize (complete=true) → self-check."""
    steps = [
        json.dumps({"kind": "plan", "title": "x",
                    "tasks": [{"id": 1, "description": "a"},
                              {"id": 2, "description": "b"}]}),
        json.dumps({"kind": "execute", "current_task_id": 1,
                    "recommended_agent": "executor", "reasoning": "x",
                    "plan": "x", "task_updates": [], "new_tasks": [],
                    "complete": False,
                    "reflection": {"critical_thinking": "x"}}),
        json.dumps({"kind": "execute", "current_task_id": 2,
                    "recommended_agent": "executor", "reasoning": "x",
                    "plan": "x", "task_updates": [], "new_tasks": [],
                    "complete": False,
                    "reflection": {"critical_thinking": "x"}}),
    ]
    if _extra_complete_step:
        steps.append(json.dumps({
            "kind": "execute", "current_task_id": 2,
            "recommended_agent": "executor", "reasoning": "done",
            "plan": "", "task_updates": [], "new_tasks": [],
            "complete": True,
            "reflection": {"critical_thinking": "delivered"},
        }))
    steps.append('{"ok": true}')
    return iter(steps)


def test_existing_file_gets_renamed_not_clobbered(tmp_path, monkeypatch):
    """Auto-save must NOT overwrite a file the user already has —
    rename to .1.py instead. Same conservatism as the cli.py path."""
    import tools as _tools_mod
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "guess.py").write_text("# old content\n")
    monkeypatch.setattr(_tools_mod, "WORKSPACE_DIR", str(workspace))

    mas = _build_mas()
    fake_content = "```python:guess.py\nimport random\n```\n"

    def fake_stream(_ctx):
        yield {"type": "content", "content": fake_content}
    mas.agents["executor"].chat_stream = fake_stream

    arch_responses = _two_task_responses()

    def fake_synthesis(*_a, **_kw):
        # Must be >20 chars so the self-check short-circuit doesn't
        # reject it. (A real synthesis is always longer; this only
        # bites in tests with truncated fakes.)
        yield {"type": "content", "content": "Wrote the requested file. Run `python foo.py` to use it."}

    with patch.object(mas, "_short_circuit_classify", return_value=None):
        with patch.object(mas, "_estimate_tool_kinds", return_value=5):
            with patch.object(mas.architect, "_chat",
                              side_effect=lambda *a, **kw: next(arch_responses, '{"ok": true}')):
                with patch.object(mas, "_synthesize_user_reply",
                                  side_effect=fake_synthesis):
                    list(mas.run("update the game"))

    # Original file untouched
    assert (workspace / "guess.py").read_text() == "# old content\n"
    # New version saved to a suffixed name — first available is .1.py
    suffixed = list(workspace.glob("guess.*.py"))
    assert suffixed, f"no renamed file created in {list(workspace.iterdir())}"
    # Pick the lowest-numbered one to verify content
    suffixed.sort(key=lambda p: int(p.stem.rsplit(".", 1)[-1]))
    assert "import random" in suffixed[0].read_text()


def test_step_output_rewritten_with_save_receipt(tmp_path, monkeypatch):
    """After auto-save, the tagged code block in step_output must be
    REPLACED with a `[wrote X]` receipt so the architect's synthesis
    sees evidence of the save (and can mention it) without the
    architect's prompt being bloated by the full code body."""
    import tools as _tools_mod
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(_tools_mod, "WORKSPACE_DIR", str(workspace))

    mas = _build_mas()
    tagged = "Here's the code:\n\n```python:foo.py\nprint('hi')\n```\n\nAll done."

    def fake_stream(_ctx):
        yield {"type": "content", "content": tagged}
    mas.agents["executor"].chat_stream = fake_stream

    arch_responses = _two_task_responses()

    # Capture the step_history that's passed to synthesis
    captured_step_history = []
    def fake_synthesis(user_input, step_history, last_step_output):
        captured_step_history.append(step_history)
        # Must clear the self-check minimum-length gate (20 chars)
        yield {"type": "content", "content": "Wrote foo.py. Run `python foo.py` to use it."}

    with patch.object(mas, "_short_circuit_classify", return_value=None):
        with patch.object(mas, "_estimate_tool_kinds", return_value=5):
            with patch.object(mas.architect, "_chat",
                              side_effect=lambda *a, **kw: next(arch_responses, '{"ok": true}')):
                with patch.object(mas, "_synthesize_user_reply",
                                  side_effect=fake_synthesis):
                    list(mas.run("create foo"))

    # Architect's synthesis saw a step output that mentions the save,
    # NOT the original code block body (which we don't want in the
    # architect's prompt budget).
    assert captured_step_history
    final_step = captured_step_history[-1][-1]
    assert "[wrote" in final_step["output"]
    # The receipt references the saved file. Architect may have run
    # the executor multiple times (one per task), saving to foo.py
    # then foo.1.py — either is fine, both reference `foo` in the name.
    assert "foo" in final_step["output"] and ".py" in final_step["output"]
    # The full code body should NOT be in step_output anymore
    assert "print('hi')" not in final_step["output"]


# ── Prompt-level: executor must prefer real tool calls ──────────────────────

def test_executor_prompt_requires_tool_call_for_file_creation():
    """The executor system prompt now leads with: file creation is a
    tool call, not prose. If this section is removed, agents regress
    to outputting Python in chat without calling write_file."""
    from multi_agent import AGENT_DEFS
    prompt = AGENT_DEFS["executor"]["system_prompt"]
    assert "File creation is a TOOL CALL" in prompt
    assert "write_file" in prompt
    # The Wrong/Right contrast must be present so the model has a
    # concrete example to anchor on
    assert "Wrong:" in prompt and "Right:" in prompt


def test_no_blocks_means_no_save_attempted(tmp_path, monkeypatch):
    """When the executor emits plain prose (no tagged blocks), the
    auto-save layer must be a no-op — no synthetic tool_start chunks,
    no files created."""
    import tools as _tools_mod
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(_tools_mod, "WORKSPACE_DIR", str(workspace))

    mas = _build_mas()
    plain = "I'd recommend reading the docs at example.com for this."

    def fake_stream(_ctx):
        yield {"type": "content", "content": plain}
    mas.agents["executor"].chat_stream = fake_stream

    arch_responses = iter([
        json.dumps({"kind": "plan", "title": "x",
                    "tasks": [{"id": 1, "description": "a"},
                              {"id": 2, "description": "b"}]}),
        json.dumps({"kind": "execute", "current_task_id": 1,
                    "recommended_agent": "executor", "reasoning": "x",
                    "plan": "x", "task_updates": [], "new_tasks": [],
                    "complete": False,
                    "reflection": {"critical_thinking": "x"}}),
        json.dumps({"kind": "execute", "current_task_id": 2,
                    "recommended_agent": "executor", "reasoning": "x",
                    "plan": "x", "task_updates": [], "new_tasks": [],
                    "complete": True,
                    "reflection": {"critical_thinking": "x"}}),
        '{"ok": true}',
    ])

    def fake_synthesis(*_a, **_kw):
        yield {"type": "content", "content": "ok"}

    chunks = []
    with patch.object(mas, "_short_circuit_classify", return_value=None):
        with patch.object(mas, "_estimate_tool_kinds", return_value=5):
            with patch.object(mas.architect, "_chat",
                              side_effect=lambda *a, **kw: next(arch_responses, '{"ok": true}')):
                with patch.object(mas, "_synthesize_user_reply",
                                  side_effect=fake_synthesis):
                    for c in mas.run("recommend something"):
                        chunks.append(c)

    # No write_file synthetic chunks
    assert not any(c.get("type") == "tool_start"
                   and c.get("name") == "write_file" for c in chunks)
    # No files created
    assert list(workspace.iterdir()) == []
