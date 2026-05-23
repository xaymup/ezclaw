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
    """During architect orchestration, sub-agent content does NOT
    reach the run() generator output. Tool chunks do. Synthesis content
    arrives at the end."""
    mas = _build_mas(monkeypatch, tmp_path)

    # Force the run() loop to enter the architect path (skip fast-route).
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: None)

    # Stub the architect to emit one step that's not complete, then one that is.
    step_count = {"n": 0}
    def fake_reflect_and_plan(*args, **kwargs):
        step_count["n"] += 1
        return {
            "recommended_agent": "executor",
            "goal": "do it",
            "complete": step_count["n"] >= 2,
            "plan": "",
        }
    monkeypatch.setattr(mas.architect, "reflect_and_plan", fake_reflect_and_plan, raising=False)

    # Stub the executor's chat_stream: content + tool + content.
    executor = mas.agents["executor"]
    def fake_stream(_input):
        yield {"type": "content", "content": "EXECUTOR_NARRATION_HIDDEN"}
        yield {"type": "tool_start", "name": "write_file", "arguments": {}, "interactive": False}
        yield {"type": "tool_end", "name": "write_file", "result": "ok"}
        yield {"type": "content", "content": "EXECUTOR_TRAILING_HIDDEN"}
    monkeypatch.setattr(executor, "chat_stream", fake_stream)

    # Stub synthesis to yield a known string.
    def fake_synth(user_input, step_history, last_step_output):
        yield {"type": "content", "content": "SYNTH_OUT"}
    monkeypatch.setattr(mas, "_synthesize_user_reply", fake_synth)

    chunks = []
    try:
        for chunk in mas.run("multi-step request"):
            chunks.append(chunk)
            if len(chunks) > 80:  # safety stop
                break
    except Exception:
        pass

    content_texts = [c["content"] for c in chunks if c.get("type") == "content"]
    tool_starts = [c for c in chunks if c.get("type") == "tool_start"]
    tool_ends = [c for c in chunks if c.get("type") == "tool_end"]

    assert all("EXECUTOR_NARRATION_HIDDEN" not in t for t in content_texts), \
        f"Sub-agent content leaked: {content_texts}"
    assert all("EXECUTOR_TRAILING_HIDDEN" not in t for t in content_texts), \
        f"Sub-agent content leaked: {content_texts}"
    assert any("SYNTH_OUT" in t for t in content_texts), \
        f"Synthesis content missing: {content_texts}"
    assert len(tool_starts) >= 1, "Tool start chunks should pass through"
    assert len(tool_ends) >= 1, "Tool end chunks should pass through"
