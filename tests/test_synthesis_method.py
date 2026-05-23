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
    mas = _build_mas(monkeypatch, tmp_path)

    class _BoomClient:
        def chat(self, **kwargs):
            raise RuntimeError("oops")

    mas.architect.client = _BoomClient()
    chunks = list(mas._synthesize_user_reply(
        user_input="x", step_history=[], last_step_output="",
    ))
    assert chunks == []


def test_user_input_embedded_in_prompt(monkeypatch, tmp_path):
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
