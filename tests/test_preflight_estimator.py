"""Unit tests for the pre-flight tool-line parser."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent import _parse_tool_lines


def test_empty_string_returns_empty_set():
    assert _parse_tool_lines("") == set()


def test_two_clean_tool_names():
    assert _parse_tool_lines("write_file\nrun_shell") == {"write_file", "run_shell"}


def test_dash_and_number_prefixes_stripped():
    text = "- write_file\n1. run_shell\n* read_file"
    assert _parse_tool_lines(text) == {"write_file", "run_shell", "read_file"}


def test_unknown_names_dropped_and_duplicates_collapsed():
    text = "write_file\nfoo_bar\nwrite_file"
    assert _parse_tool_lines(text) == {"write_file"}


def test_prose_lines_rejected():
    text = "first I'd read_file then write_file"
    assert _parse_tool_lines(text) == set()


def test_case_insensitive_recognition():
    text = "WRITE_FILE\nRun_Shell"
    assert _parse_tool_lines(text) == {"write_file", "run_shell"}


def test_trailing_whitespace_tolerated():
    text = "write_file   \n  run_shell\t"
    assert _parse_tool_lines(text) == {"write_file", "run_shell"}


def test_parenthesised_or_bracketed_names_stripped():
    text = "(write_file)\n[run_shell]"
    assert _parse_tool_lines(text) == {"write_file", "run_shell"}


def test_lines_with_inline_prose_after_name_rejected():
    text = "write_file to make the script"
    assert _parse_tool_lines(text) == set()


# ── _estimate_tool_kinds (stub the LLM client) ──────────────────────────────


def _make_mas_with_stubbed_architect(monkeypatch, llm_response: str, tmp_path):
    """Build a MultiAgentSystem whose architect.client.chat returns a canned
    response. Avoids touching real ollama and skips load_skills."""
    import multi_agent

    monkeypatch.setattr(multi_agent, "load_skills", lambda: [])
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "preflight.db"))

    from multi_agent import MultiAgentSystem
    mas = MultiAgentSystem(session_id=None)

    class _StubClient:
        def chat(self, **kwargs):
            return {"message": {"content": llm_response}}

    mas.architect.client = _StubClient()
    return mas


def test_estimate_returns_kind_count(monkeypatch, tmp_path):
    mas = _make_mas_with_stubbed_architect(
        monkeypatch, "write_file\nrun_shell\nrun_shell", tmp_path
    )
    assert mas._estimate_tool_kinds("anything") == 2


def test_estimate_returns_one_for_single_kind(monkeypatch, tmp_path):
    mas = _make_mas_with_stubbed_architect(monkeypatch, "read_file", tmp_path)
    assert mas._estimate_tool_kinds("show me foo.py") == 1


def test_estimate_returns_none_when_response_is_pure_prose(monkeypatch, tmp_path):
    mas = _make_mas_with_stubbed_architect(
        monkeypatch, "I would first read the file then write it", tmp_path
    )
    assert mas._estimate_tool_kinds("anything") is None


def test_estimate_returns_none_when_chat_raises(monkeypatch, tmp_path):
    mas = _make_mas_with_stubbed_architect(monkeypatch, "ignored", tmp_path)

    class _BoomClient:
        def chat(self, **kwargs):
            raise RuntimeError("ollama down")

    mas.architect.client = _BoomClient()
    assert mas._estimate_tool_kinds("anything") is None


def test_estimate_sends_request_text_in_prompt(monkeypatch, tmp_path):
    captured = {}

    mas = _make_mas_with_stubbed_architect(monkeypatch, "write_file", tmp_path)

    class _CapturingClient:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"message": {"content": "write_file"}}

    mas.architect.client = _CapturingClient()
    mas._estimate_tool_kinds("build me a snake game")
    assert any(
        "build me a snake game" in m.get("content", "")
        for m in captured.get("messages", [])
    )
