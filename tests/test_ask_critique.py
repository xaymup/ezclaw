"""Tests for the ask_user and critique tools — UI/LLM back-channels."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools


# ── ask_user ────────────────────────────────────────────────────────────────

def test_ask_user_returns_callback_answer():
    """When a UI handler is registered, ask_user delegates to it and returns
    whatever the user typed."""
    tools.set_ask_user_callback(lambda q: "the answer is 42")
    try:
        out = tools.ask_user("what is the meaning of life?")
        assert out == "the answer is 42"
    finally:
        tools.set_ask_user_callback(None)


def test_ask_user_strips_whitespace():
    tools.set_ask_user_callback(lambda q: "   spaced   \n")
    try:
        out = tools.ask_user("anything")
        assert out == "spaced"
    finally:
        tools.set_ask_user_callback(None)


def test_ask_user_empty_question_is_an_error():
    """Empty or whitespace-only questions should not even reach the UI."""
    tools.set_ask_user_callback(lambda q: pytest_fail("should not be called"))
    try:
        out = tools.ask_user("")
        assert "Error" in out and "empty" in out.lower()
        out = tools.ask_user("   ")
        assert "Error" in out and "empty" in out.lower()
    finally:
        tools.set_ask_user_callback(None)


def test_ask_user_no_handler_falls_back_with_question_visible():
    """Without a registered UI handler, the tool must NOT block forever — it
    should return a string that tells the model to continue with a default."""
    tools.set_ask_user_callback(None)
    out = tools.ask_user("which version?")
    assert "ask_user fallback" in out
    assert "which version?" in out


def test_ask_user_handler_exception_surfaces_as_tool_error():
    def boom(q):
        raise RuntimeError("UI thread crashed")
    tools.set_ask_user_callback(boom)
    try:
        out = tools.ask_user("anything")
        assert "Error" in out and "ask_user failed" in out
        assert "UI thread crashed" in out
    finally:
        tools.set_ask_user_callback(None)


def test_ask_user_empty_response_returns_marker():
    """If the user submits an empty answer (just pressing Enter), we return
    a recognizable marker rather than an empty string — otherwise the
    model can't tell silence apart from absence-of-tool-output."""
    tools.set_ask_user_callback(lambda q: "")
    try:
        out = tools.ask_user("anything?")
        assert "no answer" in out.lower()
    finally:
        tools.set_ask_user_callback(None)


# ── critique ────────────────────────────────────────────────────────────────

def test_critique_uses_registered_callback():
    captured = {}

    def cb(draft, ctx):
        captured["draft"] = draft
        captured["ctx"] = ctx
        return "missing edge case: empty input"

    tools.set_critique_callback(cb)
    try:
        out = tools.critique("def f(x): return x*2", context="the function should double its input")
        assert out == "missing edge case: empty input"
        assert captured["draft"] == "def f(x): return x*2"
        assert "double its input" in captured["ctx"]
    finally:
        tools.set_critique_callback(None)


def test_critique_optional_context_defaults_to_empty():
    captured = {}
    tools.set_critique_callback(lambda d, c: captured.setdefault("ctx", c) or "fine")
    try:
        tools.critique("a draft")
        assert captured["ctx"] == ""
    finally:
        tools.set_critique_callback(None)


def test_critique_empty_draft_is_an_error():
    tools.set_critique_callback(lambda d, c: pytest_fail("should not be called"))
    try:
        out = tools.critique("")
        assert "Error" in out and "empty" in out.lower()
    finally:
        tools.set_critique_callback(None)


def test_critique_no_handler_returns_unavailable_marker():
    tools.set_critique_callback(None)
    out = tools.critique("anything")
    assert "critique unavailable" in out.lower()


def test_critique_handler_exception_surfaces_as_tool_error():
    def boom(d, c):
        raise RuntimeError("LLM timeout")
    tools.set_critique_callback(boom)
    try:
        out = tools.critique("a draft")
        assert "Error" in out and "critique failed" in out
        assert "LLM timeout" in out
    finally:
        tools.set_critique_callback(None)


def pytest_fail(msg):  # local helper used inside lambdas above
    raise AssertionError(msg)
