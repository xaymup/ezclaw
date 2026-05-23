"""Unit tests for ChatUI._render_intent_lines (Spec D)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_ui_stub():
    """ChatUI.__new__ without invoking __init__; sufficient for testing
    _render_intent_lines because it touches no instance state."""
    from cli import ChatUI
    return ChatUI.__new__(ChatUI)


def test_full_intent_returns_five_lines():
    ui = _make_ui_stub()
    intent = {
        "agent": "executor",
        "goal": "write the game skeleton",
        "observation": "pygame installed cleanly",
        "critical_thinking": "using 600x400 to keep tests fast",
        "reasoning": "I'll write the module then run it once",
    }
    lines = ui._render_intent_lines(intent)
    assert len(lines) == 5


def test_missing_reasoning_drops_that_line():
    ui = _make_ui_stub()
    intent = {
        "agent": "executor",
        "goal": "g",
        "observation": "o",
        "critical_thinking": "c",
    }
    lines = ui._render_intent_lines(intent)
    assert len(lines) == 4


def test_missing_agent_drops_that_line():
    ui = _make_ui_stub()
    intent = {
        "goal": "g",
        "observation": "o",
        "critical_thinking": "c",
        "reasoning": "r",
    }
    lines = ui._render_intent_lines(intent)
    assert len(lines) == 4


def test_empty_string_fields_are_omitted():
    ui = _make_ui_stub()
    intent = {
        "agent": "executor",
        "goal": "",
        "observation": "o",
        "critical_thinking": "",
        "reasoning": "r",
    }
    lines = ui._render_intent_lines(intent)
    assert len(lines) == 3


def test_empty_intent_returns_empty_list():
    ui = _make_ui_stub()
    assert ui._render_intent_lines({}) == []


def test_each_line_starts_with_eight_space_indent():
    ui = _make_ui_stub()
    intent = {
        "agent": "executor",
        "goal": "g",
        "observation": "o",
        "critical_thinking": "c",
        "reasoning": "r",
    }
    lines = ui._render_intent_lines(intent)
    for line in lines:
        assert line.plain.startswith("        ")
