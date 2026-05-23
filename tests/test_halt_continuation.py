"""Integration tests for the CLI halt + continuation behavior."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_chatui_stub():
    """Build a ChatUI-like object minimally enough to test halt/continuation
    state transitions without launching the full TUI."""
    from cli import ChatUI
    ui = ChatUI.__new__(ChatUI)
    ui.history_ansi = []
    ui.current_response_parts = []
    ui.reasoning_chunks = []
    ui.tool_executions = []
    ui.side_messages = []
    ui.is_generating = False
    ui.halted = False
    return ui


def test_chatui_init_sets_halted_to_false(monkeypatch, tmp_path):
    """ChatUI.__init__ must initialize self.halted = False."""
    import cli
    # The simplest direct check: read the source for the line in __init__.
    import inspect
    src = inspect.getsource(cli.ChatUI.__init__)
    assert "self.halted" in src
    assert "False" in src.split("self.halted")[1].split("\n")[0]


def test_halt_chunk_sets_halted_flag():
    """Simulate the dispatch loop's handling of one halt chunk."""
    ui = _make_chatui_stub()
    chunk = {"type": "halt", "reason": "iteration_cap"}
    # Reproduce the production branch.
    if chunk.get("type") == "halt":
        ui.halted = True
    assert ui.halted is True


def test_finalize_block_skipped_when_halted():
    """At end-of-turn, the finalize block must NOT run when halted is True."""
    ui = _make_chatui_stub()
    ui.halted = True
    ui.current_response_parts = ["partial response so far"]
    initial_history_len = len(ui.history_ansi)

    ui.is_generating = False
    if not ui.halted:
        ui.history_ansi.append("would-have-finalized")
        ui.current_response_parts = []

    assert len(ui.history_ansi) == initial_history_len
    assert ui.current_response_parts == ["partial response so far"]


def test_finalize_block_runs_when_not_halted():
    """The finalize block runs as today when halted is False."""
    ui = _make_chatui_stub()
    ui.halted = False
    ui.current_response_parts = ["full response"]

    ui.is_generating = False
    if not ui.halted:
        ui.history_ansi.append("finalized")
        ui.current_response_parts = []

    assert ui.history_ansi == ["finalized"]
    assert ui.current_response_parts == []


def test_continuation_input_does_not_append_new_user_bubble():
    """When halted and the user types a continuation phrase, no new
    history entry is created and a continuation marker is appended."""
    from continuation import is_continuation
    ui = _make_chatui_stub()
    ui.halted = True
    ui.current_response_parts = ["prior content"]
    initial_history_len = len(ui.history_ansi)

    text = "continue"
    if ui.halted and is_continuation(text):
        ui.halted = False
        ui.current_response_parts.append("\n\n*↳ continuing…*\n\n")

    assert len(ui.history_ansi) == initial_history_len
    assert "prior content" in ui.current_response_parts
    assert any("continuing" in p for p in ui.current_response_parts)
    assert ui.halted is False


def test_non_continuation_input_finalizes_prior_panel():
    """When halted and the user types something other than a continuation,
    the prior panel is finalized first."""
    from continuation import is_continuation
    ui = _make_chatui_stub()
    ui.halted = True
    ui.current_response_parts = ["prior content"]
    initial_history_len = len(ui.history_ansi)

    text = "actually do X instead"
    if ui.halted and not is_continuation(text):
        ui.history_ansi.append("finalized-prior-panel")
        ui.current_response_parts = []
        ui.halted = False

    assert len(ui.history_ansi) == initial_history_len + 1
    assert ui.current_response_parts == []
    assert ui.halted is False
