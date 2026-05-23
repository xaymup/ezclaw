"""Tests for ChatUI._resolve_user_name — the priority chain that decides
what to put on the user-message bubble."""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli


def _mk_ui_with_memories(memories):
    """Construct a minimal ChatUI instance with a stubbed db.search_memories_hybrid
    that returns the given list of memory strings."""
    ui = cli.ChatUI.__new__(cli.ChatUI)
    ui.agent = MagicMock()
    ui.agent.db.search_memories_hybrid.return_value = memories
    return ui


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("EZCLAW_USER", "Captain")
    ui = _mk_ui_with_memories(["my name is Alice"])
    assert ui._resolve_user_name() == "Captain"


def test_env_override_alternate_var(monkeypatch):
    monkeypatch.delenv("EZCLAW_USER", raising=False)
    monkeypatch.setenv("EZCLAW_USER_NAME", "Lulu")
    ui = _mk_ui_with_memories([])
    assert ui._resolve_user_name() == "Lulu"


def test_falls_back_to_memory_when_no_env(monkeypatch):
    monkeypatch.delenv("EZCLAW_USER", raising=False)
    monkeypatch.delenv("EZCLAW_USER_NAME", raising=False)
    ui = _mk_ui_with_memories(["my name is alice"])
    assert ui._resolve_user_name() == "Alice"


def test_memory_pattern_i_am(monkeypatch):
    monkeypatch.delenv("EZCLAW_USER", raising=False)
    monkeypatch.delenv("EZCLAW_USER_NAME", raising=False)
    ui = _mk_ui_with_memories(["I am Bob, a software engineer"])
    assert ui._resolve_user_name() == "Bob"


def test_memory_pattern_users_name_is(monkeypatch):
    monkeypatch.delenv("EZCLAW_USER", raising=False)
    monkeypatch.delenv("EZCLAW_USER_NAME", raising=False)
    ui = _mk_ui_with_memories(["The user's name is Charlie"])
    assert ui._resolve_user_name() == "Charlie"


def test_memory_filters_garbage_words(monkeypatch):
    """The regex is loose; if it accidentally grabs a junk word like
    'the' or 'a', the filter list should reject it and we should fall
    back further down the chain."""
    monkeypatch.delenv("EZCLAW_USER", raising=False)
    monkeypatch.delenv("EZCLAW_USER_NAME", raising=False)
    ui = _mk_ui_with_memories(["I am the user", "name is going forward"])
    # Both should be filtered → falls through to getpass.getuser()
    name = ui._resolve_user_name()
    assert name not in ("The", "Going")
    # The fallback returns the system username capitalized; just confirm
    # we got SOMETHING non-empty and not the rejected words.
    assert isinstance(name, str) and len(name) > 0


def test_falls_back_to_system_user_when_memory_empty(monkeypatch):
    monkeypatch.delenv("EZCLAW_USER", raising=False)
    monkeypatch.delenv("EZCLAW_USER_NAME", raising=False)
    ui = _mk_ui_with_memories([])
    import getpass
    expected = getpass.getuser().capitalize()
    assert ui._resolve_user_name() == expected


def test_memory_lookup_failure_falls_back_silently(monkeypatch):
    monkeypatch.delenv("EZCLAW_USER", raising=False)
    monkeypatch.delenv("EZCLAW_USER_NAME", raising=False)
    ui = cli.ChatUI.__new__(cli.ChatUI)
    ui.agent = MagicMock()
    ui.agent.db.search_memories_hybrid.side_effect = RuntimeError("db down")
    # Must not raise; falls through to system user
    name = ui._resolve_user_name()
    assert isinstance(name, str) and len(name) > 0
