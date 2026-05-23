"""Unit tests for the continuation-phrase recognizer."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from continuation import _CONTINUATION_PHRASES, is_continuation


def test_known_phrases_match_exact_case():
    for phrase in _CONTINUATION_PHRASES:
        assert is_continuation(phrase) is True


def test_known_phrases_match_uppercase():
    for phrase in _CONTINUATION_PHRASES:
        assert is_continuation(phrase.upper()) is True


def test_phrases_with_whitespace_match():
    assert is_continuation("  continue  ") is True
    assert is_continuation("\tcontinue\n") is True
    assert is_continuation("go on  ") is True


def test_empty_string_returns_false():
    assert is_continuation("") is False
    assert is_continuation("   ") is False
    assert is_continuation("\n") is False


def test_extra_words_reject():
    assert is_continuation("continue please") is False
    assert is_continuation("yes continue") is False


def test_unrelated_text_rejects():
    assert is_continuation("ok") is False
    assert is_continuation("y") is False
    assert is_continuation("hello") is False
    assert is_continuation("now do X instead") is False


def test_continuation_phrase_set_has_expected_members():
    expected = {"continue", "go on", "keep going", "more", "next"}
    assert _CONTINUATION_PHRASES == expected
