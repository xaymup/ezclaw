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
