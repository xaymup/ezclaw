"""Unit tests for inline_code_saver."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inline_code_saver import ParsedBlock, parse_tagged_blocks


def test_no_blocks_returns_empty():
    assert parse_tagged_blocks("just some prose") == []


def test_untagged_block_ignored():
    text = "Before\n```python\nprint('hi')\n```\nAfter"
    assert parse_tagged_blocks(text) == []


def test_single_tagged_block_parsed():
    text = "intro\n```python:src/foo.py\nprint('hi')\n```\nout"
    blocks = parse_tagged_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].lang == "python"
    assert blocks[0].path == "src/foo.py"
    assert blocks[0].body == "print('hi')"


def test_empty_lang_allowed():
    text = "```:scripts/run.sh\necho hi\n```"
    blocks = parse_tagged_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].lang == ""
    assert blocks[0].path == "scripts/run.sh"


def test_multiple_tagged_blocks_in_one_text():
    text = (
        "first\n```python:a.py\nA = 1\n```\n"
        "middle\n```text:b.txt\nB\n```\n"
        "end"
    )
    blocks = parse_tagged_blocks(text)
    assert len(blocks) == 2
    assert {b.path for b in blocks} == {"a.py", "b.txt"}


def test_tagged_among_untagged_only_tagged_returned():
    text = (
        "```python\nsnippet only\n```\n"
        "```python:saved.py\nsaved = True\n```"
    )
    blocks = parse_tagged_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].path == "saved.py"


def test_trailing_whitespace_on_opener_tolerated():
    text = "```python:src/foo.py   \nbody\n```"
    blocks = parse_tagged_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].path == "src/foo.py"


def test_body_preserved_verbatim_without_trailing_newline():
    text = "```python:foo.py\nline1\nline2\n```"
    blocks = parse_tagged_blocks(text)
    assert blocks[0].body == "line1\nline2"


def test_block_offsets_captured():
    text = "```python:foo.py\nbody\n```"
    blocks = parse_tagged_blocks(text)
    assert blocks[0].start == 0
    assert blocks[0].end == len(text)


def test_unclosed_block_skipped():
    text = "```python:foo.py\nbody but no closing fence"
    assert parse_tagged_blocks(text) == []


# ── plan_saves ──────────────────────────────────────────────────────────────

from inline_code_saver import PlannedSave, plan_saves


def test_plan_rejects_absolute_path(tmp_path):
    block = ParsedBlock(
        raw_open_fence="```python:/etc/passwd",
        lang="python", path="/etc/passwd", body="x",
        start=0, end=0,
    )
    plans = plan_saves([block], workspace_root=str(tmp_path))
    assert len(plans) == 1
    assert plans[0].error is not None
    assert plans[0].abs_target is None


def test_plan_rejects_dotdot_traversal(tmp_path):
    block = ParsedBlock(
        raw_open_fence="```python:../escape.py",
        lang="python", path="../escape.py", body="x",
        start=0, end=0,
    )
    plans = plan_saves([block], workspace_root=str(tmp_path))
    assert plans[0].error is not None


def test_plan_accepts_clean_relative_path(tmp_path):
    block = ParsedBlock(
        raw_open_fence="```python:src/foo.py",
        lang="python", path="src/foo.py", body="x",
        start=0, end=0,
    )
    plans = plan_saves([block], workspace_root=str(tmp_path))
    assert plans[0].error is None
    assert plans[0].abs_target == str(tmp_path / "src" / "foo.py")
    assert plans[0].exists is False


def test_plan_flags_collision(tmp_path):
    target = tmp_path / "existing.py"
    target.write_text("old content")
    block = ParsedBlock(
        raw_open_fence="```python:existing.py",
        lang="python", path="existing.py", body="new",
        start=0, end=0,
    )
    plans = plan_saves([block], workspace_root=str(tmp_path))
    assert plans[0].error is None
    assert plans[0].exists is True


def test_plan_handles_multiple_blocks_in_order(tmp_path):
    blocks = [
        ParsedBlock(raw_open_fence="```python:a.py", lang="python",
                    path="a.py", body="A", start=0, end=0),
        ParsedBlock(raw_open_fence="```python:b.py", lang="python",
                    path="b.py", body="B", start=0, end=0),
    ]
    plans = plan_saves(blocks, workspace_root=str(tmp_path))
    assert len(plans) == 2
    assert plans[0].block.path == "a.py"
    assert plans[1].block.path == "b.py"
