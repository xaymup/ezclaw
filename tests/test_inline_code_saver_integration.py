"""Integration tests: end-of-turn pipeline saves tagged blocks and
rewrites the response text with save badges. Collision callback is
stubbed."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inline_code_saver import parse_tagged_blocks, plan_saves, apply_save, SaveResult


def _run_end_of_turn(joined_response: str, workspace_root: str,
                     collision_callback=lambda plan: "write"):
    """Simulate the CLI's end-of-turn save pipeline."""
    blocks = parse_tagged_blocks(joined_response)
    plans = plan_saves(blocks, workspace_root=workspace_root)
    results = []
    for plan in plans:
        if plan.error is not None:
            results.append(SaveResult(plan=plan, status="rejected", final_path=None))
            continue
        if plan.exists:
            choice = collision_callback(plan)
            results.append(apply_save(plan, choice))
        else:
            results.append(apply_save(plan, "write"))
    return results


def test_pipeline_writes_two_blocks_to_correct_paths(tmp_path):
    response = (
        "Here are two files:\n"
        "```python:a.py\nA = 1\n```\n"
        "and\n"
        "```text:b.txt\nB\n```\n"
    )
    results = _run_end_of_turn(response, str(tmp_path))
    assert len(results) == 2
    assert (tmp_path / "a.py").read_text() == "A = 1"
    assert (tmp_path / "b.txt").read_text() == "B"
    assert all(r.status == "succeeded" for r in results)


def test_pipeline_collision_rename(tmp_path):
    (tmp_path / "auth.py").write_text("OLD")
    response = "```python:auth.py\nNEW\n```"
    results = _run_end_of_turn(
        response, str(tmp_path),
        collision_callback=lambda plan: "rename",
    )
    assert results[0].status == "renamed"
    assert (tmp_path / "auth.py").read_text() == "OLD"
    assert (tmp_path / "auth.1.py").read_text() == "NEW"


def test_pipeline_collision_skip(tmp_path):
    (tmp_path / "auth.py").write_text("OLD")
    response = "```python:auth.py\nNEW\n```"
    results = _run_end_of_turn(
        response, str(tmp_path),
        collision_callback=lambda plan: "skip",
    )
    assert results[0].status == "skipped"
    assert (tmp_path / "auth.py").read_text() == "OLD"


def test_pipeline_rejected_absolute_path(tmp_path):
    response = "```python:/etc/passwd\nx\n```"
    results = _run_end_of_turn(response, str(tmp_path))
    assert results[0].status == "rejected"
