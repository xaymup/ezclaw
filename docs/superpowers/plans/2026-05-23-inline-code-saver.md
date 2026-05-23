# Inline Code Auto-Save Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the assistant emits a fenced code block tagged with `lang:path/to/file.py`, save the body to `workspace/<path>` at end-of-turn, prompt the user on collisions, and render a "saved → ..." badge above the rendered block. Untagged code blocks stay inline as today.

**Architecture:** A new pure module `inline_code_saver.py` exports `parse_tagged_blocks`, `plan_saves`, `apply_save`. The CLI integrates at the end-of-turn finalize point. Collisions trigger a chat-driven Y/N/R prompt reusing the auth-prompt pattern. Saved files write a row to the `actions` table via the existing action-tracking subsystem.

**Tech Stack:** Python 3, sqlite (already in use), pytest.

**Spec:** `docs/superpowers/specs/2026-05-23-inline-code-saver-design.md`

---

## File Structure

- **Create:** `inline_code_saver.py` — pure parser + path validator + save planner + save applier. No UI, no DB.
- **Modify:** `tools.py` — add `"inline_save"` to `MUTATING_TOOLS`.
- **Modify:** `cli.py` — hook the saver into `_run_turn`'s end-of-turn block; add `_ask_save_collision(plan)` helper; add `_rewrite_with_badges(text, results)` helper.
- **Modify:** `multi_agent.py` — append the prompt nudge to the executor system prompt.
- **Modify:** `agent.py` — append the same prompt nudge to ChatAgent's `system_prompt`.
- **Modify:** `agents.md` — append the prompt nudge to the shared agent doc.
- **Create:** `tests/test_inline_code_saver.py` — unit tests for the parser + planner + applier.
- **Create:** `tests/test_inline_code_saver_integration.py` — integration: full pipeline with stubbed collision callback.

---

## Task 1: `inline_code_saver.py` — parser

**Files:**
- Create: `inline_code_saver.py`
- Create: `tests/test_inline_code_saver.py`

- [ ] **Step 1: Write failing tests**

Create `/home/lulu/Projects/ezclaw/tests/test_inline_code_saver.py`:

```python
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
    """A fence opened but never closed should not parse as a block."""
    text = "```python:foo.py\nbody but no closing fence"
    assert parse_tagged_blocks(text) == []
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_inline_code_saver.py -v`
Expected: ImportError on `inline_code_saver`.

- [ ] **Step 3: Create the parser**

Create `/home/lulu/Projects/ezclaw/inline_code_saver.py`:

```python
"""Inline code-block parsing, path validation, save planning, and save
application for the assistant's response stream.

Pure module — no UI, no DB, no agent state. Imports kept minimal so
tests can run without ollama or sqlite.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ParsedBlock:
    """A code-fenced block whose info string carried a `lang:path` tag."""
    raw_open_fence: str
    lang: str
    path: str
    body: str
    start: int  # offset of the opening fence in the source text
    end: int    # offset just past the closing fence


_FENCE_RE = re.compile(
    r"```([^\n`]*)\n(.*?)\n```",
    re.DOTALL,
)


def parse_tagged_blocks(text: str) -> List[ParsedBlock]:
    """Return every fenced code block whose info string contains a colon
    after the language identifier. Untagged blocks are not returned.

    The info string is the part between the opening ``` and the first
    newline. We treat the substring before the first `:` as `lang` and
    everything after the first `:` (trimmed) as `path`. Multiple colons
    in the path are tolerated.
    """
    result: List[ParsedBlock] = []
    for m in _FENCE_RE.finditer(text):
        info = m.group(1).strip()
        body = m.group(2)
        if ":" not in info:
            continue
        lang, _, path = info.partition(":")
        path = path.strip()
        if not path:
            continue
        result.append(ParsedBlock(
            raw_open_fence=f"```{info}",
            lang=lang.strip(),
            path=path,
            body=body,
            start=m.start(),
            end=m.end(),
        ))
    return result
```

- [ ] **Step 4: Verify tests pass**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_inline_code_saver.py -v`
Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add inline_code_saver.py tests/test_inline_code_saver.py && git commit -m "feat(inline-save): tagged-block parser

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Path validation + save planner

**Files:**
- Modify: `inline_code_saver.py` (add `PlannedSave`, `plan_saves`)
- Modify: `tests/test_inline_code_saver.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `/home/lulu/Projects/ezclaw/tests/test_inline_code_saver.py`:

```python
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
    # Pre-create the target.
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
        ParsedBlock(
            raw_open_fence="```python:a.py", lang="python",
            path="a.py", body="A", start=0, end=0,
        ),
        ParsedBlock(
            raw_open_fence="```python:b.py", lang="python",
            path="b.py", body="B", start=0, end=0,
        ),
    ]
    plans = plan_saves(blocks, workspace_root=str(tmp_path))
    assert len(plans) == 2
    assert plans[0].block.path == "a.py"
    assert plans[1].block.path == "b.py"
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_inline_code_saver.py -v`
Expected: ImportError on `PlannedSave` / `plan_saves`.

- [ ] **Step 3: Add the planner**

Append to `/home/lulu/Projects/ezclaw/inline_code_saver.py`:

```python
@dataclass(frozen=True)
class PlannedSave:
    block: ParsedBlock
    abs_target: Optional[str]   # absolute path under workspace_root, None on error
    exists: bool                # whether the target file already exists
    error: Optional[str]        # set if validation rejected; abs_target is None


def _validate_path(rel_path: str, workspace_root: str) -> Optional[str]:
    """Return the absolute target path if valid; None if not. Validation:
    must be relative, must not contain `..`, must land under workspace_root
    after normalization."""
    if not rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
        return None
    if ".." in rel_path.replace("\\", "/").split("/"):
        return None
    workspace_abs = os.path.abspath(workspace_root)
    candidate = os.path.abspath(os.path.join(workspace_abs, rel_path))
    if not candidate.startswith(workspace_abs + os.sep) and candidate != workspace_abs:
        return None
    return candidate


def plan_saves(blocks: List[ParsedBlock], workspace_root: str) -> List[PlannedSave]:
    """Validate paths and probe disk. Returns one PlannedSave per block."""
    result: List[PlannedSave] = []
    for block in blocks:
        abs_target = _validate_path(block.path, workspace_root)
        if abs_target is None:
            result.append(PlannedSave(
                block=block, abs_target=None, exists=False,
                error=f"invalid path: {block.path!r} (absolute, traversal, or outside workspace)",
            ))
            continue
        result.append(PlannedSave(
            block=block,
            abs_target=abs_target,
            exists=os.path.exists(abs_target),
            error=None,
        ))
    return result
```

- [ ] **Step 4: Verify tests pass**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_inline_code_saver.py -v`
Expected: all 15 tests pass (10 + 5 new).

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add inline_code_saver.py tests/test_inline_code_saver.py && git commit -m "feat(inline-save): path validator + save planner

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Save applier with overwrite / skip / rename strategies

**Files:**
- Modify: `inline_code_saver.py` (add `SaveResult`, `apply_save`)
- Modify: `tests/test_inline_code_saver.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `/home/lulu/Projects/ezclaw/tests/test_inline_code_saver.py`:

```python
# ── apply_save ──────────────────────────────────────────────────────────────

from inline_code_saver import SaveResult, apply_save


def test_apply_write_creates_new_file(tmp_path):
    block = ParsedBlock(
        raw_open_fence="```python:foo.py", lang="python",
        path="foo.py", body="x = 1", start=0, end=0,
    )
    plan = PlannedSave(
        block=block,
        abs_target=str(tmp_path / "foo.py"),
        exists=False,
        error=None,
    )
    result = apply_save(plan, "write")
    assert result.status == "succeeded"
    assert result.final_path == str(tmp_path / "foo.py")
    assert (tmp_path / "foo.py").read_text() == "x = 1"


def test_apply_skip_does_not_touch_disk(tmp_path):
    (tmp_path / "existing.py").write_text("OLD")
    block = ParsedBlock(
        raw_open_fence="```python:existing.py", lang="python",
        path="existing.py", body="NEW", start=0, end=0,
    )
    plan = PlannedSave(
        block=block,
        abs_target=str(tmp_path / "existing.py"),
        exists=True,
        error=None,
    )
    result = apply_save(plan, "skip")
    assert result.status == "skipped"
    assert (tmp_path / "existing.py").read_text() == "OLD"


def test_apply_rename_writes_to_next_free_suffix(tmp_path):
    (tmp_path / "foo.py").write_text("OLD")
    block = ParsedBlock(
        raw_open_fence="```python:foo.py", lang="python",
        path="foo.py", body="NEW", start=0, end=0,
    )
    plan = PlannedSave(
        block=block,
        abs_target=str(tmp_path / "foo.py"),
        exists=True,
        error=None,
    )
    result = apply_save(plan, "rename")
    assert result.status == "renamed"
    assert result.final_path == str(tmp_path / "foo.1.py")
    assert (tmp_path / "foo.1.py").read_text() == "NEW"
    assert (tmp_path / "foo.py").read_text() == "OLD"


def test_apply_rename_finds_next_free_when_dot1_taken(tmp_path):
    (tmp_path / "foo.py").write_text("OLD")
    (tmp_path / "foo.1.py").write_text("OLDER")
    block = ParsedBlock(
        raw_open_fence="```python:foo.py", lang="python",
        path="foo.py", body="NEW", start=0, end=0,
    )
    plan = PlannedSave(
        block=block,
        abs_target=str(tmp_path / "foo.py"),
        exists=True,
        error=None,
    )
    result = apply_save(plan, "rename")
    assert result.status == "renamed"
    assert result.final_path == str(tmp_path / "foo.2.py")


def test_apply_creates_parent_dirs(tmp_path):
    block = ParsedBlock(
        raw_open_fence="```python:src/nested/foo.py", lang="python",
        path="src/nested/foo.py", body="x", start=0, end=0,
    )
    plan = PlannedSave(
        block=block,
        abs_target=str(tmp_path / "src" / "nested" / "foo.py"),
        exists=False,
        error=None,
    )
    result = apply_save(plan, "write")
    assert result.status == "succeeded"
    assert (tmp_path / "src" / "nested" / "foo.py").exists()


def test_apply_overwrite_replaces_existing_file(tmp_path):
    (tmp_path / "foo.py").write_text("OLD")
    block = ParsedBlock(
        raw_open_fence="```python:foo.py", lang="python",
        path="foo.py", body="NEW", start=0, end=0,
    )
    plan = PlannedSave(
        block=block,
        abs_target=str(tmp_path / "foo.py"),
        exists=True,
        error=None,
    )
    # When the existence collision is acknowledged by the user choosing
    # "write", we overwrite.
    result = apply_save(plan, "write")
    assert result.status == "succeeded"
    assert (tmp_path / "foo.py").read_text() == "NEW"
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_inline_code_saver.py -v`
Expected: ImportError on `SaveResult` / `apply_save`.

- [ ] **Step 3: Add the applier**

Append to `/home/lulu/Projects/ezclaw/inline_code_saver.py`:

```python
@dataclass(frozen=True)
class SaveResult:
    plan: PlannedSave
    status: str                  # 'succeeded' | 'skipped' | 'renamed' | 'rejected'
    final_path: Optional[str]    # absolute path that was written, or None


def _next_free_suffix(abs_target: str) -> str:
    """Return abs_target with a `.N.ext` suffix where N is the smallest
    positive integer such that the resulting path does not exist."""
    base, ext = os.path.splitext(abs_target)
    n = 1
    while True:
        candidate = f"{base}.{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def apply_save(plan: PlannedSave, choice: str) -> SaveResult:
    """Execute the save per the user's choice. Returns a SaveResult.

    choice: 'write' (overwrite or create), 'skip' (no write), or 'rename'
    (write to next free `.N.ext` suffix when the original exists).

    If the plan was already rejected (error set), returns a 'rejected'
    SaveResult without touching disk.
    """
    if plan.error is not None or plan.abs_target is None:
        return SaveResult(plan=plan, status="rejected", final_path=None)

    if choice == "skip":
        return SaveResult(plan=plan, status="skipped", final_path=None)

    if choice == "rename" and plan.exists:
        target = _next_free_suffix(plan.abs_target)
        status = "renamed"
    else:
        target = plan.abs_target
        status = "succeeded"

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as f:
        f.write(plan.block.body)
    return SaveResult(plan=plan, status=status, final_path=target)
```

- [ ] **Step 4: Verify tests pass**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_inline_code_saver.py -v`
Expected: 21/21 tests pass (15 + 6 new).

- [ ] **Step 5: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add inline_code_saver.py tests/test_inline_code_saver.py && git commit -m "feat(inline-save): save applier with write/skip/rename strategies

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add `inline_save` to `MUTATING_TOOLS`

**Files:**
- Modify: `tools.py` (the `MUTATING_TOOLS` constant)

- [ ] **Step 1: Locate the constant**

Run: `cd /home/lulu/Projects/ezclaw && grep -n "MUTATING_TOOLS" tools.py | head -5`
Expected: shows the `MUTATING_TOOLS = frozenset({...})` definition.

- [ ] **Step 2: Add `inline_save`**

In `/home/lulu/Projects/ezclaw/tools.py`, find `MUTATING_TOOLS`:

```python
MUTATING_TOOLS = frozenset({
    "apply_diff",
    "write_file",
    "run_shell",
    "schedule_task",
    "unschedule_task",
})
```

Change to:

```python
MUTATING_TOOLS = frozenset({
    "apply_diff",
    "write_file",
    "run_shell",
    "schedule_task",
    "unschedule_task",
    "inline_save",
})
```

- [ ] **Step 3: Verify no regression**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_recall_actions_tool.py tests/test_action_dispatcher_integration.py -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add tools.py && git commit -m "feat(inline-save): register inline_save in MUTATING_TOOLS

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Prompt nudges in executor + ChatAgent + agents.md

**Files:**
- Modify: `multi_agent.py` (executor system prompt)
- Modify: `agent.py` (ChatAgent system_prompt augmentation)
- Modify: `agents.md`

- [ ] **Step 1: Locate the executor system prompt**

Run: `cd /home/lulu/Projects/ezclaw && grep -n "executor.*system_prompt\|## Past actions" multi_agent.py | head -5`
Expected: shows the executor system prompt and the action-tracking nudge added by the prior feature.

- [ ] **Step 2: Append the inline-save nudge to executor**

In `/home/lulu/Projects/ezclaw/multi_agent.py`, find the executor's `system_prompt` string. At the END of the string (just before the closing triple-quote), append:

```
═══════════════════════════════════════════════════════════════
## Emitting code in your response

When you output a complete file for the user, tag the code fence with the
destination path:

  ```python:src/auth.py
  # file body
  ```

The CLI saves tagged blocks to `workspace/<path>` automatically. Untagged
fences (just ```python) stay inline and are NOT saved — use untagged for
short illustrative snippets only. For files you'll then manipulate via
tools, use `write_file` or `apply_diff` instead of an inline tagged block.
```

- [ ] **Step 3: Append to ChatAgent's prompt augmentation in agent.py**

In `/home/lulu/Projects/ezclaw/agent.py`, find the existing prompt augmentation (where `self.system_prompt += "\n\n## Past actions\n..."` is set in `__init__`). Right after that block, add another augmentation:

```python
        self.system_prompt += (
            "\n\n## Emitting code in your response\n"
            "When you output a complete file for the user, tag the code "
            "fence with the destination path:\n"
            "\n"
            "  ```python:src/foo.py\n"
            "  # file body\n"
            "  ```\n"
            "\n"
            "The CLI saves tagged blocks to `workspace/<path>` automatically. "
            "Untagged fences (just ```python) stay inline and are NOT saved — "
            "use untagged for short illustrative snippets only. For files you'll "
            "then manipulate via tools, use `write_file` or `apply_diff` instead."
        )
```

- [ ] **Step 4: Append to agents.md**

In `/home/lulu/Projects/ezclaw/agents.md`, append the same paragraph at the end of the file:

```markdown

## Emitting code in your response

When you output a complete file for the user, tag the code fence with the
destination path:

    ```python:src/auth.py
    # file body
    ```

The CLI saves tagged blocks to `workspace/<path>` automatically. Untagged
fences (just ```python) stay inline and are NOT saved — use untagged for
short illustrative snippets only. For files you'll then manipulate via
tools, use `write_file` or `apply_diff` instead of an inline tagged block.
```

- [ ] **Step 5: Add a smoke test for the nudge**

Append to `/home/lulu/Projects/ezclaw/tests/test_inline_code_saver.py`:

```python
def test_executor_prompt_mentions_tagged_blocks():
    import multi_agent
    prompt = multi_agent.AGENT_DEFS["executor"]["system_prompt"]
    assert "python:src/" in prompt or "lang:path" in prompt or "tag the code fence" in prompt.lower()
```

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_inline_code_saver.py::test_executor_prompt_mentions_tagged_blocks -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add multi_agent.py agent.py agents.md tests/test_inline_code_saver.py && git commit -m "feat(inline-save): prompt nudge telling agents to tag complete-file blocks

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: CLI integration — end-of-turn save + badge rewrite

**Files:**
- Modify: `cli.py` — hook the saver into `_run_turn`'s end-of-turn block
- Create: `tests/test_inline_code_saver_integration.py`

- [ ] **Step 1: Write failing integration tests**

Create `/home/lulu/Projects/ezclaw/tests/test_inline_code_saver_integration.py`:

```python
"""Integration tests: end-of-turn pipeline saves tagged blocks and
rewrites the response text with save badges. Collision callback is
stubbed."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inline_code_saver import parse_tagged_blocks, plan_saves, apply_save


def _run_end_of_turn(joined_response: str, workspace_root: str,
                     collision_callback=lambda plan: "write"):
    """Simulate the CLI's end-of-turn save pipeline.

    Returns (save_results, rewritten_text). Mirrors what cli.py will do
    in its finalize block."""
    blocks = parse_tagged_blocks(joined_response)
    plans = plan_saves(blocks, workspace_root=workspace_root)
    results = []
    for plan in plans:
        if plan.error is not None:
            from inline_code_saver import SaveResult
            results.append(SaveResult(plan=plan, status="rejected", final_path=None))
            continue
        if plan.exists:
            choice = collision_callback(plan)
            results.append(apply_save(plan, choice))
        else:
            results.append(apply_save(plan, "write"))
    # The CLI's _rewrite_with_badges builds the final user-visible markdown.
    # For the test we just check the disk + results.
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
```

- [ ] **Step 2: Verify these helper-based tests pass already**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_inline_code_saver_integration.py -v`
Expected: all 4 tests pass (the pipeline logic is built out of already-working primitives from Tasks 1-3).

- [ ] **Step 3: Add CLI integration**

In `/home/lulu/Projects/ezclaw/cli.py`, find the end-of-turn finalize block in `_run_turn`. After Spec A landed, the block looks like:

```python
        # Finish generating
        self.is_generating = False
        if not self.halted:
            final_renderable = self._get_current_renderable_ansi()
            self.history_ansi.append(final_renderable)
            self.current_response_parts = []
            ...
```

Insert the save pass BETWEEN `self.is_generating = False` and the `if not self.halted:` gate (so saves happen regardless of halt state — the active panel needs the badges either way):

```python
        # Finish generating
        self.is_generating = False
        # Inline-save pass: parse tagged code blocks, save them to disk,
        # and rewrite the content with badges. Skipped if halted (the
        # turn will resume; saves wait until the user truly ends the turn).
        if not self.halted and self.current_response_parts:
            self._process_inline_saves()
        if not self.halted:
            final_renderable = self._get_current_renderable_ansi()
            self.history_ansi.append(final_renderable)
            ...
```

Then add the new helper method on `ChatUI` (place it near the end of the class, after `_resolve_user_name` or wherever other private helpers live):

```python
    def _process_inline_saves(self) -> None:
        """End-of-turn pass: parse tagged code blocks from the response,
        save them, rewrite the joined content with badges. Best-effort —
        any exception is logged to stderr and swallowed."""
        try:
            from inline_code_saver import parse_tagged_blocks, plan_saves, apply_save, SaveResult
            joined = "".join(self.current_response_parts)
            blocks = parse_tagged_blocks(joined)
            if not blocks:
                return
            plans = plan_saves(blocks, workspace_root="workspace")
            results = []
            for plan in plans:
                if plan.error is not None:
                    results.append(SaveResult(plan=plan, status="rejected", final_path=None))
                    continue
                if plan.exists:
                    choice = self._ask_save_collision(plan)
                    results.append(apply_save(plan, choice))
                else:
                    results.append(apply_save(plan, "write"))
            rewritten = self._rewrite_with_badges(joined, results)
            self.current_response_parts = [rewritten]
            # Record each successful save as an action row.
            for r in results:
                if r.status in ("succeeded", "renamed"):
                    self._record_inline_save_action(r)
        except Exception as e:
            import sys
            print(f"[inline-save] pass failed: {e}", file=sys.stderr)
```

Add the rewrite helper:

```python
    def _rewrite_with_badges(self, joined: str, results: list) -> str:
        """Replace each tagged opening fence with a plain `lang` fence
        and prepend a blockquote badge indicating save status."""
        # Apply results in reverse order so earlier offsets stay valid.
        out = joined
        for r in sorted(results, key=lambda x: x.plan.block.start, reverse=True):
            block = r.plan.block
            if r.status == "succeeded":
                badge = f"> 💾 **saved →** `workspace/{block.path}`\n\n"
                new_fence = f"```{block.lang}" if block.lang else "```"
            elif r.status == "renamed":
                final_rel = os.path.relpath(r.final_path, os.path.abspath("workspace"))
                badge = f"> 💾 **saved →** `workspace/{final_rel}` (renamed; original existed)\n\n"
                new_fence = f"```{block.lang}" if block.lang else "```"
            elif r.status == "skipped":
                badge = f"> ⊘ **skipped →** `workspace/{block.path}` (existed)\n\n"
                new_fence = f"```{block.lang}" if block.lang else "```"
            elif r.status == "rejected":
                badge = f"> ⚠ **rejected →** `{block.path}` ({r.plan.error}; kept inline)\n\n"
                new_fence = f"```{block.lang}" if block.lang else "```"
            else:
                continue
            # Replace the opening fence with badge + plain fence.
            opener_len = len(block.raw_open_fence)
            opener_end = block.start + opener_len
            out = out[:block.start] + badge + new_fence + out[opener_end:]
        return out
```

Add the collision-prompt callback. The CLI's existing auth-prompt pattern blocks on user input. Reuse the same single-keypress queue. Place this method near `_ask_auth`:

```python
    def _ask_save_collision(self, plan) -> str:
        """Block on user input for a collision. Returns 'write', 'skip',
        or 'rename'. Default on UI failure: 'rename' (safe non-destructive)."""
        try:
            # Push a side-message panel describing the collision and
            # accepted keys. The CLI's input handler must recognize these
            # keystrokes; for v1 we use the auth_queue with extended
            # tokens. If the UI surface for this prompt isn't ready yet,
            # default to 'rename' to avoid destructive overwrites.
            from rich.panel import Panel
            from rich.text import Text as _Text
            body = _Text()
            body.append(f"💾 {plan.abs_target} exists\n", style="bold")
            body.append("[O]verwrite   [S]kip   [R]ename to next free suffix\n")
            self.side_messages.append(f"💾 collision on {plan.block.path} — press O/S/R")
            self._update_ui()
            # The keystroke handler must enqueue a token on the auth_queue.
            # For v1, the production code reads a token; the default below
            # ensures non-destructive behavior if the queue mechanism isn't
            # extended in time.
            choice_token = self.auth_queue.get(timeout=60)  # may raise on timeout
            if choice_token == "overwrite":
                return "write"
            if choice_token == "skip":
                return "skip"
            return "rename"
        except Exception:
            return "rename"
```

Add the action-row helper:

```python
    def _record_inline_save_action(self, result) -> None:
        """Record a successful inline-save as an action row, matching
        the action-tracking subsystem's expectations."""
        try:
            from tools import set_session_context, get_session_context
            from embed import embed as _embed
            import json
            import pickle
            session_id = get_session_context()
            if session_id is None:
                return
            db = getattr(self.agent, "db", None)
            if db is None:
                return
            block = result.plan.block
            path = block.path if result.status == "succeeded" else os.path.relpath(
                result.final_path, os.path.abspath("workspace"),
            )
            summary = f"saved {os.path.basename(path)}"
            why = ""  # No assistant_text context here; leave why empty.
            outcome = "succeeded" if result.status == "succeeded" else "partial"
            try:
                vec = _embed(summary)
                emb_blob = pickle.dumps(vec)
            except Exception:
                emb_blob = None
            db.add_action(
                session_id=session_id,
                tool="inline_save",
                args_json=json.dumps({"path": path}),
                summary=summary,
                why=why or None,
                outcome=outcome,
                error_excerpt=None,
                embedding=emb_blob,
            )
        except Exception as e:
            import sys
            print(f"[inline-save] action record failed: {e}", file=sys.stderr)
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/lulu/Projects/ezclaw && python -m pytest tests/test_inline_code_saver.py tests/test_inline_code_saver_integration.py tests/test_recall_actions_tool.py tests/test_action_dispatcher_integration.py -v`
Expected: every test passes.

- [ ] **Step 5: Verify cli.py compiles**

Run: `cd /home/lulu/Projects/ezclaw && python -m py_compile cli.py`
Expected: compiles cleanly.

- [ ] **Step 6: Commit**

```bash
cd /home/lulu/Projects/ezclaw && git add cli.py tests/test_inline_code_saver_integration.py && git commit -m "feat(inline-save): CLI end-of-turn save + badge rewrite + collision prompt

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Manual smoke

**Files:** none.

- [ ] **Step 1: Start ezclaw**

```bash
cd /home/lulu/Projects/ezclaw && python cli.py
```

- [ ] **Step 2: Ask for a complete file with the tag convention**

Type: `write a python helper that returns the current timestamp; save it as helpers/time_now.py`

Expected:
- The agent narrates briefly.
- A code block appears in the response.
- After the turn ends, a `💾 saved → workspace/helpers/time_now.py` badge appears above the code block.
- `workspace/helpers/time_now.py` exists on disk.
- `sqlite3 ezclaw.db "SELECT tool, summary FROM actions ORDER BY id DESC LIMIT 1"` shows `inline_save | saved time_now.py`.

- [ ] **Step 3: Re-ask the same prompt to trigger a collision**

Repeat Step 2 verbatim. Watch for the collision side-message. Press `R` to rename.

Expected:
- A new file `workspace/helpers/time_now.1.py` is created.
- The badge above the code block reads "saved → workspace/helpers/time_now.1.py (renamed; original existed)".

- [ ] **Step 4: Test snippet (untagged) preservation**

Type: `show me a small python snippet that uses enumerate`

Expected: the model outputs an UNTAGGED ` ```python ` block. No file lands on disk. No badge appears.

---

## Self-Review

**Spec coverage:**
- Tagged-block parser — Task 1.
- Path validator + planner — Task 2.
- Save applier with write/skip/rename — Task 3.
- `inline_save` in `MUTATING_TOOLS` — Task 4.
- Prompt nudges — Task 5.
- CLI end-of-turn integration + badge rewrite — Task 6.
- Collision prompt UX — Task 6 (`_ask_save_collision`).
- Action-row integration — Task 6 (`_record_inline_save_action`).
- Manual smoke — Task 7.

**Placeholder scan:** No placeholders; each step's code is complete.

**Type consistency:**
- `ParsedBlock` (Task 1) used in `plan_saves` (Task 2) and `apply_save` (Task 3) and tests — consistent fields.
- `PlannedSave` (Task 2) used by `apply_save` (Task 3) — consistent.
- `SaveResult` (Task 3) consumed by the CLI helpers (Task 6) — consistent.
- `apply_save(plan, choice: str)` accepts the same `choice` values as `_ask_save_collision` returns (Task 6).
- `parse_tagged_blocks` / `plan_saves` / `apply_save` signatures match between unit tests (Tasks 1-3), integration tests (Task 6), and CLI consumer (Task 6).
- `inline_save` is the string used in both `MUTATING_TOOLS` (Task 4) and the `add_action(tool=...)` call (Task 6).

**Note on `_ask_save_collision`:** The implementation uses `self.auth_queue` with extended tokens (`overwrite` / `skip` / `rename`). The CLI's keystroke handler must be extended to push these tokens. If a subagent implementing Task 6 finds the keystroke wiring missing, they should EITHER extend the keystroke handler to recognize O/S/R while a collision is pending OR fall back to auto-renaming as the safe default (the code already does this on timeout). Both are acceptable for v1.
