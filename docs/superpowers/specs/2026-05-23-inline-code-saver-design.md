# Inline Code Auto-Save — Design

**Date:** 2026-05-23
**Status:** Approved, pending implementation plan.

## Problem

When the agent inlines code in a chat response, the user has to manually copy-paste it into a file. The `write_file` / `apply_diff` tools exist but the prompt only encourages them — there's no rule against inline. As a result, complete files end up trapped in transcripts. The user wants any complete-file code emitted in a response to land on disk automatically; only small illustrative snippets should stay inline.

## Goal

When the assistant emits a fenced code block tagged with a destination path, the CLI saves the block's body to `workspace/<path>` at end-of-turn, displays a badge above the rendered block indicating the save destination, and falls back to a chat-driven prompt on collisions. Untagged blocks stay inline as today.

## Non-goals

- Editing tagged blocks across multiple turns. Each turn is independent; re-emitting the same path triggers the collision flow.
- Diff display when overwriting (user runs `git diff` after).
- Marker variations beyond `lang:path` (no `lang title=...`).
- Streaming-time saves. Saves happen once per turn at finalize.

## Convention

The model annotates a code fence with the destination path after the language identifier, separated by a colon:

````
```python:src/auth.py
# full file content here
```
````

A bare fence (`` ```python ``) or any fence whose info string lacks a colon is a snippet — stays inline, no save.

Strictness:
- Exactly one colon separates lang from path.
- Path must be relative, no leading `/`, no `..` segments.
- Lang may be empty (`` ```:scripts/run.sh `` is allowed; rich renders without highlighting).

Invalid tags emit a `⚠ rejected` badge and the block stays inline as a snippet.

## Architecture

A new pure module `inline_code_saver.py` at project root performs parsing, path validation, and save planning. `cli.py` integrates at the end-of-turn finalize point. No agent-side or DB-schema changes beyond a re-used action row.

### Module: `inline_code_saver.py`

```python
@dataclass(frozen=True)
class ParsedBlock:
    raw_open_fence: str       # e.g. "```python:src/auth.py"
    lang: str                  # e.g. "python"
    path: str                  # e.g. "src/auth.py"
    body: str                  # block content, no trailing newline
    start: int                 # offset in source text where fence starts
    end: int                   # offset where closing fence ends


@dataclass(frozen=True)
class PlannedSave:
    block: ParsedBlock
    abs_target: str            # absolute path under workspace_root
    exists: bool               # whether the target file already exists
    error: Optional[str]       # set if path validation rejected; abs_target is None
```

Functions:

```python
def parse_tagged_blocks(text: str) -> List[ParsedBlock]:
    """Find every fenced block whose info string contains a colon.
    Untagged blocks are not returned."""

def plan_saves(blocks: List[ParsedBlock], workspace_root: str) -> List[PlannedSave]:
    """Validate paths and probe disk. Returns one PlannedSave per block,
    each marked either OK-to-write, collision, or rejected."""

def apply_save(plan: PlannedSave, choice: str) -> SaveResult:
    """choice in {'write','skip','rename'}. Writes the body to disk per
    the chosen strategy. Returns final path + status."""
```

`apply_save` is the only function that touches disk.

### CLI integration

The end-of-turn finalize block (currently `cli.py:2094-2102`) gains a save pass before the existing history-append:

```python
if not self.halted:                  # from Spec A
    joined = "".join(self.current_response_parts)
    blocks = parse_tagged_blocks(joined)
    plans = plan_saves(blocks, workspace_root="workspace")

    # Pending collisions surface as chat prompts. Sequential, blocking.
    save_results = []
    for plan in plans:
        if plan.error:
            save_results.append(SaveResult(plan=plan, status="rejected", final_path=None))
            continue
        if plan.exists:
            choice = self._ask_save_collision(plan)        # new method
            save_results.append(apply_save(plan, choice))
        else:
            save_results.append(apply_save(plan, "write"))

    # Rewrite the joined content so rich renders cleanly and the badge appears.
    rendered_content = _rewrite_with_badges(joined, save_results)
    self.current_response_parts = [rendered_content]

    # Existing finalize: snapshot active area to history, clear state.
    final_renderable = self._get_current_renderable_ansi()
    self.history_ansi.append(final_renderable)
    ...
```

`_ask_save_collision(plan)` shows a small Panel in the active area:

```
┌──────────────────────────────────────────────┐
│  💾  workspace/src/auth.py exists            │
│  [O]verwrite   [S]kip   [R]ename to .1.py    │
└──────────────────────────────────────────────┘
```

Reuses the existing single-keypress capture mechanism that the auth panel uses (`cli.py:596`). Returns `"write"`, `"skip"`, or `"rename"`.

`_rewrite_with_badges(joined, save_results)` does a regex-based rewrite per block:

- For successful writes: replace `` ```python:src/auth.py `` (just the opener) with `> 💾 **saved →** `workspace/src/auth.py`\n` followed by `` ```python ``.
- For skipped: `> ⊘ **skipped →** ... (existed)`.
- For rejected: `> ⚠ **rejected →** ... (kept inline as snippet)`. In this case, also strip the `:path` part of the opener so it stays a valid lang tag.

The blockquote-then-fence pattern renders correctly in `rich.markdown.Markdown`.

### Prompt nudge

Append to the executor system prompt in `multi_agent.py` (placed after the existing "## Past actions" block from the action-tracking spec):

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

Add the same paragraph to `agents.md` (the ChatAgent's prompt source) and to the ChatAgent prompt augmentation in `agent.py:__init__` so single-agent mode honours it too.

### Action-table integration

Each successful save records a row in the `actions` table (from the previous spec):

```python
self.db.add_action(
    session_id=self.session_id,
    tool="inline_save",
    args_json=json.dumps({"path": final_path}),
    summary=f"saved {os.path.basename(final_path)}",
    why=extract_why(joined),
    outcome="succeeded",  # or "partial" for skipped/renamed
    error_excerpt=None,
    embedding=...,
)
```

`MUTATING_TOOLS` in `tools.py` gains the string `"inline_save"` so future audit queries surface these alongside `write_file` and `apply_diff` results.

## Testing

### Unit — `tests/test_inline_code_saver.py`

- `parse_tagged_blocks("...just text...")` → `[]`.
- `parse_tagged_blocks(text_with_one_tagged_block)` → length 1, correct lang/path/body.
- `parse_tagged_blocks(text_with_untagged)` → `[]` (untagged blocks ignored).
- `parse_tagged_blocks(text_with_three_blocks_mixed)` → length 1 (only the tagged one).
- Tolerates trailing whitespace on the opener (`` ```python:src/auth.py   ``).
- Captures the closing fence correctly even when body contains backticks.
- `plan_saves` rejects absolute paths (`/etc/passwd`), `..` traversal (`../parent.py`), empty path after colon. Each yields `error` set.
- `plan_saves` flags `exists=True` when a file is already on disk inside the temp workspace.
- `apply_save` with `choice="write"` writes the body verbatim and returns the canonical path.
- `apply_save` with `choice="skip"` returns `SaveResult(status="skipped")` and does NOT touch disk.
- `apply_save` with `choice="rename"` produces `.1.ext`; if that exists, `.2.ext`; etc.

### Integration — `tests/test_inline_code_saver_integration.py`

Stub the CLI's collision-prompt callback to return a canned choice.

- Pipeline: text with two tagged blocks → both written to expected workspace paths → results contain two `"succeeded"` rows.
- Pipeline: first block writes new file; second block targets the same path → collision callback fires once → `"rename"` chosen → both files exist (`auth.py`, `auth.1.py`).
- Pipeline: rejected absolute path → no file written → status `"rejected"`, error message present.
- Each successful save inserts one row in the `actions` table with `tool="inline_save"`.

### Manual UX

Run ezclaw, ask the agent for a small Python module. Verify the badge appears, the file lands, the chat shows the code, action-table query returns the new row.

## Risks & mitigations

- **Model fails to tag.** Mitigation: prompt nudge is explicit. If the user notices untagged complete-file output, they can ask the agent to redo. Long-term: a heuristic that flags suspiciously-large untagged blocks as a `[did you mean to tag this?]` warning. Out of scope for v1.
- **Path validation gaps.** Reuse `tools.get_workspace_path()` (already exists, hardened). Don't reinvent.
- **Badge text causes Markdown rendering glitches.** The blockquote `> ` plus the existing code fence pattern is well-tested in rich; backtick path-names render as inline code.
- **Streaming partial blocks.** Saves happen at end-of-turn, so a streamed block that's not yet closed when generation ends is simply not parsed (parser only returns closed blocks). No partial-write risk.
- **Collision UX blocks the agent.** Saves happen AFTER generation finishes, so blocking on user input doesn't delay the model. The active panel just doesn't finalize to history until all collisions are resolved.

## Out of scope (explicit deferrals)

- Streaming-time partial-block detection.
- Untagged-but-suspiciously-large block warnings.
- Diff display on overwrite.
- Multi-marker syntax (`lang title=foo path=...`).
- Per-block metadata in the `actions` row (e.g., line counts, language).
