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
    start: int
    end: int


_FENCE_RE = re.compile(
    r"```([^\n`]*)\n(.*?)\n```",
    re.DOTALL,
)


def parse_tagged_blocks(text: str) -> List[ParsedBlock]:
    """Return every fenced code block whose info string contains a colon
    after the language identifier. Untagged blocks are not returned.

    The info string is the part between the opening ``` and the first
    newline. We treat the substring before the first `:` as `lang` and
    everything after the first `:` (trimmed) as `path`.
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


@dataclass(frozen=True)
class PlannedSave:
    block: ParsedBlock
    abs_target: Optional[str]
    exists: bool
    error: Optional[str]


def _validate_path(rel_path: str, workspace_root: str) -> Optional[str]:
    """Return the absolute target path if valid; None if not."""
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
