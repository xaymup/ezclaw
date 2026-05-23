"""Pure helpers for the action-tracking subsystem.

Kept dependency-free so they can be unit-tested without touching sqlite,
ollama, or any agent state.
"""

import os
from typing import Any, Dict, Optional, Tuple


def classify_outcome(result: str) -> Tuple[str, Optional[str]]:
    """Inspect a tool's stringified result and return (outcome, error_excerpt).

    Outcome is one of: 'succeeded', 'failed', 'partial'.
    error_excerpt is the first 300 chars of `result` when outcome != 'succeeded',
    else None.
    """
    if not result:
        return "succeeded", None
    head = result.lstrip()
    failed_markers = ("Error:", "Traceback", "Exception")
    if any(head.startswith(m) or m in head[:500] for m in failed_markers):
        return "failed", result[:300]
    partial_markers = ("exit code 1", "exit code 2", "non-zero exit", "exit status 1")
    if any(m in result for m in partial_markers):
        return "partial", result[:300]
    return "succeeded", None


def extract_why(assistant_text: str) -> Optional[str]:
    """Return the last non-empty sentence of `assistant_text`, trimmed to 200 chars.

    LLMs reliably narrate intent before invoking a tool — this captures that
    intent. Returns None if there is no usable text.
    """
    if not assistant_text or not assistant_text.strip():
        return None
    text = assistant_text.strip()
    parts = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".!?":
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    if not parts:
        return None
    last = parts[-1]
    return last[:200]


def summarize_action(tool_name: str, args: Dict[str, Any]) -> str:
    """One-line human description of a tool call, suitable for embedding."""
    if tool_name == "apply_diff":
        path = args.get("path", "?")
        return f"edited {os.path.basename(path)}"
    if tool_name == "write_file":
        path = args.get("path", "?")
        return f"wrote {os.path.basename(path)}"
    if tool_name == "run_shell":
        cmd = args.get("command", "")
        return f"ran: {cmd[:60]}"
    if tool_name == "schedule_task":
        desc = args.get("description", "")
        return f"scheduled: {desc[:60]}"
    if tool_name == "unschedule_task":
        return f"unscheduled task {args.get('task_id', '?')}"
    return f"called {tool_name}"
