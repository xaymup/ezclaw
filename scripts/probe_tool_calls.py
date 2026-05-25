"""Tool-call wire-format probe.

The qwen2.5-coder:14b lesson: some local models emit tool calls as
JSON inside `message.content` instead of populating Ollama's structured
`message.tool_calls` field. ezclaw's executor loop only iterates the
structured field, so files never get written and tools never run —
the model can claim "Done." while having done nothing.

This probe catches that failure mode in ~30s before we spend 10+ min
on a full audit. It hands the model ONE tool (`add_numbers`) and an
unambiguous request to call it, then streams the response and looks
for the tool call in BOTH places (the structured field — what we want
— and embedded JSON in content — the failure mode). Reports the
verdict in plain text so a single human glance answers "is this model
runtime-compatible with ezclaw at all?".

Usage:
  python scripts/probe_tool_calls.py qwen3:14b
  python scripts/probe_tool_calls.py qwen3:14b gpt-oss:20b phi4-reasoning:plus
  python scripts/probe_tool_calls.py --timeout 90 llama3.1:8b

Exit code 0 if every model passed, 1 if any failed (so it composes
into shell pipelines: `probe X Y Z && audit ...`).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from typing import Tuple

# Make the repo importable.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import ollama  # noqa: E402


TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two integers and return their sum. Use this for any arithmetic.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "first number"},
                "b": {"type": "integer", "description": "second number"},
            },
            "required": ["a", "b"],
        },
    },
}

TEST_PROMPT = "Use the add_numbers tool to compute 47 + 53. Call the tool — do not just write the answer."

# Match a JSON tool-call written into chat content (the qwen2.5-coder
# failure mode). Captures either {"name": "add_numbers", ...} or the
# OpenAI-style {"tool_calls": [...]} envelope.
_EMBEDDED_TOOLCALL_RE = re.compile(
    r'\{\s*"(?:name|tool_calls|function)"\s*:',
    re.MULTILINE,
)


def probe(model: str, timeout: float = 60.0) -> Tuple[bool, str]:
    """Run the probe against `model`. Returns (passed, one_line_reason)."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    client = ollama.Client(host=base_url)
    started = time.time()

    structured_calls = []
    content_buf = ""
    chunk_count = 0
    finish_reason = None

    try:
        stream = client.chat(
            model=model,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            tools=[TEST_TOOL],
            stream=True,
            options={"temperature": 0.0, "num_ctx": 4096},
            keep_alive="5m",
        )
        for chunk in stream:
            if time.time() - started > timeout:
                return False, f"timed out after {timeout:.0f}s (got {chunk_count} chunks)"
            chunk_count += 1
            msg = chunk.get("message") or {}
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = (tc.get("function") or {})
                    structured_calls.append({
                        "name": fn.get("name"),
                        "args": fn.get("arguments"),
                    })
            if msg.get("content"):
                content_buf += msg["content"]
            if chunk.get("done"):
                finish_reason = chunk.get("done_reason") or "done"
                break
    except Exception as e:
        return False, f"client error: {type(e).__name__}: {e}"

    elapsed = time.time() - started

    # Verdict order: structured tool_calls is the WIN. Embedded JSON in
    # content means the model emitted intent but in the wrong wire format
    # — flag it loudly so we don't burn an audit on it.
    if structured_calls:
        names = [c["name"] for c in structured_calls]
        if "add_numbers" in names:
            return True, f"PASS — structured tool_calls fired (`{names[0]}`) in {elapsed:.1f}s"
        return False, f"structured tool_calls fired but with wrong name(s): {names}"

    if _EMBEDDED_TOOLCALL_RE.search(content_buf):
        snippet = _EMBEDDED_TOOLCALL_RE.search(content_buf).group(0)
        return False, (
            f"BROKEN — model emitted tool call as JSON in content (`{snippet}`) "
            f"instead of structured tool_calls. Won't work with ezclaw."
        )

    if not content_buf.strip():
        return False, f"no output at all in {elapsed:.1f}s — model may not be loaded"

    return False, (
        f"no tool call attempted — model ignored the tool. "
        f"Content was: {content_buf[:150]!r}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="+", help="model tags to probe (e.g. qwen3:14b)")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-model wall cap (s)")
    args = ap.parse_args()

    print(f"Probing {len(args.models)} model(s) with timeout {args.timeout:.0f}s each.\n", flush=True)
    width = max(len(m) for m in args.models)
    failures = 0
    for m in args.models:
        print(f"  {m:<{width}}  ", end="", flush=True)
        ok, reason = probe(m, timeout=args.timeout)
        mark = "✅" if ok else "❌"
        print(f"{mark}  {reason}", flush=True)
        if not ok:
            failures += 1

    print(f"\n{len(args.models) - failures}/{len(args.models)} compatible.", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
