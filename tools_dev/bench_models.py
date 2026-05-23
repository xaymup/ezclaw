"""Benchmark candidate models for each agent role.

For each role, runs the actual production system prompt + a
representative task against multiple candidate models. Measures
latency and checks the output for the shape we need for that role:

  - executor: did it emit a tool_call for write_file? (tools schema)
  - architect: did it return valid JSON matching the execute() shape?
  - debugger: did it identify the root cause and propose a fix?
  - general / researcher: did it stay on-topic and produce a usable answer?

Run:
  python tools_dev/bench_models.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama

OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TIMEOUT = 90

# Minimal write_file tool schema for the executor tests.
WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path inside workspace/"},
                "content": {"type": "string", "description": "File body to write"},
            },
            "required": ["path", "content"],
        },
    },
}

EXECUTOR_SYS = (
    "You are EzClaw's Executor. When the task is to create or modify a file, "
    "you MUST call the write_file tool — generating code in your response "
    "without the tool call means the file is never written. Be concise."
)

ARCHITECT_SYS = (
    "You are EzClaw's Architect. Classify and route requests. Return ONLY "
    "a JSON object of the form:\n"
    "{\"kind\": \"plan\", \"title\": \"<short>\", \"tasks\": [{\"id\": <int>, \"description\": \"<short>\"}]}\n"
    "or {\"kind\": \"single\"} for trivial chat. No prose, no <think>, no markdown fences."
)

DEBUGGER_SYS = (
    "You are EzClaw's Debugger. Given a Python error trace, identify the "
    "ROOT CAUSE (not just the symptom) and propose a concrete fix. Return:\n"
    "1. Root cause: <one sentence>\n"
    "2. Proposed fix:\n"
    "   - <step>\n"
)

GENERAL_SYS = (
    "You are EzClaw's General Assistant — friendly, concise, on-topic. "
    "Answer directly without preamble."
)


def _chat(client, model, system, user, **opts):
    """Single-turn chat. Returns (latency_ms, message_dict)."""
    t0 = time.time()
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
        options={"temperature": 0.0, "num_ctx": 8192, **opts},
        **({"tools": opts.pop("tools")} if "tools" in opts else {}),
    )
    return (time.time() - t0) * 1000, resp.get("message", {})


def _chat_with_tools(client, model, system, user, tools):
    t0 = time.time()
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=tools,
        stream=False,
        options={"temperature": 0.0, "num_ctx": 8192},
    )
    return (time.time() - t0) * 1000, resp.get("message", {})


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def section(title):
    print(f"\n{'═' * 70}")
    print(f" {title}")
    print(f"{'═' * 70}")


def test_executor(client, models):
    section("EXECUTOR · 'Write a function double(x) and save it to math_utils.py'")
    user = (
        "Create a file called math_utils.py with a single function "
        "double(x) that returns x * 2. Use the write_file tool."
    )
    for m in models:
        try:
            ms, msg = _chat_with_tools(
                client, m, EXECUTOR_SYS, user, tools=[WRITE_FILE_TOOL]
            )
            tool_calls = getattr(msg, "tool_calls", None) or msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()
            print(f"\n  ── {m}  ({ms:.0f}ms)")
            if tool_calls:
                tc = tool_calls[0]
                name = tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name")
                args = tc.function.arguments if hasattr(tc, "function") else tc.get("function", {}).get("arguments")
                args_str = json.dumps(args)[:140] if args else "{}"
                print(f"     ✓ tool_call:  {name}({args_str})")
            else:
                print(f"     ✗ NO tool_call")
                print(f"        content excerpt: {content[:140]}")
        except Exception as e:
            print(f"  ── {m}  ERROR: {type(e).__name__}: {e}")


def test_architect(client, models):
    section("ARCHITECT · plan parsing → 'Add a function double(x) to utils.py and write a test'")
    user = (
        "PLAN REQUEST:\nAdd a function double(x) to utils.py and write a "
        "test for it. Return the plan JSON."
    )
    for m in models:
        try:
            ms, msg = _chat(client, m, ARCHITECT_SYS, user, format="json")
        except Exception:
            try:
                ms, msg = _chat(client, m, ARCHITECT_SYS, user)
            except Exception as e:
                print(f"\n  ── {m}  ERROR: {type(e).__name__}: {e}")
                continue
        content = _strip_think(msg.get("content") or "")
        print(f"\n  ── {m}  ({ms:.0f}ms)")
        # Try to parse JSON
        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(json_match.group(0)) if json_match else {}
            kind = data.get("kind")
            tasks = data.get("tasks") or []
            print(f"     ✓ kind={kind}, {len(tasks)} task(s)")
            for t in tasks[:3]:
                print(f"        - {t.get('description', '?')[:70]}")
        except Exception as e:
            print(f"     ✗ invalid JSON: {e}")
            print(f"        excerpt: {content[:200]}")


def test_debugger(client, models):
    section("DEBUGGER · 'AttributeError: NoneType has no attribute foo'")
    user = (
        "Diagnose this Python error.\n\n"
        "  Traceback (most recent call last):\n"
        "    File \"app.py\", line 42, in <module>\n"
        "      result = config.foo\n"
        "  AttributeError: 'NoneType' object has no attribute 'foo'\n\n"
        "context: `config = load_config(path)` was called above the failing line.\n"
        "load_config returns None when the file does not exist."
    )
    for m in models:
        try:
            ms, msg = _chat(client, m, DEBUGGER_SYS, user)
            content = _strip_think(msg.get("content") or "")
            print(f"\n  ── {m}  ({ms:.0f}ms)")
            # Heuristic check: response should mention the root cause
            ok_root = (
                "load_config" in content
                or "returns None" in content
                or "None" in content and "file" in content
            )
            ok_fix = (
                "check" in content.lower()
                or "exists" in content.lower()
                or "raise" in content.lower()
                or "default" in content.lower()
            )
            mark_root = "✓" if ok_root else "✗"
            mark_fix = "✓" if ok_fix else "✗"
            print(f"     {mark_root} root cause identified  |  {mark_fix} fix proposed")
            print(f"     excerpt: {content[:200]}")
        except Exception as e:
            print(f"  ── {m}  ERROR: {type(e).__name__}: {e}")


def test_general(client, models):
    section("GENERAL · 'how do I start a daily walking habit?'")
    user = "I want to start a daily walking habit. Give me 3 concrete tips."
    for m in models:
        try:
            ms, msg = _chat(client, m, GENERAL_SYS, user)
            content = _strip_think(msg.get("content") or "")
            print(f"\n  ── {m}  ({ms:.0f}ms)")
            on_topic = "walk" in content.lower() or "step" in content.lower() or "habit" in content.lower()
            length_ok = 80 < len(content) < 1200
            mark_topic = "✓" if on_topic else "✗"
            mark_len = "✓" if length_ok else "✗"
            print(f"     {mark_topic} on-topic  |  {mark_len} concise (got {len(content)} chars)")
            print(f"     excerpt: {content[:240]}")
        except Exception as e:
            print(f"  ── {m}  ERROR: {type(e).__name__}: {e}")


def main():
    client = ollama.Client(host=OLLAMA_HOST, timeout=TIMEOUT)

    # Models pulled locally — see `ollama list`
    EXECUTOR_CANDIDATES   = ["qwen3:14b", "qwen2.5-coder:14b"]
    ARCHITECT_CANDIDATES  = ["deepseek-r1:14b", "phi4-reasoning:plus"]
    DEBUGGER_CANDIDATES   = ["deepseek-r1:14b", "phi4-reasoning:plus"]
    GENERAL_CANDIDATES    = ["qwen3.5:9b"]   # baseline — fast is the only constraint

    test_executor(client, EXECUTOR_CANDIDATES)
    test_architect(client, ARCHITECT_CANDIDATES)
    test_debugger(client, DEBUGGER_CANDIDATES)
    test_general(client, GENERAL_CANDIDATES)


if __name__ == "__main__":
    main()
