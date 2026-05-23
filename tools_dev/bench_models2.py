"""Round 2 of the model benchmark — fills in gaps from round 1:

- Researcher: test that qwen3.5:9b can produce a usable summary of
  a researched topic.
- General vs qwen3:14b: is the smaller model really enough, or are we
  losing quality on more involved conversational queries?
- Tool filtering: verify the production tool-selection path (the
  embedding-similarity top-N filter) doesn't accidentally drop tools
  the executor needs for a typical request.
- Conversational shortcut: verify routing examples actually map the
  intended queries to the intended agents.
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

RESEARCHER_SYS = (
    "You are EzClaw's Researcher. Synthesize info crisply. Cite sources "
    "when given to you. Keep responses tight (≤300 words)."
)

GENERAL_SYS = (
    "You are EzClaw's General Assistant — friendly, concise, on-topic."
)


def _chat(client, model, system, user):
    t0 = time.time()
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
        options={"temperature": 0.0, "num_ctx": 8192},
    )
    return (time.time() - t0) * 1000, resp.get("message", {})


def _strip(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def section(title):
    print(f"\n{'═' * 70}")
    print(f" {title}")
    print(f"{'═' * 70}")


def test_researcher(client, models):
    section("RESEARCHER · 'summarize what asyncio's event loop does'")
    user = "Summarize what asyncio's event loop does in Python. ≤200 words."
    for m in models:
        try:
            ms, msg = _chat(client, m, RESEARCHER_SYS, user)
            content = _strip(msg.get("content") or "")
            on_topic = "event loop" in content.lower() and "asyncio" in content.lower()
            mentions_coro = "coroutine" in content.lower() or "await" in content.lower()
            words = len(content.split())
            length_ok = 30 < words <= 300
            print(f"\n  ── {m}  ({ms:.0f}ms)")
            print(f"     {'✓' if on_topic else '✗'} on-topic  "
                  f"|  {'✓' if mentions_coro else '✗'} mentions coroutines/await  "
                  f"|  {'✓' if length_ok else '✗'} word count ({words})")
            print(f"     excerpt: {content[:250]}")
        except Exception as e:
            print(f"  ── {m}  ERROR: {type(e).__name__}: {e}")


def test_general_size(client, models):
    section("GENERAL — is qwen3.5:9b enough or do we need a 14b?")
    user = (
        "Help me think through whether to learn Rust or Go next. "
        "I'm a Python dev. ≤200 words."
    )
    for m in models:
        try:
            ms, msg = _chat(client, m, GENERAL_SYS, user)
            content = _strip(msg.get("content") or "")
            mentions_both = (
                "rust" in content.lower() and "go" in content.lower()
            )
            mentions_python = "python" in content.lower()
            words = len(content.split())
            length_ok = 50 < words <= 250
            print(f"\n  ── {m}  ({ms:.0f}ms)")
            print(f"     {'✓' if mentions_both else '✗'} compares both  "
                  f"|  {'✓' if mentions_python else '✗'} considers python background  "
                  f"|  {'✓' if length_ok else '✗'} word count ({words})")
            print(f"     excerpt: {content[:250]}")
        except Exception as e:
            print(f"  ── {m}  ERROR: {type(e).__name__}: {e}")


def test_routing_classifier():
    """Smoke-test the conversational shortcut: route each query and
    check it lands on the expected agent."""
    section("ROUTING · _short_circuit_classify on representative prompts")
    from unittest.mock import patch, MagicMock
    with patch("multi_agent.build_agent_client", return_value=(MagicMock(), "m")):
        with patch("multi_agent.build_architect_client", return_value=(MagicMock(), "m")):
            with patch.object(__import__("multi_agent").SpecializedAgent, "_pre_embed_tools", lambda self: None):
                with patch("multi_agent.load_skills", return_value=[]):
                    with patch("multi_agent.create_memory_tools", lambda db: None):
                        with patch("multi_agent.create_action_tracking_tools", lambda db: None):
                            from multi_agent import MultiAgentSystem
                            mas = MultiAgentSystem()

    cases = [
        ("hi how's it going", "general"),
        ("thanks!", "general"),
        ("help me create a morning routine", "general"),
        ("give me ideas for a side project", "general"),
        ("what should I do about my noisy neighbor", "general"),
        ("Write a function that adds two numbers", "executor"),
        ("add a print statement to main.py", "executor"),
        ("fix the bug in the auth module", "executor"),
        ("run the tests for the auth module", "executor"),
        ("git status please", None),     # ambiguous — either route is fine
        ("look up documentation for httpx", "researcher"),
        ("find the latest news on python 3.14", "researcher"),
    ]
    pass_count = 0
    for query, expected in cases:
        got = mas._short_circuit_classify(query)
        if expected is None:
            mark = "·"
        else:
            ok = got == expected
            mark = "✓" if ok else "✗"
            if ok:
                pass_count += 1
        target = expected or "(any)"
        print(f"  {mark}  {query!r:<55} → {got or '(no shortcut, → architect)':10s}  expected={target}")
    countable = sum(1 for _, e in cases if e is not None)
    print(f"\n  {pass_count}/{countable} classifications match expected agent")


def test_tool_filter_for_executor():
    """The executor uses an embedding similarity filter to pick top-N
    tools relevant to the user input. Verify that for 'create a file
    X', write_file ranks in the top 5."""
    section("TOOL FILTERING · top-5 tools for code-writing requests")
    from unittest.mock import patch, MagicMock
    with patch("multi_agent.build_agent_client", return_value=(MagicMock(), "m")):
        with patch("multi_agent.build_architect_client", return_value=(MagicMock(), "m")):
            with patch("multi_agent.load_skills", return_value=[]):
                with patch("multi_agent.create_memory_tools", lambda db: None):
                    with patch("multi_agent.create_action_tracking_tools", lambda db: None):
                        from multi_agent import MultiAgentSystem
                        mas = MultiAgentSystem()
    executor = mas.agents["executor"]
    for prompt in (
        "Create a file called game.py with a number guessing game",
        "fix the bug in line 42 of auth.py",
        "list the files in the workspace",
        "search the web for what xyz library does",
        "remember that my favorite color is blue",
    ):
        try:
            tools = executor._select_relevant_tools(prompt, top_n=5)
            names = [t["function"]["name"] for t in tools]
            print(f"\n  prompt: {prompt!r}")
            for n in names:
                print(f"     · {n}")
        except Exception as e:
            print(f"  ── ERROR on {prompt!r}: {type(e).__name__}: {e}")


def main():
    client = ollama.Client(host=OLLAMA_HOST, timeout=TIMEOUT)
    test_researcher(client, ["qwen3.5:9b"])
    test_general_size(client, ["qwen3.5:9b", "qwen3:14b"])
    test_routing_classifier()
    test_tool_filter_for_executor()


if __name__ == "__main__":
    main()
