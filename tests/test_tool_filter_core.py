"""Tool filter — core workhorse tools must always be available.

The embedding-similarity ranker is good at finding semantically
matching tools, but it has blind spots: 'Create a file' ranked
`learn_skill` ahead of `write_file` in a real benchmark, and
'remember that X' dropped `remember` out of the top 5 entirely.

Core tools (read_file, write_file, apply_diff, list_dir, run_shell,
grep_codebase, delegate, current_datetime) get an allowlist
guarantee — they're ALWAYS in the returned set, even if their
embedding similarity score is too low to rank.
"""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_executor():
    from multi_agent import MultiAgentSystem, SpecializedAgent
    with patch("multi_agent.build_agent_client", return_value=(MagicMock(), "m")):
        with patch("multi_agent.build_architect_client", return_value=(MagicMock(), "m")):
            with patch("multi_agent.load_skills", return_value=[]):
                with patch("multi_agent.create_memory_tools", lambda db: None):
                    with patch("multi_agent.create_action_tracking_tools", lambda db: None):
                        mas = MultiAgentSystem()
    return mas.agents["executor"]


def _names(tool_defs):
    return {t["function"]["name"] for t in tool_defs}


def test_core_tools_always_present_with_top_n_5():
    """Even with the tightest top_n that production never uses, the
    core workhorse tools survive. If a model is given top-5 tools and
    the LLM picks one, it must be able to pick `write_file` for a
    code-creation request."""
    executor = _build_executor()
    selected = executor._select_relevant_tools(
        "remember that my favorite color is blue", top_n=5,
    )
    names = _names(selected)
    # Core tools we promise to never drop
    for required in ("read_file", "write_file", "list_dir", "run_shell"):
        assert required in names, (
            f"core tool {required!r} dropped from top-5: {names}"
        )


def test_core_tools_dont_overshoot_top_n():
    """The replacement logic must not GROW the surface — top_n is a
    promise about ceiling, not just floor."""
    executor = _build_executor()
    selected = executor._select_relevant_tools(
        "do something completely orthogonal to tools", top_n=10,
    )
    assert len(selected) <= 10


def test_high_similarity_tool_still_first():
    """Adding the core-tool allowlist must not displace genuinely
    relevant tools that happened to rank high naturally. For
    'web search for X', web_search must remain in the result."""
    executor = _build_executor()
    selected = executor._select_relevant_tools(
        "search the web for the latest python docs", top_n=8,
    )
    names = _names(selected)
    assert "web_search" in names


def test_filter_is_a_passthrough_when_tools_below_top_n():
    """When the total tool count <= top_n, no filtering happens at
    all. Verify by setting top_n higher than the total."""
    executor = _build_executor()
    total = len(executor.tools)
    selected = executor._select_relevant_tools("anything", top_n=total + 5)
    assert len(selected) == total


def test_filter_returns_top_n_on_embed_failure():
    """When the embed() call raises (e.g. ollama down), fall back to
    the first N tools — not crash, not infinite-loop. Pin this
    contract."""
    executor = _build_executor()
    with patch("multi_agent.embed", side_effect=RuntimeError("ollama is down")):
        selected = executor._select_relevant_tools("anything", top_n=5)
    assert len(selected) == 5


def test_write_file_present_for_file_creation_queries():
    """Regression: 'Create a file' should not push write_file out by
    favoring `learn_skill`. With the core-tool guarantee, write_file
    is always there."""
    executor = _build_executor()
    for query in (
        "Create a file called game.py with a number guessing game",
        "save the snake game to snake.py",
        "write a new module called helpers.py",
    ):
        selected = executor._select_relevant_tools(query, top_n=10)
        names = _names(selected)
        assert "write_file" in names, (
            f"write_file missing for {query!r}: {sorted(names)}"
        )


def test_read_file_present_for_bug_fix_queries():
    """Regression: 'fix the bug in line 42 of auth.py' previously
    dropped read_file from top-5. To fix a bug you need to read the
    file first; the core-tool guarantee ensures it's always there."""
    executor = _build_executor()
    for query in (
        "fix the bug in line 42 of auth.py",
        "the function in module.py is wrong, please fix it",
        "debug the failing test in tests/test_main.py",
    ):
        selected = executor._select_relevant_tools(query, top_n=8)
        names = _names(selected)
        assert "read_file" in names, (
            f"read_file missing for {query!r}: {sorted(names)}"
        )
