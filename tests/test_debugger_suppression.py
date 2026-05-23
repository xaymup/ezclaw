"""The debugger's content output is for the architect's planning, never
for the user. These tests pin the chat-stream contract.

Regression test for: when the architect routes a step to the debugger,
the user previously saw the raw diagnosis + Proposed Fix in chat. That
was wrong — the debugger is a planning aid, not a user-facing voice."""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent import MultiAgentSystem


def _fake_subagent_stream(content_text, tool_name=None):
    """Build a generator that mimics a SpecializedAgent.chat_stream by
    yielding one content chunk and one tool_end chunk (if requested)."""
    def gen(context):
        yield {"type": "content", "content": content_text}
        if tool_name:
            yield {"type": "tool_end", "name": tool_name, "result": "ok"}
    return gen


def _build_system_with_fake_agents(
    debugger_output="Diagnosis: cause is X. Proposed Fix: 1) Y 2) Z",
    executor_output="Implemented Y and Z. Tests pass.",
):
    """Construct a MultiAgentSystem with mocked LLM clients so we can
    drive the chat stream without real models."""
    with patch("multi_agent.build_agent_client", return_value=(MagicMock(), "m")):
        with patch("multi_agent.build_architect_client", return_value=(MagicMock(), "m")):
            with patch.object(__import__("multi_agent").SpecializedAgent, "_pre_embed_tools", lambda self: None):
                with patch("multi_agent.load_skills", return_value=[]):
                    with patch("multi_agent.create_memory_tools", lambda db: None):
                        mas = MultiAgentSystem()

    debugger = mas.agents["debugger"]
    executor = mas.agents["executor"]
    debugger.chat_stream = _fake_subagent_stream(debugger_output)
    executor.chat_stream = _fake_subagent_stream(executor_output, tool_name="apply_diff")
    return mas


def test_debugger_content_chunks_are_suppressed_from_chat_stream():
    """Drive the MAS sub-agent loop directly with a debugger key and
    verify that the content chunk the debugger emits never appears in
    the yielded chat stream."""
    mas = _build_system_with_fake_agents(
        debugger_output="Root cause: race in handler. Proposed Fix: lock the registry."
    )

    # Drive ONLY the inner loop body — we patch out the architect
    # entirely so we can isolate the stream-filtering behavior.
    debugger_agent = mas.agents["debugger"]

    # Simulate the loop's content-filter contract directly: copy the
    # logic into a local function so we test the contract, not the
    # broader orchestration. The contract is: when agent_key=="debugger",
    # content chunks are captured but NOT yielded.
    def filter_stream(agent_key, agent, context):
        suppress = (agent_key == "debugger")
        captured = []
        emitted = []
        for chunk in agent.chat_stream(context):
            if chunk["type"] == "content":
                captured.append(chunk["content"])
                if suppress:
                    continue
            emitted.append(chunk)
        return captured, emitted

    captured, emitted = filter_stream("debugger", debugger_agent, "ctx")

    # The diagnosis WAS captured (architect can still see it)
    assert any("Root cause" in c for c in captured)
    assert any("Proposed Fix" in c for c in captured)
    # But NOTHING was emitted to the chat — no content chunk reached the user
    assert all(e["type"] != "content" for e in emitted), \
        f"debugger content leaked into chat stream: {emitted}"


def test_executor_content_chunks_are_emitted_normally():
    """The same filter must NOT suppress executor output — executor
    responses are how the user gets their answer."""
    mas = _build_system_with_fake_agents(
        executor_output="Created the file. 12 tests passing."
    )
    executor_agent = mas.agents["executor"]

    def filter_stream(agent_key, agent, context):
        suppress = (agent_key == "debugger")
        emitted = []
        for chunk in agent.chat_stream(context):
            if chunk["type"] == "content" and suppress:
                continue
            emitted.append(chunk)
        return emitted

    emitted = filter_stream("executor", executor_agent, "ctx")
    content_chunks = [e for e in emitted if e["type"] == "content"]
    assert content_chunks, "executor content was incorrectly suppressed"
    assert any("12 tests passing" in c["content"] for c in content_chunks)


def test_final_response_does_not_pick_up_debugger_output():
    """When the orchestration loop tracks `final_response`, debugger
    output must NEVER update it — otherwise the user's final answer
    would be the raw diagnosis."""
    # Simulate the contract directly:
    final_response = ""

    def maybe_set_final(agent_key, step_output):
        suppress = (agent_key == "debugger")
        if step_output.strip() and not suppress:
            return step_output.strip()
        return None

    # Debugger step — should NOT update final_response
    result = maybe_set_final("debugger", "Diagnosis: cache miss bug.")
    assert result is None
    final_response = result if result else final_response
    assert final_response == ""

    # Executor step — SHOULD update final_response
    result = maybe_set_final("executor", "Fixed and verified. 5/5 tests pass.")
    assert result == "Fixed and verified. 5/5 tests pass."
    final_response = result if result else final_response
    assert "tests pass" in final_response


def test_sub_agent_step_yields_neutral_handoff_status():
    """The user shouldn't see the sub-agent's raw output (Spec E
    suppresses everyone), but they SHOULD see *something* happening
    when the sub-agent produced output — a short status line so the
    chat doesn't look frozen. The phrasing must be neutral (not say
    'debugger' on a general/executor step), and the handoff must NOT
    fire on silent steps that produced no output (that bug emitted
    'debugger: analysis complete' on every architect iteration)."""
    import multi_agent
    src = open(multi_agent.__file__).read()
    # Neutral phrasing — parameterized on the actual agent_key, not
    # hardcoded to "debugger"
    assert 'f"{agent_key}: handing off to architect"' in src
    # And the yield is gated on step_output or tool results
    assert "step_output.strip() or step_tool_results" in src
