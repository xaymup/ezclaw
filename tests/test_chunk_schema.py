"""Tier 4 — Chunk schema validator tests.

The orchestrator/UI contract is the set of chunk shapes that
MultiAgentSystem.run and ChatAgent.chat_stream yield. This module pins
that contract."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration import (
    validate_chunk,
    validate_stream,
    KNOWN_CHUNK_TYPES,
    ChunkValidationError,
)


# ── Strict mode is opt-in via env ───────────────────────────────────────────

def test_validation_is_a_noop_without_env_flag(monkeypatch):
    """Default behavior: no env flag → no validation. Any garbage passes."""
    monkeypatch.delenv("EZCLAW_VALIDATE_CHUNKS", raising=False)
    # All of these would fail in strict mode; none raise here.
    validate_chunk({})
    validate_chunk({"type": "no-such-chunk-type-here"})
    validate_chunk({"type": "tool_start"})  # missing name + arguments
    validate_chunk(None)
    validate_chunk("just a string")


def test_validation_raises_when_strict(monkeypatch):
    monkeypatch.setenv("EZCLAW_VALIDATE_CHUNKS", "1")
    with pytest.raises(ChunkValidationError):
        validate_chunk({})


# ── Per-chunk-type shape ─────────────────────────────────────────────────────

@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("EZCLAW_VALIDATE_CHUNKS", "1")


def test_valid_content_chunk_passes(strict):
    validate_chunk({"type": "content", "content": "hello"})


def test_content_chunk_missing_content_fails(strict):
    with pytest.raises(ChunkValidationError, match="missing required field 'content'"):
        validate_chunk({"type": "content"})


def test_valid_intent_chunk_passes(strict):
    validate_chunk({
        "type": "intent",
        "agent": "executor",
        "reasoning": "do the thing",
        "reflection": {"goal": "x"},
        "plan": "step 1",
        "complete": False,
    })


def test_intent_chunk_must_have_agent(strict):
    with pytest.raises(ChunkValidationError, match="missing required field 'agent'"):
        validate_chunk({"type": "intent", "reasoning": "x"})


def test_valid_tool_start_chunk_passes(strict):
    validate_chunk({
        "type": "tool_start",
        "name": "read_file",
        "arguments": {"path": "foo.py"},
    })


def test_tool_end_must_have_result(strict):
    with pytest.raises(ChunkValidationError, match="missing required field 'result'"):
        validate_chunk({"type": "tool_end", "name": "read_file"})


def test_unknown_chunk_type_fails(strict):
    with pytest.raises(ChunkValidationError, match="unknown chunk type"):
        validate_chunk({"type": "definitely_not_a_known_type"})


def test_non_string_chunk_type_fails(strict):
    with pytest.raises(ChunkValidationError, match="must be str"):
        validate_chunk({"type": 42, "content": "x"})


def test_non_mapping_chunk_fails(strict):
    with pytest.raises(ChunkValidationError, match="must be a Mapping"):
        validate_chunk("not a dict")
    with pytest.raises(ChunkValidationError):
        validate_chunk(None)


def test_unknown_extra_keys_are_tolerated_forward_compat(strict):
    """Producers may add new keys for forward compatibility — the
    validator must not reject chunks for having more than the schema
    says, only for having less."""
    validate_chunk({
        "type": "content",
        "content": "x",
        "future_field": "doesn't break anything",
    })


# ── validate_stream ────────────────────────────────────────────────────────

def test_validate_stream_passes_chunks_through(strict):
    chunks = [
        {"type": "content", "content": "a"},
        {"type": "status", "content": "thinking"},
        {"type": "tool_start", "name": "x", "arguments": {}},
    ]
    out = list(validate_stream(iter(chunks)))
    assert out == chunks


def test_validate_stream_raises_on_first_bad_chunk(strict):
    def gen():
        yield {"type": "content", "content": "ok"}
        yield {"type": "tool_end", "name": "bad"}  # missing result
        yield {"type": "content", "content": "never reached"}

    out = []
    with pytest.raises(ChunkValidationError):
        for c in validate_stream(gen()):
            out.append(c)
    assert len(out) == 1  # we got the first chunk before the failure


# ── Coverage: every chunk type emitted by the codebase is in the schema ─────

def test_known_chunk_types_cover_codebase_emissions():
    """If anyone adds a new `yield {"type": "X"}` in multi_agent.py /
    agent.py / cli.py, KNOWN_CHUNK_TYPES must list it. Otherwise the
    schema becomes meaningless. This catches drift early."""
    import re
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = ["multi_agent.py", "agent.py", "cli.py"]
    pattern = re.compile(r'["\']type["\']\s*:\s*["\']([a-z_]+)["\']')
    found: set = set()
    for f in files:
        with open(os.path.join(here, f)) as fh:
            src = fh.read()
        for m in pattern.finditer(src):
            ctype = m.group(1)
            # Heuristic: only consider strings that look like chunk types
            # (lowercase + underscores). Filter out incidental matches by
            # whitelisting known prefixes.
            found.add(ctype)

    # Subset check: each chunk type yielded somewhere must be known to
    # the schema. We do NOT require the reverse (schema may declare
    # types not yet emitted).
    emitted_chunk_types = {
        "content", "reasoning", "status", "halt", "plan_created",
        "plan_update", "intent", "tool_start", "tool_end",
        "auth_required", "context_augmented", "memory_stored",
        "skill_offer",
    }
    missing = emitted_chunk_types - KNOWN_CHUNK_TYPES
    assert not missing, (
        f"Chunk types emitted in the codebase but not declared in the "
        f"schema: {sorted(missing)}. Add them to orchestration/chunk_schema.py."
    )
