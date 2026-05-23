"""Schema for the chunks yielded by MultiAgentSystem.run / ChatAgent.chat_stream.

The orchestrator's generator is the contract between the agent layer
and the UI. When the contract drifts (e.g., a new chunk_type appears
silently, a required field gets dropped), the UI breaks in subtle ways
the test suite doesn't catch — silent yield of a malformed `tool_end`
just shows the wrong thing in the tool panel, no crash.

This module pins the contract. Each known chunk_type has a small spec
of required + optional fields. Validation is loud (raises) when
EZCLAW_VALIDATE_CHUNKS=1 is set in the environment; otherwise it's a
no-op (free in production).

Usage:
    from orchestration import validate_stream
    for chunk in validate_stream(mas.run(prompt)):
        ...

Or one chunk at a time:
    from orchestration import validate_chunk
    validate_chunk(chunk)

Schemas
-------
"content"          : {"type", "content"}
"reasoning"        : {"type", "content"}                # sub-agent <think>
"status"           : {"type", "content"}                # short status line
"halt"             : {"type"}                           # cap-hit signal
"plan_created"     : {"type", "plan"}                   # plan.Plan instance
"plan_update"      : {"type", "plan"}
"intent"           : {"type", "agent", "reasoning"} + optional "reflection", "plan", "complete"
"tool_start"       : {"type", "name", "arguments"} + optional "interactive"
"tool_end"         : {"type", "name", "result"}
"auth_required"    : {"type", "name", "arguments"}
"context_augmented": {"type", "memories"}
"memory_stored"    : {"type", "fact"}
"skill_offer"      : {"type", "draft"}                  # draft has {name, description, procedure}
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator, Mapping, Any


class ChunkValidationError(ValueError):
    """Raised when a chunk fails schema validation in strict mode."""


# Required and known-optional keys per chunk type. A chunk MAY carry
# additional keys (forward-compat); unknown keys do not fail validation.
# A chunk MUST have all `required` keys. Optional are documented for
# producers but not enforced.
_SCHEMAS: dict = {
    "content":           {"required": ("type", "content"),                  "optional": ()},
    "reasoning":         {"required": ("type", "content"),                  "optional": ()},
    "status":            {"required": ("type", "content"),                  "optional": ()},
    "halt":              {"required": ("type",),                            "optional": ("reason",)},
    "plan_created":      {"required": ("type", "plan"),                     "optional": ()},
    "plan_update":       {"required": ("type", "plan"),                     "optional": ()},
    "intent":            {"required": ("type", "agent"),                    "optional": ("reasoning", "reflection", "plan", "complete", "current_task_id")},
    "tool_start":        {"required": ("type", "name", "arguments"),        "optional": ("interactive",)},
    "tool_end":          {"required": ("type", "name", "result"),           "optional": ()},
    "auth_required":     {"required": ("type", "name", "arguments"),        "optional": ()},
    "context_augmented": {"required": ("type", "memories"),                 "optional": ()},
    "memory_stored":     {"required": ("type", "fact"),                     "optional": ()},
    "skill_offer":       {"required": ("type", "draft"),                    "optional": ()},
}

KNOWN_CHUNK_TYPES: frozenset = frozenset(_SCHEMAS.keys())


def _strict() -> bool:
    """Validation is opt-in via env. Default off so production runs
    take no overhead. Tests / dev sessions can flip the flag."""
    return os.environ.get("EZCLAW_VALIDATE_CHUNKS", "").strip() == "1"


def validate_chunk(chunk: Any) -> None:
    """Validate a single chunk against its schema. No-op when the
    EZCLAW_VALIDATE_CHUNKS env var is unset/empty/"0".

    Raises ChunkValidationError when:
      - chunk is not a Mapping
      - chunk lacks a 'type' key
      - chunk.type is not in KNOWN_CHUNK_TYPES
      - any required key for that type is missing
    """
    if not _strict():
        return
    if not isinstance(chunk, Mapping):
        raise ChunkValidationError(f"chunk must be a Mapping, got {type(chunk).__name__}")
    if "type" not in chunk:
        raise ChunkValidationError(f"chunk is missing 'type' key: {chunk!r}")
    ctype = chunk["type"]
    if not isinstance(ctype, str):
        raise ChunkValidationError(f"chunk['type'] must be str, got {type(ctype).__name__}")
    if ctype not in _SCHEMAS:
        raise ChunkValidationError(
            f"unknown chunk type {ctype!r} — known: {sorted(KNOWN_CHUNK_TYPES)}"
        )
    schema = _SCHEMAS[ctype]
    for key in schema["required"]:
        if key not in chunk:
            raise ChunkValidationError(
                f"chunk type {ctype!r} missing required field {key!r}: {dict(chunk)!r}"
            )


def validate_stream(stream: Iterable[Any]) -> Iterator[Any]:
    """Wrap a chunk-yielding iterator with schema validation. Each
    chunk is validated before being passed through. When the env flag
    is off this is a transparent pass-through.

    Usage:
        for chunk in validate_stream(mas.run(prompt)):
            # ... handle chunk normally
    """
    for chunk in stream:
        validate_chunk(chunk)
        yield chunk
