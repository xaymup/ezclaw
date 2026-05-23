"""Continuation-phrase recognizer for the halt-then-resume UX.

When the agent emits a halt event (max_iterations reached), the CLI
keeps the active panel open. The next user input is treated as a
continuation if it matches one of these phrases exactly (after trim
and case fold); otherwise the prior panel is finalized to history and
a fresh turn begins.

Strict matching avoids false positives. Users learn the rule quickly:
"continue" (or one of the listed alternatives) extends; anything else
starts fresh.
"""

_CONTINUATION_PHRASES = frozenset({
    "continue",
    "go on",
    "keep going",
    "more",
    "next",
})


def is_continuation(text: str) -> bool:
    """Return True iff `text`, trimmed and case-folded, is one of the
    canonical continuation phrases."""
    if not text:
        return False
    return text.strip().lower() in _CONTINUATION_PHRASES
