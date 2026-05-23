"""Whimsical 🦀 crab-themed status vocabulary for the TUI.

One categorized list per kind of internal event. `pick(category)` returns
a random variant — call it ONCE at status-emit time (not on every UI
render) so the chosen phrase is stable while it's displayed; the next
emission rolls a fresh one for variety across a long session.

Conventions:
  - All phrases are short, lowercase, no trailing punctuation; callers
    append "…" or "(step n/m)" as needed.
  - Tone is playful but each phrase still maps clearly to what's going on.
"""

import random

# ── Connecting to the model ─────────────────────────────────────────────────
CONNECTING = [
    "scuttling over",
    "waving pincers at the model",
    "rapping on the shell",
    "sniffing the current",
]

# ── Architect: planning pass ────────────────────────────────────────────────
ARCHITECT_PLANNING = [
    "brewing the plan",
    "scribbling the catch list",
    "consulting the tide charts",
    "marking coordinates on the shell",
    "weighing the haul",
]

# ── Architect: per-step analysis ────────────────────────────────────────────
ARCHITECT_THINKING = [
    "shellgazing",
    "checking the currents",
    "tide-pool divining",
    "claw-tapping the diagram",
    "rereading the kelp scrolls",
]

# ── Architect: wrapping up ──────────────────────────────────────────────────
ARCHITECT_FINALIZING = [
    "wrapping up the catch",
    "tying the net",
    "double-checking the haul",
    "rinsing the sand off",
]

# ── Sub-agent doing work ────────────────────────────────────────────────────
AGENT_WORKING = [
    "bambaloozing",
    "snipping",
    "scuttling sideways",
    "rummaging in the sand",
    "stewing",
    "clattering pincers",
    "rolling up the sleeves (no sleeves)",
]

# ── Tool still running (the panel "running…" indicator) ─────────────────────
TOOL_RUNNING = [
    "simmering",
    "brewing",
    "stewing",
    "bubbling away",
]

# ── Escalation: fast-route gave up, going to the architect ──────────────────
ESCALATING = [
    "murky waters — calling the senior crab",
    "this needs the architect's pincers",
    "kicking it upstairs",
]

# ── Loop detection — agent is stuck ─────────────────────────────────────────
LOOP_DETECTED = [
    "going in circles — taking a breather",
    "chasing the same wave",
    "we already tried this — stopping",
    "the kelp's tangled",
]

# ── Hit max steps ───────────────────────────────────────────────────────────
STEP_LIMIT = [
    "ran out of pincer-power",
    "the kettle's gone dry",
    "too many tides — stopping here",
]

# ── Pivot recovery (auto-retry at higher temperature) ───────────────────────
PIVOT = [
    "shaking off the sand, trying something else",
    "new approach — flipping the shell",
    "pivoting sideways like a proper crab",
]

# ── Task completed ──────────────────────────────────────────────────────────
COMPLETED = [
    "all snipped and shipped",
    "catch landed",
    "back to the shore",
    "shell sealed, job done",
]

# ── "thinking" reasoning panel badge ────────────────────────────────────────
THINKING_BADGE = "shellgazing"   # singular; renders every UI tick, no rotation
THOUGHT_BADGE = "musings"


def pick(category) -> str:
    """Return a random variant. Call once per event — the result should
    be stored in status state, not recomputed on every render tick."""
    return random.choice(category)
