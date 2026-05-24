"""Whimsical 🦀 crab-themed status vocabulary for the TUI.

One module-level list per kind of internal event. `pick(category)` returns
a random variant — call it ONCE at status-emit time (not on every UI
render) so the chosen phrase is stable while it's displayed; the next
emission rolls a fresh one for variety.

Phrase quality bar (set after a streamline pass):
- 2-5 words, lowercase, no trailing punctuation.
- Each phrase must MAP CLEARLY to what's actually happening. Pure
  nonsense like "bambaloozing" was removed because a user staring at a
  status bar gets no information from it. Crab/coastal flavor is
  encouraged where it doesn't obscure meaning.
- 15+ variants per common category so the same phrase doesn't appear
  twice within a single multi-step turn.

Augmentation:
- `augment_with_llm(client, model_name)` calls the running Ollama (or
  DeepSeek) model to brainstorm fresh phrases per category and merges
  them into the in-memory lists. Persisted to ~/.ezclaw/phrase_pool.json
  so subsequent sessions inherit them. Triggered by the `/phrases
  refresh` slash command — never on startup, so cold-start latency is
  unchanged.
"""

import json
import os
import random
from pathlib import Path
from typing import Iterable

# ── Connecting to the model ─────────────────────────────────────────────────
CONNECTING = [
    "scuttling over",
    "waving pincers at the model",
    "rapping on the shell",
    "sniffing the current",
    "knocking on ollama's door",
    "opening the line",
    "warming up the model",
    "tapping out a greeting",
    "wiring up the cable",
    "shaking hands with the daemon",
    "raising the antennas",
    "sending the first ping",
    "starting the call",
    "calling the model",
    "loading the conversation",
]

# ── Architect: planning pass ────────────────────────────────────────────────
ARCHITECT_PLANNING = [
    "brewing the plan",
    "scribbling the catch list",
    "consulting the tide charts",
    "marking coordinates on the shell",
    "weighing the haul",
    "drawing up tasks",
    "sketching the route",
    "breaking it into steps",
    "deciding what to do first",
    "laying out the plan",
    "ordering the steps",
    "splitting the work",
    "outlining the approach",
    "picking the entry point",
    "mapping the dive",
    "naming the steps",
    "shaping the plan",
]

# ── Architect: per-step analysis ────────────────────────────────────────────
ARCHITECT_THINKING = [
    "shellgazing",
    "checking the currents",
    "tide-pool divining",
    "claw-tapping the diagram",
    "rereading the kelp scrolls",
    "deciding next step",
    "reviewing the last result",
    "picking the next move",
    "checking what's done",
    "weighing the options",
    "thinking it through",
    "considering routes",
    "checking the plan",
    "re-reading the request",
    "deciding who's next",
    "looking at the board",
    "consulting the log",
]

# ── Architect: wrapping up ──────────────────────────────────────────────────
ARCHITECT_FINALIZING = [
    "wrapping up the catch",
    "tying the net",
    "double-checking the haul",
    "rinsing the sand off",
    "writing the summary",
    "composing the reply",
    "polishing the answer",
    "finalizing the response",
    "checking once more",
    "drafting the recap",
    "putting it all together",
    "preparing to hand back",
    "stitching the answer",
    "tidying the shell",
    "sealing the report",
]

# ── Sub-agent doing work ────────────────────────────────────────────────────
AGENT_WORKING = [
    # NB: "bambaloozing" used to live here. Removed during the phrase
    # streamline pass — a user pointed out (correctly) that it conveys
    # nothing about what the agent is doing.
    "snipping",
    "scuttling sideways",
    "rummaging in the sand",
    "stewing",
    "clattering pincers",
    "rolling up the sleeves (no sleeves)",
    "running the call",
    "thinking",
    "working on it",
    "computing",
    "processing",
    "crunching",
    "evaluating",
    "on it",
    "executing",
    "drafting",
    "in the workshop",
    "elbows deep",
    "noodling",
    "head down",
]

# ── Tool still running (the panel "running…" indicator) ─────────────────────
TOOL_RUNNING = [
    "simmering",
    "brewing",
    "stewing",
    "bubbling away",
    "running",
    "in progress",
    "working on it",
    "underway",
    "ticking along",
    "chugging",
    "humming",
    "spinning up",
    "executing",
    "in flight",
]

# ── Escalation: fast-route gave up, going to the architect ──────────────────
ESCALATING = [
    "murky waters — calling the senior crab",
    "this needs the architect's pincers",
    "kicking it upstairs",
    "calling the architect",
    "going up the chain",
    "promoting to architect",
    "asking the planner",
    "stepping it up",
    "this needs a plan",
    "bigger than expected",
    "passing to the architect",
]

# ── Loop detection — agent is stuck ─────────────────────────────────────────
LOOP_DETECTED = [
    "going in circles — taking a breather",
    "chasing the same wave",
    "we already tried this — stopping",
    "the kelp's tangled",
    "stuck in a loop",
    "same step twice — halting",
    "spinning wheels",
    "no progress detected — pausing",
    "looping on the same step",
    "this isn't moving — stopping",
]

# ── Hit max steps ───────────────────────────────────────────────────────────
STEP_LIMIT = [
    "ran out of pincer-power",
    "the kettle's gone dry",
    "too many tides — stopping here",
    "step budget exhausted",
    "max steps reached",
    "ran out of moves",
    "hit the step ceiling",
    "out of attempts",
    "stopping at the step limit",
]

# ── Pivot recovery (auto-retry at higher temperature) ───────────────────────
PIVOT = [
    "shaking off the sand, trying something else",
    "new approach — flipping the shell",
    "pivoting sideways like a proper crab",
    "trying a different angle",
    "switching tactic",
    "fresh approach",
    "different route",
    "trying it another way",
    "shaking it up",
    "changing strategy",
    "reaching for plan B",
]

# ── Task completed ──────────────────────────────────────────────────────────
COMPLETED = [
    "all snipped and shipped",
    "catch landed",
    "back to the shore",
    "shell sealed, job done",
    "done",
    "finished",
    "wrapped up",
    "task complete",
    "all done",
    "delivered",
    "handed back",
    "ready to go",
    "shipped",
    "sent it",
    "off the bench",
]

# ── "thinking" reasoning panel badge ────────────────────────────────────────
THINKING_BADGE = "shellgazing"   # singular; renders every UI tick, no rotation
THOUGHT_BADGE = "musings"


# ── Augmentation cache ──────────────────────────────────────────────────────
# All categories in one place so the augmentation + cache loader can iterate.
_CATEGORIES: dict = {
    "CONNECTING": CONNECTING,
    "ARCHITECT_PLANNING": ARCHITECT_PLANNING,
    "ARCHITECT_THINKING": ARCHITECT_THINKING,
    "ARCHITECT_FINALIZING": ARCHITECT_FINALIZING,
    "AGENT_WORKING": AGENT_WORKING,
    "TOOL_RUNNING": TOOL_RUNNING,
    "ESCALATING": ESCALATING,
    "LOOP_DETECTED": LOOP_DETECTED,
    "STEP_LIMIT": STEP_LIMIT,
    "PIVOT": PIVOT,
    "COMPLETED": COMPLETED,
}

# Snapshot the built-in lists BEFORE load_cache() runs. `reset_cache()`
# uses this to truncate each list back to its original length without
# requiring a session restart. Taken on import while every list still
# holds only the hardcoded entries this module ships with.
_BUILTIN_LENGTHS: dict = {name: len(lst) for name, lst in _CATEGORIES.items()}

_CACHE_PATH = Path(os.path.expanduser("~/.ezclaw/phrase_pool.json"))


def categories() -> dict:
    """Return the {name: list} mapping. Mutating the returned lists
    affects future picks (this is intentional — augment_with_llm uses
    this for in-place extension)."""
    return _CATEGORIES


def counts() -> dict:
    """Return {category_name: len(category)} for the /phrases status view."""
    return {name: len(lst) for name, lst in _CATEGORIES.items()}


def pick(category) -> str:
    """Return a random variant. Call once per event — the result should
    be stored in status state, not recomputed on every render tick."""
    return random.choice(category)


def _merge_in_place(category_name: str, new_phrases: Iterable[str]) -> int:
    """Add `new_phrases` to a category's list if they aren't already
    present. Returns the count actually added. Whitespace / case folded
    for the dup check."""
    target = _CATEGORIES.get(category_name)
    if target is None:
        return 0
    seen_norm = {p.lower().strip() for p in target}
    added = 0
    for p in new_phrases:
        if not isinstance(p, str):
            continue
        clean = p.strip().rstrip(".!?").lower()
        if not clean or clean in seen_norm:
            continue
        # Trust the LLM's casing exactly as it produced it, only stripped.
        target.append(p.strip().rstrip(".!?"))
        seen_norm.add(clean)
        added += 1
    return added


def load_cache() -> int:
    """Load persisted LLM-generated phrases from `~/.ezclaw/phrase_pool.json`
    and merge into the in-memory lists. Returns total phrases added.
    Best-effort — any failure (missing file, bad JSON, wrong shape) is
    silently ignored."""
    try:
        if not _CACHE_PATH.exists():
            return 0
        data = json.loads(_CACHE_PATH.read_text())
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    total = 0
    for name, plist in data.items():
        if isinstance(plist, list):
            total += _merge_in_place(name, plist)
    return total


def _persist_cache(generated: dict) -> None:
    """Write the generated bundle to disk so subsequent sessions inherit
    it. The on-disk format mirrors the LLM response shape:
    {category_name: [phrase, ...]}. Existing cache is MERGED (the user
    can refresh repeatedly to grow the pool)."""
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if _CACHE_PATH.exists():
            try:
                existing = json.loads(_CACHE_PATH.read_text())
                if not isinstance(existing, dict):
                    existing = {}
            except Exception:
                existing = {}
        for name, plist in generated.items():
            if not isinstance(plist, list):
                continue
            merged = list(existing.get(name, []))
            seen = {p.lower().strip() for p in merged}
            for p in plist:
                if isinstance(p, str) and p.strip().lower() not in seen:
                    merged.append(p.strip().rstrip(".!?"))
                    seen.add(p.strip().lower())
            existing[name] = merged
        _CACHE_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    except Exception:
        # Cache failure is non-fatal. The in-memory merge already happened.
        pass


def augment_with_llm(client, model: str, per_category: int = 5, timeout: float = 30.0) -> dict:
    """Ask the running LLM to brainstorm fresh status phrases.

    Returns {category_name: [new_phrases_actually_added]} so the caller
    can report what changed. Best-effort — on any failure returns an
    empty dict and leaves the in-memory lists unchanged.

    `client` should be an ollama.Client (or any object with a `.chat()`
    method matching Ollama's signature). `model` is the model name to
    use — pass the fastest available; this is a one-shot creative call,
    not a code task. `per_category` is how many phrases per category to
    request; the model usually returns close to that but may produce
    duplicates which get filtered.

    The call is bounded by `timeout` seconds. Set EZCLAW_PHRASES_DEBUG=1
    to print the raw model response for tuning the prompt.
    """
    cat_names = list(_CATEGORIES.keys())
    existing_summary = {
        name: ", ".join(_CATEGORIES[name][:8]) + ("…" if len(_CATEGORIES[name]) > 8 else "")
        for name in cat_names
    }
    prompt = (
        "Generate playful crab/coastal-themed status phrases for a CLI agent's "
        "status bar. Constraints per phrase:\n"
        "- 2-5 words, lowercase, no trailing punctuation\n"
        "- Each phrase must legibly map to what's happening (semantic clarity "
        "  matters more than the pun)\n"
        "- Crab/ocean/coastal flavor is encouraged but not required\n"
        "- Avoid nonsense words (e.g. 'bambaloozing'); avoid duplicates of the existing ones below\n\n"
        f"Generate exactly {per_category} new phrases for each category.\n\n"
        "Categories and what each describes:\n"
        "- CONNECTING: opening the connection to the local model\n"
        "- ARCHITECT_PLANNING: the architect agent is decomposing a request into tasks\n"
        "- ARCHITECT_THINKING: the architect is deciding the next step mid-run\n"
        "- ARCHITECT_FINALIZING: the architect is composing the final reply\n"
        "- AGENT_WORKING: a sub-agent is mid-call (executor, researcher, etc.)\n"
        "- TOOL_RUNNING: a specific tool (read_file, run_shell, web_fetch, …) is running\n"
        "- ESCALATING: the fast-route gave up; the request is being kicked to the architect\n"
        "- LOOP_DETECTED: the same step ran twice with no progress; halting\n"
        "- STEP_LIMIT: the maximum step count for a turn was hit\n"
        "- PIVOT: the run hit a wall and is auto-retrying with a different approach\n"
        "- COMPLETED: the user's request is fully satisfied\n\n"
        "Existing phrases (do NOT repeat these): " + json.dumps(existing_summary) + "\n\n"
        "Return ONLY a JSON object whose keys are the category names above and "
        "whose values are lists of new phrases. No prose before or after."
    )
    debug = bool(os.environ.get("EZCLAW_PHRASES_DEBUG"))
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            # 8k is enough headroom for the long instruction + JSON
            # response across all 11 categories. The previous 4k limit
            # silently truncated the prompt on some models.
            options={"temperature": 1.0, "num_ctx": 8192},
        )
        content = resp["message"]["content"].strip()
    except Exception as e:
        if debug:
            print(f"[phrases.augment_with_llm] client.chat raised: {type(e).__name__}: {e}")
        return {}

    if debug:
        print("[phrases.augment_with_llm raw]:", content[:2000])

    # Try strict JSON first, fall back to extracting the first {...} block.
    parsed = None
    try:
        parsed = json.loads(content)
    except Exception:
        import re
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
    if not isinstance(parsed, dict):
        return {}

    added_bundle: dict = {}
    for name in cat_names:
        plist = parsed.get(name)
        if not isinstance(plist, list):
            continue
        before = list(_CATEGORIES[name])
        n_added = _merge_in_place(name, plist)
        if n_added > 0:
            added_bundle[name] = [p for p in _CATEGORIES[name] if p not in before]

    if added_bundle:
        _persist_cache(added_bundle)
    return added_bundle


def reset_cache() -> None:
    """Delete the on-disk cache and revert the in-memory lists to their
    built-in defaults. Used by `/phrases reset`.

    Implementation: every list in `_CATEGORIES` is mutated in place
    (`load_cache()` / `augment_with_llm()` both `.append()` to the same
    objects), so truncating each list back to `_BUILTIN_LENGTHS[name]`
    drops everything the augmentation added — both this session's adds
    AND any historical adds the cache loaded at import.
    """
    try:
        if _CACHE_PATH.exists():
            _CACHE_PATH.unlink()
    except Exception:
        pass
    for name, original_len in _BUILTIN_LENGTHS.items():
        lst = _CATEGORIES.get(name)
        if lst is not None and len(lst) > original_len:
            del lst[original_len:]


# Load any persisted augmentations on import so a refreshed pool survives
# across sessions without the user needing to re-run /phrases refresh.
load_cache()
