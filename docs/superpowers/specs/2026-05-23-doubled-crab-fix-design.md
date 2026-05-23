# Doubled-Crab Fix — Design

**Date:** 2026-05-23
**Status:** Approved, one-line fix.

## Problem

While the agent is generating, the user sees two 🦀 mascots simultaneously:

- **Bottom status bar** (`cli.py:764`): a pulsing crab segment that's always present.
- **Active-generation spinner row** (`cli.py:964`): a second crab prepended to the spinner's status text whenever `is_generating` is True.

Both render at the same time during a generation.

## Decision

Suppress the **bottom-bar crab** while `self.is_generating` is True. Keep the spinner-row crab. When idle, the bar's crab shows. When generating, the spinner-row crab takes over.

## Why this direction

The user picked this over dropping the spinner-row crab. The bar still has the cycling `ACTIVITY_FRAMES` glyph (line 763) so it remains visibly alive while generating; the bar doesn't go dead.

## Change

In `_build_status_segments` in `cli.py`, around line 762-769. The `segments` list literal currently includes the mascot segment unconditionally. Move that segment behind an `is_generating` check so it only appears when the agent is idle.

Concretely:

```python
        segments = [
            (BG + f"bold {act_color}", f"  {glyph}  "),
        ]
        if not self.is_generating:
            segments.append((BG + f"bold {mascot_color}", MASCOT + " "))
        segments.extend([
            (BG + "#7a7570", f"{auth_icon}  "),
            (BG + "#ffd166", f"{mode_glyph} {model_info}"),
            divider,
            (BG + "#c8c4be", f"{msg_count} msgs"),
        ])
```

`mascot_color` only needs to be computed when the segment is appended; gate that too if cheap.

## No new tests

This is a visual change; the existing test suite doesn't cover the status bar rendering, and adding a UI test for one conditional isn't worth the harness setup. Verification is manual — start ezclaw, watch the bar during a generation, confirm only one crab.

## Non-goals

- Don't touch the spinner row.
- Don't touch the welcome banner.
- Don't change colors.
