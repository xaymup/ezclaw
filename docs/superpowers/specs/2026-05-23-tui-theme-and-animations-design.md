# TUI Theme Layer & Animated Polish

**Status:** Draft
**Date:** 2026-05-23
**Owner:** Mohamed Kotp

## Problem

`cli.py` has accumulated visual styling inline: hex colors (`#5fafff`,
`#ffd700`, `#9999cc`), glyph choices (`→`, `·`, `⚙`, `✂`), spinner
configuration, border styles, and role-color mappings are scattered across
panel-render code, status-bar code, and the welcome panel. Every tweak
touches multiple call sites.

The UI is also visually flat: one global dots spinner regardless of which
agent is working, no visual cue when routing handoffs happen, all tool
panels render in the same color regardless of whether the tool touched a
file, ran a shell command, or fetched from the web.

## Goal

Two outcomes:

1. **Theming layer.** A new `theme.py` module owns every visual constant
   (palette, role bundles, tool-kind bundles, spinner frames, icons). `cli.py`
   reads from it. Future restyling — adding a calm mode, a light theme, or a
   user-customizable theme — becomes a one-file change.

2. **Modern/animated polish on four surfaces:** welcome banner, working
   spinner & status, tool execution panels, architect chip & assistant
   output borders. Discrete frame-stepped animation only — no attempt at
   continuous-motion effects terminals can't deliver.

## Non-goals

- Calm mode / animation-off toggle. Trivial to add later via a
  `THEME.motion_level` field; not in v1.
- Light/dark theme switching. The theming infra makes it cheap to add later;
  v1 ships a single dark theme.
- Layout changes. Panel placement, status-bar position, scroll behavior,
  keybindings all unchanged.
- Welcome banner *reveal* animation. Gradient styling yes, animated reveal
  no — reveals on every paint feel gimmicky after the first viewing.
- Changes to `multi_agent.py`, `agent.py`, `tools.py`, `model_client.py`, or
  `memory.py`. This is a pure UI change.

## Architecture

Two layers:

1. **`theme.py`** (new) — pure data module exporting a singleton `THEME`
   instance. No I/O, no side effects, no rendering. Just typed dataclasses
   and lookup methods.
2. **`cli.py`** (edit) — imports `THEME`, replaces every hardcoded color
   string and glyph with a theme lookup, and adds the new render logic for
   each polished surface.

`THEME` is constructed at import time. Lookups are O(1) dict reads. The
animation tick (already 10fps via `_start_animation_loop`) is unchanged —
new visuals consume tick events but don't add new ones.

## Components

### 1. `theme.py`

```python
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class Palette:
    primary: str    # gold — "#ffd700"
    secondary: str  # cyan — "#5fd7d7"
    accent: str     # pink — "#ff5fd7"
    dim: str        # grey — "#5f5f5f"
    warn: str       # yellow — "#d7d75f"
    err: str        # red   — "#ff5f5f"
    bg_panel: str   # used for panel backgrounds where applicable

@dataclass(frozen=True)
class RoleStyle:
    color: str          # border/icon color
    icon: str           # single glyph prefix
    spinner_frames: List[str]
    flash_color: str    # one-tick brighter variant for role-change flash

@dataclass(frozen=True)
class ToolKindStyle:
    color: str
    icon: str

@dataclass(frozen=True)
class Theme:
    palette: Palette
    roles: Dict[str, RoleStyle]
    tool_kinds: Dict[str, ToolKindStyle]
    default_role: RoleStyle
    default_tool_kind: ToolKindStyle

    def role(self, name: str) -> RoleStyle:
        return self.roles.get(name, self.default_role)

    def tool_kind(self, tool_name: str) -> ToolKindStyle:
        kind = _TOOL_NAME_TO_KIND.get(tool_name, "default")
        return self.tool_kinds.get(kind, self.default_tool_kind)
```

`_TOOL_NAME_TO_KIND` is a module-level dict in `theme.py`:

```python
_TOOL_NAME_TO_KIND = {
    "read_file": "file", "write_file": "file", "list_dir": "file",
    "generate_codebase_map": "file",
    "run_shell": "shell",
    "web_fetch": "web",
    "remember": "memory", "recall": "memory", "forget": "memory",
}
```

Concrete content:

- **Role bundles** for `executor`, `researcher`, `debugger`, `general`,
  `architect`. Spinner frames per role:
  - `executor`: braille dots `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`
  - `researcher`: sparkles `✦ ✧ ⋆ ✧ ✦`
  - `debugger`: warning pulse `⚠ ⚡ ⚠ ⚡`
  - `architect`: diamonds `◇ ◈ ◆ ◈`
  - `general`: dots `· ·· ··· ····`
- **Tool-kind bundles**: `file` (📄, blue), `shell` (⚡, gold), `web` (🌐,
  green), `memory` (🧠, magenta), `default` (•, dim grey). Tool-name → kind
  mapping covers `read_file`/`write_file`/`list_dir` → file;
  `run_shell` → shell; `web_fetch` → web; `remember`/`recall`/`forget` →
  memory; rest → default.

### 2. `cli.py` changes

Five touchpoints, scoped:

**a. Imports & constants.** Drop the hex hardcoded constants at the top
(`PRIMARY`, `SECONDARY`, etc.) — re-export them as
`PRIMARY = THEME.palette.primary` for backwards compat with any code still
using those names. Drop `_ROLE_COLOR` dict — replaced by `THEME.role(...)`.

**b. Welcome banner (`_get_welcome_panel`).** Replace the plain "EzClaw"
string with a per-character styled `Text` ramping across a fixed
6-color sequence held in `theme.py` as `TITLE_GRADIENT = ["#ffd700",
"#ffaa55", "#ff7755", "#ff5fd7", "#ffaa55", "#ffd700"]` (gold → orange →
red → magenta → orange → gold). Each character takes the next color in the
ramp, wrapping if the string is longer than the ramp. Each agent line
prefixed with `THEME.role(name).icon`. Section separator (`─` × width)
between header and footer.

**c. Working spinner.** The current single `self.spinner = Spinner("dots", ...)`
is replaced by a method `self._spinner_for(role)` that returns a fresh
`Spinner` with the role's frames. The renderable at the bottom of
`_get_current_renderable_ansi` picks the spinner for the *most recent*
routed role. New field `self.current_role` updated when an `intent` chunk
arrives.

**d. Tool execution panels (`_build_tool_panel`).** Title gains the
tool-kind icon. Border style switches from a fixed color to
`THEME.tool_kind(tool['name']).color`. Collapsed panels gain a one-line
summary header: the first non-empty stripped line of `tool['result']`,
truncated to 80 chars, rendered dim italic above the body fold.

**e. Architect chip (`_render_architect_intent`).** Already renders one
line when collapsed; add the role icon prefix from
`THEME.role(agent).icon`. Detect role transition via a new
`self._last_chip_role` field — when the current role differs, render with
`flash_color` for exactly one tick, then reset. Implementation: stash the
new role on render, compare against previous, set a `_chip_flash_until`
timestamp 0.15s in the future, use `flash_color` while now < flash_until.

**f. Status bar (`_get_status_bar`).** Section dividers swap from `·` to
`╱` between major segments. Add an animated activity glyph (rotating
`◐◓◑◒` at 2 frames/sec) on the left when `self.is_generating`. The existing
keybinding-hints tail stays.

### 3. State additions on `ChatUI`

Three new instance fields, initialized in `__init__`:

- `self.current_role: Optional[str] = None` — most recent routed role
- `self._last_chip_role: Optional[str] = None` — previous role for flash detection
- `self._chip_flash_until: float = 0.0` — monotonic timestamp

Updated in the `_agent_worker` chunk-dispatch loop when an `intent` chunk
arrives:

```python
if chunk["type"] == "intent":
    self.architect_intent = chunk
    new_role = chunk.get("agent")
    if new_role and new_role != self._last_chip_role:
        self._chip_flash_until = time.time() + 0.15
        self._last_chip_role = new_role
    self.current_role = new_role
```

### 4. No new dependencies

Pure `rich` + `prompt_toolkit`. `rich.spinner.Spinner` already supports
custom frame lists via `frames=...`. `Text.append(char, style=...)` supports
per-character styling for the gradient title.

## Animation strategy & terminal-honesty

- All visual change is **frame-stepped at 10fps** (the existing
  `_start_animation_loop` tick). No new tick infrastructure.
- "Animations" are discrete glyph/color swaps. The two primitives are:
  - **Spinner cycling**: handled by `rich.spinner.Spinner` internally.
  - **Flash on transition**: one-tick (or ~1-2 frames at 10fps) brighter
    color, then settle. Triggered on role change.
- No constant pulsing of static panels. Idle UI is *still*. Motion signals
  state change.

## Risks

| Risk | Mitigation |
|---|---|
| Unicode glyphs (✦, ⚠, ◇, 📄, etc.) may render as missing boxes in some terminals | Choose glyphs from CJK-safe blocks where possible; provide an ASCII fallback dict the user can swap to in `theme.py` if needed |
| 10fps redraw cap means "flash" can look jerky on slow terminals | Flash duration is 150ms = ~1.5 ticks, which is the minimum reliably visible. Anything shorter would be invisible; longer would feel sluggish |
| `Spinner(frames=...)` may not accept custom frames in older `rich` versions | Verify version compatibility in the implementation step; fall back to `Spinner("dots", style=role_color)` if needed |
| Per-character gradient on welcome title may not align if the user's font has variable-width Unicode | Restrict gradient to ASCII characters (`EzClaw` itself), leave the version suffix plain |

## Testing

This is a visual-only change. Testing is manual:

1. Smoke run: `python cli.py`, send a simple message, confirm:
   - Welcome banner renders with gradient title
   - Working spinner shows role-appropriate frames
   - Tool panels show kind icon & colored border
   - Architect chip shows role icon
2. Multi-step run: send a task that routes across executor → researcher →
   debugger, confirm the chip flashes on each handoff.
3. Terminal compatibility: spot-check in at least one alternate terminal
   (whatever the user has handy) for glyph rendering.
4. Regression: confirm F2 (copy mode) and F3 (strategy panel) still work,
   `/help` / `/diagnose` / `/clear` still render correctly, scroll behavior
   unchanged.

No unit tests — there's no headless way to assert on rich's ANSI output
that would survive future styling tweaks. A snapshot test would lock in
the current visuals and break on every intentional change.

## Files

- **NEW**: `theme.py` (~150 lines)
- **EDIT**: `cli.py` (~80 lines changed across the six touchpoints above)
- **UNCHANGED**: everything else

## Rollout

Single commit. No feature flag. Reversible by `git revert`.
