# TUI Theme Layer & Animated Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `theme.py` module as the single source of truth for visual constants, and polish four TUI surfaces (welcome banner, working spinner, tool panels, architect chip + status bar) with role-aware visuals and discrete frame-stepped animation.

**Architecture:** Pure UI change. New `theme.py` exports a singleton `THEME` with `Palette`, role bundles, and tool-kind bundles. `cli.py` reads from `THEME` instead of inline hex strings, and gains role-aware spinner instantiation, per-role chip flash on handoff, tool-kind icons and border colors, and status-bar polish. No changes to `multi_agent.py`, `agent.py`, `tools.py`, `model_client.py`, or `memory.py`. No new dependencies.

**Tech Stack:** Python, `rich` (Panel, Text, Spinner, Group, Markdown), `prompt_toolkit` (Application, KeyBindings).

**Spec reference:** `docs/superpowers/specs/2026-05-23-tui-theme-and-animations-design.md`

---

## File Structure

- **Create:** `theme.py` (~150 lines) — palette, role bundles, tool-kind bundles, gradient, singleton `THEME`, helpers
- **Modify:** `cli.py` (~80 lines across 6 touchpoints):
  - Replace inline color constants with re-exports from `THEME`
  - Welcome panel gradient title + role icons
  - Role-aware spinner via `_spinner_for(role)`
  - Tool panel kind icons + colored borders + collapsed-summary header
  - Architect chip role flash on handoff
  - Status bar dividers + animated activity glyph
- **No other files touched.**

---

## Task 1: Create `theme.py`

**Files:**
- Create: `theme.py`
- Test (smoke): import + lookup verification via `python -c`

- [ ] **Step 1: Create the theme module**

Create `theme.py` with the following exact content:

```python
"""Single source of truth for TUI visual constants.

This module is import-time only — no I/O, no rendering. It exports a
singleton `THEME` that `cli.py` reads to find colors, role icons,
spinner frame lists, and tool-kind styling. Adding a new role or
tool kind is a one-file change here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Palette:
    primary: str = "#ffd700"    # gold
    secondary: str = "#bdbdbd"  # grey74
    accent: str = "#00ff00"     # green
    dim: str = "#808080"        # grey50
    warn: str = "#ff8c00"       # dark orange
    err: str = "#ff0000"        # red


@dataclass(frozen=True)
class RoleStyle:
    color: str
    icon: str
    spinner_frames: Tuple[str, ...]
    flash_color: str  # one-tick brighter variant rendered on role change


@dataclass(frozen=True)
class ToolKindStyle:
    color: str
    icon: str


# Maps a concrete tool function name (as registered in tools.py) to a
# tool-kind key. Anything not listed falls back to "default".
_TOOL_NAME_TO_KIND: Dict[str, str] = {
    "read_file": "file",
    "write_file": "file",
    "list_dir": "file",
    "generate_codebase_map": "file",
    "run_shell": "shell",
    "web_fetch": "web",
    "remember": "memory",
    "recall": "memory",
    "forget": "memory",
}


# Per-character gradient applied to the welcome banner title. The string
# is mapped to this ramp character-by-character, wrapping if longer.
TITLE_GRADIENT: Tuple[str, ...] = (
    "#ffd700",  # gold
    "#ffaa55",  # orange
    "#ff7755",  # red-orange
    "#ff5fd7",  # magenta
    "#ffaa55",  # orange
    "#ffd700",  # gold
)


# Activity glyph cycled (1 frame per ~250ms = every 2-3 UI ticks at 10fps)
# in the bottom status bar when generating.
ACTIVITY_FRAMES: Tuple[str, ...] = ("◐", "◓", "◑", "◒")


@dataclass(frozen=True)
class Theme:
    palette: Palette
    roles: Dict[str, RoleStyle]
    tool_kinds: Dict[str, ToolKindStyle]
    default_role: RoleStyle
    default_tool_kind: ToolKindStyle

    def role(self, name: str) -> RoleStyle:
        if not name:
            return self.default_role
        return self.roles.get(name, self.default_role)

    def tool_kind(self, tool_name: str) -> ToolKindStyle:
        kind = _TOOL_NAME_TO_KIND.get(tool_name, "default")
        return self.tool_kinds.get(kind, self.default_tool_kind)


_PALETTE = Palette()

_ROLES: Dict[str, RoleStyle] = {
    "executor": RoleStyle(
        color="#5fafff",
        icon="◇",
        spinner_frames=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
        flash_color="#afd7ff",
    ),
    "researcher": RoleStyle(
        color="#5fd75f",
        icon="✦",
        spinner_frames=("✦", "✧", "⋆", "✧"),
        flash_color="#afffaf",
    ),
    "debugger": RoleStyle(
        color="#ff6f6f",
        icon="⚠",
        spinner_frames=("⚠", "⚡", "⚠", "⚡"),
        flash_color="#ffafaf",
    ),
    "general": RoleStyle(
        color="#d75fd7",
        icon="●",
        spinner_frames=("·", "··", "···", "····"),
        flash_color="#ffafff",
    ),
    "architect": RoleStyle(
        color="#ffd700",
        icon="◈",
        spinner_frames=("◇", "◈", "◆", "◈"),
        flash_color="#ffeb7a",
    ),
}

_DEFAULT_ROLE = RoleStyle(
    color=_PALETTE.dim,
    icon="•",
    spinner_frames=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
    flash_color=_PALETTE.secondary,
)

_TOOL_KINDS: Dict[str, ToolKindStyle] = {
    "file":    ToolKindStyle(color="#5fafff", icon="📄"),
    "shell":   ToolKindStyle(color="#ffd700", icon="⚡"),
    "web":     ToolKindStyle(color="#5fd75f", icon="🌐"),
    "memory":  ToolKindStyle(color="#d75fd7", icon="🧠"),
    "default": ToolKindStyle(color=_PALETTE.dim, icon="•"),
}

THEME = Theme(
    palette=_PALETTE,
    roles=_ROLES,
    tool_kinds=_TOOL_KINDS,
    default_role=_DEFAULT_ROLE,
    default_tool_kind=_TOOL_KINDS["default"],
)
```

- [ ] **Step 2: Verify the module imports cleanly and lookups work**

Run:

```bash
python -c "
from theme import THEME, TITLE_GRADIENT, ACTIVITY_FRAMES
assert THEME.role('executor').icon == '◇'
assert THEME.role('researcher').color == '#5fd75f'
assert THEME.role('nonexistent').icon == '•'
assert THEME.tool_kind('read_file').icon == '📄'
assert THEME.tool_kind('run_shell').color == '#ffd700'
assert THEME.tool_kind('unknown_tool').icon == '•'
assert len(TITLE_GRADIENT) == 6
assert len(ACTIVITY_FRAMES) == 4
print('theme.py OK')
"
```

Expected output: `theme.py OK`

- [ ] **Step 3: Commit**

```bash
git add theme.py
git commit -m "feat(theme): add theme.py with palette, role and tool-kind bundles"
```

---

## Task 2: Wire `cli.py` constants to `THEME`

**Files:**
- Modify: `cli.py:62-67` (replace inline hex constants with re-exports from THEME)
- Modify: `cli.py` (import line)

This is a non-breaking refactor: existing `PRIMARY`, `SECONDARY`, etc. usages continue working, but the source of truth moves to `theme.py`.

- [ ] **Step 1: Add import and replace constants**

Find the existing block at `cli.py:62-67`:

```python
PRIMARY = "#ffd700"  # gold1
SECONDARY = "#bdbdbd" # grey74
ACCENT = "#00ff00"   # green
WARN = "#ff8c00"     # dark_orange
ERR = "#ff0000"      # red
DIM = "#808080"      # grey50
```

Replace with:

```python
from theme import THEME, TITLE_GRADIENT, ACTIVITY_FRAMES

PRIMARY = THEME.palette.primary
SECONDARY = THEME.palette.secondary
ACCENT = THEME.palette.accent
WARN = THEME.palette.warn
ERR = THEME.palette.err
DIM = THEME.palette.dim
```

- [ ] **Step 2: Remove the `_ROLE_COLOR` class attribute**

Find the existing block (around `cli.py:439-445`):

```python
    _ROLE_COLOR = {
        "executor": "#5fafff",   # blue
        "researcher": "#5fd75f", # green
        "debugger":   "#ff6f6f", # red
        "general":    "#d75fd7", # magenta
        "architect":  "#ffd700", # gold (matches PRIMARY)
    }
```

Delete it entirely. The two call sites (`_render_architect_intent` and `_get_welcome_panel`) will be updated in later tasks to use `THEME.role(name).color` directly.

For now, search for `self._ROLE_COLOR.get(` and `self._ROLE_COLOR[`. There are two existing references in `_render_architect_intent` (`role_color = self._ROLE_COLOR.get(agent, SECONDARY)`) and `_get_welcome_panel` (`color = self._ROLE_COLOR.get("architect", PRIMARY)` and `color = self._ROLE_COLOR.get(role, SECONDARY)`).

Replace those three lines:

- In `_render_architect_intent`, change:
  ```python
  role_color = self._ROLE_COLOR.get(agent, SECONDARY)
  ```
  to:
  ```python
  role_color = THEME.role(agent).color
  ```

- In `_get_welcome_panel`, change:
  ```python
  color = self._ROLE_COLOR.get("architect", PRIMARY)
  ```
  to:
  ```python
  color = THEME.role("architect").color
  ```
  and:
  ```python
  color = self._ROLE_COLOR.get(role, SECONDARY)
  ```
  to:
  ```python
  color = THEME.role(role).color
  ```

- [ ] **Step 3: Syntax check**

Run:

```bash
python -c "import ast; ast.parse(open('cli.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Import check**

Run:

```bash
python -c "
import cli
assert cli.PRIMARY == '#ffd700'
assert cli.DIM == '#808080'
print('cli imports OK')
"
```

Expected: `cli imports OK`

- [ ] **Step 5: Commit**

```bash
git add cli.py
git commit -m "refactor(cli): read colors from THEME, drop _ROLE_COLOR dict"
```

---

## Task 3: Welcome banner gradient + role icons

**Files:**
- Modify: `cli.py` — the body of `_get_welcome_panel` (around `cli.py:514-547`)

- [ ] **Step 1: Replace `_get_welcome_panel` body**

Find the method (signature on line 514). Replace the entire method body with:

```python
    def _get_welcome_panel(self):
        body = Text()

        # Per-character gradient on the "EzClaw" title — wraps the ramp
        # if the string is longer than TITLE_GRADIENT.
        title = "EzClaw"
        for i, ch in enumerate(title):
            body.append(ch, style=f"bold {TITLE_GRADIENT[i % len(TITLE_GRADIENT)]}")
        body.append(" ", "")
        body.append("v2.2 (Full TUI)\n", style=f"dim {DIM}")
        body.append("─" * 40 + "\n", style=f"dim {DIM}")

        if ENABLE_MULTI_AGENT and hasattr(self.agent, "agents"):
            body.append("⚡ Multi-agent\n", style=f"bold {PRIMARY}")
            arch_model = getattr(self.agent.architect, "model", "?") if hasattr(self.agent, "architect") else "?"
            arch_style = THEME.role("architect")
            body.append(f"  {arch_style.icon} architect   ", style=f"bold {arch_style.color}")
            body.append(f"{arch_model}\n", style=SECONDARY)
            for role, sub_agent in self.agent.agents.items():
                rs = THEME.role(role)
                body.append(f"  {rs.icon} {role:<9} ", style=f"bold {rs.color}")
                body.append(f"{sub_agent.model}\n", style=SECONDARY)
        else:
            body.append("● Single-agent\n", style=f"bold {PRIMARY}")
            body.append(f"  model       ", style=f"bold {SECONDARY}")
            body.append(f"{self.agent.model}\n", style=SECONDARY)

        body.append("─" * 40 + "\n", style=f"dim {DIM}")
        body.append(f"  workspace   ", style=f"bold {DIM}")
        body.append(f"./workspace\n", style=f"dim {DIM}")
        body.append("\n", "")
        body.append("Commands: ", style="bold")
        body.append("/help  /diagnose  /clear  /thinking  /settings  /authorize  /expand  /collapse",
                    style=f"dim {DIM}")
        return Panel(
            body,
            box=ROUNDED, padding=(1, 2), border_style=DIM,
            title=f"[bold {PRIMARY}]EzClaw[/bold {PRIMARY}]",
        )
```

- [ ] **Step 2: Syntax check**

Run:

```bash
python -c "import ast; ast.parse(open('cli.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Visual smoke test**

Run:

```bash
python -c "
import cli
ui = cli.ChatUI()
panel = ui._get_welcome_panel()
from rich.console import Console
Console().print(panel)
"
```

Expected: A panel renders with "EzClaw" in mixed gold/orange/red/magenta colors, each agent line prefixed with a role icon (◈ for architect, ◇ for executor, ✦ for researcher, ⚠ for debugger, ● for general), and `─` separators above/below the agent block. No tracebacks.

- [ ] **Step 4: Commit**

```bash
git add cli.py
git commit -m "feat(ui): gradient welcome title and role icons on agent list"
```

---

## Task 4: Role-aware spinner

**Files:**
- Modify: `cli.py:133` (drop the global spinner)
- Modify: `cli.py` `__init__` (add `current_role` state)
- Modify: `cli.py` `_get_current_renderable_ansi` (around line 427-428: spinner selection)
- Modify: `cli.py` `_agent_worker` (around line 870: update `current_role` on intent chunk)
- Add: `_spinner_for(role)` helper method

- [ ] **Step 1: Drop the single shared spinner and add state**

Find line 133:

```python
        self.spinner = Spinner("dots", style=f"bold {ACCENT}")
```

Delete that line. Spinners are now created on demand per role.

Find the `welcome_shown` initialization in `__init__` (around line 174-175). Right after it, add:

```python
        # Track the most recently routed role so the spinner and chip
        # can pick role-specific visuals.
        self.current_role: str | None = None
```

- [ ] **Step 2: Add module-level spinner helper**

`rich.spinner.Spinner` accepts a registered name (`"dots"`, `"line"`, etc.) and reads frames from an internal table. To use custom frames per role, we construct a default spinner and override its `frames`/`interval` attributes. Add this helper at the **module top**, immediately after the existing `from rich.spinner import Spinner` import (around line 113):

```python
def _make_custom_spinner(frames, color):
    """Build a Spinner with a custom frame list and 10fps cadence."""
    sp = Spinner(name="dots", text="", style=f"bold {color}")
    sp.frames = list(frames)
    sp.interval = 0.1
    return sp
```

- [ ] **Step 3: Add `_spinner_for(role)` method on `ChatUI`**

Add this method on the `ChatUI` class. A natural place is right above `_get_current_renderable_ansi` (around line 362). Insert:

```python
    def _spinner_for(self, role):
        """Return a Spinner instance using the role's frame list and color."""
        rs = THEME.role(role or "")
        return _make_custom_spinner(rs.spinner_frames, rs.color)
```

- [ ] **Step 4: Use `_spinner_for` in the render path**

Find lines 425-428:

```python
        if self.is_generating:
            elapsed = time.time() - self.generation_start_time
            idle_time = time.time() - self.last_chunk_time
            status_text = f" {self.current_status}  [{elapsed:.1f}s]"
            if idle_time > 15:
                status_text += f"  ⚠ idle {idle_time:.0f}s"
            self.spinner.text = Text(status_text, style=f"bold {ACCENT}")
            parts.append(self.spinner)
```

Replace with:

```python
        if self.is_generating:
            elapsed = time.time() - self.generation_start_time
            idle_time = time.time() - self.last_chunk_time
            rs = THEME.role(self.current_role or "")
            status_text = f" {rs.icon} {self.current_status}  [{elapsed:.1f}s]"
            if idle_time > 15:
                status_text += f"  ⚠ idle {idle_time:.0f}s"
            spinner = self._spinner_for(self.current_role)
            spinner.text = Text(status_text, style=f"bold {rs.color}")
            parts.append(spinner)
```

- [ ] **Step 5: Update `current_role` when an intent chunk arrives**

Find the `_agent_worker` method (around line 860). Within the chunk-dispatch loop, locate:

```python
                if chunk["type"] == "intent":
                    self.architect_intent = chunk
```

Replace with:

```python
                if chunk["type"] == "intent":
                    self.architect_intent = chunk
                    new_role = chunk.get("agent")
                    if new_role:
                        self.current_role = new_role
```

- [ ] **Step 6: Reset `current_role` on each new user turn**

Find the `handle_input` method that resets per-turn state (search for `self.architect_intent = None`). Right after that line, add:

```python
        self.current_role = None
```

- [ ] **Step 7: Syntax check**

```bash
python -c "import ast; ast.parse(open('cli.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 8: Import check + spinner constructor smoke test**

```bash
python -c "
import cli
ui = cli.ChatUI()
sp = ui._spinner_for('executor')
assert sp.frames[0] == '⠋'
sp2 = ui._spinner_for('researcher')
assert sp2.frames[0] == '✦'
sp3 = ui._spinner_for(None)
assert len(sp3.frames) > 0
print('spinner OK')
"
```

Expected: `spinner OK`

- [ ] **Step 9: Commit**

```bash
git add cli.py
git commit -m "feat(ui): role-aware spinner driven by current_role state"
```

---

## Task 5: Tool panel kind icons, colored borders, collapsed summary

**Files:**
- Modify: `cli.py` `_build_tool_panel` (around line 549-612)

- [ ] **Step 1: Update the header construction and border style**

In `_build_tool_panel`, find:

```python
        idx_label = f"[{index}] " if index is not None else ""
        state_label = "  ⇣ expanded" if expanded else ""
        header = Text.assemble(
            ("▸ ", f"bold {WARN}"),
            (idx_label, f"dim {DIM}"),
            (tool_name, "bold"),
            (state_label, f"dim {ACCENT}"),
        )
```

Replace with:

```python
        tool_kind = THEME.tool_kind(tool_name)
        idx_label = f"[{index}] " if index is not None else ""
        state_label = "  ⇣ expanded" if expanded else ""
        header = Text.assemble(
            (f"{tool_kind.icon} ", f"bold {tool_kind.color}"),
            (idx_label, f"dim {DIM}"),
            (tool_name, f"bold {tool_kind.color}"),
            (state_label, f"dim {ACCENT}"),
        )
```

- [ ] **Step 2: Add the collapsed-summary header**

Just before the closing `return Panel(...)` of `_build_tool_panel`, find:

```python
        return Panel(Group(*tool_parts), title=header, border_style=DIM, box=ROUNDED)
```

Replace with:

```python
        # Collapsed panels gain a single-line summary pulled from the first
        # non-empty stripped line of the tool result. Helps scan a long
        # transcript without expanding every panel.
        summary = None
        if not expanded and result:
            for line in str(result).splitlines():
                stripped = line.strip()
                if stripped:
                    summary = stripped[:80] + ("…" if len(stripped) > 80 else "")
                    break
        if summary:
            summary_text = Text(f"  ↳ {summary}", style=f"dim {DIM} italic")
            tool_parts.insert(0, summary_text)

        return Panel(
            Group(*tool_parts),
            title=header,
            border_style=f"dim {tool_kind.color}",
            box=ROUNDED,
        )
```

- [ ] **Step 3: Syntax check**

```bash
python -c "import ast; ast.parse(open('cli.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Smoke-test the panel build**

```bash
python -c "
import cli
ui = cli.ChatUI()
panel = ui._build_tool_panel({
    'name': 'read_file',
    'args': {'path': 'foo.py'},
    'result': 'def add(a, b):\n    return a + b\n',
}, index=1)
from rich.console import Console
Console().print(panel)
"
```

Expected: A panel rendered with `📄 [1] read_file` in blue (file-kind color), a blue border, and a `↳ def add(a, b):` summary line above the args/output blocks. No tracebacks.

- [ ] **Step 5: Commit**

```bash
git add cli.py
git commit -m "feat(ui): tool-kind icons, colored borders, collapsed summary line"
```

---

## Task 6: Architect chip role-flash on handoff

**Files:**
- Modify: `cli.py` `__init__` (add flash state)
- Modify: `cli.py` `handle_input` (reset flash state per turn)
- Modify: `cli.py` `_render_architect_intent` (use flash_color when active)
- Modify: `cli.py` `_agent_worker` (set flash deadline on role change)

- [ ] **Step 1: Add flash state to `__init__`**

Right after the `self.current_role` line added in Task 4 (in `__init__`), add:

```python
        # When the routed role changes, render the chip in flash_color for
        # one tick of ~150ms to signal the handoff. `_chip_flash_until`
        # is a monotonic timestamp; the chip checks `time.time() < x`.
        self._last_chip_role: str | None = None
        self._chip_flash_until: float = 0.0
```

- [ ] **Step 2: Reset flash state per turn**

In `handle_input`, just below the line `self.current_role = None` added in Task 4, add:

```python
        self._last_chip_role = None
        self._chip_flash_until = 0.0
```

- [ ] **Step 3: Update `_agent_worker` to set the flash deadline**

Find the block updated in Task 4 step 5:

```python
                if chunk["type"] == "intent":
                    self.architect_intent = chunk
                    new_role = chunk.get("agent")
                    if new_role:
                        self.current_role = new_role
```

Replace with:

```python
                if chunk["type"] == "intent":
                    self.architect_intent = chunk
                    new_role = chunk.get("agent")
                    if new_role:
                        if self._last_chip_role and new_role != self._last_chip_role:
                            self._chip_flash_until = time.time() + 0.15
                        self._last_chip_role = new_role
                        self.current_role = new_role
```

- [ ] **Step 4: Use `flash_color` in the chip render**

In `_render_architect_intent`, find the compact-chip branch (the `if not self.show_architect:` block). The current code reads:

```python
        if not self.show_architect:
            # Compact: dim one-liner — keeps routing context visible without
            # the screen-eating panels. F3 expands.
            headline = goal or (plan.splitlines()[0] if plan else reasoning) or "thinking…"
            if len(headline) > 110:
                headline = headline[:107] + "…"
            line = Text()
            line.append("→ ", style=f"dim {DIM}")
            line.append(agent, style=f"bold {role_color}")
            line.append(" · ", style=f"dim {DIM}")
            line.append(headline, style=f"italic {DIM}")
            line.append("   [F3] expand", style=f"dim {DIM}")
            return line
```

Replace with:

```python
        if not self.show_architect:
            # Compact: dim one-liner — keeps routing context visible without
            # the screen-eating panels. F3 expands.
            rs = THEME.role(agent)
            flashing = time.time() < self._chip_flash_until
            chip_color = rs.flash_color if flashing else rs.color
            headline = goal or (plan.splitlines()[0] if plan else reasoning) or "thinking…"
            if len(headline) > 110:
                headline = headline[:107] + "…"
            line = Text()
            line.append(f"{rs.icon} ", style=f"bold {chip_color}")
            line.append(agent, style=f"bold {chip_color}")
            line.append(" · ", style=f"dim {DIM}")
            line.append(headline, style=f"italic {DIM}")
            line.append("   [F3] expand", style=f"dim {DIM}")
            return line
```

- [ ] **Step 5: Syntax check**

```bash
python -c "import ast; ast.parse(open('cli.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Smoke-test the chip render**

```bash
python -c "
import cli, time
ui = cli.ChatUI()
ui.show_architect = False
intent = {
    'agent': 'executor',
    'reflection': {'goal': 'Read agent.py and report max_iterations'},
    'reasoning': '',
    'plan': '',
}
chip = ui._render_architect_intent(intent)
from rich.console import Console
Console().print(chip)

# Simulate a role flash and re-render
ui._chip_flash_until = time.time() + 0.5
chip2 = ui._render_architect_intent(intent)
Console().print(chip2)
print('chip OK')
"
```

Expected: Two lines printed — the first in normal executor blue, the second in a brighter blue (flash variant). Followed by `chip OK`.

- [ ] **Step 7: Commit**

```bash
git add cli.py
git commit -m "feat(ui): architect chip flashes on role handoff between steps"
```

---

## Task 7: Status bar dividers + animated activity glyph

**Files:**
- Modify: `cli.py` `_get_status_text` (around line 346-360)

- [ ] **Step 1: Replace `_get_status_text` body**

Find the method (signature on line 346). Replace the entire body with:

```python
    def _get_status_text(self):
        auth_icon = "◉" if self.agent.session_authorized else "○"
        mode = "⚡" if ENABLE_MULTI_AGENT else "●"
        model_info = self.agent.model.split(",")[0][:45] if "," in self.agent.model else self.agent.model[:45]
        msg_count = len(self.agent.messages) if hasattr(self.agent, 'messages') and self.agent.messages else 0

        # Animated activity glyph: cycles through ACTIVITY_FRAMES at ~4Hz
        # while generating. Idle state shows a static dot.
        if self.is_generating:
            frame_idx = int(time.time() * 4) % len(ACTIVITY_FRAMES)
            activity = ACTIVITY_FRAMES[frame_idx]
        else:
            activity = "·"

        live = ""
        if self.is_generating:
            n_tools = len(self.tool_executions)
            elapsed = time.time() - self.generation_start_time if self.generation_start_time else 0
            live = f"  ╱  ⚙ {n_tools} tool{'s' if n_tools != 1 else ''}  ╱  {elapsed:.1f}s"

        copy_badge = "  ╱  ✂ COPY MODE" if not self._mouse_capture else ""
        arch_badge = "  ╱  🧠 STRATEGY" if self.show_architect else ""
        return (
            f"  {activity}  {auth_icon}  {mode} {model_info}  ╱  {msg_count} msgs"
            f"{live}{copy_badge}{arch_badge}  │  [Ctrl+C] Exit  [F2] Copy  [F3] Strategy  [Arrows] Scroll"
        )
```

- [ ] **Step 2: Syntax check**

```bash
python -c "import ast; ast.parse(open('cli.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Smoke-test the status text**

```bash
python -c "
import cli, time
ui = cli.ChatUI()
ui.is_generating = False
print(repr(ui._get_status_text()))
ui.is_generating = True
ui.generation_start_time = time.time() - 4.2
print(repr(ui._get_status_text()))
"
```

Expected: Two printed lines — first idle (with `·` activity glyph and `╱` section dividers), second generating (with one of ◐◓◑◒ as the activity glyph and a tool/elapsed segment).

- [ ] **Step 4: Commit**

```bash
git add cli.py
git commit -m "feat(ui): status bar dividers and animated activity glyph"
```

---

## Task 8: Full TUI smoke test + commit any tweaks

**Files:**
- Possibly modify: `theme.py` (only if a glyph renders as a missing-box in the user's terminal)

- [ ] **Step 1: Launch the TUI and exercise it**

In a separate terminal (NOT the agent's run), launch:

```bash
cd /home/lulu/Projects/ezclaw
python cli.py
```

Verify each surface:

1. **Welcome banner** — "EzClaw" title shows mixed colors across characters (gold/orange/red/magenta). Role icons (`◈`, `◇`, `✦`, `⚠`, `●`) appear next to each agent line. Two `─` separators bracket the agent block.
2. **Status bar** — Bottom bar shows `╱` between segments instead of `·`. When idle, leftmost glyph is `·`; when typing/generating, it cycles through `◐◓◑◒`.
3. **Working spinner** — Send a multi-step prompt (e.g. "read tools.py and list its functions"). While the executor is working, the spinner shows braille dots `⠋⠙⠹…` in blue. If the prompt requires the researcher, it switches to sparkles `✦✧⋆✧` in green.
4. **Tool panels** — Each tool execution panel shows the kind icon (`📄` for read_file, `⚡` for run_shell) in its title, with a border color matching the kind. Collapsed panels show `↳ <first line of output>` above the body.
5. **Architect chip** — Compact chip shows `◇ executor · <goal>`. When the architect routes to a different agent (e.g. handoff from executor to debugger after a failure), the chip momentarily flashes brighter.
6. **F2 and F3 still work** — Toggle copy mode and the strategy panel; both should function as before.

- [ ] **Step 2: If any glyph renders as a missing box (□)**

Edit `theme.py` and swap the offending glyph for an ASCII fallback. For example, if `📄` doesn't render, swap to `>`:

```python
"file": ToolKindStyle(color="#5fafff", icon=">"),
```

Re-run the TUI smoke test. Repeat as needed.

- [ ] **Step 3: If no tweaks needed, no commit. If tweaks made, commit**

```bash
git add theme.py
git commit -m "fix(theme): swap glyphs for terminal compatibility"
```

- [ ] **Step 4: Final verification — full diff**

```bash
git log --oneline origin/main..HEAD
```

Expected: 7 or 8 commits (depending on whether Task 8 needed a tweak):

```
<hash> fix(theme): swap glyphs for terminal compatibility   (only if needed)
<hash> feat(ui): status bar dividers and animated activity glyph
<hash> feat(ui): architect chip flashes on role handoff between steps
<hash> feat(ui): tool-kind icons, colored borders, collapsed summary line
<hash> feat(ui): role-aware spinner driven by current_role state
<hash> feat(ui): gradient welcome title and role icons on agent list
<hash> refactor(cli): read colors from THEME, drop _ROLE_COLOR dict
<hash> feat(theme): add theme.py with palette, role and tool-kind bundles
```

---

## Notes for the implementer

- **No TDD here.** This is a visual change with no headless assertion target. Each task verifies via smoke import + a `python -c` rendering of the changed component. The final task does interactive verification.
- **Commit between every task.** If a later task introduces a regression, you can `git revert` the offending commit without losing the others.
- **Don't fold `theme.py` into `cli.py`.** The whole point of this work is the separation. If you find yourself wanting to add a color directly in `cli.py`, stop and add it to `theme.py` first.
- **Glyph fallbacks are real.** Unicode glyphs in `theme.py` (especially the emoji-family ones like 📄 🌐 🧠) may render differently or be missing in some terminals. Task 8 has the swap procedure.
- **No new dependencies.** If you're tempted to install something, you're off the plan.
