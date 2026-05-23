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
