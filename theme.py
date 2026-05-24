"""Single source of truth for TUI visual constants.

This module is import-time only — no I/O, no rendering. It exports a
singleton `THEME` that `cli.py` reads to find colors, role icons,
spinner frame lists, and tool-kind styling. Adding a new role or
tool kind is a one-file change here.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Palette:
    """Warm coastal palette inspired by the 🦀 mascot — amber carapace,
    sandy greys, sage seaweed, coral red. All colors chosen to be legible
    on a dark terminal background and to coordinate with the role colors
    in `_ROLES` below (which lean cool to contrast the warm base)."""
    primary: str = "#ffb84d"    # warm amber (was #ffd700 gold)
    secondary: str = "#c8c4be"  # warm light grey (was cool #bdbdbd)
    accent: str = "#7fd070"     # sage green (was eye-burning #00ff00)
    dim: str = "#7a7570"        # warm grey-brown (was cool #808080)
    warn: str = "#f5a623"       # honey amber (was #ff8c00)
    err: str = "#e85a5a"        # coral red (was eye-burning #ff0000)


# The 🦀 mascot. Used in the welcome banner and the working spinner.
MASCOT = "🦀"


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
    "code_outline": "file",
    "apply_diff": "file",
    "grep_codebase": "file",
    "run_shell": "shell",
    "run_tests": "shell",
    "python_eval": "shell",
    "git_diff": "shell",
    "git_log": "shell",
    "git_blame": "shell",
    "web_fetch": "web",
    "web_search": "web",
    "remember": "memory",
    "recall": "memory",
    "forget": "memory",
}


# Per-character gradient applied to the welcome banner title. The string
# is mapped to this ramp character-by-character, wrapping if longer.
# Sunset ramp — amber through coral, mirrored back. Coordinates with the
# warm `Palette` above so the title reads as part of the same scheme.
TITLE_GRADIENT: Tuple[str, ...] = (
    "#ffd166",  # honey
    "#ffb84d",  # amber
    "#ff8c5c",  # coral
    "#e85a8a",  # rose
    "#ff8c5c",  # coral
    "#ffb84d",  # amber
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
        color=_PALETTE.primary,    # tracks the palette so the chip + welcome agree
        icon="◈",
        spinner_frames=("◇", "◈", "◆", "◈"),
        flash_color="#ffd9a0",     # brighter variant of primary amber
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


# Per-model emoji used in the status bar and the settings panel. Lookup is
# by prefix on the model family — e.g. "qwen3:14b" and "qwen3.5:9b-q4_K_M"
# both resolve to the qwen entry. Order in this list matters: the FIRST
# matching prefix wins, so put longer / more specific keys before shorter
# ones. Unknown models get the default 🤖 glyph.
#
# Each emoji is chosen to evoke the model family without leaning on the
# vendor's logo character (which doesn't render in most terminals):
#   🐳 deepseek — whale, their open-source brand
#   🧧 qwen     — Alibaba's red envelope (qwen-coder gets 🛠 to distinguish)
#   💎 gemma    — gem
#   🦙 llama    — the obvious one
#   🌬 mistral  — wind, matches the name
#   🤖 gpt-oss  — generic OSS GPT
#   🔬 phi      — Microsoft research line
#   🧮 embed    — embeddings are computation, not chat
#   🐍 codellama / coder-variants → 💾  → coder gets a tool icon
_MODEL_EMOJI_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("mxbai-embed", "🧮"),
    ("nomic-embed", "🧮"),
    ("all-minilm",  "🧮"),
    ("qwen2.5-coder", "🛠"),
    ("qwen3-coder",   "🛠"),
    ("codellama",     "💾"),
    ("starcoder",     "💾"),
    ("deepseek-coder","💾"),
    ("deepseek-r1",   "🐳"),
    ("deepseek",      "🐳"),
    ("phi4",          "🔬"),
    ("phi3",          "🔬"),
    ("phi",           "🔬"),
    ("qwen3.5",       "🧧"),
    ("qwen3",         "🧧"),
    ("qwen2.5",       "🧧"),
    ("qwen2",         "🧧"),
    ("qwen",          "🧧"),
    ("gemma4",        "💎"),
    ("gemma3",        "💎"),
    ("gemma2",        "💎"),
    ("gemma",         "💎"),
    ("llama4",        "🦙"),
    ("llama3",        "🦙"),
    ("llama2",        "🦙"),
    ("llama",         "🦙"),
    ("mistral",       "🌬"),
    ("mixtral",       "🌬"),
    ("gpt-oss",       "🤖"),
    ("gpt-",          "🤖"),
    ("claude",        "🪶"),
)

_DEFAULT_MODEL_EMOJI = "🤖"


def model_emoji(name: str) -> str:
    """Return a single emoji for the given model name.

    Lookup is prefix-based on the model family — "qwen3:14b" returns 🧧,
    "deepseek-r1:14b" returns 🐳, etc. Unknown models return the default
    🤖. Comparison is case-insensitive; tag and quantization suffixes
    (`:14b`, `-q4_K_M`, `:latest`) are ignored because matching happens
    on a prefix of the bare name.
    """
    if not name:
        return _DEFAULT_MODEL_EMOJI
    n = name.lower().strip()
    for prefix, glyph in _MODEL_EMOJI_PREFIXES:
        if n.startswith(prefix):
            return glyph
    return _DEFAULT_MODEL_EMOJI


# Task-state styling for the plan panel. The state key matches Plan task
# statuses (plan.TASK_STATUSES). Each entry is (icon_glyph, color_hex).
TASK_STATE_STYLE: dict = {
    "pending":     ("○", _PALETTE.dim),
    "in_progress": ("▸", "#5fafff"),     # executor blue
    "done":        ("●", "#5fd75f"),     # green
    "failed":      ("✗", _PALETTE.err),
    "skipped":     ("⊘", _PALETTE.dim),
}

# Brighter variant of each state's color, used for the one-tick flash
# when a task transitions between statuses.
TASK_STATE_FLASH: dict = {
    "pending":     _PALETTE.secondary,
    "in_progress": "#afd7ff",            # brighter executor blue
    "done":        "#afffaf",            # brighter green
    "failed":      "#ffafaf",            # brighter red
    "skipped":     _PALETTE.secondary,
}
