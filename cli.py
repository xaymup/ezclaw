#!/usr/bin/env python3
import os
import re
import sys
import time
import threading
import asyncio
import subprocess
import io
import queue
from datetime import datetime
from typing import List, Dict, Any, Optional

# Venv check
_venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
if os.path.isdir(_venv) and not sys.prefix.startswith(_venv):
    _python = os.path.join(_venv, "bin", "python3")
    os.execv(_python, [_python] + sys.argv)

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED
from rich.syntax import Syntax
from rich.spinner import Spinner

from prompt_toolkit.application import Application
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.keys import Keys as _PtKeys


# ── Shift+Enter support ────────────────────────────────────────────────────
# By default prompt_toolkit's parser aliases every "Enter-with-modifier"
# escape sequence to Keys.ControlM (same as plain Enter), so the app can't
# tell Enter from Shift+Enter. We rebind the modifier variants to distinct
# Keys.F24 / F23 / F22 (unused function keys) so the keybinding layer
# below can react to them separately. We also rely on the terminal having
# been put into either CSI u or xterm modifyOtherKeys mode so it actually
# sends those byte sequences — `ChatUI.run` enables both modes at startup
# and disables them on exit.
SHIFT_ENTER = _PtKeys.F24      # repurposed unused function key
CTRL_ENTER = _PtKeys.F23
CTRL_SHIFT_ENTER = _PtKeys.F22

# Modified Backspace under kitty CSI u arrives as \x1b[127;<mod>u
# (keycode 127 = DEL) or \x1b[8;<mod>u (keycode 8 = BS). prompt_toolkit
# has no discrete Keys.* for these, so we route them to two synthetic
# keys: WORD_DELETE_BACK for Ctrl/Alt+Backspace (conventional word-
# delete in most editors), and SHIFT_BACKSPACE for Shift+Backspace
# (folded into a regular Backspace so nothing weird happens).
WORD_DELETE_BACK = _PtKeys.F21
SHIFT_BACKSPACE = _PtKeys.ControlH  # = same key as plain Backspace

# Both CSI u (kitty keyboard protocol) and xterm modifyOtherKeys encode
# Enter modifiers as `<bitmask>+1`. Bits:
#   1=Shift  2=Alt  4=Ctrl  8=Super  16=Hyper  32=Meta
#   64=CapsLock  128=NumLock
# We care about the Shift / Ctrl / Shift+Ctrl combos, with any of the
# four lock-key states the user might have on (none / Caps / Num /
# both). Missing a lock-state variant means the parser doesn't match
# the full sequence and inserts the suffix bytes as literal text in
# the prompt — observed real-world: kitty + NumLock sent
# `\x1b[13;130u` (130 = 1 + 128 + 1 = Shift + NumLock + 1) which used
# to land as the literal text `[13;130u` in the input box.
_MOD_SHIFT = 1
_MOD_CTRL = 4
_MOD_VARIANTS = (
    (_MOD_SHIFT,           SHIFT_ENTER),
    (_MOD_CTRL,            CTRL_ENTER),
    (_MOD_SHIFT | _MOD_CTRL, CTRL_SHIFT_ENTER),
)
for _base, _key in _MOD_VARIANTS:
    for _lock in (0, 64, 128, 64 | 128):  # none / Caps / Num / both
        _modval = _base + _lock + 1  # protocol: bitmask + 1
        ANSI_SEQUENCES[f"\x1b[13;{_modval}u"] = _key            # CSI u
        ANSI_SEQUENCES[f"\x1b[27;{_modval};13~"] = _key         # modifyOtherKeys

# Once kitty's CSI u flag is active, Ctrl+letter is no longer sent as
# the legacy ASCII control byte (\x01..\x1a). Kitty instead emits e.g.
# \x1b[99;5u for Ctrl+C, \x1b[99;133u for Ctrl+NumLock+C, etc. Without
# the inverse mapping, prompt_toolkit's `c-c` binding (which listens for
# Keys.ControlC = \x03) doesn't fire — the user observed Ctrl+C stopped
# interrupting the app. Map every Ctrl+letter sequence (both CSI u and
# modifyOtherKeys forms) back to the corresponding Keys.Control<X> so
# the existing bindings keep working.
for _codepoint in range(ord("a"), ord("z") + 1):
    _letter = chr(_codepoint)
    _ctrl_key = getattr(_PtKeys, f"Control{_letter.upper()}")
    for _lock in (0, 64, 128, 64 | 128):
        _modval = _MOD_CTRL + _lock + 1
        ANSI_SEQUENCES[f"\x1b[{_codepoint};{_modval}u"] = _ctrl_key
        ANSI_SEQUENCES[f"\x1b[27;{_modval};{_codepoint}~"] = _ctrl_key
        # Ctrl+Shift+letter — prompt_toolkit doesn't ship discrete
        # Keys.ControlShift<X> entries, but the byte sequence has to
        # map to SOMETHING so it doesn't leak as literal text into the
        # input. Re-use the plain Ctrl+letter key — the user pressing
        # Ctrl+Shift+Z gets treated as Ctrl+Z, which is what nearly all
        # CLI apps do anyway.
        _modval_shift = _MOD_SHIFT | _MOD_CTRL + _lock + 1  # Shift+Ctrl + lock + 1
        # Above is wrong precedence — explicit:
        _modval_shift = (_MOD_SHIFT | _MOD_CTRL) + _lock + 1
        ANSI_SEQUENCES[f"\x1b[{_codepoint};{_modval_shift}u"] = _ctrl_key
        ANSI_SEQUENCES[f"\x1b[27;{_modval_shift};{_codepoint}~"] = _ctrl_key

# Same shape as the Ctrl+letter problem, but for modified arrow keys.
# Without NumLock prompt_toolkit knows e.g. \x1b[1;5D (Ctrl+Left), but
# with NumLock kitty sends \x1b[1;133D which wasn't mapped — pressing
# Ctrl+Left inserted "[1;133D" as literal text into the prompt instead
# of moving the cursor word-left. Map all the lock-key permutations.
#
# Arrow direction letters: A=Up, B=Down, C=Right, D=Left.
# Modifier combos worth mapping: Shift / Ctrl / Shift+Ctrl.
_ARROW_KEYS = {
    # (modifier_bits, direction_letter)  → Keys.<Modifier><Direction>
    (_MOD_SHIFT, "A"):           _PtKeys.ShiftUp,
    (_MOD_SHIFT, "B"):           _PtKeys.ShiftDown,
    (_MOD_SHIFT, "C"):           _PtKeys.ShiftRight,
    (_MOD_SHIFT, "D"):           _PtKeys.ShiftLeft,
    (_MOD_CTRL, "A"):            _PtKeys.ControlUp,
    (_MOD_CTRL, "B"):            _PtKeys.ControlDown,
    (_MOD_CTRL, "C"):            _PtKeys.ControlRight,
    (_MOD_CTRL, "D"):            _PtKeys.ControlLeft,
    (_MOD_SHIFT | _MOD_CTRL, "A"): _PtKeys.ControlShiftUp,
    (_MOD_SHIFT | _MOD_CTRL, "B"): _PtKeys.ControlShiftDown,
    (_MOD_SHIFT | _MOD_CTRL, "C"): _PtKeys.ControlShiftRight,
    (_MOD_SHIFT | _MOD_CTRL, "D"): _PtKeys.ControlShiftLeft,
}
for (_base, _letter), _key in _ARROW_KEYS.items():
    for _lock in (0, 64, 128, 64 | 128):
        _modval = _base + _lock + 1
        ANSI_SEQUENCES[f"\x1b[1;{_modval}{_letter}"] = _key

# Modified Backspace under kitty CSI u. Maps all
# (Shift / Alt / Ctrl / Shift+Ctrl / Shift+Alt / Ctrl+Alt) × 4 lock
# states × 2 keycodes (127=DEL, 8=BS) to the right synthetic key.
# Shift+Backspace folds to a regular Backspace; Ctrl/Alt/Ctrl+Alt
# variants go to WORD_DELETE_BACK so word-delete works.
_MOD_ALT = 2
_BACKSPACE_MOD_VARIANTS = (
    (_MOD_SHIFT,                      SHIFT_BACKSPACE),
    (_MOD_ALT,                        WORD_DELETE_BACK),
    (_MOD_CTRL,                       WORD_DELETE_BACK),
    (_MOD_SHIFT | _MOD_ALT,           WORD_DELETE_BACK),
    (_MOD_SHIFT | _MOD_CTRL,          WORD_DELETE_BACK),
    (_MOD_ALT | _MOD_CTRL,            WORD_DELETE_BACK),
    (_MOD_SHIFT | _MOD_ALT | _MOD_CTRL, WORD_DELETE_BACK),
)
for _base, _key in _BACKSPACE_MOD_VARIANTS:
    for _lock in (0, 64, 128, 64 | 128):
        _modval = _base + _lock + 1
        for _kc in (127, 8):
            ANSI_SEQUENCES[f"\x1b[{_kc};{_modval}u"] = _key
            ANSI_SEQUENCES[f"\x1b[27;{_modval};{_kc}~"] = _key
# Alt+Backspace via the classic Esc-prefix is also a common encoding
# (most terminals when "Option as Esc+" is on). Bind that too.
ANSI_SEQUENCES["\x1b\x7f"] = WORD_DELETE_BACK
ANSI_SEQUENCES["\x1b\x08"] = WORD_DELETE_BACK

# Plain (unmodified) navigation keys with lock-state variants.
# When kitty has modifyOtherKeys on and the user has NumLock or
# CapsLock down, kitty includes the lock bit in the modifier value
# EVEN FOR UNMODIFIED KEYS. So plain Left arrow becomes \x1b[1;129D
# (NumLock = 128, +1 = 129) instead of legacy \x1b[D. prompt_toolkit
# doesn't ship the lock-only variants, so they were getting split as
# unknown CSI and dropped by the filter — arrows stopped moving.
# Map every "modifier = lock-only" combination back to the plain key.
_PLAIN_ARROW_LIKE = {
    # letter → Keys.<name>  (no-modifier nav keys)
    "A": _PtKeys.Up,
    "B": _PtKeys.Down,
    "C": _PtKeys.Right,
    "D": _PtKeys.Left,
    "H": _PtKeys.Home,
    "F": _PtKeys.End,
}
_PLAIN_TILDE_NAV = {
    # numeric prefix → Keys.<name>  (no-modifier nav keys, ~ terminator)
    "2": _PtKeys.Insert,
    "3": _PtKeys.Delete,
    "5": _PtKeys.PageUp,
    "6": _PtKeys.PageDown,
    # F-keys 11..24 use the same ~ pattern in some terminals
    "15": getattr(_PtKeys, "F5", None),
    "17": getattr(_PtKeys, "F6", None),
    "18": getattr(_PtKeys, "F7", None),
    "19": getattr(_PtKeys, "F8", None),
    "20": getattr(_PtKeys, "F9", None),
    "21": getattr(_PtKeys, "F10", None),
    "23": getattr(_PtKeys, "F11", None),
    "24": getattr(_PtKeys, "F12", None),
}
# "Modifier = lock-only" values: 1 (no mod, but kitty may include this
# spurious 1), 65 (Caps), 129 (Num), 193 (both).
for _modval in (1, 65, 129, 193):
    for _letter, _key in _PLAIN_ARROW_LIKE.items():
        ANSI_SEQUENCES[f"\x1b[1;{_modval}{_letter}"] = _key
    for _prefix, _key in _PLAIN_TILDE_NAV.items():
        if _key is None:
            continue
        ANSI_SEQUENCES[f"\x1b[{_prefix};{_modval}~"] = _key

# Home (`H`) and End (`F`) under modifiers + locks — same pattern.
# Bare modified Home/End: prompt_toolkit ships ShiftHome / ControlHome
# etc. Lock variants need explicit mapping.
_HOME_END_KEYS = {
    (_MOD_SHIFT, "H"):           getattr(_PtKeys, "ShiftHome", None),
    (_MOD_SHIFT, "F"):           getattr(_PtKeys, "ShiftEnd", None),
    (_MOD_CTRL, "H"):            getattr(_PtKeys, "ControlHome", None),
    (_MOD_CTRL, "F"):            getattr(_PtKeys, "ControlEnd", None),
}
for (_base, _letter), _key in _HOME_END_KEYS.items():
    if _key is None:
        continue
    for _lock in (0, 64, 128, 64 | 128):
        _modval = _base + _lock + 1
        ANSI_SEQUENCES[f"\x1b[1;{_modval}{_letter}"] = _key

# Catch-all that DOESN'T need any terminal protocol: a fair number of
# terminals send a literal CR+LF (\r\n) for Shift+Enter and bare \r for
# plain Enter. Without an entry for \r\n in ANSI_SEQUENCES the parser
# fires Keys.ControlM (submit) immediately on the \r and then the \n
# is processed against an empty buffer — so Shift+Enter visibly
# submits. Adding \r\n as a known sequence makes the parser wait for
# the \n before firing, which we then route to SHIFT_ENTER. Plain
# Enter still sends a bare \r (no \n behind it) so submit still fires
# on the next flush.
ANSI_SEQUENCES["\r\n"] = SHIFT_ENTER
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.widgets import Frame, TextArea
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.application.current import get_app
from prompt_toolkit.history import FileHistory
from prompt_toolkit.filters import Condition
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.keys import Keys

class SimpleAnsiLexer(Lexer):
    def lex_document(self, document):
        # Cache line fragments to speed up rendering
        lines = document.text.splitlines()
        def get_line(i):
            try:
                # We need to use ANSI() on each line to preserve formatting
                return to_formatted_text(ANSI(lines[i]))
            except IndexError:
                return []
        return get_line
from agent import ChatAgent
from multi_agent import MultiAgentSystem
from dotenv import load_dotenv

load_dotenv()

# ── Theme (opencode-inspired) ───────────────────────────────────
from theme import THEME, TITLE_GRADIENT, ACTIVITY_FRAMES, MASCOT, model_emoji

PRIMARY = THEME.palette.primary
SECONDARY = THEME.palette.secondary
ACCENT = THEME.palette.accent
WARN = THEME.palette.warn
ERR = THEME.palette.err
DIM = THEME.palette.dim

# Global console for rendering
console = Console(file=io.StringIO(), force_terminal=True, width=100)

def _fmt_duration(seconds: float) -> str:
    """Human-readable duration that doesn't blow out at long elapsed
    times. Used by the spinner status, the status-bar cost group, and
    the per-turn cook-time annotation so they all read the same way.
        12.3 → "12.3s"
        125  → "2m 5s"
        4500 → "1h 15m"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


# Full ECMA-48 CSI sequence: ESC `[`, optional parameter bytes (0x30-0x3f),
# optional intermediate bytes (0x20-0x2f), then a final byte (0x40-0x7e).
# SGR is the final byte 'm' (0x6d) — kept so colored output still renders;
# everything else (cursor movement, screen clear, alt-screen entry like
# `\x1b[?1049h`) is dropped before subprocess output reaches the TUI.
# Without this, vim/top/htop/less inside the interactive pane would
# clear or scramble the host terminal.
_CSI_RE = re.compile(r'\x1b\[[\x30-\x3f]*[\x20-\x2f]*([\x40-\x7e])')
# OSC sequences (window title, hyperlinks) and simple ESC designators.
_OSC_RE = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)')
_SIMPLE_ESC_RE = re.compile(r'\x1b[78cMEH=>()]')

def _sanitize_subprocess_text(text: str) -> str:
    """Strip cursor-positioning escapes and other dangerous control bytes
    from raw subprocess output before we render it inside the TUI. Keeps
    SGR color codes (so colored `ls`/`git` output still looks right) but
    drops anything that would move the cursor, clear the screen, or enter
    an alt-screen — all of which would otherwise leak through `ANSI()` and
    clobber prompt_toolkit's layout."""
    text = _CSI_RE.sub(lambda m: m.group(0) if m.group(1) == 'm' else '', text)
    text = _OSC_RE.sub('', text)
    text = _SIMPLE_ESC_RE.sub('', text)
    return text.replace('\r', '').replace('\b', '')


def render_to_ansi(renderable) -> str:
    with io.StringIO() as f:
        c = Console(file=f, force_terminal=True, width=100)
        c.print(renderable)
        return f.getvalue()


# Global state
SHOW_THINKING = os.getenv("SHOW_THINKING", "true").lower() == "true"
ENABLE_MULTI_AGENT = os.getenv("ENABLE_MULTI_AGENT", "false").lower() == "true"

def run_diagnostics_raw():
    output = []
    output.append(f"[bold {SECONDARY}]Running System Diagnostics...[/bold {SECONDARY}]")
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader", shell=True, text=True)
        output.append(Panel(gpu_info.strip(), title=f"[bold {ACCENT}]NVIDIA GPU Status[/bold {ACCENT}]", border_style=ACCENT))
    except:
        output.append(f"[{WARN}]NVIDIA GPU (nvidia-smi) not found or failed.[/{WARN}]")
    try:
        ollama_ps = subprocess.check_output("ollama ps", shell=True, text=True)
        output.append(Panel(ollama_ps.strip() or "No models currently loaded in memory.", title=f"[bold {SECONDARY}]Ollama Active Models[/bold {SECONDARY}]", border_style=SECONDARY))
    except:
        output.append(f"[{WARN}]Ollama CLI (ollama ps) not found or failed.[/{WARN}]")
    return Group(*output)

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.syntax import Syntax
from rich.spinner import Spinner
from rich.console import Group
from rich.box import ROUNDED


def _make_custom_spinner(frames, color):
    """Build a Spinner with a custom frame list and 10fps cadence."""
    sp = Spinner(name="dots", text="", style=f"bold {color}")
    sp.frames = list(frames)
    sp.interval = 100  # milliseconds — rich divides by 1000 internally
    return sp

class ChatUI:
    def __init__(self):
        self.agent = MultiAgentSystem() if ENABLE_MULTI_AGENT else ChatAgent()
        self._wrap_tools()
        self.history_ansi = []
        self.current_response_parts = []
        self.reasoning_chunks = []
        # self.reasoning_log initialized further down (line ~268)
        self.tool_executions = []
        self.side_messages = []
        self.halted = False
        self.is_generating = False
        self.auth_queue = queue.Queue()
        self.auth_active = False
        self.current_auth_chunk = None
        self._auth_requests = queue.Queue()
        
        # Persistent rich objects for animations
        self.r_console = Console(file=io.StringIO(), force_terminal=True, width=100)

        # Generation timing & health tracking
        self.generation_start_time = 0.0
        self.last_chunk_time = 0.0

        # Session-wide accounting (Feature 2 — status bar tokens + energy).
        # Tokens are a char/4 estimate, not exact counts from Ollama — the
        # status bar surfaces them with a `~` prefix so the imprecision is
        # honest. Energy is wall-clock generation seconds × GPU TDP, an
        # over-counting Fermi estimate good enough for "this turn was
        # expensive" intuition but not for billing.
        self.session_tokens_in = 0
        self.session_tokens_out = 0
        self.session_energy_wh = 0.0
        # Default to RTX 4080 (320W) — matches README's hardware section.
        try:
            self.gpu_tdp_watts = float(os.environ.get("EZCLAW_GPU_TDP_W", "320"))
        except (TypeError, ValueError):
            self.gpu_tdp_watts = 320.0
        # Most recently completed turn's cook time (seconds). Used for the
        # "· 12.3s" annotation under the assistant bubble (Feature 1) when
        # the streamed turn has settled into history.
        self.last_cook_time = 0.0

        # ── Reflection line (Feature: ambient wisdom) ─────────────────
        # A short, dim italic one-liner shown between the status bar and
        # the input field. Refreshed every REFLECTION_REFRESH_SEC from
        # the architect LLM with a "give me one short aphorism / fun
        # fact / insight" prompt, optionally grounded in user-memory
        # facts. The user can force a refresh with `/wisdom`.
        self.reflection: str = ""
        self.reflection_at: float = 0.0
        self.reflection_refresh_sec: float = float(
            os.environ.get("EZCLAW_REFLECTION_REFRESH_SEC", "900")
        )
        # Set to True while a background thread is mid-fetch so we don't
        # launch a second one on top of it.
        self._reflection_inflight: bool = False

        # ── Paste-as-block ────────────────────────────────────────────
        # Large pastes are collapsed into a single-line placeholder
        # ("[pasted #N: L lines, C chars]") so the input area stays
        # compact. The real content lives in `_paste_blocks` keyed by
        # the exact placeholder string; the Enter handler expands every
        # known placeholder back to its original content before
        # submitting. Cleared at submit / /clear.
        self._paste_blocks: dict = {}
        self._paste_seq: int = 0
        # Pastes smaller than these thresholds are inserted verbatim —
        # collapsing a 2-line paste would feel surprising.
        self._paste_collapse_lines: int = 3
        self._paste_collapse_chars: int = 200

        self.history_file = os.path.expanduser("~/.ezclaw_history")
        self.prompt_history = FileHistory(self.history_file)

        # Native buffer for history.
        # `_history_read_only` is a depth counter (not a boolean) so the
        # read-only state is correctly maintained across overlapping
        # _update_ui calls from the worker thread and the animation tick.
        self._history_read_only = 1  # >0 means read-only
        self._update_lock = threading.RLock()
        self.history_buffer = Buffer(read_only=Condition(lambda: self._history_read_only > 0))

        self.kb = KeyBindings()

        self._setup_keybindings()

        self.layout = self._create_layout()
        # Warm-coastal palette for the bottom status bar — replaces the
        # default white-background eyesore from `reverse #ffffff`.
        # Deep warm brown bg, honey-amber text. Coordinates with the
        # 🦀 mascot palette and the Palette dataclass in theme.py.
        self.style = Style.from_dict({
            'status': 'bg:#1f160e #ffd166',
            'status.badge.copy': 'bg:#1f160e #ff8c5c bold',
            'status.badge.strategy': 'bg:#1f160e #ff5fd7 bold',
            'status.badge.tools': 'bg:#1f160e #7fd070 bold',
            'status.activity': 'bg:#1f160e bold',
            'status.divider': 'bg:#1f160e #7a7570',
            'status.keys': 'bg:#1f160e #c8c4be',
            'prompt': f'bold {PRIMARY}',
            'frame.border': f'{DIM}',
        })

        # Mouse capture: OFF by default so click-drag selection works
        # in the terminal natively (the user can copy chat output with
        # their terminal's normal Ctrl+Shift+C / Cmd+C / right-click).
        # F2 toggles it ON to re-enable our ScrollUp/Down bindings
        # (PgUp/PgDn / Home/End / Up/Down arrows still scroll regardless
        # of this setting, so most users never need to flip it).
        # Defaulting OFF: most users want to copy, far fewer rely on
        # mouse-wheel scrolling — and the keyboard scroll keys cover
        # that case anyway.
        self._mouse_capture = False
        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=True,
            mouse_support=Condition(lambda: self._mouse_capture),
        )

        # ── feed_multiple wrapper ──────────────────────────────────────
        # Wraps the input loop's batch path with TWO behaviors:
        #
        # 1. ALWAYS: filter out the "split unknown CSI sequence" pattern.
        #    When prompt_toolkit's Vt100Parser sees a sequence it doesn't
        #    recognize (e.g. \x1b[3;199~), it gives up and delivers each
        #    character as a separate KeyPress: Keys.Escape, then '[',
        #    '3', ';', '1', '9', '9', '~'. The non-escape chars then get
        #    inserted into the input buffer as literal text — which is
        #    why unmapped modified keys (DEL, F-keys, etc.) leaked
        #    garbage in the prompt. We detect that pattern at the
        #    feed_multiple boundary and drop the whole split sequence.
        #
        # 2. OPTIONAL (EZCLAW_KEYDEBUG=1): log every keypress's resolved
        #    Key + raw byte data to /tmp/ezclaw_keys.log for diagnostics.
        _original_feed_multiple = self.app.key_processor.feed_multiple
        from prompt_toolkit.keys import Keys as _PtKeys2
        _NOISE_KEYS = {_PtKeys2.Vt100MouseEvent, _PtKeys2.CPRResponse}
        _KEYDEBUG = bool(os.environ.get("EZCLAW_KEYDEBUG"))

        # CSI terminator: a letter (A-Z, a-z) or '~'. Stops scanning the
        # split body so we know where the orphaned sequence ends.
        def _is_csi_terminator(ch: str) -> bool:
            return len(ch) == 1 and (ch.isalpha() or ch == "~")

        def _filter_split_csi(key_presses):
            """Drop runs that look like [Keys.Escape, '[', <body>, terminator].
            Leaves recognized single-KeyPress sequences (Keys.Left, etc.)
            untouched because they aren't split."""
            out = []
            i = 0
            n = len(key_presses)
            while i < n:
                kp = key_presses[i]
                if kp.key == _PtKeys2.Escape and i + 1 < n:
                    nxt = key_presses[i + 1]
                    if isinstance(nxt.data, str) and nxt.data == "[":
                        # Find the terminator (CSI runs end on a letter or '~').
                        j = i + 2
                        while j < n and not _is_csi_terminator(key_presses[j].data):
                            j += 1
                        if j < n:
                            # Drop [i .. j] (inclusive of terminator).
                            i = j + 1
                            continue
                        # No terminator yet — keep the Esc, let the parser
                        # accumulate more bytes next read.
                out.append(kp)
                i += 1
            return out

        def _wrapped_feed_multiple(key_presses, first=False):
            filtered = _filter_split_csi(key_presses)
            if _KEYDEBUG:
                try:
                    with open("/tmp/ezclaw_keys.log", "a") as _f:
                        if len(filtered) != len(key_presses):
                            _f.write(
                                f"{time.strftime('%H:%M:%S')}  dropped {len(key_presses) - len(filtered)} bytes from unknown CSI\n"
                            )
                        for kp in filtered:
                            if kp.key in _NOISE_KEYS:
                                continue
                            _f.write(
                                f"{time.strftime('%H:%M:%S')}  key={kp.key!r:<22}  data={kp.data!r}\n"
                            )
                except Exception:
                    pass
            return _original_feed_multiple(filtered, first=first)

        self.app.key_processor.feed_multiple = _wrapped_feed_multiple
        if _KEYDEBUG:
            try:
                with open("/tmp/ezclaw_keys.log", "a") as _f:
                    _f.write(f"\n--- ezclaw session start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            except Exception:
                pass

        self.welcome_shown = False

        # Track the most recently routed role so the spinner and chip
        # can pick role-specific visuals.
        self.current_role: str | None = None

        # When the routed role changes, render the chip in flash_color for
        # one tick of ~150ms to signal the handoff. `_chip_flash_until`
        # is a monotonic timestamp; the chip checks `time.time() < x`.
        self._last_chip_role: str | None = None
        self._chip_flash_until: float = 0.0

        # Active plan from the most recent run. None means "no plan, hide panel".
        self.current_plan = None
        # Track previous statuses so we can detect transitions and flash the row.
        self._last_task_states: dict = {}
        # task_id → monotonic timestamp at which the flash window expires.
        self._task_flash_until: dict = {}

        # Cache one Spinner per role — rich's Spinner derives its current
        # frame from (now - start_time) / interval, so re-creating a fresh
        # spinner each tick would always reset start_time and freeze the
        # animation on frame 0. Cache key is the role string.
        self._spinner_cache: dict = {}

        # Architect strategy panels: hidden by default — they dominated the
        # screen with mostly-redundant information. The compact chip below
        # still shows the routing decision; F3 toggles the full panels back.
        self.show_architect = False

        # Tool panels: compact (inline args in title, no args sub-panel,
        # no summary line) by default. F4 toggles to the full layout that
        # shows args in their own panel and the redundant ↳ summary line.
        self.compact_tools = True

        # Feature 3 — right-side editor pane with tabs.
        #
        # `_open_files` maps {path -> {content, opened_at, revealed_chars}}.
        # Every write_file or apply_diff that touches a path opens (or
        # updates) the corresponding entry. The pane shows the active
        # file's content with a tab row at the top listing every open
        # file; F6/F7 cycles, F5 toggles the whole pane.
        #
        # Reveal animation: on first appearance, `revealed_chars` starts
        # at 0 and the animation tick walks it up to len(content) over
        # ~1.5s so the file appears to be typed in by the agent. Once
        # fully revealed, subsequent updates (apply_diff growing the
        # file) snap to the new content — the typing illusion only fires
        # the first time you see the file.
        self._show_editor: bool = True
        self._open_files: dict = {}
        self._active_editor_path: str | None = None
        # Legacy alias — some callers still reference this; we expose a
        # property below so existing code paths keep working until they
        # migrate.

        # When True, the next _update_ui will auto-scroll to bottom even if
        # the user has scrolled up. Set on new-prompt submit and on End-key
        # press so the user always sees their own message + the start of
        # the agent's reply.
        self._force_scroll_next_update = False

        # Active embedded interactive shell session, or None. While set,
        # user input typed into the prompt is forwarded to the subprocess
        # via the session's input_queue instead of being sent to the agent.
        self._interactive_session = None

        # Tracks the last status string we sent a desktop notification for
        # so identical consecutive statuses don't fire twice.
        self._last_notified_status = None

        # Resolved display name for the user bubble. Cached so the memory
        # lookup doesn't run on every turn — invalidated whenever the
        # agent stores a new memory (a `remember` tool call could be
        # depositing a name).
        self.user_name = self._resolve_user_name()

        # ask_user back-channel: when the agent calls the ask_user tool,
        # _pending_user_question holds the question text and the worker
        # thread blocks on _user_answer_queue.get() until handle_input
        # forwards the user's typed reply.
        self._pending_user_question: str | None = None
        self._user_answer_queue: "queue.Queue[str]" = queue.Queue()

        # Skill-learning offer: after a turn ends with signs that the
        # user educated us, the architect drafts a skill and yields a
        # `skill_offer` chunk. The draft sits here until the user
        # presses Y (save) or N (skip).
        self._pending_skill_offer: dict | None = None

        # ── Unified reasoning log (Tier 2.1) ───────────────────────────
        # All reasoning-style content flows into this single log,
        # rendered by _render_reasoning_panel as ONE collapsible panel.
        # Sources:
        #   - architect intent reflections (goal / observation /
        #     critical_thinking, one row per architect step)
        #   - sub-agent <think> chunks (`reasoning` chunks)
        #   - skill match notices
        #   - self-check verdicts
        #
        # Each entry: {kind, label, body, time}.
        # Toggle: Ctrl+R or /reasoning. Default state follows the
        # SHOW_THINKING env var so existing user config is honored.
        self.reasoning_log: list[dict] = []
        self.show_reasoning: bool = SHOW_THINKING

        self._wire_tool_callbacks()

    def _wrap_tools(self):
        from tools import registry
        original_run_shell = registry.tools.get('run_shell')
        if not original_run_shell:
            return

        def wrapped_run_shell(command: str, interactive: bool = False) -> str:
            if not interactive:
                return original_run_shell(command, interactive=False)

            # ── UI-embedded interactive shell ─────────────────────────────
            # Subprocess output streams into a live tool panel inside the
            # chat. User input typed into the prompt is routed to the
            # subprocess via input_queue. The TUI is never suspended.
            input_q: "queue.Queue[bytes]" = queue.Queue()
            live_buffer: list[bytes] = []

            # Find the tool execution dict that was just appended for this
            # call (the agent emitted tool_start → cli appended an entry).
            # We tag it with our streaming buffer so the panel render can
            # show output as it arrives.
            tool_dict = None
            for t in reversed(self.tool_executions):
                if t.get("name") == "run_shell" and t.get("result") is None:
                    tool_dict = t
                    tool_dict["_live_buffer"] = live_buffer
                    tool_dict["interactive"] = True
                    tool_dict["expanded"] = True  # show output live
                    break

            self._interactive_session = {
                "input_queue": input_q,
                "tool": tool_dict,
                "command": command,
                "last_output_at": time.time(),
                "notified_waiting": False,
                # The subprocess handle, filled in by on_proc_spawn once
                # _run_shell_interactive starts it. Ctrl+C and SIGINT
                # forwarding use this.
                "proc": None,
                # Input routing toggle. False → keystrokes go to the
                # subprocess via input_queue (default for new interactive
                # sessions). True → keystrokes go to ezclaw's chat handler
                # (slash commands work; plain messages get a hint).
                # Esc toggles this while a session is active.
                "chat_mode": False,
            }

            def on_proc_spawn(proc):
                if self._interactive_session is not None:
                    self._interactive_session["proc"] = proc

            def on_output(data: bytes):
                # Cap the in-memory buffer so a runaway subprocess can't
                # balloon UI state. Keep the last ~64 KB.
                live_buffer.append(data)
                total = sum(len(c) for c in live_buffer)
                while total > 64 * 1024 and len(live_buffer) > 1:
                    total -= len(live_buffer.pop(0))
                # Refresh the "last output" timestamp so the panel knows
                # the subprocess is still talking — clears any "waiting"
                # cue from the prior idle window.
                if self._interactive_session is not None:
                    self._interactive_session["last_output_at"] = time.time()
                    self._interactive_session["notified_waiting"] = False
                try:
                    self._update_ui()
                except Exception:
                    pass

            def input_provider():
                try:
                    val = input_q.get_nowait()
                except queue.Empty:
                    return None
                # User just typed — reset the timer so the panel reverts
                # to "live" while the subprocess processes the input.
                if self._interactive_session is not None:
                    self._interactive_session["last_output_at"] = time.time()
                    self._interactive_session["notified_waiting"] = False
                return val

            try:
                from tools import (
                    _run_shell_interactive, _build_sandbox_env,
                    _ensure_workspace_self_symlink, WORKSPACE_DIR,
                )
                workspace_cwd = os.path.abspath(WORKSPACE_DIR)
                os.makedirs(workspace_cwd, exist_ok=True)
                _ensure_workspace_self_symlink(workspace_cwd)
                result = _run_shell_interactive(
                    command,
                    workspace_cwd=workspace_cwd,
                    sandbox_env=_build_sandbox_env(),
                    on_output=on_output,
                    input_provider=input_provider,
                    on_proc_spawn=on_proc_spawn,
                )
            except Exception as exc:
                # Defensive: any failure during the embedded interactive
                # run gets caught here so the TUI never crashes. The agent
                # sees a clear error string as the tool result.
                result = (
                    f"[Interactive shell session — crashed]\n"
                    f"Command: {command}\n"
                    f"Error: {type(exc).__name__}: {exc}\n"
                )
            finally:
                self._interactive_session = None
                try:
                    self._update_ui()
                except Exception:
                    pass

            return result

        registry.tools['run_shell'] = wrapped_run_shell

    def _wire_tool_callbacks(self):
        """Register the runtime back-channels for ask_user and critique.

        The tool functions themselves live in tools.py and call into
        these callbacks when invoked by an agent. We register them once
        at startup so they pick up changes to ChatUI state automatically.
        """
        import tools as _tools
        _tools.set_ask_user_callback(self._on_ask_user_tool)
        _tools.set_critique_callback(self._on_critique_tool)

    def _on_ask_user_tool(self, question: str) -> str:
        """Block the agent worker until the user types an answer.

        Runs on the agent worker thread — NEVER call from the UI thread,
        or the UI thread will deadlock waiting for itself.
        """
        # Drain any stale answer from a prior race
        try:
            while True:
                self._user_answer_queue.get_nowait()
        except queue.Empty:
            pass

        self._pending_user_question = question
        self._update_ui()

        # Surface a desktop notification so the user can come back from
        # another window — same pattern as the auth prompt.
        try:
            from notifications import notify, URGENCY_NORMAL
            notify(
                "🦀 ezclaw has a question",
                question[:140],
                urgency=URGENCY_NORMAL,
            )
        except Exception:
            pass

        try:
            answer = self._user_answer_queue.get()
        finally:
            self._pending_user_question = None
            self._update_ui()
        return answer or ""

    def _on_critique_tool(self, draft: str, context: str) -> str:
        """Run a quick adversarial critique using the architect's model.

        Returns a short bulleted critique. Best-effort — on any failure
        we return a string the tool wraps as the result so the calling
        agent never sees a raw exception.
        """
        if not ENABLE_MULTI_AGENT:
            return "[critique requires multi-agent mode]"
        arch = getattr(self.agent, "architect", None)
        if arch is None or arch.client is None:
            return "[critique unavailable — no architect client]"

        system_msg = (
            "You are a careful adversarial reviewer. Read the draft and "
            "identify weaknesses, missing edge cases, factual errors, and "
            "unstated assumptions. Be specific and concrete. Return 3-6 "
            "short bullet points, each starting with one of: MISSING, "
            "WRONG, RISKY, UNSTATED. If the draft looks solid, say so "
            "in one bullet and stop."
        )
        user_msg = f"DRAFT:\n{draft}\n"
        if context:
            user_msg += f"\nCONTEXT THE CRITIC SHOULD KNOW:\n{context}\n"

        try:
            resp = arch.client.chat(
                model=arch.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                stream=False,
                options={"temperature": 0.4, "num_predict": 600},
            )
            text = (resp.get("message", {}) or {}).get("content", "")
            return text.strip() or "[critique returned no content]"
        except Exception as e:
            return f"[critique failed: {type(e).__name__}: {e}]"

    def _accept_skill_offer(self):
        """Save the parked skill draft via the learn_skill tool. Shows
        the result in the chat history and reloads the in-memory skill
        list so the next turn can match against it."""
        draft = self._pending_skill_offer
        self._pending_skill_offer = None
        if not draft:
            self._update_ui()
            return
        try:
            from tools import registry
            tool = registry.tools.get("learn_skill")
            if tool is None:
                self.history_ansi.append(render_to_ansi(
                    Text("Error: learn_skill tool not registered.", style=ERR)
                ))
                self._update_ui()
                return
            result = tool(
                name=draft.get("name", "unnamed"),
                description=draft.get("description", ""),
                procedure=draft.get("procedure", ""),
            )
            self.history_ansi.append(render_to_ansi(
                Text(f"💡 {result}", style=f"bold {ACCENT}")
            ))
            # Reload skills so the next turn can match against the new
            # entry without restarting ezclaw.
            try:
                from agent import load_skills
                if hasattr(self.agent, "skills"):
                    self.agent.skills = load_skills()
            except Exception:
                pass
        except Exception as e:
            self.history_ansi.append(render_to_ansi(
                Text(f"Error saving skill: {e}", style=ERR)
            ))
        self._update_ui()

    def _decline_skill_offer(self):
        """Dismiss the parked skill draft without saving."""
        self._pending_skill_offer = None
        self.history_ansi.append(render_to_ansi(
            Text("Skipped saving the skill candidate.", style=DIM)
        ))
        self._update_ui()

    def _setup_keybindings(self):
        @self.kb.add('c-c')
        def _(event):
            """Ctrl+C is repurposed away from "exit ezclaw":
               - Interactive shell active → forward SIGINT to the subprocess
                 process group (Ctrl+C in their shell, the way you'd expect)
               - Generating → cancel the current run (sets is_generating
                 False; the worker observes it on the next chunk)
               - Idle → no-op with a side hint. Exit is via the literal
                 'exit' / 'quit' commands.
            """
            session = self._interactive_session
            if session is not None and session.get("proc") is not None:
                import signal
                proc = session["proc"]
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                self.side_messages.append("↯ SIGINT sent to subprocess")
                self._update_ui()
                return
            if self.is_generating:
                self.is_generating = False
                self.side_messages.append("↯ generation cancelled")
                self._update_ui()
                return
            if self.halted:
                # User cancelled during halt — finalize the partial panel
                # so the cancel doesn't leave a phantom open block.
                final_renderable = self._get_current_renderable_ansi()
                self.history_ansi.append(final_renderable)
                self.current_response_parts = []
                self.halted = False
                self._update_ui()
                return
            self.side_messages.append("↯ press is harmless — type 'exit' to leave ezclaw")
            self._update_ui()

        @self.kb.add('escape', eager=True)
        def _(event):
            """Toggle input routing during an interactive session.

            When a subprocess is live, the default is to send keystrokes
            to it via input_queue. Press Esc to flip into "chat mode":
            keystrokes go to ezclaw's normal chat handler (slash commands
            work; plain messages get a hint that the agent is busy).
            Press Esc again to flip back to subprocess.

            Outside of an interactive session, Esc is a no-op.
            """
            session = self._interactive_session
            if session is None:
                return
            session["chat_mode"] = not session.get("chat_mode", False)
            mode = "ezclaw chat" if session["chat_mode"] else "subprocess input"
            self.side_messages.append(f"↻ input now routed to: {mode}")
            self._update_ui()
            event.app.invalidate()

        @self.kb.add('tab')
        def _(event):
            event.app.layout.focus_next()

        @self.kb.add('f1')
        def _(event):
            # Instant help — same as typing /help but doesn't disturb a
            # prompt the user may already be composing.
            self._show_help()
            self._update_ui()
            event.app.invalidate()

        @self.kb.add('f2')
        def _(event):
            # Toggle terminal-native text selection. With mouse capture off,
            # click-drag selects text the way the terminal natively expects,
            # so the user can copy. Press F2 again to re-enable scroll bindings.
            self._mouse_capture = not self._mouse_capture
            event.app.invalidate()

        @self.kb.add('f3')
        def _(event):
            self.show_architect = not self.show_architect
            self._update_ui()
            event.app.invalidate()

        @self.kb.add('f4')
        def _(event):
            self.compact_tools = not self.compact_tools
            self._update_ui()
            event.app.invalidate()

        # Feature 3: F5 toggles the right-side editor panel (which
        # auto-shows whenever the agent calls write_file while toggled on).
        @self.kb.add('f5')
        def _(event):
            self._show_editor = not self._show_editor
            self._update_ui()
            event.app.invalidate()

        # F6 / F7 cycle through open file tabs in the editor pane.
        @self.kb.add('f6')
        def _(event):
            self._cycle_editor_tab(-1)
            self._update_ui()
            event.app.invalidate()

        @self.kb.add('f7')
        def _(event):
            self._cycle_editor_tab(+1)
            self._update_ui()
            event.app.invalidate()

        # Editor pane scrolling. Alt+Up / Alt+Down scroll by one line;
        # Alt+PageUp / Alt+PageDown scroll by ten. Only fires when the
        # editor pane is currently rendered (toggle on + an open file).
        # Reaches editor_window.vertical_scroll directly — the same
        # mechanism the chat scrollback uses for its history window.
        _editor_visible = Condition(
            lambda: self._show_editor and bool(self._open_files)
        )

        @self.kb.add('escape', 'up', filter=_editor_visible)
        def _editor_scroll_up(event):
            self.editor_window.vertical_scroll = max(
                0, self.editor_window.vertical_scroll - 1
            )
            event.app.invalidate()

        @self.kb.add('escape', 'down', filter=_editor_visible)
        def _editor_scroll_down(event):
            self.editor_window.vertical_scroll += 1
            event.app.invalidate()

        @self.kb.add('escape', 'pageup', filter=_editor_visible)
        def _editor_pageup(event):
            self.editor_window.vertical_scroll = max(
                0, self.editor_window.vertical_scroll - 10
            )
            event.app.invalidate()

        @self.kb.add('escape', 'pagedown', filter=_editor_visible)
        def _editor_pagedown(event):
            self.editor_window.vertical_scroll += 10
            event.app.invalidate()

        # Reset scroll to top whenever the user cycles tabs or opens a
        # new file — feels more natural than landing in the middle of
        # the new file's content at the previous file's scroll offset.
        # (The actual reset happens in _cycle_editor_tab and
        # _open_or_update_file via direct vertical_scroll = 0 calls.)

        # Authorization keys (active only while an auth_required panel
        # is up — see self.auth_active). Semantics:
        #   Y  remember THIS tool for the rest of the session (most common
        #      user intent — previously this was one-shot, which surprised
        #      users who pressed Y and were immediately re-prompted for
        #      the same tool the next turn).
        #   O  allow ONCE (true one-shot — useful when you want to inspect
        #      one call but stay strict on the rest).
        #   A  allow ALL tools session-wide.
        #   N  deny this call (no state change).
        #
        # Both lowercase AND uppercase bound for each — the panel shows
        # `[Y]` `[O]` `[N]` `[A]` in uppercase, and users naturally press
        # Shift+letter (especially with CapsLock on or on small phone
        # keyboards). Without the uppercase binding, those keypresses
        # produced no action and the user saw the panel re-appear on
        # the next tool call because no auth had actually been granted.
        _auth_filter = Condition(lambda: self.auth_active)

        def _auth(answer: str):
            def _h(event):
                self.auth_queue.put(answer)
            return _h

        for _k in ("y", "Y"):
            self.kb.add(_k, filter=_auth_filter)(_auth("allow_tool"))
        for _k in ("o", "O"):
            self.kb.add(_k, filter=_auth_filter)(_auth("allow"))
        for _k in ("n", "N"):
            self.kb.add(_k, filter=_auth_filter)(_auth("deny"))
        for _k in ("a", "A"):
            self.kb.add(_k, filter=_auth_filter)(_auth("allow_session"))

        # Reasoning panel toggle (Tier 2.1) — Ctrl+R flips
        # show_reasoning. /reasoning command does the same via _handle_command.
        @self.kb.add('c-r')
        def _(event):
            self.show_reasoning = not self.show_reasoning
            self._update_ui()

        # Ctrl/Alt+Backspace → delete previous word. The synthetic key
        # WORD_DELETE_BACK (F21) catches every modifier+lock permutation
        # mapped at module top. Implementation walks left across
        # trailing whitespace, then across non-whitespace, and deletes
        # the span — same shape as readline's `backward-kill-word`.
        @self.kb.add(WORD_DELETE_BACK, filter=Condition(lambda: not self.auth_active))
        def _delete_word_back(event):
            buf = event.current_buffer
            text = buf.text
            pos = buf.cursor_position
            if pos == 0:
                return
            # Skip trailing whitespace immediately left of cursor.
            i = pos
            while i > 0 and text[i - 1].isspace():
                i -= 1
            # Then skip the word itself (non-whitespace).
            while i > 0 and not text[i - 1].isspace():
                i -= 1
            if i < pos:
                buf.text = text[:i] + text[pos:]
                buf.cursor_position = i

        # Swallow Ctrl+Z so it doesn't leak its raw byte data into the
        # input buffer. Observed in kitty + NumLock: pressing Ctrl+Z
        # resolved to Keys.ControlZ but no binding fired, and
        # prompt_toolkit's default fallback surfaced the literal
        # sequence in the prompt as garbled characters. Explicit no-op
        # consumes the event cleanly. Other Ctrl+letter keys with
        # legitimate conventional meanings (Ctrl+L clear screen,
        # Ctrl+P / Ctrl+N history, Ctrl+W word-delete, etc.) are left
        # alone so prompt_toolkit's default TextArea bindings apply.
        @self.kb.add('c-z', filter=Condition(lambda: not self.auth_active))
        def _(event):
            pass

        # (The Keys.Any catch-all that used to live here was removed —
        # it took precedence over the Buffer's default cursor-movement
        # bindings, so plain arrow keys stopped working. The actual fix
        # for unmapped-CSI-sequence leaks lives in the feed_multiple
        # wrapper in __init__, which filters out the entire
        # "Keys.Escape + split per-char CSI body" pattern before the
        # KeyProcessor sees it. That fixes the leak at parse time
        # without intercepting any recognized key.)

        # Skill-offer Y/N — active only when a draft is parked and no
        # auth prompt is competing for the same keys.
        _skill_offer_pending = Condition(
            lambda: self._pending_skill_offer is not None and not self.auth_active
        )

        # Both cases bound, same reason as the auth keys above —
        # Shift+Y / CapsLock'd Y otherwise produced no action.
        for _k in ("y", "Y"):
            self.kb.add(_k, filter=_skill_offer_pending)(lambda e: self._accept_skill_offer())
        for _k in ("n", "N"):
            self.kb.add(_k, filter=_skill_offer_pending)(lambda e: self._decline_skill_offer())

        @self.kb.add('enter', filter=Condition(lambda: not self.auth_active))
        def _(event):
            # Submit on plain Enter UNLESS the buffer ends with a
            # backslash — in which case the user signaled "this Enter
            # is a newline, not a submit" with the trailing `\`. We
            # remove the backslash and insert `\n` instead. This is
            # the always-works escape for terminals that can't send
            # Shift+Enter distinctly (macOS Terminal.app, default
            # GNOME Terminal, etc.) — every terminal in existence can
            # type a `\` followed by Enter.
            buf = self.input_field.buffer
            text = buf.text
            if text.endswith("\\"):
                buf.text = text[:-1] + "\n"
                buf.cursor_position = len(buf.text)
                return
            stripped = text.strip()
            if stripped:
                # Expand paste placeholders back to their original
                # content before submitting. A placeholder that the
                # user deleted partially won't match exactly and stays
                # as literal text in the submission — acceptable
                # trade-off; the user can re-paste if they wanted the
                # block back.
                if self._paste_blocks:
                    for placeholder, content in list(self._paste_blocks.items()):
                        if placeholder in stripped:
                            stripped = stripped.replace(placeholder, content)
                    # Turn boundary — drop the cached blocks so a future
                    # paste with the same #N doesn't collide.
                    self._paste_blocks.clear()
                # Add to history (the placeholder form, not the expanded
                # one — keeps Up-arrow recall compact).
                buf.append_to_history()
                self.input_field.text = ""
                self.handle_input(stripped)

        # Newline insertion for the multiline input.
        #
        # Plan: Enter submits, Shift+Enter inserts a newline. Most
        # terminals send the same byte for Shift+Enter as for Enter
        # (both = `\r`), so the application layer can't tell them apart
        # at the byte stream level. To make Shift+Enter work the way
        # users expect (and the way Claude Code's CLI does it), we:
        #   1. Enable two keyboard protocols at app startup (see
        #      `_enable_extended_keyboard_modes` called from `run()`):
        #      xterm modifyOtherKeys level 2 and kitty CSI u. With
        #      either active, the terminal sends a distinct escape
        #      sequence for Shift+Enter instead of plain `\r`.
        #   2. Override the ANSI_SEQUENCES table (top of this module)
        #      so prompt_toolkit's parser routes those distinct
        #      sequences to a synthetic `SHIFT_ENTER` key (F24) instead
        #      of aliasing them back to ControlM.
        #   3. Bind SHIFT_ENTER below to insert `\n`.
        #
        # Fallbacks for terminals that DON'T support either protocol:
        #   - `c-j` (Ctrl+J)         — the literal `\n` byte; works
        #                              universally.
        #   - `escape, enter`        — prompt_toolkit's documented
        #                              alt-enter convention; also fires
        #                              when many terminals send the
        #                              Esc+Enter pair for Alt+Enter.
        # Plain insert-newline handler. The earlier diagnostic version
        # logged WHICH key fired to /tmp/ezclaw_keys.log — that's now
        # gated on EZCLAW_KEYDEBUG above (via the feed_multiple wrap),
        # which captures everything anyway, so this stays clean.
        def _make_newline_handler(label: str):
            def _h(event):
                event.current_buffer.insert_text("\n")
            return _h

        # Shift+Enter — works on terminals that honor either kitty
        # CSI u (\x1b[13;2u) or xterm modifyOtherKeys level 2
        # (\x1b[27;2;13~). Those sequences are rerouted to Keys.F24 at
        # module top, and _enable_extended_keyboard_modes asks the
        # terminal to emit them.
        self.kb.add(SHIFT_ENTER, filter=Condition(lambda: not self.auth_active))(
            _make_newline_handler("Shift+Enter (Keys.F24)")
        )
        # Ctrl+Enter — same idea, distinct rerouting (Keys.F23).
        self.kb.add(CTRL_ENTER, filter=Condition(lambda: not self.auth_active))(
            _make_newline_handler("Ctrl+Enter (Keys.F23)")
        )
        # Universal fallbacks: Ctrl+J (the literal newline byte) and
        # Esc-then-Enter / Alt+Enter — both work in every terminal,
        # including ones that don't support the extended protocols.
        self.kb.add('c-j', filter=Condition(lambda: not self.auth_active))(
            _make_newline_handler("Ctrl+J")
        )
        self.kb.add('escape', 'enter', filter=Condition(lambda: not self.auth_active))(
            _make_newline_handler("Esc,Enter / Alt+Enter")
        )

        # Bracketed paste: collapse large blobs into a single-line
        # placeholder. prompt_toolkit's bracketed-paste handler ships
        # the full pasted body as `event.data`. Without this, pasting
        # 100 lines into the prompt would expand the input box to its
        # full height; users typically want the SEND action, not to
        # edit the pasted content. Small pastes (<3 lines, <200 chars)
        # are inserted verbatim so a one-line code snippet still feels
        # natural.
        @self.kb.add(Keys.BracketedPaste, filter=Condition(lambda: not self.auth_active))
        def _on_paste(event):
            pasted = event.data or ""
            line_count = pasted.count("\n") + 1 if pasted else 0
            char_count = len(pasted)
            if (
                line_count < self._paste_collapse_lines
                and char_count < self._paste_collapse_chars
            ):
                event.current_buffer.insert_text(pasted)
                return
            self._paste_seq += 1
            placeholder = (
                f"[pasted #{self._paste_seq}: "
                f"{line_count} line{'s' if line_count != 1 else ''}, "
                f"{char_count} char{'s' if char_count != 1 else ''}]"
            )
            self._paste_blocks[placeholder] = pasted
            event.current_buffer.insert_text(placeholder)

        # Up / Down for the multiline input.
        #
        # TextArea(history=...) defaults to binding Up/Down to history
        # navigation buffer-wide, so even with multiple lines composed
        # the cursor couldn't move between them. Bind explicit
        # app-level handlers that use the canonical prompt_toolkit
        # Buffer methods:
        #   - cursor_up()       — move cursor up one row (or no-op at top)
        #   - cursor_down()     — move cursor down one row (or no-op at bottom)
        #   - history_backward() / history_forward() at the line boundary
        #
        # No focus filter: app-level bindings already only fire when the
        # app is taking input, and getting the wrong buffer is fine
        # because event.current_buffer points to the focused one.
        # Up / Down in the input field:
        #   - if there's a line above/below, move the cursor between
        #     lines (real multi-line editing)
        #   - at the boundary, fall through to scrolling the CHAT
        #     HISTORY window — NOT prompt history. Calling
        #     history_backward() here used to nuke the user's typed
        #     buffer with a recalled past prompt, which felt broken.
        #
        # If you actually want prompt-history recall, prompt_toolkit's
        # default buffer bindings keep Ctrl+P (previous) and Ctrl+N
        # (next) available — those don't interfere with the
        # natural-feeling chat-scroll behavior on Up/Down.
        _kd = bool(os.environ.get("EZCLAW_KEYDEBUG"))

        def _arrow_log(direction, buf):
            if not _kd:
                return
            try:
                with open("/tmp/ezclaw_keys.log", "a") as _f:
                    _f.write(
                        f"{time.strftime('%H:%M:%S')}  arrow {direction}: "
                        f"row={buf.document.cursor_position_row} "
                        f"col={buf.document.cursor_position_col} "
                        f"line_count={buf.document.line_count} "
                        f"text={buf.text[:60]!r}\n"
                    )
            except Exception:
                pass

        @self.kb.add('up', filter=Condition(lambda: not self.auth_active), eager=True)
        def _(event):
            buf = event.current_buffer
            _arrow_log("UP", buf)
            if buf.document.cursor_position_row > 0:
                buf.cursor_up()
            else:
                # Boundary: scroll chat history up by moving the history
                # buffer's cursor. See the ScrollUp comment for why we
                # can't just set vertical_scroll.
                self.history_buffer.cursor_up(count=3)
                event.app.invalidate()

        @self.kb.add('down', filter=Condition(lambda: not self.auth_active), eager=True)
        def _(event):
            buf = event.current_buffer
            _arrow_log("DOWN", buf)
            if buf.document.cursor_position_row < buf.document.line_count - 1:
                buf.cursor_down()
            else:
                # Boundary: scroll chat history down. See ScrollUp comment.
                self.history_buffer.cursor_down(count=3)
                event.app.invalidate()

        # Helpers for treating `[pasted #N: ...]` placeholders as atomic
        # units when navigating / deleting. Each helper scans the live
        # placeholder set, so they're cheap when there are no pastes
        # active and only do real work in the (rare) turns where one is.
        def _placeholder_ending_at(text, pos):
            """Return placeholder string if `text[:pos]` ends with one."""
            if not self._paste_blocks:
                return None
            for placeholder in self._paste_blocks:
                if text[:pos].endswith(placeholder):
                    return placeholder
            return None

        def _placeholder_starting_at(text, pos):
            """Return placeholder string if `text[pos:]` starts with one."""
            if not self._paste_blocks:
                return None
            for placeholder in self._paste_blocks:
                if text[pos:].startswith(placeholder):
                    return placeholder
            return None

        # Left / Right wrap at \n boundaries AND skip placeholders as
        # one atomic unit. Default Keys.Left stopped at column 0; we
        # also need to teach it about placeholders so the cursor never
        # lands inside `[pasted #N: ...]`.
        @self.kb.add('left', filter=Condition(lambda: not self.auth_active), eager=True)
        def _(event):
            buf = event.current_buffer
            _arrow_log("LEFT", buf)
            text = buf.text
            pos = buf.cursor_position
            if pos <= 0:
                return
            ph = _placeholder_ending_at(text, pos)
            if ph is not None:
                # Cursor is right after a placeholder — jump to its start.
                buf.cursor_position = pos - len(ph)
                return
            buf.cursor_position = pos - 1

        @self.kb.add('right', filter=Condition(lambda: not self.auth_active), eager=True)
        def _(event):
            buf = event.current_buffer
            _arrow_log("RIGHT", buf)
            text = buf.text
            pos = buf.cursor_position
            if pos >= len(text):
                return
            ph = _placeholder_starting_at(text, pos)
            if ph is not None:
                # Cursor is right before a placeholder — jump past it.
                buf.cursor_position = pos + len(ph)
                return
            buf.cursor_position = pos + 1

        # Smart Backspace: at the end of a paste placeholder, one
        # Backspace deletes the WHOLE placeholder AND drops its entry
        # from `_paste_blocks` (so the collapsed paste is fully
        # un-staged from the eventual submit). Otherwise normal
        # single-char delete.
        @self.kb.add('c-h', filter=Condition(lambda: not self.auth_active), eager=True)
        def _smart_backspace(event):
            buf = event.current_buffer
            text = buf.text
            pos = buf.cursor_position
            if pos <= 0:
                return
            ph = _placeholder_ending_at(text, pos)
            if ph is not None:
                start = pos - len(ph)
                buf.text = text[:start] + text[pos:]
                buf.cursor_position = start
                self._paste_blocks.pop(ph, None)
                return
            buf.text = text[:pos - 1] + text[pos:]
            buf.cursor_position = pos - 1

        # Smart Delete (forward-delete). At the start of a paste
        # placeholder, one DEL deletes the whole placeholder + drops
        # its entry. Otherwise normal single-char forward delete.
        @self.kb.add('delete', filter=Condition(lambda: not self.auth_active), eager=True)
        def _smart_delete(event):
            buf = event.current_buffer
            text = buf.text
            pos = buf.cursor_position
            if pos >= len(text):
                return
            ph = _placeholder_starting_at(text, pos)
            if ph is not None:
                end = pos + len(ph)
                buf.text = text[:pos] + text[end:]
                self._paste_blocks.pop(ph, None)
                return
            buf.text = text[:pos] + text[pos + 1:]

        # Chat-history scrolling. We move `history_buffer.cursor_position`
        # (via cursor_up/cursor_down) rather than poking the Window's
        # `vertical_scroll` directly — prompt_toolkit clamps vertical_scroll
        # on every render to keep the cursor visible, so writing it without
        # also moving the cursor snaps right back on the next animation
        # tick. To the user that looks like "scrolling does nothing."
        @self.kb.add(Keys.ScrollUp)
        def _(event):
            self.history_buffer.cursor_up(count=3)
            event.app.invalidate()

        @self.kb.add(Keys.ScrollDown)
        def _(event):
            self.history_buffer.cursor_down(count=3)
            event.app.invalidate()

        @self.kb.add('end')
        def _(event):
            # Jump to bottom and re-enable auto-scroll.
            self._force_scroll_next_update = True
            self._scroll_to_bottom()
            event.app.invalidate()

        @self.kb.add('home')
        def _(event):
            # Jump to top of history.
            self.history_buffer.cursor_position = 0
            event.app.invalidate()

        @self.kb.add('pageup')
        def _(event):
            info = self.history_window.render_info
            jump = info.window_height - 1 if info else 10
            self.history_buffer.cursor_up(count=jump)
            event.app.invalidate()

        @self.kb.add('pagedown')
        def _(event):
            info = self.history_window.render_info
            jump = info.window_height - 1 if info else 10
            self.history_buffer.cursor_down(count=jump)
            event.app.invalidate()

    def _create_layout(self):
        self.history_control = BufferControl(
            buffer=self.history_buffer,
            lexer=SimpleAnsiLexer(),
            focusable=True,
        )
        self.history_window = Window(
            content=self.history_control,
            right_margins=[ScrollbarMargin(display_arrows=True)],
            wrap_lines=True,
        )

        self.status_control = FormattedTextControl(self._get_status_text)
        self.status_window = Window(content=self.status_control, height=1, style='class:status')

        # Multiline TextArea — required for the prompt to handle pasted
        # multi-line content as a single block instead of firing one
        # submit per pasted newline (the old `multiline=False` interpreted
        # every embedded `\n` as a press of Enter, so a 5-line paste
        # produced 5 garbled half-turns).
        #
        # The `enter` keybinding below still submits the buffer; Shift+
        # Enter, Ctrl+J, and Esc-then-Enter insert a literal newline so
        # the user can compose multi-line prompts. The visible height
        # grows with content (1 → 8 lines) so short prompts stay
        # one-line while pastes / multi-line composition expand naturally.
        self.input_field = TextArea(
            height=D(min=1, max=8, preferred=1),
            prompt="ezclaw > ",
            multiline=True,
            wrap_lines=True,
            scrollbar=True,
            history=self.prompt_history,
        )

        # Feature 3: right-side editor panel. Lazy ANSI render driven by
        # `_get_editor_text`. Only visible when `_show_editor` AND at
        # least one file is open — keeps the chat full-width when the
        # agent isn't writing code.
        self.editor_control = FormattedTextControl(self._get_editor_text)
        # Make the editor pane scrollable: a ScrollbarMargin lets the
        # user see scroll position, and `vertical_scroll` on the window
        # can be adjusted via the Alt+Up / Alt+Down bindings below.
        # Stored on self so the binding handlers can reach it.
        self.editor_window = Window(
            content=self.editor_control,
            wrap_lines=False,
            width=64,
            right_margins=[ScrollbarMargin(display_arrows=True)],
        )
        editor_frame = Frame(self.editor_window, title="Editor")
        editor_container = ConditionalContainer(
            content=editor_frame,
            filter=Condition(lambda: self._show_editor and bool(self._open_files)),
        )

        chat_frame = Frame(self.history_window, title="EzClaw Chat")
        chat_plus_editor = VSplit([chat_frame, editor_container])

        # Dedicated interactive-shell pane. Renders the subprocess's live
        # stdout outside the chat so the user can see what's happening at
        # full width without scrolling through the tool log. The pane is
        # shown ONLY while `_interactive_session` is active — when the
        # subprocess exits, the ConditionalContainer collapses and the
        # chat returns to full height. Keystrokes already route to the
        # subprocess via existing chat_mode handling (Esc toggles); the
        # status bar's KEYS→SUBPROC badge tells the user where keys go.
        self.shell_pane_control = FormattedTextControl(self._get_shell_pane_text)
        shell_pane_window = Window(
            content=self.shell_pane_control,
            wrap_lines=False,
            height=12,
        )
        shell_pane_frame = Frame(shell_pane_window, title="Interactive Shell")
        shell_pane_container = ConditionalContainer(
            content=shell_pane_frame,
            filter=Condition(lambda: self._interactive_session is not None),
        )

        # One-line hint below the input frame telling the user how to
        # insert a newline. Shift+Enter only works on terminals that
        # support kitty CSI u or xterm modifyOtherKeys — most don't.
        # Three escape paths in order of universality:
        #   1. `\` + Enter  → ALWAYS works (the Enter handler treats a
        #                     trailing backslash as "insert newline,
        #                     don't submit"). Pure-software, terminal-
        #                     independent.
        #   2. Ctrl+J       → ALWAYS works (it's the literal newline byte).
        #   3. Alt+Enter    → works when the terminal sends Alt as Esc-
        #                     prefix (most Linux terminals, iTerm2 in
        #                     "Esc+" mode).
        self._newline_hint_control = FormattedTextControl(
            lambda: [("class:status #7a7570 italic",
                      "  ⏎ Enter: send   ·   newline:  Ctrl+J   or   Alt+Enter   or   end line with \\ + Enter  ")]
        )
        newline_hint_window = Window(
            content=self._newline_hint_control,
            height=1,
            style="class:status",
        )

        return Layout(
            HSplit([
                chat_plus_editor,
                shell_pane_container,
                self.status_window,
                Frame(self.input_field, height=D(min=3, max=10, preferred=3)),
                newline_hint_window,
            ]),
            focused_element=self.input_field
        )

    def _maybe_refresh_reflection(self, force: bool = False) -> None:
        """Kick off a background reflection fetch if it's due.

        `force=True` (from `/wisdom`) bypasses the time gate. We launch
        in a daemon thread so the UI never blocks on the LLM call.
        Re-entry is guarded by `_reflection_inflight` so a fast user
        spamming /wisdom doesn't pile up requests.
        """
        if self._reflection_inflight:
            return
        if not force and (time.time() - self.reflection_at < self.reflection_refresh_sec):
            return
        self._reflection_inflight = True
        threading.Thread(target=self._fetch_reflection, daemon=True).start()

    def _fetch_reflection(self) -> None:
        """Call the architect LLM for one short reflection.

        Best-effort: any exception, timeout, or unparseable response is
        swallowed and the previous reflection stays on screen. Pulls a
        few recent memory facts (if any) so the model can ground the
        line in something the user has shared.
        """
        try:
            arch = getattr(self.agent, "architect", None)
            client = getattr(arch, "client", None) or getattr(self.agent, "client", None)
            model = getattr(arch, "model", None) or getattr(self.agent, "model", None)
            if client is None or not model:
                return

            # Pull up to 5 short memory facts for grounding. Best-effort.
            memory_facts: list = []
            try:
                facts = self.agent.db.search_memories_hybrid("recent personal facts", alpha=0.6)
                if isinstance(facts, list):
                    memory_facts = [str(f)[:200] for f in facts[:5]]
            except Exception:
                memory_facts = []

            user_label = self.user_name or "the user"
            # Randomize the prompt's "kind" so consecutive refreshes feel
            # varied rather than producing the same Hallmark-card register
            # every time. The previous prompt drifted toward sappy
            # aphorisms ("Wisdom is X, courage is Y") — explicitly steer
            # AWAY from that here.
            import random as _r
            kinds = [
                "a movie quote (short, recognizable, attribute briefly if you want)",
                "an obscure fun fact you'd actually tell someone at a bar",
                "a single-sentence weird piece of trivia",
                "a witty programmer / hacker observation",
                "a one-line dry joke",
                "a memorable line from a song",
                "a tiny piece of dev folklore",
                "a brief odd observation about computers, terminals, or code",
                "a vivid one-line image (not a metaphor about life)",
            ]
            kind = _r.choice(kinds)
            prompt = (
                "Write ONE short single-line caption for a developer's terminal "
                f"status bar. Make it: {kind}.\n\n"
                "Hard constraints:\n"
                "- Under 60 characters total. Brevity matters more than depth.\n"
                "- No greeting, no name, no preface like 'here is' / 'reflection:'.\n"
                "- No surrounding quotes around the whole line.\n"
                "- NEVER produce sappy aphorisms like 'wisdom is X, courage is Y',\n"
                "  'success is when...', 'life is a journey', or any Hallmark-card\n"
                "  pattern. If you find yourself writing 'is the X of Y', stop and\n"
                "  pick a different angle.\n"
                "- Tone: dry, playful, specific, like something you'd see scrawled\n"
                "  in a coworker's terminal — not on an inspirational poster.\n\n"
                "Return ONLY the line, nothing else."
            )
            if memory_facts:
                prompt += "\n\nOptional context (use only if it sparks something concrete):\n"
                for f in memory_facts[:3]:
                    prompt += f"- {f}\n"

            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 1.05, "num_ctx": 4096},
            )
            text = (resp.get("message") or {}).get("content", "").strip()
            # Strip leading/trailing quotes (model adds them sometimes
            # despite being told not to) and any meta-prefix like
            # "Reflection: ..." or "Here's a thought: ...".
            import re as _re
            text = _re.sub(r"^(?:reflection|here'?s? .{0,40}?[:\-])\s*", "", text, flags=_re.IGNORECASE)
            text = text.strip().strip("'\"")
            # Drop any <think>…</think> a thinking model emits.
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
            # Reasonable length floor — drop if model returned trash.
            if 5 < len(text) <= 200:
                self.reflection = text.splitlines()[0].strip()
                self.reflection_at = time.time()
                try:
                    self._update_ui()
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._reflection_inflight = False

    def _get_shell_pane_text(self):
        """ANSI for the dedicated interactive-shell pane.

        Pulls from the active session's `live_buffer` (a list of bytes
        chunks captured by on_output) and shows the tail. Caps at the
        last ~80 lines so a chatty subprocess doesn't blow out the panel.
        """
        session = self._interactive_session
        if session is None:
            return ANSI("")
        tool = session.get("tool") or {}
        live_buffer = tool.get("_live_buffer") or []
        # Bytes → text, tolerating partial UTF-8. Then sanitize: subprocess
        # cursor / screen-clear escapes would pass through ANSI() and
        # corrupt the host terminal's layout. SGR colors survive.
        try:
            text = b"".join(live_buffer).decode("utf-8", errors="replace")
        except Exception:
            text = ""
        text = _sanitize_subprocess_text(text)
        lines = text.splitlines()
        if len(lines) > 80:
            lines = lines[-80:]
        body = "\n".join(lines) if lines else "(no output yet)"

        chat_mode = session.get("chat_mode", False)
        cmd = session.get("command", "")
        header = Text()
        header.append(" ⌘ ", style=f"bold {ACCENT}")
        header.append(f"{cmd[:90]}", style=f"bold {SECONDARY}")
        header.append("   ", style="")
        if chat_mode:
            header.append(" KEYS → CHAT  (Esc: switch to subprocess) ",
                          style=f"bold reverse {ACCENT}")
        else:
            header.append(" KEYS → SUBPROCESS  (Esc: switch to chat) ",
                          style=f"bold reverse {WARN}")
        return ANSI(self._render_to_ansi(Group(header, Text(""), Text(body))))

    def _open_or_update_file(self, path: str, content: str, fresh: bool) -> None:
        """Open / update a tab in the editor pane.

        `fresh=True` means this is a brand-new write that should
        animate (reveal_chars reset to 0). `fresh=False` is for updates
        that should snap (e.g. apply_diff growing an already-open file).
        """
        existing = self._open_files.get(path)
        if existing is None:
            self._open_files[path] = {
                "content": content or "",
                "opened_at": time.time(),
                "revealed_chars": 0 if fresh else len(content or ""),
            }
        else:
            existing["content"] = content or ""
            if fresh:
                existing["revealed_chars"] = 0
            else:
                existing["revealed_chars"] = len(content or "")
        # New/updated file → start at top of pane.
        if hasattr(self, "editor_window"):
            self.editor_window.vertical_scroll = 0

    def _cycle_editor_tab(self, direction: int) -> None:
        """Advance the active editor tab by `direction` (±1)."""
        paths = list(self._open_files.keys())
        if not paths:
            return
        try:
            idx = paths.index(self._active_editor_path) if self._active_editor_path in paths else 0
        except ValueError:
            idx = 0
        new_idx = (idx + direction) % len(paths)
        self._active_editor_path = paths[new_idx]
        # Switching tabs → reset scroll to top of the new file.
        if hasattr(self, "editor_window"):
            self.editor_window.vertical_scroll = 0

    def _get_editor_text(self):
        """Return ANSI for the right-side editor panel (Feature 3).

        Shows the most recent write_file's path + content using rich.Syntax
        so the user can see what the agent is writing without scrolling the
        tool log. Content is the *intended* final file — write_file is
        atomic in this codebase, not streamed. The panel updates once per
        tool call.
        """
        # Helpers for the editor pane defined just above the renderer.
        # (Body continues below.)
        if not self._open_files or self._active_editor_path is None:
            return ANSI("")
        # Bring the reveal animation forward by a tick. Walking up to ~6
        # chars per UI tick (40-60 per second at the 10fps anim cadence)
        # makes a 1.5s "typing" feel for short files, and longer files
        # finish revealing within a few seconds — fast enough not to
        # block the user, slow enough to register as motion.
        REVEAL_STEP = 64
        for f in self._open_files.values():
            target = len(f.get("content") or "")
            if f.get("revealed_chars", 0) < target:
                f["revealed_chars"] = min(target, f.get("revealed_chars", 0) + REVEAL_STEP)

        path = self._active_editor_path
        f = self._open_files.get(path) or {}
        full_content = f.get("content", "") or ""
        revealed = full_content[: f.get("revealed_chars", len(full_content))]

        # Tab row at the top — one chip per open file, the active one in
        # bright primary, others in dim. Truncated paths so the tab row
        # fits in the panel width.
        tabs = Text()
        for p in self._open_files.keys():
            label = os.path.basename(p) or p
            if len(label) > 16:
                label = label[:13] + "…"
            if p == path:
                tabs.append(f" {label} ", style=f"bold reverse {PRIMARY}")
            else:
                tabs.append(f" {label} ", style=f"dim {DIM}")
            tabs.append(" ", style="")
        if len(self._open_files) > 1:
            tabs.append("  [F6/F7]", style=f"dim {DIM}")

        # Header line (active file's full path + reveal progress).
        target = len(full_content)
        revealed_n = f.get("revealed_chars", target)
        header = Text()
        header.append(path, style=f"bold {PRIMARY}")
        if revealed_n < target:
            header.append(f"  {revealed_n}/{target}", style=f"italic {DIM}")

        # Syntax-highlight the full revealed content. The window's
        # vertical_scroll determines which slice is on-screen (set by
        # the Alt+Up / Alt+Down / Alt+PageUp / Alt+PageDown bindings).
        # Hard cap at 2000 lines for safety — beyond that, rendering
        # every UI tick gets expensive. Tail-truncated so the most-
        # recent content (typically what the agent just wrote) is the
        # visible portion when first opened.
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        lang_map = {
            "py": "python", "js": "javascript", "ts": "typescript",
            "tsx": "tsx", "jsx": "jsx", "sh": "bash", "bash": "bash",
            "md": "markdown", "rs": "rust", "go": "go", "java": "java",
            "c": "c", "cpp": "cpp", "h": "c", "hpp": "cpp",
            "json": "json", "yaml": "yaml", "yml": "yaml", "toml": "toml",
            "html": "html", "css": "css",
        }
        lang = lang_map.get(ext, "text")
        max_lines = 2000
        lines = revealed.splitlines()
        truncated = False
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
            truncated = True
        body = "\n".join(lines)
        renderables = [tabs, header]
        if truncated:
            renderables.append(Text(
                f"… (showing last {max_lines} of {len(revealed.splitlines())} lines — file too large to render in full)",
                style=DIM,
            ))
        if body:
            renderables.append(Syntax(body, lang, theme="monokai", line_numbers=True, word_wrap=False))
        return ANSI(self._render_to_ansi(Group(*renderables)))

    def _update_ui(self):
        # Serialize across threads — the worker thread and the asyncio
        # animation tick both call this; without the lock, they race on
        # the read-only flag and the buffer write raises EditReadOnlyBuffer.
        with self._update_lock:
            full_ansi = "\n".join(self.history_ansi)
            full_ansi += "\n" + self._get_current_renderable_ansi()

            # Only auto-scroll to bottom if the user was already there.
            # Otherwise preserve their scroll position so they can read
            # earlier content while generation continues. Anchored via
            # render_info from the previous render — accurate enough.
            was_at_bottom = self._is_at_bottom()

            self._history_read_only -= 1  # 1 -> 0, now writable
            try:
                self.history_buffer.text = full_ansi
            finally:
                self._history_read_only += 1  # back to read-only

            if was_at_bottom or self._force_scroll_next_update:
                self._scroll_to_bottom()
                self._force_scroll_next_update = False
            if self.app.is_running:
                self.app.invalidate()

    def _is_at_bottom(self) -> bool:
        """Return True iff the chat window is scrolled to (or within 2 lines
        of) the bottom on the last render. Used to decide whether the next
        `_update_ui` should auto-scroll — if the user has scrolled up to
        read earlier content, we preserve their position."""
        info = self.history_window.render_info
        if info is None:
            return True  # first paint — default to auto-scroll
        total_lines = info.ui_content.line_count
        last_visible = info.vertical_scroll + info.window_height
        return last_visible >= total_lines - 2
    def _get_status_text(self):
        """Status bar as a list of (inline-style, text) tuples.

        Streamlined into four left-to-right groups with `│` between them:
          1. State  — single glyph indicating idle vs generating
          2. Identity — auth · model emoji(s) · primary model · msg count
          3. Cost   — session tokens · energy estimate · (gen) elapsed · (gen) tools
          4. Mode   — only non-default toggles surface (COPY, EDITOR,
                      shell-routing). Default state shows nothing here so
                      the bar reads as clean.

        F-key hints used to live in this bar but were removed — they
        consumed most of the horizontal real estate and the same info is
        available via /help and F1. Keep the bar information-dense, not
        instruction-dense.
        """
        BG = "bg:#1f160e "
        SEP = (BG + "#5a4a3a", "  │  ")
        DOT = (BG + "#5a4a3a", "  ·  ")

        # ── Group 1: state glyph ──────────────────────────────────────
        if self.is_generating:
            # 1Hz cycle — slow enough to read as a steady heartbeat
            # rather than the previous 4Hz flicker.
            frame_idx = int(time.time()) % len(ACTIVITY_FRAMES)
            glyph = ACTIVITY_FRAMES[frame_idx]
            glyph_color = PRIMARY
        else:
            glyph = "·"
            glyph_color = "#7a7570"

        segments: list = [
            (BG + f"bold {glyph_color}", f"  {glyph}  "),
            SEP,
        ]

        # ── Group 2: identity ─────────────────────────────────────────
        auth_icon = "◉" if self.agent.session_authorized else "○"

        def _bare_name(token: str) -> str:
            return token.split(": ", 1)[1].strip() if ": " in token else token.strip()

        raw_tokens = [m.strip() for m in self.agent.model.split(",") if m.strip()]
        raw_models = [_bare_name(t) for t in raw_tokens]
        arch_model = getattr(getattr(self.agent, "architect", None), "model", None)
        if arch_model and arch_model not in raw_models:
            raw_models.append(arch_model)
        primary_model = raw_models[0] if raw_models else self.agent.model
        seen_glyphs: list = []
        for m in raw_models:
            g = model_emoji(m)
            if g not in seen_glyphs:
                seen_glyphs.append(g)
        model_glyphs = "".join(seen_glyphs) or model_emoji(primary_model)

        msg_count = len(self.agent.messages) if hasattr(self.agent, "messages") and self.agent.messages else 0

        segments.extend([
            (BG + "#7a7570", f"{auth_icon} "),
            (BG + "#ffd166", f"{model_glyphs} "),
            (BG + "#ffd166", f"{primary_model[:35]}"),
            DOT,
            (BG + "#c8c4be", f"{msg_count} msg"),
            SEP,
        ])

        # ── Group 3: cost ─────────────────────────────────────────────
        total_tokens = self.session_tokens_in + self.session_tokens_out
        if total_tokens >= 1000:
            tok_label = f"~{total_tokens / 1000:.1f}k tok"
        else:
            tok_label = f"~{total_tokens} tok"
        segments.extend([
            (BG + "#c8c4be", tok_label),
            DOT,
            (BG + "#c8c4be", f"{self.session_energy_wh:.2f} Wh"),
        ])
        if self.is_generating:
            n_tools = len(self.tool_executions)
            elapsed = time.time() - self.generation_start_time if self.generation_start_time else 0
            segments.extend([
                DOT,
                (BG + "#7fd070", f"⚙{n_tools}"),
                DOT,
                (BG + "#ff8c5c", _fmt_duration(elapsed)),
            ])

        # ── Group 4: reflection (subtle, one segment, dim) ────────────
        # Sits between Cost and Mode badges so it's visible by default
        # but doesn't push status-critical info off the right edge of
        # narrow terminals. Dim grey, no italic, no prefix glyph — the
        # idea is that the line BELONGS in the bar, not screams from it.
        # When the LLM hasn't produced one yet, this segment is omitted.
        if self.reflection:
            text = self.reflection
            # Hard-cap so the prompt-toolkit single-row status bar never
            # has to handle wrapped content. 60 chars is a tight ceiling
            # the LLM prompt aims for — this is a defensive truncation
            # in case it goes long.
            if len(text) > 60:
                text = text[:57] + "…"
            segments.append(SEP)
            segments.append((BG + "#7a7570", text))

        # ── Group 5: active toggles (only when non-default) ───────────
        badges: list = []
        # _mouse_capture defaults False (terminal-native selection on).
        # Show a badge only in the unusual ON state where prompt_toolkit
        # owns mouse events for scrolling and selection is disabled.
        if self._mouse_capture:
            badges.append((BG + "bold #ff8c5c", "🖱 SCROLL (F2 = copy)"))
        if self._show_editor and self._open_files:
            n = len(self._open_files)
            badges.append((BG + "bold #7fd070", f"⊟ EDITOR ({n})" if n > 1 else "⊟ EDITOR"))
        if self._interactive_session is not None:
            chat_mode = self._interactive_session.get("chat_mode", False)
            badges.append(
                (BG + "bold #7fd070", "↳ KEYS→CHAT") if chat_mode
                else (BG + "bold #ffd166", "↳ KEYS→SUBPROC")
            )
        if badges:
            segments.append(SEP)
            for i, b in enumerate(badges):
                if i > 0:
                    segments.append(DOT)
                segments.append(b)

        return segments

    def _spinner_for(self, role):
        """Return a cached Spinner instance for the given role.

        Cached so rich's frame counter (now - start_time / interval)
        actually advances across UI ticks instead of resetting to zero
        on every render.
        """
        key = role or ""
        spinner = self._spinner_cache.get(key)
        if spinner is None:
            rs = THEME.role(key)
            spinner = _make_custom_spinner(rs.spinner_frames, rs.color)
            self._spinner_cache[key] = spinner
        return spinner

    def _get_current_renderable_ansi(self):

        if not self.is_generating and not self.current_response_parts and not self.reasoning_chunks and not self.tool_executions:
            # Re-render every tick when idle so the welcome banner's
            # gradient flow and 🦀 color pulse actually animate. The
            # previous one-shot `welcome_shown` gate locked the banner
            # to a single frozen frame after first paint.
            if self.history_ansi:
                return ""  # post-conversation idle — keep history clean
            return render_to_ansi(self._get_welcome_panel())
            
        parts = []
        # ── Layout order (top → bottom) ────────────────────────────────────
        # 1. Side messages         — context notes (memory stored, etc).
        # 2. Unattached tools      — tool calls not nested under a plan task.
        # 3. Bottom slot (one of):
        #      - Auth prompt       (when waiting on Y/O/N/A)
        #      - ask_user question (when the agent asked for input)
        #      - Skill offer       (when a draft skill is parked for Y/N)
        #      - Unified panel     (plan tree)
        # 4. Assistant response    — the model's final reply, BELOW the
        #                            panel that produced it. Previously
        #                            sat above the panel which read as
        #                            "answer first, then the work that
        #                            produced it" — backwards.
        # 5. Spinner               — role-aware status line, last.
        #
        # Tools attached to a plan step nest inside the unified panel under
        # their task row; flat tool panels above are the fallback for
        # single-step / pre-plan tools.

        # 1. Side messages (capped to last 3)
        for msg in self.side_messages[-3:]:
            parts.append(Text(msg, style=f"dim {DIM} italic"))

        # 2. Unattached tool panels — tools that ran outside any plan
        # task. Plan-attached tools render inside the unified panel
        # via _render_nested_tool_row.
        for idx, tool in enumerate(self.tool_executions, 1):
            if self.current_plan is not None and tool.get("task_id") is not None:
                continue
            parts.append(self._build_tool_panel(tool, idx))

        # 3. Bottom slot: exactly one of {auth, question, skill offer,
        # unified panel} renders here. Auth and ask_user are blocking on
        # user input so they take precedence over the plan/reasoning.
        if self.auth_active and self.current_auth_chunk:
            parts.append(self._build_auth_panel(self.current_auth_chunk))
        elif self._pending_user_question is not None:
            parts.append(Panel(
                Text.assemble(
                    ("The agent is asking:", f"bold {ACCENT}"),
                    ("\n\n", ""),
                    (self._pending_user_question, "bold"),
                    ("\n\nType your answer below and press Enter.", f"dim {DIM}"),
                ),
                title=f"[bold {ACCENT}]Question for you[/bold {ACCENT}]",
                border_style=ACCENT,
                box=ROUNDED,
                padding=(1, 2),
            ))
        elif self._pending_skill_offer is not None:
            draft = self._pending_skill_offer
            body = Text()
            body.append("I learned something this turn. Save it as a skill?\n\n", style=f"bold {ACCENT}")
            body.append("Name: ", style=f"bold {DIM}")
            body.append(f"{draft.get('name', '?')}\n", style="bold")
            desc = draft.get("description", "")
            if desc:
                body.append("Description: ", style=f"bold {DIM}")
                body.append(f"{desc}\n", style="")
            body.append("Procedure:\n", style=f"bold {DIM}")
            for line in (draft.get("procedure", "") or "").splitlines()[:8]:
                body.append(f"  {line}\n", style=f"{SECONDARY}")
            reason = draft.get("reason")
            if reason:
                body.append(f"\nWhy: ", style=f"dim {DIM}")
                body.append(f"{reason}\n", style=f"italic {DIM}")
            body.append("\nPress ", style="")
            body.append("[Y]", style="bold")
            body.append(" to save, ", style="")
            body.append("[N]", style="bold")
            body.append(" to skip.", style="")
            parts.append(Panel(
                body,
                title=f"[bold {ACCENT}]💡 New skill candidate[/bold {ACCENT}]",
                border_style=ACCENT,
                box=ROUNDED,
                padding=(1, 2),
            ))
        else:
            unified_panel = self._render_unified_panel()
            if unified_panel is not None:
                parts.append(unified_panel)

        # 4. Assistant response — rendered AFTER the unified panel so the
        # user reads "plan → result" top-to-bottom. Previously this sat
        # at position 2 (above the panel), which made the response feel
        # disconnected from the work that produced it.
        current_content = "".join(self.current_response_parts)
        if current_content:
            parts.append(Markdown(current_content))

        if self.is_generating:
            elapsed = time.time() - self.generation_start_time
            idle_time = time.time() - self.last_chunk_time
            rs = THEME.role(self.current_role or "")
            # Build a multi-color status line so the 🦀 mascot can pulse
            # through the sunset palette independently of the role color
            # used for the rest of the line.
            mascot_color = self._cycle_palette_color(TITLE_GRADIENT)
            status_text = Text()
            status_text.append(f" {MASCOT}", style=f"bold {mascot_color}")
            status_text.append(f" {rs.icon} ", style=f"bold {rs.color}")
            status_text.append(f"{self.current_status}  [{_fmt_duration(elapsed)}]", style=f"bold {rs.color}")
            if idle_time > 15:
                status_text.append(f"  ⚠ idle {idle_time:.0f}s", style=f"bold {WARN}")
            spinner = self._spinner_for(self.current_role)
            spinner.text = status_text
            parts.append(spinner)

        return self._render_to_ansi(Group(*parts))

    def _render_to_ansi(self, renderable):
        """Helper to render a rich object to ANSI using a persistent console."""
        # Clear the internal buffer
        self.r_console.file.seek(0)
        self.r_console.file.truncate()
        self.r_console.print(renderable)
        return self.r_console.file.getvalue()

    @staticmethod
    def _clean_field(value):
        if value is None:
            return ""
        if isinstance(value, list):
            value = "\n".join(str(v) for v in value)
        s = str(value).strip()
        if not s or s.lower() in ("n/a", "none", "null"):
            return ""
        return s

    def _group_reasoning_into_steps(self) -> list:
        """Group the flat reasoning_log into a TODO-style list of steps.

        Each `architect` entry starts a new step. Subsequent non-architect
        entries (agent <think>, skill matches, self-check verdicts) are
        attached as substeps of the latest step. Pre-architect events
        get a synthetic placeholder so they don't get dropped.

        Returns: list[{head: entry, subs: [entries], status: str}]
        where status is "done" for all but the last, which is "in_progress"
        (or "complete" if the last substep was a self-check that passed).
        """
        steps: list = []
        for entry in self.reasoning_log:
            if entry["kind"] == "architect":
                steps.append({"head": entry, "subs": []})
            else:
                if not steps:
                    # Edge case: a sub-agent emitted thinking before any
                    # architect step (fast-route path). Synthesize a head
                    # so the substep still has somewhere to attach.
                    steps.append({
                        "head": {
                            "kind": "agent",
                            "label": entry.get("label", "agent"),
                            "body": "(direct route — no architect plan)",
                            "time": entry.get("time", 0),
                        },
                        "subs": [],
                    })
                steps[-1]["subs"].append(entry)

        # Status assignment. Last step is in_progress while the run is
        # ongoing; everything before it is done.
        for i, step in enumerate(steps):
            if i < len(steps) - 1:
                step["status"] = "done"
            else:
                # If the last substep was a self_check with an 'ok'
                # verdict, the step itself completed cleanly.
                last_sub = step["subs"][-1] if step["subs"] else None
                if (
                    last_sub is not None
                    and last_sub["kind"] == "self_check"
                    and "ok" in (last_sub["body"] or "").lower()
                    and "missing" not in (last_sub["body"] or "").lower()
                ):
                    step["status"] = "done"
                elif not self.is_generating:
                    # Run finished — the last step is also done.
                    step["status"] = "done"
                else:
                    step["status"] = "in_progress"
        return steps

    def _render_unified_panel(self):
        """Merged Plan + Reasoning panel.

        Replaces the previously-separate `_render_plan_panel` and
        `_render_reasoning_panel` so the user has ONE place to look for
        "what's happening." Layout inside the panel:

            (plan tree)                ← when self.current_plan is set
            ── Reasoning ──            ← divider, only if both sections exist
            (reasoning timeline)       ← when reasoning_log has entries

        Cases handled:
        - Both plan and reasoning present → both sections, divider between.
        - Plan only (no reasoning yet) → plan section, no divider.
        - Reasoning only (single-step request, no plan) → reasoning section
          alone; title falls back to "🧠 Reasoning · step N".
        - Neither → returns None (callers omit the panel entirely).
        - `show_reasoning` False → reasoning section collapses to a chip
          line ("🧠 reasoning · step N · …  [Ctrl+R] expand"). Plan stays
          fully rendered because it's load-bearing for in-flight work.
        """
        from theme import TASK_STATE_STYLE, TASK_STATE_FLASH
        REASON_COLOR = "#9999cc"

        has_plan = self.current_plan is not None and self.current_plan.tasks
        has_reasoning = bool(self.reasoning_log)
        if not has_plan and not has_reasoning:
            return None

        body_renderables: list = []

        # ── Plan section ───────────────────────────────────────────────
        plan_done = plan_total = 0
        plan_title_text = ""
        if has_plan:
            plan = self.current_plan
            plan_done, plan_total = plan.progress()
            plan_title_text = f"Plan: {plan.title}  ·  {plan_done}/{plan_total}"
            now = time.time()

            tools_by_task: dict = {}
            for idx, tool in enumerate(self.tool_executions, 1):
                tid = tool.get("task_id")
                if tid is None:
                    continue
                tools_by_task.setdefault(tid, []).append((idx, tool))

            plan_lines: list = []

            def render_subtree(task, depth: int) -> None:
                icon, color = TASK_STATE_STYLE.get(task.status, ("•", DIM))
                flashing = now < self._task_flash_until.get(task.id, 0.0)
                line_color = TASK_STATE_FLASH.get(task.status, color) if flashing else color
                display_icon = icon
                if flashing and task.status == "done":
                    display_icon = "✨"
                    line_color = self._cycle_palette_color(TITLE_GRADIENT, period_sec=0.15)
                weight = "bold " if task.status == "in_progress" else ""
                indent = "  " + ("    " * depth)
                connector = "↳ " if depth > 0 else ""
                line_text = Text()
                line_text.append(f"{indent}", style="")
                if connector:
                    line_text.append(connector, style=f"dim {DIM}")
                line_text.append(f"{display_icon} ", style=f"{weight}{line_color}")
                line_text.append(f"{task.id}. ", style=f"dim {DIM}")
                line_text.append(task.description, style=f"{weight}{line_color}")
                plan_lines.append(line_text)
                if task.status == "in_progress" and self.architect_intent:
                    plan_lines.extend(self._render_intent_lines(self.architect_intent))
                for tidx, tool in tools_by_task.get(task.id, []):
                    plan_lines.append(self._render_nested_tool_row(tidx, tool))
                for child in plan.children_of(task.id):
                    render_subtree(child, depth + 1)

            roots = plan.roots() if hasattr(plan, "roots") else plan.tasks
            for root in roots:
                render_subtree(root, 0)
            body_renderables.extend(plan_lines)

        # The separate "Reasoning" section that used to live here was
        # removed: every architect step's `goal` field is the same string
        # as the active plan task's description (it's literally what the
        # architect promoted into a task), so the section duplicated the
        # plan tree verbatim. Architect intent (goal / observation /
        # critical_thinking) is now rendered inline under the in-progress
        # task via _render_intent_lines, which is the only non-duplicate
        # piece of reasoning content. Substeps that DO carry independent
        # info (skill matches, self-check verdicts) still come through
        # via the reasoning_log → side_messages path elsewhere.
        #
        # When the plan is empty but reasoning has fired (rare — single-
        # step runs that didn't produce a plan), surface the latest
        # architect goal as a single italic line so the panel isn't
        # completely empty.
        if not has_plan and has_reasoning:
            steps = self._group_reasoning_into_steps()
            if steps:
                head = steps[-1]["head"] or {}
                headline = (head.get("body") or "").split("·")[0].strip()
                if headline.startswith("goal:"):
                    headline = headline[len("goal:"):].strip()
                if len(headline) > 110:
                    headline = headline[:107] + "…"
                if headline:
                    line = Text()
                    line.append("  ", style="")
                    line.append(head.get("label") or "agent", style=f"dim {REASON_COLOR}")
                    line.append(" — ", style=f"dim {DIM}")
                    line.append(headline, style=f"italic {REASON_COLOR}")
                    body_renderables.append(line)

        # ── Wrap in one outer Panel ────────────────────────────────────
        # Animation policy: the panel sits on screen the whole turn and
        # shouldn't visually pulse. Title color drifts slowly (period 8s),
        # border stays a steady dim warm.
        PANEL_TITLE_PERIOD_SEC = 8.0
        if has_plan:
            title_color = self._cycle_palette_color(TITLE_GRADIENT, period_sec=PANEL_TITLE_PERIOD_SEC)
            border_style = f"dim {title_color}"
            title_markup = f"[bold {title_color}]{plan_title_text}[/bold {title_color}]"
        else:
            # Reasoning-only fallback — no plan was created.
            border_style = f"dim {REASON_COLOR}"
            title_markup = f"[bold {REASON_COLOR}]Working[/bold {REASON_COLOR}]"

        return Panel(
            Group(*body_renderables),
            title=title_markup,
            border_style=border_style,
            box=ROUNDED,
            padding=(0, 1),
        )

    def _render_nested_tool_row(self, idx: int, tool: dict):
        """One-line summary for a tool call rendered under its plan step.

        Format: `    ├─ ⚡ run_shell  ls -la …  ✓` — branch glyph, kind
        icon, name, first arg summary, status marker. Indented two extra
        spaces from the task glyph so the hierarchy reads at a glance.
        """
        kind = THEME.tool_kind(tool.get("name", ""))
        name = tool.get("name", "?")
        args = tool.get("args", {}) or {}
        result = tool.get("result")

        # Pick the most identifying arg to show inline.
        first_value = ""
        if isinstance(args, dict) and args:
            for key in ("command", "path", "file_path", "url", "query", "question", "draft"):
                if key in args and args[key]:
                    first_value = str(args[key]).splitlines()[0].strip()
                    break
            if not first_value:
                # Fall back to first arg in declaration order.
                k0 = next(iter(args))
                v0 = args.get(k0)
                if v0 is not None:
                    first_value = str(v0).splitlines()[0].strip()
        if len(first_value) > 50:
            first_value = first_value[:47] + "…"

        if result is None:
            status_icon = "…"
            status_color = WARN
        elif isinstance(result, str) and result.lower().startswith("error"):
            status_icon = "✗"
            status_color = ERR
        else:
            status_icon = "✓"
            status_color = ACCENT

        line = Text()
        line.append("     ├─ ", style=f"dim {DIM}")
        line.append(f"{kind.icon} ", style=f"bold {kind.color}")
        line.append(name, style=f"bold {kind.color}")
        if first_value:
            line.append(f"  {first_value}", style=f"italic {SECONDARY}")
        line.append(f"  {status_icon}", style=f"bold {status_color}")
        return line

    def _render_architect_intent(self, intent):
        agent = intent.get("agent") or "?"
        reflection = intent.get("reflection") or {}
        goal = self._clean_field(reflection.get("goal"))
        obs = self._clean_field(reflection.get("observation"))
        ct = self._clean_field(reflection.get("critical_thinking"))
        reasoning = self._clean_field(intent.get("reasoning"))
        plan = self._clean_field(intent.get("plan"))

        if not self.show_architect:
            # Compact: dim one-liner — keeps routing context visible without
            # the screen-eating panels. F3 expands.
            rs = THEME.role(agent)
            flashing = time.time() < self._chip_flash_until
            # The icon ALWAYS pulses subtly between the role's base color
            # and a brightened flash color — even when not transitioning —
            # so the chip feels alive instead of static. Flash on
            # role-change still spikes to the full flash color.
            icon_pulse_t = (time.time() * 1.4) % 2.0
            icon_color = rs.flash_color if flashing or icon_pulse_t < 0.4 else rs.color
            chip_color = rs.flash_color if flashing else rs.color
            from phrases import THINKING_BADGE as _TB
            headline = goal or (plan.splitlines()[0] if plan else reasoning) or f"{_TB}…"
            if len(headline) > 110:
                headline = headline[:107] + "…"
            # Subtle headline shimmer: one character at a time rendered
            # bright at ~1.5Hz sweep — gentle but unmistakably "alive".
            shimmer_pos = int(time.time() * 6) % max(len(headline), 1)
            line = Text()
            line.append(f"{rs.icon} ", style=f"bold {icon_color}")
            line.append(agent, style=f"bold {chip_color}")
            line.append(" · ", style=f"dim {DIM}")
            for i, ch in enumerate(headline):
                if i == shimmer_pos:
                    line.append(ch, style=f"italic bold {SECONDARY}")
                else:
                    line.append(ch, style=f"italic {DIM}")
            line.append("   [F3] expand", style=f"dim {DIM}")
            return line

        # Expanded view: only render fields with real content; collapse
        # reasoning into plan when it's a substring/prefix to avoid the
        # double-printed prose that bloated the old layout.
        role_color = THEME.role(agent).color
        body = Text()
        body.append("agent: ", style=f"bold {DIM}")
        body.append(agent + "\n", style=f"bold {role_color}")
        if goal:
            body.append("goal: ", style=f"bold {PRIMARY}")
            body.append(goal + "\n", "")
        if obs:
            body.append("observation: ", style=f"bold {WARN}")
            body.append(obs + "\n", "")
        if ct:
            body.append("critical thinking: ", style=f"bold {ACCENT}")
            body.append(ct + "\n", "")

        sections = [body]
        if reasoning and not (plan and reasoning.lower() in plan.lower()):
            sections.append(Text(reasoning, style="italic"))
        if plan:
            sections.append(Markdown(plan))

        return Panel(
            Group(*sections),
            title=f"[bold {role_color}]architect[/bold {role_color}]  [dim {DIM}]([F3] collapse)[/dim {DIM}]",
            border_style=f"dim {role_color}",
            box=ROUNDED,
            padding=(0, 1),
        )

    def _render_intent_lines(self, intent: dict) -> list:
        """Return a list of Text lines representing the architect's intent,
        indented to sit under the active plan task row.

        The format mirrors the standalone architect chip but as inline
        lines rather than a Panel. Empty fields are omitted.
        """
        lines = []
        if not intent:
            return lines
        agent = intent.get("agent")
        if agent:
            role_color = THEME.role(agent).color
            t = Text()
            t.append("        agent: ", style=f"dim {DIM}")
            t.append(agent, style=f"bold {role_color}")
            lines.append(t)
        for field, label, style, italic in (
            ("goal", "goal", PRIMARY, False),
            ("observation", "observation", WARN, False),
            ("critical_thinking", "critical", ACCENT, False),
            ("reasoning", "reasoning", DIM, True),
        ):
            value = intent.get(field)
            if value:
                t = Text()
                t.append(f"        {label}: ", style=f"bold {style}")
                t.append(value, style="italic" if italic else "")
                lines.append(t)
        return lines

    def _get_welcome_panel(self):
        body = Text()

        # Static gradient title — colors are positional (each letter gets
        # its index's slot in TITLE_GRADIENT) so the banner reads as a
        # single composed piece, not a marquee. The previous shimmer
        # sweep + 2.5Hz offset shift looked like a screensaver in the
        # corner of the screen during idle; the streamline pass removed
        # both. The welcome panel only paints when there's no history,
        # so the user sees it once per fresh session — animation isn't
        # earning its visual cost.
        body.append(f"{MASCOT} ", style=f"bold {PRIMARY}")
        title = "EzClaw"
        for i, ch in enumerate(title):
            body.append(ch, style=f"bold {TITLE_GRADIENT[i % len(TITLE_GRADIENT)]}")
        body.append(" ", "")
        body.append("v2.3 (Full TUI)\n", style=f"dim {DIM}")
        body.append("─" * 40 + "\n", style=f"dim {DIM}")

        if ENABLE_MULTI_AGENT and hasattr(self.agent, "agents"):
            body.append("⚡ Multi-agent\n", style=f"bold {PRIMARY}")
            arch_model = getattr(self.agent.architect, "model", "?") if hasattr(self.agent, "architect") else "?"
            arch_style = THEME.role("architect")
            body.append(f"  {arch_style.icon} {'architect':<10} ", style=f"bold {arch_style.color}")
            body.append(f"{arch_model}\n", style=SECONDARY)
            for role, sub_agent in self.agent.agents.items():
                rs = THEME.role(role)
                body.append(f"  {rs.icon} {role:<10} ", style=f"bold {rs.color}")
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
        body.append(
            "/help [F1]  /settings  /queue  /skills  /memory  /phrases  /wisdom  /diagnose  /clear",
            style=f"dim {DIM}",
        )
        body.append("\n", "")
        body.append("Keys:     ", style="bold")
        body.append(
            "[F2] copy  [F3] strategy  [F4] tools  [F5] editor  [F6/F7] tab  [Ctrl+R] reasoning  "
            "[Ctrl+J] newline",
            style=f"dim {DIM}",
        )
        return Panel(
            body,
            box=ROUNDED, padding=(1, 2), border_style=DIM,
            title=f"[bold {PRIMARY}]{MASCOT} EzClaw[/bold {PRIMARY}]",
        )

    @staticmethod
    def _cycle_palette_color(palette, period_sec: float = 1.5) -> str:
        """Return a color from `palette` based on the current time. Used to
        animate single glyphs (🦀, ◐, etc.) by re-picking each UI tick —
        the animation loop redraws every ~100ms so the human sees a smooth
        ~600ms transition between colors at the default 1.5s period."""
        idx = int(time.time() / period_sec) % len(palette)
        return palette[idx]

    @classmethod
    def _gradient_offset(cls) -> int:
        """Index offset into TITLE_GRADIENT, advancing once per ~0.4s.
        Used to flow the welcome title gradient across letters over time."""
        return int(time.time() * 2.5) % len(TITLE_GRADIENT)

    def _one_line_tool_head(self, tool_kind, tool_name, args, index):
        """Build the icon + index + name + (args) prefix used by the
        one-line collapsed/running tool entries.

        Brightness hierarchy:
          icon       — pulsing kind color (most prominent)
          tool name  — solid kind color, bold
          args       — SECONDARY (warm light grey, plainly readable)
          [N] index  — dim (low-priority)
          parens     — dim (chrome)
        """
        pulse_t = (time.time() * 1.2) % 2.0
        icon_color = (
            self._cycle_palette_color(TITLE_GRADIENT, period_sec=1.5)
            if pulse_t < 0.4 else tool_kind.color
        )
        line = Text()
        line.append(f"{tool_kind.icon} ", style=f"bold {icon_color}")
        if index is not None:
            line.append(f"[{index}] ", style=f"{DIM}")
        line.append(tool_name, style=f"bold {tool_kind.color}")
        if args:
            sig = self._format_args_inline(args)
            line.append("(", style=f"{DIM}")
            line.append(sig, style=f"{SECONDARY}")
            line.append(")", style=f"{DIM}")
        return line

    @staticmethod
    def _format_args_inline(args, max_len=60):
        """Render an args dict as a function-call signature for the title.

        Quotes strings, leaves numbers/bools bare, truncates long values, and
        caps the total length so the title stays on one line.
        """
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            if isinstance(v, (bool, int, float)):
                rendered = str(v)
            else:
                s = str(v).replace("\n", "\\n")
                if len(s) > 30:
                    s = s[:27] + "…"
                rendered = f'"{s}"'
            parts.append(f"{k}={rendered}")
        sig = ", ".join(parts)
        if len(sig) > max_len:
            sig = sig[: max_len - 1] + "…"
        return sig

    def _build_auth_panel(self, chunk: dict):
        """Render the security-check panel shown while waiting on a user
        decision about an auth_required tool.

        Layout goal: the four choices read as a row of equal-weight chips
        with their key and consequence side by side, so the user can pick
        without re-reading the help text every time. The most-common
        intent (Y = remember this tool) is highlighted; the rare one (A =
        blanket session allow) is dimmed.
        """
        tool_name = chunk.get("name", "?")
        args = chunk.get("arguments", {}) or {}
        # Pick the most identifying arg to render inline so the user can
        # see WHAT this call is about, not just which tool. Mirrors the
        # nested-tool-row logic.
        arg_summary = ""
        if isinstance(args, dict) and args:
            for key in ("command", "path", "file_path", "url", "query"):
                if key in args and args[key]:
                    arg_summary = str(args[key]).splitlines()[0].strip()
                    break
            if not arg_summary:
                k0 = next(iter(args))
                v0 = args.get(k0)
                if v0 is not None:
                    arg_summary = f"{k0}={str(v0).splitlines()[0]}"
        if len(arg_summary) > 90:
            arg_summary = arg_summary[:87] + "…"

        # Header block — tool + identifying arg.
        header = Text()
        header.append(" Authorization required", style=f"bold {WARN}")
        header.append("  · ", style=f"dim {DIM}")
        header.append(tool_name, style="bold")
        if arg_summary:
            header.append(f"  {arg_summary}", style=f"italic {SECONDARY}")
        header.append("\n")

        # Choice rows. Each row is: [key]  name — what it actually does.
        # Order is meaningful — Y first because it's the recommended
        # default; A last because it's the broadest grant.
        choices = [
            ("Y", ACCENT,  "Allow this tool",  "won't ask for the same tool again this session"),
            ("O", PRIMARY, "Allow once",       "this single call only — ask again next time"),
            ("N", ERR,     "Deny",             "skip this call; agent continues without the result"),
            ("A", DIM,     "Allow all tools",  "blanket auth for the rest of the session"),
        ]
        rows = Text()
        for key, color, name, desc in choices:
            rows.append(" ", style="")
            rows.append(f" {key} ", style=f"bold reverse {color}")
            rows.append("  ", style="")
            rows.append(f"{name:<17}", style=f"bold {color}")
            rows.append(f"  {desc}\n", style=f"dim {DIM}")

        return Panel(
            Group(header, Text(""), rows),
            title=f"[bold {WARN}] 🛡  Security check[/bold {WARN}]",
            border_style=f"bold {WARN}",
            box=ROUNDED,
            padding=(1, 2),
        )

    def _build_tool_panel(self, tool, index=None):
        tool_name = tool["name"]
        args = tool.get("args", {})
        result = tool.get("result")
        expanded = tool.get("expanded", False)
        compact = self.compact_tools

        # Truncation caps only matter when expanded — the collapsed view
        # shows just a one-line summary, so the body is never rendered.
        diff_cap = 2000
        read_cap = 5000
        output_cap = 5000
        args_cap = 200

        tool_kind = THEME.tool_kind(tool_name)

        # ── Running: one-line indicator (no border) ────────────────────────
        if not result:
            start_time = tool.get("start_time")
            elapsed = time.time() - start_time if start_time else 0
            verb = tool.get("_running_verb")
            if verb is None:
                from phrases import pick as _pick, TOOL_RUNNING as _TR
                verb = _pick(_TR)
                tool["_running_verb"] = verb
            elapsed_str = f" ({_fmt_duration(elapsed)})" if elapsed > 1 else ""

            # Embedded interactive shell — render the streaming subprocess
            # output as a live tool panel inside the chat. The panel
            # expands as bytes arrive; the user types into the chat box
            # and their keystrokes are forwarded to the subprocess.
            live_buffer = tool.get("_live_buffer")
            if tool.get("interactive") and live_buffer is not None:
                # Detect "subprocess waiting for input": no output for
                # ~1.5s while session is still active. Promote the panel
                # to a high-attention state and (once per waiting event)
                # ping the user via OS notification + sound.
                session = self._interactive_session or {}
                last_out = session.get("last_output_at", time.time())
                idle = time.time() - last_out
                waiting = idle > 1.5

                if waiting and not session.get("notified_waiting"):
                    session["notified_waiting"] = True
                    try:
                        from notifications import notify, URGENCY_CRITICAL
                        notify(
                            "🦀 ezclaw: subprocess waiting for input",
                            f"`{tool_name}` is waiting on a prompt — switch to ezclaw to type.",
                            urgency=URGENCY_CRITICAL,
                        )
                    except Exception:
                        pass

                head = self._one_line_tool_head(tool_kind, tool_name, args, index)
                chat_mode = session.get("chat_mode", False)
                if waiting:
                    # Bright attention-grabbing label, animated pulse on
                    # the ✋ glyph so the eye finds it immediately.
                    pulse_color = self._cycle_palette_color(
                        ("#ff8c5c", "#ffd166", "#ff5fd7"), period_sec=0.5
                    )
                    head.append(f"   ✋ ", style=f"bold {pulse_color}")
                    head.append(
                        f"AWAITING INPUT  ({idle:.0f}s idle) — type below",
                        style=f"bold {WARN}",
                    )
                    border = WARN
                else:
                    head.append(f"   ⏳ live{elapsed_str}", style=f"bold {WARN}")
                    border = f"dim {WARN}"
                # Mode indicator — tells the user where their keystrokes
                # are going. Esc toggles between the two.
                if chat_mode:
                    head.append("   ↳ keys → ezclaw chat", style=f"bold #7fd070")
                    head.append("  (Esc to send to subprocess)", style=f"{SECONDARY} italic")
                else:
                    head.append("   ↳ keys → subprocess", style=f"bold {WARN}")
                    head.append("  (Esc to chat, Ctrl+C to interrupt)", style=f"{SECONDARY} italic")

                # Decode the streaming bytes; sanitize cursor/screen-clear
                # escapes (which would corrupt our layout) while keeping
                # SGR colors so things like `ls --color` look right. Cap
                # at ~40 lines so a chatty subprocess can't grow this
                # panel indefinitely.
                try:
                    full = b"".join(live_buffer).decode("utf-8", errors="ignore")
                except Exception:
                    full = ""
                full = _sanitize_subprocess_text(full)
                lines = [l for l in full.splitlines() if l.strip("\r")][-40:]
                body = Text("\n".join(lines), style="")
                return Panel(
                    Group(head, body),
                    border_style=border,
                    box=ROUNDED,
                )

            line = self._one_line_tool_head(tool_kind, tool_name, args, index)
            # Running-verb status: readable warm light grey, italic to set
            # it apart from the args signature without disappearing.
            line.append(f"   ⏳ {verb}…{elapsed_str}", style=f"{SECONDARY} italic")
            return line

        # We have a result. Compute the one-line summary and whether the
        # output has more content than the summary captures.
        renderable_result = str(result)
        non_empty_lines = [l for l in renderable_result.splitlines() if l.strip()]
        summary = None
        if non_empty_lines:
            first = non_empty_lines[0].strip()
            # Cap a bit tighter in one-line view so the row fits without
            # wrapping on typical terminal widths.
            summary = first[:60] + ("…" if len(first) > 60 else "")
        has_more = len(non_empty_lines) > 1 or (
            bool(non_empty_lines) and len(non_empty_lines[0]) > 60
        )

        # ── Collapsed: ONE LINE, no border ─────────────────────────────────
        if not expanded:
            line = self._one_line_tool_head(tool_kind, tool_name, args, index)
            if summary:
                # Summary is the at-a-glance result the user reads while
                # scanning the chat — must be plainly visible, not dimmed.
                # `↳` arrow stays muted (chrome); the summary text uses
                # SECONDARY (warm light grey) so it sits at body-text
                # readability level.
                line.append("   ↳ ", style=f"{DIM}")
                line.append(summary, style=f"{SECONDARY} italic")
            if has_more and index is not None:
                line.append(f"   [/expand {index}]", style=f"dim {DIM} italic")
            return line

        # ── Expanded: full bordered panel ──────────────────────────────────
        idx_label = f"[{index}] " if index is not None else ""
        state_label = "  ▴ expanded"
        header_parts = [
            (f"{tool_kind.icon} ", f"bold {tool_kind.color}"),
            (idx_label, f"dim {DIM}"),
            (tool_name, f"bold {tool_kind.color}"),
        ]
        if compact and args:
            sig = self._format_args_inline(args)
            header_parts.append(("(", f"dim {DIM}"))
            header_parts.append((sig, f"dim {DIM}"))
            header_parts.append((")", f"dim {DIM}"))
        header_parts.append((state_label, f"dim {DIM}"))
        header = Text.assemble(*header_parts)

        tool_parts = []

        # ── Expanded: render the full body ─────────────────────────────────
        if not compact and args:
            arg_lines = []
            for k, v in args.items():
                arg_lines.append(Text.assemble((f"{k}: ", "bold"), (str(v), "")))
            arg_text = Text("\n").join(arg_lines)
            tool_parts.append(Panel(self._truncate_text(arg_text, max_lines=args_cap),
                                    title="args", border_style=f"dim {DIM}"))

        if tool_name == "write_file" and "Diff:" in renderable_result:
            parts_of_result = renderable_result.split("Diff:\n", 1)
            if len(parts_of_result) > 1:
                tool_parts.append(Text(parts_of_result[0]))
                diff_content = self._truncate_text(parts_of_result[1], diff_cap)
                if isinstance(diff_content, Text):
                    diff_content = diff_content.plain
                tool_parts.append(Syntax(diff_content, "diff", theme="monokai", background_color="default"))
            else:
                truncated = self._truncate_text(renderable_result, max_lines=output_cap)
                tool_parts.append(truncated if compact else Panel(truncated, title="output", border_style=DIM))
        elif tool_name == "read_file":
            path_arg = str(args.get("path", "")) if args else ""
            lang = self._detect_lang(path_arg)
            file_content = self._truncate_text(renderable_result, read_cap)
            if isinstance(file_content, Text):
                file_content = file_content.plain
            tool_parts.append(Syntax(file_content, lang, theme="monokai", background_color="default"))
        else:
            truncated = self._truncate_text(Text(renderable_result), max_lines=output_cap)
            tool_parts.append(truncated if compact else Panel(truncated, title="output", border_style=DIM))

        if index is not None:
            tool_parts.append(Text(
                f"  /collapse {index}  to hide output",
                style=f"dim {DIM} italic",
            ))

        return Panel(
            Group(*tool_parts),
            title=header,
            border_style=f"dim {tool_kind.color}",
            box=ROUNDED,
        )

    _LANG_BY_EXT = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "tsx", ".jsx": "jsx", ".rs": "rust", ".go": "go",
        ".rb": "ruby", ".sh": "bash", ".bash": "bash", ".zsh": "bash",
        ".fish": "fish", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".md": "markdown", ".sql": "sql", ".html": "html",
        ".css": "css", ".c": "c", ".cc": "cpp", ".cpp": "cpp", ".h": "c",
        ".hpp": "cpp", ".java": "java", ".kt": "kotlin", ".swift": "swift",
        ".lua": "lua", ".vim": "vim", ".env": "ini", ".cfg": "ini",
        ".ini": "ini", ".dockerfile": "dockerfile",
    }

    @classmethod
    def _detect_lang(cls, path: str) -> str:
        path_lower = path.lower()
        if path_lower.endswith(("dockerfile",)):
            return "dockerfile"
        for ext, lang in cls._LANG_BY_EXT.items():
            if path_lower.endswith(ext):
                return lang
        return "text"

    def _truncate_text(self, content: Any, max_lines: int = 1000) -> Any:
        """Truncates text or Text objects and appends a styled notice."""
        if isinstance(content, Text):
            lines = content.split("\n")
            if len(lines) > max_lines:
                new_content = Text("\n").join(lines[:max_lines])
                new_content.append(f"\n\n... (Output truncated at {max_lines} lines) ...", style=f"bold {WARN}")
                return new_content
            return content
            
        # Fallback for strings
        text = str(content)
        lines = text.splitlines()
        if len(lines) > max_lines:
            # We use Text.from_markup here to ensure the styling is parsed correctly
            msg = f"\n\n[bold {WARN}]... (Output truncated at {max_lines} lines) ...[/bold {WARN}]"
            return Text.assemble("\n".join(lines[:max_lines]), Text.from_markup(msg))
        return text

    def handle_input(self, text):
        if text.lower() in ["exit", "quit"]:
            self.app.exit()
            return

        # An ask_user tool call is pending — route this keystroke straight
        # to the answer queue and unblock the agent worker. Slash commands
        # still work (let them fall through to the dispatcher) so the user
        # can /cancel out of a question.
        if self._pending_user_question is not None and not text.startswith("/"):
            self.history_ansi.append(render_to_ansi(Panel(
                text, title=self.user_name, border_style=PRIMARY,
            )))
            self._user_answer_queue.put(text)
            self._update_ui()
            return

        # An embedded interactive shell session is active. Routing depends
        # on the chat_mode toggle (flipped via Esc):
        #   chat_mode=False (default) → keystroke → subprocess input_queue
        #   chat_mode=True           → keystroke → normal chat handling,
        #     but slash commands are the only thing that does anything
        #     useful (the agent worker is blocked on the subprocess).
        if self._interactive_session is not None and not self._interactive_session.get("chat_mode"):
            try:
                self._interactive_session["input_queue"].put((text + "\n").encode())
            except Exception:
                pass
            self.history_ansi.append(render_to_ansi(
                Text(f"→ {text}", style=f"italic {SECONDARY}")
            ))
            self._update_ui()
            return

        # In chat_mode during an interactive session: slash commands fall
        # through to _handle_command below; plain text gets a friendly
        # note instead of being silently dropped or sent to a blocked
        # agent worker.
        if self._interactive_session is not None and self._interactive_session.get("chat_mode"):
            if not text.startswith("/") and text.lower() not in ("exit", "quit"):
                self.side_messages.append(
                    "agent is busy in an interactive shell — Esc to send input to subprocess, "
                    "or use a slash command (/help)"
                )
                self._update_ui()
                return

        if text.startswith("/"):
            self._handle_command(text)
            return

        if self.halted:
            from continuation import is_continuation
            if is_continuation(text):
                # Continuation path: extend the existing panel.
                self.halted = False
                self.current_response_parts.append("\n\n*↳ continuing…*\n\n")
                self._force_scroll_next_update = True
                self.is_generating = True
                self._start_animation_loop()
                threading.Thread(
                    target=self._agent_worker, args=(text,), daemon=True
                ).start()
                return
            # Non-continuation: finalize the prior halted panel first.
            final_renderable = self._get_current_renderable_ansi()
            self.history_ansi.append(final_renderable)
            self.current_response_parts = []
            self.reasoning_chunks = []
            self.reasoning_log = []
            self.tool_executions = []
            self.side_messages = []
            self.halted = False
            # Fall through to the existing fresh-turn path below.

        # Regular message — always snap to bottom on new prompt so the user
        # sees their own message and the start of the reply, even if they had
        # scrolled up while reading older history.
        self._force_scroll_next_update = True
        self._last_notified_status = None  # fresh turn, allow notifications again
        self.history_ansi.append(render_to_ansi(Panel(
            text, title=self.user_name, border_style=PRIMARY,
        )))
        self.is_generating = True
        self.current_response_parts = []
        self.reasoning_chunks = []
        self.reasoning_log = []
        self.tool_executions = []
        self.side_messages = []
        self.architect_intent = None
        self.current_role = None
        self._last_chip_role = None
        self._chip_flash_until = 0.0
        self.current_plan = None
        self._last_task_states = {}
        self._task_flash_until = {}
        from phrases import pick as _pick, CONNECTING as _CN
        self.current_status = f"{_pick(_CN)}…"
        self.generation_start_time = time.time()
        self.last_chunk_time = time.time()
        # Feature 2: tally the prompt as input tokens.
        self.session_tokens_in += max(1, len(text) // 4)

        self._update_ui()
        # Start redraw loop for animations
        self._start_animation_loop()
        threading.Thread(target=self._agent_worker, args=(text,), daemon=True).start()

    def _start_animation_loop(self):
        """Run a continuous UI tick so animations are visible at all times.

        - During generation: 10fps (matches the spinner cadence)
        - At idle: 4fps — enough for the welcome banner's gradient flow
          and mascot color pulse to be visible, cheap enough that the CPU
          cost is negligible.
        """
        loop = getattr(self.app, 'loop', None)
        if not loop:
            return
        # Guard against double-starting the loop (handle_input restarts it
        # on every new prompt). If a tick is already scheduled, this no-ops.
        if getattr(self, "_anim_loop_active", False):
            return
        self._anim_loop_active = True

        def tick():
            try:
                self._update_ui()
            except Exception:
                pass
            delay = 0.1 if self.is_generating else 0.25
            loop.call_later(delay, tick)

        loop.call_later(0.1, tick)

    def _handle_command(self, cmd):
        global SHOW_THINKING
        cmd = cmd.lower().strip()
        if cmd.startswith("/thinking"):
            SHOW_THINKING = "on" in cmd or ("off" not in cmd and not SHOW_THINKING)
            msg = f"Thinking visualization: {'ON' if SHOW_THINKING else 'OFF'}"
            self.history_ansi.append(render_to_ansi(Text(msg, style=DIM)))
        elif cmd.startswith("/notify"):
            # Runtime toggle for desktop notifications. Mirrors /thinking.
            current = os.environ.get("EZCLAW_NOTIFY", "1") != "0"
            if "on" in cmd:
                target = True
            elif "off" in cmd:
                target = False
            else:
                target = not current
            os.environ["EZCLAW_NOTIFY"] = "1" if target else "0"
            self.history_ansi.append(render_to_ansi(
                Text(f"Desktop notifications: {'ON' if target else 'OFF'}", style=DIM)
            ))
        elif cmd == "/authorize":
            self.agent.session_authorized = not self.agent.session_authorized
            status = "ENABLED (Always Allow)" if self.agent.session_authorized else "DISABLED (Ask per tool)"
            self.history_ansi.append(render_to_ansi(Text(f"Session authorization: {status}", style=ACCENT)))
        elif cmd.startswith("/reasoning"):
            # Toggle the unified reasoning panel (Tier 2.1). Accepts
            # `on` / `off` argument or no-arg for plain toggle.
            if " on" in cmd:
                target = True
            elif " off" in cmd:
                target = False
            else:
                target = not self.show_reasoning
            self.show_reasoning = target
            self.history_ansi.append(render_to_ansi(
                Text(f"Reasoning panel: {'ON' if target else 'OFF'}", style=DIM)
            ))
        elif cmd == "/cancel":
            # Escape hatch: unblock a pending ask_user so the user can
            # abandon a question without typing an answer (the agent
            # receives the empty-answer marker and decides what to do).
            if self._pending_user_question is not None:
                self._user_answer_queue.put("")
                self.history_ansi.append(render_to_ansi(
                    Text("Cancelled pending question.", style=DIM)
                ))
            else:
                self.history_ansi.append(render_to_ansi(
                    Text("Nothing to cancel.", style=DIM)
                ))
        elif cmd == "/settings":
            self._show_settings()
        elif cmd == "/queue":
            self._show_scheduled_queue()
        elif cmd == "/skills":
            self._show_skills()
        elif cmd.startswith("/phrases"):
            self._handle_phrases_command(cmd)
        elif cmd.startswith("/wisdom"):
            # Force a fresh reflection line. The fetch runs in the
            # background; the row updates as soon as the model replies.
            self._maybe_refresh_reflection(force=True)
            self.history_ansi.append(render_to_ansi(
                Text("Asking for a fresh reflection…", style=f"italic {DIM}")
            ))
        elif cmd.startswith("/memory"):
            query = cmd[len("/memory"):].strip()
            self._show_memory(query)
        elif cmd == "/help":
            self._show_help()
        elif cmd == "/diagnose":
            self.history_ansi.append(render_to_ansi(run_diagnostics_raw()))
        elif cmd == "/clear":
            self.agent.clear_session_history()
            self.history_ansi = []
            # Feature 2: /clear resets session accounting too.
            self.session_tokens_in = 0
            self.session_tokens_out = 0
            self.session_energy_wh = 0.0
            self.last_cook_time = 0.0
            self._open_files = {}
            self._active_editor_path = None
            self._paste_blocks.clear()
            self._paste_seq = 0
        elif cmd.startswith("/expand") or cmd.startswith("/collapse"):
            self._toggle_tool_expansion(cmd)
        elif cmd.startswith("/copy"):
            self._copy_to_clipboard(cmd)
        else:
            self.history_ansi.append(render_to_ansi(Text(f"Unknown command: {cmd}", style=ERR)))

        self._update_ui()

    def _build_agents_models_rows(self, history_size: int) -> list:
        """Per-role model rows for the settings panel.

        In multi-agent mode, expand each specialized agent's model on its
        own row prefixed with the model-family emoji (theme.model_emoji)
        so the user can see at a glance who runs what. In single-agent
        mode, show just the primary model.
        """
        rows: list = [
            ("Mode", "multi-agent" if ENABLE_MULTI_AGENT else "single-agent", "ENABLE_MULTI_AGENT env"),
        ]
        if ENABLE_MULTI_AGENT and hasattr(self.agent, "agents"):
            # Surface every role's model with its emoji. `self.agent.agents`
            # is the {name: SpecializedAgent} dict on MultiAgentSystem.
            for role_name, role_agent in self.agent.agents.items():
                model_name = getattr(role_agent, "model", "?")
                rows.append((
                    role_name,
                    f"{model_emoji(model_name)} {model_name}",
                    f"OLLAMA_{role_name.upper()}_MODEL env",
                ))
            # Architect runs through a separate client; pull its model too.
            arch_model = getattr(getattr(self.agent, "architect", None), "model", None)
            if arch_model:
                rows.append((
                    "architect",
                    f"{model_emoji(arch_model)} {arch_model}",
                    "OLLAMA_ARCHITECT_MODEL env",
                ))
        else:
            primary = getattr(self.agent, "model", "?")
            rows.append((
                "Primary model",
                f"{model_emoji(primary)} {primary}",
                "OLLAMA_MODEL env",
            ))
        rows.append(("History", f"{history_size} messages", "/clear to reset"))
        return rows

    def _show_settings(self) -> None:
        """Reorganized settings view — grouped by concern, each row shows
        the current value AND the command/keybind to change it."""
        from rich.table import Table
        notify_on = os.environ.get("EZCLAW_NOTIFY", "1") != "0"
        sound_on = bool(os.environ.get("EZCLAW_NOTIFY_SOUND", "default").strip())
        history_size = (
            len(self.agent.messages)
            if hasattr(self.agent, "messages") and self.agent.messages else 0
        )

        sections = [
            ("Display", [
                ("Thinking panel",     "on" if SHOW_THINKING else "off", "/thinking"),
                ("Strategy panel",     "expanded" if self.show_architect else "compact chip", "F3"),
                ("Tool panels",        "full" if not self.compact_tools else "compact (one-line)", "F4"),
                ("Mouse mode",         "scroll (selection disabled)" if self._mouse_capture else "select & copy", "F2"),
            ]),
            ("Behavior", [
                ("Session auth",       "always allow" if self.agent.session_authorized else "ask per tool", "/authorize"),
                ("User name",          self.user_name, "EZCLAW_USER env"),
            ]),
            ("Agents & models", self._build_agents_models_rows(history_size)),
            ("Notifications", [
                ("Desktop alerts",     "on" if notify_on else "off", "/notify"),
                ("Sound",              "on" if (notify_on and sound_on) else "off", "EZCLAW_NOTIFY_SOUND env"),
            ]),
            ("Scheduling", [
                ("Heartbeat poll",     "every 30s", "(automatic)"),
                ("View queue",         "—", "/queue"),
            ]),
        ]
        for title, rows in sections:
            t = Table.grid(padding=(0, 2))
            t.add_column(style=f"bold {PRIMARY}")
            t.add_column(style="")
            t.add_column(style=f"dim {DIM} italic")
            for label, value, control in rows:
                t.add_row(label, str(value), control)
            self.history_ansi.append(render_to_ansi(Panel(
                t, title=f"[bold]{title}[/bold]", border_style=f"dim {PRIMARY}", box=ROUNDED,
            )))

    def _show_scheduled_queue(self) -> None:
        """List active (Pending/Notified) scheduled tasks from heartbeat.md."""
        try:
            from scheduler import Scheduler
            tasks = Scheduler().list_pending()
        except Exception as e:
            self.history_ansi.append(render_to_ansi(
                Text(f"Error reading scheduler: {e}", style=ERR)
            ))
            return
        if not tasks:
            self.history_ansi.append(render_to_ansi(
                Panel(Text("No scheduled tasks pending.", style=f"dim {DIM} italic"),
                      title="queue", border_style=f"dim {PRIMARY}", box=ROUNDED)
            ))
            return
        from rich.table import Table
        t = Table.grid(padding=(0, 2))
        t.add_column(style=f"bold {PRIMARY}")
        t.add_column()
        t.add_column(style=f"dim {DIM}")
        t.add_column()
        for task in sorted(tasks, key=lambda x: x.time):
            t.add_row(f"#{task.id}", task.time_str, task.status, task.description)
        self.history_ansi.append(render_to_ansi(Panel(
            t, title=f"[bold]{len(tasks)} scheduled task{'s' if len(tasks) != 1 else ''}[/bold]",
            border_style=f"dim {PRIMARY}", box=ROUNDED,
        )))

    def _handle_phrases_command(self, cmd: str) -> None:
        """`/phrases`             — show how many variants live in each category.
        `/phrases refresh`     — ask the running LLM for fresh phrases,
                                 merged into the pool + persisted to
                                 ~/.ezclaw/phrase_pool.json so subsequent
                                 sessions inherit them.
        `/phrases reset`       — delete the cache file (in-memory state
                                 settles on next restart)."""
        import phrases as _ph
        rest = cmd[len("/phrases"):].strip().lower()
        from rich.table import Table

        if rest in ("", "show", "list"):
            t = Table.grid(padding=(0, 2))
            t.add_column(style=f"bold {PRIMARY}")
            t.add_column(style="")
            for name, n in _ph.counts().items():
                t.add_row(name, str(n))
            self.history_ansi.append(render_to_ansi(Panel(
                t,
                title=f"[bold]Phrase pool[/bold]  [dim {DIM}](/phrases refresh to add more)[/dim {DIM}]",
                border_style=f"dim {PRIMARY}",
                box=ROUNDED,
            )))
            return

        if rest == "refresh":
            # Use the architect's client and a quick model — the call is
            # one-shot and creative, not load-bearing. Architect client
            # works for both ollama and deepseek backends.
            client = getattr(getattr(self.agent, "architect", None), "client", None)
            model = getattr(getattr(self.agent, "architect", None), "model", None)
            if client is None or model is None:
                # Fall back to a per-agent client (single-agent mode).
                client = getattr(self.agent, "client", None)
                model = getattr(self.agent, "model", None)
            if client is None or model is None:
                self.history_ansi.append(render_to_ansi(
                    Text("No LLM client available to refresh phrases.", style=ERR)
                ))
                return

            self.history_ansi.append(render_to_ansi(
                Text(f"Asking {model} for fresh phrases…", style=f"italic {DIM}")
            ))
            self._update_ui()
            added = _ph.augment_with_llm(client, model)
            if not added:
                self.history_ansi.append(render_to_ansi(
                    Text("No new phrases added (model returned nothing usable).",
                         style=f"dim {WARN}")
                ))
                return

            t = Table.grid(padding=(0, 2))
            t.add_column(style=f"bold {ACCENT}")
            t.add_column(style="")
            total = 0
            for name, plist in added.items():
                t.add_row(f"+{len(plist)} {name}", ", ".join(plist[:6]))
                total += len(plist)
            self.history_ansi.append(render_to_ansi(Panel(
                t,
                title=f"[bold {ACCENT}]Added {total} new phrase{'s' if total != 1 else ''}[/bold {ACCENT}]",
                border_style=f"dim {ACCENT}",
                box=ROUNDED,
            )))
            return

        if rest == "reset":
            _ph.reset_cache()
            self.history_ansi.append(render_to_ansi(Text(
                "Phrase cache deleted. Restart the session to drop the in-memory additions.",
                style=f"dim {DIM}",
            )))
            return

        self.history_ansi.append(render_to_ansi(Text(
            "Usage: /phrases [refresh | reset]", style=ERR,
        )))

    def _show_skills(self) -> None:
        """List skills the agent has learned (in ~/.ezclaw/skills/)."""
        try:
            from tools import SKILLS_DIR, _migrate_legacy_skills_once
            _migrate_legacy_skills_once()
            if not os.path.isdir(SKILLS_DIR):
                names = []
            else:
                names = sorted(
                    f[:-3] for f in os.listdir(SKILLS_DIR) if f.endswith(".md")
                )
        except Exception as e:
            self.history_ansi.append(render_to_ansi(
                Text(f"Error reading skills: {e}", style=ERR)
            ))
            return
        if not names:
            body = Text("No skills learned yet. Tell the agent to remember a procedure and it'll save one here.",
                        style=f"dim {DIM} italic")
        else:
            body = Text("\n".join(f"• {n}" for n in names), style="")
        self.history_ansi.append(render_to_ansi(Panel(
            body, title=f"[bold]{len(names)} learned skill{'s' if len(names) != 1 else ''}[/bold]",
            border_style=f"dim {PRIMARY}", box=ROUNDED,
        )))

    def _show_memory(self, query: str) -> None:
        """Inspect the agent's memory store. With no query, show recent
        facts. With a query, run hybrid semantic search and show matches."""
        try:
            db = self.agent.db
            if query:
                results = db.search_memories_hybrid(query, alpha=0.6, threshold=0.2)
                heading = f"memory search: \"{query[:60]}\""
            else:
                results = db.search_memories("", limit=20)
                heading = "recent memories"
        except Exception as e:
            self.history_ansi.append(render_to_ansi(
                Text(f"Error reading memory: {e}", style=ERR)
            ))
            return
        if not results:
            body = Text("No matching memories." if query else "Memory is empty.",
                        style=f"dim {DIM} italic")
        else:
            body = Text("\n".join(f"• {m}" for m in results[:20]), style="")
        self.history_ansi.append(render_to_ansi(Panel(
            body, title=f"[bold]{heading}[/bold]",
            border_style=f"dim {PRIMARY}", box=ROUNDED,
        )))

    def _show_help(self) -> None:
        """Help grouped by category — easier to scan than a flat command list."""
        sections = [
            ("Input & navigation", [
                ("[Enter]",             "send the prompt"),
                ("[Shift+Enter]",       "newline (requires kitty / WezTerm / iTerm2 with CSI u)"),
                ("[Ctrl+J]",            "newline (universal, works in any terminal)"),
                ("[Alt+Enter]",         "newline (Esc-prefix terminals; most Linux + iTerm2)"),
                ("end line with \\ + Enter", "newline (always works — backslash gets swapped for \\n)"),
                ("paste",               "large pastes collapse to [pasted #N: L lines, C chars]"),
                ("[PgUp] / [PgDn]",     "scroll one screen"),
                ("[Home] / [End]",      "jump to top / bottom"),
                ("[Ctrl+C]",            "interrupt / quit"),
            ]),
            ("Panel toggles", [
                ("[F1]",                "open this help"),
                ("[F2]",                "copy mode (terminal-native mouse selection)"),
                ("[F3]",                "expanded architect-strategy panels"),
                ("[F4]",                "compact / full tool-panel layout"),
                ("[F5]",                "show / hide right-side editor pane"),
                ("[F6] / [F7]",         "cycle editor tabs (prev / next)"),
                ("[Alt+↑] / [Alt+↓]",   "scroll editor pane (one line)"),
                ("[Alt+PgUp/PgDn]",     "scroll editor pane (10 lines)"),
                ("[Ctrl+R]",            "toggle reasoning visibility"),
                ("[Esc]",               "(while interactive shell active) toggle input → subprocess / chat"),
            ]),
            ("Authorization (when 🛡 panel is up)", [
                ("[Y]",                 "allow this tool for the rest of the session"),
                ("[O]",                 "allow once only"),
                ("[N]",                 "deny this call"),
                ("[A]",                 "allow all tools session-wide"),
            ]),
            ("Slash commands — inspection", [
                ("/settings",           "all settings + how to change each"),
                ("/queue",              "list active scheduled tasks (incl. recurring)"),
                ("/skills",             "list learned skills (~/.ezclaw/skills/)"),
                ("/memory [query]",     "show stored memories (optional search)"),
                ("/diagnose",           "GPU / Ollama / system probe"),
            ]),
            ("Slash commands — display", [
                ("/thinking [on|off]",  "show / hide reasoning content"),
                ("/notify  [on|off]",   "desktop notifications on/off"),
                ("/reasoning [on|off]", "alias for /thinking"),
                ("/expand [N|all]",     "expand a tool panel (defaults to last)"),
                ("/collapse [N|all]",   "re-collapse a tool panel"),
                ("/copy [last|all|N]",  "copy assistant text (OSC52 clipboard)"),
            ]),
            ("Slash commands — flavor", [
                ("/wisdom",             "refresh the reflection line in the status bar"),
                ("/phrases",            "show built-in status phrase counts"),
                ("/phrases refresh",    "ask the LLM for new phrases, persist to disk"),
                ("/phrases reset",      "revert phrase pool to defaults, delete cache"),
            ]),
            ("Slash commands — session", [
                ("/clear",              "wipe history + reset session tokens / energy / cook time"),
                ("/authorize",          "toggle session-wide tool authorization"),
                ("/cancel",             "cancel a pending ask_user question"),
                ("exit / quit",         "leave EzClaw"),
            ]),
            ("Scheduled tasks (agent tools)", [
                ("schedule_task(time, description)",  "one-shot at YYYY-MM-DD HH:MM"),
                ("schedule_task(..., recurrence=...)", "recurring: every Nm/Nh/Nd, hourly, daily, weekly, weekdays"),
                ("unschedule_task(id)",                "cancel a pending/recurring task by ID"),
                ("list_scheduled_tasks()",             "see the queue (same as /queue)"),
            ]),
            ("Direct commands (bypass the LLM)", [
                ("list tasks / show tasks",   "→ list_scheduled_tasks (instant)"),
                ("list skills",                "→ list_skills"),
                ("system info",                "→ get_system_info"),
                ("what time is it",            "→ current_datetime"),
            ]),
        ]
        from rich.table import Table
        for title, rows in sections:
            t = Table.grid(padding=(0, 2))
            t.add_column(style=f"bold {PRIMARY}")
            t.add_column(style=f"dim {DIM}")
            for k, v in rows:
                t.add_row(k, v)
            self.history_ansi.append(render_to_ansi(Panel(
                t, title=f"[bold]{title}[/bold]",
                border_style=f"dim {PRIMARY}", box=ROUNDED,
            )))

    def _copy_to_clipboard(self, cmd: str):
        """Push content to the system clipboard via OSC52.

        Forms: /copy           -> last assistant response
               /copy all       -> entire visible chat history
               /copy last N    -> last N assistant responses (joined)
        """
        import base64

        parts = cmd.split()
        target = parts[1] if len(parts) >= 2 else "last"

        # Pull conversation pairs the multi-agent system tracked, or fall
        # back to the rendered history_ansi if we're in single-agent mode.
        history = getattr(self.agent, "_conversation_history", None)
        if history is None:
            messages = getattr(self.agent, "messages", []) or []
            history = [
                {"user": m.get("content", ""), "assistant": ""}
                for m in messages if m.get("role") == "user"
            ]

        if not history:
            self.history_ansi.append(render_to_ansi(Text("Nothing to copy yet.", style=ERR)))
            return

        if target == "all":
            payload = "\n\n".join(
                f"You: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
                for turn in history
            )
        elif target == "last":
            payload = history[-1].get("assistant", "") or ""
        else:
            try:
                n = int(target)
                payload = "\n\n".join(
                    turn.get("assistant", "") for turn in history[-n:] if turn.get("assistant")
                )
            except ValueError:
                self.history_ansi.append(render_to_ansi(
                    Text("Usage: /copy | /copy all | /copy <N>", style=ERR)
                ))
                return

        if not payload.strip():
            self.history_ansi.append(render_to_ansi(
                Text("Nothing to copy (target was empty).", style=f"dim {DIM}")
            ))
            return

        # OSC52: ESC ] 52 ; c ; <base64> BEL.  Most modern terminals support
        # this (alacritty, kitty, foot, iTerm2, modern xterm, wezterm).
        b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        # Write directly to stdout — prompt_toolkit will redraw on top after.
        sys.stdout.write(f"\033]52;c;{b64}\007")
        sys.stdout.flush()

        n_chars = len(payload)
        self.history_ansi.append(render_to_ansi(
            Text(f"✓ Copied {n_chars} chars to clipboard (OSC52). If clipboard is empty, your terminal may not support OSC52 — toggle copy mode with F2 and select manually.",
                 style=f"dim {ACCENT}")
        ))

    def _toggle_tool_expansion(self, cmd: str):
        """Handle /expand and /collapse commands.

        Forms: /expand N | /expand last | /expand all
               /collapse N | /collapse all
        Only operates on tools of the currently-rendering turn. Historical
        tool panels already frozen into history_ansi cannot be re-expanded
        from here — re-issue the prompt if you need them open.
        """
        parts = cmd.split()
        action = parts[0]
        target_state = (action == "/expand")
        if not self.tool_executions:
            self.history_ansi.append(render_to_ansi(
                Text("No tools in the current turn to expand. (Historical tools are frozen.)",
                     style=f"dim {DIM}")
            ))
            return

        if len(parts) < 2:
            # Default target for /expand is "last", for /collapse is "all".
            target = "last" if target_state else "all"
        else:
            target = parts[1].strip()

        if target == "all":
            for t in self.tool_executions:
                t["expanded"] = target_state
        elif target == "last":
            self.tool_executions[-1]["expanded"] = target_state
        else:
            try:
                idx = int(target) - 1
                if 0 <= idx < len(self.tool_executions):
                    self.tool_executions[idx]["expanded"] = target_state
                else:
                    self.history_ansi.append(render_to_ansi(
                        Text(f"No tool with index {target}. (Tools 1..{len(self.tool_executions)})", style=ERR)
                    ))
                    return
            except ValueError:
                self.history_ansi.append(render_to_ansi(
                    Text(f"Usage: {action} <N> | {action} last | {action} all", style=ERR)
                ))
                return

    def _agent_worker(self, user_input):
        gen = self.agent.chat_stream(user_input)
        try:
            chunk = next(gen)
            while True:
                self.last_chunk_time = time.time()
                if chunk["type"] == "plan_created":
                    self.current_plan = chunk["plan"]
                    self._last_task_states = {t.id: t.status for t in self.current_plan.tasks}
                elif chunk["type"] == "plan_update":
                    new_plan = chunk["plan"]
                    now = time.time()
                    prev_in_progress = (
                        {t.id for t in self.current_plan.tasks if t.status == "in_progress"}
                        if self.current_plan else set()
                    )
                    for task in new_plan.tasks:
                        prev = self._last_task_states.get(task.id)
                        if prev is not None and prev != task.status:
                            self._task_flash_until[task.id] = now + 0.15
                        self._last_task_states[task.id] = task.status
                    self.current_plan = new_plan
                    new_in_progress = {t.id for t in new_plan.tasks if t.status == "in_progress"}
                    if new_in_progress != prev_in_progress:
                        # The previously-active task moved; any held intent is
                        # about the prior task. Wait for the next intent chunk.
                        self.architect_intent = None
                elif chunk["type"] == "halt":
                    self.halted = True
                elif chunk["type"] == "intent":
                    self.architect_intent = chunk
                    new_role = chunk.get("agent")
                    if new_role:
                        if self._last_chip_role and new_role != self._last_chip_role:
                            self._chip_flash_until = time.time() + 0.15
                        self._last_chip_role = new_role
                        self.current_role = new_role
                    # Tier 2.1: archive each architect step into the
                    # unified reasoning log so we can show the FULL
                    # history of decisions, not just the latest.
                    refl = chunk.get("reflection") or {}
                    summary_bits = []
                    for k in ("goal", "observation", "critical_thinking"):
                        v = (refl.get(k) or "").strip()
                        if v:
                            summary_bits.append(f"{k}: {v}")
                    body = " · ".join(summary_bits) or (chunk.get("reasoning") or "")
                    if body:
                        self.reasoning_log.append({
                            "kind": "architect",
                            "label": f"architect → {new_role or '?'}",
                            "body": body.strip(),
                            "time": time.time(),
                        })
                elif chunk["type"] == "reasoning":
                    self.reasoning_chunks.append(chunk["content"])
                    # Also feed into the unified log for the new panel.
                    text = (chunk.get("content") or "").strip()
                    if text:
                        self.reasoning_log.append({
                            "kind": "agent",
                            "label": self.current_role or "agent",
                            "body": text,
                            "time": time.time(),
                        })
                elif chunk["type"] == "content":
                    self.current_response_parts.append(chunk["content"])
                    # Feature 2: tally output tokens (char/4 heuristic).
                    self.session_tokens_out += max(1, len(chunk["content"]) // 4)
                elif chunk["type"] == "status":
                    self.current_status = chunk["content"].strip()
                    # Pipe self-check verdicts into the reasoning log so
                    # users can see why the loop extended a plan.
                    if self.current_status.startswith("self-check:"):
                        self.reasoning_log.append({
                            "kind": "self_check",
                            "label": "self-check",
                            "body": self.current_status[len("self-check:"):].strip(),
                            "time": time.time(),
                        })
                    # The architect chip / spinner status line already
                    # surface the current status; don't ALSO stack each
                    # one as a dim italic line in side_messages — that
                    # produced 20+ duplicate lines on long turns.
                    # Side messages are now reserved for true side-channel
                    # events (memory stored, context augmented, blockers).
                    if self.current_status.startswith("⚠") or self.current_status.startswith("⚠ blocker"):
                        self.side_messages.append(self.current_status)
                    # Attention triggers: any status starting with ⚠ blocker
                    # means the architect has stopped and needs user input.
                    # Long-running task completion is also a notify
                    # candidate so the user can come back from another
                    # window when ezclaw is done.
                    self._maybe_notify_from_status(self.current_status)
                elif chunk["type"] == "auth_required":
                    self.auth_active = True
                    self._update_ui()
                    choice = self._ask_auth(chunk)
                    chunk = gen.send(choice)
                    self.auth_active = False
                    continue
                elif chunk["type"] == "memory_stored":
                    self.side_messages.append(f"📝 {chunk['fact']}")
                    # The agent might have just stored "my name is X" via
                    # the remember tool — re-resolve the bubble title so
                    # subsequent turns pick up the new name.
                    fact_text = str(chunk.get("fact", "")).lower()
                    if any(k in fact_text for k in ("name is", "i am ", "i'm ", "name:")):
                        self.user_name = self._resolve_user_name()
                elif chunk["type"] == "context_augmented":
                    for m in chunk["memories"]:
                        self.side_messages.append(f"📎 {m}")
                elif chunk["type"] == "skill_offer":
                    # The architect detected the user educated us into
                    # success and drafted a skill. Park it for the user
                    # to accept/decline via Y/N keys.
                    self._pending_skill_offer = chunk.get("draft")
                elif chunk["type"] == "tool_start":
                    # Feature 3: light the right-side editor pane when a
                    # file is touched. write_file carries full content
                    # in its `content` arg; apply_diff carries a `diff`
                    # we can render as-is in a separate "diff" pseudo
                    # entry so the user can see the patch take effect.
                    name = chunk.get("name")
                    args = chunk.get("arguments") or {}
                    if name == "write_file":
                        ed_path = args.get("path") or args.get("file_path") or ""
                        ed_content = args.get("content", "")
                        if ed_path:
                            self._open_or_update_file(ed_path, ed_content, fresh=True)
                            self._active_editor_path = ed_path
                    elif name == "apply_diff":
                        ed_path = args.get("path") or args.get("file_path") or ""
                        diff_text = args.get("diff", "")
                        if ed_path:
                            # Show the diff itself in the editor pane until
                            # we have the post-image (which we don't from
                            # the args alone). Tagged with a `.diff`
                            # suffix so the lexer treats it as a diff.
                            tab_path = f"{ed_path}  (patch)"
                            self._open_or_update_file(tab_path, diff_text, fresh=True)
                            self._active_editor_path = tab_path
                    is_int = chunk.get("interactive", False)
                    # Tag the call with the plan step it belongs to so the
                    # plan panel can render tools nested under their task.
                    # Cascading fallback so EVERY tool gets a home in the
                    # plan tree (was previously orphaned as a flat panel
                    # above the plan when none of the conditions met):
                    #   1. architect's explicit current_task_id
                    #   2. any in_progress task
                    #   3. most recently touched task (latest non-pending,
                    #      scanning from the end — this is the task the
                    #      architect was working on, just hasn't formally
                    #      transitioned to in_progress this turn)
                    #   4. last task in the plan (last resort — guarantees
                    #      no orphans when a plan exists)
                    active_task_id = None
                    if self.current_plan is not None:
                        active_task_id = self.current_plan.current_task_id
                        if active_task_id is None:
                            for t in self.current_plan.tasks:
                                if t.status == "in_progress":
                                    active_task_id = t.id
                                    break
                        if active_task_id is None:
                            for t in reversed(self.current_plan.tasks):
                                if t.status != "pending":
                                    active_task_id = t.id
                                    break
                        if active_task_id is None and self.current_plan.tasks:
                            active_task_id = self.current_plan.tasks[-1].id
                    self.tool_executions.append({
                        "name": chunk["name"],
                        "args": chunk["arguments"],
                        "result": None,
                        "interactive": is_int,
                        "start_time": time.time(),
                        "expanded": False,
                        "task_id": active_task_id,
                    })
                elif chunk["type"] == "tool_end":
                    for tool in reversed(self.tool_executions):
                        if tool["name"] == chunk["name"] and tool["result"] is None:
                            tool["result"] = chunk["result"]
                            break
                
                self._update_ui()
                chunk = next(gen)
        except StopIteration:
            pass
        except Exception as e:
            self.current_response_parts.append(f"\n[bold {ERR}]Error:[/bold {ERR}] {str(e)}")
        
        # Finish generating
        self.is_generating = False
        # Features 1+2: freeze cook time and account for energy spent on this
        # turn. The energy figure is wall-clock × TDP and is therefore an
        # overestimate (GPU isn't pinned the whole time); see spec.
        cook = (time.time() - self.generation_start_time) if self.generation_start_time else 0.0
        self.last_cook_time = cook
        if not self.halted and cook > 0:
            self.session_energy_wh += cook * self.gpu_tdp_watts / 3600.0
        # Inline-save pass: parse tagged code blocks, save them, rewrite
        # the joined content with badges. Skipped if halted (the turn will
        # resume; saves wait until the user truly ends the turn).
        if not self.halted and self.current_response_parts:
            self._process_inline_saves()
        if not self.halted:
            final_renderable = self._get_current_renderable_ansi()
            # Feature 1: append a dim cook-time annotation under the bubble
            # so the user can scan "how long did each response take" while
            # scrolling history. Format: nicely placed on its own
            # right-margined line with a clock glyph + human-readable
            # duration. Previously a bare "  · 12.3s" with default font
            # rendering looked detached/orphaned.
            if cook > 0:
                final_renderable += render_to_ansi(
                    Text(f"⏱  {_fmt_duration(cook)}", style=f"italic {DIM}", justify="right")
                )
            self.history_ansi.append(final_renderable)
            self.current_response_parts = []
            self.reasoning_chunks = []
            self.reasoning_log = []
            self.tool_executions = []
            self.side_messages = []
            # Feature 3: editor pane closes at the end of a turn. The
            # ConditionalContainer auto-hides on the next render once
            # _open_files is empty.
            self._open_files = {}
            self._active_editor_path = None
        # End-of-turn: refresh the reflection line in the background if
        # the gate (15 min by default) has elapsed. Free if not due.
        self._maybe_refresh_reflection()
        self._update_ui()

    def _resolve_user_name(self) -> str:
        """Decide what to put on the user-message bubble title.

        Priority:
          1. EZCLAW_USER (or EZCLAW_USER_NAME) env var — explicit override
          2. The agent's memory — search for "my name is X" / "I am X" /
             "user's name is X" facts (the `remember` tool may have stored
             one in a past session)
          3. System login name from getpass.getuser(), capitalized
          4. The literal string "User" if even getpass fails

        Cached on self.user_name; re-resolved on memory_stored chunks so a
        live `remember("my name is …")` updates the bubble immediately.
        """
        explicit = os.environ.get("EZCLAW_USER") or os.environ.get("EZCLAW_USER_NAME")
        if explicit and explicit.strip():
            return explicit.strip()

        try:
            memories = self.agent.db.search_memories_hybrid(
                "my name is", alpha=0.6, threshold=0.3
            )
            pat = re.compile(
                r"(?:my name is|i am|i'm|user'?s? name is|name:|user:)\s+"
                r"([A-Za-z][A-Za-z\-]{0,30})",
                re.IGNORECASE,
            )
            for m in memories or []:
                match = pat.search(str(m))
                if match:
                    name = match.group(1).strip()
                    if name and name.lower() not in (
                        "the", "a", "an", "ezclaw", "user", "trying", "going",
                    ):
                        return name.capitalize()
        except Exception:
            pass

        try:
            import getpass
            login = getpass.getuser()
            if login:
                return login.capitalize()
        except Exception:
            pass
        return "User"

    def _process_inline_saves(self) -> None:
        """End-of-turn pass: parse tagged code blocks from the response,
        save them, rewrite the joined content with badges. Best-effort —
        any exception is logged to stderr and swallowed."""
        try:
            from inline_code_saver import parse_tagged_blocks, plan_saves, apply_save, SaveResult
            joined = "".join(self.current_response_parts)
            blocks = parse_tagged_blocks(joined)
            if not blocks:
                return
            plans = plan_saves(blocks, workspace_root="workspace")
            results = []
            for plan in plans:
                if plan.error is not None:
                    results.append(SaveResult(plan=plan, status="rejected", final_path=None))
                    continue
                if plan.exists:
                    choice = self._ask_save_collision(plan)
                    results.append(apply_save(plan, choice))
                else:
                    results.append(apply_save(plan, "write"))
            rewritten = self._rewrite_with_badges(joined, results)
            self.current_response_parts = [rewritten]
            for r in results:
                if r.status in ("succeeded", "renamed"):
                    self._record_inline_save_action(r)
        except Exception as e:
            import sys
            print(f"[inline-save] pass failed: {e}", file=sys.stderr)

    def _rewrite_with_badges(self, joined: str, results: list) -> str:
        """Replace each tagged opening fence with a plain `lang` fence
        and prepend a blockquote badge indicating save status."""
        import os as _os
        out = joined
        # Apply in reverse offset order so earlier offsets stay valid.
        for r in sorted(results, key=lambda x: x.plan.block.start, reverse=True):
            block = r.plan.block
            if r.status == "succeeded":
                badge = f"> 💾 **saved →** `workspace/{block.path}`\n\n"
                new_fence = f"```{block.lang}" if block.lang else "```"
            elif r.status == "renamed":
                final_rel = _os.path.relpath(
                    r.final_path, _os.path.abspath("workspace"),
                )
                badge = f"> 💾 **saved →** `workspace/{final_rel}` (renamed; original existed)\n\n"
                new_fence = f"```{block.lang}" if block.lang else "```"
            elif r.status == "skipped":
                badge = f"> ⊘ **skipped →** `workspace/{block.path}` (existed)\n\n"
                new_fence = f"```{block.lang}" if block.lang else "```"
            elif r.status == "rejected":
                err_msg = r.plan.error or "rejected"
                badge = f"> ⚠ **rejected →** `{block.plan.path if hasattr(block, 'plan') else block.path}` ({err_msg}; kept inline)\n\n"
                new_fence = f"```{block.lang}" if block.lang else "```"
            else:
                continue
            opener_len = len(block.raw_open_fence)
            opener_end = block.start + opener_len
            out = out[:block.start] + badge + new_fence + out[opener_end:]
        return out

    def _ask_save_collision(self, plan) -> str:
        """Block on user input for a collision. Returns 'write', 'skip',
        or 'rename'. Defaults to 'rename' (safe non-destructive) on UI
        failure or timeout. For v1, the CLI's keystroke wiring isn't yet
        extended to recognize O/S/R while a collision is pending; until
        that exists, this method falls back to 'rename' immediately to
        guarantee non-destructive behavior."""
        try:
            self.side_messages.append(
                f"💾 collision on {plan.block.path} — auto-renamed to next free suffix "
                f"(extend cli to enable O/S/R keystrokes)"
            )
            self._update_ui()
            return "rename"
        except Exception:
            return "rename"

    def _record_inline_save_action(self, result) -> None:
        """Record a successful inline-save as an action row."""
        try:
            import os as _os
            import json as _json
            import pickle as _pickle
            from tools import get_session_context
            from embed import embed as _embed
            session_id = get_session_context()
            if session_id is None:
                return
            db = getattr(self.agent, "db", None)
            if db is None:
                return
            if result.status == "succeeded":
                path = result.plan.block.path
            else:
                path = _os.path.relpath(
                    result.final_path, _os.path.abspath("workspace"),
                )
            summary = f"saved {_os.path.basename(path)}"
            outcome = "succeeded" if result.status == "succeeded" else "partial"
            try:
                vec = _embed(summary)
                emb_blob = _pickle.dumps(vec)
            except Exception:
                emb_blob = None
            db.add_action(
                session_id=session_id,
                tool="inline_save",
                args_json=_json.dumps({"path": path}),
                summary=summary,
                why=None,
                outcome=outcome,
                error_excerpt=None,
                embedding=emb_blob,
            )
        except Exception as e:
            import sys
            print(f"[inline-save] action record failed: {e}", file=sys.stderr)

    def _maybe_notify_from_status(self, status: str) -> None:
        """Send a desktop notification when a status string signals that
        ezclaw needs the user's attention or has finished a long run.

        Three trigger families:
          - "⚠ blocker" prefix          → critical (loop halted, needs input)
          - "🦀 all snipped and shipped" → low (long task done, may have stepped away)
          - "🦀 …" success variants       → low (same, alternate wording)
        Notifications are throttled so the same status doesn't fire twice
        in a row, and the "completed" notify only fires when the turn
        actually took >30s (otherwise it's just a chat reply, no need).
        """
        from notifications import notify, URGENCY_CRITICAL, URGENCY_LOW
        if status == getattr(self, "_last_notified_status", None):
            return
        self._last_notified_status = status

        if status.startswith("⚠ blocker"):
            notify(
                "🦀 ezclaw blocked",
                status.lstrip("⚠ ").strip(),
                urgency=URGENCY_CRITICAL,
            )
            return

        # Completion notifications only for runs that took some time.
        # A 2-second greeting reply doesn't need an OS notification.
        completion_markers = (
            "all snipped and shipped",
            "catch landed",
            "back to the shore",
            "shell sealed",
        )
        if any(m in status for m in completion_markers):
            elapsed = time.time() - self.generation_start_time if self.generation_start_time else 0
            if elapsed >= 30:
                notify(
                    "🦀 ezclaw finished",
                    f"Task done in {elapsed:.0f}s. Check the chat for details.",
                    urgency=URGENCY_LOW,
                )

    def _ask_auth(self, chunk):
        self.current_auth_chunk = chunk
        self.auth_active = True
        self.input_field.read_only = True
        self._update_ui()

        # The auth prompt blocks all further agent progress until the
        # user responds. Ping them via the OS notification system so
        # they can come back to ezclaw from another window.
        from notifications import notify, URGENCY_CRITICAL
        tool_name = chunk.get("name", "tool")
        notify(
            f"🦀 ezclaw needs approval",
            f"Run `{tool_name}`?  Switch to ezclaw to answer.",
            urgency=URGENCY_CRITICAL,
        )

        choice = self.auth_queue.get()
        
        self.auth_active = False
        self.current_auth_chunk = None
        self.input_field.read_only = False
        self._update_ui()
        return choice

    def _run_interactive_tool(self, chunk):
        pass

    def _scroll_to_bottom(self):
        # Move cursor to end of buffer for native scrolling
        self.history_buffer.cursor_position = len(self.history_buffer.text)

    def run(self):
        # Start heartbeat monitor
        threading.Thread(target=self._heartbeat_monitor, daemon=True).start()
        # Initial update
        self._update_ui()
        # Kick off the animation loop NOW (was only starting on first user
        # input). This makes the welcome-banner gradient flow and 🦀 color
        # pulse visible from the moment the app starts, not just during
        # generation.
        self._start_animation_loop()
        # Fetch an initial reflection in the background so the line below
        # the status bar gets populated within a few seconds of startup —
        # the welcome banner has time to be visible before this lands.
        self._maybe_refresh_reflection(force=True)
        # Conservative keyboard-protocol push: enable ONLY xterm
        # modifyOtherKeys level 2. This is narrower than kitty's CSI u
        # disambiguate flag — it adds distinct reporting for modified
        # printable keys (so Shift+Enter arrives as \x1b[27;2;13~
        # which we map to SHIFT_ENTER → newline) but leaves plain
        # arrow keys, DEL, Home/End, and other navigation keys in
        # their legacy encoding that prompt_toolkit handles natively.
        # Kitty supports modifyOtherKeys for xterm compat; in
        # terminals that don't, this is a no-op and Shift+Enter falls
        # back to indistinguishable-from-Enter — users use Ctrl+J or
        # Alt+Enter for newlines instead.
        def _on_first_render_done(_app):
            try:
                out = self.app.output
                out.write_raw("\x1b[>4;2m")  # modifyOtherKeys level 2
                out.flush()
            except Exception:
                pass
            try:
                self.app.after_render -= _on_first_render_done
            except Exception:
                pass

        self.app.after_render += _on_first_render_done

        try:
            self.app.run()
        finally:
            # Restore modifyOtherKeys to off on exit.
            try:
                self.app.output.write_raw("\x1b[>4;0m")
                self.app.output.flush()
            except Exception:
                pass

    def _heartbeat_monitor(self):
        """Poll the Scheduler every 30s for due tasks. When a task fires,
        feed its description into the agent as if the user had typed it —
        the architect plans, sub-agents execute. If the agent is already
        busy on a user-driven turn, the firing is deferred to the next tick.
        """
        from scheduler import Scheduler
        sched = Scheduler()
        while True:
            try:
                # Don't fire scheduled tasks while a user-driven turn is in
                # flight — they'd collide in the same generator. Try again
                # next tick.
                if not self.is_generating:
                    due = sched.find_due()
                    for task in due:
                        # Mark Notified immediately so a long-running auto-run
                        # doesn't get re-fired on the next tick.
                        sched.mark_status(task.id, "Notified")
                        self._fire_scheduled_task(task, sched)
                        # Only one auto-fire per tick — the agent is now busy
                        # for this one; remaining due tasks wait.
                        break
            except Exception:
                pass
            time.sleep(30)

    def _fire_scheduled_task(self, task, sched):
        """Show a scheduled-task banner and run the task description through
        the agent as if it were a user prompt. Status moves Notified → Done
        when the agent finishes, or Failed if it raises."""
        from rich.panel import Panel as _Panel
        from notifications import notify, URGENCY_NORMAL

        # OS-level desktop notification — the user may have stepped away
        # from ezclaw when this fires. Title says what's happening; body
        # carries the task description so they can decide if it needs
        # their attention or if they can let it run.
        notify(
            f"🦀 ezclaw scheduled task #{task.id}",
            task.description,
            urgency=URGENCY_NORMAL,
        )

        banner = _Panel(
            Text(f"🔔 [#{task.id}] {task.description}", style=f"bold {WARN}"),
            title=f"[bold {WARN}]Scheduled task auto-running[/bold {WARN}]",
            border_style=WARN,
            box=ROUNDED,
        )
        self.history_ansi.append(render_to_ansi(banner))
        self._force_scroll_next_update = True

        # Reuse the same per-turn reset that handle_input does — clean
        # plan state, fresh chip, etc.
        self.is_generating = True
        self.current_response_parts = []
        self.reasoning_chunks = []
        self.reasoning_log = []
        self.tool_executions = []
        self.side_messages = []
        self.architect_intent = None
        self.current_role = None
        self._last_chip_role = None
        self._chip_flash_until = 0.0
        self.current_plan = None
        self._last_task_states = {}
        self._task_flash_until = {}
        self.current_status = "auto-running…"
        self.generation_start_time = time.time()
        self.last_chunk_time = time.time()

        self._update_ui()
        self._start_animation_loop()

        def worker():
            try:
                self._agent_worker(task.description)
                # Reach here only after the worker generator exhausts.
                # _agent_worker handles its own is_generating reset.
                # For recurring tasks this re-arms the row to the next
                # occurrence; for one-shots it marks Done as before.
                sched.complete_or_reschedule(task.id, success=True)
            except Exception:
                sched.complete_or_reschedule(task.id, success=False)

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    ui = ChatUI()
    ui.run()
