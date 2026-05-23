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
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.layout.layout import Layout
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
from theme import THEME, TITLE_GRADIENT, ACTIVITY_FRAMES, MASCOT

PRIMARY = THEME.palette.primary
SECONDARY = THEME.palette.secondary
ACCENT = THEME.palette.accent
WARN = THEME.palette.warn
ERR = THEME.palette.err
DIM = THEME.palette.dim

# Global console for rendering
console = Console(file=io.StringIO(), force_terminal=True, width=100)

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
        self.tool_executions = []
        self.side_messages = []
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

        # Mouse capture is toggleable so the user can drop into terminal-native
        # text selection (F2). When True, prompt_toolkit owns the mouse (enables
        # our ScrollUp/Down bindings); when False, the terminal handles
        # click-drag selection so the user can copy text from the chat.
        self._mouse_capture = True
        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=True,
            mouse_support=Condition(lambda: self._mouse_capture),
        )

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
            }

            def on_output(data: bytes):
                # Cap the in-memory buffer so a runaway subprocess can't
                # balloon UI state. Keep the last ~64 KB.
                live_buffer.append(data)
                total = sum(len(c) for c in live_buffer)
                while total > 64 * 1024 and len(live_buffer) > 1:
                    total -= len(live_buffer.pop(0))
                try:
                    self._update_ui()
                except Exception:
                    pass

            def input_provider():
                try:
                    return input_q.get_nowait()
                except queue.Empty:
                    return None

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

    def _setup_keybindings(self):
        @self.kb.add('c-c')
        def _(event):
            if self.is_generating:
                self.is_generating = False
            else:
                event.app.exit()

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

        @self.kb.add('y', filter=Condition(lambda: self.auth_active))
        def _(event):
            self.auth_queue.put("allow")

        @self.kb.add('n', filter=Condition(lambda: self.auth_active))
        def _(event):
            self.auth_queue.put("deny")

        @self.kb.add('a', filter=Condition(lambda: self.auth_active))
        def _(event):
            self.auth_queue.put("allow_session")

        @self.kb.add('enter', filter=Condition(lambda: not self.auth_active))
        def _(event):
            text = self.input_field.text.strip()
            if text:
                # Add to history
                self.input_field.buffer.append_to_history()
                self.input_field.text = ""
                self.handle_input(text)

        # Mouse scroll speed improvements
        @self.kb.add(Keys.ScrollUp)
        def _(event):
            # Scroll the history window up by 3 lines
            self.history_window.vertical_scroll = max(0, self.history_window.vertical_scroll - 3)

        @self.kb.add(Keys.ScrollDown)
        def _(event):
            # Scroll the history window down by 3 lines
            if self.history_window.render_info:
                # We can't easily know the max scroll without render_info
                # But we can just increment and prompt_toolkit will clamp it
                self.history_window.vertical_scroll += 3

        @self.kb.add('end')
        def _(event):
            # Jump to bottom and re-enable auto-scroll.
            self._force_scroll_next_update = True
            self._scroll_to_bottom()
            event.app.invalidate()

        @self.kb.add('home')
        def _(event):
            # Jump to top of history.
            self.history_window.vertical_scroll = 0
            event.app.invalidate()

        @self.kb.add('pageup')
        def _(event):
            info = self.history_window.render_info
            jump = info.window_height - 1 if info else 10
            self.history_window.vertical_scroll = max(
                0, self.history_window.vertical_scroll - jump
            )
            event.app.invalidate()

        @self.kb.add('pagedown')
        def _(event):
            info = self.history_window.render_info
            jump = info.window_height - 1 if info else 10
            self.history_window.vertical_scroll += jump
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

        self.input_field = TextArea(
            height=3,
            prompt="ezclaw > ",
            multiline=False,
            history=self.prompt_history,
        )

        return Layout(
            HSplit([
                Frame(self.history_window, title="EzClaw Chat"),
                self.status_window,
                Frame(self.input_field, height=5)
            ]),
            focused_element=self.input_field
        )

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
        """Return the status bar as a list of (inline-style, text) tuples.

        We use inline styles (`bg:#xxx fg`) instead of class-based styles
        so each segment can carry its own color while still inheriting
        the bar-wide warm-dark background defined in self.style. The
        activity glyph and 🦀 mascot both color-cycle per tick — visible
        because the animation loop re-renders the status bar at 4-10fps.
        """
        BG = "bg:#1f160e "
        auth_icon = "◉" if self.agent.session_authorized else "○"
        mode_glyph = "⚡" if ENABLE_MULTI_AGENT else "●"
        model_info = self.agent.model.split(",")[0][:45] if "," in self.agent.model else self.agent.model[:45]
        msg_count = len(self.agent.messages) if hasattr(self.agent, 'messages') and self.agent.messages else 0

        if self.is_generating:
            frame_idx = int(time.time() * 4) % len(ACTIVITY_FRAMES)
            glyph = ACTIVITY_FRAMES[frame_idx]
            act_color = self._cycle_palette_color(TITLE_GRADIENT)
        else:
            glyph = "·"
            act_color = "#7a7570"

        mascot_color = self._cycle_palette_color(TITLE_GRADIENT, period_sec=1.2)
        divider = (BG + "#5a4a3a", "  ╱  ")

        segments = [
            (BG + f"bold {act_color}", f"  {glyph}  "),
            (BG + f"bold {mascot_color}", MASCOT + " "),
            (BG + "#7a7570", f"{auth_icon}  "),
            (BG + "#ffd166", f"{mode_glyph} {model_info}"),
            divider,
            (BG + "#c8c4be", f"{msg_count} msgs"),
        ]

        if self.is_generating:
            n_tools = len(self.tool_executions)
            elapsed = time.time() - self.generation_start_time if self.generation_start_time else 0
            segments.append(divider)
            segments.append((BG + "#7fd070", f"⚙ {n_tools} tool{'s' if n_tools != 1 else ''}"))
            segments.append(divider)
            segments.append((BG + "#ff8c5c", f"{elapsed:.1f}s"))

        if not self._mouse_capture:
            segments.append(divider)
            segments.append((BG + "bold #ff8c5c", "✂ COPY"))
        if self.show_architect:
            segments.append(divider)
            segments.append((BG + "bold #ff5fd7", "🧠 STRATEGY"))
        if not self.compact_tools:
            segments.append(divider)
            segments.append((BG + "bold #7fd070", "⊞ FULL TOOLS"))

        segments.append((BG + "#5a4a3a", "  │  "))
        segments.append((BG + "#a89884", "[F1] help  [^C] exit  [F2] copy  [F3] strategy  [F4] tools"))
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
        # Cap to the last 3 side_messages — without this cap, a long
        # multi-step turn stacked 20+ dim italic lines at the top of the
        # active area, duplicating what the architect chip already shows.
        for msg in self.side_messages[-3:]:
            parts.append(Text(msg, style=f"dim {DIM} italic"))
            
        current_reasoning = "".join(self.reasoning_chunks)
        if SHOW_THINKING and current_reasoning:
            stripped = current_reasoning.strip()
            char_count = len(stripped)
            word_count = len(stripped.split())
            still_thinking = self.is_generating and not self.current_response_parts
            from phrases import THINKING_BADGE, THOUGHT_BADGE
            badge = f"{THINKING_BADGE}…" if still_thinking else THOUGHT_BADGE
            # Soft purple-grey for reasoning so it visually recedes vs. the
            # main response (which renders as markdown). Italic + dim caps
            # are preserved; border picks up the role hue.
            REASON_COLOR = "#9999cc"
            body_text = Text()
            body_text.append(stripped, style=f"italic {REASON_COLOR}")
            parts.append(Panel(
                body_text,
                title=f"[bold {REASON_COLOR}]{badge}[/bold {REASON_COLOR}]  [dim {DIM}]({word_count}w · {char_count}c)[/dim {DIM}]",
                border_style=f"dim {REASON_COLOR}",
                box=ROUNDED,
                padding=(0, 1),
            ))
            
        for idx, tool in enumerate(self.tool_executions, 1):
            parts.append(self._build_tool_panel(tool, idx))
            
        if self.auth_active and self.current_auth_chunk:
             parts.append(Panel(
                 Text.assemble(
                     ("Authorization Required", f"bold {WARN}"),
                     ("\n\nTool: ", ""), (self.current_auth_chunk['name'], "bold"),
                     ("\nArgs: ", ""), (str(self.current_auth_chunk['arguments']), f"dim {DIM}"),
                     ("\n\nPress ", ""), ("[Y]", "bold"), (" to allow, ", ""),
                     ("[N]", "bold"), (" to deny, ", ""),
                     ("[A]", "bold"), (" to allow for session", "")
                 ),
                 title="[bold red]Security Check[/bold red]",
                 border_style="red",
                 box=ROUNDED,
                 padding=(1, 2)
             ))

        plan_panel = self._render_plan_panel()
        if plan_panel is not None:
            parts.append(plan_panel)

        if self.architect_intent:
            parts.append(self._render_architect_intent(self.architect_intent))

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
            status_text.append(f"{self.current_status}  [{elapsed:.1f}s]", style=f"bold {rs.color}")
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

    def _render_plan_panel(self):
        """Render the active plan as a sticky panel. Returns None when no plan."""
        if self.current_plan is None or not self.current_plan.tasks:
            return None
        from theme import TASK_STATE_STYLE, TASK_STATE_FLASH

        plan = self.current_plan
        done, total = plan.progress()
        now = time.time()

        body_lines = []
        for task in plan.tasks:
            icon, color = TASK_STATE_STYLE.get(task.status, ("•", DIM))
            flashing = now < self._task_flash_until.get(task.id, 0.0)
            line_color = TASK_STATE_FLASH.get(task.status, color) if flashing else color

            # Celebrate `done` transitions: during the 150ms flash window
            # after a task lands on `done`, swap the icon for a sparkle ✨
            # and animate its color through the gradient. The eye catches
            # it for one tick, then it settles back to the green ● dot.
            display_icon = icon
            if flashing and task.status == "done":
                display_icon = "✨"
                line_color = self._cycle_palette_color(TITLE_GRADIENT, period_sec=0.15)

            # Bold for in_progress so the eye finds it immediately.
            weight = "bold " if task.status == "in_progress" else ""
            line_text = Text()
            line_text.append(f"  {display_icon} ", style=f"{weight}{line_color}")
            line_text.append(f"{task.id}. ", style=f"dim {DIM}")
            line_text.append(task.description, style=f"{weight}{line_color}")
            body_lines.append(line_text)

        # Title color shifts through the sunset palette every ~1.5s so the
        # plan panel reads as actively alive. Border breathes between dim
        # and full saturation at ~0.6Hz — slow enough to feel meditative.
        title_color = self._cycle_palette_color(TITLE_GRADIENT, period_sec=1.5)
        border_breath = (time.time() * 1.2) % 2.0
        border_prefix = "" if border_breath < 1.0 else "dim "
        title = f"Plan: {plan.title}  ·  {done}/{total}"
        return Panel(
            Group(*body_lines),
            title=f"[bold {title_color}]{title}[/bold {title_color}]",
            border_style=f"{border_prefix}{title_color}",
            box=ROUNDED,
            padding=(0, 1),
        )

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

    def _get_welcome_panel(self):
        body = Text()

        # Mascot prefix + animated gradient title.
        # The gradient offset shifts ~2.5Hz so colors flow across the letters.
        # A SHIMMER position (one letter at a time, sweeping left-to-right
        # at ~2Hz) gets rendered bright white on top of the gradient — a
        # clearly-visible "light moving across the title" effect.
        body.append(f"{MASCOT} ", style=f"bold {self._cycle_palette_color(TITLE_GRADIENT)}")
        offset = self._gradient_offset()
        title = "EzClaw"
        # Shimmer cell: sweeps 0..len(title)-1 then pauses for a beat
        # before restarting. The pause makes the next sweep feel intentional
        # rather than a continuous strobe.
        shimmer_period = len(title) + 2
        shimmer_pos = int(time.time() * 2) % shimmer_period
        for i, ch in enumerate(title):
            base_color = TITLE_GRADIENT[(i + offset) % len(TITLE_GRADIENT)]
            if i == shimmer_pos:
                body.append(ch, style=f"bold reverse {base_color}")
            else:
                body.append(ch, style=f"bold {base_color}")
        body.append(" ", "")
        body.append("v2.2 (Full TUI)\n", style=f"dim {DIM}")
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
        body.append("/help [F1]  /settings  /queue  /skills  /memory  /diagnose  /clear  /thinking  /notify",
                    style=f"dim {DIM}")
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
        one-line collapsed/running tool entries. The kind icon pulses
        subtly between its base color and a brightened sunset shade per
        tick — gives running tool panels a heartbeat."""
        # Pulse only the icon at ~0.6Hz (so the eye catches it without
        # being distracting), keep the name + signature solid.
        pulse_t = (time.time() * 1.2) % 2.0
        icon_color = (
            self._cycle_palette_color(TITLE_GRADIENT, period_sec=1.5)
            if pulse_t < 0.4 else tool_kind.color
        )
        line = Text()
        line.append(f"{tool_kind.icon} ", style=f"bold {icon_color}")
        if index is not None:
            line.append(f"[{index}] ", style=f"dim {DIM}")
        line.append(tool_name, style=f"bold {tool_kind.color}")
        if args:
            sig = self._format_args_inline(args)
            line.append("(", style=f"dim {DIM}")
            line.append(sig, style=f"dim {DIM}")
            line.append(")", style=f"dim {DIM}")
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
            elapsed_str = f" ({elapsed:.1f}s)" if elapsed > 1 else ""

            # Embedded interactive shell — render the streaming subprocess
            # output as a live tool panel inside the chat. The panel
            # expands as bytes arrive; the user types into the chat box
            # and their keystrokes are forwarded to the subprocess.
            live_buffer = tool.get("_live_buffer")
            if tool.get("interactive") and live_buffer is not None:
                head = self._one_line_tool_head(tool_kind, tool_name, args, index)
                head.append(f"   ⏳ live{elapsed_str}", style=f"bold {WARN}")
                head.append("   (type to send input to subprocess)", style=f"dim {DIM} italic")

                # Decode the streaming bytes and strip ANSI; keep the last
                # ~40 lines so the panel doesn't grow indefinitely on a
                # very chatty subprocess.
                try:
                    full = b"".join(live_buffer).decode("utf-8", errors="ignore")
                except Exception:
                    full = ""
                full = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", full)
                lines = [l for l in full.splitlines() if l.strip("\r")][-40:]
                body = Text("\n".join(lines), style="")
                return Panel(
                    Group(head, body),
                    border_style=f"dim {WARN}",
                    box=ROUNDED,
                )

            line = self._one_line_tool_head(tool_kind, tool_name, args, index)
            line.append(f"   ⏳ {verb}…{elapsed_str}", style=f"dim {DIM} italic")
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
                line.append("   ↳ ", style=f"dim {DIM}")
                line.append(summary, style=f"dim {DIM} italic")
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

        # An embedded interactive shell session is active — every keystroke
        # the user types goes to the subprocess via the session's input
        # queue (with a trailing newline so subprocesses like `sudo` see
        # a complete line). The agent loop stays paused for the duration.
        if self._interactive_session is not None:
            try:
                self._interactive_session["input_queue"].put((text + "\n").encode())
            except Exception:
                pass
            self.history_ansi.append(render_to_ansi(
                Text(f"→ {text}", style=f"italic dim {DIM}")
            ))
            self._update_ui()
            return

        if text.startswith("/"):
            self._handle_command(text)
            return

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
        elif cmd == "/settings":
            self._show_settings()
        elif cmd == "/queue":
            self._show_scheduled_queue()
        elif cmd == "/skills":
            self._show_skills()
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
        elif cmd.startswith("/expand") or cmd.startswith("/collapse"):
            self._toggle_tool_expansion(cmd)
        elif cmd.startswith("/copy"):
            self._copy_to_clipboard(cmd)
        else:
            self.history_ansi.append(render_to_ansi(Text(f"Unknown command: {cmd}", style=ERR)))

        self._update_ui()

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
                ("Mouse copy mode",    "on" if not self._mouse_capture else "off", "F2"),
            ]),
            ("Behavior", [
                ("Session auth",       "always allow" if self.agent.session_authorized else "ask per tool", "/authorize"),
                ("User name",          self.user_name, "EZCLAW_USER env"),
            ]),
            ("Agents & models", [
                ("Mode",               "multi-agent" if ENABLE_MULTI_AGENT else "single-agent", "ENABLE_MULTI_AGENT env"),
                ("Primary model",      self.agent.model, "OLLAMA_MODEL env"),
                ("History",            f"{history_size} messages", "/clear to reset"),
            ]),
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
            ("Navigation", [
                ("[PgUp] / [PgDn]",    "scroll one screen"),
                ("[Home] / [End]",     "jump to top / bottom"),
                ("[F1]",               "open this help"),
                ("[Arrows] / mouse",   "scroll"),
            ]),
            ("Display toggles", [
                ("[F2]",               "copy mode (terminal-native selection)"),
                ("[F3]",               "expanded strategy / reflection panel"),
                ("[F4]",               "compact / full tool panel layout"),
                ("/thinking [on|off]", "show / hide reasoning panel"),
                ("/notify  [on|off]",  "desktop notification on/off"),
            ]),
            ("Inspection", [
                ("/settings",          "show all settings + how to change each"),
                ("/queue",             "list active scheduled tasks"),
                ("/skills",            "list learned skills"),
                ("/memory [query]",    "show stored memories (with optional search)"),
                ("/diagnose",          "GPU / Ollama / system probe"),
            ]),
            ("Session", [
                ("/clear",             "wipe current session history"),
                ("/authorize",         "toggle session-wide tool authorization"),
                ("/expand [N|all]",    "expand a tool panel (defaults to last)"),
                ("/collapse [N|all]",  "re-collapse a tool panel"),
                ("/copy [last|all|N]", "copy assistant text (OSC52 clipboard)"),
                ("exit / quit",        "leave EzClaw"),
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
                    for task in new_plan.tasks:
                        prev = self._last_task_states.get(task.id)
                        if prev is not None and prev != task.status:
                            self._task_flash_until[task.id] = now + 0.15
                        self._last_task_states[task.id] = task.status
                    self.current_plan = new_plan
                elif chunk["type"] == "intent":
                    self.architect_intent = chunk
                    new_role = chunk.get("agent")
                    if new_role:
                        if self._last_chip_role and new_role != self._last_chip_role:
                            self._chip_flash_until = time.time() + 0.15
                        self._last_chip_role = new_role
                        self.current_role = new_role
                elif chunk["type"] == "reasoning":
                    self.reasoning_chunks.append(chunk["content"])
                elif chunk["type"] == "content":
                    self.current_response_parts.append(chunk["content"])
                elif chunk["type"] == "status":
                    self.current_status = chunk["content"].strip()
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
                elif chunk["type"] == "tool_start":
                    is_int = chunk.get("interactive", False)
                    self.tool_executions.append({"name": chunk["name"], "args": chunk["arguments"], "result": None, "interactive": is_int, "start_time": time.time(), "expanded": False})
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
        final_renderable = self._get_current_renderable_ansi()
        self.history_ansi.append(final_renderable)
        self.current_response_parts = []
        self.reasoning_chunks = []
        self.tool_executions = []
        self.side_messages = []
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
        self.app.run()

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
                sched.mark_status(task.id, "Done")
            except Exception:
                sched.mark_status(task.id, "Failed")

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    ui = ChatUI()
    ui.run()
