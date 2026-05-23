#!/usr/bin/env python3
import os
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

from prompt_toolkit.eventloop import call_soon_threadsafe
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED
from rich.syntax import Syntax
from rich.spinner import Spinner

from prompt_toolkit.application import Application, run_in_terminal
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
from theme import THEME, TITLE_GRADIENT, ACTIVITY_FRAMES

PRIMARY = THEME.palette.primary
SECONDARY = THEME.palette.secondary
ACCENT = THEME.palette.accent
WARN = THEME.palette.warn
ERR = THEME.palette.err
DIM = THEME.palette.dim

class Theme:
    primary = PRIMARY
    secondary = SECONDARY
    accent = ACCENT
    warn = WARN
    err = ERR
    dim = DIM
    box = ROUNDED

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

def _make_custom_spinner(frames, color):
    """Build a Spinner with a custom frame list and 10fps cadence."""
    sp = Spinner(name="dots", text="", style=f"bold {color}")
    sp.frames = list(frames)
    sp.interval = 0.1
    return sp
from rich.box import ROUNDED

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
        self.style = Style.from_dict({
            'status': f'reverse #ffffff bg:#333333',
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

        # Architect strategy panels: hidden by default — they dominated the
        # screen with mostly-redundant information. The compact chip below
        # still shows the routing decision; F3 toggles the full panels back.
        self.show_architect = False

    def _wrap_tools(self):
        from tools import registry
        original_run_shell = registry.tools.get('run_shell')
        if original_run_shell:
            def wrapped_run_shell(command: str, interactive: bool = False) -> str:
                if interactive:
                    res_queue = queue.Queue()
                    def _trigger():
                        async def _run_async():
                            def _run():
                                from rich.console import Console
                                from rich.panel import Panel
                                from rich.text import Text
                                r_console = Console()
                                
                                r_console.print("\n")
                                r_console.print(Panel(
                                    Text.assemble(
                                        ("EZCLAW SUSPENDED\n\n", "bold yellow"),
                                        ("Command: ", ""), (command, "bold cyan"),
                                        ("\n\nInstructions: ", "bold"), 
                                        ("Provide any required input (password, confirmation, etc.).\n", ""),
                                        ("Wait for the command to finish completely.\n", ""),
                                        ("If the command hangs after finishing, press [Enter].", "italic dim")
                                    ),
                                    title="[bold yellow]Interactive Shell Session[/bold yellow]",
                                    border_style="yellow",
                                    expand=False
                                ))
                                
                                result = original_run_shell(command, interactive=True)
                                
                                r_console.print(Panel(
                                    "Interactive session finished. Resuming EzClaw TUI...",
                                    title="[bold green]Success[/bold green]",
                                    border_style="green",
                                    expand=False
                                ))
                                r_console.print("\n")
                                return result
                            
                            # run_in_terminal suspends the TUI and gives control to the terminal
                            result = await run_in_terminal(_run, render_cli_done=False)
                            res_queue.put(result)
                        
                        # Create background task on the main event loop
                        self.app.create_background_task(_run_async())
                    
                    loop = getattr(self.app, 'loop', None)
                    call_soon_threadsafe(_trigger, loop=loop)
                    
                    # Block the agent thread until the user is done with the terminal
                    return res_queue.get()
                return original_run_shell(command, interactive=False)
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

            self._history_read_only -= 1  # 1 -> 0, now writable
            try:
                self.history_buffer.text = full_ansi
            finally:
                self._history_read_only += 1  # back to read-only

            self._scroll_to_bottom()
            if self.app.is_running:
                self.app.invalidate()
    def _get_status_text(self):
        auth_icon = "◉" if self.agent.session_authorized else "○"
        mode = "⚡" if ENABLE_MULTI_AGENT else "●"
        model_info = self.agent.model.split(",")[0][:45] if "," in self.agent.model else self.agent.model[:45]
        msg_count = len(self.agent.messages) if hasattr(self.agent, 'messages') and self.agent.messages else 0

        live = ""
        if self.is_generating:
            n_tools = len(self.tool_executions)
            elapsed = time.time() - self.generation_start_time if self.generation_start_time else 0
            live = f"  ·  ⚙ {n_tools} tool{'s' if n_tools != 1 else ''}  ·  {elapsed:.1f}s"

        copy_badge = "  ·  ✂ COPY MODE" if not self._mouse_capture else ""
        arch_badge = "  ·  🧠 STRATEGY" if self.show_architect else ""
        return f"  {auth_icon}  {mode} {model_info}  ·  {msg_count} msgs{live}{copy_badge}{arch_badge}  |  [Ctrl+C] Exit  [F2] Copy  [F3] Strategy  [Arrows] Scroll"

    def _spinner_for(self, role):
        """Return a Spinner instance using the role's frame list and color."""
        rs = THEME.role(role or "")
        return _make_custom_spinner(rs.spinner_frames, rs.color)

    def _get_current_renderable_ansi(self):

        if not self.is_generating and not self.current_response_parts and not self.reasoning_chunks and not self.tool_executions:
            if not self.welcome_shown:
                self.welcome_shown = True
                return render_to_ansi(self._get_welcome_panel())
            return ""
            
        parts = []
        for msg in self.side_messages:
            parts.append(Text(msg, style=f"dim {DIM} italic"))
            
        current_reasoning = "".join(self.reasoning_chunks)
        if SHOW_THINKING and current_reasoning:
            stripped = current_reasoning.strip()
            char_count = len(stripped)
            word_count = len(stripped.split())
            still_thinking = self.is_generating and not self.current_response_parts
            badge = "thinking…" if still_thinking else "thought"
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

        if self.architect_intent:
            parts.append(self._render_architect_intent(self.architect_intent))

        current_content = "".join(self.current_response_parts)
        if current_content:
            parts.append(Markdown(current_content))
            
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

    def _render_architect_intent(self, intent):
        agent = intent.get("agent") or "?"
        reflection = intent.get("reflection") or {}
        goal = self._clean_field(reflection.get("goal"))
        obs = self._clean_field(reflection.get("observation"))
        ct = self._clean_field(reflection.get("critical_thinking"))
        reasoning = self._clean_field(intent.get("reasoning"))
        plan = self._clean_field(intent.get("plan"))

        role_color = THEME.role(agent).color

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

        # Expanded view: only render fields with real content; collapse
        # reasoning into plan when it's a substring/prefix to avoid the
        # double-printed prose that bloated the old layout.
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
        body.append("/help  /diagnose  /clear  /thinking  /settings  /authorize  /expand  /collapse",
                    style=f"dim {DIM}")
        return Panel(
            body,
            box=ROUNDED, padding=(1, 2), border_style=DIM,
            title=f"[bold {PRIMARY}]EzClaw[/bold {PRIMARY}]",
        )

    def _build_tool_panel(self, tool, index=None):
        tool_name = tool["name"]
        args = tool.get("args", {})
        result = tool.get("result")
        expanded = tool.get("expanded", False)

        # Truncation limits flex based on expand state. Expanded uses a generous
        # cap so we still avoid runaway 10MB dumps, but show effectively all
        # normal tool output.
        args_cap = 200 if expanded else 5
        diff_cap = 2000 if expanded else 20
        read_cap = 5000 if expanded else 25
        output_cap = 5000 if expanded else 100

        idx_label = f"[{index}] " if index is not None else ""
        state_label = "  ⇣ expanded" if expanded else ""
        header = Text.assemble(
            ("▸ ", f"bold {WARN}"),
            (idx_label, f"dim {DIM}"),
            (tool_name, "bold"),
            (state_label, f"dim {ACCENT}"),
        )
        tool_parts = []
        if args:
            arg_lines = []
            for k, v in args.items():
                arg_lines.append(Text.assemble((f"{k}: ", "bold"), (str(v), "")))
            arg_text = Text("\n").join(arg_lines)
            tool_parts.append(Panel(self._truncate_text(arg_text, max_lines=args_cap), title="args", border_style=f"dim {DIM}"))

        if result:
            renderable_result = str(result)
            if tool_name == "write_file" and "Diff:" in renderable_result:
                parts_of_result = renderable_result.split("Diff:\n", 1)
                if len(parts_of_result) > 1:
                    tool_parts.append(Text(parts_of_result[0]))
                    diff_content = self._truncate_text(parts_of_result[1], diff_cap)
                    if isinstance(diff_content, Text):
                        diff_content = diff_content.plain
                    tool_parts.append(Syntax(diff_content, "diff", theme="monokai", background_color="default"))
                else:
                    tool_parts.append(Panel(self._truncate_text(renderable_result, max_lines=output_cap), title="output", border_style=DIM))
            elif tool_name == "read_file":
                path_arg = str(args.get("path", "")) if args else ""
                lang = self._detect_lang(path_arg)
                file_content = self._truncate_text(renderable_result, read_cap)
                if isinstance(file_content, Text):
                    file_content = file_content.plain
                tool_parts.append(Syntax(file_content, lang, theme="monokai", background_color="default"))
            else:
                tool_parts.append(Panel(self._truncate_text(Text(renderable_result), max_lines=output_cap), title="output", border_style=DIM))

            # Hint only when truncation could have hidden something AND not expanded.
            if not expanded and index is not None:
                result_lines = renderable_result.count("\n") + 1
                # Heuristic: if result is multi-line and likely above caps, surface the hint.
                if result_lines > 20:
                    tool_parts.append(Text(f"  /expand {index}  to show full output", style=f"dim {DIM} italic"))
        else:
            start_time = tool.get("start_time")
            elapsed = time.time() - start_time if start_time else 0
            label = f"running... ({elapsed:.1f}s)" if elapsed > 1 else "running..."
            tool_parts.append(Text(label, style=f"dim {DIM}"))
        return Panel(Group(*tool_parts), title=header, border_style=DIM, box=ROUNDED)

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

        if text.startswith("/"):
            self._handle_command(text)
            return

        # Regular message
        self.history_ansi.append(render_to_ansi(Panel(text, title="User", border_style=PRIMARY)))
        self.is_generating = True
        self.current_response_parts = []
        self.reasoning_chunks = []
        self.tool_executions = []
        self.side_messages = []
        self.architect_intent = None
        self.current_role = None
        self.current_status = "connecting..."
        self.generation_start_time = time.time()
        self.last_chunk_time = time.time()
        
        self._update_ui()
        # Start redraw loop for animations
        self._start_animation_loop()
        threading.Thread(target=self._agent_worker, args=(text,), daemon=True).start()

    def _start_animation_loop(self):
        """Periodically triggers UI updates to animate spinners using the native event loop."""
        loop = getattr(self.app, 'loop', None)
        if not loop:
            return

        def tick():
            if self.is_generating:
                self._update_ui()
                loop.call_later(0.1, tick)
        
        loop.call_later(0.1, tick)

    def _handle_command(self, cmd):
        global SHOW_THINKING
        cmd = cmd.lower().strip()
        if cmd.startswith("/thinking"):
            SHOW_THINKING = "on" in cmd or ("off" not in cmd and not SHOW_THINKING)
            msg = f"Thinking visualization: {'ON' if SHOW_THINKING else 'OFF'}"
            self.history_ansi.append(render_to_ansi(Text(msg, style=DIM)))
        elif cmd == "/authorize":
            self.agent.session_authorized = not self.agent.session_authorized
            status = "ENABLED (Always Allow)" if self.agent.session_authorized else "DISABLED (Ask per tool)"
            self.history_ansi.append(render_to_ansi(Text(f"Session authorization: {status}", style=ACCENT)))
        elif cmd == "/settings":
            model_info = self.agent.model
            auth = "Always Allow" if self.agent.session_authorized else "Ask per tool"
            history_size = len(self.agent.messages) if hasattr(self.agent, 'messages') and self.agent.messages else 0
            info = (
                f"**Model:** `{model_info}`\n"
                f"**Thinking:** `{'Enabled' if SHOW_THINKING else 'Disabled'}`\n"
                f"**Session Auth:** `{auth}`\n"
                f"**History Size:** `{history_size}` messages"
            )
            self.history_ansi.append(render_to_ansi(Panel(Markdown(info), title="settings", border_style=DIM)))
        elif cmd == "/help":
            help_text = (
                "## Commands\n\n"
                "| Command | Description |\n"
                "|---------|-------------|\n"
                "| `/help` | Show this help message |\n"
                "| `/diagnose` | Run GPU and Ollama diagnostics |\n"
                "| `/clear` | Clear current session history |\n"
                "| `/thinking [on|off]` | Toggle thinking visualization |\n"
                "| `/settings` | Show system settings |\n"
                "| `/authorize` | Toggle session-wide tool authorization |\n"
                "| `/expand [N|last|all]` | Expand a truncated tool panel (defaults to last) |\n"
                "| `/collapse [N|all]` | Re-collapse an expanded tool panel (defaults to all) |\n"
                "| `/copy [last\\|all\\|N]` | Copy assistant text to clipboard (OSC52) |\n"
                "| `[F2]` | Toggle copy mode (terminal-native selection) |\n"
                "| `[F3]` | Toggle architect strategy/reflection panel |\n"
                "| `exit` / `quit` | Exit EzClaw |\n"
            )
            self.history_ansi.append(render_to_ansi(Panel(Markdown(help_text), title="help", border_style=DIM)))
        elif cmd == "/diagnose":
            self.history_ansi.append(render_to_ansi(run_diagnostics_raw()))
        elif cmd == "/clear":
            self.agent.clear_session_history()
            self.history_ansi = []
            self.welcome_shown = False
        elif cmd.startswith("/expand") or cmd.startswith("/collapse"):
            self._toggle_tool_expansion(cmd)
        elif cmd.startswith("/copy"):
            self._copy_to_clipboard(cmd)
        else:
            self.history_ansi.append(render_to_ansi(Text(f"Unknown command: {cmd}", style=ERR)))

        self._update_ui()

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
                if chunk["type"] == "intent":
                    self.architect_intent = chunk
                    new_role = chunk.get("agent")
                    if new_role:
                        self.current_role = new_role
                elif chunk["type"] == "reasoning":
                    self.reasoning_chunks.append(chunk["content"])
                elif chunk["type"] == "content":
                    self.current_response_parts.append(chunk["content"])
                elif chunk["type"] == "status":
                    self.current_status = chunk["content"].strip()
                    self.side_messages.append(self.current_status)
                elif chunk["type"] == "auth_required":
                    self.auth_active = True
                    self._update_ui()
                    choice = self._ask_auth(chunk)
                    chunk = gen.send(choice)
                    self.auth_active = False
                    continue
                elif chunk["type"] == "memory_stored":
                    self.side_messages.append(f"📝 {chunk['fact']}")
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

    def _ask_auth(self, chunk):
        self.current_auth_chunk = chunk
        self.auth_active = True
        self.input_field.read_only = True
        self._update_ui()
        
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
        self.app.run()

    def _heartbeat_monitor(self):
        while True:
            try:
                if os.path.exists("heartbeat.md"):
                    with open("heartbeat.md", "r") as f:
                        lines = f.readlines()
                    updated = False
                    new_lines = []
                    for line in lines:
                        if "| Pending |" in line:
                            parts = [p.strip() for p in line.split("|")]
                            if len(parts) >= 4:
                                try:
                                    task_time = datetime.strptime(parts[1], '%Y-%m-%d %H:%M')
                                    if task_time <= datetime.now():
                                        msg = f"🔔 SCHEDULED TASK DUE: {parts[2]}"
                                        self.history_ansi.append(render_to_ansi(Text(msg, style=WARN)))
                                        self._update_ui()
                                        line = line.replace("| Pending |", "| Notified |")
                                        updated = True
                                except: pass
                        new_lines.append(line)
                    if updated:
                        with open("heartbeat.md", "w") as f:
                            f.writelines(new_lines)
            except Exception: pass
            time.sleep(30)

if __name__ == "__main__":
    ui = ChatUI()
    ui.run()
