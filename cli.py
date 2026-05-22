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
PRIMARY = "#ffd700"  # gold1
SECONDARY = "#bdbdbd" # grey74
ACCENT = "#00ff00"   # green
WARN = "#ff8c00"     # dark_orange
ERR = "#ff0000"      # red
DIM = "#808080"      # grey50

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

        self.history_file = os.path.expanduser("~/.ezclaw_history")
        self.prompt_history = FileHistory(self.history_file)

        # Native buffer for history
        self._history_read_only = True
        self.history_buffer = Buffer(read_only=Condition(lambda: self._history_read_only))

        self.kb = KeyBindings()

        self._setup_keybindings()

        self.layout = self._create_layout()
        self.style = Style.from_dict({
            'status': f'reverse #ffffff bg:#333333',
            'prompt': f'bold {PRIMARY}',
            'frame.border': f'{DIM}',
        })

        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=True,
            mouse_support=True,
        )

        self.welcome_shown = False

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
                                return original_run_shell(command, interactive=True)
                            result = await run_in_terminal(_run, render_cli_done=True)
                            res_queue.put(result)
                        self.app.create_background_task(_run_async())
                    
                    loop = getattr(self.app, 'loop', None)
                    call_soon_threadsafe(_trigger, loop=loop)
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
        # Update the buffer text
        full_ansi = "\n".join(self.history_ansi)
        full_ansi += "\n" + self._get_current_renderable_ansi()

        self._history_read_only = False
        self.history_buffer.text = full_ansi
        self._history_read_only = True

        self._scroll_to_bottom()
        if self.app.is_running:
            self.app.invalidate()
    def _get_status_text(self):
        auth_icon = "◉" if self.agent.session_authorized else "○"
        mode = "⚡" if ENABLE_MULTI_AGENT else "●"
        model_info = self.agent.model.split(",")[0][:45] if "," in self.agent.model else self.agent.model[:45]
        msg_count = len(self.agent.messages) if hasattr(self.agent, 'messages') and self.agent.messages else 0
        return f"  {auth_icon}  {mode} {model_info}  ·  {msg_count} msgs  |  [Ctrl+C] Exit  [Arrows/Wheel] Scroll"

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
            parts.append(Panel(
                Text(current_reasoning, style=f"italic {DIM}"),
                title=f"[bold {PRIMARY}]thinking[/bold {PRIMARY}]",
                border_style=DIM, box=ROUNDED,
            ))
            
        for tool in self.tool_executions:
            parts.append(self._build_tool_panel(tool))
            
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

        current_content = "".join(self.current_response_parts)
        if current_content:
            parts.append(Markdown(current_content))
            
        if not parts and self.is_generating:
            parts.append(Spinner("dots", text=f"[dim {DIM}]connecting...[/dim {DIM}]"))
            
        return render_to_ansi(Group(*parts))

    def _get_welcome_panel(self):
        return Panel(
            Text.assemble(
                ("EzClaw ", f"bold {PRIMARY}"),
                ("v2.2 (Full TUI)", f"dim {DIM}"),
                ("\n\n", ""),
                (f"{'⚡' if ENABLE_MULTI_AGENT else '●'} ", ""),
                (f"{self.agent.model}", f"{SECONDARY}"),
                ("\n", ""),
                (f"./workspace", f"dim {DIM}"),
                ("\n\n", ""),
                ("Commands: ", "bold"),
                ("/help · /diagnose · /clear · /thinking · /settings · /authorize", f"dim {DIM}"),
            ),
            box=ROUNDED, padding=(1, 2), border_style=DIM,
            title=f"[bold {PRIMARY}]EzClaw[/bold {PRIMARY}]",
        )

    def _build_tool_panel(self, tool):
        tool_name = tool["name"]
        args = tool.get("args", {})
        result = tool.get("result")
        header = Text.assemble(
            ("▸ ", f"bold {WARN}"),
            (tool_name, f"bold"),
        )
        tool_parts = []
        if args:
            arg_str = "\n".join([f"[bold]{k}:[/bold] {v}" for k, v in args.items()])
            tool_parts.append(Panel(self._truncate_text(arg_str, max_lines=5), title="args", border_style=f"dim {DIM}"))
        if result:
            renderable_result = str(result)
            if tool_name == "write_file" and "Diff:" in renderable_result:
                parts_of_result = renderable_result.split("Diff:\n", 1)
                if len(parts_of_result) > 1:
                    tool_parts.append(Text(parts_of_result[0]))
                    tool_parts.append(Syntax(self._truncate_text(parts_of_result[1], 20), "diff", theme="monokai", background_color="default"))
                else:
                    tool_parts.append(Panel(self._truncate_text(renderable_result), title="output", border_style=DIM))
            elif tool_name == "read_file":
                tool_parts.append(Syntax(self._truncate_text(renderable_result, 25), "python", theme="monokai", background_color="default"))
            else:
                tool_parts.append(Panel(self._truncate_text(renderable_result), title="output", border_style=DIM))
        else:
            tool_parts.append(Text("running...", style=f"dim {DIM}"))
        return Panel(Group(*tool_parts), title=header, border_style=DIM, box=ROUNDED)

    def _truncate_text(self, text: str, max_lines: int = 1000) -> str:
        lines = text.splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n\n[bold {WARN}]... (Output truncated at {max_lines} lines) ...[/bold {WARN}]"
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
        
        self._update_ui()
        threading.Thread(target=self._agent_worker, args=(text,), daemon=True).start()

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
                "| `exit` / `quit` | Exit EzClaw |\n"
            )
            self.history_ansi.append(render_to_ansi(Panel(Markdown(help_text), title="help", border_style=DIM)))
        elif cmd == "/diagnose":
            self.history_ansi.append(render_to_ansi(run_diagnostics_raw()))
        elif cmd == "/clear":
            self.agent.clear_session_history()
            self.history_ansi = []
            self.welcome_shown = False
        else:
            self.history_ansi.append(render_to_ansi(Text(f"Unknown command: {cmd}", style=ERR)))
        
        self._update_ui()

    def _agent_worker(self, user_input):
        gen = self.agent.chat_stream(user_input)
        try:
            chunk = next(gen)
            while True:
                if chunk["type"] == "reasoning":
                    self.reasoning_chunks.append(chunk["content"])
                elif chunk["type"] == "content":
                    self.current_response_parts.append(chunk["content"])
                elif chunk["type"] == "status":
                    self.side_messages.append(chunk["content"].strip())
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
                    self.tool_executions.append({"name": chunk["name"], "args": chunk["arguments"], "result": None, "interactive": is_int})
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
