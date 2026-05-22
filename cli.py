import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.box import ROUNDED, HEAVY
from rich.syntax import Syntax
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.layout import FormattedTextControl, Window, HSplit
from prompt_toolkit.application import get_app
from agent import ChatAgent
from multi_agent import MultiAgentSystem
from dotenv import load_dotenv

load_dotenv()

# ── Theme ──────────────────────────────────────────────────────
PRIMARY = "magenta"
SECONDARY = "cyan"
ACCENT = "green"
WARN = "yellow"
ERR = "red"
DIM = "grey50"

class Theme:
    primary = PRIMARY
    secondary = SECONDARY
    accent = ACCENT
    warn = WARN
    err = ERR
    dim = DIM
    box = ROUNDED

console = Console()

# Global state
SHOW_THINKING = os.getenv("SHOW_THINKING", "true").lower() == "true"
ENABLE_MULTI_AGENT = os.getenv("ENABLE_MULTI_AGENT", "false").lower() == "true"

def heartbeat_monitor():
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
                                    console.print(f"\n[bold {WARN}]🔔 SCHEDULED TASK DUE:[/bold {WARN}] {parts[2]}")
                                    line = line.replace("| Pending |", "| Notified |")
                                    updated = True
                            except: pass
                    new_lines.append(line)
                if updated:
                    with open("heartbeat.md", "w") as f:
                        f.writelines(new_lines)
        except Exception: pass
        time.sleep(30)

def run_diagnostics():
    console.print(f"[bold {SECONDARY}]Running System Diagnostics...[/bold {SECONDARY}]")
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader", shell=True, text=True)
        console.print(Panel(gpu_info.strip(), title=f"[bold {ACCENT}]NVIDIA GPU Status[/bold {ACCENT}]", border_style=ACCENT))
    except:
        console.print(f"[{WARN}]NVIDIA GPU (nvidia-smi) not found or failed.[/{WARN}]")
    try:
        ollama_ps = subprocess.check_output("ollama ps", shell=True, text=True)
        console.print(Panel(ollama_ps.strip() or "No models currently loaded in memory.", title=f"[bold {SECONDARY}]Ollama Active Models[/bold {SECONDARY}]", border_style=SECONDARY))
    except:
        console.print(f"[{WARN}]Ollama CLI (ollama ps) not found or failed.[/{WARN}]")

def truncate_text(text: str, max_lines: int = 15) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n\n[bold {WARN}]... (Output truncated, {len(lines) - max_lines} lines more) ...[/bold {WARN}]"
    return text

def build_status_bar(agent):
    auth_icon = "🔓" if agent.session_authorized else "🔒"
    mode = "Multi" if ENABLE_MULTI_AGENT else "Single"
    model_info = agent.model.split(",")[0][:50] if "," in agent.model else agent.model[:50]
    msg_count = len(agent.messages) if hasattr(agent, 'messages') and agent.messages else 0
    return [
        ("class:bar.text", f"  {auth_icon} "),
        ("class:bar.auth", "AUTH " if agent.session_authorized else "LOCKED "),
        ("class:bar.sep", "│"),
        ("class:bar.text", f" {mode} "),
        ("class:bar.sep", "│"),
        ("class:bar.text", f" {model_info} "),
        ("class:bar.sep", "│"),
        ("class:bar.text", f" {msg_count} msgs "),
    ]

def main():
    global SHOW_THINKING
    monitor_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    monitor_thread.start()

    pt_style = Style.from_dict({
        'prompt': f'bold {PRIMARY}',
        'bar.text': f'bold {SECONDARY}',
        'bar.auth': f'bold {ACCENT}',
        'bar.sep': f'dim',
    })

    session = PromptSession(
        history=FileHistory(os.path.expanduser("~/.ezclaw_history")),
        style=pt_style,
        bottom_toolbar=lambda: build_status_bar(agent),
    )
    agent = MultiAgentSystem() if ENABLE_MULTI_AGENT else ChatAgent()
    tool_executions = []

    welcome = Panel(
        Text.assemble(
            (f"EzClaw ", f"bold {PRIMARY}"), ("v2.2", f"bold {WARN}"),
            ("\n\nModel: ", "bold"), (f"{agent.model}", f"{SECONDARY}"),
            ("\nMode: ", "bold"), (f"{'Multi-Agent' if ENABLE_MULTI_AGENT else 'Single-Agent'}", f"{SECONDARY}"),
            ("\nWorkspace: ", "bold"), ("./workspace", f"{SECONDARY}"),
            ("\n\nCommands: ", "bold"),
            ("/diagnose, /clear, /thinking, /settings, /authorize", f"{DIM}"),
        ),
        box=ROUNDED, padding=(1, 2), border_style=PRIMARY,
        title="[bold]EzClaw[/bold]",
    )
    console.print(welcome)

    while True:
        try:
            user_input = session.prompt("ezclaw > ")
            if not user_input.strip(): continue
            if user_input.lower() in ["exit", "quit"]: break

            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                if cmd.startswith("/thinking"):
                    SHOW_THINKING = "on" in cmd or ("off" not in cmd and not SHOW_THINKING)
                    console.print(f"[{DIM}]Thinking visualization: {'ON' if SHOW_THINKING else 'OFF'}[/{DIM}]")
                elif cmd == "/authorize":
                    agent.session_authorized = not agent.session_authorized
                    status = "ENABLED (Always Allow)" if agent.session_authorized else "DISABLED (Ask per tool)"
                    console.print(f"[bold {ACCENT}]Session authorization: {status}[/bold {ACCENT}]")
                elif cmd == "/settings":
                    model_info = agent.model
                    auth = "Always Allow" if agent.session_authorized else "Ask per tool"
                    history = len(agent.messages) if hasattr(agent, 'messages') and agent.messages else 0
                    info = (
                        f"**Model:** `{model_info}`\n"
                        f"**Thinking:** `{'Enabled' if SHOW_THINKING else 'Disabled'}`\n"
                        f"**Session Auth:** `{auth}`\n"
                        f"**History Size:** `{history}` messages"
                    )
                    console.print(Panel(Markdown(info), title="[bold]System Settings[/bold]", border_style=SECONDARY))
                elif cmd == "/diagnose":
                    run_diagnostics()
                elif cmd == "/clear":
                    agent.clear_session_history()
                    console.print(f"[{ACCENT}]Session cleared.[/{ACCENT}]")
                elif cmd == "/last":
                    if tool_executions and tool_executions[-1].get("result"):
                        with console.pager(styles=True):
                            console.print(Panel(str(tool_executions[-1]["result"]), title=f"Full Output: {tool_executions[-1]['name']}", border_style=ACCENT))
                    else:
                        console.print(f"[{WARN}]No recent tool output to display.[/{WARN}]")
                continue

            current_reasoning = ""
            current_content = ""
            tool_executions.clear()
            side_messages = []

            def get_renderable():
                parts = []
                for msg in side_messages:
                    parts.append(Text(msg, style=f"dim italic"))
                if SHOW_THINKING and current_reasoning:
                    parts.append(Panel(
                        Text(current_reasoning, style=f"italic {DIM}"),
                        title=f"[bold {SECONDARY}]Thinking Process[/bold {SECONDARY}]",
                        border_style=SECONDARY, box=ROUNDED,
                    ))
                for tool in tool_executions:
                    tool_name = tool["name"]
                    args = tool.get("args", {})
                    result = tool.get("result")
                    header = Text.assemble(
                        ("⚒  Tool: ", f"bold {WARN}"),
                        (tool_name, f"bold {SECONDARY}"),
                    )
                    tool_parts = []
                    if args:
                        arg_str = "\n".join([f"[bold]{k}:[/bold] {v}" for k, v in args.items()])
                        tool_parts.append(Panel(truncate_text(arg_str, max_lines=5), title="Arguments", border_style=f"dim"))
                    if result:
                        renderable_result = str(result)
                        if tool_name == "write_file" and "Diff:" in renderable_result:
                            parts_of_result = renderable_result.split("Diff:\n", 1)
                            if len(parts_of_result) > 1:
                                tool_parts.append(Text(parts_of_result[0]))
                                tool_parts.append(Syntax(truncate_text(parts_of_result[1], 20), "diff", theme="monokai", background_color="default"))
                            else:
                                tool_parts.append(Panel(truncate_text(renderable_result), title="Output", border_style=ACCENT))
                        elif tool_name == "read_file":
                            tool_parts.append(Syntax(truncate_text(renderable_result, 25), "python", theme="monokai", background_color="default"))
                        else:
                            tool_parts.append(Panel(truncate_text(renderable_result), title="Output", border_style=ACCENT))
                    else:
                        tool_parts.append(Text("Executing...", style=f"blink {WARN}"))
                    parts.append(Panel(Group(*tool_parts), title=header, border_style=WARN, box=ROUNDED))
                if current_content:
                    parts.append(Markdown(current_content))
                if not parts:
                    return Spinner("dots", text=f"[dim]Connecting to Ollama...[/dim]")
                return Group(*parts)

            with Live(get_renderable(), refresh_per_second=10, console=console) as live:
                gen = agent.chat_stream(user_input)
                try:
                    chunk = next(gen)
                    while True:
                        if chunk["type"] == "reasoning":
                            current_reasoning += chunk["content"]
                            if SHOW_THINKING:
                                live.update(get_renderable())
                        elif chunk["type"] == "content":
                            current_content += chunk["content"]
                            live.update(get_renderable())
                        elif chunk["type"] == "auth_required":
                            live.stop()
                            console.print(Panel(
                                Text.assemble(
                                    ("⚒  Authorization Required: ", f"bold {WARN}"),
                                    (chunk['name'], f"bold {SECONDARY}"),
                                    ("\nArguments: ", "bold"),
                                    (str(chunk['arguments']), f"{DIM}"),
                                ),
                                border_style=ERR, title="[bold]Security Check[/bold]",
                            ))
                            choice = ""
                            while choice not in ["y", "n", "a"]:
                                choice = console.input(f"[bold {ERR}]Authorize? (y/n/a): [/bold {ERR}]").lower().strip()
                            if choice == "y":
                                chunk = gen.send("allow")
                            elif choice == "n":
                                chunk = gen.send("deny")
                            elif choice == "a":
                                chunk = gen.send("allow_session")
                            live.start()
                            continue
                        elif chunk["type"] == "memory_stored":
                            side_messages.append(f"💭 Memory stored: {chunk['fact']}")
                            live.update(get_renderable())
                        elif chunk["type"] == "context_augmented":
                            for m in chunk["memories"]:
                                side_messages.append(f"🧠 Context augmented: {m}")
                            live.update(get_renderable())
                        elif chunk["type"] == "tool_start":
                            is_int = chunk.get("interactive", False)
                            tool_executions.append({"name": chunk["name"], "args": chunk["arguments"], "result": None, "interactive": is_int})
                            if is_int:
                                live.stop()
                                console.print(Panel(
                                    f"[bold {WARN}]Entering interactive mode for: {chunk['name']}[/bold {WARN}]",
                                    border_style=WARN,
                                ))
                            else:
                                live.update(get_renderable())
                        elif chunk["type"] == "tool_end":
                            for tool in reversed(tool_executions):
                                if tool["name"] == chunk["name"] and tool["result"] is None:
                                    tool["result"] = chunk["result"]
                                    if tool.get("interactive"):
                                        console.print(Panel(
                                            f"[bold {ACCENT}]Interactive mode completed: {chunk['name']}[/bold {ACCENT}]",
                                            border_style=ACCENT,
                                        ))
                                        live.start()
                                    break
                            live.update(get_renderable())
                        chunk = next(gen)
                except StopIteration:
                    pass
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            console.print(f"[bold {ERR}]Error:[/bold {ERR}] {str(e)}")

if __name__ == "__main__":
    main()
