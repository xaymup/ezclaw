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
from rich.box import ROUNDED
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from agent import ChatAgent
from dotenv import load_dotenv

load_dotenv()

console = Console()

# Global state
SHOW_THINKING = os.getenv("SHOW_THINKING", "true").lower() == "true"

def heartbeat_monitor():
    """Background thread for scheduled tasks."""
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
                                    console.print(f"\n[bold yellow]🔔 SCHEDULED TASK DUE:[/bold yellow] {parts[2]}")
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
    """Check GPU and Ollama status."""
    console.print("\n[bold cyan]Running System Diagnostics...[/bold cyan]")
    
    # Check NVIDIA GPU
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader", shell=True, text=True)
        console.print(Panel(gpu_info.strip(), title="NVIDIA GPU Status", border_style="green"))
    except:
        console.print("[yellow]NVIDIA GPU (nvidia-smi) not found or failed.[/yellow]")

    # Check Ollama Models
    try:
        ollama_ps = subprocess.check_output("ollama ps", shell=True, text=True)
        console.print(Panel(ollama_ps.strip() or "No models currently loaded in memory.", title="Ollama Active Models", border_style="blue"))
    except:
        console.print("[yellow]Ollama CLI (ollama ps) not found or failed.[/yellow]")

def main():
    global SHOW_THINKING
    monitor_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    monitor_thread.start()

    style = Style.from_dict({'prompt': 'bold magenta'})
    session = PromptSession(history=FileHistory(os.path.expanduser("~/.ezclaw_history")), style=style)
    agent = ChatAgent()
    
    console.print(Panel(
        Text.assemble(
            ("EzClaw ", "bold magenta"), ("v2.2", "bold yellow"),
            ("\nWorkspace: ", "bold"), ("./workspace", "cyan"),
            ("\nCommands: ", "bold"), ("/diagnose, /clear, /thinking, /settings", "dim")
        ),
        box=ROUNDED, padding=(1, 2), border_style="magenta"
    ))

    while True:
        try:
            user_input = session.prompt("ezclaw > ")
            if not user_input.strip(): continue
            if user_input.lower() in ["exit", "quit"]: break

            # Internal Commands
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                if cmd.startswith("/thinking"):
                    SHOW_THINKING = "on" in cmd or ("off" not in cmd and not SHOW_THINKING)
                    console.print(f"[dim]Thinking visualization: {'ON' if SHOW_THINKING else 'OFF'}[/dim]")
                elif cmd == "/settings":
                    console.print(Panel(Markdown(f"**Model:** `{agent.model}`\n**Thinking:** `{'Enabled' if SHOW_THINKING else 'Disabled'}`\n**History Size:** {len(agent.messages)} messages"), title="System Settings", border_style="cyan"))
                elif cmd == "/diagnose":
                    run_diagnostics()
                elif cmd == "/clear":
                    agent.clear_session_history()
                    console.print("[green]Current session history cleared (locally).[/green]")
                continue

            current_reasoning = ""
            current_content = ""

            def get_renderable():
                parts = []
                if SHOW_THINKING and current_reasoning:
                    parts.append(Panel(Text(current_reasoning, style="italic grey50"), title="[bold cyan]Thinking Process[/bold cyan]", border_style="cyan", box=ROUNDED))
                if current_content:
                    parts.append(Markdown(current_content))
                if not parts:
                    return Spinner("dots", text="[dim]Connecting to Ollama...[/dim]")
                return Group(*parts)

            # Streaming UI - Lower refresh rate to save CPU
            with Live(get_renderable(), refresh_per_second=10, console=console, transient=False) as live:
                for chunk in agent.chat_stream(user_input):
                    if chunk["type"] == "reasoning":
                        current_reasoning += chunk["content"]
                        if SHOW_THINKING:
                            live.update(get_renderable())
                    
                    elif chunk["type"] == "content":
                        current_content += chunk["content"]
                        live.update(get_renderable())
                    
                    elif chunk["type"] == "tool_start":
                        # For tools, we might want to show them temporarily or keep them.
                        # Let's keep them in the flow for now by adding to a status list or just updating.
                        live.update(Group(
                            get_renderable(),
                            Panel(Text.assemble(("⚒  Executing: ", "bold yellow"), (chunk['name'], "bold cyan")), border_style="yellow", box=ROUNDED)
                        ))
                    
                    elif chunk["type"] == "tool_end":
                        live.update(get_renderable())

            console.print()

        except KeyboardInterrupt: continue
        except EOFError: break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
