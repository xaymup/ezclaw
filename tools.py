import os
import subprocess
import httpx
import re
import difflib
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional
import inspect

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, func: Callable = None, *, auth_required: bool = False):
        if func is None:
            def decorator(f):
                f.auth_required = auth_required
                self.tools[f.__name__] = f
                return f
            return decorator
        func.auth_required = auth_required
        self.tools[func.__name__] = func
        return func

    def get_tool_functions(self) -> List[Callable]:
        return list(self.tools.values())

registry = ToolRegistry()

WORKSPACE_DIR = "workspace"
SKILLS_DIR = "skills"

def clean_html(html: str) -> str:
    """Basic HTML cleaning to remove tags and extra whitespace."""
    # Remove scripts and styles
    html = re.sub(r'<(script|style)\b[^>]*>.*?</\1>', '', html, flags=re.DOTALL)
    # Remove all other tags
    text = re.sub(r'<[^>]+>', '', html)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_workspace_path(path: str) -> str:
    """Ensure path is within the workspace directory."""
    # Prevent directory traversal
    safe_path = path.lstrip("./").lstrip("/")
    return os.path.join(WORKSPACE_DIR, safe_path)

@registry.register(auth_required=True)
def run_shell(command: str, interactive: bool = False) -> str:
    """
    Execute a shell command. 
    Set interactive=True for commands that need user input (e.g. ssh, apt, manual confirmation).
    """
    try:
        if interactive:
            # Run in foreground, using same stdin/out
            result = subprocess.run(command, shell=True, capture_output=False, text=True)
            return f"Interactive command '{command}' completed."
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return output or "Command executed successfully with no output."
    except Exception as e:
        return f"Error executing command: {str(e)}"

@registry.register(auth_required=True)
def read_file(path: str) -> str:
    """
    Read the content of a file within the workspace.
    """
    try:
        full_path = get_workspace_path(path)
        if not os.path.exists(full_path):
            return f"Error: File '{path}' does not exist."
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@registry.register(auth_required=True)
def write_file(path: str, content: str) -> str:
    """
    Write or update a file. Shows a diff if the file already exists.
    """
    try:
        full_path = get_workspace_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        old_content = ""
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                old_content = f.read()
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if old_content:
            diff = difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}"
            )
            diff_text = "".join(diff)
            return f"File updated: {path}\n\nDiff:\n{diff_text}" if diff_text else "No changes detected."
        return f"File created: {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@registry.register
def list_dir(path: str = ".") -> str:
    """
    List files in the workspace. Use this to explore the project structure.
    """
    try:
        full_path = get_workspace_path(path)
        if not os.path.exists(full_path):
            return f"Error: Directory '{path}' does not exist."
        items = os.listdir(full_path)
        return "\n".join(items) if items else "Directory is empty."
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@registry.register
def web_fetch(url: str) -> str:
    """
    Fetch URL content for research, documentation, or news.
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=20.0) as client:
            headers = {"User-Agent": "Mozilla/5.0 (EzClaw/2.0 ResearchBot)"}
            response = client.get(url, headers=headers)
            response.raise_for_status()
            text = clean_html(response.text)
            
            # Context window protection: Limit to 12,000 characters
            if len(text) > 12000:
                return f"--- CONTENT FROM {url} (TRUNCATED) ---\n" + text[:12000] + "\n... (Content truncated for length) ..."
            return f"--- CONTENT FROM {url} ---\n" + text
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

@registry.register
def schedule_task(scheduled_time: str, description: str) -> str:
    """
    Schedule a task (YYYY-MM-DD HH:MM).
    """
    try:
        datetime.strptime(scheduled_time, '%Y-%m-%d %H:%M')
        row = f"| {scheduled_time} | {description} | Pending |\n"
        with open("heartbeat.md", "a") as f:
            f.write(row)
        return f"Task scheduled: {description} at {scheduled_time}"
    except ValueError:
        return "Error: Use 'YYYY-MM-DD HH:MM' format."

@registry.register(auth_required=True)
def learn_skill(name: str, description: str, procedure: str) -> str:
    """
    Save a reusable step-by-step procedure/workflow.
    """
    try:
        os.makedirs(SKILLS_DIR, exist_ok=True)
        filename = f"{name.lower().replace(' ', '_')}.md"
        path = os.path.join(SKILLS_DIR, filename)
        content = f"# Skill: {name}\n\n## Description\n{description}\n\n## Procedure\n{procedure}\n"
        with open(path, 'w') as f:
            f.write(content)
        return f"Skill '{name}' saved to {path}."
    except Exception as e:
        return f"Error saving skill: {str(e)}"

@registry.register
def get_skill(name: str) -> str:
    """Retrieve a learned skill."""
    try:
        path = os.path.join(SKILLS_DIR, f"{name.lower().replace(' ', '_')}.md")
        if not os.path.exists(path): return f"Skill '{name}' not found."
        with open(path, 'r') as f: return f.read()
    except Exception as e: return str(e)

@registry.register
def list_skills() -> str:
    """List all known skills."""
    if not os.path.exists(SKILLS_DIR): return "No skills learned yet."
    skills = [f.replace('.md', '') for f in os.listdir(SKILLS_DIR) if f.endswith('.md')]
    return "\n".join(skills) if skills else "No skills learned yet."

def create_memory_tools(db: Any):
    """Integrates Database-backed memory tools."""
    @registry.register
    def remember(fact: str, tags: Optional[str] = None) -> str:
        """Store a fact, preference, or project detail in long-term memory."""
        db.add_memory(fact, tags)
        return f"Memory stored: {fact}"

    @registry.register
    def recall(query: str) -> str:
        """Search long-term memory for relevant information."""
        memories = db.search_memories(query)
        if not memories: return f"No memories found for '{query}'."
        return "Relevant memories:\n- " + "\n- ".join(memories)

    @registry.register
    def forget(query: str) -> str:
        """Remove facts from long-term memory using keywords."""
        deleted = db.delete_memory(query)
        if not deleted: return f"No memories found matching '{query}' to forget."
        return "Deleted memories:\n- " + "\n- ".join(deleted)

    return remember, recall, forget