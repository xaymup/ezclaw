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
            # Called as @register(auth_required=True)
            def decorator(f):
                f.auth_required = auth_required
                self.tools[f.__name__] = f
                return f
            return decorator
        
        # Called as @register
        func.auth_required = auth_required
        self.tools[func.__name__] = func
        return func

    def get_tool_functions(self) -> List[Callable]:
        return list(self.tools.values())

registry = ToolRegistry()

WORKSPACE_DIR = "workspace"

def clean_html(html: str) -> str:
    """Basic HTML cleaning to remove tags and extra whitespace."""
    html = re.sub(r'<(script|style)\b[^>]*>.*?</\1>', '', html, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_workspace_path(path: str) -> str:
    """Ensure path is within the workspace directory."""
    # Remove leading slash/dots to prevent escaping
    safe_path = path.lstrip("./").lstrip("/")
    return os.path.join(WORKSPACE_DIR, safe_path)

@registry.register(auth_required=True)
def run_shell(command: str) -> str:
    """
    Execute a shell command and return the output.
    Use this for system tasks, running scripts, or gathering info.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
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
            return f"Error: File '{path}' does not exist in workspace."
        with open(full_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@registry.register(auth_required=True)
def write_file(path: str, content: str) -> str:
    """
    Write or overwrite a file within the workspace.
    """
    try:
        full_path = get_workspace_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        old_content = ""
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                old_content = f.read()
        
        with open(full_path, 'w') as f:
            f.write(content)
        
        if old_content:
            diff = difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}"
            )
            diff_text = "".join(diff)
            if diff_text:
                return f"File updated: workspace/{path}\n\nDiff:\n{diff_text}"
            else:
                return f"File workspace/{path} remains unchanged (content identical)."
        else:
            return f"File successfully created: workspace/{path}\n\nContent:\n{content}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@registry.register
def list_dir(path: str = ".") -> str:
    """
    List the files and directories in the workspace.
    """
    try:
        full_path = get_workspace_path(path)
        if not os.path.exists(full_path):
            return f"Error: Directory '{path}' does not exist in workspace."
        items = os.listdir(full_path)
        return "\n".join(items) if items else "Directory is empty."
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@registry.register
def web_fetch(url: str) -> str:
    """
    Fetch the content of a URL and return it as clean text.
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=15.0) as client:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = client.get(url, headers=headers)
            response.raise_for_status()
            clean_text = clean_html(response.text)
            return clean_text[:8000]
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

@registry.register
def schedule_task(scheduled_time: str, description: str) -> str:
    """
    Schedule a task to be performed at a specific time. Format: 'YYYY-MM-DD HH:MM'.
    """
    try:
        datetime.strptime(scheduled_time, '%Y-%m-%d %H:%M')
        row = f"| {scheduled_time} | {description} | Pending |\n"
        with open("heartbeat.md", "a") as f:
            f.write(row)
        return f"Task scheduled for {scheduled_time}: {description}"
    except ValueError:
        return "Error: Invalid time format. Please use 'YYYY-MM-DD HH:MM'."
    except Exception as e:
        return f"Error scheduling task: {str(e)}"

@registry.register
def list_tasks() -> str:
    """
    List all scheduled tasks.
    """
    try:
        if not os.path.exists("heartbeat.md"):
            return "No tasks scheduled."
        with open("heartbeat.md", "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading heartbeat.md: {str(e)}"

SKILLS_DIR = "skills"

@registry.register(auth_required=True)
def learn_skill(name: str, description: str, procedure: str) -> str:
    """
    Save a new skill or procedure that the agent has learned.
    'name': A short, unique name for the skill (e.g., 'git_workflow').
    'description': What this skill does.
    'procedure': The step-by-step instructions or code.
    """
    try:
        os.makedirs(SKILLS_DIR, exist_ok=True)
        filename = f"{name.lower().replace(' ', '_')}.md"
        path = os.path.join(SKILLS_DIR, filename)
        
        content = f"# Skill: {name}\n\n## Description\n{description}\n\n## Procedure\n{procedure}\n"
        
        with open(path, 'w') as f:
            f.write(content)
        return f"Skill '{name}' has been learned and saved to {path}."
    except Exception as e:
        return f"Error learning skill: {str(e)}"

@registry.register
def get_skill(name: str) -> str:
    """
    Retrieve the details of a specific learned skill.
    """
    try:
        filename = f"{name.lower().replace(' ', '_')}.md"
        path = os.path.join(SKILLS_DIR, filename)
        if not os.path.exists(path):
            return f"Skill '{name}' not found."
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error retrieving skill: {str(e)}"

@registry.register
def list_skills() -> str:
    """
    List all skills currently learned by the agent.
    """
    try:
        if not os.path.exists(SKILLS_DIR):
            return "No skills learned yet."
        skills = [f.replace('.md', '') for f in os.listdir(SKILLS_DIR) if f.endswith('.md')]
        return "\n".join(skills) if skills else "No skills learned yet."
    except Exception as e:
        return f"Error listing skills: {str(e)}"

def create_memory_tools(db: Any):
    @registry.register
    def remember(fact: str, tags: Optional[str] = None) -> str:
        """Store a fact in long-term memory."""
        db.add_memory(fact, tags)
        return f"Fact remembered: {fact}"

    @registry.register
    def recall(query: str) -> str:
        """Search and retrieve information from long-term memory."""
        memories = db.search_memories(query)
        if not memories:
            return f"No memories found for query: '{query}'"
        return "Relevant memories:\n- " + "\n- ".join(memories)
    
    return remember, recall
