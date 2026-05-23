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

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Generate Ollama-compatible tool definitions for all registered tools.
        """
        definitions = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            doc = func.__doc__.strip() if func.__doc__ else "No description provided."
            
            # Simple docstring parsing: use first line as description
            description = doc.split("\n")[0].strip()
            
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param_name, param in sig.parameters.items():
                # Map Python types to JSON Schema types
                p_type = "string"
                if param.annotation == int: p_type = "integer"
                elif param.annotation == bool: p_type = "boolean"
                elif param.annotation == float: p_type = "number"
                elif param.annotation == list: p_type = "array"
                elif param.annotation == dict: p_type = "object"
                
                parameters["properties"][param_name] = {
                    "type": p_type,
                    "description": f"Parameter {param_name}"
                }
                
                if param.default is inspect.Parameter.empty:
                    parameters["required"].append(param_name)
            
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            })
        return definitions

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

class WorkspacePathError(ValueError):
    """Raised when a tool path cannot be safely resolved inside the workspace."""

def get_workspace_path(path: str) -> str:
    """Resolve a tool-supplied path to an absolute path inside the workspace.

    Failure modes the agent kept hitting:
      - Passes `/home/lulu/Projects/ezclaw/foo.py` (an absolute project path).
        The old code did `lstrip('/')` and joined it onto workspace/, producing
        `workspace/home/lulu/Projects/ezclaw/foo.py`. Now we raise instead, so
        the agent gets a clear error and corrects itself.
      - Passes `workspace/foo.py` or even `workspace/workspace/foo.py`.
        We strip every leading `workspace/` segment, not just one.
      - Uses `..` to escape. Caught by the final boundary check.
    """
    if not isinstance(path, str) or not path.strip():
        raise WorkspacePathError("Path is empty.")

    base_dir = os.path.abspath(WORKSPACE_DIR)
    base_with_sep = base_dir + os.path.sep
    norm_path = os.path.normpath(path)

    if os.path.isabs(norm_path):
        # Absolute is only allowed if it already lives inside the sandbox.
        # We require a separator boundary so /foo/workspace doesn't masquerade
        # as a child of /foo/workspace-other.
        if norm_path == base_dir or norm_path.startswith(base_with_sep):
            return norm_path
        raise WorkspacePathError(
            f"Path '{path}' is outside the workspace sandbox. "
            f"Pass a relative path (e.g. 'foo.py', 'subdir/bar.py'), "
            f"not an absolute path. The sandbox root is '{base_dir}'."
        )

    # Strip every leading `workspace/` segment the agent may have tacked on.
    rel_parts = [p for p in norm_path.split(os.path.sep) if p not in ("", ".")]
    while rel_parts and rel_parts[0] == WORKSPACE_DIR:
        rel_parts = rel_parts[1:]
    norm_path = os.path.sep.join(rel_parts) if rel_parts else "."

    final_path = os.path.abspath(os.path.join(base_dir, norm_path))

    if final_path != base_dir and not final_path.startswith(base_with_sep):
        raise WorkspacePathError(
            f"Path '{path}' escapes the workspace sandbox via '..' or similar."
        )

    return final_path

@registry.register
def get_system_info() -> str:
    """
    Get information about the current system (OS, shell, user, package manager).
    Use this to adapt your commands to the local environment.
    """
    try:
        import platform
        import getpass
        info = [
            f"OS: {platform.system()} {platform.release()}",
            f"User: {getpass.getuser()}",
            f"Shell: {os.getenv('SHELL', 'unknown')}",
            f"Python: {platform.python_version()}",
        ]
        
        # Check for package managers
        managers = []
        for pm in ["apt", "dnf", "yum", "pacman", "brew", "pip"]:
            if subprocess.run(f"which {pm}", shell=True, capture_output=True).returncode == 0:
                managers.append(pm)
        if managers:
            info.append(f"Package Managers: {', '.join(managers)}")
            
        return "\n".join(info)
    except Exception as e:
        return f"Error getting system info: {str(e)}"

@registry.register(auth_required=True)
def run_shell(command: str, interactive: bool = False) -> str:
    """
    Execute a shell command from the workspace directory.

    All commands run with cwd=workspace/ so that shell paths line up with
    the paths read_file/write_file/list_dir use. This eliminates the
    silent split where the agent edited workspace/X but built ./X.

    Use interactive=True for commands that require user input (sudo, vim, ssh, interactive scripts, installers).
    For 'ls', 'grep', 'cat', etc., keep interactive=False for faster, cleaner output.
    """
    workspace_cwd = os.path.abspath(WORKSPACE_DIR)
    os.makedirs(workspace_cwd, exist_ok=True)
    try:
        if interactive and os.name != 'nt':
            import pty
            import sys

            output_data = []
            def read(fd):
                data = os.read(fd, 1024)
                if data:
                    try:
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    except Exception:
                        pass
                    output_data.append(data)
                return data

            # pty.spawn doesn't take a cwd kwarg, so we shift to workspace
            # in the parent before spawning. pty.spawn forks; the child
            # inherits cwd. We restore the parent's cwd afterward so other
            # tools (like the embedding client or DB lookups) aren't
            # affected.
            prev_cwd = os.getcwd()
            os.chdir(workspace_cwd)
            try:
                pty.spawn(['/bin/sh', '-c', command], read)
            finally:
                os.chdir(prev_cwd)

            full_output = b"".join(output_data).decode('utf-8', errors='ignore')
            clean_output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', full_output)
            return clean_output or "Interactive command completed."

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=workspace_cwd,
        )
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

@registry.register
def generate_codebase_map(path: str = ".") -> str:
    """
    Generate a structural map of the codebase, identifying classes, functions, and key files.
    Use this to get a high-level overview of the project architecture.
    """
    try:
        import ast
        full_path = get_workspace_path(path)
        if not os.path.exists(full_path):
            return f"Error: Path '{path}' does not exist."

        ignore_dirs = {'.git', 'venv', '__pycache__', 'node_modules', '.gemini'}
        map_lines = [f"Codebase Map for: {path}"]
        
        for root, dirs, files in os.walk(full_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            rel_path = os.path.relpath(root, full_path)
            indent = "  " * (0 if rel_path == "." else rel_path.count(os.sep) + 1)
            
            if rel_path != ".":
                map_lines.append(f"{indent}📁 {os.path.basename(root)}/")

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    map_lines.append(f"{indent}  📄 {file}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            tree = ast.parse(f.read())
                        for node in tree.body:
                            if isinstance(node, ast.ClassDef):
                                map_lines.append(f"{indent}    class {node.name}")
                                for subnode in node.body:
                                    if isinstance(subnode, ast.FunctionDef):
                                        map_lines.append(f"{indent}      def {subnode.name}")
                            elif isinstance(node, ast.FunctionDef):
                                map_lines.append(f"{indent}    def {node.name}")
                    except Exception:
                        pass
                elif file.endswith(('.js', '.ts', '.go', '.rs', '.c', '.cpp', '.h')):
                     map_lines.append(f"{indent}  📄 {file}")

        return "\n".join(map_lines)
    except Exception as e:
        return f"Error generating map: {str(e)}"

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

@registry.register
def delegate(agent_key: str, instruction: str) -> str:
    """
    Delegate a task to another agent. Use when you need capabilities outside your scope.
    agent_key: one of executor, researcher, debugger, general
    instruction: exactly what the target agent should do
    """
    allowed = ("executor", "researcher", "debugger", "general")
    if agent_key not in allowed:
        return f"Error: agent_key must be one of {allowed}"
    return f"[DELEGATE:{agent_key}]{instruction}[/DELEGATE]"