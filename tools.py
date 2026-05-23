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

# Skills live OUTSIDE the project root so the agent has zero write-paths
# anywhere except ./workspace/. The bwrap jail in run_shell enforces this
# for arbitrary shell commands; the file tools (read_file/write_file/etc.)
# enforce it via get_workspace_path; and skill storage lives in user's
# home so a "learn_skill" call can't accidentally touch project source.
SKILLS_DIR = os.path.join(os.path.expanduser("~"), ".ezclaw", "skills")

# Legacy location — if files exist there from before this change they
# get migrated on first skill access.
_LEGACY_SKILLS_DIR = "skills"


def _migrate_legacy_skills_once() -> None:
    """One-time copy from project-root ./skills/ to ~/.ezclaw/skills/.
    Idempotent — if the destination already exists with files, do nothing.
    The source files are left in place (we don't delete user data)."""
    if not os.path.isdir(_LEGACY_SKILLS_DIR):
        return
    os.makedirs(SKILLS_DIR, exist_ok=True)
    for name in os.listdir(_LEGACY_SKILLS_DIR):
        if not name.endswith(".md"):
            continue
        src = os.path.join(_LEGACY_SKILLS_DIR, name)
        dst = os.path.join(SKILLS_DIR, name)
        if os.path.exists(dst):
            continue
        try:
            with open(src, "r", encoding="utf-8") as fsrc:
                content = fsrc.read()
            with open(dst, "w", encoding="utf-8") as fdst:
                fdst.write(content)
        except OSError:
            pass

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

# Environment variables the sandbox passes through to children. Everything
# else (including API keys, OLLAMA_* config, PYTHONPATH, VIRTUAL_ENV) is
# stripped. Keep this set conservative.
_ENV_ALLOWLIST = frozenset({
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "TMPDIR",
    "TZ",
})

# Hard resource ceilings for shell children (POSIX only). These cap a runaway
# command's blast radius without preventing reasonable builds.
_RLIMIT_AS_BYTES = 2 * 1024 * 1024 * 1024     # 2 GB virtual address space
_RLIMIT_CPU_SECONDS = 60                       # 60 s CPU time (matches the 60 s wall timeout)
_RLIMIT_FSIZE_BYTES = 100 * 1024 * 1024        # 100 MB max single-file write
_RLIMIT_CORE = 0                               # no core dumps
# RLIMIT_NPROC was removed: it broke bwrap's namespace creation
# (CLONE_NEWUSER hits EAGAIN under tight nproc caps). The bwrap mount
# namespace already destroys all child processes when the jail exits,
# so a fork bomb in there can't affect the host.


def _build_sandbox_env() -> dict:
    """Return a scrubbed environment dict for shell children.

    Only variables in `_ENV_ALLOWLIST` pass through. We add a sentinel
    `EZCLAW_SANDBOX=1` so nested invocations can detect they're inside
    the sandbox. `PWD` is set to the workspace; the child's cwd will
    match (Popen.cwd= takes precedence anyway, this is just for shells
    that consult $PWD for prompts).
    """
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env["EZCLAW_SANDBOX"] = "1"
    env["PWD"] = os.path.abspath(WORKSPACE_DIR)
    # PATH is mandatory for command lookup; provide a sane default if the
    # parent had none (rare but possible in stripped CI environments).
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return env


def _ensure_workspace_self_symlink(workspace_cwd: str) -> None:
    """Create `./workspace/workspace -> .` so a bogus extra `workspace/`
    segment in a shell path is a filesystem-level no-op.

    Local LLMs sometimes do `run_shell("touch workspace/foo.py")` even
    though run_shell is already cwd'd to ./workspace/. That used to land
    at ./workspace/workspace/foo.py and stay there. With this self-symlink,
    the path `workspace/foo.py` from inside `./workspace/` resolves via
    `workspace -> .` back to `./workspace/foo.py` — the right place.

    Idempotent. Skips if a real directory or file already occupies the
    path (so we never clobber existing data — the user can clean that
    up manually with `mv workspace/workspace/* workspace/ && rmdir
    workspace/workspace`).
    """
    link_path = os.path.join(workspace_cwd, WORKSPACE_DIR)
    if os.path.islink(link_path):
        return
    if os.path.exists(link_path):
        return  # real directory in the way — leave it alone
    try:
        os.symlink(".", link_path)
    except OSError:
        pass  # filesystem doesn't support symlinks (rare) — best-effort


def _apply_sandbox_rlimits() -> None:
    """preexec_fn that caps the child's resource usage. POSIX only.

    Runs in the forked child between fork() and exec(). Must not raise
    or the child will die before exec — every limit is wrapped in its
    own try/except because not all platforms support every rlimit
    (e.g. RLIMIT_NPROC is missing on macOS in some builds).
    """
    if os.name == "nt":
        return
    try:
        import resource
    except ImportError:
        return
    for name, value in (
        ("RLIMIT_AS",    _RLIMIT_AS_BYTES),
        ("RLIMIT_CPU",   _RLIMIT_CPU_SECONDS),
        ("RLIMIT_FSIZE", _RLIMIT_FSIZE_BYTES),
        ("RLIMIT_CORE",  _RLIMIT_CORE),
    ):
        which = getattr(resource, name, None)
        if which is None:
            continue
        try:
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            # The hard limit may already be lower (e.g. RLIMIT_NPROC on a
            # systemd-restricted user). Don't fail the run for that.
            pass


@registry.register(auth_required=True)
def run_shell(command: str, interactive: bool = False) -> str:
    """
    Execute a shell command from the workspace directory.

    Process-level protections (always on):
      - cwd is `./workspace/`. The parent process's cwd is never mutated,
        even for interactive commands — cwd is passed via Popen, not via
        os.chdir.
      - Environment is scrubbed to a small allowlist (PATH, HOME, USER,
        SHELL, TERM, LANG, LC_*, TMPDIR, TZ). API keys, OLLAMA_* config,
        PYTHONPATH, and VIRTUAL_ENV are dropped. `EZCLAW_SANDBOX=1` is set.
      - The child runs in its own session/process group (`start_new_session`)
        so signals don't cross the boundary.
      - POSIX rlimits: 2 GB virtual memory, 60 s CPU time, 100 MB max file
        size, no core dumps.

    NOTE: there is NO filesystem jail. The shell can technically write
    anywhere the user has permission to write. The convention is that
    the agent confines all writes to `./workspace/`; the file tools
    (read_file/write_file/list_dir) enforce this via WorkspacePathError,
    but `run_shell` runs arbitrary shell — it's the agent's responsibility
    to keep its writes inside the workspace.

    interactive=True allocates a pty so commands that need a tty (sudo,
    ssh, vim, installers) work correctly.
    """
    workspace_cwd = os.path.abspath(WORKSPACE_DIR)
    os.makedirs(workspace_cwd, exist_ok=True)
    _ensure_workspace_self_symlink(workspace_cwd)
    sandbox_env = _build_sandbox_env()

    try:
        if interactive and os.name != "nt":
            return _run_shell_interactive(command, workspace_cwd, sandbox_env)

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=workspace_cwd,
            env=sandbox_env,
            start_new_session=True,
            preexec_fn=_apply_sandbox_rlimits if os.name != "nt" else None,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return output or "Command executed successfully with no output."
    except Exception as e:
        return f"Error executing command: {str(e)}"


def _run_shell_interactive(command: str, workspace_cwd: str, sandbox_env: dict) -> str:
    """Interactive shell via a pty pair, with the same sandboxing as the
    non-interactive path. The parent's cwd is NEVER mutated — cwd flows
    through Popen's `cwd=` kwarg directly to the forked child.
    """
    import pty
    import select
    import sys

    master_fd, slave_fd = pty.openpty()
    output_data: list[bytes] = []
    proc = None
    try:
        proc = subprocess.Popen(
            ["/bin/sh", "-c", command],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=workspace_cwd,
            env=sandbox_env,
            start_new_session=True,
            preexec_fn=_apply_sandbox_rlimits,
            close_fds=True,
        )
        # Close the slave in the parent — the child owns it now.
        os.close(slave_fd)
        slave_fd = -1

        stdin_fd = sys.stdin.fileno() if sys.stdin.isatty() else None

        # Pump bytes between the user's terminal and the pty master.
        while True:
            if proc.poll() is not None:
                # Drain any final output then exit
                try:
                    while True:
                        data = os.read(master_fd, 4096)
                        if not data:
                            break
                        try:
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                        except Exception:
                            pass
                        output_data.append(data)
                except OSError:
                    pass
                break

            rlist = [master_fd]
            if stdin_fd is not None:
                rlist.append(stdin_fd)
            try:
                r, _, _ = select.select(rlist, [], [], 0.1)
            except (OSError, ValueError):
                break

            if master_fd in r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                try:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                except Exception:
                    pass
                output_data.append(data)

            if stdin_fd is not None and stdin_fd in r:
                try:
                    user_input = os.read(stdin_fd, 4096)
                except OSError:
                    user_input = b""
                if user_input:
                    try:
                        os.write(master_fd, user_input)
                    except OSError:
                        pass
    finally:
        try:
            if slave_fd >= 0:
                os.close(slave_fd)
        except OSError:
            pass
        try:
            os.close(master_fd)
        except OSError:
            pass
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    full_output = b"".join(output_data).decode("utf-8", errors="ignore")
    clean_output = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", full_output)
    return clean_output or "Interactive command completed."

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
    Schedule a task to run at a future time (YYYY-MM-DD HH:MM).

    When the time arrives, ezclaw's heartbeat monitor will auto-execute
    the description as if the user had typed it — the agent plans and
    runs it autonomously. Returns the new task's stable ID, which can be
    passed to unschedule_task() to cancel before it fires.
    """
    from scheduler import Scheduler
    try:
        task = Scheduler().schedule(scheduled_time, description)
        return f"Task scheduled [#{task.id}]: {task.description} at {task.time_str}"
    except ValueError:
        return "Error: Use 'YYYY-MM-DD HH:MM' format."
    except Exception as e:
        return f"Error scheduling task: {e}"


@registry.register
def unschedule_task(task_id: int) -> str:
    """
    Cancel a pending scheduled task by its ID.

    Returns success or an explanation of why it couldn't be cancelled
    (e.g. unknown ID, task already completed). Use list_scheduled_tasks
    to see active IDs.
    """
    from scheduler import Scheduler
    try:
        task = Scheduler().unschedule(int(task_id))
        if task is None:
            return f"No pending task with ID {task_id} (already done, cancelled, or unknown)."
        return f"Cancelled task [#{task.id}]: {task.description} (was scheduled for {task.time_str})"
    except (TypeError, ValueError):
        return f"Error: task_id must be an integer, got {task_id!r}"
    except Exception as e:
        return f"Error unscheduling task: {e}"


@registry.register
def list_scheduled_tasks() -> str:
    """
    List all active (Pending or Notified) scheduled tasks with their IDs.

    Useful before calling unschedule_task or to remind the user what's
    queued up.
    """
    from scheduler import Scheduler
    try:
        tasks = Scheduler().list_pending()
        if not tasks:
            return "No scheduled tasks pending."
        lines = ["Active scheduled tasks:"]
        for t in sorted(tasks, key=lambda x: x.time):
            lines.append(f"  [#{t.id}] {t.time_str}  {t.status}  — {t.description}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing tasks: {e}"

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

def _skill_filename(name: str) -> str:
    """Convert a user-facing skill name to a safe filename. Strips any
    path-traversal attempts (../, leading /, etc.) and reduces to
    [a-z0-9_-] so a malicious name can't escape SKILLS_DIR."""
    safe = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_")
    if not safe:
        safe = "unnamed"
    return f"{safe}.md"


@registry.register(auth_required=True)
def learn_skill(name: str, description: str, procedure: str) -> str:
    """
    Save a reusable step-by-step procedure to ~/.ezclaw/skills/.
    """
    try:
        _migrate_legacy_skills_once()
        os.makedirs(SKILLS_DIR, exist_ok=True)
        path = os.path.join(SKILLS_DIR, _skill_filename(name))
        content = f"# Skill: {name}\n\n## Description\n{description}\n\n## Procedure\n{procedure}\n"
        with open(path, "w") as f:
            f.write(content)
        return f"Skill '{name}' saved to {path}."
    except Exception as e:
        return f"Error saving skill: {str(e)}"


@registry.register
def get_skill(name: str) -> str:
    """Retrieve a learned skill."""
    try:
        _migrate_legacy_skills_once()
        path = os.path.join(SKILLS_DIR, _skill_filename(name))
        if not os.path.exists(path):
            return f"Skill '{name}' not found."
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return str(e)


@registry.register
def list_skills() -> str:
    """List all known skills."""
    _migrate_legacy_skills_once()
    if not os.path.exists(SKILLS_DIR):
        return "No skills learned yet."
    skills = [f.replace(".md", "") for f in os.listdir(SKILLS_DIR) if f.endswith(".md")]
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