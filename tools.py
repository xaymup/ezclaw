import os
import subprocess
import httpx
import re
import difflib
import urllib.parse
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional
import inspect
import contextvars

# Whitelist of tools whose calls are recorded as actions. Reads/searches
# are excluded — only state-changing operations qualify.
MUTATING_TOOLS = frozenset({
    "apply_diff",
    "write_file",
    "run_shell",
    "schedule_task",
    "unschedule_task",
})

# Set by the agent at session start; read by recall_actions to scope its
# search. Module-level so it survives across the tool's invocation
# without threading a parameter through every call site.
_session_id_var: contextvars.ContextVar = contextvars.ContextVar(
    "ezclaw_session_id", default=None
)


def set_session_context(session_id):
    """Called by the agent at session start. Pass None to clear."""
    _session_id_var.set(session_id)


def get_session_context():
    """Return the current session id, or None if unset."""
    return _session_id_var.get()


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
def current_datetime() -> str:
    """Get the current local date and time, formatted multiple ways for
    easy reasoning. Useful for scheduling tasks (compute "tomorrow at 9am"
    from the current moment), for timestamping notes, and for any
    request that involves relative time expressions ("in 2 hours",
    "next Monday", etc.).
    """
    from datetime import datetime, timezone
    now = datetime.now()
    now_utc = datetime.now(timezone.utc)
    tz_offset = now.astimezone().strftime("%z") or "+0000"
    return (
        f"Local date:  {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})\n"
        f"Local time:  {now.strftime('%H:%M:%S')}\n"
        f"Timezone:    UTC{tz_offset[:3]}:{tz_offset[3:]}\n"
        f"ISO format:  {now.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"For scheduling (schedule_task format): {now.strftime('%Y-%m-%d %H:%M')}"
    )


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


def _run_shell_interactive(
    command: str,
    workspace_cwd: str,
    sandbox_env: dict,
    on_output=None,
    input_provider=None,
    on_proc_spawn=None,
) -> str:
    """Interactive shell via a pty pair, with the same sandboxing as the
    non-interactive path. The parent's cwd is NEVER mutated — cwd flows
    through Popen's `cwd=` kwarg directly to the forked child.

    UI-embeddable: when `on_output` and `input_provider` are supplied,
    output bytes go to the callback (instead of sys.stdout) and input
    bytes come from the provider (instead of the real tty). This lets
    the TUI stream subprocess output into a live tool panel and route
    user chat input to the subprocess, without suspending the TUI.

    When both are None (back-compat), behavior matches the old terminal-
    takeover path: output → sys.stdout, input ← sys.stdin.

    Returns a structured report — header with command, exit code,
    duration, and byte count, followed by the cleaned terminal output.
    """
    import pty
    import select
    import sys
    import time as _time

    start_time = _time.time()
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
        # Expose the spawned proc to the caller (cli wrapper) so it can
        # forward signals — Ctrl+C in the TUI sends SIGINT to the
        # subprocess group instead of killing ezclaw.
        if on_proc_spawn is not None:
            try:
                on_proc_spawn(proc)
            except Exception:
                pass
        # Close the slave in the parent — the child owns it now.
        os.close(slave_fd)
        slave_fd = -1

        # When no callbacks are supplied, fall back to the legacy
        # terminal-takeover behavior (write to sys.stdout, read from
        # sys.stdin). The TUI uses the callback path; smoke tests and
        # any future headless caller can still rely on the legacy path.
        use_callbacks = on_output is not None
        stdin_fd = None
        if not use_callbacks:
            stdin_fd = sys.stdin.fileno() if sys.stdin.isatty() else None

        def _emit(data: bytes) -> None:
            output_data.append(data)
            if use_callbacks:
                try:
                    on_output(data)
                except Exception:
                    pass
            else:
                try:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                except Exception:
                    pass

        def _check_user_input():
            """Forward one chunk of user input to the subprocess if any
            is pending. Two sources: input_provider (UI-embedded path),
            or the real tty (terminal-takeover path)."""
            user_input = b""
            if use_callbacks and input_provider is not None:
                try:
                    val = input_provider()
                except Exception:
                    val = None
                if val:
                    user_input = val if isinstance(val, (bytes, bytearray)) else str(val).encode()
            elif stdin_fd is not None and stdin_fd in r:
                try:
                    user_input = os.read(stdin_fd, 4096)
                except OSError:
                    user_input = b""
            if user_input:
                try:
                    os.write(master_fd, user_input)
                except OSError:
                    pass

        # Pump bytes between the subprocess and the active I/O source.
        while True:
            if proc.poll() is not None:
                # Drain any final output then exit
                try:
                    while True:
                        try:
                            data = os.read(master_fd, 4096)
                        except OSError:
                            break
                        if not data:
                            break
                        _emit(data)
                except OSError:
                    pass
                break

            rlist = [master_fd]
            if stdin_fd is not None:
                rlist.append(stdin_fd)
            try:
                r, _, _ = select.select(rlist, [], [], 0.05)
            except (OSError, ValueError):
                break

            if master_fd in r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                _emit(data)

            _check_user_input()
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
    # Also strip carriage-return-only lines that leak through cursor moves
    # (e.g. progress bars), and trim trailing blank lines.
    clean_output = "\n".join(
        line.rstrip("\r") for line in clean_output.splitlines()
        if line.strip("\r")
    ).strip()

    duration = _time.time() - start_time
    exit_code = proc.returncode if proc is not None else None
    status = (
        "succeeded" if exit_code == 0
        else f"failed (exit code {exit_code})" if exit_code is not None
        else "did not exit cleanly"
    )

    # Structured report so the agent's next turn can reason about what
    # happened. Without this, an empty output stream + missing exit code
    # made successful interactive commands look indistinguishable from
    # failed ones in the model's context.
    header = (
        f"[Interactive shell session — {status}]\n"
        f"Command: {command}\n"
        f"Exit code: {exit_code}\n"
        f"Duration: {duration:.1f}s\n"
        f"Captured terminal output ({len(clean_output)} chars):\n"
        f"───\n"
    )
    body = clean_output if clean_output else (
        "(no terminal output captured — common for commands that exit "
        "silently like a successful `sudo true`, a background daemon "
        "launch, or a command whose only side effect is filesystem changes)"
    )
    return header + body

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


_DDG_LITE_LINK_RE = re.compile(
    r"""<a[^>]*?href=["']([^"']+)["'][^>]*?class=['"]result-link['"][^>]*>(.*?)</a>""",
    re.S | re.I,
)
_DDG_LITE_SNIPPET_RE = re.compile(
    r"""<td[^>]*class=['"]result-snippet['"][^>]*>(.*?)</td>""",
    re.S | re.I,
)
_DDG_REDIRECT_RE = re.compile(r"uddg=([^&]+)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&(?:amp|quot|lt|gt|nbsp|#x?\w+);")
_HTML_ENTITY_MAP = {
    "&amp;": "&", "&quot;": '"', "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
    "&#39;": "'", "&#x27;": "'", "&#x2f;": "/", "&#x2F;": "/",
}


def _ddg_lite_strip(html_fragment: str) -> str:
    """Remove HTML tags and decode the small handful of entities DDG emits."""
    out = _HTML_TAG_RE.sub("", html_fragment)
    out = _HTML_ENTITY_RE.sub(lambda m: _HTML_ENTITY_MAP.get(m.group(0), m.group(0)), out)
    return out.strip()


def _ddg_lite_unwrap(href: str) -> str:
    """DDG wraps real URLs in //duckduckgo.com/l/?uddg=<encoded>&rut=... —
    unwrap to the destination."""
    m = _DDG_REDIRECT_RE.search(href)
    if m:
        return urllib.parse.unquote(m.group(1))
    if href.startswith("//"):
        return "https:" + href
    return href


def _search_ddg_lite(query: str, limit: int) -> list:
    """DuckDuckGo Lite HTML — anti-bot tolerant general web search. The
    /lite/ endpoint is designed for old browsers and returns a plain
    table of results; it gates much less aggressively than the JSON
    endpoint or html.duckduckgo.com.

    No API key. Free. Real general-purpose results (the same engine
    used by ddg.gg)."""
    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
    with httpx.Client(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": ua, "Accept-Language": "en-US,en;q=0.7"},
    ) as client:
        resp = client.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
        )
        resp.raise_for_status()
        html = resp.text

    links = _DDG_LITE_LINK_RE.findall(html)
    snippets = _DDG_LITE_SNIPPET_RE.findall(html)

    results = []
    for i, (href, raw_title) in enumerate(links[:limit]):
        title = _ddg_lite_strip(raw_title)
        if not title:
            continue
        url = _ddg_lite_unwrap(href)
        snippet = _ddg_lite_strip(snippets[i]) if i < len(snippets) else ""
        results.append({
            "title": title[:200],
            "url": url,
            "snippet": snippet[:400],
            "source": "ddg-lite",
        })
    return results


def _search_brave(query: str, limit: int, api_key: str) -> list:
    """Brave Search API — real general-purpose search. Best results, but
    requires an API key (free tier 2000/mo at search.brave.com/app/api)."""
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(limit, 20)},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    results = []
    for r in (data.get("web") or {}).get("results", [])[:limit]:
        results.append({
            "title": r.get("title", "").strip(),
            "url": r.get("url", ""),
            "snippet": r.get("description", "").strip(),
            "source": "brave",
        })
    return results


def _search_wikipedia(query: str, limit: int) -> list:
    """Wikipedia opensearch — free, no key, no anti-bot. Excellent for
    library names, language features, documented APIs; useless for raw
    error messages."""
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        resp = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch", "search": query,
                "limit": limit, "format": "json",
            },
            headers={"User-Agent": "ezclaw/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, list) or len(data) < 4:
        return []
    _, titles, snippets, urls = data[0], data[1], data[2], data[3]
    results = []
    for i in range(min(len(titles), limit)):
        results.append({
            "title": titles[i],
            "url": urls[i] if i < len(urls) else "",
            "snippet": snippets[i] if i < len(snippets) else "",
            "source": "wikipedia",
        })
    return results


def _search_ddg_instant(query: str) -> list:
    """DDG Instant Answer — narrow but reliable. Returns at most one
    encyclopedic entry for common topics."""
    with httpx.Client(timeout=10.0, follow_redirects=True) as client:
        resp = client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers={"User-Agent": "ezclaw/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    out = []
    if data.get("Abstract"):
        out.append({
            "title": data.get("Heading") or query,
            "url": data.get("AbstractURL", ""),
            "snippet": data.get("AbstractText", ""),
            "source": "ddg-instant",
        })
    for rt in (data.get("RelatedTopics") or [])[:4]:
        if isinstance(rt, dict) and rt.get("FirstURL"):
            out.append({
                "title": rt.get("Text", "")[:80] or query,
                "url": rt.get("FirstURL", ""),
                "snippet": rt.get("Text", ""),
                "source": "ddg-instant",
            })
    return out


@registry.register
def web_search(query: str, limit: int = 5) -> str:
    """
    Search the web and return top results (title, URL, snippet).

    Used to GROUND analysis in real sources rather than training-data
    guesses — when you hit an unfamiliar error, library behavior, or
    framework concept, search for the exact phrase before diagnosing.

    Backend chain (best → fallback):
      1. Brave Search API — real search, requires BRAVE_SEARCH_API_KEY
         (free tier 2000 queries/month at search.brave.com/app/api).
      2. DuckDuckGo Lite (/lite/) — real general-purpose search via
         the lite HTML endpoint, no key, anti-bot tolerant. This is
         the default workhorse when no Brave key is set.
      3. Wikipedia opensearch — encyclopedic backup for concept queries.
      4. DuckDuckGo Instant Answer — narrow, mostly Wikipedia-derived.

    Returns markdown-formatted results so the agent can cite URLs in
    its reasoning. Results are deduplicated by URL across backends.
    """
    if not query or not query.strip():
        return "Error: empty query."

    results = []
    errors = []

    def _extend(new_items: list):
        """Merge new results, deduplicating by URL."""
        for r in new_items:
            u = r.get("url", "")
            if u and any(u == existing["url"] for existing in results):
                continue
            results.append(r)
            if len(results) >= limit:
                return

    # Try backends in priority order. Continue past failures so a flaky
    # backend doesn't kill the whole search.
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if api_key:
        try:
            _extend(_search_brave(query, limit, api_key))
        except Exception as e:
            errors.append(f"brave: {e}")

    if len(results) < limit:
        try:
            _extend(_search_ddg_lite(query, limit - len(results)))
        except Exception as e:
            errors.append(f"ddg-lite: {e}")

    if len(results) < limit:
        try:
            _extend(_search_wikipedia(query, limit - len(results)))
        except Exception as e:
            errors.append(f"wikipedia: {e}")

    if len(results) < limit:
        try:
            _extend(_search_ddg_instant(query))
        except Exception as e:
            errors.append(f"ddg-instant: {e}")

    if not results:
        msg = f"No results found for {query!r}."
        if errors:
            msg += "\n\nBackend errors: " + "; ".join(errors)
        msg += (
            "\n\nIf this was a live-data query (weather, stock price, "
            "score), try web_fetch on a specific endpoint instead — e.g. "
            "https://wttr.in/<city>?format=3 for weather."
        )
        return msg

    lines = [f"## Search results for: {query}", ""]
    for i, r in enumerate(results[:limit], 1):
        title = r["title"] or "(untitled)"
        lines.append(f"{i}. **{title}**  [_{r['source']}_]")
        if r["url"]:
            lines.append(f"   {r['url']}")
        if r["snippet"]:
            snippet = r["snippet"][:300]
            if len(r["snippet"]) > 300:
                snippet += "…"
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()

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


def create_action_tracking_tools(db: Any):
    """Register the `recall_actions` tool. Bound to `db`; reads the current
    session id from the module-level contextvar set by the agent."""

    @registry.register
    def recall_actions(query: str, limit: int = 5) -> str:
        """Search this session's past mutating actions by what they did or why.

        Call when the user asks about a past action — 'what did you do
        about X', 'did you fix Y', 'which files did you edit earlier'.
        Returns one line per matching action: time, summary, why, outcome.
        """
        session_id = get_session_context()
        if session_id is None:
            return "No active session — action history unavailable."
        rows = db.search_actions(session_id=session_id, query=query, limit=limit)
        if not rows:
            return "No matching actions in this session."
        lines = []
        for r in rows:
            ts = (r.get("created_at") or "")[-8:-3] or "??:??"
            head = f"[{ts}] {r['summary']}"
            if r.get("why"):
                head += f" — why: \"{r['why']}\""
            head += f" — {r['outcome']}"
            if r.get("error_excerpt"):
                head += f": {r['error_excerpt'][:120]}"
            lines.append(head)
        return "\n".join(lines)


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


# ── ask_user / critique — wired to the UI at runtime ─────────────────────
#
# These tools require an interactive back-channel to either the human
# user (ask_user) or a second LLM (critique). The hosting environment
# registers a callable via the setters below; tools.py keeps the API
# stable while cli.py provides the actual implementation.

_ask_user_callback = None
_critique_callback = None


def set_ask_user_callback(cb):
    """Register the callback used by the ask_user tool. cli.py calls
    this at startup with a function that shows a prompt panel and blocks
    on user input."""
    global _ask_user_callback
    _ask_user_callback = cb


def set_critique_callback(cb):
    """Register the callback used by the critique tool. cli.py wires
    this to a second LLM call (architect model by default)."""
    global _critique_callback
    _critique_callback = cb


@registry.register
def ask_user(question: str) -> str:
    """
    Pause and ask the user a clarifying question instead of guessing.

    Use when:
      - the request is genuinely ambiguous (multiple files match,
        conflicting interpretations, missing key parameter)
      - you need a value only the user knows (server hostname, account
        name, target version)
      - you want explicit confirmation before a destructive action

    DON'T use when:
      - the answer is in the workspace (use read_file / grep_codebase)
      - the answer is online (use web_search)
      - you can reasonably guess and the cost of being wrong is low

    Returns the user's answer as a plain string. The agent pauses
    until the user types something.
    """
    q = (question or "").strip()
    if not q:
        return "Error: empty question."
    if _ask_user_callback is None:
        return (
            f"[ask_user fallback — no UI handler registered] question was: {q}\n"
            "Skip this question and continue with a reasonable default."
        )
    try:
        answer = _ask_user_callback(q)
        return answer.strip() if answer else "(user provided no answer)"
    except Exception as e:
        return f"Error: ask_user failed: {e}"


@registry.register
def critique(draft: str, context: str = "") -> str:
    """
    Get an adversarial second opinion on a draft answer, plan, or
    decision. Returns weaknesses, missing considerations, and factual
    issues the draft glossed over.

    Use sparingly — burns an extra LLM call. Best for:
      - high-stakes deliverables (a refactor plan, a debugging
        diagnosis, a security-touching change)
      - drafts that look "confident but wrong" smell-test wise
      - long final answers where you want a fresh pair of eyes

    `context` (optional): any relevant background — file content the
    draft references, the original user request, prior turns — that
    the critic should know to assess the draft fairly.
    """
    d = (draft or "").strip()
    if not d:
        return "Error: empty draft."
    if _critique_callback is None:
        return "[critique unavailable — no critic LLM registered in this environment]"
    try:
        return _critique_callback(d, context or "")
    except Exception as e:
        return f"Error: critique failed: {e}"


# ── Code-intelligence tools ────────────────────────────────────────────────

@registry.register
def code_outline(path: str) -> str:
    """
    Extract the symbol tree from a source file (imports, classes,
    functions, top-level assignments) without reading the full file.

    Saves context budget on large files: a 2000-line module is ~4000
    tokens to read in full vs ~200 tokens for its outline. Use this
    FIRST when exploring an unfamiliar file; read_file second only if
    you need a specific symbol's body.

    Python files use the AST for accurate parsing. Other languages fall
    back to a regex pass that catches def/function/class/fn/func
    declarations — useful but less precise.
    """
    try:
        full_path = get_workspace_path(path)
        if not os.path.exists(full_path):
            return f"Error: File '{path}' does not exist."

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()

        if path.endswith(".py"):
            return _code_outline_python(source, path)
        # Generic fallback: catches common declaration patterns
        return _code_outline_generic(source, path)
    except Exception as e:
        return f"Error outlining {path}: {e}"


def _code_outline_python(source: str, path: str) -> str:
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Error: SyntaxError in {path}: {e}"

    out = [f"# Outline: {path}", f"# {len(source.splitlines())} lines total", ""]

    def emit(node, depth=0):
        indent = "  " * depth
        if isinstance(node, ast.Import):
            names = ", ".join(a.name for a in node.names)
            out.append(f"{indent}import {names}  L{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = ", ".join(a.name for a in node.names)
            out.append(f"{indent}from {mod} import {names}  L{node.lineno}")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(_unparse_short(b) for b in node.bases)
            head = f"class {node.name}({bases})" if bases else f"class {node.name}"
            out.append(f"{indent}{head}:  L{node.lineno}")
            for child in node.body:
                emit(child, depth + 1)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            out.append(f"{indent}{kw} {node.name}({args})  L{node.lineno}")
        elif isinstance(node, ast.Assign) and depth == 0:
            targets = ", ".join(_unparse_short(t) for t in node.targets)
            out.append(f"{indent}{targets} = …  L{node.lineno}")

    for top in tree.body:
        emit(top, depth=0)
    return "\n".join(out)


def _unparse_short(node) -> str:
    import ast
    try:
        return ast.unparse(node)[:60]
    except Exception:
        return getattr(node, "id", "?")


def _code_outline_generic(source: str, path: str) -> str:
    """Regex-based outline for non-Python source. Catches common
    declaration patterns across C/Go/Rust/JS/TS."""
    lines = source.splitlines()
    patterns = [
        # JS/TS function / class / arrow exports
        (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)"), "function"),
        (re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"), "class"),
        (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*="), "binding"),
        # C / C++
        (re.compile(r"^\s*(?:static\s+|inline\s+|extern\s+)*(?:[\w*&]+\s+){1,3}(\w+)\s*\([^;]*\)\s*\{"), "func"),
        (re.compile(r"^\s*(?:typedef\s+)?(?:struct|enum|union)\s+(\w+)"), "type"),
        # Go
        (re.compile(r"^\s*func\s+(?:\([^)]*\)\s+)?(\w+)"), "func"),
        (re.compile(r"^\s*type\s+(\w+)\s+(?:struct|interface)"), "type"),
        # Rust
        (re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"), "fn"),
        (re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)"), "type"),
    ]
    out = [f"# Outline: {path}", f"# {len(lines)} lines total", ""]
    for lineno, line in enumerate(lines, 1):
        for pat, kind in patterns:
            m = pat.match(line)
            if m:
                out.append(f"{kind} {m.group(1)}  L{lineno}")
                break
    if len(out) == 3:
        # Nothing matched — emit first 30 lines as a fallback
        out.append("(no declarations matched; first 30 lines:)")
        out.append("")
        out.extend(lines[:30])
    return "\n".join(out)


# ── Apply diff ─────────────────────────────────────────────────────────────

@registry.register(auth_required=True)
def apply_diff(path: str, diff: str) -> str:
    """
    Apply a unified diff to a file in the workspace.

    Use this instead of `write_file` for small edits: it sends just the
    hunks, not the whole file. Saves context AND makes review clearer.

    `diff` should be a standard unified diff:
        @@ -10,3 +10,4 @@
         context line
        -removed line
        +added line 1
        +added line 2
         context line

    Tolerances (what the applier corrects for you):
    - Line numbers in `@@ -N,n +M,m @@` may be wrong — the applier
      fuzzy-searches the file for the pre-image and lands on the
      closest match to the header hint.
    - Context lines missing their leading space are still treated as
      context (e.g. `alpha` instead of ` alpha`).
    - Trailing whitespace and CRLF/LF differences are ignored.

    Hard requirements (you MUST get these right):
    - Removed (`-`) and context lines must match the file character-
      for-character on the meaningful content (indentation included
      — tabs vs spaces is a real difference).
    - At least one context or `-` line per hunk (no zero-pre-image
      hunks except pure insertions).

    On failure, the error lists each rejected hunk with its expected
    pre-image and the actual file content at that position so you can
    self-correct without re-reading the whole file.
    """
    try:
        full_path = get_workspace_path(path)
    except WorkspacePathError as e:
        return f"Error: {e}"
    if not os.path.exists(full_path):
        return f"Error: File '{path}' does not exist. Use write_file to create it."

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            original = f.read().splitlines(keepends=True)
    except Exception as e:
        return f"Error reading {path}: {e}"

    new_content, applied, rejections = _apply_unified_diff(original, diff)
    if rejections:
        return _format_rejections(rejections, path)
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("".join(new_content))
    except Exception as e:
        return f"Error writing {path}: {e}"

    return f"Applied {applied} hunk(s) to {path}."


def _apply_unified_diff(original_lines, diff_text):
    """Unified-diff applier with fuzzy line-number recovery.

    Returns (new_lines, applied_count, rejections). `rejections` is a
    list of dicts describing each failed hunk so the caller can build
    a diagnostic error message — empty list means everything applied.

    Each hunk is parsed into pre-image + post-image then matched
    against the file. We try the header-claimed offset first, and if
    that fails (LLMs frequently hallucinate line numbers), we scan
    the whole file for a unique position whose pre-image matches.
    """
    lines = diff_text.splitlines()
    new_content = list(original_lines)
    applied = 0
    rejections: list = []
    i = 0
    # Track cumulative offset from prior hunks so later hunks land at
    # the right line numbers after additions/deletions.
    offset = 0
    hunk_header = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

    hunk_idx = 0
    while i < len(lines):
        line = lines[i]
        m = hunk_header.match(line)
        if not m:
            i += 1
            continue

        hunk_idx += 1
        old_start = int(m.group(1))
        header_text = line
        # Collect hunk body until next @@ or end
        body = []
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            body.append(lines[i])
            i += 1

        pre, post = _parse_hunk_body(body)

        # Pure-insertion hunk (no pre-image): insert at the header offset.
        if not pre:
            idx = max(0, old_start - 1 + offset) if old_start else 0
            new_content[idx:idx] = post
            offset += len(post)
            applied += 1
            continue

        # Try the header-claimed offset first.
        idx = old_start - 1 + offset if old_start else 0
        if _pre_image_matches_at(new_content, idx, pre):
            new_content[idx:idx + len(pre)] = post
            offset += len(post) - len(pre)
            applied += 1
            continue

        # Fuzzy: scan the whole file for a position whose pre-image
        # matches. Prefer the position closest to the header hint.
        found = _scan_for_pre_image(new_content, pre, prefer_near=idx)
        if found is not None:
            new_content[found:found + len(pre)] = post
            offset += len(post) - len(pre) + (found - idx)
            applied += 1
            continue

        rejections.append({
            "index": hunk_idx,
            "header": header_text,
            "expected": [p.rstrip("\n") for p in pre],
            "found_near": _peek_window(new_content, idx, len(pre)),
        })

    return new_content, applied, rejections


def _parse_hunk_body(body):
    """Split a hunk body into pre-image / post-image lists, each ending
    with `\\n` so we can splice straight into the file content list."""
    pre, post = [], []
    for bl in body:
        if not bl:
            # Truly empty body line = a context line for an empty
            # line in the file.
            pre.append("\n")
            post.append("\n")
            continue
        tag, rest = bl[0], bl[1:]
        # Restore the trailing newline so the line matches lines read
        # with splitlines(keepends=True).
        line_with_nl = rest if rest.endswith("\n") else rest + "\n"
        if tag == "-":
            pre.append(line_with_nl)
        elif tag == "+":
            post.append(line_with_nl)
        elif tag == " ":
            pre.append(line_with_nl)
            post.append(line_with_nl)
        elif tag == "\\":
            # "\ No newline at end of file" — drop the trailing newline
            # from the most recently appended line in each arr.
            for arr in (pre, post):
                if arr and arr[-1].endswith("\n"):
                    arr[-1] = arr[-1][:-1]
        else:
            # Tolerate prefix-less context lines (an LLM mistake where
            # the leading space got eaten). Treat as context.
            line_with_nl = bl if bl.endswith("\n") else bl + "\n"
            pre.append(line_with_nl)
            post.append(line_with_nl)
    return pre, post


def _pre_image_matches_at(content, start_idx, pre):
    """True iff `pre` lines match `content[start_idx:]` line-for-line
    using rstrip-aware equality (tolerates trailing whitespace and
    CRLF/LF differences)."""
    if start_idx < 0 or start_idx + len(pre) > len(content):
        return False
    for j, expected in enumerate(pre):
        if content[start_idx + j].rstrip() != expected.rstrip():
            return False
    return True


def _scan_for_pre_image(content, pre, prefer_near=0):
    """Find an offset where `pre` matches the file. When multiple
    matches exist, pick the one closest to `prefer_near` (the
    header-claimed offset)."""
    if not pre:
        return None
    matches = []
    max_start = len(content) - len(pre)
    for start in range(0, max_start + 1):
        if _pre_image_matches_at(content, start, pre):
            matches.append(start)
    if not matches:
        return None
    # Closest-to-hint wins; ties broken by lower index.
    matches.sort(key=lambda s: (abs(s - prefer_near), s))
    return matches[0]


def _peek_window(content, idx, length):
    """Return a small slice of the file around `idx` for error messages."""
    if not content:
        return []
    lo = max(0, idx)
    hi = min(len(content), idx + max(length, 3))
    return [content[j].rstrip("\n") for j in range(lo, hi)]


def _format_rejections(rejections, path):
    """Build a human-readable error block when one or more hunks fail.
    Designed for the agent to read and self-correct, not the user."""
    parts = [
        f"Error: {len(rejections)} hunk(s) failed to apply to {path}.",
        "",
    ]
    for rej in rejections:
        parts.append(f"── Hunk #{rej['index']}  {rej['header']}")
        parts.append("  Expected (pre-image):")
        for line in rej["expected"][:8]:
            parts.append(f"    │ {line}")
        if len(rej["expected"]) > 8:
            parts.append(f"    │ … ({len(rej['expected']) - 8} more lines)")
        parts.append("  Found at that position in the file:")
        if rej["found_near"]:
            for line in rej["found_near"][:8]:
                parts.append(f"    │ {line}")
        else:
            parts.append("    (past end of file)")
        parts.append("")
    parts.append(
        "The hunk's context lines don't match the file. Re-read the file "
        "with read_file to get the exact lines, then regenerate the diff. "
        "If the change is small, write_file is simpler."
    )
    return "\n".join(parts)


# ── Codebase grep ──────────────────────────────────────────────────────────

@registry.register
def grep_codebase(pattern: str, path: str = ".", max_results: int = 100) -> str:
    """
    Search the workspace for lines matching `pattern` (Python regex).

    Returns matches as `path:line: matched_line`, capped at max_results.
    Skips binary files, .git/, node_modules/, venv/, __pycache__/, dist/,
    build/, .pytest_cache/, target/, *.pyc.

    Use this for "find every place X is used / defined / referenced"
    instead of constructing fragile `run_shell("grep -rn ...")` calls.
    """
    if not pattern:
        return "Error: empty pattern."
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex: {e}"
    try:
        base = get_workspace_path(path)
    except WorkspacePathError as e:
        return f"Error: {e}"
    if not os.path.exists(base):
        return f"Error: path {path!r} does not exist in workspace."

    SKIP_DIRS = {
        ".git", "node_modules", "venv", ".venv", "__pycache__", "dist",
        "build", ".pytest_cache", "target", ".cache", ".next",
    }
    SKIP_SUFFIXES = (".pyc", ".pyo", ".o", ".so", ".dylib", ".class",
                     ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip",
                     ".gz", ".tar", ".webp", ".ico", ".woff", ".woff2",
                     ".ttf", ".otf", ".eot")

    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if fname.endswith(SKIP_SUFFIXES):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        if compiled.search(line):
                            rel = os.path.relpath(fpath, base)
                            hits.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(hits) >= max_results:
                                hits.append(f"… (capped at {max_results} matches)")
                                return "\n".join(hits)
            except (OSError, UnicodeDecodeError):
                continue
    if not hits:
        return f"No matches for {pattern!r} under {path!r}."
    return "\n".join(hits)


# ── Test runner ────────────────────────────────────────────────────────────

@registry.register(auth_required=True)
def run_tests(target: str = "") -> str:
    """
    Auto-detect the project's test framework and run it.

    Detection (first match wins, within the workspace):
      - pyproject.toml or pytest.ini or tests/conftest.py → pytest
      - package.json with a "test" script        → npm test
      - Cargo.toml                               → cargo test
      - go.mod                                   → go test ./...
      - Makefile with a 'test' target            → make test

    `target` optionally narrows: passed as the final argument to the
    framework (e.g., a test path, module name, or test pattern).

    Returns the test output. Honors the standard process-level sandbox
    (env scrub, cwd=workspace, rlimits).
    """
    workspace_cwd = os.path.abspath(WORKSPACE_DIR)
    os.makedirs(workspace_cwd, exist_ok=True)

    def exists(*names):
        return any(os.path.exists(os.path.join(workspace_cwd, n)) for n in names)

    if exists("pyproject.toml", "pytest.ini", "tests/conftest.py", "setup.cfg"):
        cmd = f"pytest -v {target}".strip()
    elif exists("package.json"):
        cmd = f"npm test -- {target}".strip() if target else "npm test"
    elif exists("Cargo.toml"):
        cmd = f"cargo test {target}".strip()
    elif exists("go.mod"):
        cmd = f"go test ./... {target}".strip()
    elif exists("Makefile") and _makefile_has_target(os.path.join(workspace_cwd, "Makefile"), "test"):
        cmd = f"make test {target}".strip()
    else:
        return (
            "Error: no test framework detected. "
            "Looked for: pyproject.toml / pytest.ini / package.json / "
            "Cargo.toml / go.mod / Makefile-with-test-target. "
            "Either add one or call run_shell with the exact test command."
        )
    return run_shell(cmd, interactive=False)


def _makefile_has_target(path: str, target: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return any(line.lstrip().startswith(f"{target}:") for line in f)
    except OSError:
        return False


# ── Python eval ────────────────────────────────────────────────────────────

@registry.register
def python_eval(expr: str) -> str:
    """
    Evaluate a Python expression in a sandboxed subprocess.

    Use this for math, datetime arithmetic, JSON parsing, list/dict
    manipulation — anything you'd otherwise compute in your head and
    likely get wrong. Returns repr() of the result.

    Sandbox: subprocess with scrubbed env, cwd=workspace, 10s wall timeout,
    no network access by convention (the sandbox doesn't enforce this —
    just don't write expressions that fetch URLs).

    Examples:
      python_eval("2**32 - 1")            → 4294967295
      python_eval("sum(range(100))")      → 4950
      python_eval("[x*2 for x in range(5)]") → [0, 2, 4, 6, 8]
      python_eval("import json; json.dumps({'a': 1})") → '{"a": 1}'
    """
    if not expr or not expr.strip():
        return "Error: empty expression."
    # Wrap so we can exec multi-line snippets too (e.g., the json import case).
    # Use a small driver that prints repr of the LAST expression's result.
    driver = (
        "import sys\n"
        "_src = sys.stdin.read()\n"
        "try:\n"
        "    _ast = compile(_src, '<eval>', 'exec')\n"
        "    _ns = {}\n"
        "    exec(_ast, _ns)\n"
        "    # Find the last expression's value via re-parsing\n"
        "    import ast as _astmod\n"
        "    _tree = _astmod.parse(_src)\n"
        "    if _tree.body and isinstance(_tree.body[-1], _astmod.Expr):\n"
        "        _last = _astmod.Expression(_tree.body[-1].value)\n"
        "        _val = eval(compile(_last, '<eval>', 'eval'), _ns)\n"
        "        sys.stdout.write(repr(_val))\n"
        "    else:\n"
        "        sys.stdout.write('(executed; no expression result)')\n"
        "except Exception as e:\n"
        "    sys.stdout.write(f'Error: {type(e).__name__}: {e}')\n"
    )
    sandbox_env = _build_sandbox_env()
    try:
        result = subprocess.run(
            ["python", "-c", driver],
            input=expr, text=True, capture_output=True,
            timeout=10, cwd=os.path.abspath(WORKSPACE_DIR),
            env=sandbox_env, start_new_session=True,
            preexec_fn=_apply_sandbox_rlimits if os.name != "nt" else None,
        )
    except subprocess.TimeoutExpired:
        return "Error: python_eval timed out after 10s."
    except Exception as e:
        return f"Error launching python_eval: {e}"
    if result.returncode != 0:
        return f"Error: {result.stderr.strip() or 'non-zero exit'}"
    return result.stdout.strip() or "(empty result)"


# ── Git helpers ────────────────────────────────────────────────────────────

def _git_in_workspace(args: list, path: str = ".") -> str:
    """Shared helper: run `git <args>` inside the workspace path. Returns
    output or a clear error if the path isn't a git repo."""
    try:
        cwd = get_workspace_path(path)
    except WorkspacePathError as e:
        return f"Error: {e}"
    # Make sure it's a git repo
    if not os.path.exists(os.path.join(cwd, ".git")):
        # Walk up to find a .git in a parent within workspace
        probe = cwd
        ws_root = os.path.abspath(WORKSPACE_DIR)
        found = False
        while probe and probe.startswith(ws_root):
            if os.path.exists(os.path.join(probe, ".git")):
                cwd = probe
                found = True
                break
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if not found:
            return f"Error: no git repository found at or above {path!r} in workspace."
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd, capture_output=True, text=True,
            timeout=30, env=_build_sandbox_env(),
        )
    except subprocess.TimeoutExpired:
        return f"Error: git {' '.join(args)} timed out."
    except FileNotFoundError:
        return "Error: git not installed."
    output = result.stdout
    if result.stderr:
        output += ("\n" if output else "") + result.stderr
    return output or f"(git {' '.join(args)} produced no output)"


@registry.register
def git_diff(ref: str = "HEAD", path: str = ".") -> str:
    """
    Show git diff against `ref` for a path in the workspace.

    Defaults to comparing the working tree against HEAD. Useful for
    seeing what's changed before committing, or comparing two refs
    with `ref` like "main..feature-branch".
    """
    args = ["diff", "--no-color"]
    if ref:
        args.append(ref)
    return _git_in_workspace(args, path)


@registry.register
def git_log(path: str = ".", limit: int = 10) -> str:
    """
    Show recent commits for a path in the workspace.

    Returns one line per commit: `<short_hash> <date> <author>: <subject>`.
    Default last 10 commits.
    """
    args = ["log", f"-{limit}", "--no-color",
            "--pretty=format:%h %ad %an: %s", "--date=short"]
    return _git_in_workspace(args, path)


@registry.register
def git_blame(file: str, line: Optional[int] = None) -> str:
    """
    Show git blame for a file in the workspace, optionally narrowed
    to a specific line number.

    Each output line is `<short_hash> (<author> <date>) <code>`.
    Useful for "who added this and why" — pair with git_log on the
    returned hash for the commit message.
    """
    # `git blame --no-color` is ambiguous (clashes with --no-color-lines /
    # --no-color-by-age). Default ANSI in output is fine — the caller can
    # strip if needed.
    args = ["blame"]
    if line is not None:
        try:
            n = int(line)
            args.extend(["-L", f"{n},{n}"])
        except (TypeError, ValueError):
            return f"Error: line must be an integer, got {line!r}"
    # The file path is relative to the repo root, so resolve it inside
    # the workspace via get_workspace_path before passing to git blame.
    try:
        full = get_workspace_path(file)
    except WorkspacePathError as e:
        return f"Error: {e}"
    args.append(full)
    return _git_in_workspace(args, ".")