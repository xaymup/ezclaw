"""Tests for the run_shell sandbox hardening.

These tests cover the three isolation properties we promise:
1. Environment scrubbing: secrets and config don't leak into the child.
2. Parent cwd is never mutated, even for interactive commands.
3. The child runs in its own process group with rlimits applied.
"""

import os
import sys
import pytest


# Add project root to sys.path so we can import tools — conftest also does this
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    _build_sandbox_env,
    _ENV_ALLOWLIST,
    _ensure_workspace_self_symlink,
    _run_shell_interactive,
    _skill_filename,
    run_shell,
    WORKSPACE_DIR,
)


# ── _build_sandbox_env ──────────────────────────────────────────────────────

def test_sandbox_env_drops_api_keys(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-1234")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other")
    env = _build_sandbox_env()
    assert "DEEPSEEK_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env


def test_sandbox_env_drops_ollama_config(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_NUM_GPU", "999")
    env = _build_sandbox_env()
    assert "OLLAMA_BASE_URL" not in env
    assert "OLLAMA_NUM_GPU" not in env


def test_sandbox_env_drops_python_pollution(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/private/path")
    monkeypatch.setenv("VIRTUAL_ENV", "/home/user/venv")
    env = _build_sandbox_env()
    assert "PYTHONPATH" not in env
    assert "VIRTUAL_ENV" not in env


def test_sandbox_env_keeps_path(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")
    env = _build_sandbox_env()
    assert env.get("PATH") == "/usr/local/bin:/usr/bin:/bin"


def test_sandbox_env_keeps_locale_and_term(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    env = _build_sandbox_env()
    assert env.get("TERM") == "xterm-256color"
    assert env.get("LANG") == "en_US.UTF-8"


def test_sandbox_env_sets_sentinel():
    env = _build_sandbox_env()
    assert env.get("EZCLAW_SANDBOX") == "1"


def test_allowlist_does_not_include_api_keys():
    """Belt-and-suspenders: even if the allowlist is edited carelessly, it
    must not let any obviously-secret variable through."""
    forbidden = {"DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN"}
    assert _ENV_ALLOWLIST.isdisjoint(forbidden)


# ── run_shell process isolation ─────────────────────────────────────────────

def test_run_shell_does_not_leak_env_to_child(monkeypatch):
    """A `printenv` invocation must not contain the parent's secrets."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-leak-canary")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://leak-canary:11434")
    out = run_shell("printenv")
    assert "sk-leak-canary" not in out
    assert "leak-canary" not in out


def test_run_shell_env_has_sandbox_sentinel():
    out = run_shell("echo \"$EZCLAW_SANDBOX\"")
    assert "1" in out.strip().splitlines()


def test_run_shell_does_not_mutate_parent_cwd(tmp_path, monkeypatch):
    """Even for interactive=False, the parent's cwd must be unchanged."""
    monkeypatch.chdir(tmp_path)
    before = os.getcwd()
    run_shell("ls")
    after = os.getcwd()
    assert before == after


def test_run_shell_runs_in_workspace():
    """The child's cwd matches the workspace dir."""
    out = run_shell("pwd").strip()
    expected_workspace = os.path.abspath(WORKSPACE_DIR)
    assert expected_workspace in out


def test_run_shell_runs_in_new_process_group():
    """The child should be in its own session/pgid so signals stay contained."""
    out = run_shell("ps -o pgid= -p $$").strip()
    # ps -o pgid= prints the child's process group id. The parent (this test
    # runner) has its own pgid; the child's pgid should equal its own pid
    # because start_new_session=True implies setsid().
    child_pgid_lines = [line.strip() for line in out.splitlines() if line.strip().isdigit()]
    assert child_pgid_lines, f"Expected a numeric pgid in: {out!r}"
    child_pgid = int(child_pgid_lines[0])
    # Parent's pgid — definitely not equal if isolation worked.
    parent_pgid = os.getpgid(0)
    assert child_pgid != parent_pgid


# ── Workspace self-symlink (defeats workspace/workspace/ doubling) ─────────

def test_ensure_workspace_self_symlink_creates_link(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _ensure_workspace_self_symlink(str(ws))
    link = ws / WORKSPACE_DIR
    assert link.is_symlink()
    assert os.readlink(str(link)) == "."


def test_ensure_workspace_self_symlink_is_idempotent(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _ensure_workspace_self_symlink(str(ws))
    _ensure_workspace_self_symlink(str(ws))  # second call must be a no-op
    link = ws / WORKSPACE_DIR
    assert link.is_symlink()


def test_ensure_workspace_self_symlink_does_not_clobber_real_dir(tmp_path):
    """If a real workspace/workspace/ directory already exists (from past
    mistakes), we must not touch it — the user's data is in there."""
    ws = tmp_path / "ws"
    ws.mkdir()
    real_subdir = ws / WORKSPACE_DIR
    real_subdir.mkdir()
    (real_subdir / "data.txt").write_text("user data")

    _ensure_workspace_self_symlink(str(ws))
    assert real_subdir.is_dir()
    assert not real_subdir.is_symlink()
    assert (real_subdir / "data.txt").read_text() == "user data"


def test_workspace_workspace_path_resolves_through_symlink_to_workspace():
    """End-to-end: agent runs `touch workspace/canary.txt` from inside the
    workspace. The file should land at workspace/canary.txt (one level),
    NOT at workspace/workspace/canary.txt (two levels)."""
    canary = "_double_workspace_canary.txt"
    workspace_abs = os.path.abspath(WORKSPACE_DIR)
    canary_at_root = os.path.join(workspace_abs, canary)
    canary_at_nested = os.path.join(workspace_abs, WORKSPACE_DIR, canary)
    # Clean up any stragglers from previous runs
    for p in (canary_at_root, canary_at_nested):
        if os.path.lexists(p):
            os.remove(p)

    # Agent's mistake: extra workspace/ prefix in the shell command
    run_shell(f"touch {WORKSPACE_DIR}/{canary}")

    # Whether resolved via the symlink or accidentally created two-deep,
    # the file should be readable at the one-level path.
    assert os.path.exists(canary_at_root), (
        f"Expected file at {canary_at_root} after touch via workspace/ prefix"
    )

    # Cleanup
    if os.path.lexists(canary_at_root):
        os.remove(canary_at_root)


# ── Interactive shell: structured output for the agent ────────────────────

@pytest.mark.skipif(os.name == "nt", reason="pty is POSIX-only")
def test_interactive_returns_structured_header_with_output():
    """The agent's next turn sees a clear summary: command + exit code +
    duration + captured output. Used to be just the raw stream which
    made empty output indistinguishable from a failed command."""
    workspace = os.path.abspath(WORKSPACE_DIR)
    os.makedirs(workspace, exist_ok=True)
    env = _build_sandbox_env()
    out = _run_shell_interactive(
        "echo 'hello' && echo 'world'",
        workspace_cwd=workspace,
        sandbox_env=env,
    )
    assert "[Interactive shell session" in out
    assert "succeeded" in out
    assert "Exit code: 0" in out
    assert "Duration:" in out
    assert "hello" in out
    assert "world" in out


@pytest.mark.skipif(os.name == "nt", reason="pty is POSIX-only")
def test_interactive_explains_empty_output_case():
    """Commands like `true` produce zero stdout. The result must STILL
    tell the agent it succeeded, not return a bare 'Interactive command
    completed' that hides the exit code."""
    workspace = os.path.abspath(WORKSPACE_DIR)
    os.makedirs(workspace, exist_ok=True)
    env = _build_sandbox_env()
    out = _run_shell_interactive("true", workspace_cwd=workspace, sandbox_env=env)
    assert "Exit code: 0" in out
    assert "succeeded" in out
    assert "no terminal output captured" in out


@pytest.mark.skipif(os.name == "nt", reason="pty is POSIX-only")
def test_interactive_callback_path_streams_output():
    """When on_output is provided, output goes to the callback instead of
    sys.stdout. The TUI uses this to stream subprocess bytes into a live
    tool panel inside the chat."""
    workspace = os.path.abspath(WORKSPACE_DIR)
    os.makedirs(workspace, exist_ok=True)
    chunks = []
    _run_shell_interactive(
        "echo 'streamed via callback'",
        workspace_cwd=workspace,
        sandbox_env=_build_sandbox_env(),
        on_output=chunks.append,
    )
    joined = b"".join(chunks).decode("utf-8", errors="ignore")
    assert "streamed via callback" in joined


@pytest.mark.skipif(os.name == "nt", reason="pty is POSIX-only")
def test_interactive_input_provider_feeds_subprocess():
    """A subprocess that reads a line from stdin should receive the bytes
    returned by input_provider. The TUI uses this to route user chat
    input to the subprocess while a session is active."""
    workspace = os.path.abspath(WORKSPACE_DIR)
    os.makedirs(workspace, exist_ok=True)
    fed = [b"hello-from-test\n"]

    def provider():
        return fed.pop(0) if fed else None

    out = _run_shell_interactive(
        "read line && echo \"got: $line\"",
        workspace_cwd=workspace,
        sandbox_env=_build_sandbox_env(),
        on_output=lambda _b: None,
        input_provider=provider,
    )
    assert "got: hello-from-test" in out


@pytest.mark.skipif(os.name == "nt", reason="pty is POSIX-only")
def test_interactive_surfaces_nonzero_exit_code():
    """A failing command must be marked failed in the header so the agent
    knows to retry or escalate, not silently treat it as success."""
    workspace = os.path.abspath(WORKSPACE_DIR)
    os.makedirs(workspace, exist_ok=True)
    env = _build_sandbox_env()
    out = _run_shell_interactive("false", workspace_cwd=workspace, sandbox_env=env)
    assert "Exit code: 1" in out
    assert "failed" in out


# ── Skill path-traversal protection ────────────────────────────────────────

def test_skill_filename_strips_traversal():
    """A malicious skill name like `../../etc/passwd` must be sanitized
    into a flat filename that can never escape SKILLS_DIR."""
    assert _skill_filename("../../etc/passwd") == "etc_passwd.md"
    assert _skill_filename("/abs/path") == "abs_path.md"
    assert _skill_filename("normal name") == "normal_name.md"
    assert _skill_filename("") == "unnamed.md"
    # Path separators in the result would let os.path.join place the file
    # outside SKILLS_DIR — make sure none survive sanitization.
    for bad in ("a/b", "a\\b", "../x", "/etc/passwd", "x/../y"):
        out = _skill_filename(bad)
        assert "/" not in out and "\\" not in out and ".." not in out


# ── Resource caps (POSIX rlimits) ──────────────────────────────────────────

@pytest.mark.skipif(os.name == "nt", reason="rlimits are POSIX-only")
def test_run_shell_caps_file_size():
    """RLIMIT_FSIZE should prevent the child from writing a >100 MB file."""
    # Write 200 MB; bash should die or truncate.
    out = run_shell(
        "dd if=/dev/zero of=big.bin bs=1M count=200 2>&1; ls -la big.bin 2>&1; rm -f big.bin"
    )
    # We don't assert on the exact error message — different shells emit
    # different things. We assert that big.bin (if it exists at the ls stage)
    # is below the rlimit, OR dd reported an error.
    # Quick sanity: SIGXFSZ or "File size limit exceeded" or "Disk quota" or
    # the file simply doesn't get past 100 MB.
    if "big.bin" in out:
        # If the listing showed up, ensure no line claims 200M
        assert "200" not in out or "exceeded" in out.lower() or "limit" in out.lower()
