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
