"""Tests for the new intelligence-amplifying tools: code_outline,
apply_diff, grep_codebase, run_tests, python_eval, git_*."""

import os
import sys
import subprocess
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools


# ── code_outline ────────────────────────────────────────────────────────────

@pytest.fixture
def py_file(tmp_path, monkeypatch):
    """Create a Python file inside the workspace dir so get_workspace_path resolves."""
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    src = textwrap.dedent('''
        """Sample module."""
        import os
        from typing import List

        CONST = 42

        class Foo:
            def method(self, x):
                return x

            class Inner:
                pass

        async def async_func(arg):
            await something(arg)

        def helper():
            pass
    ''').strip()
    fpath = tmp_path / "sample.py"
    fpath.write_text(src)
    return fpath


def test_code_outline_python_extracts_symbols(py_file, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(py_file.parent))
    out = tools.code_outline("sample.py")
    assert "import os" in out
    assert "from typing import List" in out
    assert "class Foo" in out
    assert "def method(self, x)" in out
    assert "class Inner" in out
    assert "async def async_func(arg)" in out
    assert "def helper()" in out
    assert "CONST = …" in out


def test_code_outline_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    out = tools.code_outline("nope.py")
    assert "Error" in out and "does not exist" in out


def test_code_outline_generic_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "thing.js").write_text(
        "export function hello() {}\n"
        "export class Widget {}\n"
        "export const X = 1\n"
    )
    out = tools.code_outline("thing.js")
    assert "function hello" in out
    assert "class Widget" in out
    assert "binding X" in out


# ── apply_diff ──────────────────────────────────────────────────────────────

def test_apply_diff_simple_addition(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    fpath = tmp_path / "foo.txt"
    fpath.write_text("line 1\nline 2\nline 3\n")
    diff = textwrap.dedent("""\
        @@ -1,3 +1,4 @@
         line 1
         line 2
        +inserted
         line 3
    """)
    out = tools.apply_diff("foo.txt", diff)
    assert "Applied 1 hunk" in out
    assert fpath.read_text() == "line 1\nline 2\ninserted\nline 3\n"


def test_apply_diff_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    fpath = tmp_path / "foo.txt"
    fpath.write_text("aaa\nbbb\nccc\n")
    diff = textwrap.dedent("""\
        @@ -1,3 +1,3 @@
         aaa
        -bbb
        +BBB
         ccc
    """)
    tools.apply_diff("foo.txt", diff)
    assert fpath.read_text() == "aaa\nBBB\nccc\n"


def test_apply_diff_rejects_when_context_doesnt_match(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    fpath = tmp_path / "foo.txt"
    fpath.write_text("real content\n")
    bad_diff = textwrap.dedent("""\
        @@ -1,1 +1,2 @@
         expected something else
        +added
    """)
    out = tools.apply_diff("foo.txt", bad_diff)
    assert "Error" in out and "hunk" in out


def test_apply_diff_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    out = tools.apply_diff("nonexistent.txt", "@@ -1 +1 @@\n-a\n+b\n")
    assert "Error" in out and "does not exist" in out


# ── grep_codebase ───────────────────────────────────────────────────────────

def test_grep_codebase_finds_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "a.py").write_text("def foo():\n    return 'bar'\n")
    (tmp_path / "b.py").write_text("class Bar:\n    pass\n")
    (tmp_path / "c.txt").write_text("the word bar appears here\n")
    out = tools.grep_codebase("bar", ".")
    assert "a.py:" in out
    assert "c.txt:" in out


def test_grep_codebase_skips_binary_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "should_skip.py").write_text("secret")
    (tmp_path / "ignored.pyc").write_bytes(b"\x00\x01secret\x02")
    (tmp_path / "visible.py").write_text("findable secret\n")
    out = tools.grep_codebase("secret", ".")
    assert "visible.py" in out
    assert "should_skip" not in out
    assert "ignored.pyc" not in out


def test_grep_codebase_invalid_regex(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    out = tools.grep_codebase("[unclosed", ".")
    assert "Error" in out and "regex" in out


def test_grep_codebase_no_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "f.txt").write_text("hello world\n")
    out = tools.grep_codebase("zzzzz", ".")
    assert "No matches" in out


# ── python_eval ─────────────────────────────────────────────────────────────

def test_python_eval_arithmetic(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    assert tools.python_eval("2 ** 32 - 1") == "4294967295"


def test_python_eval_collections(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    assert tools.python_eval("[x*2 for x in range(5)]") == "[0, 2, 4, 6, 8]"


def test_python_eval_handles_exception_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    out = tools.python_eval("1 / 0")
    assert "Error" in out and "ZeroDivisionError" in out


def test_python_eval_empty_input(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    out = tools.python_eval("")
    assert "Error" in out and "empty" in out.lower()


def test_python_eval_multistatement_returns_last_expr(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    out = tools.python_eval("x = 10\ny = 20\nx + y")
    assert out == "30"


# ── run_tests ───────────────────────────────────────────────────────────────

def test_run_tests_no_framework(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    out = tools.run_tests()
    assert "Error" in out and "no test framework" in out.lower()


def test_run_tests_detects_makefile_target(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "Makefile").write_text("test:\n\t@echo ran make test\n")
    out = tools.run_tests()
    assert "ran make test" in out


def test_run_tests_makefile_without_test_target(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    (tmp_path / "Makefile").write_text("build:\n\t@echo built\n")
    out = tools.run_tests()
    assert "Error" in out  # has Makefile but no test target → fall through


# ── git tools ───────────────────────────────────────────────────────────────

def _init_repo(path):
    """Create a tiny git repo with two commits for testing."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)
    (path / "hello.txt").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=path, check=True)
    (path / "hello.txt").write_text("hello\nworld\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=path, check=True)


def test_git_log_returns_recent_commits(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    _init_repo(tmp_path)
    out = tools.git_log(".")
    assert "first" in out
    assert "second" in out


def test_git_diff_shows_working_tree_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    _init_repo(tmp_path)
    (tmp_path / "hello.txt").write_text("hello\nworld\nNEW\n")
    out = tools.git_diff("HEAD")
    assert "+NEW" in out


def test_git_blame_reports_author(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    _init_repo(tmp_path)
    out = tools.git_blame("hello.txt", line=1)
    assert "Tester" in out


def test_git_log_no_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", str(tmp_path))
    out = tools.git_log(".")
    assert "Error" in out and "no git repository" in out
