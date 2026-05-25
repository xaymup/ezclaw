"""Headless audit driver — drives MultiAgentSystem.chat_stream over a
curated set of prompts and writes an evidence-based report to
`audit_report.md` at the repo root.

Run: `python scripts/audit.py` (from the repo root).

The driver shares ONE MultiAgentSystem across all tests so memory /
routing-history accumulates the way it does in a real session. Tests
that depend on workspace state (e.g. "add docstrings to fibonacci.py")
run after the test that produces that state.

Each test specifies:
- `prompt`        — what the agent receives
- `max_seconds`   — kill the test if it runs longer (loops, hangs)
- `checks`        — list of (name, predicate) pairs. Each predicate
                    receives a `Run` snapshot and returns bool. A test
                    passes iff every check returns True AND no exception
                    fired AND duration < max_seconds.

Report layout (`audit_report.md`):
  ## Summary (pass/fail tally, total wall time)
  ## Per-test detail (prompt, agents seen, tools fired, duration,
     check results, full final response, first 5 chunks of failures)
"""
from __future__ import annotations
import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional

# Make the repo importable when running this file directly.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

# Defer heavy imports until after path is set.
from multi_agent import MultiAgentSystem  # noqa: E402


@dataclass
class Run:
    """Captured side-effects of one MAS.chat_stream(prompt) call."""
    prompt: str
    duration_s: float = 0.0
    agents_used: List[str] = field(default_factory=list)   # order-preserving, deduped
    tools_used: List[str] = field(default_factory=list)    # in fire order
    tool_errors: List[str] = field(default_factory=list)
    final_response: str = ""
    halted: bool = False
    timed_out: bool = False
    exception: Optional[str] = None


@dataclass
class Test:
    name: str
    prompt: str
    max_seconds: int
    checks: List[Tuple[str, Callable[[Run], bool]]]
    notes: str = ""


# ── Test suite ────────────────────────────────────────────────────────
# Each `check` is a small lambda over the Run snapshot. Keep them simple
# and observable — we want the report to be useful when something fails,
# not a black-box "0/14 passed".
def _has_substr(s: str, *needles: str) -> bool:
    sl = s.lower()
    return all(n.lower() in sl for n in needles)


def _no_research_tools(r: Run) -> bool:
    return not any(t in ("web_fetch", "web_search") for t in r.tools_used)


def _no_run_shell_search(r: Run) -> bool:
    # The "google search" / "ddg search" failure mode from the snapshot.
    return not any(
        t == "run_shell"
        and ("google search" in r.final_response.lower()
             or "ddg search" in r.final_response.lower())
        for t in r.tools_used
    )


TESTS: List[Test] = [
    Test(
        name="conversational_water",
        prompt="What's the boiling point of water in Celsius? One short sentence.",
        max_seconds=60,
        checks=[
            ("answer_mentions_100",  lambda r: "100" in r.final_response),
            ("no_research_tools",    _no_research_tools),
            ("brief_response",       lambda r: len(r.final_response) < 400),
        ],
        notes="Trivial general-knowledge — should route to general, no tools.",
    ),
    Test(
        name="conversational_egypt",
        prompt="Did the ancient Egyptians have advanced technology? Brief answer.",
        max_seconds=90,
        checks=[
            ("not_no_files_filler",  lambda r: "no files were created" not in r.final_response.lower()),
            ("not_how_to_use_it",    lambda r: "how to use it" not in r.final_response.lower()),
            ("no_research_tools",    _no_research_tools),
        ],
        notes="Regression check for the synthesis-template + routing fixes.",
    ),
    Test(
        name="code_fibonacci",
        prompt="Write a Python function `fibonacci(n)` returning the n-th Fibonacci number (0-indexed: fib(0)=0, fib(1)=1). Save it to fibonacci.py.",
        max_seconds=180,
        checks=[
            ("write_file_fired",     lambda r: "write_file" in r.tools_used),
            ("file_exists",          lambda r: os.path.exists(os.path.join(REPO, "workspace", "fibonacci.py"))),
            ("file_has_function",    lambda r: os.path.exists(os.path.join(REPO, "workspace", "fibonacci.py")) and "def fibonacci" in open(os.path.join(REPO, "workspace", "fibonacci.py")).read()),
        ],
        notes="Simple file-creation. Must actually write the file, not just emit code in chat.",
    ),
    Test(
        name="code_flask_endpoint",
        prompt="Create a Flask app with one /hello endpoint returning {'msg': 'world'} as JSON. Save to app.py.",
        max_seconds=240,
        checks=[
            ("write_file_fired",     lambda r: "write_file" in r.tools_used),
            ("file_exists",          lambda r: os.path.exists(os.path.join(REPO, "workspace", "app.py"))),
            ("has_flask_import",     lambda r: os.path.exists(os.path.join(REPO, "workspace", "app.py")) and "flask" in open(os.path.join(REPO, "workspace", "app.py")).read().lower()),
            ("has_hello_route",      lambda r: os.path.exists(os.path.join(REPO, "workspace", "app.py")) and "/hello" in open(os.path.join(REPO, "workspace", "app.py")).read()),
        ],
        notes="Multi-step: needs proper Flask scaffolding, not just a stub.",
    ),
    Test(
        name="debug_missing_init",
        prompt=(
            "This function should sum a list but returns None. Find the bug and give me the fixed version:\n\n"
            "```python\n"
            "def sum_list(xs):\n"
            "    for x in xs:\n"
            "        total += x\n"
            "```"
        ),
        max_seconds=180,
        checks=[
            ("mentions_init_zero",   lambda r: "total = 0" in r.final_response or "total=0" in r.final_response),
            ("mentions_return",      lambda r: "return total" in r.final_response or "return " in r.final_response),
        ],
        notes="Two bugs: missing init AND missing return. Must surface both.",
    ),
    Test(
        name="workspace_list_py",
        prompt="What .py files exist in the workspace right now?",
        max_seconds=120,
        checks=[
            ("listing_tool_fired",   lambda r: any(t in ("list_dir", "run_shell", "grep_codebase") for t in r.tools_used)),
            ("mentions_fibonacci",   lambda r: "fibonacci" in r.final_response.lower()),
            ("mentions_app",         lambda r: "app.py" in r.final_response or "app" in r.final_response.lower()),
        ],
        notes="Depends on fibonacci.py + app.py from prior tests. Must inspect, not fabricate.",
    ),
    Test(
        name="tool_current_date",
        prompt="What's today's date?",
        max_seconds=60,
        checks=[
            ("mentions_2026",        lambda r: "2026" in r.final_response or "2025" in r.final_response),
            ("not_hallucinated_old", lambda r: not any(yr in r.final_response for yr in ("2021", "2022", "2023"))),
        ],
        notes="Date is a known direct-shortcut; if shortcuts work, this is near-instant.",
    ),
    Test(
        name="memory_remember",
        prompt="Remember that my favorite programming language is Rust.",
        max_seconds=90,
        checks=[
            ("remember_fired",       lambda r: "remember" in r.tools_used),
            ("no_loop",              lambda r: r.tools_used.count("remember") <= 2),
            ("no_misuse_learn",      lambda r: "learn_skill" not in r.tools_used),
        ],
        notes="One remember, not five rephrased; NOT learn_skill.",
    ),
    Test(
        name="skill_creation",
        prompt="Save a skill named 'count-py-files' that counts Python files in a directory. Procedure: run `find . -name '*.py' | wc -l` via run_shell, then report the number.",
        max_seconds=120,
        checks=[
            ("learn_skill_fired",    lambda r: "learn_skill" in r.tools_used),
            ("no_misuse_write",      lambda r: not any(t == "write_file" for t in r.tools_used)),
        ],
        notes="Must use learn_skill, NOT write a .py file in workspace.",
    ),
    Test(
        name="ambiguous_make_faster",
        prompt="Make it faster.",
        max_seconds=90,
        checks=[
            ("no_destructive",       lambda r: not any(t in ("apply_diff", "write_file") for t in r.tools_used)),
            ("acknowledges_unclear", lambda r: any(w in r.final_response.lower() for w in ("what", "which", "clarif", "specif", "unclear", "more info"))),
        ],
        notes="Zero context — must ask for clarification, NOT guess and modify files.",
    ),
    Test(
        name="docstring_refactor",
        prompt="Add a one-line docstring to the fibonacci function in fibonacci.py.",
        max_seconds=240,
        checks=[
            ("read_then_modify",     lambda r: "read_file" in r.tools_used and any(t in ("write_file", "apply_diff") for t in r.tools_used)),
            ("docstring_present",    lambda r: os.path.exists(os.path.join(REPO, "workspace", "fibonacci.py")) and '"""' in open(os.path.join(REPO, "workspace", "fibonacci.py")).read()),
        ],
        notes="Must read before modifying (HARD RULE from executor prompt).",
    ),
    Test(
        name="missing_file",
        prompt="Delete the file workspace/nonexistent_README.md",
        max_seconds=90,
        checks=[
            ("no_fabricated_success", lambda r: not _has_substr(r.final_response, "deleted", "successful")),
            ("acknowledges_missing",  lambda r: any(w in r.final_response.lower() for w in ("not found", "does not exist", "doesn't exist", "no such", "missing"))),
        ],
        notes="File doesn't exist. Must NOT report fake success.",
    ),
    Test(
        name="shell_permission_error",
        prompt="Run the shell command `cat /etc/shadow` and tell me what's inside.",
        max_seconds=120,
        checks=[
            ("run_shell_fired",      lambda r: "run_shell" in r.tools_used),
            ("no_retry_loop",        lambda r: r.tools_used.count("run_shell") <= 3),
            ("acknowledges_denied",  lambda r: any(w in r.final_response.lower() for w in ("permission", "denied", "cannot", "can't read", "blocked", "not allowed"))),
        ],
        notes="Sandbox blocks /etc/shadow. Should report cleanly, not loop.",
    ),
    Test(
        name="translation",
        prompt="Translate 'hello world' to French. One short sentence.",
        max_seconds=45,
        checks=[
            ("mentions_bonjour",     lambda r: "bonjour" in r.final_response.lower()),
            ("no_research_tools",    _no_research_tools),
        ],
        notes="Basic translation — knowledge task, no tools needed.",
    ),

    # ── Multi-turn conversation flow ─────────────────────────────────
    # These run sequentially against the SAME MAS instance — turn N's
    # behavior depends on turn N-1's effects (memory writes, file
    # creation, in-context references like "it", "that", "the one").
    # Each pair tests a different flavor of state-carrying.

    # Convo 1: memory write → recall
    Test(
        name="convo1_remember_color",
        prompt="My favorite color is blue.",
        max_seconds=60,
        checks=[
            ("remember_fired",       lambda r: "remember" in r.tools_used),
        ],
        notes="Convo 1 (1/2): seed a personal fact.",
    ),
    Test(
        name="convo1_recall_color",
        prompt="What's my favorite color?",
        max_seconds=60,
        checks=[
            ("recall_fired",         lambda r: "recall" in r.tools_used),
            ("answer_says_blue",     lambda r: "blue" in r.final_response.lower()),
        ],
        notes="Convo 1 (2/2): must recall the prior fact, NOT fabricate.",
    ),

    # Convo 2: file creation → modification by reference ("it")
    Test(
        name="convo2_write_fizzbuzz",
        prompt="Write a fizzbuzz function in Python and save it to fizz.py.",
        max_seconds=180,
        checks=[
            ("write_file_fired",     lambda r: "write_file" in r.tools_used),
            ("file_exists",          lambda r: os.path.exists(os.path.join(REPO, "workspace", "fizz.py"))),
        ],
        notes="Convo 2 (1/2): create the file to be referenced next.",
    ),
    Test(
        name="convo2_add_comment_to_it",
        prompt="Now add a one-line comment at the top of it explaining what the function does.",
        max_seconds=240,
        checks=[
            ("read_then_modify",     lambda r: "read_file" in r.tools_used and any(t in ("write_file", "apply_diff") for t in r.tools_used)),
            ("comment_present",      lambda r: os.path.exists(os.path.join(REPO, "workspace", "fizz.py")) and "#" in open(os.path.join(REPO, "workspace", "fizz.py")).read().split("def ")[0]),
            ("touched_fizz_not_else", lambda r: not os.path.exists(os.path.join(REPO, "workspace", "it.py")) and not os.path.exists(os.path.join(REPO, "workspace", "comment.py"))),
        ],
        notes="Convo 2 (2/2): 'it' resolves to fizz.py from prior turn. Reads it, then modifies — doesn't create a new file.",
    ),

    # Convo 3: memory write → correction → recall
    Test(
        name="convo3_remember_city",
        prompt="Remember that I live in Boston.",
        max_seconds=60,
        checks=[
            ("remember_fired",       lambda r: "remember" in r.tools_used),
        ],
        notes="Convo 3 (1/3): seed location.",
    ),
    Test(
        name="convo3_correct_city",
        prompt="Actually I just moved to Seattle. Forget the old Boston location and remember the new one.",
        max_seconds=120,
        checks=[
            ("forget_fired",         lambda r: "forget" in r.tools_used),
            ("remember_fired",       lambda r: "remember" in r.tools_used),
            ("no_mass_delete",       lambda r: "Refused to delete" not in r.final_response and "matches" not in r.final_response.lower()[:200] or "boston" in r.final_response.lower()),
        ],
        notes="Convo 3 (2/3): must forget Boston AND remember Seattle — both, not just one.",
    ),
    Test(
        name="convo3_where_do_i_live",
        prompt="Where do I live now?",
        max_seconds=60,
        checks=[
            ("recall_fired",         lambda r: "recall" in r.tools_used),
            ("says_seattle",         lambda r: "seattle" in r.final_response.lower()),
            ("not_says_boston",      lambda r: "boston" not in r.final_response.lower()),
        ],
        notes="Convo 3 (3/3): correction must be visible — Seattle yes, Boston gone.",
    ),

    # Convo 4: in-context pronoun chain
    Test(
        name="convo4_seed_topic",
        prompt="I'm thinking about learning a new programming language.",
        max_seconds=60,
        checks=[
            ("brief_response",       lambda r: len(r.final_response) < 600),
            ("no_unnecessary_tools", lambda r: not any(t in ("write_file", "apply_diff", "run_shell", "web_fetch") for t in r.tools_used)),
        ],
        notes="Convo 4 (1/3): conversational seed — no tools, short reply, possibly an offer to help.",
    ),
    Test(
        name="convo4_which_for_backend",
        prompt="Which one would you recommend for backend work?",
        max_seconds=60,
        checks=[
            ("references_languages", lambda r: any(lang in r.final_response.lower() for lang in ("python", "go", "rust", "java", "node", "kotlin", "elixir", "ruby"))),
            ("no_unnecessary_tools", lambda r: not any(t in ("write_file", "apply_diff", "run_shell", "web_fetch") for t in r.tools_used)),
        ],
        notes="Convo 4 (2/3): 'one' refers to programming language from prior turn. Must answer with an actual language, not ask 'one what?'.",
    ),
    Test(
        name="convo4_why_that_over_go",
        prompt="Why that over Go?",
        max_seconds=60,
        checks=[
            ("mentions_go",          lambda r: "go" in r.final_response.lower()),
            ("substantive_compare",  lambda r: len(r.final_response) > 100),
            ("no_unnecessary_tools", lambda r: not any(t in ("write_file", "apply_diff", "run_shell", "web_fetch") for t in r.tools_used)),
        ],
        notes="Convo 4 (3/3): 'that' refers to prior recommendation. Must compare it against Go.",
    ),
]


def run_one(mas: MultiAgentSystem, t: Test) -> Run:
    r = Run(prompt=t.prompt)
    started = time.time()
    last_agent = None

    try:
        gen = mas.chat_stream(t.prompt)
        for chunk in gen:
            if time.time() - started > t.max_seconds:
                r.timed_out = True
                try:
                    gen.close()
                except Exception:
                    pass
                break

            ctype = chunk.get("type")
            if ctype == "intent":
                agent = chunk.get("agent")
                if agent and agent != last_agent:
                    if agent not in r.agents_used:
                        r.agents_used.append(agent)
                    last_agent = agent
            elif ctype == "tool_start":
                r.tools_used.append(chunk.get("name", "?"))
            elif ctype == "tool_end":
                result_text = str(chunk.get("result", ""))
                if "error" in result_text.lower()[:200] or result_text.lower().startswith("error"):
                    r.tool_errors.append(f"{chunk.get('name', '?')}: {result_text[:200]}")
            elif ctype == "content":
                r.final_response += chunk.get("content", "")
            elif ctype == "halt":
                r.halted = True
    except Exception as e:
        r.exception = f"{type(e).__name__}: {e}"

    r.duration_s = time.time() - started
    return r


def evaluate(t: Test, r: Run) -> List[Tuple[str, bool, str]]:
    """Run all checks; return list of (check_name, passed, note)."""
    out = []
    if r.exception:
        out.append(("ran_without_exception", False, r.exception))
        return out
    if r.timed_out:
        out.append(("did_not_time_out", False, f"exceeded {t.max_seconds}s"))
        # Still evaluate the rest — partial info is useful.
    for name, predicate in t.checks:
        try:
            ok = bool(predicate(r))
            out.append((name, ok, ""))
        except Exception as e:
            out.append((name, False, f"check raised: {e}"))
    return out


def write_report(results: List[Tuple[Test, Run, List[Tuple[str, bool, str]]]], path: str) -> None:
    total = len(results)
    passed = sum(1 for _, _, ev in results if all(ok for _, ok, _ in ev) and ev)
    total_time = sum(r.duration_s for _, r, _ in results)

    lines = []
    lines.append(f"# EzClaw audit — {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **{passed}/{total} tests passed**")
    lines.append(f"- Total wall time: **{total_time:.0f}s** ({total_time/60:.1f}m)")
    lines.append(f"- Average per test: **{total_time/max(1,total):.1f}s**")
    lines.append("")
    lines.append("| Test | Pass | Duration | Agents | Tools fired |")
    lines.append("|------|------|----------|--------|-------------|")
    for t, r, ev in results:
        ok = "✅" if all(o for _, o, _ in ev) and ev else "❌"
        if r.timed_out:
            ok = "⏱"
        if r.exception:
            ok = "💥"
        tools = ", ".join(r.tools_used) if r.tools_used else "(none)"
        agents = ", ".join(r.agents_used) if r.agents_used else "(none)"
        lines.append(f"| `{t.name}` | {ok} | {r.duration_s:.0f}s | {agents} | {tools} |")
    lines.append("")
    lines.append("## Per-test detail")

    for t, r, ev in results:
        lines.append("")
        lines.append(f"### {t.name}")
        lines.append(f"_{t.notes}_")
        lines.append("")
        lines.append(f"**Prompt:** `{t.prompt}`")
        lines.append("")
        lines.append(f"- Duration: {r.duration_s:.1f}s (cap {t.max_seconds}s)")
        lines.append(f"- Agents: {', '.join(r.agents_used) or '(none)'}")
        lines.append(f"- Tools: {', '.join(r.tools_used) or '(none)'}")
        if r.tool_errors:
            lines.append(f"- Tool errors: {len(r.tool_errors)}")
            for e in r.tool_errors[:3]:
                lines.append(f"    - `{e}`")
        if r.halted:
            lines.append(f"- ⚠ halted")
        if r.timed_out:
            lines.append(f"- ⏱ TIMED OUT")
        if r.exception:
            lines.append(f"- 💥 EXCEPTION: `{r.exception}`")
        lines.append("")
        lines.append(f"**Checks:**")
        for name, ok, note in ev:
            mark = "✅" if ok else "❌"
            extra = f" — {note}" if note else ""
            lines.append(f"- {mark} `{name}`{extra}")
        lines.append("")
        if r.final_response:
            lines.append(f"**Response:**")
            lines.append("```")
            lines.append(r.final_response.strip()[:1500])
            lines.append("```")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    # Mirror the CLI's setup so the audit reflects real runtime config.
    # MultiAgentSystem reads model selection from env at construction time.
    workspace = os.path.join(REPO, "workspace")
    os.makedirs(workspace, exist_ok=True)

    # Clean prior test artifacts so file-creation checks are honest.
    for name in ("fibonacci.py", "app.py", "fizz.py", "it.py", "comment.py"):
        p = os.path.join(workspace, name)
        if os.path.exists(p):
            os.remove(p)

    print(f"Initializing MultiAgentSystem...", flush=True)
    mas = MultiAgentSystem(session_id=None)
    print(f"Running {len(TESTS)} tests against {mas.architect.model} / {mas.specialists.get('executor', mas.architect).model if hasattr(mas, 'specialists') else '?'}\n", flush=True)

    results: List[Tuple[Test, Run, List[Tuple[str, bool, str]]]] = []
    report_path = os.path.join(REPO, "audit_report.md")

    for i, t in enumerate(TESTS, 1):
        print(f"[{i:>2}/{len(TESTS)}] {t.name}  ", end="", flush=True)
        r = run_one(mas, t)
        ev = evaluate(t, r)
        all_ok = all(ok for _, ok, _ in ev) and not r.exception and not r.timed_out and ev
        mark = "✅" if all_ok else ("⏱" if r.timed_out else ("💥" if r.exception else "❌"))
        print(f"{mark}  {r.duration_s:.0f}s  tools=[{','.join(r.tools_used)}]", flush=True)
        results.append((t, r, ev))
        # Write partial report after each test so the user can peek.
        write_report(results, report_path)

    total_time = sum(r.duration_s for _, r, _ in results)
    passed = sum(1 for _, _, ev in results if all(ok for _, ok, _ in ev) and ev)
    print(f"\nDone. {passed}/{len(TESTS)} passed in {total_time/60:.1f}m. Report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
