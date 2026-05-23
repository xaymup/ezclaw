import ollama
import os
import json
import re
from typing import List, Dict, Any, Optional, Iterator
from memory import Database
import pickle
from tools import registry, create_memory_tools, MUTATING_TOOLS, create_action_tracking_tools, set_session_context
from action_tracking import classify_outcome, extract_why, summarize_action
from agent import load_skills, match_skills, format_skills_block
from embed import embed, cosine_similarity, classify_by_similarity
from model_client import build_architect_client, build_agent_client, extract_json
from dotenv import load_dotenv

load_dotenv()


# Phrases that, when present in recent user messages, indicate the user
# was correcting or teaching the assistant. Used as a cheap precheck
# before we spend an architect LLM call on skill drafting.
_LEARNING_CUES = (
    "no, ", "no.", "nope", "actually", "instead", "the right way",
    "you should", "should use", "should be", "use ", "try ",
    "wrong", "incorrect", "doesn't work", "didn't work",
    "fix it", "correct way", "not like that", "rather than",
    "the correct", "what i meant", "i mean", "to be clear",
    "the proper", "the URL is", "the endpoint is", "the command is",
)


def _has_learning_signal(recent_history: str) -> bool:
    """Cheap keyword precheck: did the user's recent messages contain
    corrective/instructive language? Only when this returns True do
    we pay for the architect call that drafts the skill."""
    if not recent_history:
        return False
    text = recent_history.lower()
    return any(cue in text for cue in _LEARNING_CUES)


PREFLIGHT_KIND_THRESHOLD = 3  # >= this many distinct tool kinds → architect

_KNOWN_TOOL_NAMES = frozenset({
    "apply_diff", "write_file", "run_shell", "read_file", "list_dir",
    "grep_codebase", "web_search", "web_fetch", "recall", "code_outline",
    "git_diff", "git_log", "git_blame", "run_tests", "python_eval",
    "schedule_task", "unschedule_task", "ask_user", "current_datetime",
    "get_system_info",
})


def _parse_tool_lines(text: str) -> set:
    """Extract known tool names from a newline-separated LLM response.

    Lowercases, strips leading list markers (`-`, `*`, digits, `.`, `)`,
    `(`, `[`, `]`), strips whitespace, filters to _KNOWN_TOOL_NAMES,
    returns the unique set. Tolerates simple list formats. Rejects prose
    (lines whose normalized form is not exactly a known tool name).
    """
    result: set = set()
    if not text:
        return result
    for raw_line in text.splitlines():
        token = raw_line.strip().lower()
        if not token:
            continue
        # Strip leading list markers / punctuation.
        token = token.lstrip("-*().[] \t0123456789")
        # Strip trailing punctuation.
        token = token.rstrip(" \t.,;:)]")
        if token in _KNOWN_TOOL_NAMES:
            result.add(token)
    return result


OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

AGENT_DEFS = {
    "executor": {
        "model": os.getenv("OLLAMA_MODEL", "qwen3:14b"),
        "system_prompt": """You are EzClaw's **Executor** — you receive a numbered plan and execute it step by step using tools.

## Core Rules
- **Verification-Driven Autonomy (Test-First)**: For every coding task or bug fix:
    1. **Reproduce**: Create or identify a test/script that fails due to the issue.
    2. **Fix**: Implement the changes.
    3. **Verify**: Run the test/script to confirm the fix works.
    - A task is NOT complete until the verification command passes.
- **Environment Awareness**: At the start of a task, use `get_system_info` to understand the OS, user permissions, and available package managers.
- **Interactive Shell Handling**: 
    - ALWAYS use `interactive=True` for commands that require user input (e.g., `sudo`, `pacman`, `apt`, `pip` installs that might prompt, `vim`, `ssh`).
    - When running an interactive command, tell the user in your reasoning that they may need to provide input (like a password).
- **Workspace Sandbox (STRICT)**: ALL file and shell tools operate from the SAME `workspace/` directory. `read_file("foo.py")` reads `workspace/foo.py`; `run_shell("make")` runs `make` from inside `workspace/`. Pass paths as relative names ("foo.py", "blex_os/Makefile") — NEVER prefix with `workspace/` yourself, that double-prefixes to `workspace/workspace/foo.py`.
- **Writes stay in workspace.** `write_file` enforces this automatically (paths outside `workspace/` raise WorkspacePathError). For `run_shell`, the enforcement is YOUR job — never run `echo X > /path/outside/workspace`, never `cp file /elsewhere`, never `git commit` on anything but a repo inside `workspace/`. Reads outside workspace via shell (e.g., `cat /etc/os-release` for system info, or `git log` on a sibling repo) are fine; writes are not. The project source (`cli.py`, `tools.py`, `multi_agent.py`, etc.) at the parent directory is OFF-LIMITS for modification.
- **Don't pre-validate**: Just call `read_file(path)` — if it errors, then act. Do NOT `list_dir` first to "check if the file exists."
- **Use the efficient tools when they fit** — they save context budget and avoid common mistakes:
    - `code_outline(file)` BEFORE `read_file` on large source files. The outline tells you which symbols exist; only `read_file` after you know which part you want.
    - `apply_diff(file, diff)` INSTEAD of `write_file` for small edits. It sends just the hunks, not the whole file.
    - `grep_codebase(pattern)` INSTEAD of `run_shell("grep -rn …")`. Structured output, skips binary/cache dirs automatically.
    - `run_tests([target])` INSTEAD of guessing pytest/jest/cargo invocations.
    - `python_eval(expr)` INSTEAD of mental math, datetime arithmetic, or JSON manipulation. Use it whenever the answer is computable.
    - `git_diff` / `git_log` / `git_blame` INSTEAD of `run_shell("git …")` for structured output.
- **Verify after every change.** If you `write_file` or run any mutating shell command (make, pip install, git commit, mv, rm, etc.), you MUST IN THE SAME TURN re-run the verification command (e.g. `make`, the test suite, `cat` the resulting file) BEFORE declaring success. Editing the file is not the same as fixing it. NEVER write "build succeeded" or "fix applied" based on a successful write_file alone — base success on a successful build/test/cat output.
- **Verification First**: Before modifying or copying a file, verify its existence and content to avoid redundant work.
- After each tool result, proceed to the NEXT step. Do NOT repeat a step unless it failed and you have a new approach.
- Do NOT ask questions. Do NOT say "how can I help". Just execute.
- If a step fails, retry once with adjusted input, then report and move on.
- When all steps are done, provide a comprehensive summary of what was accomplished and the final state of the task.

## Output
- **Lead with the answer.** First line states the result ("Wrote add.py with the add(a,b) function.", "Tests pass: 8/8.", "Found 3 matches: ..."). No preamble, no "In this task I will..." narration.
- For multi-step work, follow the lead line with a short bulleted recap of what each step did. One line per step. Skip steps that did nothing notable.
- Show diffs for edits, key lines for command output, summaries for long output. Do NOT paste entire tool outputs back to the user — the TUI already shows tool panels.
- If something failed, say so directly on the lead line ("Could not X because Y") and stop — don't dress up failures.

## Reasoning style (when you think aloud / produce <think> content)
- Be direct. State the decision or the next action, not the process of arriving at it.
- Skip openers like "Let me think...", "First, I'll consider...", "Now I need to...". Just say what you're doing.
- One short paragraph or 1-3 short lines is usually enough. If the task is simple, a single line is fine.
- When evaluating tool results, lead with what changed: "The read returned X, so next I'll Y."
- Never restate the user's request back to them.

═══════════════════════════════════════════════════════════════
## Past actions

When the user asks what you did about a past task, file, bug, or feature
("what did you do about X", "did you fix Y", "earlier you changed
something in Z"), call `recall_actions(query)` BEFORE answering.
`recall_actions` is authoritative for this session's mutating actions —
don't reconstruct from memory or guess.""",
    },
    "researcher": {
        "model": os.getenv("OLLAMA_RESEARCHER_MODEL", "qwen3.5:9b"),
        "system_prompt": """You are EzClaw's **Researcher** — gather and synthesize information from the web.

Rules:
- Respond in plain text. No JSON.
- Lead with the answer, not commentary.

Research strategy:
1. Start broad: search google/ddg for the topic.
2. Skim results, identify 1-2 most relevant links.
3. Fetch those links for details.
4. Synthesize into a concise summary (2-4 bullet points or 1-2 paragraphs).
5. Never exceed 4 web_fetch calls.

Source quality: Prefer official docs, reputable sources, recent dates. Note when info might be stale.""",
    },
    "debugger": {
        "model": os.getenv("OLLAMA_DEBUGGER_MODEL", "deepseek-r1:14b"),
        "system_prompt": """You are EzClaw's **Debugger** — find root causes, not just symptoms. **Ground every diagnosis in evidence: code you've read, errors you've reproduced, and (when relevant) documentation you've looked up.**

Rules:
- Respond in plain text. No JSON.
- Provide a clear, actionable fix that an Executor can apply.
- Don't trust training-data recall of error messages — **verify with `web_search` when the error is library-specific, version-sensitive, or unfamiliar.**

Debug methodology:
1. **Reproduce**: Run the code/command to see the error yourself.
2. **Isolate**: Read relevant files. Identify the exact line/component failing.
3. **Ground** (when the error is non-trivial): call `web_search` with the EXACT error message (or a representative phrase from it). Skim the top results. Look for:
     - upstream bug reports / GitHub issues / mailing list threads
     - official documentation describing the function's behavior
     - Stack Overflow answers with verified fixes
   Then `web_fetch` the most relevant result for full content. Cite the source URL in your Analysis.
4. **Root cause**: What is the fundamental issue?
5. **Fix**: Provide a minimal, targeted, and VERIFIABLE fix.

## When to skip the search step
- Trivial syntax errors you can fix from inspection alone
- Issues in the user's own code where the bug is staring at you in the file
- Anything where you're 100% sure of the cause and the standard fix

When in doubt, search — a 5-second search beats a confident-but-wrong diagnosis every time.

## Output Format:
## Analysis
(What you examined, the flow, your reasoning. If you searched, include "Grounded by:" lines with the URLs you consulted.)

## Root Cause
(One sentence: what, where, why)

## Proposed Fix
(The EXACT change needed, with file path and line references. Provide the code block clearly.)

## Verification
(How the executor should verify the fix works)""",
    },
    "general": {
        "model": os.getenv("OLLAMA_GENERAL_MODEL", "qwen3.5:9b"),
        "system_prompt": """You are EzClaw's **General Assistant** — friendly, concise, and context-aware.

Rules:
- Respond in plain text. Be concise — no preambles, no fluff.
- When user shares personal info ("my name is X", "I like Y"), use `remember` to store it.
- When user asks about themselves ("what's my name", "do you know me"), use `recall` to check.
- Use `forget` if the user asks you to delete something.
- If recall returns nothing relevant, say so directly — don't fabricate.
- **For ANY question about workspace content** — a project, file, directory, build, "the code", "how to run X", "is X done" — your training data doesn't contain the user's workspace. Use `list_dir` to see what's there, `read_file` to inspect specific files, and `run_shell` (with `interactive=false`) to check things like `make -n` or `ls`. Don't fabricate answers from prior knowledge when the workspace has the actual data.""",
    },
}


class _SharedAuthState:
    """One-bit shared state for "user pressed [A] = allow for session".

    Each SpecializedAgent reads/writes session_authorized through this
    shared object, so when ONE agent's user-prompt session is authorized,
    ALL sibling agents see it immediately. Previously each agent kept its
    own bool, so pressing [A] in an executor prompt only authorized the
    executor; the next architect step to (say) the researcher would
    re-prompt for auth, defeating the [A] key.
    """
    __slots__ = ("authorized",)

    def __init__(self):
        self.authorized = False


class SpecializedAgent:
    """Self-contained agent with its own model, prompt, tools, and history."""

    def __init__(self, name: str, config: dict, db: Database, auth_state: "_SharedAuthState" = None, session_id: int = None):
        self.name = name
        self.client = build_agent_client()
        self.model = config["model"]
        self.system_prompt = config["system_prompt"]
        # Give all agents access to all registered tools
        self.tools = registry.get_tool_definitions()
        self.db = db
        self.session_id = session_id
        self.messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", 16384))
        self.options = {
            "temperature": 0.0,
            "num_ctx": self.num_ctx,
            "top_p": 0.9,
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", 999)),
        }
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m")
        # Shared with sibling agents — see _SharedAuthState docstring.
        self._auth_state = auth_state if auth_state is not None else _SharedAuthState()
        self._pre_embed_tools()

    def _record_action(self, tool_name: str, args: dict, result: str, assistant_text: str) -> None:
        """Record one mutating tool call to the actions table. Best-effort —
        never propagates out and never blocks the tool."""
        try:
            if self.session_id is None:
                return
            summary = summarize_action(tool_name, args)
            why = extract_why(assistant_text)
            outcome, error_excerpt = classify_outcome(result)
            embed_text = f"{summary} {why or ''}".strip()
            try:
                vec = embed(embed_text)
                emb_blob = pickle.dumps(vec)
            except Exception:
                emb_blob = None
            self.db.add_action(
                session_id=self.session_id,
                tool=tool_name,
                args_json=json.dumps(args, default=str),
                summary=summary,
                why=why,
                outcome=outcome,
                error_excerpt=error_excerpt,
                embedding=emb_blob,
            )
        except Exception as e:
            import sys
            print(f"[action-tracking] record failed: {e}", file=sys.stderr)

    @property
    def session_authorized(self) -> bool:
        return self._auth_state.authorized

    @session_authorized.setter
    def session_authorized(self, value: bool) -> None:
        self._auth_state.authorized = bool(value)

    def _pre_embed_tools(self):
        self._tool_embeddings = {}
        self._tool_name_map = {}
        for t in self.tools:
            name = t['function']['name']
            desc = f"{name}: {t['function']['description']}"
            self._tool_name_map[name] = t
            try:
                self._tool_embeddings[name] = embed(desc)
            except Exception:
                self._tool_embeddings[name] = None

    def _select_relevant_tools(self, user_input: str, top_n: int = 20) -> List[Dict[str, Any]]:
        if len(self.tools) <= top_n:
            return self.tools
        try:
            q_vec = embed(user_input)
        except Exception:
            return self.tools[:top_n]
        scored = []
        for name, t_def in self._tool_name_map.items():
            t_vec = self._tool_embeddings.get(name)
            if t_vec is not None:
                sim = cosine_similarity(q_vec, t_vec)
                scored.append((sim, t_def))
            else:
                scored.append((0.0, t_def))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_n]]

    def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
        # Truncate user input to prevent context overflow
        if len(user_input) > 4000:
            user_input = user_input[:4000] + "\n... (truncated)"

        # Proactive pruning: if agent already has a long history, trim before appending
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        max_chars = int(self.num_ctx * 2.5)
        if total_chars > max_chars or len(self.messages) > 12:
            self.messages = [self.messages[0]] + self.messages[-10:]

        self.messages.append({"role": "user", "content": user_input})

        last_tool_hash = None

        selected_tools = self._select_relevant_tools(user_input, top_n=20)

        max_messages = max(6, int(self.num_ctx / 2048))  # scale with context size

        for _ in range(10):
            full_response, full_reasoning, tool_calls = "", "", []
            in_thinking, raw_buffer = False, ""

            # Re-check total context length before each LLM call
            total_chars = sum(len(m.get("content", "")) for m in self.messages)
            if total_chars > max_chars:
                if len(self.messages) > max_messages:
                    self.messages = [self.messages[0]] + self.messages[-(max_messages - 1):]
                else:
                    # Can't trim by count — truncate individual message content instead
                    for m in self.messages:
                        if m["role"] == "system":
                            continue
                        if len(m.get("content", "")) > 1500:
                            m["content"] = m["content"][:1500] + "\n... (truncated)"

            try:
                stream = self.client.chat(
                    model=self.model, messages=self.messages,
                    tools=selected_tools or None,
                    options=self.options, keep_alive=self.keep_alive, stream=True,
                )
                first_chunk = next(stream)
            except Exception as e:
                if "does not support tools" in str(e).lower() or "400" in str(e):
                    stream = self.client.chat(
                        model=self.model, messages=self.messages,
                        options=self.options, keep_alive=self.keep_alive, stream=True,
                    )
                    first_chunk = next(stream)
                else:
                    yield {"type": "content", "content": f"[System: Agent error - {str(e)}]"}
                    break

            def stream_with_first(s, f):
                yield f
                for c in s:
                    yield c

            for chunk in stream_with_first(stream, first_chunk):
                reasoning = getattr(chunk.message, 'reasoning', None) or (chunk.message.get('reasoning') if isinstance(chunk.message, dict) else None)
                if reasoning:
                    full_reasoning += reasoning
                    yield {"type": "reasoning", "content": reasoning}

                if chunk.message.content:
                    raw_buffer += chunk.message.content

                    while True:
                        if not in_thinking:
                            match = re.search(r'<(think|thought|reasoning)\b[^>]*>', raw_buffer, re.IGNORECASE)
                            if match:
                                pre = raw_buffer[:match.start()]
                                if pre:
                                    full_response += pre
                                    yield {"type": "content", "content": pre}
                                in_thinking = True
                                raw_buffer = raw_buffer[match.end():]
                                continue

                            if '<' in raw_buffer:
                                last_bracket = raw_buffer.rfind('<')
                                if any(tag.startswith(raw_buffer[last_bracket:].lower()) for tag in ["<think", "<thought", "<reasoning"]):
                                    pre = raw_buffer[:last_bracket]
                                    if pre:
                                        full_response += pre
                                        yield {"type": "content", "content": pre}
                                    raw_buffer = raw_buffer[last_bracket:]
                                    break
                                else:
                                    full_response += raw_buffer
                                    yield {"type": "content", "content": raw_buffer}
                                    raw_buffer = ""
                                    break
                            else:
                                if raw_buffer:
                                    full_response += raw_buffer
                                    yield {"type": "content", "content": raw_buffer}
                                    raw_buffer = ""
                                break
                        else:
                            match = re.search(r'</(think|thought|reasoning)\s*>', raw_buffer, re.IGNORECASE)
                            if match:
                                pre = raw_buffer[:match.start()]
                                if pre:
                                    full_reasoning += pre
                                    yield {"type": "reasoning", "content": pre}
                                in_thinking = False
                                raw_buffer = raw_buffer[match.end():]
                                continue

                            if '</' in raw_buffer:
                                last_bracket = raw_buffer.rfind('</')
                                if any(tag.startswith(raw_buffer[last_bracket:].lower()) for tag in ["</think>", "</thought>", "</reasoning"]):
                                    pre = raw_buffer[:last_bracket]
                                    if pre:
                                        full_reasoning += pre
                                        yield {"type": "reasoning", "content": pre}
                                    raw_buffer = raw_buffer[last_bracket:]
                                    break
                                else:
                                    full_reasoning += raw_buffer
                                    yield {"type": "reasoning", "content": raw_buffer}
                                    raw_buffer = ""
                                    break
                            else:
                                if raw_buffer:
                                    full_reasoning += raw_buffer
                                    yield {"type": "reasoning", "content": raw_buffer}
                                    raw_buffer = ""
                                break

                if getattr(chunk.message, "tool_calls", None):
                    tool_calls.extend(chunk.message.tool_calls)

            if raw_buffer:
                if in_thinking:
                    full_reasoning += raw_buffer
                    yield {"type": "reasoning", "content": raw_buffer}
                else:
                    full_response += raw_buffer
                    yield {"type": "content", "content": raw_buffer}
                raw_buffer = ""

            if not tool_calls:
                if full_response.strip() or full_reasoning.strip():
                    msg = {"role": "assistant", "content": full_response}
                    if full_reasoning:
                        msg["reasoning"] = full_reasoning
                    self.messages.append(msg)
                    break
                break

            current_hash = hash(
                str([(t.function.name, t.function.arguments) for t in tool_calls])
            )
            if current_hash == last_tool_hash:
                from phrases import pick as _pick, LOOP_DETECTED as _LD
                yield {"type": "content", "content": f"\n[🦀 {_pick(_LD)}.]"}
                break
            last_tool_hash = current_hash

            self.messages.append({
                "role": "assistant", "content": full_response,
                "tool_calls": [
                    {"function": {"name": t.function.name, "arguments": t.function.arguments}}
                    for t in tool_calls
                ],
            })

            for tool_call in tool_calls:
                tool_func = registry.tools.get(tool_call.function.name)
                is_interactive = tool_call.function.arguments.get("interactive", False)

                if tool_func and getattr(tool_func, "auth_required", False) and not self.session_authorized:
                    auth = yield {
                        "type": "auth_required",
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                    if auth == "deny":
                        self.messages.append({
                            "role": "tool", "content": "Authorization denied.",
                            "name": tool_call.function.name,
                        })
                        yield {"type": "tool_end", "name": tool_call.function.name, "result": "Authorization denied."}
                        continue
                    elif auth == "allow_session":
                        self.session_authorized = True

                yield {
                    "type": "tool_start",
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                    "interactive": is_interactive,
                }
                try:
                    result = tool_func(**tool_call.function.arguments) if tool_func else "Tool not found."
                except Exception as e:
                    result = f"Error: {str(e)}"

                result_str = str(result)
                full_result = result_str
                if len(result_str) > 8000:
                    head = result_str[:5000]
                    tail = result_str[-2500:]
                    result_str = f"{head}\n\n... ({len(full_result)} chars total, middle truncated) ...\n\n{tail}"
                self.messages.append({
                    "role": "tool", "content": result_str, "name": tool_call.function.name,
                })
                if tool_call.function.name in MUTATING_TOOLS:
                    self._record_action(
                        tool_name=tool_call.function.name,
                        args=tool_call.function.arguments,
                        result=full_result,
                        assistant_text=full_response,
                    )
                yield {"type": "tool_end", "name": tool_call.function.name, "result": full_result}


class Architect:
    """Routes tasks to specialized agents and tracks the plan."""

    def __init__(self, db: Database):
        self.client, self.model = build_architect_client()
        self.db = db
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", 16384))
        self.use_deepseek = os.getenv("ARCHITECT_PROVIDER", "ollama") == "deepseek"
        self.messages: List[Dict] = [{
            "role": "system",
            "content": """You are the **Architect** — a senior systems designer orchestrating a multi-agent workflow.

You operate in TWO modes. Each user prompt will start with either `## PLANNING REQUEST` or `## EXECUTION REQUEST`. Read the header carefully and respond with the JSON shape required by that mode.

## Routing roles available
- **executor**: File edits, shell commands, code implementation, memory management, verification runs.
- **researcher**: Web searching, documentation gathering.
- **debugger**: Root-cause analysis. Only route here for UNEXPECTED errors.
- **general**: Conversational responses.

═══════════════════════════════════════════════════════════════
## Mode 1: PLANNING REQUEST
═══════════════════════════════════════════════════════════════

You receive the user's original request plus context. Classify the request and produce ONE of:

**Multi-step request** → return a structured plan:
{
  "kind": "plan",
  "title": "<one-line summary, ≤60 chars, no trailing period>",
  "tasks": [
    {"id": 1, "description": "<short user-facing line, ≤80 chars>"},
    {"id": 2, "description": "..."}
  ]
}

Rules for plans:
- Between 2 and 7 tasks. If you cannot decompose into ≥2 meaningful tasks, return `kind: single` instead.
- Tasks must be in execution order. No out-of-order dependencies.
- Descriptions are USER-FACING summaries, not implementation jargon.
- IDs are 1-based, contiguous, ascending.

**Single-step / conversational request** → return:
{"kind": "single", "reason": "<one-line explanation>"}

Return `kind: single` ONLY when the request is purely workspace-independent:
- Greetings, acknowledgments, social niceties ("hi", "thanks", "lol")
- Definitions of general concepts ("what is recursion", "explain async/await")
- Arithmetic, simple computations the model can do in its head

**Anything that names something in the workspace requires inspection** — a project name (blex_os, the API server), a file (cli.py, README), a directory, "the build", "the tests", "this code", "the bug", "how to run X" where X is in `./workspace/` — return `kind: plan`. The executor needs `read_file` / `list_dir` / `run_shell` to actually look at what's there; the model's training data does NOT contain the user's workspace.

═══════════════════════════════════════════════════════════════
## Mode 2: EXECUTION REQUEST
═══════════════════════════════════════════════════════════════

You receive: the current plan (with task statuses), the task_context (recent step results), memory/skills/routing context. Decide the NEXT step.

Return:
{
  "kind": "execute",
  "current_task_id": <int>,
  "recommended_agent": "executor|general|researcher|debugger",
  "reasoning": "<one short line on the routing choice>",
  "plan": "<numbered step-by-step instructions for the routed agent, addressing the current task>",
  "task_updates": [{"id": <int>, "status": "done|failed|skipped"}],
  "new_tasks": [{"after_id": <int>, "description": "<short line>"}],
  "complete": false,
  "reflection": {
    "goal": "<noun phrase>",
    "observation": "<what happened in the last step>",
    "critical_thinking": "<one line on why this action>"
  }
}

Rules for execution:
- `current_task_id` must reference an existing task in the plan (not yet `done`/`failed`/`skipped`).
- `task_updates` is for tasks finishing in the current step. Only mark `done` after a successful verification. Mark `failed` only after retries are exhausted. Mark `skipped` only when the task is genuinely no longer needed.
- `new_tasks` is for genuinely-new work discovered during execution. Leave empty most of the time. Each entry's `after_id` must reference an existing task.
- **After a `debugger` step:** the debugger returns a diagnosis plus a numbered "Proposed Fix". This output is INTERNAL — the user never sees it. DO NOT re-narrate that diagnosis or paste its steps into `plan`. Convert each step of the Proposed Fix into a `new_tasks` entry (`after_id` = the debugger task's id, one entry per concrete step), mark the debugger task `done`, route the NEXT turn to the agent that should execute the first new step (usually `executor`). Keep `reflection.observation` to one short sentence.
- **The user's chat only ever shows what addresses their original prompt.** When you set `complete: true`, the LAST visible step's output is what the user reads as the answer. Route the final step to an agent whose output naturally responds to the user (executor for "did it work?" recaps; researcher/general for explanatory questions). Never let the debugger be the final visible step — its content is suppressed from chat by design.
- Set `complete: true` ONLY when every task in the plan is `done` or `skipped` AND the user's full original intent is verifiably satisfied. Premature completion is forbidden.
- `plan` is the step-by-step instruction the routed agent will execute this turn. Make it concrete and actionable: "Read sse_handler.py, find the handle_disconnect function, add a `connection.cleanup()` call before the return." Not "Work on the leak."
- `reflection.observation` is one short sentence describing what actually happened in the previous step. Skip if first step.

═══════════════════════════════════════════════════════════════
## When no plan is active (single-step path)
═══════════════════════════════════════════════════════════════

If the EXECUTION REQUEST says `Plan: (none — single-step request)`, the routed agent will respond once or twice. Set `current_task_id` to `0`, leave `task_updates` and `new_tasks` empty.

**Routing in single-step mode:**
- `general` — pure chat ONLY (greetings, definitions of general concepts, arithmetic). The general agent uses no tools by default; route here only when there's nothing in the workspace to look at.
- `executor` — anything that needs file/shell access. If the user mentions a workspace file, project, or "the code", route here even in single-step.
- `researcher` — web/documentation lookup the executor can't do locally.
- `debugger` — root-cause analysis of an unexpected error.

For completion:
- **Set `complete: true`** as soon as the routed agent has produced a reply that addresses the user's request.
- **Keep `complete: false`** only if the agent errored out, returned obviously incomplete output, or the routing was wrong and you need to retry with a different agent.

## Persistence

Keep iterating until the user's primary goal is achieved. Re-read the original request each step; don't drift to easier sub-goals or declare victory on a partial result. On step failure: diagnose, switch agent or tool path, insert a corrective task — don't halt. Only stop with `complete: false` for genuine blockers: missing credentials, ambiguous requirements, external service unavailable. The orchestrator allows 40 steps and 3 pivots per run.

## Use available skills

When the `⚑ MATCHED SKILLS` block is present, the listed procedure(s) take precedence over generic web search, scattershot fetches, or freshly-invented approaches. The block is only populated when the matcher confirmed a genuine fit (semantic similarity above threshold).

**Rules for skill-matched requests:**
- Your `plan` field (the instruction the sub-agent runs this turn) MUST reference the skill by name and pass through its concrete steps. Example: "Follow the 'Weather Forecast Retrieval' skill: web_fetch https://wttr.in/Giza?format=3, then summarize the result for the user."
- Do NOT route to `researcher` or call `web_search` for a request a skill already solves. The skill exists because the freeform approach failed before.
- If you genuinely believe the skill does NOT fit (wrong location semantics, stale URL, etc.), say so in `reasoning` and proceed with a custom approach. Silence = use the skill.

If the matched-skills block is empty, plan freshly.

## Conversation flow

References like "that file", "fix it", "make it shorter" point at past turns in `## Conversation History` — each turn lists the user's text, the assistant's reply, tools used, and any plan summary. Resolve such references from history; don't treat the new request as fresh context.

═══════════════════════════════════════════════════════════════
## General style
═══════════════════════════════════════════════════════════════

- One short sentence per reflection field. Direct, no filler.
- Return ONLY the JSON object. No prose before or after. No markdown fences.
- Empty arrays and empty strings are fine where no new information applies.""",
        }]

    def _prune_messages(self):
        """Keep architect history bounded — system prompt + last 3 turns."""
        if len(self.messages) > 7:
            self.messages = [self.messages[0]] + self.messages[-6:]

    def _chat(self, prompt: str, temperature: float = 0.0) -> str:
        self._prune_messages()
        # Truncate prompt if it alone would overflow
        max_prompt_chars = int(self.num_ctx * 3) - sum(len(m.get("content", "")) for m in self.messages)
        if max_prompt_chars < 500:
            self._prune_messages()
            self.messages = self.messages[:1]
            max_prompt_chars = int(self.num_ctx * 3) - sum(len(m.get("content", "")) for m in self.messages)
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max(500, max_prompt_chars)] + "\n... (truncated)"

        if self.use_deepseek:
            import openai
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages + [{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        resp = self.client.chat(
            model=self.model,
            messages=self.messages + [{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": temperature, "num_ctx": self.num_ctx},
        )
        return resp["message"]["content"].strip()

    def plan(
        self,
        user_input: str,
        memory_block: str = "",
        skills_block: str = "",
        history_block: str = "",
        map_block: str = "",
    ):
        """Planning pass. Returns a Plan object if the request is multi-step,
        or None if it's single-step / conversational.

        Retries once on malformed JSON, then returns None.
        """
        from plan import Plan, Task

        prompt = f"""## PLANNING REQUEST

## User Request
{user_input}

## Conversation History
{history_block}

## Codebase Map
{map_block}

## Memory & Skills
{memory_block}{skills_block}

## Decision Required
Classify this request. If multi-step, return a `kind: plan` JSON with 2-7 tasks. If single-step or conversational, return `kind: single`.

**If the `⚑ MATCHED SKILLS` block above contains a procedure that fully covers this request, prefer `kind: single` — the next step will be one direct application of the skill, not a multi-task plan.** Multi-task plans are for genuinely multi-stage work (read file, modify, test, commit). A skill that says "fetch URL → parse → summarize" is single-step from the orchestrator's view.

Return ONLY the JSON object."""

        for attempt in range(2):
            try:
                content = self._chat(
                    prompt if attempt == 0 else prompt + "\n\nCRITICAL: Return ONLY valid JSON.",
                    temperature=0.0,
                )
                data = extract_json(content)
                kind = data.get("kind")
                if kind == "single":
                    return None
                if kind == "plan":
                    title = str(data.get("title", "")).strip()
                    raw_tasks = data.get("tasks", [])
                    if not isinstance(raw_tasks, list) or not raw_tasks:
                        continue
                    tasks = []
                    for i, t in enumerate(raw_tasks, start=1):
                        if not isinstance(t, dict):
                            continue
                        desc = str(t.get("description", "")).strip()
                        if not desc:
                            continue
                        tasks.append(Task(id=i, description=desc[:120]))
                    if len(tasks) < 2:
                        return None
                    return Plan(title=title or "Plan", tasks=tasks)
                return None
            except Exception:
                continue
        return None

    def execute(
        self,
        plan,
        task_context: str,
        memory_block: str = "",
        skills_block: str = "",
        experiences_block: str = "",
        routing_block: str = "",
        history_block: str = "",
        map_block: str = "",
        temperature: float = 0.0,
        pivot_hint: str = "",
    ) -> "Dict[str, Any]":
        """Execution-mode call. Returns the structured intent dict. When
        `plan` is None, the architect operates in single-step mode (no plan
        active); when present, the plan is rendered into the prompt so the
        architect knows which task to advance.

        On parse failure, returns a safe fallback intent that does not
        mutate the plan.
        """
        if plan is not None:
            plan_render_lines = [f"Plan: {plan.title}"]
            for t in plan.tasks:
                marker = {
                    "pending": " ", "in_progress": "▸", "done": "✓",
                    "failed": "✗", "skipped": "⊘",
                }.get(t.status, " ")
                plan_render_lines.append(f"  [{marker}] {t.id}. {t.description}  ({t.status})")
            plan_block = "\n".join(plan_render_lines)
        else:
            plan_block = "Plan: (none — single-step request)"

        pivot_block = ""
        if pivot_hint:
            pivot_block = (
                "## CRITICAL PIVOT REQUIRED\n"
                f"{pivot_hint}\n"
                "Do NOT re-issue the previous plan. Choose a fundamentally different "
                "approach: switch the recommended_agent, decompose differently, "
                "or use a different tool.\n\n"
            )

        prompt = f"""## EXECUTION REQUEST

{pivot_block}{plan_block}

## Task Context
{task_context}

## Conversation History
{history_block}

## Auxiliary Context
### Codebase Map
{map_block}
### Past Experiences
{experiences_block}
{memory_block}{skills_block}{routing_block}

## Decision Required
Decide the next routing step. Return the EXECUTION JSON object."""

        for attempt in range(2):
            try:
                content = self._chat(
                    prompt if attempt == 0 else prompt + "\n\nCRITICAL: Return ONLY valid JSON.",
                    temperature=temperature,
                )
                intent = extract_json(content)
                intent.setdefault("kind", "execute")
                intent.setdefault("current_task_id", 0)
                intent.setdefault("recommended_agent", "executor")
                intent.setdefault("reasoning", "")
                intent.setdefault("plan", "")
                intent.setdefault("task_updates", [])
                intent.setdefault("new_tasks", [])
                intent.setdefault("complete", False)
                intent.setdefault("reflection", {})

                valid_agents = {"executor", "general", "researcher", "debugger"}
                if intent.get("recommended_agent") not in valid_agents:
                    intent["recommended_agent"] = "executor"
                if not isinstance(intent.get("task_updates"), list):
                    intent["task_updates"] = []
                if not isinstance(intent.get("new_tasks"), list):
                    intent["new_tasks"] = []
                return intent
            except Exception:
                continue

        # Safe fallback — keep the loop alive without mutating the plan.
        return {
            "kind": "execute",
            "current_task_id": 0,
            "recommended_agent": "executor",
            "reasoning": "Parse fallback",
            "plan": "",
            "task_updates": [],
            "new_tasks": [],
            "complete": False,
            "reflection": {"critical_thinking": "Parsing failed; routing to executor as a default."},
        }



class MultiAgentSystem:
    def __init__(self, session_id: Optional[int] = None):
        self.db = Database(os.getenv("DATABASE_PATH", "ezclaw.db"))
        create_memory_tools(self.db)
        create_action_tracking_tools(self.db)
        self.skills = load_skills()
        if session_id is None:
            session_id = self.db.get_last_session_id() or self.db.create_session()
        self.session_id = session_id
        set_session_context(self.session_id)
        self.architect = Architect(self.db)
        # Shared auth state so pressing [A] in any sub-agent's prompt
        # authorizes the entire session, not just that one role.
        self._shared_auth = _SharedAuthState()
        self.agents = {
            name: SpecializedAgent(name, cfg, self.db, auth_state=self._shared_auth, session_id=self.session_id)
            for name, cfg in AGENT_DEFS.items()
        }
        self._conversation_history: List[Dict[str, str]] = []
        # Current plan for the in-flight user request. Reset on each run.
        self.current_plan = None

    def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
        yield from self.run(user_input)

    # ── Backwards-compat properties for CLI ──────────────────────

    @property
    def session_authorized(self):
        # Reads through the shared state — single source of truth.
        return self._shared_auth.authorized

    @session_authorized.setter
    def session_authorized(self, value):
        self._shared_auth.authorized = bool(value)

    @property
    def model(self):
        return ", ".join(f"{n}: {a.model}" for n, a in self.agents.items())

    @property
    def messages(self):
        return self.architect.messages

    def _format_history(self) -> str:
        """Render the last few turns as a context block for prompts.

        Includes tool calls and plan summaries per turn so the next turn
        can resolve references like "that file" or "the bug we discussed"
        — without those, the agent only sees user+assistant text and
        loses track of what was actually done.

        Returns content only — no `## Conversation History` header (the
        caller wraps it). This avoids the double-header bug the older
        code shipped with.
        """
        if not self._conversation_history:
            return ""
        lines = []
        for turn in self._conversation_history[-5:]:
            user_text = turn["user"]
            if len(user_text) > 200:
                user_text = user_text[:200] + "…"
            lines.append(f"- User: {user_text}")

            resp = turn.get("assistant") or ""
            if resp:
                if len(resp) > 1200:
                    resp = resp[:1200] + "… [truncated]"
                lines.append(f"  Assistant: {resp}")

            tools = turn.get("tools") or []
            if tools:
                # Dedup while preserving order — same tool can appear many
                # times in a turn, the history just needs the gist.
                seen = set()
                unique_tools = [t for t in tools if not (t in seen or seen.add(t))]
                lines.append(f"  Tools: {', '.join(unique_tools[:8])}")

            plan_summary = turn.get("plan_summary")
            if plan_summary:
                lines.append(f"  Plan: {plan_summary}")
        return "\n".join(lines) + "\n"

    def _append_conversation_turn(
        self,
        user_input: str,
        assistant_text: str,
        step_history=None,
        plan=None,
    ) -> None:
        """Single entry-point for recording a completed turn. Captures
        user text + assistant reply + every tool used + a plan summary
        if a plan was active. Replaces the bare {user, assistant} dicts
        that older code appended directly."""
        all_tools = []
        if step_history:
            for step in step_history:
                all_tools.extend(step.get("tools") or [])

        plan_summary = None
        if plan is not None and getattr(plan, "tasks", None):
            done = sum(1 for t in plan.tasks if t.status == "done")
            total = len(plan.tasks)
            plan_summary = f"{plan.title} ({done}/{total} done)"

        self._conversation_history.append({
            "user": user_input,
            "assistant": assistant_text,
            "tools": all_tools,
            "plan_summary": plan_summary,
        })

    def _propose_skill_from_run(
        self,
        user_input: str,
        final_response: str,
        step_history: list,
        recent_history: str,
    ) -> Optional[dict]:
        """Ask the architect to distill the just-completed turn into a
        reusable skill IF the user clearly guided the assistant to
        success. Returns {name, description, procedure} or None.

        The architect is asked to be conservative — only propose a skill
        when the work this turn would actually save effort if recalled
        next time. Generic chats / first-try successes return None."""
        # Compress step history to keep the prompt cheap.
        steps_summary = []
        for s in step_history[:8]:
            tools = ", ".join(s.get("tools") or [])
            outcome = s.get("outcome", "?")
            out = (s.get("output") or "")[:300]
            steps_summary.append(
                f"- [{s.get('agent', '?')}|{outcome}] tools=[{tools}] → {out}"
            )

        prompt = f"""Examine this just-completed turn. The cheap keyword
detector flagged that the user may have guided/corrected me toward a
working approach. You decide whether to actually save a skill.

## User's latest request
{user_input}

## What I ended up answering
{final_response[:1200]}

## Steps taken
{chr(10).join(steps_summary) if steps_summary else "(no steps)"}

## Recent conversation (for context — were there corrections?)
{recent_history[:2000]}

## Decision
Return a JSON object:

  {{
    "propose": true | false,
    "reason": "<one line — why this is or isn't worth saving>",
    "skill": {{
      "name": "<3-6 word noun phrase, e.g. 'Weather Forecast Retrieval'>",
      "description": "<one-line: what this skill does>",
      "procedure": "<numbered steps, each starting with a verb. Concrete URLs, tool names, and parameter shapes — not abstract guidance.>"
    }}
  }}

Propose `true` ONLY when ALL of:
- The user actually said something corrective/instructive (a URL, a tool, a step, a "don't do X, do Y").
- The successful approach would actually save effort next time (genuinely reusable).
- The skill is general enough — strip user-specific details (location names, account IDs).

Propose `false` when:
- The user just asked a normal question and I got it right first try.
- The "correction" was about style / formatting / "shorter please".
- The work was so context-specific no future turn would benefit.

Return ONLY the JSON object."""

        try:
            content = self.architect._chat(prompt, temperature=0.1)
            data = extract_json(content)
        except Exception:
            return None
        if not isinstance(data, dict) or not data.get("propose"):
            return None
        skill = data.get("skill") or {}
        name = (skill.get("name") or "").strip()
        description = (skill.get("description") or "").strip()
        procedure = (skill.get("procedure") or "").strip()
        if not (name and procedure):
            return None
        return {
            "name": name[:80],
            "description": description[:300],
            "procedure": procedure[:4000],
            "reason": (data.get("reason") or "")[:200],
        }

    def clear_session_history(self):
        for a in self.agents.values():
            a.messages = [a.messages[0]]
        self.architect.messages = [self.architect.messages[0]]
        self._conversation_history.clear()

    def _prune_architect(self):
        """Keep architect history bounded — system prompt + last 3 turns."""
        if len(self.architect.messages) > 7:
            self.architect.messages = (
                [self.architect.messages[0]] + self.architect.messages[-6:]
            )

    # ── Routing Enhancements ──────────────────────────────────

    ROUTING_EXAMPLES = [
        # General chat
        ("hello", "general"), ("hi how are you", "general"), ("thanks", "general"),
        ("goodbye", "general"), ("good morning", "general"),
        # Research / web lookup
        ("search the web for", "researcher"), ("look up information about", "researcher"),
        ("find documentation for", "researcher"), ("what is the latest news", "researcher"),
        # Memory operations (executor handles via memory tools)
        ("remember that my favorite color is blue", "executor"),
        ("remember my name is John", "executor"),
        ("do you remember anything about me", "executor"),
        ("what do you know about me", "executor"),
        ("what is my favorite color", "executor"),
        ("what is my name", "executor"),
        # Development tasks — go straight to executor, skip architect cycle
        ("write a function that", "executor"),
        ("write a python script to", "executor"),
        ("implement a function", "executor"),
        ("add a new function to", "executor"),
        ("edit the file", "executor"),
        ("modify this file", "executor"),
        ("update the code in", "executor"),
        ("refactor this function", "executor"),
        ("read the file", "executor"),
        ("show me the contents of", "executor"),
        ("list files in", "executor"),
        ("run this command", "executor"),
        ("run the tests", "executor"),
        ("execute pytest", "executor"),
        ("install this package", "executor"),
        ("git status", "executor"),
        ("git diff", "executor"),
        ("commit these changes", "executor"),
        ("fix the bug in", "executor"),
        ("there is an error when", "executor"),
        ("this is failing with", "executor"),
        ("traceback shows", "executor"),
    ]

    def _estimate_tool_kinds(self, user_input: str) -> Optional[int]:
        """Single LLM call estimating the distinct tool kinds the executor
        would need to fulfill `user_input`. Returns the count, or None on
        any failure (timeout, parse failure, empty result).

        Used by run() to gate the fast-route to executor: if this returns
        a count >= PREFLIGHT_KIND_THRESHOLD, the architect plans instead.
        """
        prompt = (
            "You are pre-flighting a tool plan. List the EZCLAW tool names "
            "you would need to fulfill this request, ONE PER LINE, no prose, "
            "no numbering.\nUse ONLY these names:\n"
            "  apply_diff, write_file, run_shell, read_file, list_dir, "
            "grep_codebase, web_search, web_fetch, recall, code_outline, "
            "git_diff, git_log, git_blame, run_tests, python_eval, "
            "schedule_task, unschedule_task, ask_user, current_datetime, "
            "get_system_info\n\n"
            f"Request: {user_input}\n\nTools needed (one per line):"
        )
        try:
            resp = self.architect.client.chat(
                model=self.architect.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_ctx": 2048},
            )
            text = resp.get("message", {}).get("content", "")
            kinds = _parse_tool_lines(text)
            return len(kinds) if kinds else None
        except Exception:
            return None

    def _short_circuit_classify(self, user_input: str) -> Optional[str]:
        # Layer 1: cheap textual heuristic for trivial chat. Anything that
        # looks like a short greeting / acknowledgement / nonsense one-liner
        # ("hi", "thanks", "Captain Claw!", "yo") shouldn't burn an embedding
        # call — and definitely shouldn't reach the architect loop where it
        # can spin until the stuck-detector trips.
        stripped = user_input.strip()
        if 0 < len(stripped) <= 40:
            # No code/punctuation that suggests an actual request
            no_code_chars = not any(c in stripped for c in "(){}[];=<>|\\$/`")
            no_request_verbs = not any(
                w in stripped.lower().split()
                for w in {
                    "make", "build", "create", "write", "fix", "add", "implement",
                    "refactor", "test", "run", "read", "list", "show", "find",
                    "search", "look", "edit", "modify", "update", "delete",
                    "remember", "forget", "install", "commit", "push", "pull",
                }
            )
            if no_code_chars and no_request_verbs:
                return "general"

        # Layer 2: embedding-similarity routing across the known examples.
        examples = [ex for ex, _ in self.ROUTING_EXAMPLES]
        labels = [lb for _, lb in self.ROUTING_EXAMPLES]
        return classify_by_similarity(user_input, examples, labels, threshold=0.6)

    def _apply_intent_to_plan(self, intent):
        """Apply task_updates / new_tasks / current_task_id from an execute()
        intent to self.current_plan. Yields a plan_update chunk if a plan is
        active. This is a generator helper — callers iterate."""
        if self.current_plan is None:
            return
        for upd in intent.get("task_updates", []) or []:
            if isinstance(upd, dict) and "id" in upd and "status" in upd:
                self.current_plan.advance(upd["id"], upd["status"])
        for new in intent.get("new_tasks", []) or []:
            if isinstance(new, dict) and "after_id" in new and "description" in new:
                self.current_plan.insert(new["after_id"], new["description"])
        current_id = intent.get("current_task_id")
        if current_id and self.current_plan.get_task(current_id):
            self.current_plan.advance(current_id, "in_progress")
        yield {"type": "plan_update", "plan": self.current_plan}

    def _format_routing_priors(self, priors: List[Dict]) -> str:
        if not priors:
            return ""
        lines = ["\n[Similar Past Routing Decisions]:"]
        for p in priors:
            lines.append(f"- Query: '{p['query'][:80]}' → {p['agent']} (success={p['success']})")
        return "\n".join(lines) + "\n"

    def _summarize_experience(self, task: str, task_context: str):
        """Generates a concise summary of a successful task for episodic memory."""
        try:
            prompt = f"""Summarize this successful task for future reference.
Task: {task}
Steps taken:
{task_context}

Return a concise summary (2-3 sentences) focused on the PROBLEM and the SOLUTION pattern.
No fluff. No "In this task...". Just facts."""
            
            resp = self.architect.client.chat(
                model=self.architect.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_ctx": 4096}
            )
            summary = resp["message"]["content"].strip()
            self.db.add_experience(task, summary)
        except Exception as e:
            pass

    # ── Orchestration ──────────────────────────────────────────

    def run(self, user_input: str) -> Iterator[Dict[str, Any]]:
        if len(user_input) > 4000:
            user_input = user_input[:4000] + "\n... (truncated)"
        task_context = f"User Request: {user_input}"
        self.current_plan = None
        # Persistence-first orchestration: keep iterating until the user's
        # primary goal is achieved, only halt on a genuine blocker. Three
        # knobs cooperate:
        #   - max_steps gives generous runway for complex multi-task plans
        #   - pivot_count allows several different approaches before giving up
        #   - the stuck detector still catches true infinite loops (same plan,
        #     same agent, zero tool calls AND zero output)
        max_steps = 40
        last_step_hash = None
        stuck_repeats = 0
        STUCK_LIMIT = 3  # bumped from 2 — research/web-fetch chains often
                          # legitimately re-issue the same plan across steps
                          # while making progress on different URLs
        pivot_count = 0
        MAX_PIVOTS = 3
        agent_has_responded = False
        step_history = []
        final_response = ""

        for a in self.agents.values():
            a.messages = [a.messages[0]]

        recent_history = self._format_history()
        history_block = f"\n## Conversation History\n{recent_history}\n" if recent_history else ""

        # Pillar 2: Initial Codebase Map (Generated ONCE per run)
        codebase_map = ""
        try:
            from tools import generate_codebase_map
            codebase_map = generate_codebase_map(".")
        except: pass
        map_block = f"\n## Codebase Map\n{codebase_map[:3000]}\n" if codebase_map else ""

        short_circuit_agent = self._short_circuit_classify(user_input)
        if short_circuit_agent == "executor":
            kind_count = self._estimate_tool_kinds(user_input)
            if kind_count is None or kind_count >= PREFLIGHT_KIND_THRESHOLD:
                kind_label = "?" if kind_count is None else str(kind_count)
                yield {
                    "type": "status",
                    "content": f"[pre-flight: {kind_label} tool kinds — planning]\n",
                }
                short_circuit_agent = None
        if short_circuit_agent and short_circuit_agent != "debugger":
            agent_key = short_circuit_agent
            agent = self.agents.get(agent_key)
            if agent:
                agent.messages = [agent.messages[0]]
                memory_facts = self.db.search_memories_hybrid(user_input[:1000], alpha=0.6)
                memory_hint = f"\n[Memory]: {memory_facts}\n" if memory_facts else ""
                from phrases import pick as _pick, AGENT_WORKING as _AW
                yield {"type": "status", "content": f"[{agent_key}] {_pick(_AW)} (fast-routed)\n"}
                sc_output = ""
                sc_tool_results = []
                agent_input = f"{history_block}{memory_hint}{user_input}"
                for chunk in agent.chat_stream(agent_input):
                    if chunk["type"] == "content":
                        sc_output += chunk["content"]
                    elif chunk["type"] == "tool_end":
                        sc_tool_results.append(chunk["name"])
                    yield chunk
                if not sc_output.strip() and sc_tool_results:
                    sc_output = "Done."
                    yield {"type": "content", "content": "Done."}

                # Escape hatch: if the fast-routed agent returned a give-up signal,
                # don't trust the short-circuit — escalate to the full architect loop.
                escalate_re = re.compile(
                    r"\b(not found|could not|cannot|unable to|does not exist|"
                    r"no such file|i don'?t know|wasn'?t able|was not found)\b",
                    re.IGNORECASE,
                )
                success_re = re.compile(r"\b(i found|successfully|here is|here are|done)\b", re.IGNORECASE)
                if escalate_re.search(sc_output) and not success_re.search(sc_output):
                    from phrases import pick as _pick, ESCALATING as _ESC
                    yield {"type": "status", "content": f"\n[{_pick(_ESC)}]\n"}
                    self.db.store_routing_decision(user_input, agent_key, False)
                    agent.messages = [agent.messages[0]]
                    task_context = f"User Request: {user_input}\n\n[FAILURE] Fast-route to '{agent_key}' returned: {sc_output.strip()[:500]}"
                    # Fall through to the architect loop below.
                else:
                    yield {"type": "status", "content": "Done.\n"}
                    self.db.store_routing_decision(user_input, agent_key, True)
                    # Fast-route case: synthesize a single-step history of
                    # the one tool batch the routed agent ran (if any).
                    sc_steps = [{"agent": agent_key, "tools": sc_tool_results}] if sc_tool_results else None
                    self._append_conversation_turn(user_input, sc_output.strip(), step_history=sc_steps)
                    return

        # Hoist constant-per-turn lookups out of the step loop.
        # These all depend only on user_input, which doesn't change across architect steps —
        # running them per-step was N embed+search calls per turn for no extra signal.
        memory_facts = self.db.search_memories_hybrid(user_input[:1000], alpha=0.6)
        memory_block = f"\n[Memory]: {memory_facts}\n" if memory_facts else ""

        routing_priors = self.db.search_similar_routing(user_input[:1000], limit=3)
        routing_block = self._format_routing_priors(routing_priors)

        experiences = self.db.search_experiences(user_input[:1000], limit=2)
        exp_block = "\n## Lessons Learned from Past Tasks\n" + "\n".join(
            [f"- Task: {e['task']}\n  Result: {e['trace']}" for e in experiences]
        ) + "\n" if experiences else ""

        # Initial skill match against the raw user input — gives the
        # planning pass visibility into any learned skill that fits the
        # request right out of the gate. The per-step loop below replaces
        # this with a fresh match against the evolving task_context.
        initial_matched_skills = match_skills(user_input, self.skills)
        skills_block = format_skills_block(initial_matched_skills)

        # Planning pass: produce a structured task list, or None for single-step.
        self.current_plan = self.architect.plan(
            user_input,
            memory_block=memory_block,
            skills_block=skills_block,
            history_block=history_block,
            map_block=map_block,
        )
        if self.current_plan is not None:
            yield {"type": "plan_created", "plan": self.current_plan}

        from phrases import (
            pick, ARCHITECT_THINKING, ARCHITECT_FINALIZING, AGENT_WORKING,
            ESCALATING, LOOP_DETECTED, STEP_LIMIT, PIVOT, COMPLETED,
        )

        hit_cap = False
        cap_reason = None
        for step in range(1, max_steps + 1):
            yield {"type": "status", "content": f"🦀 {pick(ARCHITECT_THINKING)}… (step {step}/{max_steps})"}

            # Skills still computed per-step because they match against task_context,
            # which grows as steps complete.
            matched_skills = match_skills(task_context, self.skills)
            skills_block = format_skills_block(matched_skills)

            self._prune_architect()
            intent = self.architect.execute(
                self.current_plan, task_context, memory_block, skills_block,
                routing_block=routing_block, history_block=history_block,
                map_block=map_block, experiences_block=exp_block,
            )
            
            # Apply plan mutations from the execute intent
            yield from self._apply_intent_to_plan(intent)

            # Yield the full intent for UI display (Reflection + Plan)
            yield {
                "type": "intent",
                "reflection": intent.get("reflection"),
                "reasoning": intent.get("reasoning"),
                "plan": intent.get("plan"),
                "agent": intent.get("recommended_agent"),
                "complete": intent.get("complete")
            }

            # Completion paths:
            #   1. Plan-driven: architect signals complete AND the plan is done.
            #   2. Single-step (no plan): trust the architect's complete=true
            #      ONLY IF the previous step actually responded. If the
            #      architect calls complete on step 1 before any agent has
            #      run, it's hallucinating — wait for at least one productive
            #      step. (Greeting loops are now caught upstream by the
            #      textual short-circuit in _short_circuit_classify, so we
            #      don't need the old after-step-1 force-complete.)
            plan_done = (
                intent.get("complete") and agent_has_responded
                and self.current_plan is not None and self.current_plan.is_complete()
            )
            single_step_done = (
                self.current_plan is None
                and intent.get("complete")
                and agent_has_responded
            )
            if single_step_done or plan_done:
                yield {"type": "status", "content": f"🦀 {pick(COMPLETED)}."}
                final_text = final_response.strip() if final_response else "Task completed."
                self._append_conversation_turn(user_input, final_text, step_history, self.current_plan)

                # PILLAR 1: Store experience
                self._summarize_experience(user_input, task_context)

                # PILLAR 2: Offer to save a skill if the turn shows
                # signs that the user educated us into success. Cheap
                # keyword precheck guards the (slower) architect call.
                if _has_learning_signal(recent_history) and step_history:
                    try:
                        draft = self._propose_skill_from_run(
                            user_input, final_text, step_history, recent_history
                        )
                    except Exception:
                        draft = None
                    if draft:
                        yield {"type": "skill_offer", "draft": draft}
                break

            agent_key = intent.get("recommended_agent", "executor")
            agent = self.agents.get(agent_key)
            if not agent:
                # The architect named an agent that doesn't exist (rare).
                # Don't halt — fall back to the executor (the most general
                # role) so the loop can keep making progress toward the goal.
                yield {
                    "type": "status",
                    "content": f"architect named unknown agent '{agent_key}' — falling back to executor",
                }
                agent_key = "executor"
                agent = self.agents.get("executor")
                if not agent:
                    yield {
                        "type": "status",
                        "content": "⚠ blocker: executor agent missing from agent map; cannot proceed.",
                    }
                    break

            plan = intent.get("plan") or intent.get("reasoning", "") or "Executing..."

            # Ensure plan is hashable (it might be a list of steps)
            plan_str = str(plan)

            # Loop detection: repetition alone is not enough — the architect
            # often re-issues the same overall plan while progress is being
            # made on sub-steps. We only count this as "stuck" when the
            # PREVIOUS step produced no tool calls and no output. The counter
            # resets on a productive step below.
            agent_plan_hash = hash((agent_key, plan_str))
            if agent_plan_hash == last_step_hash and stuck_repeats >= STUCK_LIMIT:
                if pivot_count < MAX_PIVOTS:
                    # Auto-recovery: re-analyze with a pivot nudge.
                    # Temperature rises with each pivot attempt — start at
                    # 0.7, climb to 0.9, then 1.0 — so successive retries
                    # explore a wider space rather than re-hitting the same
                    # local minimum.
                    pivot_count += 1
                    pivot_temp = min(0.6 + 0.2 * pivot_count, 1.0)
                    yield {
                        "type": "status",
                        "content": f"🦀 stuck on '{agent_key}' — {pick(PIVOT)} (attempt {pivot_count}/{MAX_PIVOTS})…",
                    }
                    hint = (
                        f"The previous plan was sent to '{agent_key}' "
                        f"{stuck_repeats + 1} times with no progress (no tool calls, "
                        f"no output). This is pivot attempt #{pivot_count} of "
                        f"{MAX_PIVOTS}. The current approach is not working — "
                        f"try a fundamentally different agent, a different "
                        f"decomposition, or a different tool path."
                    )
                    intent = self.architect.execute(
                        self.current_plan, task_context, memory_block, skills_block,
                        routing_block=routing_block, history_block=history_block,
                        map_block=map_block, experiences_block=exp_block,
                        temperature=pivot_temp, pivot_hint=hint,
                    )
                    yield from self._apply_intent_to_plan(intent)
                    yield {
                        "type": "intent",
                        "reflection": intent.get("reflection"),
                        "reasoning": intent.get("reasoning"),
                        "plan": intent.get("plan"),
                        "agent": intent.get("recommended_agent"),
                        "complete": intent.get("complete"),
                    }
                    agent_key = intent.get("recommended_agent", "executor")
                    agent = self.agents.get(agent_key)
                    if not agent:
                        # Architect named a nonexistent agent — fall back to
                        # executor rather than silently halting. The executor
                        # is the most general path and least likely to make
                        # things worse.
                        agent_key = "executor"
                        agent = self.agents.get("executor")
                        if not agent:
                            # Truly broken setup — surface as a blocker.
                            yield {
                                "type": "status",
                                "content": "⚠ blocker: executor agent missing from agent map; cannot proceed.",
                            }
                            break
                    plan = intent.get("plan") or intent.get("reasoning", "") or "Executing..."
                    plan_str = str(plan)
                    agent_plan_hash = hash((agent_key, plan_str))
                    stuck_repeats = 0
                    last_step_hash = None
                else:
                    # All pivot attempts exhausted. Frame as a blocker that
                    # needs the user's input rather than a silent give-up.
                    yield {
                        "type": "status",
                        "content": (
                            f"⚠ blocker: tried {MAX_PIVOTS} different approaches "
                            f"on '{agent_key}' and still stuck. The agent needs your "
                            f"help — please clarify the goal, simplify the request, "
                            f"or tell me which approach to retry."
                        ),
                    }
                    yield {"type": "content", "content": (
                        "\n[System: architect ran out of pivot attempts — "
                        "pausing here. Type \"continue\" (or one of: go on / keep "
                        "going / more / next) to extend this response, or send a "
                        "new prompt to start fresh.]"
                    )}
                    yield {"type": "halt", "reason": "pivot_exhausted"}
                    hit_cap = True
                    cap_reason = "pivot_exhausted"
                    break

            yield {
                "type": "reasoning",
                "content": f"[{agent_key}] {plan}\n",
            }
            yield {"type": "status", "content": f"{agent_key}: {pick(AGENT_WORKING)}…"}

            instruction = intent.get("plan") or intent.get("reasoning", "Execute the next step.")

            prev_step_summary = ""
            if step_history:
                last = step_history[-1]
                prev_step_summary = f"\n## Previous Step Result ({last['agent']})\n{last['output'][:2000]}"
                if last.get('tools'):
                    prev_step_summary += f"\nTools used: {', '.join(last['tools'][:5])}"

            history_block = f"\n## Conversation History\n{recent_history}" if recent_history else ""
            # Sub-agents now ALSO see the matched skills — without this,
            # the executor / researcher / debugger could not apply a saved
            # procedure even when the architect's routing reasoning named it.
            agent_context = (
                f"{memory_block}{skills_block}{history_block}"
                f"## Instructions\n{instruction}\n\n"
                f"## Original Request\n{user_input}{prev_step_summary}"
            )

            # The debugger's job is to hand a diagnosis to the ARCHITECT, not
            # to the user. Suppress its content chunks from the chat stream
            # (the architect still sees them via step_output_parts so it can
            # turn the Proposed Fix into new_tasks). Tool calls and status
            # still surface so the user sees that work is happening.
            suppress_user_visible = (agent_key == "debugger")

            step_output_parts = []
            step_tool_results = []
            step_tool_names = []
            for chunk in agent.chat_stream(agent_context):
                if chunk["type"] == "content":
                    step_output_parts.append(chunk["content"])
                    if suppress_user_visible:
                        # Capture-only: architect needs this in its next
                        # prompt, but the user never sees raw debugger
                        # analysis as the response to their question.
                        continue
                elif chunk["type"] == "tool_end":
                    step_tool_names.append(chunk["name"])
                    step_tool_results.append(f"  [{chunk['name']}]: {str(chunk['result'])[:500]}")
                yield chunk

            step_output = "".join(step_output_parts)
            if not step_output.strip() and step_tool_results:
                step_output = "Done."
                if not suppress_user_visible:
                    yield {"type": "content", "content": "Done."}
            if step_output.strip() and not suppress_user_visible:
                # Only update final_response from agents whose output is
                # meant for the user. Otherwise the debugger's diagnosis
                # would become the answer when the run completes.
                final_response = step_output.strip()

            if suppress_user_visible:
                # Tell the user the debugger handed off so they can see
                # something happened — without exposing the analysis.
                yield {
                    "type": "status",
                    "content": "debugger: analysis complete → architect updating plan",
                }

            agent_has_responded = bool(step_output.strip() or step_tool_results)

            # Determine if step was a success or failure
            had_error = any(w in step_output.lower() for w in ["error:", "exception:", "traceback", "failed to"]) or \
                        any("error" in r.lower() or "not found" in r.lower() for r in step_tool_results)
            
            outcome = "FAILURE" if had_error else "SUCCESS"
            
            step_record = {
                "agent": agent_key,
                "output": step_output[:3000],
                "tools": step_tool_names,
                "outcome": outcome,
            }
            step_history.append(step_record)

            step_ctx = step_output[:3000]
            if step_tool_results:
                step_ctx += "\n\nTool results:\n" + "\n".join(step_tool_results[:10])

            task_context += f"\n\n--- Step {step} ({agent_key}) [{outcome}] ---\n{step_ctx}"
            
            # Prune task_context if it's getting too long
            parts = task_context.split("\n\n--- Step ")
            if len(parts) > 5:
                task_context = parts[0] + "\n\n... (earlier steps omitted) ...\n\n--- Step " + "\n\n--- Step ".join(parts[-4:])

            step_success = bool(step_output.strip() or step_tool_results)

            # Update loop-detection state. A productive step (any tool result
            # or any output) resets the stuck counter even if the architect
            # issues the same plan next turn — that's fine, real work happened.
            made_progress = bool(step_tool_results or step_output.strip())
            if agent_plan_hash == last_step_hash and not made_progress:
                stuck_repeats += 1
            elif not made_progress:
                # New plan but still no progress — start the counter fresh.
                stuck_repeats = 1
            else:
                stuck_repeats = 0
            last_step_hash = agent_plan_hash

            self.db.store_routing_decision(
                user_input if step == 1 else task_context[:300],
                agent_key, step_success
            )

        if step >= max_steps and not hit_cap:
            # Hitting the step cap means the task is more complex than the
            # orchestrator can autonomously drive to completion. Frame as a
            # blocker, not a give-up — the user should refine the goal,
            # split the task, or confirm what's been done so far.
            done_count = (
                f"{self.current_plan.progress()[0]}/{self.current_plan.progress()[1]} tasks done; "
                if self.current_plan is not None else ""
            )
            yield {
                "type": "status",
                "content": (
                    f"⚠ blocker: reached the {max_steps}-step ceiling on this turn. "
                    f"{done_count}the task is bigger than one run can carry. "
                    f"Please review what's been done and tell me how to continue "
                    f"(or break the goal into smaller pieces)."
                ),
            }
            yield {"type": "content", "content": (
                f"\n[System: architect ran {max_steps} orchestration steps "
                f"— pausing here. Type \"continue\" (or one of: go on / keep "
                f"going / more / next) to extend this response, or send a "
                f"new prompt to start fresh.]"
            )}
            yield {"type": "halt", "reason": "iteration_cap"}
            hit_cap = True
            cap_reason = "iteration_cap"
            if final_response:
                self._append_conversation_turn(user_input, final_response, step_history, self.current_plan)

        # Finalize plan: any lingering in_progress task → skipped, emit final update
        if self.current_plan is not None:
            for t in self.current_plan.tasks:
                if t.status == "in_progress":
                    self.current_plan.advance(t.id, "skipped")
            yield {"type": "plan_update", "plan": self.current_plan}
