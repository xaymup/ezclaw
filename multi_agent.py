import ollama
import os
import json
import re
from typing import List, Dict, Any, Optional, Iterator
from memory import Database
from tools import registry, create_memory_tools
from agent import load_skills, match_skills, format_skills_block
from embed import embed, cosine_similarity, classify_by_similarity
from model_client import build_architect_client, build_agent_client, extract_json
from dotenv import load_dotenv

load_dotenv()

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
- **Workspace Sandbox (IMPORTANT)**: All `read_file`, `write_file`, and `list_dir` paths are RESOLVED RELATIVE to the `workspace/` directory. So `read_file("foo.py")` reads `workspace/foo.py`. You CANNOT read files outside `workspace/` (e.g. project source like `agent.py` is NOT accessible — if asked about those, say so and use `run_shell` with `cat` if absolutely needed). Pass paths as relative names ("foo.py"), not as `workspace/foo.py` — that would resolve to `workspace/workspace/foo.py`. Use `list_dir(".")` to see what's in the workspace root.
- **Don't pre-validate**: Just call `read_file(path)` — if it errors, then act. Do NOT `list_dir` first to "check if the file exists."
- **Verification First**: Before modifying or copying a file, verify its existence and content to avoid redundant work.
- After each tool result, proceed to the NEXT step. Do NOT repeat a step unless it failed and you have a new approach.
- Do NOT ask questions. Do NOT say "how can I help". Just execute.
- If a step fails, retry once with adjusted input, then report and move on.
- When all steps are done, provide a comprehensive summary of what was accomplished and the final state of the task.

## Output
- **Lead with the answer.** First line states the result ("Wrote add.py with the add(a,b) function.", "Tests pass: 8/8.", "Found 3 matches: ..."). No preamble, no "In this task I will..." narration.
- For multi-step work, follow the lead line with a short bulleted recap of what each step did. One line per step. Skip steps that did nothing notable.
- Show diffs for edits, key lines for command output, summaries for long output. Do NOT paste entire tool outputs back to the user — the TUI already shows tool panels.
- If something failed, say so directly on the lead line ("Could not X because Y") and stop — don't dress up failures.""",
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
        "system_prompt": """You are EzClaw's **Debugger** — find root causes, not just symptoms.

Rules:
- Respond in plain text. No JSON.
- Provide a clear, actionable fix that an Executor can apply.

Debug methodology:
1. **Reproduce**: Run the code/command to see the error yourself.
2. **Isolate**: Read relevant files. Identify the exact line/component failing.
3. **Root cause**: What is the fundamental issue?
4. **Fix**: Provide a minimal, targeted, and VERIFIABLE fix.

## Output Format:
## Analysis
(What you examined, the flow, your reasoning)

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
- If recall returns nothing relevant, say so directly — don't fabricate.""",
    },
}


class SpecializedAgent:
    """Self-contained agent with its own model, prompt, tools, and history."""

    def __init__(self, name: str, config: dict, db: Database):
        self.name = name
        self.client = build_agent_client()
        self.model = config["model"]
        self.system_prompt = config["system_prompt"]
        # Give all agents access to all registered tools
        self.tools = registry.get_tool_definitions()
        self.db = db
        self.messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", 16384))
        self.options = {
            "temperature": 0.0,
            "num_ctx": self.num_ctx,
            "top_p": 0.9,
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", 999)),
        }
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m")
        self.session_authorized = False
        self._pre_embed_tools()

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
                yield {"type": "content", "content": "\n[Loop detected. Stopping.]"}
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
            "content": """You are the **Architect** — a senior systems designer and the COMMUNICATIONS HUB for EzClaw. Your job is to orchestrate a seamless workflow between specialized agents using advanced **Chain-of-Thought (CoT)** reasoning.

## Your Critical Thinking Protocol:
Before deciding on an action, you must perform a mandatory reflection:
1. **Goal Analysis**: What is the ultimate objective? Are we closer to it than in the previous step?
2. **Observation**: What EXACTLY happened in the latest step? Did it return [SUCCESS] or [FAILURE]? What were the tool results?
3. **Critical Pivot**: Is the current agent or plan working? If we see [FAILURE] or repetition, why is it happening, and how must the strategy change?
4. **Verification**: How will we know the final result is actually correct?

## Routing Protocol:
- **executor**: File edits, shell commands, code implementation, memory management, verification runs.
- **researcher**: Web searching, documentation gathering.
- **debugger**: Root-cause analysis. Only route here for UNEXPECTED errors.
- **general**: Conversational responses.

## Handoff Protocol:
- YOU handle all handoffs. Summarize the findings of the previous agent for the next one.
- If the current agent failed ([FAILURE]), diagnose the cause. Route to `debugger` if needed, or to `executor` with a REFINED strategy.

## Completion Rules:
- Set `complete:true` ONLY when the user's FULL original intent is satisfied AND verified.

## Response Format:
Return ONLY valid JSON:
{
  "reflection": {
    "goal": "Current objective",
    "observation": "What was learned in the last step",
    "critical_thinking": "Analysis of progress and why the next action is chosen"
  },
  "category": "technical|research|chat",
  "reasoning": "Internal logic for this routing choice",
  "recommended_agent": "executor|general|researcher|debugger",
  "plan": "Numbered steps for the agent",
  "pivot_reasoning": "If strategy changed",
  "complete": false
}""",
        }]

    def _prune_messages(self):
        """Keep architect history bounded — system prompt + last 3 turns."""
        if len(self.messages) > 7:
            self.messages = [self.messages[0]] + self.messages[-6:]

    def _chat(self, prompt: str) -> str:
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
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        resp = self.client.chat(
            model=self.model,
            messages=self.messages + [{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0, "num_ctx": self.num_ctx},
        )
        return resp["message"]["content"].strip()

    def analyze(self, task_context: str, memory_block: str = "", skills_block: str = "", experiences_block: str = "", routing_block: str = "", history_block: str = "", map_block: str = "") -> Dict[str, Any]:
        max_prompt_len = 12000
        blocks = [task_context, memory_block, skills_block, experiences_block, routing_block, history_block, map_block]
        total = sum(len(b) for b in blocks)
        
        if total > max_prompt_len:
            overflow = total - max_prompt_len
            if len(task_context) > overflow + 1000:
                parts = task_context.split("\n\n--- Step ")
                if len(parts) > 3:
                     task_context = parts[0] + "\n\n... (earlier steps truncated) ...\n\n--- Step " + "\n\n--- Step ".join(parts[-2:])
                else:
                     task_context = task_context[:1000] + "\n... (middle truncated) ...\n" + task_context[-(len(task_context)-overflow-1500):]

        # Improved error detection: look for [FAILURE] tag in the LATEST step
        latest_step_split = task_context.split("--- Step")
        latest_step_content = latest_step_split[-1] if len(latest_step_split) > 1 else task_context
        has_failure = "[FAILURE]" in latest_step_content
        debugger_ran = "(debugger)" in latest_step_content

        situation = "initial"
        if debugger_ran and not has_failure:
            situation = "debugger_finished_fix"
        elif has_failure:
            situation = "error_detected"
        elif "--- Step" in task_context:
            situation = "mid_pipeline"

        prompt = f"""## Task State: {situation}

## Current Task Overview
{task_context}

## Task History & State
{history_block}

## Auxiliary Context (For Reference Only)
### Codebase Map
{map_block}
### Past Experiences
{experiences_block}
{memory_block}{skills_block}{routing_block}

## Decision Required
Analyze the CURRENT state using the Critical Thinking Protocol. Decide the NEXT action.

Return ONLY JSON: 
{{
  "reflection": {{"goal": "...", "observation": "...", "critical_thinking": "..."}},
  "category": "technical|research|chat",
  "reasoning": "...",
  "recommended_agent": "executor|general|researcher|debugger",
  "plan": "numbered steps",
  "pivot_reasoning": "if needed",
  "complete": false
}}"""

        for attempt in range(2):
            try:
                content = self._chat(prompt if attempt == 0 else prompt + "\n\nCRITICAL: Return ONLY valid JSON.")
                content = extract_json(content)
                intent = content
                intent.setdefault("plan", "")
                intent.setdefault("reasoning", "")
                intent.setdefault("complete", False)
                intent.setdefault("reflection", {})

                valid_agents = {"executor", "general", "researcher", "debugger"}
                if intent.get("recommended_agent") not in valid_agents:
                    intent["recommended_agent"] = "executor"

                self.messages.append({"role": "assistant", "content": json.dumps(intent)})
                return intent
            except Exception:
                if attempt == 1:
                    return {
                        "category": "technical",
                        "reasoning": "Fallback routing",
                        "recommended_agent": "executor",
                        "plan": "Continue with the task.",
                        "complete": False,
                        "reflection": {"critical_thinking": "Parsing failed, falling back to execution."}
                    }
                continue


class MultiAgentSystem:
    def __init__(self, session_id: Optional[int] = None):
        self.db = Database(os.getenv("DATABASE_PATH", "ezclaw.db"))
        create_memory_tools(self.db)
        self.skills = load_skills()
        self.architect = Architect(self.db)
        self.agents = {
            name: SpecializedAgent(name, cfg, self.db)
            for name, cfg in AGENT_DEFS.items()
        }
        self._conversation_history: List[Dict[str, str]] = []

    def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
        yield from self.run(user_input)

    # ── Backwards-compat properties for CLI ──────────────────────

    @property
    def session_authorized(self):
        return any(a.session_authorized for a in self.agents.values())

    @session_authorized.setter
    def session_authorized(self, value):
        for a in self.agents.values():
            a.session_authorized = value

    @property
    def model(self):
        return ", ".join(f"{n}: {a.model}" for n, a in self.agents.items())

    @property
    def messages(self):
        return self.architect.messages

    def _format_history(self) -> str:
        if not self._conversation_history:
            return ""
        lines = ["## Conversation History"]
        for turn in self._conversation_history[-5:]:
            lines.append(f"- User: {turn['user']}")
            if turn.get('assistant'):
                # Truncate long responses but keep the core meaning
                resp = turn['assistant']
                if len(resp) > 500:
                    resp = resp[:400] + "... [truncated]"
                lines.append(f"  Assistant: {resp}")
        return "\n".join(lines) + "\n"

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

    def _short_circuit_classify(self, user_input: str) -> Optional[str]:
        examples = [ex for ex, _ in self.ROUTING_EXAMPLES]
        labels = [lb for _, lb in self.ROUTING_EXAMPLES]
        return classify_by_similarity(user_input, examples, labels, threshold=0.6)

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
        max_steps = 15
        loop_hashes = set()
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
        if short_circuit_agent and short_circuit_agent != "debugger":
            agent_key = short_circuit_agent
            agent = self.agents.get(agent_key)
            if agent:
                agent.messages = [agent.messages[0]]
                memory_facts = self.db.search_memories_hybrid(user_input[:1000], alpha=0.6)
                memory_hint = f"\n[Memory]: {memory_facts}\n" if memory_facts else ""
                yield {"type": "status", "content": f"[{agent_key}] (fast-routed)\n"}
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
                    yield {"type": "status", "content": f"\n[fast-route returned uncertain result; escalating to architect]\n"}
                    self.db.store_routing_decision(user_input, agent_key, False)
                    agent.messages = [agent.messages[0]]
                    task_context = f"User Request: {user_input}\n\n[FAILURE] Fast-route to '{agent_key}' returned: {sc_output.strip()[:500]}"
                    # Fall through to the architect loop below.
                else:
                    yield {"type": "status", "content": "Done.\n"}
                    self.db.store_routing_decision(user_input, agent_key, True)
                    self._conversation_history.append({"user": user_input, "assistant": sc_output.strip()})
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

        for step in range(1, max_steps + 1):
            yield {"type": "status", "content": f"Architect: Analyzing task state (Step {step}/{max_steps})..."}

            # Skills still computed per-step because they match against task_context,
            # which grows as steps complete.
            matched_skills = match_skills(task_context, self.skills)
            skills_block = format_skills_block(matched_skills)

            self._prune_architect()
            intent = self.architect.analyze(task_context, memory_block, skills_block, routing_block=routing_block, history_block=history_block, map_block=map_block, experiences_block=exp_block)
            
            # Yield the full intent for UI display (Reflection + Plan)
            yield {
                "type": "intent",
                "reflection": intent.get("reflection"),
                "reasoning": intent.get("reasoning"),
                "plan": intent.get("plan"),
                "agent": intent.get("recommended_agent"),
                "complete": intent.get("complete")
            }

            if intent.get("complete") and agent_has_responded:
                yield {"type": "status", "content": "Task completed successfully."}
                final_text = final_response.strip() if final_response else "Task completed."
                self._conversation_history.append({"user": user_input, "assistant": final_text})
                
                # PILLAR 1: Store experience
                self._summarize_experience(user_input, task_context)
                break

            agent_key = intent.get("recommended_agent", "executor")
            agent = self.agents.get(agent_key)
            if not agent:
                yield {"type": "status", "content": "Architect: Finalizing response..."}
                self._conversation_history.append({"user": user_input, "assistant": final_response.strip()})
                break

            plan = intent.get("plan") or intent.get("reasoning", "") or "Executing..."
            
            # Ensure plan is hashable (it might be a list of steps)
            plan_str = str(plan)
            
            # IMPROVED LOOP DETECTION: Hash (agent_key, plan)
            # This detects if the architect is stuck sending the same agent the same plan.
            agent_plan_hash = hash((agent_key, plan_str))
            if agent_plan_hash in loop_hashes:
                yield {"type": "status", "content": "System: Loop detected, stopping execution."}
                break
            loop_hashes.add(agent_plan_hash)

            yield {
                "type": "reasoning",
                "content": f"[{agent_key}] {plan}\n",
            }
            if intent.get("pivot_reasoning"):
                 yield {"type": "reasoning", "content": f"[Pivot] {intent['pivot_reasoning']}\n"}

            yield {"type": "status", "content": f"{agent_key.capitalize()}: Working..."}

            instruction = intent.get("plan") or intent.get("reasoning", "Execute the next step.")

            prev_step_summary = ""
            if step_history:
                last = step_history[-1]
                prev_step_summary = f"\n## Previous Step Result ({last['agent']})\n{last['output'][:2000]}"
                if last.get('tools'):
                    prev_step_summary += f"\nTools used: {', '.join(last['tools'][:5])}"

            history_block = f"\n## Conversation History\n{recent_history}\n" if recent_history else ""
            agent_context = f"{memory_block}{history_block}## Instructions\n{instruction}\n\n## Original Request\n{user_input}{prev_step_summary}"

            step_output_parts = []
            step_tool_results = []
            step_tool_names = []
            for chunk in agent.chat_stream(agent_context):
                if chunk["type"] == "content":
                    step_output_parts.append(chunk["content"])
                elif chunk["type"] == "tool_end":
                    step_tool_names.append(chunk["name"])
                    step_tool_results.append(f"  [{chunk['name']}]: {str(chunk['result'])[:500]}")
                yield chunk

            step_output = "".join(step_output_parts)
            if not step_output.strip() and step_tool_results:
                step_output = "Done."
                yield {"type": "content", "content": "Done."}
            if step_output.strip():
                final_response = step_output.strip()

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
            self.db.store_routing_decision(
                user_input if step == 1 else task_context[:300],
                agent_key, step_success
            )

        if step >= max_steps:
            yield {"type": "status", "content": "Step limit reached.\n"}
            if final_response:
                self._conversation_history.append({"user": user_input, "assistant": final_response})
