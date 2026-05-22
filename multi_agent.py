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
        "system_prompt": """You are EzClaw's **Executor** — execute instructions precisely using tools.

Rules:
- Respond in plain text. Lead with the answer, no preambles.
- The user message contains an instruction or plan — follow it step by step.
- If the instruction says read a file, do it. If it says create a skill, do it.
- Do NOT ask "how can I help" or "what would you like" — just execute.

Tool selection:
- `read_file` to examine code before editing.
- `write_file` for creating/updating files (skills, code, etc).
- `run_shell` for commands. interactive=True ONLY for vim/ssh/REPLs.
- `web_fetch` for news, docs, research.
- `remember`/`recall` for memory.
- `learn_skill` to save a reusable skill from what you just did.

Execution rules:
- Read files first to understand them, then act.
- After a tool returns, continue the plan — don't repeat the same tool.
- If the instruction says "create a new skill", call learn_skill with the extracted knowledge.
- If a tool fails, try once with adjusted input, then report.
- Keep outputs concise: show diffs, not full files; show summaries, not raw output.""",
        "tools": [
            "run_shell", "read_file", "write_file", "list_dir",
            "web_fetch", "schedule_task",
            "learn_skill", "get_skill", "list_skills",
            "remember", "recall", "forget",
            "delegate",
        ],
    },
    "researcher": {
        "model": os.getenv("OLLAMA_RESEARCHER_MODEL", "qwen3:14b"),
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

Source quality: Prefer official docs, reputable sources, recent dates. Note when info might be stale.

Memory: Use `remember` to save useful research findings for later. Use `recall` if the user references past research.""",
        "tools": [
            "web_fetch", "schedule_task",
            "list_skills", "get_skill",
            "remember", "recall", "forget",
            "delegate",
        ],
    },
    "debugger": {
        "model": os.getenv("OLLAMA_DEBUGGER_MODEL", "deepseek-r1:14b"),
        "system_prompt": """You are EzClaw's **Debugger** — find root causes, not just symptoms.

Rules:
- Respond in plain text. No JSON.
- Lead with the root cause, then the fix.

Debug methodology:
1. **Reproduce**: Run the code/command to see the error yourself.
2. **Isolate**: Read relevant files. Identify the exact line/component failing.
3. **Root cause**: What is the fundamental issue? (wrong logic, missing case, type error, race condition, API change)
4. **Fix**: Minimal, targeted change. Fix the cause, not the symptom.
5. **Verify**: Run again to confirm the fix works.

Tool use:
- `read_file` to examine code, `run_shell` to reproduce and test.
- `write_file` to apply your own fixes directly.
- `web_fetch` to look up error messages or API docs.
- `delegate` to executor if the fix is complex (multiple files) or to researcher for docs.

Format:
## Analysis
(what you examined, the flow, your reasoning)

## Root Cause
(one sentence: what, where, why)

## Fix
(exact change needed, with file path and line references)""",
        "tools": [
            "run_shell", "read_file", "write_file", "list_dir",
            "web_fetch",
            "remember", "recall", "forget",
            "delegate",
        ],
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
        "tools": [
            "remember", "recall", "forget",
        ],
    },
}


def filter_tools(tool_names: List[str]) -> List[Dict[str, Any]]:
    all_defs = registry.get_tool_definitions()
    return [d for d in all_defs if d["function"]["name"] in tool_names]


class SpecializedAgent:
    """Self-contained agent with its own model, prompt, tools, and history."""

    def __init__(self, name: str, config: dict, db: Database):
        self.name = name
        self.client = build_agent_client()
        self.model = config["model"]
        self.system_prompt = config["system_prompt"]
        self.tools = filter_tools(config["tools"])
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

    def _select_relevant_tools(self, user_input: str, top_n: int = 10) -> List[Dict[str, Any]]:
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

        selected_tools = self._select_relevant_tools(user_input, top_n=10)

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
                if len(result_str) > 8000:
                    result_str = result_str[:8000] + f"\n... (truncated, {len(result_str)} chars total)"
                self.messages.append({
                    "role": "tool", "content": result_str, "name": tool_call.function.name,
                })
                yield {"type": "tool_end", "name": tool_call.function.name, "result": str(result)}


class Architect:
    """Routes tasks to specialized agents and tracks the plan."""

    def __init__(self, db: Database):
        self.client, self.model = build_architect_client()
        self.db = db
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", 16384))
        self.use_deepseek = os.getenv("ARCHITECT_PROVIDER", "ollama") == "deepseek"
        self.messages: List[Dict] = [{
            "role": "system",
            "content": """You are the **Architect** — a planner that routes work to the right agent and tracks progress.

Agents:
- executor: Does things locally (shell, file ops, code, installs). Use for ANY action.
- researcher: Fetches web info (docs, news, research, lookups).
- debugger: Finds root causes of bugs. Give it context from executor first.
- general: Conversation only. No tools beyond memory.

Pipeline:
1. Debugging: executor (gather context) → debugger (analyze) → executor (apply fix)
2. Research: researcher (search & fetch) → executor (apply if needed)
3. Direct: executor for any action, general for chat

Rules:
- recommended_agent must be: executor, general, researcher, or debugger.
- **plan** must contain concrete steps for the next agent (e.g. "1. Read src/main.py 2. Run tests 3. Fix error on line 42").
- Set **complete: true** only when the full user request is done and no more agent runs are needed.
- Before setting complete, ask: did the user's full request get handled?
- Never set complete after debugger output — route its fix to executor first.
- Use [Known Facts] and <available_skills> to make better plans.
- Respond in JSON: category, reasoning, recommended_agent, plan, complete""",
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

    def analyze(self, task_context: str, memory_block: str = "", skills_block: str = "", experiences_block: str = "", routing_block: str = "") -> Dict[str, Any]:
        max_prompt_len = 12000
        blocks = [task_context, memory_block, skills_block, experiences_block, routing_block]
        total = sum(len(b) for b in blocks)
        if total > max_prompt_len:
            overflow = total - max_prompt_len
            if overflow > 0 and len(task_context) > overflow + 500:
                task_context = task_context[:-(overflow + 100)] + "\n... (truncated)"
        prompt = f"""{task_context}
{memory_block}
{skills_block}
{experiences_block}
{routing_block}

Based on the current state, what is the next action?

Decision guide:
- Web info (fetch, search, lookup, research, news) → researcher
- Local action (run, install, create, edit, read, check) → executor
- Error/bug/crash → executor first (gather context), then debugger
- Debugger just finished with a fix → executor to apply it
- Pure chat, greeting, opinion, Q&A with no action → general
- Memory recall or storage → executor (has memory tools)

Plan quality: When routing to executor, plans MUST be concrete steps (e.g. "1. Read src/main.py 2. Run tests 3. Fix error on line 42"). Vague plans cause failure.

Complete: Set complete: true ONLY when the original user request has been fully handled. Ask: did we actually answer the user or do what they asked?

Return JSON: category, reasoning, recommended_agent, plan, complete
{{"category": "technical|research|chat", "reasoning": "why this agent", "recommended_agent": "executor|general|researcher|debugger", "plan": "steps for the agent", "complete": false}}"""

        for attempt in range(2):
            try:
                content = self._chat(prompt if attempt == 0 else prompt + "\n\nCRITICAL: Previous response was not valid JSON. Return ONLY valid JSON with keys: category, reasoning, recommended_agent, plan, complete.")
                content = extract_json(content)
                intent = content
                intent.setdefault("plan", "")
                intent.setdefault("reasoning", "")
                intent.setdefault("complete", False)

                valid_agents = {"executor", "general", "researcher", "debugger"}
                if intent.get("recommended_agent") not in valid_agents:
                    intent["recommended_agent"] = "executor"

                if intent["recommended_agent"] == "debugger" and "--- Step" not in task_context:
                    intent["recommended_agent"] = "executor"
                    intent["plan"] = f"1. Read the relevant files and gather context\n2. Pass context to debugger for analysis"

                self.messages.append({"role": "assistant", "content": json.dumps(intent)})
                return intent
            except Exception:
                if attempt == 1:
                    return {
                        "category": "technical",
                        "reasoning": "Continuing execution...",
                        "recommended_agent": "executor",
                        "plan": "",
                        "complete": False,
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

    def clear_session_history(self):
        for a in self.agents.values():
            a.messages = [a.messages[0]]
        self.architect.messages = [self.architect.messages[0]]

    def _prune_architect(self):
        """Keep architect history bounded — system prompt + last 3 turns."""
        if len(self.architect.messages) > 7:
            self.architect.messages = (
                [self.architect.messages[0]] + self.architect.messages[-6:]
            )

    # ── Routing Enhancements ──────────────────────────────────

    ROUTING_EXAMPLES = [
        ("hello", "general"), ("hi how are you", "general"), ("thanks", "general"),
        ("goodbye", "general"), ("good morning", "general"),
        ("search the web for", "researcher"), ("look up information about", "researcher"),
        ("find documentation for", "researcher"), ("what is the latest news", "researcher"),
        ("remember that my favorite color is blue", "executor"),
        ("remember my name is John", "executor"),
        ("do you remember anything about me", "executor"),
        ("what do you know about me", "executor"),
        ("what is my favorite color", "executor"),
        ("what is my name", "executor"),
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

    # ── Orchestration ──────────────────────────────────────────

    def run(self, user_input: str) -> Iterator[Dict[str, Any]]:
        # Truncate long user input to prevent context overflow
        if len(user_input) > 4000:
            user_input = user_input[:4000] + "\n... (truncated)"
        task_context = f"User Request: {user_input}"
        max_steps = 5
        loop_hashes = set()
        agent_has_responded = False
        needs_debug = any(w in user_input.lower() for w in ["debug", "bug", "error", "fix", "analyze", "crash"])

        # Reset agent histories for fresh context each request
        for a in self.agents.values():
            a.messages = [a.messages[0]]

        # Short-circuit: for obvious patterns, skip architect entirely
        short_circuit_agent = self._short_circuit_classify(user_input)
        if short_circuit_agent and short_circuit_agent != "debugger":
            agent_key = short_circuit_agent
            agent = self.agents.get(agent_key)
            if agent:
                agent.messages = [agent.messages[0]]
                memory_facts = self.db.search_memories(user_input[:2000])
                memory_hint = f"\n[Relevant Memories]: {memory_facts}\n" if memory_facts else ""
                yield {"type": "status", "content": f"🚀 [{agent_key}] (classified)\n"}
                sc_output = ""
                sc_tool_results = []
                for chunk in agent.chat_stream(f"{memory_hint}{user_input}"):
                    if chunk["type"] == "content":
                        sc_output += chunk["content"]
                    elif chunk["type"] == "tool_end":
                        sc_tool_results.append(chunk["name"])
                    yield chunk
                if not sc_output.strip() and sc_tool_results:
                    yield {"type": "content", "content": "Done."}
                yield {"type": "status", "content": "✅ Task complete.\n"}
                self.db.store_routing_decision(user_input, agent_key, True)
                return

        for step in range(1, max_steps + 1):
            yield {"type": "status", "content": f"🧠 [Architect] Step {step}: Planning...\n"}

            # Search memory for facts relevant to the current conversation
            memory_facts = self.db.search_memories(task_context[:2000])
            memory_block = f"\n[Relevant Memories]: {memory_facts}\n" if memory_facts else ""
            matched_skills = match_skills(task_context, self.skills)
            skills_block = format_skills_block(matched_skills)

            # Routing priors: find similar past routing decisions
            routing_priors = self.db.search_similar_routing(task_context[:2000], limit=3)
            routing_block = self._format_routing_priors(routing_priors)

            self._prune_architect()
            intent = self.architect.analyze(task_context, memory_block, skills_block, routing_block=routing_block)

            # Save memory for agent context
            agent_memory_block = memory_block

            if intent.get("complete") and agent_has_responded:
                yield {"type": "status", "content": "✅ Task complete.\n"}
                break

            agent_key = intent.get("recommended_agent", "executor")
            agent = self.agents.get(agent_key)
            if not agent:
                yield {"type": "status", "content": "✅ Task complete.\n"}
                break

            plan = intent.get("plan") or intent.get("reasoning", "") or "Executing task..."
            yield {
                "type": "reasoning",
                "content": f"Plan: {plan}\nDelegating to: {agent_key} ({agent.model})\n",
            }
            yield {"type": "status", "content": f"🚀 [{agent_key}]\n"}

            instruction = intent.get("plan") or intent.get("reasoning", "Execute the next step.")
            matched_skills = match_skills(f"{instruction} {task_context}", self.skills)
            skills_block = format_skills_block(matched_skills)

            # Reset agent messages before each step to prevent cross-step accumulation
            agent.messages = [agent.messages[0]]

            step_output = ""
            step_tool_results = []
            agent_context = f"## Task\n{instruction}\n\n## Original Request\n{user_input}\n\n## Context So Far\n{task_context}"
            for chunk in agent.chat_stream(agent_context):
                if chunk["type"] == "content":
                    step_output += chunk["content"]
                elif chunk["type"] == "tool_end":
                    step_tool_results.append(f"  [{chunk['name']}]: {str(chunk['result'])[:500]}")
                yield chunk
            if not step_output.strip() and step_tool_results:
                step_output = "Done."
                yield {"type": "content", "content": "Done."}

            # Check for delegation request in tool results
            delegate_match = None
            for r in step_tool_results:
                m = re.search(r'\[DELEGATE:(\w+)\](.*?)\[/DELEGATE\]', r)
                if m:
                    delegate_match = (m.group(1), m.group(2))
                    break

            if delegate_match:
                target_key, delegate_instruction = delegate_match
                target_agent = self.agents.get(target_key)
                if target_agent:
                    yield {"type": "status", "content": f"🔄 {agent_key} delegated to {target_key}\n"}
                    task_context += f"\n\n--- Delegation: {agent_key} → {target_key} ---\n{delegate_instruction}"
                    step_output = ""
                    step_tool_results = []
                    for chunk in target_agent.chat_stream(
                        f"{skills_block}{delegate_instruction}\n\nContext: {task_context}"
                    ):
                        if chunk["type"] == "content":
                            step_output += chunk["content"]
                        elif chunk["type"] == "tool_end":
                            step_tool_results.append(f"  [{chunk['name']}]: {str(chunk['result'])[:500]}")
                        yield chunk
                    if not step_output.strip() and step_tool_results:
                        step_output = "Done."
                        yield {"type": "content", "content": "Done."}
                    agent_key = target_key

            agent_has_responded = bool(step_output.strip() or step_tool_results)

            step_context = step_output[:4000]
            if step_tool_results:
                step_context += "\n\nTool results:\n" + "\n".join(step_tool_results)

            task_context += f"\n\n--- Step {step} ({agent_key}) ---\n{step_context}"
            # Keep only last 3 steps to avoid context overflow
            parts = task_context.split("\n\n--- Step ")
            if len(parts) > 4:
                task_context = parts[0] + "\n\n--- Step " + "\n\n--- Step ".join(parts[-3:])

            step_decision_text = user_input if step == 1 else task_context[:300]
            step_success = bool(step_output.strip() or step_tool_results)
            self.db.store_routing_decision(step_decision_text, agent_key, step_success)

            is_debugger_output = agent_key == "debugger" and step_output.strip()
            is_error = any(w in step_context.lower() for w in ["error", "exception", "traceback", "failed", "exit code", "not found"])
            debugger_has_run = "--- Step" in task_context and "debugger" in task_context[task_context.rfind("--- Step"):]
            is_context_prep = needs_debug and not debugger_has_run and agent_key == "executor" and step_output.strip()
            if step_output.strip() or step_tool_results:
                if is_error and agent_key != "debugger":
                    yield {"type": "status", "content": "⚠️ Error detected. Re-routing to debugger.\n"}
                    continue
                if is_debugger_output:
                    yield {"type": "status", "content": "🔧 Debugger produced a fix. Routing to executor to apply...\n"}
                    continue
                if is_context_prep:
                    yield {"type": "status", "content": "📄 Context prepared. Routing to debugger for analysis...\n"}
                    continue
                yield {"type": "status", "content": "✅ Task complete.\n"}
                break

            decision_hash = hash(agent_key)
            if decision_hash in loop_hashes:
                yield {"type": "status", "content": "⚠️ Loop detected. Stopping.\n"}
                break
            loop_hashes.add(decision_hash)

        if step >= max_steps:
            yield {"type": "status", "content": "⚠️ Orchestration limit reached.\n"}
