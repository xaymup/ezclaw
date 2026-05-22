import ollama
import os
import json
import re
from typing import List, Dict, Any, Optional, Iterator
from memory import Database
from tools import registry, create_memory_tools
from agent import load_skills, match_skills, format_skills_block
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

AGENT_DEFS = {
    "executor": {
        "model": os.getenv("OLLAMA_MODEL", "qwen3:14b"),
        "system_prompt": """You are EzClaw's **Executor Agent**. Complete tasks using tools.

Rules:
- Respond in plain natural text. NEVER output JSON.
- Use tools for actionable requests (run commands, check data, edit files, write code, fetch web content).
- For greetings or simple chat, just respond directly without tools.
- Use relative paths for file operations.
- Use web_fetch for news, web searches, documentation lookups, and online information.
- Use interactive=True ONLY for editors/REPLs/ssh (vim, python, etc).
- **Proactive fixing**: When the debugger or architect provides a fix plan, implement it immediately using write_file, run_shell, etc. Do NOT ask the user for permission — just do it.
- **File analysis**: Read files first to understand their content before making changes. Use read_file to verify the current state, then write_file to apply targeted edits.""",
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
        "system_prompt": """You are EzClaw's **Researcher Agent**. Gather and synthesize information.

Rules:
- Respond in plain natural text. NEVER output JSON.
- Use web_fetch for research tasks.
- Synthesize findings into clear summaries.""",
        "tools": [
            "web_fetch", "schedule_task",
            "list_skills", "get_skill",
            "remember", "recall", "forget",
            "delegate",
        ],
    },
    "debugger": {
        "model": os.getenv("OLLAMA_DEBUGGER_MODEL", "deepseek-r1:14b"),
        "system_prompt": """You are EzClaw's **Debugger Agent**. Analyze code, find bugs, and fix issues.

Rules:
- Respond in plain natural text. NEVER output JSON.
- Use read_file to examine code, run_shell to reproduce errors and test fixes.
- Explain root causes clearly before proposing fixes.
- Suggest minimal, targeted fixes.
- Use delegate() to request another agent execute tasks outside your scope (e.g. delegate to executor to apply a fix, or delegate to researcher to look up documentation).

Format every response with clear separation:
## 🧐 Analysis
(what you examined and found)

## 🐛 Bugs Found
(numbered list of each bug with file/line references)

## 🔧 Fix Plan
(step-by-step instructions for the executor to implement)""",
        "tools": [
            "run_shell", "read_file", "write_file", "list_dir",
            "web_fetch",
            "remember", "recall", "forget",
            "delegate",
        ],
    },
    "general": {
        "model": os.getenv("OLLAMA_GENERAL_MODEL", "qwen3.5:9b"),
        "system_prompt": """You are EzClaw's **General Agent**. Be friendly and helpful.

Rules:
- Respond in plain natural text. NEVER output JSON.
- You have no tools — just chat, answer questions, and assist.
- Be concise, friendly, and helpful.""",
        "tools": [],
    },
}


def filter_tools(tool_names: List[str]) -> List[Dict[str, Any]]:
    all_defs = registry.get_tool_definitions()
    return [d for d in all_defs if d["function"]["name"] in tool_names]


class SpecializedAgent:
    """Self-contained agent with its own model, prompt, tools, and history."""

    def __init__(self, name: str, config: dict, db: Database):
        self.name = name
        self.client = ollama.Client(host=OLLAMA_HOST)
        self.model = config["model"]
        self.system_prompt = config["system_prompt"]
        self.tools = filter_tools(config["tools"])
        self.db = db
        self.messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]
        self.options = {
            "temperature": 0.0,
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", 16384)),
            "top_p": 0.9,
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", 999)),
        }
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m")
        self.session_authorized = False

    def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
        self.messages.append({"role": "user", "content": user_input})
        if len(self.messages) > 30:
            self.messages = [self.messages[0]] + self.messages[-28:]
        last_tool_hash = None

        for _ in range(10):
            full_response, full_reasoning, tool_calls = "", "", []
            in_thinking, raw_buffer = False, ""

            try:
                stream = self.client.chat(
                    model=self.model, messages=self.messages,
                    tools=self.tools or None,
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
                yield {"type": "content", "content": "[System: No response generated.]"}
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

                self.messages.append({
                    "role": "tool", "content": str(result), "name": tool_call.function.name,
                })
                yield {"type": "tool_end", "name": tool_call.function.name, "result": str(result)}


class Architect:
    """Routes tasks to specialized agents and tracks the plan."""

    def __init__(self, db: Database):
        self.client = ollama.Client(host=OLLAMA_HOST)
        self.model = os.getenv("OLLAMA_ARCHITECT_MODEL", "phi4-reasoning:plus")
        self.db = db
        self.messages: List[Dict] = [{
            "role": "system",
            "content": """You are the Super-Architect. Coordinate specialized agents to solve requests.

Agents:
- executor: Shell commands, file operations, coding, running tools, local data. Use for ANY task that does something locally.
- researcher: Web fetching, web searches, documentation lookup, online research, information synthesis. Use when info needs to come from the internet.
- debugger: Code analysis, bug finding, logic verification, reviewing code for errors.
- general: Conversation, greetings, chitchat, opinions, Q&A. ONLY for pure talk with no action needed.

Pipeline rules:
- Context Prep → Debugger: Before debugging, route to executor FIRST to read files, run diagnostics, and prepare context. The executor can use tools — the debugger cannot.
- Debugger → Executor: After the debugger produces a fix, route to executor to apply it.
- Never mark complete after the debugger — send its fix to executor for implementation.
- **plan field**: Must contain concrete step-by-step actions the next agent should take (e.g. "1. Read file X 2. Add constant 3. Apply fix to line Y").

Decision rules (in order):
1. Does the user want info from the web (fetch, search, lookup, research, news)? → use researcher.
2. Code analysis, debugging, finding bugs? → use executor FIRST (read files, gather context). The debugger will analyze afterward.
3. After debugger output, apply its fix? → use executor.
4. Does the user want to DO something locally (check, run, install, create, edit, get, send, weather)? → use executor.
5. Is it pure conversation (hello, how are you, opinions, thanks)? → use general.
6. Otherwise: is there an action to take? If yes → executor. If no → general.
- You will receive [Known Facts] from memory and <available_skills> with procedures. Use them to decide the best plan.
- recommended_agent must be one of: executor, general, researcher, debugger.
- Set complete: true only after all work is done and no more agents need to run.
- Respond in JSON with these exact keys: category, reasoning, recommended_agent, plan, complete""",
        }]
        self.options = {
            "temperature": 0.0,
            "num_ctx": min(int(os.getenv("OLLAMA_NUM_CTX", 16384)), 8192),
            "top_p": 0.9,
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", 999)),
        }

    def analyze(self, task_context: str, memory_block: str = "", skills_block: str = "") -> Dict[str, Any]:
        prompt = f"""{task_context}
{memory_block}
{skills_block}
Based on the current state, what is the next action?
- recommended_agent must be one of: executor, general, researcher, debugger.
- If the user needs web info (fetch, search, lookup, research, news): use "researcher".
- If the user wants to DO something locally (check, run, install, create, edit): use "executor".
- If the user encountered an error, exception, bug, or needs code analysis/debugging: use "executor" FIRST to gather context (read files, run diagnostics). The debugger will analyze afterward.
- DEBUGGER→EXECUTOR PIPELINE: If the debugger just produced a fix or analysis suggesting code changes, ALWAYS route to executor next to apply that fix.
- Never set complete: true after the debugger — its output needs to be implemented by the executor.
- When routing to executor, the `plan` field MUST contain concrete step-by-step actions (e.g. "1. Read file 2. Add missing constant 3. Apply edit to line X").
- If the user just chats or asks questions with no action needed: use "general".
- ALWAYS delegate to an agent on every step. Never set complete: true unless no more work is needed.

Return valid JSON with these exact keys: category, reasoning, recommended_agent, plan, complete
{{"category": "technical|research|creative|chat", "reasoning": "why you chose this agent", "recommended_agent": "executor|general|researcher|debugger", "plan": "concrete step-by-step actions for the agent", "complete": false}}"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=self.messages + [{"role": "user", "content": prompt}],
                format="json",
            )
            content = response["message"]["content"].strip()
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                content = match.group(1)
            intent = json.loads(content)
            intent.setdefault("plan", "")
            intent.setdefault("reasoning", "")
            intent.setdefault("complete", False)
            # Ensure recommended_agent is valid; if missing or unknown, infer via embeddings
            valid_agents = {"executor", "general", "researcher", "debugger"}
            agent = intent.get("recommended_agent")
            if agent not in valid_agents:
                from embed import classify_intent
                agent = classify_intent(task_context)
                intent["recommended_agent"] = agent
            # Context-prep override: always route debugger requests to executor FIRST
            if agent == "debugger" and "--- Step" not in task_context:
                intent["recommended_agent"] = "executor"
                intent["plan"] = f"1. Read the relevant files and gather context\n2. Pass context to debugger for analysis\n\nDebug task: {task_context[:200]}"
            self.messages.append({"role": "assistant", "content": json.dumps(intent)})
            return intent
        except Exception:
            from embed import classify_intent
            agent = classify_intent(task_context)
            if agent == "debugger" and "--- Step" not in task_context:
                agent = "executor"
            return {
                "category": "technical",
                "reasoning": "Continuing execution...",
                "recommended_agent": agent,
                "plan": "",
                "complete": False,
            }


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

    # ── Orchestration ──────────────────────────────────────────

    def run(self, user_input: str) -> Iterator[Dict[str, Any]]:
        task_context = f"User Request: {user_input}"
        max_steps = 5
        loop_hashes = set()
        agent_has_responded = False
        needs_debug = any(w in user_input.lower() for w in ["debug", "bug", "error", "fix", "analyze", "crash"])

        for step in range(1, max_steps + 1):
            yield {"type": "status", "content": f"🧠 [Architect] Step {step}: Planning...\n"}

            key_facts = self.db.get_key_facts()
            memory_block = f"\n[Known Facts]: {key_facts}\n" if key_facts else ""
            matched_skills = match_skills(task_context, self.skills)
            skills_block = format_skills_block(matched_skills)

            intent = self.architect.analyze(task_context, memory_block, skills_block)

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

            step_output = ""
            step_tool_results = []
            for chunk in agent.chat_stream(
                f"{skills_block}{instruction}\n\nContext: {task_context}"
            ):
                if chunk["type"] == "content":
                    step_output += chunk["content"]
                elif chunk["type"] == "tool_end":
                    step_tool_results.append(f"  [{chunk['name']}]: {str(chunk['result'])[:500]}")
                yield chunk

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
                    agent_key = target_key

            agent_has_responded = bool(step_output.strip() or step_tool_results)

            step_context = step_output[:4000]
            if step_tool_results:
                step_context += "\n\nTool results:\n" + "\n".join(step_tool_results)

            task_context += f"\n\n--- Step {step} ({agent_key}) ---\n{step_context}"

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
