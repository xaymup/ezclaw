import ollama
import os
import json
import pickle
import re
from typing import List, Dict, Any, Optional, Callable, Iterator
from memory import Database
from tools import registry, create_memory_tools
from tools import MUTATING_TOOLS, create_action_tracking_tools, set_session_context
from action_tracking import classify_outcome, extract_why, summarize_action
from embed import embed, cosine_similarity
from model_client import build_agent_client
from dotenv import load_dotenv

load_dotenv()

# Canonical skills location — import from tools so the loader and the
# learn_skill / get_skill / list_skills tools all read/write the same dir.
# Before this import, load_skills() was reading ./skills/ (legacy, empty
# for users who only ever stored skills via the new tools), so the
# matcher had no skills to consider for prompt injection.
from tools import SKILLS_DIR, _migrate_legacy_skills_once

def load_skills() -> List[Dict[str, str]]:
    """Load all skills from the canonical ~/.ezclaw/skills/ directory.
    Migrates any legacy ./skills/*.md from before the directory move."""
    _migrate_legacy_skills_once()
    skills = []
    if not os.path.exists(SKILLS_DIR):
        return skills
    for filename in os.listdir(SKILLS_DIR):
        if filename.endswith('.md'):
            filepath = os.path.join(SKILLS_DIR, filename)
            with open(filepath, 'r') as f:
                content = f.read()
            # Parse skill name from first line or filename
            first_line = content.split('\n')[0]
            name_match = re.match(r'# Skill:\s*(.+)', first_line)
            name = name_match.group(1).strip() if name_match else filename.replace('.md', '')
            skills.append({'name': name, 'content': content, 'filename': filename})
    return skills

def match_skills(user_input: str, skills: List[Dict[str, str]], top_n: int = 5, threshold: float = 0.4) -> List[Dict[str, str]]:
    """Find relevant skills using embedding similarity, fallback to keyword
    matching. Returns skills whose embedding similarity to the query
    exceeds `threshold`, capped at `top_n` and sorted by relevance.

    The threshold matters: previously this returned the top-N regardless
    of relevance, so an unrelated query like "is blex_os done?" still
    pulled in skills like `create_image_generation_skill` and
    `spotify_cli_control` because they share common English words. The
    architect prompt then told the model to "use available skills" — and
    the model dutifully tried to apply irrelevant procedures. With a
    similarity floor, the skills_block is non-empty ONLY when there's a
    genuinely related skill to surface."""
    if not skills:
        return []
    from embed import rank_by_similarity
    skill_texts = [f"{s['name']}: {s['content'][:500]}" for s in skills]
    ranked_texts = rank_by_similarity(user_input, skill_texts, top_n=top_n, threshold=threshold)
    matched_names = []
    for rt in ranked_texts:
        name = rt.split(":")[0]
        if name not in matched_names:
            matched_names.append(name)

    # Fallback: tight keyword match if embeddings returned nothing —
    # requires a 6+ char content word to appear in the user input. The
    # old fallback used 4+ chars which matched too much noise.
    if not matched_names:
        user_lower = user_input.lower()
        for skill in skills:
            skill_lower = skill['content'].lower()
            skill_words = set(re.findall(r'\b[a-z]{6,}\b', skill_lower))
            if any(word in user_lower for word in skill_words):
                matched_names.append(skill['name'])

    by_name = {s['name']: s for s in skills}
    return [by_name[n] for n in matched_names if n in by_name][:top_n]

def format_skills_block(skills: List[Dict[str, str]]) -> str:
    """Format matched skills as DIRECTIVE procedures.

    The previous wrapper (`<available_skills>` … `</available_skills>`)
    read as background context the model often ignored. The new wrapper
    is imperative: it tells the model these are proven procedures that
    take precedence over a freshly-invented approach. The model still
    has discretion when a skill genuinely doesn't fit, but the default
    is to follow."""
    if not skills:
        return ""
    block = (
        "\n## ⚑ MATCHED SKILLS — APPLY THESE PROCEDURES (do NOT re-derive)\n"
        "These are saved, verified procedures that match the current\n"
        "request. Follow them step-by-step. If you choose NOT to follow\n"
        "one, name the skill and state why in `reasoning`.\n"
    )
    for skill in skills:
        block += f"\n### Skill: {skill['name']}\n{skill['content']}\n"
    block += "\n## (end matched skills)\n"
    return block

class ChatAgent:
    def __init__(self, session_id: Optional[int] = None):
        self.client = build_agent_client()
        self.db = Database(os.getenv("DATABASE_PATH", "ezclaw.db"))
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
        
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m")
        
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", 32768))
        self.options = {
            "temperature": 0.0,
            "num_ctx": self.num_ctx,
            "top_p": 0.9,
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", 999)),
        }
        
        create_memory_tools(self.db)
        create_action_tracking_tools(self.db)
        self.tools = registry.get_tool_definitions()
        self.skills = load_skills()
        self.skill_descriptions = "\n".join([f"- {s['name']}: {s['content'].split('## Description')[1].split('##')[0].strip() if '## Description' in s['content'] else 'No description'}" for s in self.skills])
        self._pre_embed_tools()
        
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = self.db.get_last_session_id() or self.db.create_session()

        set_session_context(self.session_id)
        self.messages = self.db.get_messages(self.session_id)
        self.system_prompt = self._load_system_prompt()
        self.system_prompt += (
            "\n\n## Past actions\n"
            "When the user asks what you did about a past task, file, bug, or "
            "feature, call `recall_actions(query)` BEFORE answering. It is "
            "authoritative for this session's mutating actions."
        )
        self._ensure_system_message()
        self.session_authorized = False

    def _load_system_prompt(self) -> str:
        if os.path.exists("agents.md"):
            with open("agents.md", "r") as f: return f.read()
        return "You are EzClaw, a powerful assistant."

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

    def _ensure_system_message(self):
        if not self.messages or self.messages[0]["role"] != "system":
            system_msg = {"role": "system", "content": self.system_prompt}
            if not self.messages: self.messages.append(system_msg)
            else: self.messages[0] = system_msg
            self.db.add_message(self.session_id, "system", self.system_prompt)

    _MEMORY_PATTERNS = re.compile(
        r'\b(my|i\'m|i am|i have|i live|i work|i prefer|i like|remember|'
        r'recall|what do you know|do you remember|what\'s my|where do i|'
        r'who am i|my name|my email|my location|my project|my favorite|'
        r'last time|previously|you told me|i told you|we discussed)\b',
        re.IGNORECASE
    )

    _STORE_PATTERNS = re.compile(
        r'\b(my name is|i live in|i\'m from|i work at|i prefer|my favorite|'
        r'remember that|i am \d+|my birthday|my email is|i use|my project)\b',
        re.IGNORECASE
    )

    def _process_intent(self, user_input: str) -> Dict[str, Any]:
        intent_prompt = f"""Analyze the user's request and determine:
1. Does it require retrieving stored memory/context? (personal info, past discussions, preferences, etc.)
2. If yes, what search query should be used to retrieve relevant memories?
3. Does it contain facts that should be stored? (user stating personal information)
4. Which skill (if any) is relevant?

Available skills:
{self.skill_descriptions if self.skills else "None"}

User request: "{user_input}"

Respond with JSON only:
{{
  "needs_memory": true/false,
  "memory_query": "query to search memories" or null,
  "store_facts": ["fact to store"] or [],
  "relevant_skill": "skill_name" or null
}}"""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": intent_prompt}],
                options={"temperature": 0.0, "num_ctx": 4096},
                keep_alive=self.keep_alive,
                stream=False
            )
            
            content = response.message.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            intent = json.loads(content)
            
            return {
                "memory": intent.get("needs_memory", False),
                "memory_q": intent.get("memory_query"),
                "facts": intent.get("store_facts", []),
                "skill": intent.get("relevant_skill"),
            }
        except Exception as e:
            lower = user_input.lower()
            needs_memory = bool(self._MEMORY_PATTERNS.search(user_input))
            
            memory_q = None
            if needs_memory:
                stripped = re.sub(r'\b(what|where|who|how|when|do|does|is|are|can|could|you|please|tell|me|about)\b', '', lower)
                stripped = re.sub(r'\s+', ' ', stripped).strip()
                memory_q = stripped if len(stripped) > 3 else user_input

            facts = []
            if self._STORE_PATTERNS.search(user_input):
                facts = [user_input.strip()]

            skill = None
            if self.skills:
                matched = match_skills(user_input, self.skills)
                if matched:
                    skill = matched[0]['name']

            return {
                "memory": needs_memory,
                "memory_q": memory_q,
                "facts": facts,
                "skill": skill,
            }

    @staticmethod
    def _consume_buffer(raw_buffer: str, in_thinking: bool):
        _OPEN_TAG = re.compile(r'<(think|thought|reasoning)\b[^>]*>', re.IGNORECASE)
        _CLOSE_TAG = re.compile(r'</(think|thought|reasoning)\s*>', re.IGNORECASE)
        _OPEN_TAGS = ("<think", "<thought", "<reasoning")
        _CLOSE_TAGS = ("</think", "</thought", "</reasoning")

        while raw_buffer:
            if not in_thinking:
                match = _OPEN_TAG.search(raw_buffer)
                if match:
                    yield raw_buffer[:match.start()], (raw_buffer[match.end():], True)
                    raw_buffer = raw_buffer[match.end():]
                    in_thinking = True
                    continue
                idx = raw_buffer.rfind('<')
                if idx >= 0 and any(t.startswith(raw_buffer[idx:].lower()) for t in _OPEN_TAGS):
                    yield raw_buffer[:idx], (raw_buffer[idx:], False)
                    return
                yield raw_buffer, ("", False)
                return
            else:
                match = _CLOSE_TAG.search(raw_buffer)
                if match:
                    yield raw_buffer[:match.start()], (raw_buffer[match.end():], False)
                    raw_buffer = raw_buffer[match.end():]
                    in_thinking = False
                    continue
                idx = raw_buffer.rfind('</')
                if idx >= 0 and any(t.startswith(raw_buffer[idx:].lower()) for t in _CLOSE_TAGS):
                    yield raw_buffer[:idx], (raw_buffer[idx:], True)
                    return
                yield raw_buffer, ("", True)
                return

    def _record_action(self, tool_name: str, args: dict, result: str, assistant_text: str) -> None:
        """Record one mutating tool call to the actions table.

        Best-effort: any failure inside this method is logged to stderr
        and swallowed — never propagates out and never blocks the tool.
        """
        try:
            from embed import embed as _embed
            summary = summarize_action(tool_name, args)
            why = extract_why(assistant_text)
            outcome, error_excerpt = classify_outcome(result)
            embed_text = f"{summary} {why or ''}".strip()
            try:
                vec = _embed(embed_text)
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

    def clear_session_history(self):
        """Restores the chat history to just the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
        # Truncate long user input to prevent context overflow
        if len(user_input) > 4000:
            user_input = user_input[:4000] + "\n... (truncated)"

        # 1. Consolidated intent analysis with full agent context
        intent = self._process_intent(user_input)

        # 2. Process extracted facts immediately
        for fact in intent.get("facts", []):
            if fact:
                self.db.add_memory(fact)
                yield {"type": "memory_stored", "fact": fact}

        memory_block = ""
        if intent.get("memory"):
            query = intent.get("memory_q") or user_input
            memories = self.db.search_memories_hybrid(query, alpha=0.6, threshold=0.2)
            if memories:
                memory_block = f"\n[Memory]:\n" + "\n".join([f"- {m}" for m in memories]) + "\n"
                yield {"type": "context_augmented", "memories": memories}

        # 4. Skill matching
        skill_name = intent.get("skill")
        matched_skills = []
        if skill_name:
            matched_skills = [s for s in self.skills if s['name'].lower() == skill_name.lower()]
        
        if not matched_skills:
            matched_skills = match_skills(user_input, self.skills)
        
        skills_block = format_skills_block(matched_skills)

        # 5. History & Augmentation
        self.messages.append({"role": "user", "content": user_input})
        self.db.add_message(self.session_id, "user", user_input)
        
        augment_prefix = ""
        if memory_block or skills_block:
            augment_prefix = f"{memory_block}{skills_block}\n---\n"

        iteration_count = 0
        # Generous default cap for extensive research / multi-fetch chains.
        # Override with EZCLAW_MAX_ITERATIONS if you need more (or less).
        max_iterations = int(os.getenv("EZCLAW_MAX_ITERATIONS", "50"))
        last_tool_hash = None
        repeat_count = 0
        # Allow up to this many *consecutive* identical tool batches before
        # halting. Common legit-repeat patterns:
        #   - re-reading a file after a write to verify the change
        #   - re-running `make` after a fix
        #   - retrying a web_fetch through transient server timeouts
        REPEAT_LIMIT = 5

        # URL-specific loop detector for web_fetch: tracks the last 10
        # fetched URLs in a sliding window. If any URL appears 3+ times
        # in that window, the model is fetching the same page repeatedly
        # — likely confused about results — and we halt. Catches both
        # consecutive (A, A, A) and interleaved (A, B, A, B, A) patterns
        # while leaving genuinely diverse research uninterrupted.
        from collections import deque, Counter
        recent_fetched_urls = deque(maxlen=10)
        URL_REPEAT_LIMIT = 3

        # Tool pre-selection: only pass tools relevant to the current query
        selected_tools = self._select_relevant_tools(user_input, top_n=20) if len(self.tools) > 20 else self.tools

        while iteration_count < max_iterations:
            iteration_count += 1
            response_parts, reasoning_parts, tool_calls = [], [], []
            in_thinking, raw_buffer = False, ""

            # 6. Context Retention logic
            if len(self.messages) > 20:
                context_messages = [self.messages[0], self.messages[1]] + self.messages[-18:]
            else:
                context_messages = [m.copy() for m in self.messages]

            if augment_prefix:
                for i in range(len(context_messages) - 1, -1, -1):
                    if context_messages[i]["role"] == "user":
                        if not context_messages[i]["content"].startswith(augment_prefix):
                            context_messages[i]["content"] = augment_prefix + context_messages[i]["content"]
                        break

            # Total content length pruning: keep within ~75% of num_ctx (chars ≈ tokens * 3)
            max_chars = int(self.num_ctx * 2.5)
            total_chars = sum(len(m.get("content", "")) for m in context_messages)
            if total_chars > max_chars:
                # Prune oldest messages (keep system + last N within limit)
                while total_chars > max_chars and len(context_messages) > 6:
                    removed = context_messages.pop(1)  # remove oldest non-system
                    total_chars -= len(removed.get("content", ""))
                # If still over, truncate individual message content
                if total_chars > max_chars:
                    for m in context_messages:
                        if m["role"] == "system":
                            continue
                        if len(m.get("content", "")) > 2000:
                            m["content"] = m["content"][:2000] + "\n... (truncated)"

            try:
                stream = self.client.chat(model=self.model, messages=context_messages, tools=selected_tools, options=self.options, keep_alive=self.keep_alive, stream=True)
                # Test first chunk to see if it works
                first_chunk = next(stream)
            except Exception as e:
                # If tool calling is not supported, fallback to standard chat
                if "does not support tools" in str(e).lower() or "400" in str(e):
                    stream = self.client.chat(model=self.model, messages=context_messages, options=self.options, keep_alive=self.keep_alive, stream=True)
                    first_chunk = next(stream)
                else:
                    raise e

            def stream_with_first(s, f):
                yield f
                for c in s:
                    yield c

            for chunk in stream_with_first(stream, first_chunk):
                reasoning = getattr(chunk.message, 'reasoning', None) or (chunk.message.get('reasoning') if isinstance(chunk.message, dict) else None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield {"type": "reasoning", "content": reasoning}

                if chunk.message.content:
                    raw_buffer += chunk.message.content
                    for emitted, target in self._consume_buffer(raw_buffer, in_thinking):
                        if emitted:
                            if in_thinking:
                                reasoning_parts.append(emitted)
                            else:
                                response_parts.append(emitted)
                            yield {"type": "reasoning" if in_thinking else "content", "content": emitted}
                        raw_buffer, in_thinking = target

                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            if raw_buffer:
                target = reasoning_parts if in_thinking else response_parts
                target.append(raw_buffer)
                yield {"type": "reasoning" if in_thinking else "content", "content": raw_buffer}
                raw_buffer = ""

            full_response = "".join(response_parts)
            full_reasoning = "".join(reasoning_parts)

            current_hash = hash(str([(t.function.name, t.function.arguments) for t in tool_calls]))
            if tool_calls and current_hash == last_tool_hash:
                repeat_count += 1
                # Soft warning around the middle of the window — lets the
                # model see the loop signal and (often) self-correct by
                # trying a different argument before we hard-halt.
                if repeat_count == REPEAT_LIMIT - 2:
                    yield {
                        "type": "content",
                        "content": (
                            f"\n[Note: same tool call repeated {repeat_count + 1}× — "
                            f"if this is intentional retry, continue; otherwise try a "
                            f"different approach.]"
                        ),
                    }
                if repeat_count >= REPEAT_LIMIT:
                    yield {
                        "type": "content",
                        "content": f"\n[System: same tool call repeated {repeat_count + 1}× — stopping.]",
                    }
                    break
            else:
                repeat_count = 0
            last_tool_hash = current_hash

            # URL-specific loop detector for web_fetch — diverse research
            # with many DIFFERENT URLs slides through this freely, but if
            # the model fetches the SAME URL 3+ times within the last 10
            # calls (consecutive or interleaved), it's stuck.
            for tc in tool_calls:
                if tc.function.name == "web_fetch":
                    args = tc.function.arguments
                    if isinstance(args, dict):
                        url = str(args.get("url", ""))
                    else:
                        # ollama sometimes serializes arguments as JSON string
                        try:
                            url = str(json.loads(args).get("url", ""))
                        except Exception:
                            url = str(args)
                    if url:
                        recent_fetched_urls.append(url)
            if recent_fetched_urls:
                most_common_url, most_common_count = Counter(recent_fetched_urls).most_common(1)[0]
                if most_common_count >= URL_REPEAT_LIMIT:
                    yield {
                        "type": "content",
                        "content": (
                            f"\n[System: same URL fetched {most_common_count}× in the last "
                            f"{len(recent_fetched_urls)} web_fetch calls "
                            f"({most_common_url[:80]}…) — likely a loop. Stopping.]"
                        ),
                    }
                    break

            if not tool_calls:
                if full_response.strip() or full_reasoning.strip():
                    msg = {"role": "assistant", "content": full_response}
                    if full_reasoning: msg["reasoning"] = full_reasoning
                    self.messages.append(msg)
                    self.db.add_message(self.session_id, "assistant", full_response)
                    break
                yield {"type": "content", "content": "[System: No response generated. Retrying...]"}
                self.messages.append({"role": "user", "content": "(continue)"})
                continue

            self.messages.append({"role": "assistant", "content": full_response, "tool_calls": [{"function": {"name": t.function.name, "arguments": t.function.arguments}} for t in tool_calls]})
            self.db.add_message(self.session_id, "assistant", full_response, tool_calls=[{"function": {"name": t.function.name, "arguments": t.function.arguments}} for t in tool_calls])

            for tool in tool_calls:
                tool_func = registry.tools.get(tool.function.name)
                is_interactive = tool.function.arguments.get('interactive', False)
                
                if tool_func and getattr(tool_func, 'auth_required', False) and not self.session_authorized:
                    auth = yield {"type": "auth_required", "name": tool.function.name, "arguments": tool.function.arguments}
                    if auth == "deny":
                        res = "Authorization denied."
                        self.messages.append({'role': 'tool', 'content': res, 'name': tool.function.name})
                        yield {"type": "tool_end", "name": tool.function.name, "result": res}
                        continue
                    elif auth == "allow_session": self.session_authorized = True

                yield {"type": "tool_start", "name": tool.function.name, "arguments": tool.function.arguments, "interactive": is_interactive}
                try:
                    result = tool_func(**tool.function.arguments) if tool_func else "Tool not found."
                except Exception as e: result = f"Error: {str(e)}"
                
                result_str = str(result)
                full_result = result_str
                if len(result_str) > 8000:
                    head = result_str[:5000]
                    tail = result_str[-2500:]
                    result_str = f"{head}\n\n... ({len(result_str)} chars total, middle truncated) ...\n\n{tail}"
                tool_msg = {'role': 'tool', 'content': result_str, 'name': tool.function.name}
                self.messages.append(tool_msg)
                self.db.add_message(self.session_id, "tool", result_str)
                if tool.function.name in MUTATING_TOOLS:
                    self._record_action(
                        tool_name=tool.function.name,
                        args=tool.function.arguments,
                        result=full_result,
                        assistant_text=full_response,
                    )
                yield {"type": "tool_end", "name": tool.function.name, "result": full_result}

        if iteration_count >= max_iterations:
            yield {"type": "content", "content": (
                f"\n[System: agent ran {max_iterations} tool-call iterations "
                f"— pausing here. If the goal still needs more work, ask me to "
                f"continue, or bump EZCLAW_MAX_ITERATIONS in your env.]"
            )}
