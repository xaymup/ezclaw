import ollama
import os
import json
import re
from typing import List, Dict, Any, Optional, Callable, Iterator
from memory import Database
from tools import registry, create_memory_tools
from embed import embed, cosine_similarity
from model_client import build_agent_client
from dotenv import load_dotenv

load_dotenv()

SKILLS_DIR = "skills"

def load_skills() -> List[Dict[str, str]]:
    """Load all skills from the skills directory."""
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

def match_skills(user_input: str, skills: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Find relevant skills using embedding similarity, fallback to keyword matching."""
    if not skills:
        return []
    from embed import rank_by_similarity
    skill_texts = [f"{s['name']}: {s['content'][:500]}" for s in skills]
    ranked_texts = rank_by_similarity(user_input, skill_texts, top_n=3)
    matched_names = set()
    for rt in ranked_texts:
        name = rt.split(":")[0]
        matched_names.add(name)

    # Fallback: keyword match if embeddings returned nothing
    if not matched_names:
        user_lower = user_input.lower()
        for skill in skills:
            skill_lower = skill['content'].lower()
            skill_words = set(re.findall(r'\b[a-z]{4,}\b', skill_lower))
            if any(word in user_lower for word in skill_words):
                matched_names.add(skill['name'])

    return [s for s in skills if s['name'] in matched_names]

def format_skills_block(skills: List[Dict[str, str]]) -> str:
    """Format matched skills into a context block."""
    if not skills:
        return ""
    block = "\n<available_skills>\n"
    for skill in skills:
        block += f"\n## {skill['name']}\n{skill['content']}\n"
    block += "</available_skills>\n"
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
        self.tools = registry.get_tool_definitions()
        self.skills = load_skills()
        self.skill_descriptions = "\n".join([f"- {s['name']}: {s['content'].split('## Description')[1].split('##')[0].strip() if '## Description' in s['content'] else 'No description'}" for s in self.skills])
        self._pre_embed_tools()
        
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = self.db.get_last_session_id() or self.db.create_session()
            
        self.messages = self.db.get_messages(self.session_id)
        self.system_prompt = self._load_system_prompt()
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
        max_iterations = 8 
        last_tool_hash = None

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
                yield {"type": "content", "content": "\n[System: Loop detected. Stopping.]"}
                break
            last_tool_hash = current_hash

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
                yield {"type": "tool_end", "name": tool.function.name, "result": full_result}

        if iteration_count >= max_iterations:
            yield {"type": "content", "content": "\n[System: Action limit reached.]"}
