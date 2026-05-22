import ollama
import os
import json
import re
from typing import List, Dict, Any, Optional, Callable, Iterator
from memory import Database
from tools import registry, create_memory_tools
from embed import embed, cosine_similarity
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
        self.client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), timeout=int(os.getenv("OLLAMA_TIMEOUT", 300)))
        self.db = Database(os.getenv("DATABASE_PATH", "ezclaw.db"))
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
        
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m")
        
        self.options = {
            "temperature": 0.0,
            "num_ctx": 32768, 
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

    def _select_relevant_tools(self, user_input: str, top_n: int = 12) -> List[Dict[str, Any]]:
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

    def _process_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Context Augmentor: Unifies persona, capabilities, and memory retrieval.
        Uses robust extraction to work with any model (reasoning, markdown, etc.).
        """
        key_facts = self.db.get_key_facts()
        
        tool_info = []
        for t in self.tools:
            name = t['function']['name']
            desc = t['function']['description']
            tool_info.append(f"- {name}: {desc}")
        tool_str = "\n".join(tool_info)

        prompt = f"""You are analyzing a user request to extract intent, context needs, and tool requirements.

[Known Facts]: {key_facts}

[Available Tools]:
{tool_str}

[Available Skills]:
{self.skill_descriptions}

User Input: "{user_input}"

Analyze:
1. **Intent**: What does the user actually want? Does it need a tool? Which one?
2. **Gap Analysis**: Compare against [Known Facts]. Is anything missing that memory could fill?
3. **Memory Trigger**: Set memory:true if the user references anything personal ("my", "I", past context, names, locations, projects, preferences). Write a semantic query optimized to find the missing context, not just a repeat of the prompt.
4. **Skill Match**: Does any skill match this request?

Return ONLY valid JSON:
{{"reasoning": "your analysis", "memory": true/false, "memory_q": "optimized search query or null", "facts": [], "skill": "skill_name_or_null"}}"""
        
        # Self-Correction Loop for Intent Analysis
        for attempt in range(2):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt if attempt == 0 else prompt + "\n\nCRITICAL: Your previous response was not valid JSON. Return ONLY the JSON object."}
                    ],
                    format="json",
                    options={"temperature": 0.0, "num_ctx": 4096, "num_predict": 300}
                )
                content = response["message"]["content"].strip()
                
                # ROBUST EXTRACTION:
                # 1. Strip reasoning/thinking tags (even unclosed ones)
                content = re.sub(r'<(think|thought|reasoning)\b[^>]*>.*?(</\1>|$)', '', content, flags=re.DOTALL | re.IGNORECASE)
                
                # 2. Extract content between first { and last }
                match = re.search(r'(\{.*\})', content, re.DOTALL)
                if match:
                    content = match.group(1)
                
                # 3. Clean up common LLM JSON hallucinations
                content = re.sub(r',\s*\}', '}', content) 
                content = re.sub(r',\s*\]', ']', content)
                
                intent = json.loads(content)
                
                # Ensure memory_q exists if memory is triggered
                if intent.get("memory") and not intent.get("memory_q"):
                    intent["memory_q"] = user_input
                    
                return intent
            except Exception:
                if attempt == 1:
                    # Final fallback if even self-correction fails
                    return {"memory": False, "memory_q": None, "facts": [], "skill": None}
                continue

    def _judge_response(self, user_input: str, response: str, reasoning: str) -> Dict[str, Any]:
        prompt = f"""Evaluate this response.

User: "{user_input[:300]}"
Response: "{response[:500]}"
Reasoning: "{reasoning[:300] if reasoning else 'N/A'}"

Rate 1-10 on:
- Completeness: Does it fully answer the request?
- Correctness: Is the information accurate? Any hallucination risk?
- Conciseness: Is it as short as it should be?
- Actionability: Does it give the user what they need?

Return JSON:
{{"score": 1-10, "needs_improvement": bool, "feedback": "what to improve, or empty"}}"""
        try:
            res = self.client.chat(model=self.model, messages=[{"role": "user", "content": prompt}], format="json", options={"temperature": 0.0, "num_ctx": 4096, "num_predict": 300})
            content = res["message"]["content"].strip()
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match: content = match.group(1)
            return json.loads(content)
        except Exception:
            return {"score": 10, "needs_improvement": False, "feedback": ""}

    def clear_session_history(self):
        """Restores the chat history to just the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
        # 1. Consolidated intent analysis with full agent context
        intent = self._process_intent(user_input)

        # 2. Process extracted facts immediately
        for fact in intent.get("facts", []):
            if fact:
                self.db.add_memory(fact)
                yield {"type": "memory_stored", "fact": fact}

        # 3. Memory recall & Context Augmentation
        memory_block = ""
        if intent.get("memory"):
            query = intent.get("memory_q") or user_input
            memories = self.db.search_memories(query)
            if memories:
                memory_block = f"\n[Background Context retrieved from Memory for this request]:\n" + "\n".join([f"- {m}" for m in memories]) + "\n"
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
        
        cot_nudge = "\n[Reminder]: Analyze the request in <think> tags before responding."
        augment_prefix = f"{memory_block}{skills_block}{cot_nudge}\n[User]: "

        iteration_count = 0
        max_iterations = 8 
        last_tool_hash = None
        last_tool_vec = None

        # Tool pre-selection: only pass tools relevant to the current query
        selected_tools = self._select_relevant_tools(user_input, top_n=12) if len(self.tools) > 12 else self.tools

        while iteration_count < max_iterations:
            iteration_count += 1
            full_response, full_reasoning, tool_calls = "", "", []
            in_thinking, raw_buffer = False, ""

            # 6. Context Retention logic
            if len(self.messages) > 20:
                context_messages = [self.messages[0], self.messages[1]] + self.messages[-18:]
            else:
                context_messages = [m.copy() for m in self.messages]

            # Augment the current request message (prepend augment block)
            for i in range(len(context_messages) - 1, -1, -1):
                if context_messages[i]["role"] == "user":
                    if not context_messages[i]["content"].startswith(augment_prefix):
                        context_messages[i]["content"] = augment_prefix + context_messages[i]["content"]
                    break

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
                
                if chunk.message.tool_calls: tool_calls.extend(chunk.message.tool_calls)

            if raw_buffer:
                if in_thinking:
                    full_reasoning += raw_buffer
                    yield {"type": "reasoning", "content": raw_buffer}
                else:
                    full_response += raw_buffer
                    yield {"type": "content", "content": raw_buffer}
                raw_buffer = ""

            current_hash = hash(str([(t.function.name, t.function.arguments) for t in tool_calls]))
            if tool_calls and current_hash == last_tool_hash:
                yield {"type": "content", "content": "\n[System: Loop detected. Stopping.]"}
                break
            last_tool_hash = current_hash

            if tool_calls and last_tool_vec is not None:
                tool_summary = " ".join(f"{t.function.name}:{json.dumps(t.function.arguments, sort_keys=True)}" for t in tool_calls)
                try:
                    current_vec = embed(tool_summary)
                    sim = cosine_similarity(current_vec, last_tool_vec)
                    if sim > 0.95:
                        yield {"type": "content", "content": "\n[Semantic loop detected. Stopping.]"}
                        break
                    last_tool_vec = current_vec
                except Exception:
                    pass
            elif tool_calls:
                try:
                    tool_summary = " ".join(f"{t.function.name}:{json.dumps(t.function.arguments, sort_keys=True)}" for t in tool_calls)
                    last_tool_vec = embed(tool_summary)
                except Exception:
                    pass

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
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + f"\n... (truncated, {len(result_str)} chars total)"
                tool_msg = {'role': 'tool', 'content': result_str, 'name': tool.function.name}
                self.messages.append(tool_msg)
                self.db.add_message(self.session_id, "tool", result_str)
                yield {"type": "tool_end", "name": tool.function.name, "result": str(result)}

        if iteration_count >= max_iterations:
            yield {"type": "content", "content": "\n[System: Action limit reached.]"}
