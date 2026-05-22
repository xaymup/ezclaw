import ollama
import os
import json
import re
from typing import List, Dict, Any, Optional, Callable, Iterator
from memory import Database
from tools import registry, create_memory_tools
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
        self.client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
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

        prompt = f"""You are analyzing a user request as part of your persona's "Search-First Analysis Protocol".
Your goal is to identify if the request can be "improved", personalized, or successfully executed by retrieving context from your long-term memory or using tools.

[Known Facts]: {key_facts}

[Available Tools]:
{tool_str}

[Available Skills]:
{self.skill_descriptions}

User Input: "{user_input}"

Your priority is to determine if you have ALL the information needed to answer the user perfectly. 
If the user mentions names, locations, past projects, preferences, or specific tasks that imply previous context (e.g., "my website", "the draft we made", "my favorite X"), you MUST trigger a memory recall.

Instructions:
1. Intent: Identify the core goal. Does it require a tool?
   - For `run_shell`: Only set `interactive: true` for commands that NEED it (vim, ssh). For `ls`, `cat`, etc., use `interactive: false`.
   - For file tools (`read_file`, `write_file`, etc.): ALWAYS use relative paths. Do NOT prefix paths with 'workspace/' or use absolute paths.
2. Gap Analysis: Compare the request against [Known Facts]. Is there a missing piece of context that might be in your long-term memory?
3. Memory Trigger: Set "memory": true if there is ANY chance that past interactions or stored facts could help provide a better, more personalized response.
4. Semantic Query: If "memory" is true, write a query optimized for finding the MISSING context (e.g., if you need a city for weather, query "user location or city"). Do NOT just repeat the user's prompt.

Return ONLY a valid JSON object. 
Example: {{"reasoning": "User asked for weather but city is missing from [Known Facts]. Searching memory for location.", "memory": true, "memory_q": "user city and location", "facts": [], "skill": null}}

JSON:"""
        
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
        prompt = f"""Evaluate the quality of this response.

User Request: "{user_input}"

Agent's Response: "{response}"

Agent's Reasoning: "{reasoning[:500] if reasoning else 'N/A'}"

Criteria:
1. Completeness: Does it fully address the request?
2. Correctness: Is the information accurate?
3. Clarity: Is it well-structured?
4. Actionability: Does it provide concrete help?

Return JSON:
{{"score": 1-10, "needs_improvement": bool, "feedback": "what to improve"}}"""
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
        
        # Nudge the model to follow the CoT protocol
        cot_nudge = "\n[System Reminder]: Follow the MANDATORY CHAIN OF THOUGHT protocol using <think> tags as defined in your persona."
        augment_prefix = f"{memory_block}{skills_block}{cot_nudge}\n[User Prompt]: "

        iteration_count = 0
        max_iterations = 15 
        last_tool_hash = None

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
                stream = self.client.chat(model=self.model, messages=context_messages, tools=self.tools, options=self.options, keep_alive=self.keep_alive, stream=True)
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
                
                tool_msg = {'role': 'tool', 'content': str(result), 'name': tool.function.name}
                self.messages.append(tool_msg)
                self.db.add_message(self.session_id, "tool", str(result))
                yield {"type": "tool_end", "name": tool.function.name, "result": str(result)}

        if iteration_count >= max_iterations:
            yield {"type": "content", "content": "\n[System: Action limit reached.]"}
