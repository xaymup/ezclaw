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
    """Check if any skill is relevant to the user's input."""
    matched = []
    user_lower = user_input.lower()
    for skill in skills:
        # Check if skill description or procedure keywords match the input
        skill_lower = skill['content'].lower()
        # Extract key terms from skill (words that aren't common stop words)
        skill_words = set(re.findall(r'\b[a-z]{4,}\b', skill_lower))
        # Check if any significant skill words appear in user input
        if any(word in user_lower for word in skill_words):
            matched.append(skill)
    return matched

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
        self.tools = registry.get_tool_functions()
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
        Context Augmentor: Uses deep reasoning to identify what personal context 
        is missing OR relevant from the prompt to fulfill the user's intent.
        """
        key_facts = self.db.get_key_facts()
        
        prompt = f"""You are a Context Augmentor. Your goal is to identify if the user's input can be "improved" or personalized by using information from their long-term memory.

[Key Facts already in mind]: {key_facts}

User Input: "{user_input}"

Instructions:
1. Even if you "know" the answer from [Key Facts], if the fact is RELEVANT (e.g., you see their location and they ask about weather), you MUST set "memory": true so it can be explicitly injected into the final prompt.
2. If the user's prompt is generic (e.g., "What's the weather?", "How are you?"), search for personalizing details (location, name, preferences).
3. Generate a SEMANTIC search query for anything not covered in [Key Facts] but needed for the intent.

Return a valid JSON object:
- "reasoning": Your analysis of why context is or isn't needed.
- "memory": bool, true if context (from [Key Facts] or a new search) should be explicitly "provoked" and used.
- "memory_q": A search query for additional details or null.
- "facts": list of new facts to store.
- "skill": skill name or null.

JSON:"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0, "num_ctx": 4096, "num_predict": 200}
            )
            intent = json.loads(response["message"]["content"].strip())
            
            # If the model finds the input is relevant to personalization, 
            # we ensure a query exists to trigger the recall.
            if intent.get("memory") and not intent.get("memory_q"):
                # If no specific query, we use the user input to find the related facts again
                intent["memory_q"] = user_input
                
            return intent
        except Exception:
            return {"memory": False, "memory_q": None, "facts": [], "skill": None}

    def clear_session_history(self):
        """Restores the chat history to just the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
        # 1. Fast consolidated intent analysis
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
        # We keep the real input for history, but use augmented for the model
        self.messages.append({"role": "user", "content": user_input})
        self.db.add_message(self.session_id, "user", user_input)
        
        # Augmented content is used for ALL follow-up turns in this request
        user_msg_augmented_content = f"{memory_block}{skills_block}\n[User Prompt]: {user_input}"

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
            
            # Find the LAST user message and augment it
            for i in range(len(context_messages) - 1, -1, -1):
                if context_messages[i]["role"] == "user":
                    context_messages[i]["content"] = user_msg_augmented_content
                    break

            stream = self.client.chat(model=self.model, messages=context_messages, tools=self.tools, options=self.options, keep_alive=self.keep_alive, stream=True)

            for chunk in stream:
                if hasattr(chunk.message, 'reasoning') and chunk.message.reasoning:
                    full_reasoning += chunk.message.reasoning
                    yield {"type": "reasoning", "content": chunk.message.reasoning}
                
                if chunk.message.content:
                    raw_buffer += chunk.message.content
                    
                    if not in_thinking:
                        # Check for opening tags
                        match = re.search(r'<(think|thought)>', raw_buffer, re.IGNORECASE)
                        if match:
                            in_thinking = True
                            pre = raw_buffer[:match.start()]
                            if pre: 
                                full_response += pre
                                yield {"type": "content", "content": pre}
                            raw_buffer = raw_buffer[match.end():]
                        else:
                            # Avoid yielding if we might be in the middle of a tag
                            if '<' in raw_buffer:
                                last_bracket = raw_buffer.rfind('<')
                                if any(tag.startswith(raw_buffer[last_bracket:].lower()) for tag in ["<think>", "<thought>"]):
                                    # Yield everything BEFORE the bracket
                                    pre = raw_buffer[:last_bracket]
                                    if pre:
                                        full_response += pre
                                        yield {"type": "content", "content": pre}
                                    raw_buffer = raw_buffer[last_bracket:]
                                else:
                                    # Not a tag start, yield it all
                                    full_response += raw_buffer
                                    yield {"type": "content", "content": raw_buffer}
                                    raw_buffer = ""
                            else:
                                full_response += raw_buffer
                                yield {"type": "content", "content": raw_buffer}
                                raw_buffer = ""
                    else:
                        # Check for closing tags
                        match = re.search(r'</(think|thought)>', raw_buffer, re.IGNORECASE)
                        if match:
                            in_thinking = False
                            pre = raw_buffer[:match.start()]
                            if pre:
                                full_reasoning += pre
                                yield {"type": "reasoning", "content": pre}
                            raw_buffer = raw_buffer[match.end():]
                        else:
                            # Avoid yielding if we might be in the middle of a closing tag
                            if '</' in raw_buffer:
                                last_bracket = raw_buffer.rfind('</')
                                if any(tag.startswith(raw_buffer[last_bracket:].lower()) for tag in ["</think>", "</thought>"]):
                                    pre = raw_buffer[:last_bracket]
                                    if pre:
                                        full_reasoning += pre
                                        yield {"type": "reasoning", "content": pre}
                                    raw_buffer = raw_buffer[last_bracket:]
                                else:
                                    full_reasoning += raw_buffer
                                    yield {"type": "reasoning", "content": raw_buffer}
                                    raw_buffer = ""
                            else:
                                full_reasoning += raw_buffer
                                yield {"type": "reasoning", "content": raw_buffer}
                                raw_buffer = ""
                
                if chunk.message.tool_calls: tool_calls.extend(chunk.message.tool_calls)

            # Flush any remaining buffer at the end of stream
            if raw_buffer:
                if in_thinking:
                    full_reasoning += raw_buffer
                    yield {"type": "reasoning", "content": raw_buffer}
                else:
                    full_response += raw_buffer
                    yield {"type": "content", "content": raw_buffer}
                raw_buffer = ""

            # 7. Repeat Checker
            current_hash = hash(str([(t.function.name, t.function.arguments) for t in tool_calls]))
            if tool_calls and current_hash == last_tool_hash:
                yield {"type": "content", "content": "\n[System: Loop detected. Stopping.]"}
                break
            last_tool_hash = current_hash

            if not tool_calls:
                # If we have content, we're done.
                if full_response.strip() or full_reasoning.strip():
                    msg = {"role": "assistant", "content": full_response}
                    if full_reasoning: msg["reasoning"] = full_reasoning
                    self.messages.append(msg)
                    self.db.add_message(self.session_id, "assistant", full_response)
                    break
                
                # If it's a follow-up turn and we have NO content and NO tools, the model might be stuck.
                # We nudge it to finish.
                if iteration_count > 1:
                    self.messages.append({"role": "user", "content": "(continue)"})
                    continue
                else:
                    break

            # 8. Process Tools
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