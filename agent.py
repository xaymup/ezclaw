import ollama
import os
import json
import re
from typing import List, Dict, Any, Optional, Callable, Iterator
from memory import Database
from tools import registry, create_memory_tools
from dotenv import load_dotenv

load_dotenv()

class ChatAgent:
    def __init__(self, session_id: Optional[int] = None):
        self.client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.db = Database(os.getenv("DATABASE_PATH", "ezclaw.db"))
        self.model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
        
        ka = os.getenv("OLLAMA_KEEP_ALIVE", "60m")
        try:
            self.keep_alive = int(ka) if ka.isdigit() or ka == "-1" else ka
        except:
            self.keep_alive = "60m"
        
        self.options = {
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", 0.0)),
            "top_p": float(os.getenv("OLLAMA_TOP_P", 0.9)),
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", 16384)),
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", 999)),
            "num_thread": int(os.getenv("OLLAMA_NUM_THREAD", 16)),
            "num_batch": int(os.getenv("OLLAMA_NUM_BATCH", 2048)),
        }
        
        create_memory_tools(self.db)
        self.tools = registry.get_tool_functions()
        
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = self.db.get_last_session_id() or self.db.create_session()
            
        self.messages = self.db.get_messages(self.session_id)
        self.system_prompt = self._load_system_prompt()
        self._ensure_system_message()
        self.session_authorized = False

    def _load_system_prompt(self) -> str:
        content = "You are EzClaw, a powerful terminal-based assistant."
        if os.path.exists("agents.md"):
            try:
                with open("agents.md", "r") as f:
                    content = f.read()
            except: pass
        return content

    def _ensure_system_message(self):
        if not self.messages or self.messages[0]["role"] != "system":
            system_msg = {"role": "system", "content": self.system_prompt}
            if not self.messages:
                self.messages.append(system_msg)
            else:
                self.messages[0] = system_msg
            self.db.add_message(self.session_id, system_msg["role"], system_msg["content"])

    def clear_session_history(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def chat_stream(self, user_input: str) -> Iterator[Dict[str, Any]]:
        # Proactive Memory Recall
        memories = self.db.search_memories(user_input)
        if memories:
            memory_context = "\n[RELEVANT MEMORIES]\n- " + "\n- ".join(memories)
            # Inject memories as a hidden context hint in the user message for the model
            augmented_input = f"{user_input}\n{memory_context}"
        else:
            augmented_input = user_input

        user_msg = {"role": "user", "content": augmented_input}
        self.messages.append({"role": "user", "content": user_input}) # Keep clean version in history
        self.db.add_message(self.session_id, "user", user_input)

        while True:
            full_response_content = ""
            full_reasoning_content = ""
            tool_calls = []
            
            in_thinking = False
            raw_buffer = ""

            # Use augmented input for the latest user message in the context
            history_for_context = self.messages[:-1] + [user_msg]
            context_messages = [self.messages[0]] + history_for_context[-10:] if len(history_for_context) > 11 else history_for_context

            stream = self.client.chat(
                model=self.model,
                messages=context_messages,
                tools=self.tools,
                options=self.options,
                keep_alive=self.keep_alive,
                stream=True
            )

            for chunk in stream:
                if hasattr(chunk.message, 'reasoning') and chunk.message.reasoning:
                    full_reasoning_content += chunk.message.reasoning
                    yield {"type": "reasoning", "content": chunk.message.reasoning}
                
                if chunk.message.content:
                    text = chunk.message.content
                    raw_buffer += text
                    
                    if not in_thinking:
                        # Check for opening tags
                        match = re.search(r'<(think|thought)>', raw_buffer, re.IGNORECASE)
                        if match:
                            in_thinking = True
                            pre_tag = raw_buffer[:match.start()]
                            if pre_tag:
                                full_response_content += pre_tag
                                yield {"type": "content", "content": pre_tag}
                            raw_buffer = raw_buffer[match.end():]
                        elif not any(tag.startswith(raw_buffer.lower()[raw_buffer.lower().rfind('<'):]) for tag in ["<think>", "<thought>"] if '<' in raw_buffer):
                            # Not a potential tag, yield everything
                            full_response_content += raw_buffer
                            yield {"type": "content", "content": raw_buffer}
                            raw_buffer = ""
                        # Otherwise (is a potential tag prefix), wait for more data.
                    else:
                        # Check for closing tags
                        match = re.search(r'</(think|thought)>', raw_buffer, re.IGNORECASE)
                        if match:
                            in_thinking = False
                            pre_tag = raw_buffer[:match.start()]
                            if pre_tag:
                                full_reasoning_content += pre_tag
                                yield {"type": "reasoning", "content": pre_tag}
                            raw_buffer = raw_buffer[match.end():]
                        elif not any(tag.startswith(raw_buffer.lower()[raw_buffer.lower().rfind('</'):]) for tag in ["</think>", "</thought>"] if '</' in raw_buffer):
                            # Not a potential closing tag prefix, yield everything
                            full_reasoning_content += raw_buffer
                            yield {"type": "reasoning", "content": raw_buffer}
                            raw_buffer = ""
                
                if chunk.message.tool_calls:
                    tool_calls.extend(chunk.message.tool_calls)

            # Final flush
            if raw_buffer:
                if in_thinking:
                    full_reasoning_content += raw_buffer
                    yield {"type": "reasoning", "content": raw_buffer}
                else:
                    full_response_content += raw_buffer
                    yield {"type": "content", "content": raw_buffer}

            if not tool_calls:
                assistant_msg = {"role": "assistant", "content": full_response_content}
                if full_reasoning_content:
                    assistant_msg["reasoning"] = full_reasoning_content
                self.messages.append(assistant_msg)
                self.db.add_message(self.session_id, assistant_msg["role"], assistant_msg["content"])
                break

            # Handle tools
            self.messages.append({
                "role": "assistant", 
                "content": full_response_content,
                "tool_calls": [{"function": {"name": t.function.name, "arguments": t.function.arguments}} for t in tool_calls]
            })
            self.db.add_message(self.session_id, "assistant", full_response_content, 
                                tool_calls=[{"function": {"name": t.function.name, "arguments": t.function.arguments}} for t in tool_calls])

            for tool in tool_calls:
                tool_func = registry.tools.get(tool.function.name)
                
                # Check for authorization
                if tool_func and getattr(tool_func, 'auth_required', False) and not self.session_authorized:
                    auth_response = yield {
                        "type": "auth_required", 
                        "name": tool.function.name, 
                        "arguments": tool.function.arguments
                    }
                    if auth_response == "deny":
                        result = "Authorization denied by user."
                        tool_msg = {'role': 'tool', 'content': result, 'name': tool.function.name}
                        self.messages.append(tool_msg)
                        self.db.add_message(self.session_id, tool_msg["role"], tool_msg["content"])
                        yield {"type": "tool_end", "name": tool.function.name, "result": result}
                        continue
                    elif auth_response == "allow_session":
                        self.session_authorized = True

                yield {"type": "tool_start", "name": tool.function.name, "arguments": tool.function.arguments}
                if tool_func:
                    try:
                        result = tool_func(**tool.function.arguments)
                    except Exception as e:
                        result = f"Error executing tool: {str(e)}"
                    tool_msg = {'role': 'tool', 'content': str(result), 'name': tool.function.name}
                    self.messages.append(tool_msg)
                    self.db.add_message(self.session_id, tool_msg["role"], tool_msg["content"])
                    yield {"type": "tool_end", "name": tool.function.name, "result": str(result)}
                else:
                    error_msg = f"Tool {tool.function.name} not found."
                    tool_msg = {'role': 'tool', 'content': error_msg, 'name': tool.function.name}
                    self.messages.append(tool_msg)
                    self.db.add_message(self.session_id, tool_msg["role"], tool_msg["content"])
                    yield {"type": "tool_end", "name": tool.function.name, "result": error_msg}

    def chat(self, user_input: str, stream_callback: Optional[Callable[[str], None]] = None) -> str:
        full_content = ""
        for chunk in self.chat_stream(user_input):
            if chunk["type"] == "content":
                full_content += chunk["content"]
        return full_content
