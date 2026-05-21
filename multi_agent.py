import ollama
import os
import json
import re
from typing import List, Dict, Any, Optional, Iterator
from memory import Database
from tools import registry, create_memory_tools
from dotenv import load_dotenv

load_dotenv()

class BaseAgent:
    def __init__(self, model: str, system_prompt: str, db: Database):
        self.client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.model = model
        self.system_prompt = system_prompt
        self.db = db
        self.options = {
            "temperature": 0.0,
            "num_ctx": 32768,
            "top_p": 0.9,
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", 999)),
        }

class Dispatcher(BaseAgent):
    """The Architect: Analyzes intent and delegates to specialized agents."""
    def categorize(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Analyze the user input and categorize the task.
User Input: "{user_input}"

Categories:
- "technical": Coding, debugging, architecture, shell commands.
- "research": Current events, news, documentation lookup, general knowledge.
- "creative": Writing, brainstorming, general chat.
- "multimodal": If the user mentions an image or file analysis.

Return JSON:
{{
  "category": "technical|research|creative|multimodal",
  "reasoning": "why",
  "recommended_agent": "executor|researcher|librarian",
  "requires_vision": bool
}}"""
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            format="json"
        )
        return json.loads(response["message"]["content"])

class MultiAgentSystem:
    def __init__(self):
        self.db = Database(os.getenv("DATABASE_PATH", "ezclaw.db"))
        
        # Initialize specialized agents based on model analysis
        self.architect = Dispatcher(
            model="phi4-reasoning:plus", 
            system_prompt="You are the Architect. Plan and delegate tasks.",
            db=self.db
        )
        self.executor = BaseAgent(
            model="qwen2.5-coder:14b",
            system_prompt="You are the Executor. Write code and run tools.",
            db=self.db
        )
        self.researcher = BaseAgent(
            model="qwen3.5:9b",
            system_prompt="You are the Researcher. Search and summarize.",
            db=self.db
        )
        self.debugger = BaseAgent(
            model="deepseek-r1:14b",
            system_prompt="You are the Debugger. Solve complex logic and bugs.",
            db=self.db
        )
        
        create_memory_tools(self.db)
        self.tools = registry.get_tool_functions()

    def run(self, user_input: str) -> Iterator[Dict[str, Any]]:
        # 1. Dispatcher analyzes
        yield {"type": "content", "content": "🔍 [Architect] Analyzing intent...\n"}
        intent = self.architect.categorize(user_input)
        yield {"type": "reasoning", "content": f"Category: {intent['category']}. Recommended: {intent['recommended_agent']}\n"}

        # 2. Select Agent
        target_model = self.executor.model
        if intent["category"] == "research":
            target_model = self.researcher.model
        elif intent["category"] == "technical" and ("bug" in user_input or "error" in user_input):
            target_model = self.debugger.model
            
        # 3. Execute using the selected model via the standard chat loop logic
        # For now, we'll keep it simple and yield that we've switched
        yield {"type": "content", "content": f"🚀 [Switching to {target_model}] to handle your request.\n"}
        
        # Integration with ChatAgent's streaming logic would follow here
        # (This is a simplified prototype of the multi-agent handoff)

if __name__ == "__main__":
    mas = MultiAgentSystem()
    for chunk in mas.run("How do I fix a python recursion error?"):
        print(chunk)
