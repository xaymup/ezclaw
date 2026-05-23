import os
import json
import re

OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def extract_json(content: str) -> dict:
    content = re.sub(r'<(think|thought|reasoning)\b[^>]*>.*?(</\1>|$)', '', content, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    if match:
        content = match.group(1)
    content = re.sub(r',\s*\}', '}', content)
    return json.loads(content)

def build_architect_client():
    provider = os.getenv("ARCHITECT_PROVIDER", "ollama")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", 300))
    if provider == "deepseek":
        import openai
        return openai.OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            timeout=timeout,
        ), os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    import ollama
    return ollama.Client(host=OLLAMA_HOST, timeout=timeout), os.getenv("OLLAMA_ARCHITECT_MODEL", "qwen3:14b")

def build_agent_client():
    import ollama
    return ollama.Client(host=OLLAMA_HOST, timeout=int(os.getenv("OLLAMA_TIMEOUT", 300)))
