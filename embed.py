import ollama
import os
import numpy as np
from typing import List, Optional

OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
_client = ollama.Client(host=OLLAMA_HOST)

_AGENT_PROTOTYPES = {
    "executor": [
        "run a shell command",
        "create a file",
        "install a package",
        "check disk space",
        "list files in directory",
        "edit a Python file",
        "start a server",
        "download a file",
        "set up a project",
        "execute this code",
        "show me the contents of",
        "read the contents of a file",
        "write to a file",
        "what files are here",
        "delete a directory",
        "what is my name",
        "recall my preferences",
        "what did I tell you",
        "remember this fact",
        "look up my information",
    ],
    "researcher": [
        "search the web for",
        "find documentation about",
        "look up information on",
        "research this topic",
        "fetch a URL",
        "what is the latest news",
        "find a tutorial",
        "lookup an API reference",
        "search for a library",
        "what is the weather forecast",
        "check the weather",
        "latest stock price",
        "find information online",
        "what is the weather today",
        "tell me the weather",
    ],
    "debugger": [
        "find bugs in this code",
        "debug my application",
        "fix this error",
        "analyze this crash",
        "why is this broken",
        "check for issues in the code",
        "trace this exception",
        "review this code for errors",
        "what is wrong with",
        "why does this fail",
    ],
    "general": [
        "hello",
        "how are you",
        "what do you think",
        "tell me a joke",
        "thanks",
        "goodbye",
        "nice to meet you",
        "what is your name",
        "how does this work",
        "explain the concept",
        "nice",
        "okay",
    ],
}

_prototype_vecs: Optional[dict] = None


def _get_prototype_vecs():
    global _prototype_vecs
    if _prototype_vecs is None:
        _prototype_vecs = {}
        for agent, queries in _AGENT_PROTOTYPES.items():
            _prototype_vecs[agent] = [embed(q) for q in queries]
    return _prototype_vecs


def embed(text: str) -> List[float]:
    result = _client.embeddings(model=EMBED_MODEL, prompt=text)
    return result["embedding"]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def rank_by_similarity(query: str, candidates: List[str], top_n: Optional[int] = None) -> List[str]:
    if not candidates:
        return []
    q_vec = embed(query)
    c_vecs = [embed(c) for c in candidates]
    scored = [(cosine_similarity(q_vec, cv), c) for cv, c in zip(c_vecs, candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)
    result = [c for _, c in scored]
    return result[:top_n] if top_n else result


def classify_intent(query: str) -> str:
    """Classify user intent into one of: executor, researcher, debugger, general.
    Uses max embedding similarity against prototype queries for each agent type.
    """
    q_vec = embed(query)
    protos = _get_prototype_vecs()
    best_agent = "executor"
    best_score = -1.0

    for agent, vecs in protos.items():
        max_sim = max(cosine_similarity(q_vec, pv) for pv in vecs)
        if max_sim > best_score:
            best_score = max_sim
            best_agent = agent

    return best_agent
