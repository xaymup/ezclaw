import ollama
import os
import numpy as np
from typing import List, Optional

OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
_client = ollama.Client(host=OLLAMA_HOST)


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
