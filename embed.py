import ollama
import os
import hashlib
import numpy as np
from typing import List, Optional, Tuple
from functools import lru_cache

OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
_client = ollama.Client(host=OLLAMA_HOST, timeout=int(os.getenv("OLLAMA_TIMEOUT", 30)))

_embed_cache: dict = {}
_MAX_CACHE = 512


def embed(text: str) -> List[float]:
    if len(text) > 1000:
        text = text[:1000]
    key = hashlib.md5(text.encode()).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]
    result = _client.embeddings(model=EMBED_MODEL, prompt=text)
    vec = result["embedding"]
    if len(_embed_cache) >= _MAX_CACHE:
        oldest = next(iter(_embed_cache))
        del _embed_cache[oldest]
    _embed_cache[key] = vec
    return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def rank_by_similarity(query: str, candidates: List[str], top_n: Optional[int] = None, threshold: float = 0.0) -> List[str]:
    if not candidates:
        return []
    q_vec = embed(query)
    c_vecs = [embed(c) for c in candidates]
    scored = [(cosine_similarity(q_vec, cv), c) for cv, c in zip(c_vecs, candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)
    result = [(s, c) for s, c in scored if s >= threshold]
    items = [c for _, c in result]
    return items[:top_n] if top_n else items


def rank_by_similarity_vectors(q_vec: List[float], candidate_vectors: List[List[float]], candidates: List[str], top_n: Optional[int] = None) -> List[str]:
    scored = [(cosine_similarity(q_vec, cv), c) for cv, c in zip(candidate_vectors, candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)
    result = [c for _, c in scored]
    return result[:top_n] if top_n else result


_classify_cache: dict = {}

def classify_by_similarity(query: str, examples: List[str], labels: List[str], threshold: float = 0.6) -> Optional[str]:
    cache_key = tuple(examples)
    if cache_key not in _classify_cache:
        _classify_cache[cache_key] = [embed(ex) for ex in examples]
    ex_vecs = _classify_cache[cache_key]

    q_vec = embed(query)
    best_label, best_score = None, threshold
    for ex_vec, label in zip(ex_vecs, labels):
        sim = cosine_similarity(q_vec, ex_vec)
        if sim > best_score:
            best_score = sim
            best_label = label
    return best_label
