"""Routing tests: verify that the pre-flight kind count gates the
fast-route branch in MultiAgentSystem.run."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_mas(monkeypatch, tmp_path):
    import multi_agent
    monkeypatch.setattr(multi_agent, "load_skills", lambda: [])
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "routing.db"))
    from multi_agent import MultiAgentSystem
    return MultiAgentSystem(session_id=None)


def _drain_until_status_or_end(gen, limit: int = 20):
    """Iterate the run() generator until either a status chunk arrives
    or the generator ends. Returns list of chunks yielded."""
    chunks = []
    try:
        for _ in range(limit):
            chunk = next(gen)
            chunks.append(chunk)
            if chunk.get("type") == "status":
                break
    except StopIteration:
        pass
    return chunks


def test_low_kind_count_fast_routes(monkeypatch, tmp_path):
    """When the pre-flight returns 1 kind, the fast-route status fires
    and the architect loop is NOT entered."""
    mas = _build_mas(monkeypatch, tmp_path)
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: "executor")
    monkeypatch.setattr(mas, "_estimate_tool_kinds", lambda u: 1)

    executor = mas.agents["executor"]
    def fake_stream(_input):
        yield {"type": "content", "content": "ok"}
    monkeypatch.setattr(executor, "chat_stream", fake_stream)

    gen = mas.run("read foo.py")
    chunks = _drain_until_status_or_end(gen)
    gen.close()

    status_msgs = [c["content"] for c in chunks if c.get("type") == "status"]
    assert any("fast-routed" in m for m in status_msgs)
    assert not any("pre-flight" in m for m in status_msgs)


def test_high_kind_count_skips_fast_route(monkeypatch, tmp_path):
    """When the pre-flight returns >= PREFLIGHT_KIND_THRESHOLD, the
    pre-flight status fires and the fast-route status does NOT."""
    from multi_agent import PREFLIGHT_KIND_THRESHOLD
    mas = _build_mas(monkeypatch, tmp_path)
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: "executor")
    monkeypatch.setattr(mas, "_estimate_tool_kinds", lambda u: PREFLIGHT_KIND_THRESHOLD)

    # Stub the architect loop to fail fast so we don't hit real LLMs.
    def boom(*args, **kwargs):
        raise RuntimeError("architect short-circuit for test")
    monkeypatch.setattr(mas.architect, "reflect_and_plan", boom, raising=False)

    gen = mas.run("write me a snake game")
    chunks = _drain_until_status_or_end(gen, limit=5)
    try:
        gen.close()
    except Exception:
        pass

    status_msgs = [c["content"] for c in chunks if c.get("type") == "status"]
    assert any("pre-flight" in m and "planning" in m for m in status_msgs)
    assert not any("fast-routed" in m for m in status_msgs)


def test_preflight_none_skips_fast_route(monkeypatch, tmp_path):
    """When the pre-flight returns None (failure), the pre-flight status
    fires with '?' and the fast-route status does NOT."""
    mas = _build_mas(monkeypatch, tmp_path)
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: "executor")
    monkeypatch.setattr(mas, "_estimate_tool_kinds", lambda u: None)
    def boom(*args, **kwargs):
        raise RuntimeError("architect short-circuit for test")
    monkeypatch.setattr(mas.architect, "reflect_and_plan", boom, raising=False)

    gen = mas.run("do a thing")
    chunks = _drain_until_status_or_end(gen, limit=5)
    try:
        gen.close()
    except Exception:
        pass

    status_msgs = [c["content"] for c in chunks if c.get("type") == "status"]
    assert any("pre-flight: ?" in m for m in status_msgs)
    assert not any("fast-routed" in m for m in status_msgs)


def test_preflight_not_called_for_general_short_circuit(monkeypatch, tmp_path):
    """Pre-flight should ONLY run when _short_circuit_classify returned
    'executor'. For 'general' / 'researcher', it must NOT be called."""
    mas = _build_mas(monkeypatch, tmp_path)
    monkeypatch.setattr(mas, "_short_circuit_classify", lambda u: "general")

    estimate_calls = {"count": 0}
    def counting_estimate(_input):
        estimate_calls["count"] += 1
        return 1
    monkeypatch.setattr(mas, "_estimate_tool_kinds", counting_estimate)

    general = mas.agents["general"]
    def fake_stream(_input):
        yield {"type": "content", "content": "hi"}
    monkeypatch.setattr(general, "chat_stream", fake_stream)

    gen = mas.run("hello")
    _drain_until_status_or_end(gen)
    gen.close()

    assert estimate_calls["count"] == 0
