"""Headless trace harness — runs MultiAgentSystem.run on a single prompt
and prints a structured trace.

Use to diagnose routing/handoff/timing issues without booting the TUI.

  python tools_dev/trace_run.py "Help me create a morning routine for myself"

The trace shows: which model fired at each step, the architect's
routing decision, sub-agent <think> blocks (truncated), tool calls,
the final response, end-to-end latency, per-step latency, completion
condition that fired.
"""

from __future__ import annotations

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent import MultiAgentSystem, AGENT_DEFS


def _flush_print(*a, **kw):
    print(*a, **kw, flush=True)


def _short(s, n=200):
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def main():
    args = list(sys.argv[1:])
    validate = False
    if "--validate" in args:
        validate = True
        args.remove("--validate")
    prompt = " ".join(args) or "Help me create a morning routine for myself"
    if validate:
        # Strict chunk schema validation — raises ChunkValidationError on
        # the first malformed chunk. Use this to catch contract drift
        # between the orchestrator and the UI.
        os.environ["EZCLAW_VALIDATE_CHUNKS"] = "1"
        _flush_print(f"[--validate enabled: strict chunk-schema validation]")
    print("══════════════════════════════════════════════════════════════════")
    print(f"PROMPT: {prompt}")
    print("══════════════════════════════════════════════════════════════════")
    print()
    print("Configured models:")
    print(f"  architect : {os.getenv('OLLAMA_ARCHITECT_MODEL') or '(default)'}")
    for k, v in AGENT_DEFS.items():
        print(f"  {k:<10}: {v['model']}")
    print()

    t0 = time.time()
    mas = MultiAgentSystem()
    init_ms = (time.time() - t0) * 1000
    print(f"MAS initialized in {init_ms:.0f}ms")
    print()

    step_idx = 0
    current_agent = None
    last_chunk_time = t0
    tools_called: list = []
    final_response_parts: list = []
    completion_observed = False
    plan_state = None

    print("── stream ────────────────────────────────────────────────────────")
    try:
        # Wrap with validate_stream when --validate is on. Outside that
        # mode it's a pass-through with zero overhead.
        from orchestration import validate_stream
        for chunk in validate_stream(mas.run(prompt)):
            now = time.time()
            dt = (now - last_chunk_time) * 1000
            last_chunk_time = now
            ctype = chunk.get("type")
            if ctype == "intent":
                step_idx += 1
                current_agent = chunk.get("agent")
                reasoning = _short(chunk.get("reasoning"), 100)
                plan_line = _short(chunk.get("plan"), 100)
                refl = chunk.get("reflection") or {}
                print(f"\n[+{dt:>6.0f}ms] ▸ STEP {step_idx} → agent={current_agent}")
                print(f"             reasoning: {reasoning}")
                if plan_line:
                    print(f"             plan:      {plan_line}")
                if refl.get("goal"):
                    print(f"             goal:      {_short(refl.get('goal'), 80)}")
                if refl.get("observation"):
                    print(f"             observ:    {_short(refl.get('observation'), 80)}")
                if refl.get("critical_thinking"):
                    print(f"             critical:  {_short(refl.get('critical_thinking'), 80)}")
            elif ctype == "plan_created":
                p = chunk["plan"]
                plan_state = p
                print(f"\n[+{dt:>6.0f}ms] PLAN_CREATED: {p.title}")
                for t in p.tasks:
                    print(f"               {t.id}. [{t.status}] {t.description}")
            elif ctype == "plan_update":
                p = chunk["plan"]
                plan_state = p
                print(f"\n[+{dt:>6.0f}ms] PLAN_UPDATE:")
                for t in p.tasks:
                    print(f"               {t.id}. [{t.status}] {t.description}")
            elif ctype == "status":
                print(f"[+{dt:>6.0f}ms] status: {_short(chunk.get('content'), 120)}")
            elif ctype == "reasoning":
                print(f"[+{dt:>6.0f}ms] think:  {_short(chunk.get('content'), 100)}")
            elif ctype == "content":
                content = chunk.get("content", "")
                final_response_parts.append(content)
                print(f"[+{dt:>6.0f}ms] content[{current_agent}]: {_short(content, 100)}")
            elif ctype == "tool_start":
                tools_called.append(chunk.get("name"))
                args_str = _short(chunk.get("arguments"), 80)
                print(f"[+{dt:>6.0f}ms] tool_start: {chunk.get('name')}({args_str})")
            elif ctype == "tool_end":
                result = _short(chunk.get("result"), 80)
                print(f"[+{dt:>6.0f}ms] tool_end:   {chunk.get('name')} → {result}")
            elif ctype == "auth_required":
                print(f"[+{dt:>6.0f}ms] AUTH REQUIRED: {chunk.get('name')} — auto-denying")
                gen_send = getattr(chunk, "_send", None)
                # The generator-send pattern requires the caller; trace
                # harness just continues, which the agent treats as deny.
            elif ctype == "skill_offer":
                draft = chunk.get("draft") or {}
                print(f"[+{dt:>6.0f}ms] 💡 SKILL_OFFER: name={draft.get('name')!r}")
            else:
                print(f"[+{dt:>6.0f}ms] {ctype}: {_short(chunk, 100)}")
    except Exception as e:
        print(f"\n!!! exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    total_ms = (time.time() - t0) * 1000
    print()
    print("══════════════════════════════════════════════════════════════════")
    print("SUMMARY")
    print("══════════════════════════════════════════════════════════════════")
    print(f"  total latency : {total_ms:>7.0f}ms ({total_ms/1000:.1f}s)")
    print(f"  architect steps: {step_idx}")
    print(f"  tools called   : {tools_called}")
    if plan_state:
        done = sum(1 for t in plan_state.tasks if t.status == "done")
        print(f"  plan           : {done}/{len(plan_state.tasks)} done — {plan_state.title}")
    print()
    print("FINAL RESPONSE:")
    print("─" * 70)
    print("".join(final_response_parts)[:2500])
    print("─" * 70)


if __name__ == "__main__":
    main()
