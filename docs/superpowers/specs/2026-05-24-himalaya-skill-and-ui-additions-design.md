# 2026-05-24 — Himalaya skill test + three TUI additions

## Background

The user asked for two things in one turn:
1. **Validate the existing skill mechanism** by running two prompts through the agent:
   - "create a new skill for reading emails using himalaya. Himalaya is already configured with my email."
   - "check my recent emails"
   The expectation is that prompt 1 produces a markdown skill file in `~/.ezclaw/skills/` via the `learn_skill` tool, and prompt 2 picks it up via embedding-based skill matching (`agent.match_skills`) and ends up invoking `himalaya envelope list` through `run_shell`.
2. **Add three TUI additions** to make per-turn cost and current code-writing activity legible at a glance.

The agent layer (`MultiAgentSystem` in `multi_agent.py`) is exercised through a throwaway harness at `/tmp/ezclaw_smoke.py` — the TUI cannot be driven from a non-interactive shell, and the user explicitly opted for the direct path so we get results in seconds rather than minutes of pasting.

## Test runs (no implementation work)

Driver: `/tmp/ezclaw_smoke.py` instantiates `MultiAgentSystem()` against the real `.env` / `ezclaw.db` / `~/.ezclaw/skills/`, runs both prompts in sequence on the same session, and prints chunk events + a summary.

Pass criteria:
- Prompt 1 ends with at least one new file under `~/.ezclaw/skills/` whose content mentions `himalaya`.
- Prompt 2's transcript shows either a `get_skill`-ish tool call **or** a skill-match notice in reasoning, AND a `run_shell` call whose args contain `himalaya`.

Fail modes the test should make obvious (no work done, just reported):
- The agent confuses "skill" for "Python file" and writes Python to `./workspace/` instead of calling `learn_skill`.
- Skill matching threshold (default 0.4 cosine in `agent.match_skills`) doesn't fire for "check my recent emails" against a skill described as "reading emails using himalaya."
- `himalaya` runs but the result is empty or malformed and the agent doesn't surface a summary.

Whatever happens, the result is reported back to the user as-is. Fixes to the skill mechanism are out of scope for this spec.

## Feature 1 — Per-response cook time

**What**: at the end of every assistant message, append a small dim annotation like `· 12.3s` showing wall-clock time from prompt submission to final chunk.

**Where**:
- `cli.py:138` — `self.generation_start_time` already exists.
- `cli.py:2438` — `final_renderable = self._get_current_renderable_ansi()` is where the turn freezes into history. Wrap or extend so the cook time is part of the rendered bubble before appending.
- `cli.py:998` — `_get_current_renderable_ansi` is the right function to modify so the annotation also shows during streaming (the in-flight time already exists in the status bar, but having it under the bubble too matches the user's "next to all responses" wording).

**Format**: `· 12.3s` in `style="dim italic"`, right-aligned on its own line under the bubble. No box, no label — minimal.

**Edge case**: halted turns don't get an appended cook time (the turn isn't really "done").

## Feature 2 — Status bar: session tokens + electricity estimate

**What**: extend the status bar with two new segments: total tokens used this session and estimated energy in Wh.

**Where**:
- `cli.py:138-139` — add `self.session_tokens_in = 0`, `self.session_tokens_out = 0`, `self.session_energy_wh = 0.0`.
- `cli.py:2343` (content chunk branch) — tally `len(chunk["content"]) // 4` into `session_tokens_out`. Tally `len(user_input) // 4` once per turn at handle_input as a coarse `session_tokens_in`. This is a heuristic, not exact — `prompt_eval_count` and `eval_count` from Ollama aren't currently surfaced through `chat_stream`, and plumbing them through is out of scope. The status bar gets a `~` prefix so the estimate is honest.
- `cli.py:2431` (end-of-turn) — add `elapsed * GPU_TDP_WATTS / 3600` to `session_energy_wh`. GPU TDP defaults to 320W (RTX 4080 — matches the README's hardware section); override with `EZCLAW_GPU_TDP_W` env var.
- `cli.py:770` `_get_status_text` — append two new segments after the existing model/msgs/elapsed group:
  - `"~T 1.2k"` (rounded; "k" for ≥1000, raw int otherwise)
  - `"⚡ 0.42 Wh"` (two decimals)
  Both use the existing dim/divider styling pattern.

**Reset**: tokens/energy reset when the user runs `/clear`. They persist for the lifetime of the session (matches what "session" means everywhere else in the codebase).

**Honest caveats**:
- Tokens are a char/4 estimate, not real. The bar prefix `~T` is the signal.
- "Electricity" is wall-clock generation time × GPU TDP, which over-counts (GPU idles between chunks, daemon shares load, embedding calls aren't separated) and under-counts (CPU + memory not modeled). It's a Fermi estimate, not a wattmeter. Good enough for "this turn was expensive" intuition; not good enough for billing.

## Feature 3 — Right-side editor panel

**What**: when the agent calls `write_file`, a fixed-width right column shows the file path and content as it streams in. Closes when the turn ends or when no write is active.

**Where**:
- `cli.py:724` `_create_layout` — wrap `Frame("EzClaw Chat", history_window)` in a `VSplit` with the new `editor_window` to its right. Editor width: 50 columns, gated by a `Condition` on `self._show_editor`.
- `cli.py:2393` (tool_start branch) — when `chunk["name"] == "write_file"`, set `self._active_editor = {"path": chunk["arguments"].get("path") or chunk["arguments"].get("file_path"), "content": chunk["arguments"].get("content", ""), "started": time.time()}` and `self._show_editor = True`.
- `cli.py:2431` (end-of-turn) — `self._show_editor = False`, `self._active_editor = None`.
- New method `_render_editor_panel()` returns ANSI for the editor pane: title bar with path, body with content rendered via `Syntax` (rich) using a language guessed from extension. Truncate to last N lines if content exceeds panel height.

**Important wrinkle — content arrives all at once**: `write_file` in this codebase is **atomic** (the args object includes the full content on `tool_start`, not streamed chunks — confirmed in tools.py:619-646). So "live as the agent types" is not actually achievable without plumbing token-level stream interception, which is out of scope. The editor renders the **final intended content as soon as the tool is called**, which is the closest honest analog. The panel title reads "Writing:" rather than "Typing:" to match what's actually happening.

**Toggle**: F5 toggles `self._show_editor`. Status bar gets a `⊟ EDITOR` badge when on.

**Fallback**: if a tool fires that isn't `write_file` (e.g. `read_file`, `run_shell`), the editor panel keeps showing the last `write_file` until the turn ends. This avoids flicker.

## Order of work

1. Run the smoke test, report results to the user. (already in flight)
2. Implement Feature 2 (status bar) first — smallest blast radius, isolated to `_get_status_text` and two ints.
3. Implement Feature 1 (cook time) next — touches the assistant-bubble rendering, which is one function.
4. Implement Feature 3 (editor panel) last — touches the layout root, biggest risk of breakage.
5. Smoke-test each in turn by importing the changed module and rendering a sample frame to ANSI (existing tests/ pattern is unit-level; we can add one render test per feature).

## Out of scope (explicit)

- Token plumbing through `chat_stream` to get exact `eval_count` from Ollama.
- A real wattmeter integration (`nvidia-smi --query-gpu=power.draw`) instead of the TDP heuristic.
- Streaming write_file character-by-character.
- Fixing the skill mechanism if the smoke test reveals a regression — that's its own follow-up.
