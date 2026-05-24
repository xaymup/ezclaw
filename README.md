# EzClaw Ollama CLI

A terminal-based assistant for Ollama with a multi-agent orchestrator, persistent memory, sandboxed tool execution, and a TUI that shows the agent's plan as it works.

## Why EzClaw?

EzClaw is a focused local agent that aims for "just works" without the configuration sprawl of larger frameworks.

- **Zero-fuss configuration**: pick models in `.env`, run `python cli.py`.
- **Multi-agent by default**: an architect plans, then specialist agents (executor / researcher / debugger / general) carry out tasks. The plan stays visible in the TUI.
- **Native memory**: SQLite + embeddings, no extra services.
- **Sandboxed workspace**: tools operate from `./workspace/` so the agent can't accidentally write into your project source.

## Features

- **Plan-first orchestration** — the architect produces a structured task list (2-7 tasks) for multi-step requests, then executes against it. The TUI shows the plan with live `pending → in_progress → done` transitions.
- **Persistent chat history** — every conversation is saved in SQLite (`ezclaw.db`).
- **Long-term memory** — explicit `remember` / `recall` / `forget` tools, plus a hybrid full-text + cosine-similarity retrieval layer.
- **Tools** — `read_file`, `write_file`, `apply_diff` (unified-diff applier with fuzzy line-number recovery, explicit hunk-count + add/remove counts on success), `list_dir`, `run_shell` (interactive- and background-safe), `web_fetch`, `learn_skill`, plus memory tools. All file operations sandboxed to `./workspace/`.
- **TUI** — full-screen prompt-toolkit + rich layout with role-aware spinners, compact tool panels (F4 toggles full), strategy panel (F3 toggles), copy mode (F2), and per-response cook-time annotation under every assistant bubble.
- **Unified Plan panel** — one bordered panel renders the active task tree with tool calls nested under their step. The architect's `goal / observation / critical_thinking` for the current task appears inline under the in-progress row — no separate Reasoning timeline duplicating the task descriptions.
- **Right-side editor pane with tabs** (F5 toggle, F6/F7 cycle) — every `write_file` and `apply_diff` the agent makes opens a tab; content reveals character-by-character as the agent "types." Multiple files can be open at once; the active tab is bright, others dim.
- **Interactive shell pane** — when a tool launches a subprocess that wants stdin (`pacman`, `vim`, etc.), a dedicated pane appears outside the chat box showing the live stdout. Keystrokes route to the subprocess (Esc toggles back to chat).
- **Multiline input + paste-as-block** — the prompt is a real multiline editor. Newline keys: `Shift+Enter` (kitty / iTerm2 / WezTerm / modern xterm via CSI u or modifyOtherKeys), `Ctrl+J` (universal), `Alt+Enter`, or `\` + Enter (terminal-independent escape). Large pastes (≥3 lines or ≥200 chars) collapse to a single-line placeholder `[pasted #N: L lines, C chars]`; the real content is expanded back at submit. Keeps the input compact even when you paste 200 lines.
- **Status bar** — single line with state glyph, model emoji per family (🐳 deepseek, 🧧 qwen, 💎 gemma, 🦙 llama, 🌬 mistral, 🔬 phi, 🛠 coder variants, 🧮 embedders, 🤖 default), msg count, ~tokens, energy estimate (Wh based on GPU TDP), active-toggle badges, and a subtle one-line reflection from the agent (`/wisdom` refreshes; cached 15 min).
- **Per-tool authorization** — the security-check panel offers `[Y] allow this tool` (session-wide for that tool), `[O] allow once`, `[N] deny`, `[A] allow all tools` — pressing Y no longer re-prompts on the next call to the same tool.
- **Skill mechanism** — `learn_skill(name, description, procedure)` saves a markdown procedure to `~/.ezclaw/skills/`. Embedding-based matching surfaces relevant skills automatically on subsequent turns. Architect routes skill-creation requests to a single `learn_skill` call; a hard guard prevents re-iterating "improve the procedure" loops after a successful save.
- **Scheduled tasks (with recurrence)** — `schedule_task(time, description, recurrence=...)` queues a task in `heartbeat.md`. Recurrence accepts `every Nm/Nh/Nd`, `hourly`, `daily`, `weekly`, or `weekdays`. On firing the agent auto-executes the description as if you'd typed it. Recurring tasks re-arm to the next occurrence after each successful run.
- **Direct command shortcuts** — `list tasks`, `show skills`, `system info`, `what time is it`, etc., resolve to a single tool call with zero LLM involvement. Typed verbatim → answer in ~50ms instead of routing through the architect.
- **Loop detection + auto-pivot** — when the architect gets stuck (same plan + same agent + no progress), it auto-retries once at a higher temperature instead of giving up. Also halts and surfaces a notice when the architect routes to `general` three times in a row (clarification-loop guard).
- **Whimsical status vocabulary** — 150+ built-in playful crab/coastal status phrases across 11 categories ("scuttling over", "shellgazing", "claw-tapping the diagram"). Pool can be expanded with `/phrases refresh`, which asks the running LLM to brainstorm fresh additions and persists them to `~/.ezclaw/phrase_pool.json` so subsequent sessions inherit them.

## Recommended Multi-Agent Setup

Multi-agent mode runs four specialist roles on local Ollama. The setup below is what this repo is tested with on an RTX 4080 (16 GB VRAM) — total disk footprint ~25 GB, peak resident VRAM ~12 GB (one 14B model loaded at a time, hot-swapped per role).

```bash
# Specialist models (pull once)
ollama pull qwen3:14b           # architect + executor (default)
ollama pull qwen3.5:9b          # researcher + general (smaller, faster handoffs)
ollama pull deepseek-r1:14b     # debugger (chain-of-thought root-cause analysis)

# Embedding model — load-bearing, not optional. See "Why embeddings matter" below.
ollama pull mxbai-embed-large
```

Then in `.env`:

```ini
ENABLE_MULTI_AGENT=true
OLLAMA_MODEL=qwen3:14b               # executor
OLLAMA_ARCHITECT_MODEL=qwen3:14b
OLLAMA_RESEARCHER_MODEL=qwen3.5:9b
OLLAMA_DEBUGGER_MODEL=deepseek-r1:14b
OLLAMA_GENERAL_MODEL=qwen3.5:9b
OLLAMA_EMBED_MODEL=mxbai-embed-large
OLLAMA_NUM_CTX=16384
OLLAMA_KEEP_ALIVE=60m                # avoid re-loading between turns
```

**Variations:**
- **Lower VRAM (12 GB)**: swap `qwen3:14b` → `qwen3.5:9b` for the architect/executor too, or use `qwen2.5-coder:14b-q4_K_M` quantized.
- **Faster planning, no local compute**: set `ARCHITECT_PROVIDER=deepseek` with a free [DeepSeek API key](https://platform.deepseek.com/). The architect runs in the cloud, executor stays local.
- **Heavier reasoning**: replace the architect with `phi4-reasoning:plus` if you have headroom; it's slower but produces tighter plans.

## Performance

Measured on **NVIDIA RTX 4080 (16 GB VRAM)** with `OLLAMA_NUM_CTX=4096`, fixed coding prompt ("Write a Python function `count_words(path)`..."), temperature 0.0, one warmup pass discarded per model.

| Model | TTFT | Total | Output chars | Approx tok/s |
|-------|-----:|------:|-------------:|-------------:|
| `qwen3:14b` | 18.9s | 19.6s | 194 | ~62 |
| `qwen3.5:9b` | 22.5s | 23.0s | 163 | ~60 |
| `deepseek-r1:14b` | 6.6s | 13.2s | 1,925 | ~63 |

Notes on the numbers:
- **TTFT** is wall time to the first streamed token. High TTFT on `qwen3:14b` / `qwen3.5:9b` here reflects short outputs where prompt-eval dominates; on longer responses the ratio inverts.
- **Approx tok/s** counts streamed message chunks. For these models that's ~1 token per chunk; treat it as a rough estimate, not a precise per-token throughput.
- `deepseek-r1:14b` produces verbose chain-of-thought output, so it generates ~10× more characters from the same prompt — the higher TTFT-to-total ratio reflects more useful generation per call.
- These numbers cover the **inference** path only. End-to-end perceived latency in the TUI also includes prompt assembly, memory lookups, and embedding calls; in practice a typical multi-step task hits the architect 3-8× per user turn, so plan-stage latency compounds.

If you want to benchmark your own hardware, the `/tmp/_bench.py` style is simple: stream a fixed prompt with `ollama.Client(...).chat(stream=True)`, time first-chunk and last-chunk, divide chunk count by generation time.

## Why Embeddings Matter

A common local-agent shortcut is to skip embeddings and rely on keyword/full-text search alone for memory and recall. EzClaw doesn't: `mxbai-embed-large` (or a swap-in alternative) is **load-bearing across four places in the system**, not just memory.

1. **Hybrid memory search.** `Database.search_memories_hybrid(query, alpha=0.6)` blends 60% cosine similarity against query embeddings with 40% SQLite FTS5 keyword match. Pure keyword retrieval misses paraphrased recall ("my preferred editor" vs. "I use neovim"); pure vector retrieval misses exact-name lookups ("file path `/etc/foo.conf`"). The hybrid covers both. The architect calls this *every turn* before deciding what to do, so embedding quality directly shapes routing.
2. **Skill matching.** `match_skills(task_context, self.skills)` embeds the current task and finds saved skills (markdown procedures in `skills/`) by similarity. When you `learn_skill("deploy", ...)` once, the next "push the app to staging" routes through that skill automatically — no keyword overlap needed.
3. **Routing memory.** `Database.search_similar_routing` looks up how previous similar requests were routed (executor / researcher / debugger / general). When the architect classifies a new request, it sees "the last 3 requests semantically close to this one all routed to executor" as a strong prior. Without embeddings this would have to be a brittle keyword classifier in the prompt.
4. **Experience recall.** `Database.search_experiences` retrieves notes from past *outcomes* — what worked, what failed — for tasks similar to the current one. A failed `pip install` on a stale lockfile two weeks ago surfaces when you hit a similar dependency issue now, without you having to remember it.

**Why `mxbai-embed-large`?** It runs locally on Ollama (one-time 669 MB pull), produces 1024-dim vectors, ranks well on MTEB for retrieval, and the GPU resident cost is small (~770 MB VRAM observed via `ollama ps` on this machine). Embedding calls cost far less than chat calls — they don't show up in the per-turn timings in the Performance section above. Swap targets if you need to fit in less VRAM: `nomic-embed-text` (137M params, ~274 MB), or `all-minilm` (45M, ~91 MB) at lower retrieval quality.

**What happens if you skip it.** Set `OLLAMA_EMBED_MODEL=` empty and ezclaw falls back to FTS5-only memory. You'll see: routing decisions stop reusing past wisdom (every turn looks "novel" to the architect), skills you've saved become invisible unless you keyword-match exactly, and the agent re-discovers the same workarounds it found yesterday. Functional but markedly less aware.

## Design Choices

Why ezclaw looks different from a typical single-loop ReAct agent (no comparative performance claims — just the design decisions and what they trade off):

1. **Separate planning pass.** Most local-agent CLIs run a single ReAct loop where the model produces "thought → action → observation" each step. ezclaw's architect first produces a structured task list (`Plan` with 2-7 tasks), *then* execution loops dispatch sub-agents against that plan. Trade-off: an extra LLM call up front; payoff is a stable user-visible plan and resilience to the executor losing context mid-task.

2. **Specialist routing.** The architect picks which sub-agent runs each step (executor / researcher / debugger / general). Each sub-agent has its own system prompt and can use a different model. Trade-off: more models to manage and prompt-engineer; payoff is that the debugger model (`deepseek-r1:14b` here) only loads when you actually need root-cause analysis, not on every shell command.

3. **Sandboxed workspace.** `read_file`/`write_file`/`run_shell` all operate from `./workspace/`. An absolute path to the project source (e.g. `/home/you/projects/foo/main.py`) returns a clear `WorkspacePathError` instead of silently mangling the path. This was a recurring bug class until [tools.py:88](tools.py#L88) was hardened to raise.

4. **Plan-aware loop detection.** Naive "same tool called twice = halt" detectors false-trip on legitimate retries (re-read a file after a write; re-run `make` to verify a fix). ezclaw's detector counts only stuck repeats — the same `(agent, plan)` pair with zero tool calls and zero output — and on the third strike auto-retries at temperature 0.7 with an explicit "pivot" nudge before halting. See `multi_agent.py` `MultiAgentSystem.run`.

5. **TUI as the source of truth.** The plan, the routed agent, the active spinner, and the tool execution panels all read from a shared `theme.py` palette and `Plan` snapshots streamed as chunk events. Toggles (`F2` copy, `F3` strategy, `F4` compact tools) flip the view, not the underlying state. Trade-off: more `rich` rendering per UI tick; payoff is consistent visuals without per-component theming.

6. **No backwards-compatibility shims.** The architect's prompt and intent shape have been rewritten three times this branch. Old fields (`category`, `pivot_reasoning`, free-form `plan` string) were deleted outright rather than kept as deprecated. Trade-off: a contributor reading the system prompt sees one canonical schema; payoff is the architect doesn't get confused by stale fields the prompt no longer documents.

For factual comparisons against other local-agent projects (opencode, Hermes-based agents, Aider, etc.), I'd rather you measure than I assert. The benchmark methodology above is the same template — pull their CLI, run the same prompt, compare TTFT and tok/s on identical hardware.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Ollama:** ensure the daemon is running, then pull the models in the [Recommended Multi-Agent Setup](#recommended-multi-agent-setup) section above.

3. **Configure ezclaw:**
   ```bash
   cp .env.example .env
   # edit .env if your model names differ
   ```

4. **Run:**
   ```bash
   python cli.py
   ```

## Usage

Sample prompts:
- "Add a Python script that lists every function in tools.py with its docstring summary." → multi-step, triggers the plan panel.
- "What's the modified date of cli.py?" → single-step, no plan panel.
- "Remember that my name is Alice and I like coffee." → memory write.
- "Who am I?" → memory recall in a new session.
- `list tasks` / `show skills` / `system info` / `what time is it` → direct shortcuts (no LLM call, instant).
- "Water the plants every weekday at 10am." → recurring scheduled task; `unschedule_task(<id>)` to stop.

Keyboard:
- **Enter** submits, **Ctrl+J** (or **Alt+Enter** / **Shift+Enter** / `\` + Enter) inserts a newline.
- **F1** help · **F2** copy mode · **F3** strategy panels · **F4** full vs compact tool panels · **F5** editor pane · **F6 / F7** cycle editor tabs · **Ctrl+R** reasoning visibility · **Ctrl+C** interrupt.
- Pasting ≥3 lines or ≥200 chars collapses to `[pasted #N: L lines, C chars]`; expanded back at submit.

Slash commands: `/help` `/settings` `/queue` `/skills` `/memory` `/phrases [refresh|reset]` `/wisdom` `/diagnose` `/clear` `/copy [last|all|N]` `/expand [N|all]` `/collapse [N|all]` `/authorize`.

## Project Structure

| File | Responsibility |
|------|----------------|
| `cli.py` | TUI (prompt-toolkit layout, panels, keybindings) |
| `multi_agent.py` | Architect + sub-agent orchestration, plan/execute loop |
| `agent.py` | Single-agent fallback, tool-call loop |
| `plan.py` | `Task` / `Plan` data model (pure data, fully unit-tested) |
| `theme.py` | Single source of truth for colors, role icons, spinner frames |
| `tools.py` | Tool registry + sandboxed file/shell/web implementations |
| `memory.py` | SQLite schema, hybrid full-text + vector search |
| `model_client.py` | Ollama and DeepSeek client construction |
| `embed.py` | Embedding helpers for `mxbai-embed-large` |
| `tests/` | Unit tests for `plan.py` and architect smoke tests |

## License

See [LICENSE](LICENSE).
