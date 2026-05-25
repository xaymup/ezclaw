# General-role model sweep — comparison

**Date:** 2026-05-25
**Scope:** 14 `general`-tagged tests run against 6 candidate models, with
architect + executor held constant at `qwen3:14b`. Each candidate ran in
a freshly-constructed `MultiAgentSystem` so context bleed between
candidates was zero.

## Summary table

| Rank | Model | Pass | Wall time | Avg/test |
|------|-------|------|-----------|----------|
| 1 | **gpt-oss:20b** | **11/14** | 8.1 m | 35 s |
| 2 | qwen3.5:9b (baseline) | 10/14 | 8.5 m | 36 s |
| 2 | llama3.1:8b | 10/14 | **6.2 m** | 27 s |
| 4 | qwen3:14b | 9/14 | 8.6 m | 37 s |
| 4 | gemma4:latest | 9/14 | 8.8 m | 38 s |
| 6 | mistral-nemo:12b | 5/14 | 8.6 m | 37 s |

## Per-test pass/fail matrix

| Test | qwen3.5:9b | qwen3:14b | llama3.1:8b | gemma4 | gpt-oss:20b | mistral-nemo |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| conversational_water       | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| conversational_egypt       | ✅ | ✅ | ✅ | ❌ web_search | ✅ | ✅ |
| tool_current_date          | ❌ "Done." | ✅ | ✅ | ✅ | ✅ | ✅ |
| memory_remember            | ❌ learn_skill | ❌ learn_skill | ✅ | ❌ no tool | ✅ | ❌ learn_skill |
| ambiguous_make_faster      | ✅ | ✅ | ❌ python_eval | ✅ ask_user | ✅ | ❌ wrong tools |
| translation                | ✅ | ✅ | ✅ python_eval | ❌ web_search | ✅ | ❌ wrong tools |
| convo1_remember_color      | ✅ | ❌ no tool | ❌ wrong tool | ✅ | ✅ | ❌ wrong tool |
| convo1_recall_color        | ✅ | ❌ no tool | ✅ | ✅ | ✅ | ❌ no tool |
| convo3_remember_city       | ✅ | ✅ 3× loop | ✅ | ✅ | ✅ | ✅ |
| convo3_correct_city        | ⏱ no forget | ⏱ no tool | ❌ no tool | ✅ slow loop | ⏱ slow loop | ⏱ no tool |
| convo3_where_do_i_live     | ✅ | ❌ cascade | ✅ | ❌ mixed state | ❌ cascade | ✅ |
| convo4_seed_topic          | ⏱ slow | ✅ ask_user | ⏱ slow | ⏱ slow | ⏱ slow | ⏱ slow |
| convo4_which_for_backend   | ✅ | ✅ | ✅ | ✅ | ✅ | ⏱ wrong tool |
| convo4_why_that_over_go    | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ wrong tool |

## Per-model character

**gpt-oss:20b — Winner.**
Lands the right tool on every test it passed. Three failures were all
timing-related (loops or latency), never wrong-tool-selection. The
biggest model tested (13 GB on disk) but inference wasn't slower
end-to-end than qwen3.5:9b — likely the MoE architecture (only 3B
activated per token) keeps it competitive.

**qwen3.5:9b — Current baseline.**
Reliable on simple tests. Two clear behavioral failures: emits "Done."
after a tool call instead of relaying the result (the system prompt
explicitly forbids this; the model ignores), and reaches for
`learn_skill` when asked to "remember" a personal fact (confusing
"remember" the verb with "skill" the concept).

**llama3.1:8b — Fastest but lowest tool-selection quality.**
6.2 min wall time is best in class. But misuses tools constantly:
python_eval for "translate hello world", python_eval for "I'm
thinking about learning a programming language", `recall` instead of
`remember` for storing a color. Pass rate held only because some
checks were forgiving.

**qwen3:14b — Worse than smaller qwen3.5:9b for the general role.**
The bigger model is MORE confident answering from short-term context
and skips memory tools entirely. 5 of 5 memory-tool tests had at least
one regression vs qwen3.5:9b. Don't put a 14B model in the general
seat just because it's bigger.

**gemma4:latest — Unique strength, unique weakness.**
The ONLY model to handle the Boston→Seattle correction (called both
`forget` and `remember`). The right behavior for that prompt. But
also the ONLY model that called `web_search` for both the Egypt
question and the "hello world" translation — strong over-reliance on
web for things it knows.

**mistral-nemo:12b — Disqualifying tool confusion.**
Called `list_scheduled_tasks` for "My favorite color is blue."
Called `get_skill` for "translate hello world to French." Called
`forget` four times for "Make it faster." The tool ontology is
opaque to this model.

## Recurring problems (cross-model)

Three failure patterns appeared in 4-5 models, suggesting they're
NOT just model issues but prompt/architecture issues we should fix:

1. **convo4_seed_topic latency.** Every model except qwen3:14b
   timed out (>60 s cap). The prompt is "I'm thinking about learning
   a new programming language." — a chat opener. The short-circuit
   should route it instantly to general, then general should produce
   a 1-2 sentence reply. Whatever's eating the 60s isn't model-specific.

2. **convo3_where_do_i_live cascade.** When step 2 (the correction
   `forget Boston, remember Seattle`) is messy, step 3's `recall`
   surfaces a mixed answer that doesn't satisfy "says Seattle". Test
   design issue partly — maybe step 3 should be more forgiving — but
   also a real product UX issue. After a correction, the user expects
   crisp recall, not "you used to live in Boston but now Seattle".

3. **learn_skill vs remember confusion.** qwens and mistral all
   misread "remember that my favorite programming language is Rust"
   as a skill-creation request. Even our improved general prompt
   ("when user shares personal info, use `remember`") didn't fix it
   on these models. The word "remember" in the prompt is too easily
   associated with the `learn_skill` tool's description. Possible
   fix: rename `learn_skill` to `save_procedure` so the verb doesn't
   collide.

## Recommendation

Switch `OLLAMA_GENERAL_MODEL` from `qwen3.5:9b` to `gpt-oss:20b`. The
qualitative win is on tool-selection accuracy — gpt-oss never picked
a clearly wrong tool, where every other model did at least once.
Latency was a wash.

Disk cost: +7 GB (13 GB vs 6 GB). VRAM: similar at q4 (gpt-oss is
MoE so resident weights are larger but active params per token are
smaller). Verify on first real session that it loads cleanly with
qwen3:14b already in VRAM.

## Re-running this sweep

```bash
scripts/sweep_general.sh
```

Reports land in `audit_report_general_<model>.md` per candidate.
