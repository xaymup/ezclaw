#!/usr/bin/env bash
# Per-role model sweep — runs the `general`-tagged audit subset against
# each candidate model and writes a separate report per model.
#
# After running, the audit_report_general_<MODEL>.md files can be
# diffed/compared to pick the strongest general agent. Each candidate
# takes ~12 minutes (14 tests × ~50s avg). Total ≈ 70 min for 6 models.
#
# Architect + executor stay fixed at qwen3:14b (the audited default)
# so we isolate the effect of swapping JUST the general model.
#
# Usage:  scripts/sweep_general.sh
# Stop:   Ctrl+C — the next loop iteration will not start.
set -euo pipefail

cd "$(dirname "$0")/.."

CANDIDATES=(
    "qwen3.5:9b"        # baseline
    "qwen3:14b"         # bigger, also our executor
    "llama3.1:8b"       # Meta's tool-tuned chat model
    "gemma4:latest"     # Google's newest, multimodal-capable
    "gpt-oss:20b"       # OpenAI OSS — biggest candidate
    "mistral-nemo:12b"  # Mistral's tool-capable mid-size
)

echo "Sweep: ${#CANDIDATES[@]} candidates for the general role."
echo "Each runs the 'general' tag (14 tests). Wall ≈ 12 min/candidate."
echo

for model in "${CANDIDATES[@]}"; do
    # Sanitize model name for filename: ':' and '.' → '-'
    label="general_${model//[:.]/-}"
    echo "════════════════════════════════════════════════════════════"
    echo "→ general=${model}  (architect+executor stay qwen3:14b)"
    echo "  report: audit_report_${label}.md"
    echo "════════════════════════════════════════════════════════════"
    OLLAMA_GENERAL_MODEL="$model" \
        python scripts/audit.py --tag general --label "$label"
    echo
done

echo "Sweep complete. Reports:"
ls -1 audit_report_general_*.md 2>/dev/null
