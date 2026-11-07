#!/bin/bash
# Claude, 2026-11-07: execution-level reproduction of two GPT round-19 diagnostics on Qwen3-8B with the GitHub code
# (63e3674 snapshot in review_r19_claude/), written to a fresh directory; GPT's own outputs are only read afterwards.
#   GPU0  e27_trace_corruption.py   tqa:138 / 150 / 412, release DAS-init adapter (same command as unicode/review.md)
#   GPU1  e27_frozen_initial_controls.py generate, first three tune questions, GPT's early random / DAS adapters
# No judge phase, no training. usage: bash r19_reproduce.sh
# Future usage requires CODE_PATH and a fresh RESULTS_PATH; prior logs/results are preserved.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/r19_run_common.sh"
r19_init ALL qwen_trace qwen_random_pilot3
r19_launch 0 qwen_trace scripts/review_checks/e27_trace_corruption.py \
  --source-run "$ROOT/results_qa_das_v1u/qwen3-8b/full" \
  --adapter-run "$ROOT/results_e27_release_20261106/qwen3-8b/ce_alpha3_ep3/das_s42" \
  --split tune --pair-ids tqa:138 tqa:150 tqa:412 --out "$OUT/qwen_trace.json"
r19_launch 1 qwen_random_pilot3 scripts/review_checks/e27_frozen_initial_controls.py generate \
  --source-run "$ROOT/results_qa_das_v1u/qwen3-8b/full" \
  --random-run "$ROOT/results_e27_search_codex_20261106/qwen3-8b/ce_alpha3_ep3/random" \
  --das-run "$ROOT/results_e27_search_codex_20261106/qwen3-8b/ce_alpha3_ep3/das" \
  --run-dir "$OUT/random_controls/qwen3-8b/pilot3" --split tune --limit 3
r19_wait_all
