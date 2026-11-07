#!/bin/bash
# Claude, 2026-11-07: reproduce the generation phase of three GPT round-19 192-question tune runs with the GitHub code
# (63e3674 snapshot in review_r19_claude/), into a fresh directory; outputs are compared with GPT's afterwards.
#   GPU0  frozen-random five-arm control, Qwen3-8B   (GPT's early random / DAS adapters, as in round 19)
#   GPU1  frozen-random five-arm control, Llama-3.1-8B
#   GPU2  deployment-dose diagnostic, Qwen3-8B        (release DAS-init adapter, alpha 1 / 2 / 3)
# Generation only: no judge phase, no training. usage: bash r19_reproduce_full.sh
# Future usage requires CODE_PATH and a fresh RESULTS_PATH; prior logs/results are preserved.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/r19_run_common.sh"
r19_init FULL random_full192_qwen3-8b random_full192_llama3.1-8b dose_full192_qwen3-8b
EARLY=$ROOT/results_e27_search_codex_20261106
for M in qwen3-8b llama3.1-8b; do
  GPU=0; [ "$M" = llama3.1-8b ] && GPU=1
  r19_launch "$GPU" "random_full192_$M" scripts/review_checks/e27_frozen_initial_controls.py generate \
    --source-run "$ROOT/results_qa_das_v1u/$M/full" \
    --random-run "$EARLY/$M/ce_alpha3_ep3/random" --das-run "$EARLY/$M/ce_alpha3_ep3/das" \
    --run-dir "$OUT/random_controls/$M/full192" --split tune --all-questions
done
r19_launch 2 dose_full192_qwen3-8b scripts/review_checks/e27_generation_dose.py generate \
  --source-run "$ROOT/results_qa_das_v1u/qwen3-8b/full" \
  --adapter-run "$ROOT/results_e27_release_20261106/qwen3-8b/ce_alpha3_ep3/das_s42" \
  --run-dir "$OUT/generation_dose/qwen3-8b/full192" --all-tune
r19_wait_all
