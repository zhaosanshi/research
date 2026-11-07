#!/bin/bash
# Claude, 2026-11-07: reproduce GPT round 19's two Llama-3.1-8B GPU diagnostics with the GitHub code (63e3674 snapshot).
#   GPU0  e27_trace_corruption.py, six tune questions, release DAS-init adapter (models/real_trace_review.md)
#   GPU1  e27_pure_das_layer_controls.py generate, all 192 tune questions, original DAS at L10 / L29 (models/pure_das_layer_review.md)
# Generation only: no judge phase, no training. usage: bash r19_reproduce_llama.sh
# Future usage requires CODE_PATH and a fresh RESULTS_PATH; prior logs/results are preserved.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/r19_run_common.sh"
r19_init LLAMA llama_trace llama_pure_das_tune192
r19_launch 0 llama_trace scripts/review_checks/e27_trace_corruption.py \
  --source-run "$ROOT/results_qa_das_v1u/llama3.1-8b/full" \
  --adapter-run "$ROOT/results_e27_release_20261106/llama3.1-8b/ce_alpha3_ep3/das_s42" \
  --split tune --pair-ids tqa:138 tqa:150 tqa:412 tqa:136 tqa:647 tqa:685 --out "$OUT/llama_trace.json"
r19_launch 1 llama_pure_das_tune192 scripts/review_checks/e27_pure_das_layer_controls.py generate \
  --source-run "$ROOT/results_qa_das_v1u/llama3.1-8b/full" \
  --run-dir "$OUT/llama_pure_das_tune192" --all-questions
r19_wait_all
