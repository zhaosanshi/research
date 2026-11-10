#!/usr/bin/env bash
# E43 full run on Llama-3.1-8B (plan 21 v4, route 乙). Code = GPT-reviewed e89b0d8 byte for byte (no Claude additions).
# Single GPU (vLLM service stopped 11-10 15:28 on 赵磊's word), production caps 48/24, batch 8.
# Order: check -> freeze (96 + 192) -> train E/C L16-31 -> train E/C L0-15 -> generate tune (main arms) -> generate train
#        -> summarize -> random norm-matched control arms -> summarize. Judging (AllenAI first) is a separate step.
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
RUN=/workspace/Truth_Decoupling/results_e43_qa_prompt_das/llama3.1-8b
SRC=/workspace/Truth_Decoupling/results_qa_das_v1u/llama3.1-8b/full
CODE=/workspace/Truth_Decoupling/review_e43_claude/claude_gpt_review/from_claude/code
COMMON=(--source-run "$SRC" --run-dir "$RUN" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
LATE="16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
EARLY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
E_ARMS="none main recover md_main"
C_ARMS="none main to_conf conf_to_caut caut_to_conf md_main"
cd "$CODE"
mkdir -p "$RUN"
step() { echo "=== $(date '+%F %T') $* ==="; }
das() { python scripts/e43_qa_prompt_das.py "$@"; }

step "check E L16"
das check "${COMMON[@]}" --family E --layers 16 --pair-ids tqa:765
step "check C L16"
das check "${COMMON[@]}" --family C --layers 16 --pair-ids tqa:765

step "freeze train + tune (step 0: four prompt controls)"
das freeze "${COMMON[@]}" --splits train tune

step "train E late layers"
das train "${COMMON[@]}" --family E --layers $LATE
step "train C late layers"
das train "${COMMON[@]}" --family C --layers $LATE
step "train E early layers"
das train "${COMMON[@]}" --family E --layers $EARLY
step "train C early layers"
das train "${COMMON[@]}" --family C --layers $EARLY

step "generate tune: E late"
das generate "${COMMON[@]}" --family E --layers $LATE --split tune --arms $E_ARMS
step "generate tune: C late"
das generate "${COMMON[@]}" --family C --layers $LATE --split tune --arms $C_ARMS
step "generate tune: E early"
das generate "${COMMON[@]}" --family E --layers $EARLY --split tune --arms $E_ARMS
step "generate tune: C early"
das generate "${COMMON[@]}" --family C --layers $EARLY --split tune --arms $C_ARMS

step "generate train (in-sample diagnostic): E all"
das generate "${COMMON[@]}" --family E --layers $LATE $EARLY --split train --arms $E_ARMS
step "generate train (in-sample diagnostic): C all"
das generate "${COMMON[@]}" --family C --layers $LATE $EARLY --split train --arms $C_ARMS

step "summarize (main arms)"
das summarize --source-run "$SRC" --run-dir "$RUN"

step "random norm-matched control: E tune"
das generate "${COMMON[@]}" --family E --layers $LATE $EARLY --split tune --arms random_main
step "random norm-matched control: C tune"
das generate "${COMMON[@]}" --family C --layers $LATE $EARLY --split tune --arms random_main

step "summarize (with random controls)"
das summarize --source-run "$SRC" --run-dir "$RUN"
step "full run done"
