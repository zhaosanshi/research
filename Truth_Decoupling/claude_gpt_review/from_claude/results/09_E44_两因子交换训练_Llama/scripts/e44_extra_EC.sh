#!/usr/bin/env bash
# E+C on the plain prompt x (from_claude/36 section 3, written before any generation). Reviewed and repaired evaluation
# script e44_eval_extra.py (4438a282, GPT round 34); its real-model selfcheck passed on all 192 tuning questions.
#   STAGE=gen1 GPU=k        E fixed at E10.w.p2; C31.w.m{0.25,0.5,1}; single-write arms for reproduction checks and the
#                           missing single-E random controls at strength 2; then the read-only checks and the choice of k
#   STAGE=gen2 GPU=k KC=k   matched controls for the chosen C strength only
#   STAGE=judge GPU=k       one export of every full eval_extra generation + baseline x; AllenAI on card k, then Pro
# Authorisation for Pro: 赵磊 11-10 17:21 and 23:34 (both judges). Runs inside the container.
set -euo pipefail
: "${STAGE:?}" "${GPU:?}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
ROOT=/workspace/Truth_Decoupling
MAIN=$ROOT/results_e44_factorial_das/llama3.1-8b
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
CODE=$ROOT/review_e44_claude/claude_gpt_review/from_claude/code
W=$ROOT/review_e44_claude
COMMON=(--source-run "$SRC" --run-dir "$MAIN" --split tune --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
step() { echo "=== $(date '+%F %T') [E+C $STAGE] $* ==="; }
cd "$CODE"
sha256sum scripts/e44_eval_extra.py configs/prompts/e44_eval_extra_v1.json
case $STAGE in
  gen1)
    ARMS="x:E10.w.p2+C31.w.m0.25 x:E10.w.p2+C31.w.m0.5 x:E10.w.p2+C31.w.m1 x:E10.w.p2 x:C31.w.m0.25 x:C31.w.m0.5 x:C31.w.m1 x:E10.rand.p2 x:E10.rand.m2"
    step "generate $ARMS"
    python scripts/e44_eval_extra.py generate "${COMMON[@]}" --arms $ARMS
    step "checks"
    python "$W/e44_extra_EC_check.py" "$MAIN"
    ;;
  gen2)
    : "${KC:?}"
    ARMS="x:E10.rand.p2+C31.w.m$KC x:E10.rand.m2+C31.w.m$KC x:E10.w.p2+C31.rand.m$KC x:E10.w.p2+C31.rand.p$KC x:C31.rand.m$KC x:C31.rand.p$KC"
    step "generate $ARMS"
    python scripts/e44_eval_extra.py generate "${COMMON[@]}" --arms $ARMS
    ;;
  judge)
    export DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key
    NAME=ec_tune
    WORK=$MAIN/eval_extra/judge_tune_$NAME
    step "export every full eval_extra generation + baseline x"
    python scripts/e44_eval_extra.py export --source-run "$SRC" --run-dir "$MAIN" --split tune --name "$NAME" --overwrite | tail -n 3
    mkdir -p "$WORK/judge2/cache"
    cp "$MAIN/eval_extra/judge_tune/$NAME.json" "$WORK/judge_input.json"
    # identical answers keep identical Pro labels: start from the cache of the run that judged baseline x and x.w.p2 at L10
    cp -n "$MAIN/judge_tune_proB_E_L10/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl" "$WORK/judge2/cache/" || true
    step "cells"
    python scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
    step "allenai on card $GPU"
    python scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
      --truth-model "$ROOT/models/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$ROOT/models/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 1
    python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$WORK/single_allenai_stdout.md"
    step "pro (two questions per unique cell)"
    python scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
    python scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$WORK/single_pro_stdout.md"
    ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
step done
