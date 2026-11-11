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
  # ---- the complete-sentence background (from_claude/37 section 2, written before these runs)
  genF)    # E+C and its two single writes on x_F; needs the selfcheck for base x_F (e44_extra_selfcheck.sh ARM=x_F:...)
    ARMS="x_F:E10.w.p2+C31.w.m0.25 x_F:E10.w.p2 x_F:C31.w.m0.25"
    step "generate $ARMS"
    python scripts/e44_eval_extra.py generate "${COMMON[@]}" --arms $ARMS
    ;;
  genCF)   # must-answer E on the must-answer + complete-sentence prompt (33 section 1.7 A; GPT round 34 A2)
    ARMS="conf_F:E10.w.p1 conf_F:E10.w.p2 conf_F:E10.rand.p2 conf_F:E10.rand.m2"
    step "generate bases conf_F conf_E1_F; arms $ARMS"
    python scripts/e44_eval_extra.py generate "${COMMON[@]}" --bases conf_F conf_E1_F --arms $ARMS
    ;;
  judgeF)
    export DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key
    NAME=ecF_tune
    WORK=$MAIN/eval_extra/judge_tune_$NAME
    GEN="x_F--E10-w-p2-and-C31-w-m0-25_tune.json x_F--E10-w-p2_tune.json x_F--C31-w-m0-25_tune.json"
    GEN="$GEN conf_F--E10-w-p1_tune.json conf_F--E10-w-p2_tune.json conf_F--E10-rand-p2_tune.json conf_F--E10-rand-m2_tune.json"
    step "export $GEN + prompt control conf_E1_F"
    python scripts/e44_eval_extra.py export --source-run "$SRC" --run-dir "$MAIN" --split tune --gen $GEN --prompts-out conf_E1_F \
      --name "$NAME" --overwrite | tail -n 3
    mkdir -p "$WORK/judge2/cache"
    cp "$MAIN/eval_extra/judge_tune/$NAME.json" "$WORK/judge_input.json"
    cp -n "$MAIN/eval_extra/judge_tune_ec_tune/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl" "$WORK/judge2/cache/" || true
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
  # ---- 33 section 1.7 B and D as written there (GPT round 34 A2 / B1: four cells, real component amounts; natural is
  #      a descriptive diagnostic only, never an identity verdict)
  genS)    # the split of the learned write at layer 10 on conf; full w regenerated as a reproduction check of the driver
    ARMS="conf:E10.wpar.p2 conf:E10.wperp.p2 conf:E10.mdg.p2 conf:E10.wpar.p1 conf:E10.wperp.p1 conf:E10.mdg.p1 conf:E10.w.p2 conf:E10.w.p1"
    step "generate $ARMS"
    python scripts/e44_eval_extra.py generate "${COMMON[@]}" --arms $ARMS
    ;;
  judgeS)
    export DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key
    NAME=split_tune
    WORK=$MAIN/eval_extra/judge_tune_$NAME
    GEN="conf--E10-wpar-p2_tune.json conf--E10-wperp-p2_tune.json conf--E10-mdg-p2_tune.json conf--E10-wpar-p1_tune.json"
    GEN="$GEN conf--E10-wperp-p1_tune.json conf--E10-mdg-p1_tune.json conf--E10-w-p2_tune.json conf--E10-w-p1_tune.json"
    step "export $GEN"
    python scripts/e44_eval_extra.py export --source-run "$SRC" --run-dir "$MAIN" --split tune --gen $GEN --name "$NAME" --overwrite | tail -n 3
    mkdir -p "$WORK/judge2/cache"
    cp "$MAIN/eval_extra/judge_tune/$NAME.json" "$WORK/judge_input.json"
    cp -n "$MAIN/eval_extra/judge_tune_ecF_tune/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl" "$WORK/judge2/cache/" || true
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
  nat)     # read-only natural differences; the training split carries the alignment check, the tuning split is reported only
    for sp in train tune; do
      step "natural E layers 8 10, split $sp"
      python scripts/e44_eval_extra.py natural --source-run "$SRC" --run-dir "$MAIN" --split $sp --batch-size 8 \
        --max-new-tokens 48 --max-target-tokens 24 --family E --layers 8 10 --pairs conf_E1:conf caut_E1:caut x_E1:x
      step "natural C layers 19 31, split $sp"
      python scripts/e44_eval_extra.py natural --source-run "$SRC" --run-dir "$MAIN" --split $sp --batch-size 8 \
        --max-new-tokens 48 --max-target-tokens 24 --family C --layers 19 31 --pairs caut:conf caut_E1:conf_E1
    done
    ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
step done
