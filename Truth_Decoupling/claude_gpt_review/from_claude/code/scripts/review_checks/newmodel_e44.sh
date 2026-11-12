#!/usr/bin/env bash
# Free-form E44 (two-factor interchange DAS) for ANY model config (Claude, 2026-11-12; plan from_claude/43).
# The Llama round-1 procedure (results/09_E44_两因子交换训练_Llama/scripts: e44_step0.sh, e44_step0_pro.sh, e44_round1.sh,
# e44_round1_extra_gen.sh, e44_round1_pro.sh) with nothing model-specific in the script. The model enters only through
#   * its source directory (make_e44_source.py: questions.json byte-identical to the old models', bank.json with this
#     model's config) and
#   * the interpreter named in its model config ("python"): used for the steps that load the model.
# Judging steps (export, cells, AllenAI, DeepSeek Pro, single, tables) run with the container's default python3, as they
# did for Llama, so the judges and their runtime are unchanged. Arms, prompts, caps and batch size are Llama's.
#   MODEL=qwen3.5-9b STAGE=<stage> [GPU=k] [BRANCH=E|C] [LAYERS="..."] [LAYER=n] [ARMS="..."] newmodel_e44.sh
# stages:
#   pilot     three-question execution check on layers L/8 and L/2 (BRANCH, GPU); own pilot run dir; no judging
#   step0     freeze the 11 prompts on train + tune, reference readout without writes, export, AllenAI (GPU)
#   step0pro  DeepSeek Pro on the step-0 tune cells (no GPU)
#   run       train -> generate the tune arms -> reference readout, for LAYERS (default all) of BRANCH (GPU)
#   judge     export every generation of BRANCH (all layers must be done) + AllenAI (GPU)
#   screen    the frozen shortlist rule on BRANCH's AllenAI table (CPU; e44_round1_tables.py screen, unchanged)
#   extra     extra arms for LAYERS of BRANCH (GPU). ARMS default: E -> Llama's six pre-Pro arms (E_EXTRA_A);
#             PURPOSE=dose -> Llama's strength-sweep arms (E_DOSE / C_DOSE)
#   pro       DeepSeek Pro + AllenAI on the same cells, ONE LAYER of BRANCH (GPU). TAG=A (default): Llama's proA arms and
#             prompts; TAG=B: Llama's proB (strength sweep) arms and prompts. ARMS / BASES override.
#   tables    frozen x-base cells and cross tables for LAYERS (proA work dirs), both judges; with BRANCH=E also the
#             exploratory must-answer tables (conf base, strength 1 on proA; strength 2 on proB for DOSE_LAYERS)
# Arm lists are copied from what Llama actually ran (review_e44_claude/chain_pro_E.sh, chain_dose.sh, chain_dose2.sh and the
# export lines in logs_round1/pro*_*.log), so that every result of from_claude/31 can be computed on the new models too.
set -euo pipefail
: "${MODEL:?}" "${STAGE:?}"
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=${CODE_PATH:-$ROOT/code_newmodels_ff_20261112}
FF=${FF_ROOT:-$ROOT/results_newmodels_ff_20261112}
SRC=$FF/source/$MODEL
MAIN=$FF/e44/$MODEL
LIKE=$ROOT/results_qa_das_v1u/llama3.1-8b/full
MODELS=$ROOT/models
PROMPTS="x x_E1 x_F conf caut conf_E1 conf_E2 caut_E1 caut_E2 conf_M caut_M"
E_EXTRA_A="x.rand.m1 x_F.w.p1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E_PROA="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 conf.wq.p1 conf_E1.wq.m1 conf.iso.p1 caut.wq.p1 caut_E1.wq.m1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E_PROA_BASES="x x_F conf conf_E1 caut caut_E1"
C_PROA="x.w.p1 x.w.m1 x.md.p1 x.md.m1 x.rand.p1 x.rand.m1 conf.wq.p1 caut.wq.m1 conf_E1.wq.p1 caut_E1.wq.m1 conf.iso.p1"
C_PROA_BASES="x conf caut conf_E1 caut_E1"
E_DOSE="x.w.p0.25 x.w.p0.5 x.w.p0.75 x.w.p1.5 x.w.p2 conf.w.p0.5 conf.w.p2 x.rand.p0.5 x.rand.m0.5 x_F.w.p0.5 x.w.m0.5 x.md.p0.5 x.rand.p0.75 x.rand.m0.75 x_F.w.p0.75 x.w.m0.75 conf.rand.p2 conf.rand.m2 conf.md.p2 conf_E1.w.m1 conf_E1.w.m2"
E_PROB_BASES="x x_F conf conf_E1"
C_DOSE="x.w.p0.25 x.w.p0.5 x.w.p2 x.w.m0.25 x.w.m0.5 x.w.m2"
C_PROB_BASES="x"
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$MODELS
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES="${GPU:-}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
cfg() { "${DX[@]}" python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))" "$CODE/configs/models/$MODEL.json" "$1" "$2"; }
PYM=$(cfg python python3)                 # model steps
NL=$(cfg num_layers 0)
pym() { "${DX[@]}" "$PYM" "$@"; }
pyj() { "${DX[@]}" python3 "$@"; }        # judging steps: the old models' runtime
need_gpu() { [ -n "${GPU:-}" ] || { echo "STAGE=$STAGE needs GPU" >&2; exit 2; }; }
step() { echo "=== $(date '+%F %T') [$MODEL ${BRANCH:-} ${STAGE} gpu${GPU:-none}] $* ==="; }
host() { case "$1" in "$ROOT"/*) echo "$HOST/${1#$ROOT/}" ;; *) echo "path $1 is outside $ROOT" >&2; exit 2 ;; esac; }

if [ ! -f "$(host "$SRC")/bank.json" ]; then
  step "source directory"
  pyj scripts/review_checks/make_e44_source.py --model "$MODEL" --like "$LIKE" --out "$SRC"
fi
COMMON=(--source-run "$SRC" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
arms_of() {
  if [ "$1" = E ]; then
    GEN_ARMS="conf.wq.p1 conf_E1.wq.m1 caut.wq.p1 caut_E1.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.rand.p1"
    MC_ARMS="x.w.p1 x.w.m1 x.md.p1 x.rand.p1"
  else
    GEN_ARMS="conf.wq.p1 caut.wq.m1 conf_E1.wq.p1 caut_E1.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.md.m1 x.rand.p1 x.rand.m1"
    MC_ARMS="x.w.p1 x.w.m1"
  fi
}
allenai() {  # allenai WORK
  pyj scripts/review_checks/judge_two_axis_run.py cells "$1" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
  "${DX[@]}" python3 scripts/allenai_tqa_judge.py --cells "$1/judge2/cells.jsonl" --out "$1/judge2/allenai.jsonl" \
    --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 4
  pyj scripts/review_checks/judge_two_axis_run.py single "$1" --judge allenai > "$(host "$1")/single_allenai_stdout.md"
}

case "$STAGE" in
pilot)
  need_gpu; : "${BRANCH:?}"; arms_of "$BRANCH"
  RUN=${MAIN}_pilot_$BRANCH; L2="$((NL / 8)) $((NL / 2))"; mkdir -p "$(host "$RUN")"
  step "pilot freeze"; pym scripts/e44_factorial_das.py freeze "${COMMON[@]}" --run-dir "$RUN" --splits train tune --n-check 3
  step "train $BRANCH $L2"; pym scripts/e44_factorial_das.py train "${COMMON[@]}" --run-dir "$RUN" --family "$BRANCH" --layers $L2 --val-n 1 --n-check 3
  step "generate"; pym scripts/e44_factorial_das.py generate "${COMMON[@]}" --run-dir "$RUN" --family "$BRANCH" --layers $L2 --split tune --arms $GEN_ARMS --n-check 3
  step "readout"; pym scripts/e44_factorial_das.py mc "${COMMON[@]}" --run-dir "$RUN" --family "$BRANCH" --layers $L2 --split tune --arms $MC_ARMS --n-check 3
  pym scripts/e44_factorial_das.py summarize --source-run "$SRC" --run-dir "$RUN" --include-pilot | tail -n 70 ;;
step0)
  need_gpu; mkdir -p "$(host "$MAIN")"
  if [ ! -f "$(host "$MAIN")/frozen_tune.json" ]; then step "freeze train + tune (11 prompts)"; pym scripts/e44_factorial_das.py freeze "${COMMON[@]}" --run-dir "$MAIN" --splits train tune; fi
  for SPLIT in tune train; do
    step "reference readout without any write: $SPLIT"
    pym scripts/e44_factorial_das.py mc "${COMMON[@]}" --run-dir "$MAIN" --split "$SPLIT" --bases $PROMPTS
  done
  for SPLIT in tune train; do
    WORK=$MAIN/judge_${SPLIT}_step0
    step "export prompt controls: $SPLIT"
    pyj scripts/e44_export_judge.py --source-run "$SRC" --run-dir "$MAIN" --split "$SPLIT" --gen --name step0 | tail -n 3
    mkdir -p "$(host "$WORK")"
    [ -f "$(host "$WORK")/judge_input.json" ] || cp "$(host "$MAIN")/judge_$SPLIT/step0.json" "$(host "$WORK")/judge_input.json"
    cmp "$(host "$MAIN")/judge_$SPLIT/step0.json" "$(host "$WORK")/judge_input.json"
    step "cells + allenai + single: $SPLIT"; allenai "$WORK"
  done
  pym scripts/e44_factorial_das.py summarize --source-run "$SRC" --run-dir "$MAIN" ;;
step0pro)
  WORK=$MAIN/judge_tune_step0
  [ -f "$(host "$WORK")/judge2/cells.jsonl" ] || { echo "run STAGE=step0 first" >&2; exit 3; }
  step "pro (two questions per unique cell)"; pyj scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
  pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$(host "$WORK")/single_pro_stdout.md"
  cat "$(host "$WORK")/single_pro_stdout.md" ;;
run)
  need_gpu; : "${BRANCH:?}"; arms_of "$BRANCH"; LAYERS=${LAYERS:-$(seq 0 $((NL - 1)) | tr '\n' ' ')}
  [ -f "$(host "$MAIN")/frozen_tune.json" ] && [ -f "$(host "$MAIN")/mc/base_x_tune.json" ] \
    || { echo "run STAGE=step0 to the end first (frozen answers and the baseline readout; parallel runs would race on them)" >&2; exit 3; }
  step "train $BRANCH layers: $LAYERS"; pym scripts/e44_factorial_das.py train "${COMMON[@]}" --run-dir "$MAIN" --family "$BRANCH" --layers $LAYERS --val-n 16
  step "generate tune arms: $GEN_ARMS"; pym scripts/e44_factorial_das.py generate "${COMMON[@]}" --run-dir "$MAIN" --family "$BRANCH" --layers $LAYERS --split tune --arms $GEN_ARMS
  step "reference readout: $MC_ARMS"; pym scripts/e44_factorial_das.py mc "${COMMON[@]}" --run-dir "$MAIN" --family "$BRANCH" --layers $LAYERS --split tune --arms $MC_ARMS ;;
judge)
  need_gpu; : "${BRANCH:?}"; arms_of "$BRANCH"
  NAME=round1_$BRANCH; WORK=$MAIN/judge_tune_$NAME
  GEN_FILES=$(cd "$(host "$MAIN")/gen" && ls ${BRANCH}_L*_tune.json)
  want=$(( NL * $(echo $GEN_ARMS | wc -w) )); have=$(echo $GEN_FILES | wc -w)
  [ "$have" = "$want" ] || { echo "generation files: have $have, want $want" >&2; exit 3; }
  step "export $NAME ($have generation files)"
  pyj scripts/e44_export_judge.py --source-run "$SRC" --run-dir "$MAIN" --split tune --gen $GEN_FILES --name "$NAME" | tail -n 3
  mkdir -p "$(host "$WORK")"
  [ -f "$(host "$WORK")/judge_input.json" ] || cp "$(host "$MAIN")/judge_tune/$NAME.json" "$(host "$WORK")/judge_input.json"
  cmp "$(host "$MAIN")/judge_tune/$NAME.json" "$(host "$WORK")/judge_input.json"
  step "cells + allenai + single"; allenai "$WORK" ;;
screen)
  : "${BRANCH:?}"
  pyj scripts/review_checks/e44_round1_tables.py screen "$MAIN" "$BRANCH" "$MAIN/judge_tune_round1_$BRANCH" ;;
extra)
  need_gpu; : "${BRANCH:?}" "${LAYERS:?}"
  if [ "${PURPOSE:-proA}" = dose ]; then
    if [ "$BRANCH" = E ]; then ARMS=${ARMS:-$E_DOSE}; else ARMS=${ARMS:-$C_DOSE}; fi
  elif [ "$BRANCH" = E ]; then ARMS=${ARMS:-$E_EXTRA_A}; fi
  : "${ARMS:?}"
  step "extra generation: layers $LAYERS arms $ARMS"
  pym scripts/e44_factorial_das.py generate "${COMMON[@]}" --run-dir "$MAIN" --family "$BRANCH" --layers $LAYERS --split tune --arms $ARMS ;;
pro)
  need_gpu; : "${BRANCH:?}" "${LAYER:?}"       # GPU: AllenAI on the same cells, which the tables need (red team 4)
  TAG=${TAG:-A}
  case "$BRANCH$TAG" in
    EA) ARMS=${ARMS:-$E_PROA}; BASES=${BASES:-$E_PROA_BASES} ;;
    CA) ARMS=${ARMS:-$C_PROA}; BASES=${BASES:-$C_PROA_BASES} ;;
    EB) ARMS=${ARMS:-$E_DOSE}; BASES=${BASES:-$E_PROB_BASES} ;;
    CB) ARMS=${ARMS:-$C_DOSE}; BASES=${BASES:-$C_PROB_BASES} ;;
    *) echo "TAG must be A or B" >&2; exit 2 ;;
  esac
  NAME=pro${TAG}_${BRANCH}_L${LAYER}; WORK=$MAIN/judge_tune_$NAME
  GEN=""; for a in $ARMS; do GEN="$GEN ${BRANCH}_L${LAYER}_${a//./-}_tune.json"; done
  step "export: prompts $BASES; arms $ARMS"
  pyj scripts/e44_export_judge.py --source-run "$SRC" --run-dir "$MAIN" --split tune --gen $GEN --prompts $BASES --name "$NAME" | tail -n 3
  mkdir -p "$(host "$WORK")/judge2/cache"
  [ -f "$(host "$WORK")/judge_input.json" ] || cp "$(host "$MAIN")/judge_tune/$NAME.json" "$(host "$WORK")/judge_input.json"
  cmp "$(host "$MAIN")/judge_tune/$NAME.json" "$(host "$WORK")/judge_input.json"
  for f in "$(host "$MAIN")/judge_tune_step0/judge2/cache"/*; do    # the step-0 Pro cache: identical base answers are not paid twice
    [ -f "$f" ] || continue
    [ -e "$(host "$WORK")/judge2/cache/$(basename "$f")" ] || cp "$f" "$(host "$WORK")/judge2/cache/"
  done
  step "cells + allenai on the same cells"; allenai "$WORK"
  step "pro"; pyj scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
  pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$(host "$WORK")/single_pro_stdout.md" ;;
tables)
  : "${BRANCH:?}" "${LAYERS:?}"
  T=scripts/review_checks/e44_round1_tables.py
  for L in $LAYERS; do for J in allenai pro; do        # frozen口径: x base, strength 1, proA work dir of the layer
    pyj $T cells "$MAIN" "$BRANCH" "$MAIN/judge_tune_proA_${BRANCH}_L$L" --judge "$J" --layers "$L"
    pyj $T cross "$MAIN" "$BRANCH" "$MAIN/judge_tune_proA_${BRANCH}_L$L" --judge "$J" --layers "$L"
    if [ "$BRANCH" = E ]; then                         # exploratory, as regen_conf_tables.sh: must-answer base, strength 1
      pyj $T cells "$MAIN" E "$MAIN/judge_tune_proA_E_L$L" --judge "$J" --layers "$L" --base conf
    fi
  done; done
  if [ "$BRANCH" = E ]; then
    for L in ${DOSE_LAYERS:-}; do for J in allenai pro; do   # exploratory: must-answer base, strength 2 (proB work dir)
      pyj $T cells "$MAIN" E "$MAIN/judge_tune_proB_E_L$L" --judge "$J" --layers "$L" --base conf --main conf.w.p2 --strength 2
    done; done
  fi ;;
*) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
step "$STAGE done"
