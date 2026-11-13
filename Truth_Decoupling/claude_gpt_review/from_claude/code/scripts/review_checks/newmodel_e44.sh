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
#   judge     export every generation of BRANCH (all layers must be done; or only LAYERS) + AllenAI (GPU)
#   screen    the frozen shortlist rule on BRANCH's AllenAI table (CPU; e44_round1_tables.py screen, unchanged)
#   extra     extra arms for LAYERS of BRANCH (GPU). ARMS default: E -> Llama's six pre-Pro arms (E_EXTRA_A);
#             PURPOSE=dose -> Llama's strength-sweep arms (E_DOSE / C_DOSE)
#   pro       DeepSeek Pro + AllenAI on the same cells, ONE LAYER of BRANCH (GPU). TAG=A (default): Llama's proA arms and
#             prompts; TAG=B: Llama's proB (strength sweep) arms and prompts. ARMS / BASES override.
#   tables    frozen x-base cells and cross tables for LAYERS (proA work dirs), both judges; with BRANCH=E also the
#             exploratory must-answer tables (conf base, strength 1 on proA; strength 2 on proB for DOSE_LAYERS)
# Arm lists are copied from what Llama actually ran (review_e44_claude/chain_pro_E.sh, chain_dose.sh, chain_dose2.sh and the
# export lines in logs_round1/pro*_*.log), so that every result of from_claude/31 can be computed on the new models too.
#
# Periphery parameters added 2026-11-13 (exploration after results/12, 赵磊 11-13 08:3x: the program is a core plus a
# periphery that passes each model's own settings in as parameters). The core (core/*.py, scripts/e44_factorial_das.py and
# the scripts its identity covers) is unchanged; with none of these set, every command is the one of 11-12.
#   E_ON=E1|E2      truth sentence of the upper E level, passed to the core as --e-on; the prompt names in the arm lists
#                   follow it (conf_<E_ON>, caut_<E_ON>). Default E1 = Llama's round 1.
#   VARIANT=<name>  run directory e44/<MODEL>_<name> instead of e44/<MODEL>. It reuses the model's step 0: the frozen answers
#                   and the no-write readouts of e44/<MODEL> are copied once, byte for byte (checked on every call), and the
#                   core accepts them only under its own identity check (code, prompts, model, caps). The Pro cache of
#                   step 0 is taken from e44/<MODEL> as well. step0 / step0pro / pilot / diag refuse a VARIANT.
#                   STAGE=seed only does that copy (run it once before parallel jobs on one VARIANT). STAGE=source only
#                   makes a new model's source directory (run it once before parallel jobs of that model).
#   TAG=<other>     pro stage: a tag other than A / B names a judged set of your own; ARMS and BASES are then required.
#                   JUDGES=allenai: export + AllenAI only (GPU). JUDGES=pro: DeepSeek Pro only, on a work dir that already
#                   has its AllenAI judgement (no GPU; the export is redone and must be identical). Default: both.
#                   CACHE_FROM: the Pro reply cache to seed from (default the step-0 one), copied once just before Pro.
#                   A judge that leaves a cell without a verdict now stops the job (plan 43 section 5) instead of passing.
#                   tables stage: TAG picks the work dirs (default A) and STRENGTH the strength suffix (default 1).
#   LAYERS          judge stage: export only these layers' run arms (default: all layers, the 11-12 behaviour).
#   STAGE=diag      prompt-only diagnostics in their own run dir promptdiag/<MODEL>: freeze DIAG_PROMPTS (a prompts file
#                   under configs/prompts) on the tune questions, export, AllenAI (GPU). STAGE=diagpro: DeepSeek Pro on
#                   them (no GPU), cache seeded from CACHE_FROM (default: the model's step-0 judge cache).
# Added 2026-11-13 11:35 (from_claude/44 section 5.5: Llama's E2 directions of 11-11 are reused, not retrained):
#   RUN_DIR=<dir>   an EXISTING run directory (absolute, under /workspace/Truth_Decoupling, with frozen answers) instead of
#                   e44/<MODEL>[_<VARIANT>]; only for extra / pro / tables (judge and screen name round 1's own work dir),
#                   never with VARIANT, always with SOURCE_DIR. The core checks its own identity against that run's frozen
#                   answers and directions, so CODE_PATH must name the code copy the run was made with (e.g. Llama's 11-11
#                   run: code_e45_20261112). A Pro pass on a RUN_DIR needs CACHE_FROM (no e44/<MODEL> step 0 to default to).
#   SOURCE_DIR=<dir> an EXISTING source directory (with bank.json; never created here) instead of source/<MODEL>, e.g.
#                   the one a RUN_DIR was made from (the export checks its bank / questions hashes against the run).
# A reply-cache directory named by CACHE_FROM (or the default) that does not exist now stops the job instead of seeding
# nothing. With none of RUN_DIR / SOURCE_DIR set, every command is the one of the 09:38 version (06ba4620...).
set -euo pipefail
: "${MODEL:?}" "${STAGE:?}"
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=${CODE_PATH:-$ROOT/code_newmodels_ff_20261112}
FF=${FF_ROOT:-$ROOT/results_newmodels_ff_20261112}
existing() {  # existing NAME DIR FILE: DIR is under ROOT and already holds FILE (RUN_DIR / SOURCE_DIR are never created)
  case "$2" in "$ROOT"/*) ;; *) echo "$1 must be an absolute path under $ROOT" >&2; exit 2 ;; esac
  [ -f "$HOST/${2#"$ROOT"/}/$3" ] || { echo "$1=$2 has no $3 (an existing directory only)" >&2; exit 3; }
}
SOURCE_DIR=${SOURCE_DIR:-}; RUN_DIR=${RUN_DIR:-}
[ -z "$SOURCE_DIR" ] || existing SOURCE_DIR "$SOURCE_DIR" bank.json
SRC=${SOURCE_DIR:-$FF/source/$MODEL}
BASE_MAIN=$FF/e44/$MODEL
VARIANT=${VARIANT:-}
case "$VARIANT" in ""|*[!A-Za-z0-9]*) [ -z "$VARIANT" ] || { echo "VARIANT must be alphanumeric" >&2; exit 2; } ;; esac
if [ -n "$RUN_DIR" ]; then
  [ -z "$VARIANT" ] || { echo "RUN_DIR and VARIANT exclude each other" >&2; exit 2; }
  [ -n "$SOURCE_DIR" ] || { echo "RUN_DIR needs SOURCE_DIR (the source directory that run was made from)" >&2; exit 2; }
  case "$STAGE" in extra|pro|tables) ;; *) echo "STAGE=$STAGE does not take a RUN_DIR" >&2; exit 2 ;; esac
  existing RUN_DIR "$RUN_DIR" frozen_tune.json
fi
MAIN=${RUN_DIR:-$BASE_MAIN${VARIANT:+_$VARIANT}}
E_ON=${E_ON:-E1}
case "$E_ON" in E1|E2) ;; *) echo "E_ON must be E1 or E2" >&2; exit 2 ;; esac
CE=conf_$E_ON; KE=caut_$E_ON
LIKE=$ROOT/results_qa_das_v1u/llama3.1-8b/full
MODELS=$ROOT/models
PROMPTS="x x_E1 x_F conf caut conf_E1 conf_E2 caut_E1 caut_E2 conf_M caut_M"
E_EXTRA_A="x.rand.m1 x_F.w.p1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E_PROA="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 conf.wq.p1 $CE.wq.m1 conf.iso.p1 caut.wq.p1 $KE.wq.m1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E_PROA_BASES="x x_F conf $CE caut $KE"
C_PROA="x.w.p1 x.w.m1 x.md.p1 x.md.m1 x.rand.p1 x.rand.m1 conf.wq.p1 caut.wq.m1 $CE.wq.p1 $KE.wq.m1 conf.iso.p1"
C_PROA_BASES="x conf caut $CE $KE"
E_DOSE="x.w.p0.25 x.w.p0.5 x.w.p0.75 x.w.p1.5 x.w.p2 conf.w.p0.5 conf.w.p2 x.rand.p0.5 x.rand.m0.5 x_F.w.p0.5 x.w.m0.5 x.md.p0.5 x.rand.p0.75 x.rand.m0.75 x_F.w.p0.75 x.w.m0.75 conf.rand.p2 conf.rand.m2 conf.md.p2 $CE.w.m1 $CE.w.m2"
E_PROB_BASES="x x_F conf $CE"
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
step() { echo "=== $(date '+%F %T') [$MODEL${VARIANT:+_$VARIANT}${RUN_DIR:+ run=${RUN_DIR##*/}} ${BRANCH:-} ${STAGE} gpu${GPU:-none} e_on=$E_ON] $* ==="; }
host() { case "$1" in "$ROOT"/*) echo "$HOST/${1#$ROOT/}" ;; *) echo "path $1 is outside $ROOT" >&2; exit 2 ;; esac; }
seed_cache() {  # seed_cache FROM_DIR WORK: copy reply-cache files (identical requests are not paid twice)
  [ -d "$(host "$1")" ] || { echo "reply cache $1 does not exist (CACHE_FROM?)" >&2; exit 3; }
  mkdir -p "$(host "$2")/judge2/cache"
  for f in "$(host "$1")"/*; do
    [ -f "$f" ] || continue
    [ -e "$(host "$2")/judge2/cache/$(basename "$f")" ] || cp "$f" "$(host "$2")/judge2/cache/"
  done
}
check_judges() {  # JUDGES may only name allenai and pro (a typo must not skip both and still succeed)
  local j; [ -n "$1" ] || { echo "JUDGES is empty" >&2; exit 2; }
  for j in $1; do case "$j" in allenai|pro) ;; *) echo "JUDGES: unknown judge '$j' (allenai / pro)" >&2; exit 2 ;; esac; done
}
verdicts_complete() {  # verdicts_complete WORK JUDGE: a cell without a verdict stops the job (plan 43 section 5: never imputed)
  local n
  n=$(pyj -c "import json,sys; print(json.load(open(sys.argv[1])).get('n_no_verdict', 'missing'))" "$1/judge2/single_$2_summary.json")
  [ "$n" = 0 ] || { echo "$2 left $n cells without a verdict in $1: re-judge them before any table" >&2; exit 3; }
}
seed_variant() {  # a VARIANT run reuses the model's step 0, byte for byte
  [ -n "$VARIANT" ] || return 0
  local b m f; b=$(host "$BASE_MAIN"); m=$(host "$MAIN")
  [ -f "$b/frozen_train.json" ] && [ -f "$b/frozen_tune.json" ] && [ -f "$b/mc/base_x_tune.json" ] \
    || { echo "base run $BASE_MAIN has no complete step 0" >&2; exit 3; }
  mkdir -p "$m/mc"
  for f in frozen_train.json frozen_tune.json $(cd "$b" && ls mc/ | grep -E '^base_.*\.json$' | sed 's|^|mc/|'); do
    [ -e "$m/$f" ] || cp -p "$b/$f" "$m/$f"
    cmp -s "$b/$f" "$m/$f" || { echo "variant copy $m/$f differs from the base run" >&2; exit 3; }
  done
}

if [ ! -f "$(host "$SRC")/bank.json" ]; then
  step "source directory"
  pyj scripts/review_checks/make_e44_source.py --model "$MODEL" --like "$LIKE" --out "$SRC"
fi
COMMON=(--source-run "$SRC" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
EON=(--e-on "$E_ON")
arms_of() {
  if [ "$1" = E ]; then
    GEN_ARMS="conf.wq.p1 $CE.wq.m1 caut.wq.p1 $KE.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.rand.p1"
    MC_ARMS="x.w.p1 x.w.m1 x.md.p1 x.rand.p1"
  else
    GEN_ARMS="conf.wq.p1 caut.wq.m1 $CE.wq.p1 $KE.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.md.m1 x.rand.p1 x.rand.m1"
    MC_ARMS="x.w.p1 x.w.m1"
  fi
}
allenai() {  # allenai WORK
  pyj scripts/review_checks/judge_two_axis_run.py cells "$1" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
  "${DX[@]}" python3 scripts/allenai_tqa_judge.py --cells "$1/judge2/cells.jsonl" --out "$1/judge2/allenai.jsonl" \
    --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 4
  pyj scripts/review_checks/judge_two_axis_run.py single "$1" --judge allenai > "$(host "$1")/single_allenai_stdout.md"
  verdicts_complete "$1" allenai
}

case "$STAGE" in
  step0|step0pro|pilot|diag|diagpro) [ -z "$VARIANT" ] || { echo "STAGE=$STAGE belongs to the model's base run (unset VARIANT)" >&2; exit 2; } ;;
  *) seed_variant ;;
esac

case "$STAGE" in
pilot)
  need_gpu; : "${BRANCH:?}"; arms_of "$BRANCH"
  RUN=${MAIN}_pilot_$BRANCH; L2="$((NL / 8)) $((NL / 2))"; mkdir -p "$(host "$RUN")"
  step "pilot freeze"; pym scripts/e44_factorial_das.py freeze "${COMMON[@]}" --run-dir "$RUN" --splits train tune --n-check 3
  step "train $BRANCH $L2"; pym scripts/e44_factorial_das.py train "${COMMON[@]}" "${EON[@]}" --run-dir "$RUN" --family "$BRANCH" --layers $L2 --val-n 1 --n-check 3
  step "generate"; pym scripts/e44_factorial_das.py generate "${COMMON[@]}" "${EON[@]}" --run-dir "$RUN" --family "$BRANCH" --layers $L2 --split tune --arms $GEN_ARMS --n-check 3
  step "readout"; pym scripts/e44_factorial_das.py mc "${COMMON[@]}" "${EON[@]}" --run-dir "$RUN" --family "$BRANCH" --layers $L2 --split tune --arms $MC_ARMS --n-check 3
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
  cat "$(host "$WORK")/single_pro_stdout.md"; verdicts_complete "$WORK" pro ;;
run)
  need_gpu; : "${BRANCH:?}"; arms_of "$BRANCH"; LAYERS=${LAYERS:-$(seq 0 $((NL - 1)) | tr '\n' ' ')}
  [ -f "$(host "$MAIN")/frozen_tune.json" ] && [ -f "$(host "$MAIN")/mc/base_x_tune.json" ] \
    || { echo "run STAGE=step0 to the end first (frozen answers and the baseline readout; parallel runs would race on them)" >&2; exit 3; }
  step "train $BRANCH layers: $LAYERS"; pym scripts/e44_factorial_das.py train "${COMMON[@]}" "${EON[@]}" --run-dir "$MAIN" --family "$BRANCH" --layers $LAYERS --val-n 16
  step "generate tune arms: $GEN_ARMS"; pym scripts/e44_factorial_das.py generate "${COMMON[@]}" "${EON[@]}" --run-dir "$MAIN" --family "$BRANCH" --layers $LAYERS --split tune --arms $GEN_ARMS
  step "reference readout: $MC_ARMS"; pym scripts/e44_factorial_das.py mc "${COMMON[@]}" "${EON[@]}" --run-dir "$MAIN" --family "$BRANCH" --layers $LAYERS --split tune --arms $MC_ARMS ;;
judge)
  need_gpu; : "${BRANCH:?}"; arms_of "$BRANCH"
  NAME=round1_$BRANCH; WORK=$MAIN/judge_tune_$NAME
  if [ -z "${LAYERS:-}" ]; then                       # 11-12 behaviour: every layer, every file of the branch
    GEN_FILES=$(cd "$(host "$MAIN")/gen" && ls ${BRANCH}_L*_tune.json)
    want=$(( NL * $(echo $GEN_ARMS | wc -w) ))
  else                                                # only these layers' run arms
    GEN_FILES=""; for L in $LAYERS; do for a in $GEN_ARMS; do GEN_FILES="$GEN_FILES ${BRANCH}_L${L}_${a//./-}_tune.json"; done; done
    for f in $GEN_FILES; do [ -f "$(host "$MAIN")/gen/$f" ] || { echo "missing generation file $f" >&2; exit 3; }; done
    want=$(( $(echo $LAYERS | wc -w) * $(echo $GEN_ARMS | wc -w) ))
  fi
  have=$(echo $GEN_FILES | wc -w)
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
  pym scripts/e44_factorial_das.py generate "${COMMON[@]}" "${EON[@]}" --run-dir "$MAIN" --family "$BRANCH" --layers $LAYERS --split tune --arms $ARMS ;;
seed)
  [ -n "$VARIANT" ] || { echo "STAGE=seed copies a base run's step 0 into a VARIANT run dir: set VARIANT" >&2; exit 2; } ;;
source)
  ;;   # only the source directory (made above when missing): run once before parallel jobs of a new model
pro)
  : "${BRANCH:?}" "${LAYER:?}"
  JUDGES=${JUDGES:-allenai pro}; check_judges "$JUDGES"
  case " $JUDGES " in *" allenai "*) need_gpu ;; esac   # GPU: AllenAI on the same cells, which the tables need (red team 4)
  TAG=${TAG:-A}
  case "$TAG" in ""|*[!A-Za-z0-9]*) echo "TAG must be alphanumeric" >&2; exit 2 ;; esac
  case "$BRANCH$TAG" in
    EA) ARMS=${ARMS:-$E_PROA}; BASES=${BASES:-$E_PROA_BASES} ;;
    CA) ARMS=${ARMS:-$C_PROA}; BASES=${BASES:-$C_PROA_BASES} ;;
    EB) ARMS=${ARMS:-$E_DOSE}; BASES=${BASES:-$E_PROB_BASES} ;;
    CB) ARMS=${ARMS:-$C_DOSE}; BASES=${BASES:-$C_PROB_BASES} ;;
    *) : "${ARMS:?TAG=$TAG needs ARMS}" "${BASES:?TAG=$TAG needs BASES}" ;;
  esac
  NAME=pro${TAG}_${BRANCH}_L${LAYER}; WORK=$MAIN/judge_tune_$NAME
  GEN=""; for a in $ARMS; do GEN="$GEN ${BRANCH}_L${LAYER}_${a//./-}_tune.json"; done
  step "export: prompts $BASES; arms $ARMS"
  pyj scripts/e44_export_judge.py --source-run "$SRC" --run-dir "$MAIN" --split tune --gen $GEN --prompts $BASES --name "$NAME" | tail -n 3
  mkdir -p "$(host "$WORK")/judge2/cache"
  [ -f "$(host "$WORK")/judge_input.json" ] || cp "$(host "$MAIN")/judge_tune/$NAME.json" "$(host "$WORK")/judge_input.json"
  cmp "$(host "$MAIN")/judge_tune/$NAME.json" "$(host "$WORK")/judge_input.json"
  case " $JUDGES " in
    *" allenai "*) step "cells + allenai on the same cells"; allenai "$WORK" ;;
    *) [ -f "$(host "$WORK")/judge2/single_allenai.jsonl" ] \
         || { echo "no AllenAI judgement in $WORK yet: run this layer with JUDGES=allenai first" >&2; exit 3; }
       verdicts_complete "$WORK" allenai ;;
  esac
  case " $JUDGES " in
    *" pro "*)
      # reply cache: the step-0 one by default (identical base answers are not paid twice); CACHE_FROM names a superset
      # (e.g. an earlier work dir of the same layer that judged some of the same answers). Seeded only before Pro, once.
      [ -z "$RUN_DIR" ] || [ -n "${CACHE_FROM:-}" ] || { echo "a Pro pass on RUN_DIR needs CACHE_FROM (the step-0 reply cache of that run)" >&2; exit 2; }
      seed_cache "${CACHE_FROM:-$BASE_MAIN/judge_tune_step0/judge2/cache}" "$WORK"
      step "pro"; pyj scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
      pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$(host "$WORK")/single_pro_stdout.md"
      verdicts_complete "$WORK" pro ;;
    *) step "pro skipped (JUDGES=$JUDGES)" ;;
  esac ;;
tables)
  : "${BRANCH:?}" "${LAYERS:?}"
  T=scripts/review_checks/e44_round1_tables.py
  TT=${TAG:-A}; S=${STRENGTH:-1}; check_judges "${JUDGES:-allenai pro}"
  echo "TABLES run=$MAIN e_on=$E_ON work=judge_tune_pro${TT}_${BRANCH}_L<layer> strength=$S judges=${JUDGES:-allenai pro}"
  for L in $LAYERS; do for J in ${JUDGES:-allenai pro}; do   # frozen口径: x base, strength 1, proA work dir of the layer
    pyj $T cells "$MAIN" "$BRANCH" "$MAIN/judge_tune_pro${TT}_${BRANCH}_L$L" --judge "$J" --layers "$L" --strength "$S"
    pyj $T cross "$MAIN" "$BRANCH" "$MAIN/judge_tune_pro${TT}_${BRANCH}_L$L" --judge "$J" --layers "$L"
    if [ "$BRANCH" = E ] && [ "$S" = 1 ]; then         # exploratory, as regen_conf_tables.sh: must-answer base, strength 1
      pyj $T cells "$MAIN" E "$MAIN/judge_tune_pro${TT}_E_L$L" --judge "$J" --layers "$L" --base conf
    fi
  done; done
  if [ "$BRANCH" = E ]; then
    for L in ${DOSE_LAYERS:-}; do for J in ${JUDGES:-allenai pro}; do   # exploratory: must-answer base, strength 2 (proB work dir)
      pyj $T cells "$MAIN" E "$MAIN/judge_tune_proB_E_L$L" --judge "$J" --layers "$L" --base conf --main conf.w.p2 --strength 2
    done; done
  fi ;;
diag)
  need_gpu; : "${DIAG_PROMPTS:?name a prompts file under configs/prompts}"
  RUN=$FF/promptdiag/$MODEL; WORK=$RUN/judge_tune_diag; mkdir -p "$(host "$RUN")"
  step "freeze tune: $DIAG_PROMPTS"
  pym scripts/e44_factorial_das.py freeze "${COMMON[@]}" --prompts-e44 "$CODE/configs/prompts/$DIAG_PROMPTS" --run-dir "$RUN" --splits tune
  step "export prompt controls"
  pyj scripts/e44_export_judge.py --source-run "$SRC" --run-dir "$RUN" --split tune --gen --name diag | tail -n 3
  mkdir -p "$(host "$WORK")"
  [ -f "$(host "$WORK")/judge_input.json" ] || cp "$(host "$RUN")/judge_tune/diag.json" "$(host "$WORK")/judge_input.json"
  cmp "$(host "$RUN")/judge_tune/diag.json" "$(host "$WORK")/judge_input.json"
  step "cells + allenai + single"; allenai "$WORK" ;;
diagpro)
  RUN=$FF/promptdiag/$MODEL; WORK=$RUN/judge_tune_diag
  [ -f "$(host "$WORK")/judge2/cells.jsonl" ] || { echo "run STAGE=diag first" >&2; exit 3; }
  seed_cache "${CACHE_FROM:-$BASE_MAIN/judge_tune_step0/judge2/cache}" "$WORK"
  step "pro"; pyj scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
  pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$(host "$WORK")/single_pro_stdout.md"
  cat "$(host "$WORK")/single_pro_stdout.md"; verdicts_complete "$WORK" pro ;;
*) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
step "$STAGE done"
