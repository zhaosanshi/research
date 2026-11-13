#!/usr/bin/env bash
# E+C on the plain prompt x for a new model (Claude, 2026-11-13; from_claude/44 section 5.8). The Llama procedure of
# from_claude/36 section 3 (results/09_E44_两因子交换训练_Llama/scripts/e44_extra_EC.sh) with nothing model-specific in the
# script: the model, its runs and the layers / strengths enter as parameters (赵磊 11-13: core plus periphery).
# The E direction comes from E_RUN (e.g. the 9B's E2 variant run), the C direction from C_RUN (the model's round-1 run):
# e44_eval_extra.py --c-run-dir, which checks that both runs share one identity and one set of frozen training answers.
# Selfcheck, generation and export run e44_eval_extra.py unchanged (4438a282, the version GPT reviewed for Llama) with the
# model's own interpreter (the model config's "python"), so that every phase of that script runs in the runtime the run was
# made with; the judging steps run with the container's python3, as in newmodel_e44.sh.
#   MODEL=qwen3.5-9b CODE_PATH=<code copy> E_RUN=<dir> C_RUN=<dir> E_ON=E2 C_E_ON=E1 EL=15 CL=16 ES="1 2" KS="0.25 0.5 1 2" \
#     TAG=<name> STAGE=<stage> [GPU=k] [CACHE_FROM="<dir> [<dir> ...]"] ff_explore_ec.sh
# stages:
#   selfcheck  real-model acceptance of the two-write engine for (E<EL>, C<CL>) on base x, whole tuning split (GPU)
#   gen1       E fixed at each strength of ES, with C<CL> toward confident (sign m) at each strength of KS; the single writes
#              (E at each ES, C at each KS) and the single-E random controls (both signs, each ES); then the read-only check
#              ff_explore_ec_check.py: reproduction against the drivers' own files, rule abstentions, and the choice of k
#              per E strength written before any judging (eval_extra/ec_choice_<TAG>.json) (GPU)
#   gen2       for each E strength and its chosen k only: E -> random (both signs) + C, E + C -> random (both signs), single
#              C random (both signs) (GPU; needs the choice file, checked against this run, these layers and ES)
#   judge      one export of exactly the gen1 + gen2 files (named, never "every file") + baseline x; AllenAI (GPU); the
#              DeepSeek Pro reply caches named in CACHE_FROM merged into the work dir's cache (judge2_cache_merge.py; a
#              conflicting key stops the job); DeepSeek Pro (identical requests are not paid twice)
#   describe   read-only description tables, both judges (ff_explore_ec_describe.py)
set -euo pipefail
: "${MODEL:?}" "${STAGE:?}" "${CODE_PATH:?}" "${E_RUN:?}" "${C_RUN:?}" "${EL:?}" "${CL:?}" "${TAG:?}"
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$CODE_PATH
FF=${FF_ROOT:-$ROOT/results_newmodels_ff_20261112}
SRC=${SOURCE_DIR:-$FF/source/$MODEL}
E_ON=${E_ON:-E1}; C_E_ON=${C_E_ON:-E1}
ES=${ES:-2}; KS=${KS:-0.25 0.5 1}
MODELS=$ROOT/models
case "$TAG" in ""|*[!A-Za-z0-9]*) echo "TAG must be alphanumeric" >&2; exit 2 ;; esac
case "$EL$CL" in *[!0-9]*) echo "EL and CL must be layer numbers" >&2; exit 2 ;; esac
[ "$EL" != "$CL" ] || { echo "the E and C writes must act at two different layers" >&2; exit 2; }
for s in $ES $KS; do case "$s" in ""|*[!0-9.]*|.*|*.|*.*.*) echo "strength '$s' is not a number like 1 or 0.25" >&2; exit 2 ;; esac; done
host() { case "$1" in "$ROOT"/*) echo "$HOST/${1#$ROOT/}" ;; *) echo "path $1 is outside $ROOT" >&2; exit 2 ;; esac; }
for d in "$E_RUN" "$C_RUN"; do [ -f "$(host "$d")/frozen_tune.json" ] || { echo "$d is not a finished run (no frozen_tune.json)" >&2; exit 3; }; done
[ -f "$(host "$SRC")/bank.json" ] || { echo "no source directory $SRC" >&2; exit 3; }
for f in scripts/e44_eval_extra.py scripts/review_checks/ff_explore_ec_check.py scripts/review_checks/ff_explore_ec_describe.py \
         scripts/review_checks/e44_round1_tables.py scripts/review_checks/judge2_cache_merge.py; do
  [ -f "$(host "$CODE")/$f" ] || { echo "code copy $CODE has no $f (copy it there before running)" >&2; exit 3; }
done
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$MODELS
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES="${GPU:-}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
cfg() { "${DX[@]}" python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))" "$CODE/configs/models/$MODEL.json" "$1" "$2"; }
PYM=$(cfg python python3)                 # every phase of e44_eval_extra.py: the runtime the run was made with
pym() { "${DX[@]}" "$PYM" "$@"; }
pyj() { "${DX[@]}" python3 "$@"; }        # judging steps: the old models' runtime
need_gpu() { [ -n "${GPU:-}" ] || { echo "STAGE=$STAGE needs GPU" >&2; exit 2; }; }
step() { echo "=== $(date '+%F %T') [$MODEL E+C $TAG $STAGE gpu${GPU:-none} E$EL($E_ON) C$CL($C_E_ON)] $* ==="; }
two() {  # two E_PART C_PART: the two-write arm on x, lower layer first (the grammar of e44_eval_extra.py)
  if [ "$EL" -lt "$CL" ]; then echo "x:$1+$2"; else echo "x:$2+$1"; fi
}
verdicts_complete() {  # verdicts_complete WORK JUDGE: a cell without a verdict stops the job (never imputed)
  local n
  n=$(pyj -c "import json,sys; print(json.load(open(sys.argv[1])).get('n_no_verdict', 'missing'))" "$1/judge2/single_$2_summary.json")
  [ "$n" = 0 ] || { echo "$2 left $n cells without a verdict in $1: re-judge them before any table" >&2; exit 3; }
}
COMMON=(--source-run "$SRC" --run-dir "$E_RUN" --e-on "$E_ON" --c-run-dir "$C_RUN" --c-e-on "$C_E_ON" --split tune
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
CHOICE=$E_RUN/eval_extra/ec_choice_$TAG.json
NAME=${TAG}_tune
WORK=$E_RUN/eval_extra/judge_tune_$NAME
LAST_ES=${ES##* }

arms1() {  # the first generation stage: E+C grid, single E, single-E random controls, single C
  local s k out=""
  for s in $ES; do for k in $KS; do out="$out $(two "E$EL.w.p$s" "C$CL.w.m$k")"; done; done
  for s in $ES; do out="$out x:E$EL.w.p$s x:E$EL.rand.p$s x:E$EL.rand.m$s"; done
  for k in $KS; do out="$out x:C$CL.w.m$k"; done
  echo $out
}
chosen_k() {  # chosen_k S: the k of E strength S from the choice file, after checking the file belongs to this run
  pyj -c "
import json, sys
c = json.load(open(sys.argv[1]))
want = {'run_dir': sys.argv[2], 'c_run_dir': sys.argv[3], 'e_layer': int(sys.argv[4]), 'c_layer': int(sys.argv[5])}
got = {k: c.get(k) for k in want}
if got != want or sorted(c['choice']) != sorted(sys.argv[6].split()):
    sys.exit('choice file %s is for another run, other layers or other E strengths: %s vs %s' % (sys.argv[1], got, want))
print(c['choice'][sys.argv[7]])" "$CHOICE" "$E_RUN" "$C_RUN" "$EL" "$CL" "$ES" "$1"
}
arms2() {  # the second generation stage, for each E strength and its chosen k only
  local s k out="" seen=" "
  [ -f "$(host "$CHOICE")" ] || { echo "no choice file $CHOICE: run STAGE=gen1 to the end first" >&2; return 3; }
  for s in $ES; do
    k=$(chosen_k "$s") || return 3
    out="$out $(two "E$EL.rand.p$s" "C$CL.w.m$k") $(two "E$EL.rand.m$s" "C$CL.w.m$k")"
    out="$out $(two "E$EL.w.p$s" "C$CL.rand.m$k") $(two "E$EL.w.p$s" "C$CL.rand.p$k")"
    case "$seen" in *" $k "*) ;; *) out="$out x:C$CL.rand.m$k x:C$CL.rand.p$k"; seen="$seen$k " ;; esac
  done
  echo $out
}
token() { local a=$1; a=${a//:/--}; a=${a//+/-and-}; echo "${a//./-}"; }   # e44_eval_extra.py's arm_token

case "$STAGE" in
selfcheck)
  need_gpu
  ARM=$(two "E$EL.w.p$LAST_ES" "C$CL.w.m1")
  step "selfcheck $ARM (whole tuning split)"
  pym scripts/e44_eval_extra.py selfcheck "${COMMON[@]}" --arms "$ARM" ;;
gen1)
  need_gpu
  ARMS=$(arms1)
  step "generate $ARMS"
  pym scripts/e44_eval_extra.py generate "${COMMON[@]}" --arms $ARMS
  step "checks and the choice of k"
  pyj scripts/review_checks/ff_explore_ec_check.py --run-dir "$E_RUN" --c-run-dir "$C_RUN" --el "$EL" --cl "$CL" \
    --es $ES --ks $KS --out "$CHOICE" ;;
gen2)
  need_gpu
  ARMS=$(arms2)
  step "generate $ARMS"
  pym scripts/e44_eval_extra.py generate "${COMMON[@]}" --arms $ARMS ;;
judge)
  need_gpu
  : "${CACHE_FROM:?a Pro pass needs CACHE_FROM (reply caches that judged baseline x and the single writes of this model)}"
  for d in $CACHE_FROM; do [ -f "$(host "$d")/judge2_cache_deepseek-v4-pro.jsonl" ] || { echo "no Pro reply cache in $d" >&2; exit 3; }; done
  A1=$(arms1); A2=$(arms2)
  GEN=""
  for a in $A1 $A2; do
    f="$(token "$a")_tune.json"
    [ -f "$(host "$E_RUN")/eval_extra/gen/$f" ] || { echo "missing generation file eval_extra/gen/$f" >&2; exit 3; }
    GEN="$GEN $f"
  done
  step "export$GEN + baseline x"
  pym scripts/e44_eval_extra.py export "${COMMON[@]}" --gen $GEN --name "$NAME"
  mkdir -p "$(host "$WORK")/judge2/cache"
  [ -f "$(host "$WORK")/judge_input.json" ] || cp "$(host "$E_RUN")/eval_extra/judge_tune/$NAME.json" "$(host "$WORK")/judge_input.json"
  cmp "$(host "$E_RUN")/eval_extra/judge_tune/$NAME.json" "$(host "$WORK")/judge_input.json"
  step "cells + allenai + single"
  pyj scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
  "${DX[@]}" python3 scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
    --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 4
  pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$(host "$WORK")/single_allenai_stdout.md"
  verdicts_complete "$WORK" allenai
  step "merge the Pro reply caches $CACHE_FROM into the work dir's cache"
  pyj scripts/review_checks/judge2_cache_merge.py "$WORK/judge2/cache" $CACHE_FROM
  pyj -c "
import json, sys
k = json.load(open(sys.argv[1])).get('keys', {})
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop (never pick one by convenience)' % n if n else 0)" \
    "$WORK/judge2/cache/conflicts.json"
  step "pro"
  pyj scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
  pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$(host "$WORK")/single_pro_stdout.md"
  verdicts_complete "$WORK" pro ;;
describe)
  for J in allenai pro; do
    step "describe, judge $J"
    pyj scripts/review_checks/ff_explore_ec_describe.py "$WORK" --judge "$J" --el "$EL" --cl "$CL" --choice "$CHOICE" \
      > "$(host "$WORK")/describe_$J.md"
    cat "$(host "$WORK")/describe_$J.md"
  done ;;
*) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
step "$STAGE done"
