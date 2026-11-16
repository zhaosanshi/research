#!/usr/bin/env bash
# The equal-N comparison of from_claude/39 for an E+C run of ff_explore_ec.sh (Claude, 2026-11-13; from_claude/44): at the
# same number N of non-abstained answers, E+C against "answering only the N answers the model itself is most confident
# in" (the decisive comparator: E on the must-answer prompt, then keep the N most confident). The Llama procedure
# (results/09_E44_两因子交换训练_Llama/scripts: genS of e44_extra_EC.sh, e44_confidence.py, e44_EC_compare.py) with nothing
# model-specific in the script; the two Python steps are ff_explore_ec_confidence.py and ff_explore_ec_compare.py, which
# reproduce Llama's 11-12 confidence file token for token (max difference 0) and its comparison table number for number.
# Same parameters as ff_explore_ec.sh (same E_RUN / C_RUN / EL / CL / ES / KS / TAG; the E+C TAG's choice file must exist,
# and its work dir must be judged before compare).
#   STAGE=genconf GPU=k     E alone on the must-answer prompt at each E strength (conf:E<EL>.w.p<s>), e44_eval_extra.py;
#                           the driver's own files of the same arms, where they exist, must be reproduced exactly
#   STAGE=judgeconf GPU=k   one export of those files + baseline conf; AllenAI (GPU), then DeepSeek Pro (caches CACHE_FROM
#                           merged, a conflicting key stops the job)
#   STAGE=confidence GPU=k  ff_explore_ec_confidence.py with the model's own interpreter (refuses an existing output);
#                           SCORE_BATCH (default 8, as for Llama) is its scoring batch size
#   STAGE=compare           ff_explore_ec_compare.py (read-only), both judges; table and JSON under the E+C work dir
# SPLIT (2026-11-14, stage B of from_claude/45 R6; default tune = the behaviour before, every call the same apart from
# the confidence step passing its default --split tune): SPLIT=test runs on the 529 test questions through the
# evaluation-only entries (e44_split_test.py extra for genconf, ff_test_confidence.py for the confidence step); file and
# work-dir names carry the split. On test, judgeconf is not run: the stage-B E work dir (ff_test_eval.sh, both judges)
# already judged E on the must-answer prompt and baseline conf, and genconf requires the engine to reproduce the driver's
# test files of those arms exactly, so that work dir is the conf work dir (CONF_WORK). The E+C work dir must be judged
# only for compare (genconf / confidence read generations, never verdicts).
set -euo pipefail
: "${MODEL:?}" "${STAGE:?}" "${CODE_PATH:?}" "${E_RUN:?}" "${C_RUN:?}" "${EL:?}" "${CL:?}" "${TAG:?}"
SPLIT=${SPLIT:-tune}
case "$SPLIT" in
  tune) XE=(scripts/e44_eval_extra.py); CX=(scripts/review_checks/ff_explore_ec_confidence.py) ;;
  test) XE=(scripts/review_checks/e44_split_test.py extra); CX=(scripts/review_checks/ff_test_confidence.py) ;;
  *) echo "SPLIT must be tune or test" >&2; exit 2 ;;
esac
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$CODE_PATH
TCODE=${TABLE_CODE:-$CODE_PATH}
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
for d in "$E_RUN" "$C_RUN"; do [ -f "$(host "$d")/frozen_$SPLIT.json" ] || { echo "$d has no frozen_$SPLIT.json (not a finished run, or the split is not frozen)" >&2; exit 3; }; done
[ -f "$(host "$SRC")/bank.json" ] || { echo "no source directory $SRC" >&2; exit 3; }
for f in scripts/e44_eval_extra.py scripts/review_checks/judge2_cache_merge.py scripts/review_checks/judge_two_axis_run.py \
         scripts/review_checks/ff_explore_ec_confidence.py ${XE[0]} ${CX[0]}; do
  [ -f "$(host "$CODE")/$f" ] || { echo "code copy $CODE has no $f (copy it there before running)" >&2; exit 3; }
done
for f in ff_explore_ec_compare.py e44_round1_tables.py; do
  [ -f "$(host "$TCODE")/scripts/review_checks/$f" ] || { echo "table code copy $TCODE has no $f" >&2; exit 3; }
done
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$MODELS
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES="${GPU:-}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
cfg() { "${DX[@]}" python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))" "$CODE/configs/models/$MODEL.json" "$1" "$2"; }
PYM=$(cfg python python3)
pym() { "${DX[@]}" "$PYM" "$@"; }
pyj() { "${DX[@]}" python3 "$@"; }
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$TCODE -w $TCODE truth_dec python3)
pyt() { "${TX[@]}" "$@"; }
need_gpu() { [ -n "${GPU:-}" ] || { echo "STAGE=$STAGE needs GPU" >&2; exit 2; }; }
step() { echo "=== $(date '+%F %T') [$MODEL E+C selection $TAG $STAGE gpu${GPU:-none} E$EL($E_ON) C$CL($C_E_ON)] $* ==="; }
verdicts_complete() {
  local n
  n=$(pyj -c "import json,sys; print(json.load(open(sys.argv[1])).get('n_no_verdict', 'missing'))" "$1/judge2/single_$2_summary.json")
  [ "$n" = 0 ] || { echo "$2 left $n cells without a verdict in $1: re-judge them before any table" >&2; exit 3; }
}
COMMON=(--source-run "$SRC" --run-dir "$E_RUN" --e-on "$E_ON" --c-run-dir "$C_RUN" --c-e-on "$C_E_ON" --split "$SPLIT"
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
CHOICE=$E_RUN/eval_extra/ec_choice_$TAG.json
ECWORK=$E_RUN/eval_extra/judge_${SPLIT}_${TAG}_$SPLIT
if [ "$SPLIT" = tune ]; then
  CNAME=${TAG}conf_tune
  CWORK=$E_RUN/eval_extra/judge_tune_$CNAME
else
  : "${CONF_WORK:?SPLIT=test needs CONF_WORK: the stage-B E work dir of this E run (judge_test_testD_E_L<EL>)}"
  [ "$CONF_WORK" = "$E_RUN/judge_test_testD_E_L$EL" ] || { echo "CONF_WORK must be $E_RUN/judge_test_testD_E_L$EL (got $CONF_WORK)" >&2; exit 2; }
  CWORK=$CONF_WORK
fi
CONF_OUT=$E_RUN/eval_extra/confidence_${TAG}.json
[ -f "$(host "$CHOICE")" ] || { echo "no choice file $CHOICE: the E+C run of TAG=$TAG is not done" >&2; exit 3; }
[ "$SPLIT" = tune ] || pyj -c "
import json, sys
c = json.load(open(sys.argv[1]))
want = {'run_dir': sys.argv[2], 'c_run_dir': sys.argv[3], 'e_layer': int(sys.argv[4]), 'c_layer': int(sys.argv[5])}
got = {k: c.get(k) for k in want}
if got != want or c.get('split', 'tune') != sys.argv[6]:
    sys.exit('choice file %s is for another run, other layers or another split: %s / %s' % (sys.argv[1], got, c.get('split')))
" "$CHOICE" "$E_RUN" "$C_RUN" "$EL" "$CL" "$SPLIT"

case "$STAGE" in
genconf)
  need_gpu
  ARMS=""; for s in $ES; do ARMS="$ARMS conf:E$EL.w.p$s"; done
  step "generate$ARMS"
  pym "${XE[@]}" generate "${COMMON[@]}" --arms $ARMS
  step "reproduction against the driver's own files"
  for s in $ES; do
    # on test the driver file must exist: its judged answers stand in for these (CONF_WORK), so they must be the same
    pyj -c "
import json, sys
e = json.load(open(sys.argv[1])); d = sys.argv[2]
import os
if not os.path.exists(d):
    if sys.argv[3] == 'test':
        sys.exit('- %s: no driver file to compare, which SPLIT=test needs' % sys.argv[1].rsplit('/', 1)[-1])
    print('- %s: no driver file to compare' % sys.argv[1].rsplit('/', 1)[-1]); sys.exit(0)
d = json.load(open(d))
a = {r['pair_id']: r for r in e['rows']}; b = {r['pair_id']: r for r in d['rows']}
same = sum(1 for q in a if q in b and all(a[q][f] == b[q][f] for f in ('token_ids', 'finished', 'stop_id', 'text')))
print('- %s: %d / %d identical with the driver file' % (sys.argv[1].rsplit('/', 1)[-1], same, len(a)))
sys.exit(0 if same == len(a) == len(b) else 1)" "$E_RUN/eval_extra/gen/conf--E$EL-w-p${s//./-}_$SPLIT.json" "$E_RUN/gen/E_L${EL}_conf-w-p${s//./-}_$SPLIT.json" "$SPLIT"
  done ;;
judgeconf)
  need_gpu
  [ "$SPLIT" = tune ] || { echo "SPLIT=test: nothing to judge here; CONF_WORK (the stage-B E work dir) judged these answers" >&2; exit 2; }
  : "${CACHE_FROM:?a Pro pass needs CACHE_FROM}"
  for d in $CACHE_FROM; do [ -f "$(host "$d")/judge2_cache_deepseek-v4-pro.jsonl" ] || { echo "no Pro reply cache in $d" >&2; exit 3; }; done
  GEN=""; for s in $ES; do f="conf--E$EL-w-p${s//./-}_tune.json"; [ -f "$(host "$E_RUN")/eval_extra/gen/$f" ] || { echo "missing $f" >&2; exit 3; }; GEN="$GEN $f"; done
  step "export$GEN + baseline conf"
  pym scripts/e44_eval_extra.py export "${COMMON[@]}" --gen $GEN --name "$CNAME"
  mkdir -p "$(host "$CWORK")/judge2/cache"
  [ -f "$(host "$CWORK")/judge_input.json" ] || cp "$(host "$E_RUN")/eval_extra/judge_tune/$CNAME.json" "$(host "$CWORK")/judge_input.json"
  cmp "$(host "$E_RUN")/eval_extra/judge_tune/$CNAME.json" "$(host "$CWORK")/judge_input.json"
  step "cells + allenai + single"
  pyj scripts/review_checks/judge_two_axis_run.py cells "$CWORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
  "${DX[@]}" python3 scripts/allenai_tqa_judge.py --cells "$CWORK/judge2/cells.jsonl" --out "$CWORK/judge2/allenai.jsonl" \
    --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 4
  pyj scripts/review_checks/judge_two_axis_run.py single "$CWORK" --judge allenai > "$(host "$CWORK")/single_allenai_stdout.md"
  verdicts_complete "$CWORK" allenai
  step "merge the Pro reply caches $CACHE_FROM"
  pyj scripts/review_checks/judge2_cache_merge.py "$CWORK/judge2/cache" $CACHE_FROM
  pyj -c "
import json, sys
k = json.load(open(sys.argv[1])).get('keys', {})
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" "$CWORK/judge2/cache/conflicts.json"
  step "pro"
  pyj scripts/review_checks/judge_two_axis_run.py llm "$CWORK" --judge pro
  pyj scripts/review_checks/judge_two_axis_run.py single "$CWORK" --judge pro > "$(host "$CWORK")/single_pro_stdout.md"
  verdicts_complete "$CWORK" pro ;;
confidence)
  need_gpu
  [ ! -e "$(host "$CONF_OUT")" ] || { echo "$CONF_OUT exists (never overwritten)" >&2; exit 3; }
  step "confidence of the model in its own answers"
  pym "${CX[@]}" "$TAG" --source-run "$SRC" --run-dir "$E_RUN" --e-on "$E_ON" \
    --c-run-dir "$C_RUN" --c-e-on "$C_E_ON" --el "$EL" --cl "$CL" --es $ES --ks $KS --batch "${SCORE_BATCH:-8}" --split "$SPLIT" ;;
compare)
  [ -f "$(host "$CONF_OUT")" ] || { echo "no confidence file $CONF_OUT" >&2; exit 3; }
  [ -f "$(host "$ECWORK")/judge2/single_pro_summary.json" ] || { echo "the E+C export of TAG=$TAG is not judged ($ECWORK)" >&2; exit 3; }
  [ -f "$(host "$CWORK")/judge2/single_pro_summary.json" ] || { echo "the conf export is not judged ($CWORK)" >&2; exit 3; }
  step "compare at the same N, both judges"
  pyt scripts/review_checks/ff_explore_ec_compare.py "$CONF_OUT" --ec-work "$ECWORK" --conf-work "$CWORK" --choice "$CHOICE" \
    --json-out "$ECWORK/compare_same_N.json" > "$(host "$ECWORK")/compare_same_N.md.tmp"
  mv -f "$(host "$ECWORK")/compare_same_N.md.tmp" "$(host "$ECWORK")/compare_same_N.md"
  grep -A 20 "pre-registered readings" "$(host "$ECWORK")/compare_same_N.md" ;;
*) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
step "$STAGE done"
