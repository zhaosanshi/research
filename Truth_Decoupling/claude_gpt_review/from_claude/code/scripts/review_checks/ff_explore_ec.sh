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
#     TAG=<name> STAGE=<stage> [GPU=k] [CACHE_FROM="<dir> [<dir> ...]"] [TABLE_CODE=<code copy>] ff_explore_ec.sh
# TABLE_CODE (added 20:3x for Llama, whose run must use the 11-11 code copy code_e45_20261112): the code copy of the two
# read-only helpers ff_explore_ec_check.py / ff_explore_ec_describe.py and of e44_round1_tables.py, which the old copy
# lacks. Default CODE_PATH (the 9B: one copy for everything, as before). Model steps and judging always use CODE_PATH.
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
# SPLIT (added 23:xx for stage B of from_claude/45; default tune = the same behaviour as before: every docker call is the
# same except that gen1's check now passes --split tune, its default; chosen_k also checks the choice file's split, which
# the tune choice files carry): SPLIT=test runs the
# same stages on the 529 test questions through the evaluation-only entry scripts/review_checks/e44_split_test.py (it
# runs e44_eval_extra.py unchanged and only widens the accepted split set by "test"); file and work-dir names carry the
# split instead of "tune"; CACHE_FROM is optional there (no test answer was ever judged; with none, nothing is merged).
set -euo pipefail
: "${MODEL:?}" "${STAGE:?}" "${CODE_PATH:?}" "${E_RUN:?}" "${C_RUN:?}" "${EL:?}" "${CL:?}" "${TAG:?}"
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$CODE_PATH
TCODE=${TABLE_CODE:-$CODE_PATH}
FF=${FF_ROOT:-$ROOT/results_newmodels_ff_20261112}
SRC=${SOURCE_DIR:-$FF/source/$MODEL}
E_ON=${E_ON:-E1}; C_E_ON=${C_E_ON:-E1}
ES=${ES:-2}; KS=${KS:-0.25 0.5 1}
SPLIT=${SPLIT:-tune}
case "$SPLIT" in tune) XE=(scripts/e44_eval_extra.py) ;; test) XE=(scripts/review_checks/e44_split_test.py extra) ;;
  *) echo "SPLIT must be tune or test" >&2; exit 2 ;; esac
# CSIGN and EXPORT (2026-11-14, rules v1.1 of from_claude/45 section 3.4; defaults m / all = exactly the behaviour before):
#   CSIGN=p  the C write of the E+C arms and of the single C arms goes toward caution (for an E strength that alone answers
#            MORE than the baseline); the choice file then carries "c_sign": "p", and gen2 / describe follow it
#   EXPORT=chosen  the judge stage exports, of the gen1 grid, only the E+C arm and the single C arm at the chosen k of each E
#            strength (plus every E single and E random arm and all of gen2); the other grid points served only the choice
CSIGN=${CSIGN:-m}
case "$CSIGN" in m|p) ;; *) echo "CSIGN must be m (toward confident) or p (toward caution)" >&2; exit 2 ;; esac
EXPORT=${EXPORT:-all}
case "$EXPORT" in all|chosen) ;; *) echo "EXPORT must be all or chosen" >&2; exit 2 ;; esac
#   ABSTAIN_EMPTY=1 (rules v1.1, red team 10:49; default 0 = as before): the choice of k counts an empty answer as an
#            abstention, as both judges do (ff_explore_ec_check.py --empty-abstains); the choice file records it
ABSTAIN_EMPTY=${ABSTAIN_EMPTY:-0}
case "$ABSTAIN_EMPTY" in 0) EA=() ;; 1) EA=(--empty-abstains) ;; *) echo "ABSTAIN_EMPTY must be 0 or 1" >&2; exit 2 ;; esac
MODELS=$ROOT/models
case "$TAG" in ""|*[!A-Za-z0-9]*) echo "TAG must be alphanumeric" >&2; exit 2 ;; esac
case "$EL$CL" in *[!0-9]*) echo "EL and CL must be layer numbers" >&2; exit 2 ;; esac
[ "$EL" != "$CL" ] || { echo "the E and C writes must act at two different layers" >&2; exit 2; }
for s in $ES $KS; do case "$s" in ""|*[!0-9.]*|.*|*.|*.*.*) echo "strength '$s' is not a number like 1 or 0.25" >&2; exit 2 ;; esac; done
host() { case "$1" in "$ROOT"/*) echo "$HOST/${1#$ROOT/}" ;; *) echo "path $1 is outside $ROOT" >&2; exit 2 ;; esac; }
for d in "$E_RUN" "$C_RUN"; do [ -f "$(host "$d")/frozen_$SPLIT.json" ] || { echo "$d has no frozen_$SPLIT.json (not a finished run, or the split is not frozen yet)" >&2; exit 3; }; done
[ -f "$(host "$SRC")/bank.json" ] || { echo "no source directory $SRC" >&2; exit 3; }
for f in scripts/e44_eval_extra.py scripts/review_checks/judge2_cache_merge.py scripts/review_checks/judge_two_axis_run.py ${XE[0]}; do
  [ -f "$(host "$CODE")/$f" ] || { echo "code copy $CODE has no $f" >&2; exit 3; }
done
for f in ff_explore_ec_check.py ff_explore_ec_describe.py e44_round1_tables.py; do
  [ -f "$(host "$TCODE")/scripts/review_checks/$f" ] || { echo "table code copy $TCODE has no $f (copy it there before running)" >&2; exit 3; }
done
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$MODELS
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES="${GPU:-}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
cfg() { "${DX[@]}" python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))" "$CODE/configs/models/$MODEL.json" "$1" "$2"; }
PYM=$(cfg python python3)                 # every phase of e44_eval_extra.py: the runtime the run was made with
pym() { "${DX[@]}" "$PYM" "$@"; }
pyj() { "${DX[@]}" python3 "$@"; }        # judging steps: the old models' runtime
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$TCODE -w $TCODE truth_dec python3)
pyt() { "${TX[@]}" "$@"; }                 # the read-only helpers (check, describe), from TABLE_CODE
need_gpu() { [ -n "${GPU:-}" ] || { echo "STAGE=$STAGE needs GPU" >&2; exit 2; }; }
step() { echo "=== $(date '+%F %T') [$MODEL E+C $TAG $STAGE gpu${GPU:-none} E$EL($E_ON) C$CL($C_E_ON)] $* ==="; }
two() {  # two E_PART C_PART: the two-write arm on x, lower layer first (the grammar of e44_eval_extra.py)
  if [ "$EL" -lt "$CL" ]; then echo "x:$1+$2"; else echo "x:$2+$1"; fi
}
verdicts_complete() {  # verdicts_complete WORK JUDGE: a cell without a verdict stops the job (never imputed)
  local out bad empty
  # an EMPTY answer is never sent to a judge and counts as an abstention (two-axis standard of 2026-11-08: "空输出一律 C";
  # first seen on the 27B, 2026-11-14): it is counted and reported, not blocking; every other missing verdict still stops
  # questions listed in judge2/excluded_questions.json (from_claude/45 section 3.5: a judge could not score them; written
  # reason per question) leave every table of the work dir, so their cells are not counted here; e44_round1_tables.load
  # checks that each listed question really holds an unscored cell
  out=$(pyj -c "
import json, os, sys
xq_path = os.path.join(os.path.dirname(sys.argv[1]), 'excluded_questions.json')
xq = json.load(open(xq_path, encoding='utf-8')) if os.path.exists(xq_path) else {}
if not isinstance(xq, dict) or not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in xq.items()):
    sys.exit('excluded_questions.json must map each pair_id to a written reason')
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
# red team 17:58: the same test as e44_round1_tables.load -- a listed question's missing verdicts here must all be the
# judge declining (pro_declined); anything else (an unfinished pass, a conflict, a malformed reply) is never hidden
for q in xq:
    b = set(r.get('basis') for r in rows if r['pair_id'] == q and r['status'] != 'scored'
            and not (r.get('basis') == 'empty_output_no_judge_vote' and r.get('empty_output') is True))
    if b - {'pro_declined'}:
        sys.exit('excluded question %s holds missing verdicts of another kind here: %s' % (q, sorted(map(str, b))))
kept = [r for r in rows if r['pair_id'] not in xq]
empty = [r for r in kept if r['status'] != 'scored' and r.get('basis') == 'empty_output_no_judge_vote' and r.get('empty_output') is True]
print(sum(r['status'] != 'scored' for r in kept) - len(empty), len(empty), len(xq), len(rows) - len(kept))" "$1/judge2/single_$2.jsonl") \
    || { echo "cannot read the $2 verdicts of $1" >&2; exit 3; }
  read -r bad empty nxq nxc <<< "$out"
  [ "$nxq" = 0 ] || echo "$2: $nxq question(s) listed in judge2/excluded_questions.json of $1 ($nxc cells) leave its tables (written reasons there)"
  [ "$empty" = 0 ] || echo "$2: $empty empty answer(s) in $1 count as abstentions (never judged; two-axis standard of 11-08)"
  [ "$bad" = 0 ] || { echo "$2 left $bad cells without a verdict in $1: re-judge them before any table" >&2; exit 3; }
}
COMMON=(--source-run "$SRC" --run-dir "$E_RUN" --e-on "$E_ON" --c-run-dir "$C_RUN" --c-e-on "$C_E_ON" --split "$SPLIT"
        --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
CHOICE=$E_RUN/eval_extra/ec_choice_$TAG.json
NAME=${TAG}_$SPLIT
WORK=$E_RUN/eval_extra/judge_${SPLIT}_$NAME
LAST_ES=${ES##* }

arms1() {  # the first generation stage: E+C grid, single E, single-E random controls, single C
  local s k out=""
  for s in $ES; do for k in $KS; do out="$out $(two "E$EL.w.p$s" "C$CL.w.$CSIGN$k")"; done; done
  for s in $ES; do out="$out x:E$EL.w.p$s x:E$EL.rand.p$s x:E$EL.rand.m$s"; done
  for k in $KS; do out="$out x:C$CL.w.$CSIGN$k"; done
  echo $out
}
chosen_k() {  # chosen_k S: the k of E strength S from the choice file, after checking the file belongs to this run
  # (and, since SPLIT, to this split: the tune choice files carry "split": "tune", written by ff_explore_ec_check.py;
  # since CSIGN, to this sign: a file without "c_sign" is an m file)
  pyj -c "
import json, sys
c = json.load(open(sys.argv[1]))
want = {'run_dir': sys.argv[2], 'c_run_dir': sys.argv[3], 'e_layer': int(sys.argv[4]), 'c_layer': int(sys.argv[5]), 'split': sys.argv[8]}
got = {k: c.get(k) for k in want}
if got != want or sorted(c['choice']) != sorted(sys.argv[6].split()) or c.get('c_sign', 'm') != sys.argv[9] or \
        bool(c.get('empty_counts_as_abstention', False)) != (sys.argv[10] == '1'):
    sys.exit('choice file %s is for another run, split, sign, empty-answer rule, other layers or other E strengths: %s vs %s' % (sys.argv[1], got, want))
print(c['choice'][sys.argv[7]])" "$CHOICE" "$E_RUN" "$C_RUN" "$EL" "$CL" "$ES" "$1" "$SPLIT" "$CSIGN" "$ABSTAIN_EMPTY"
}
arms1_export() {  # the gen1 arms the judge stage exports: all of them (EXPORT=all), or only the chosen k's (EXPORT=chosen)
  local s k out="" seen=" "
  [ "$EXPORT" = all ] && { arms1; return 0; }
  for s in $ES; do
    k=$(chosen_k "$s") || return 3
    out="$out $(two "E$EL.w.p$s" "C$CL.w.$CSIGN$k") x:E$EL.w.p$s x:E$EL.rand.p$s x:E$EL.rand.m$s"
    case "$seen" in *" $k "*) ;; *) out="$out x:C$CL.w.$CSIGN$k"; seen="$seen$k " ;; esac
  done
  echo $out
}
arms2() {  # the second generation stage, for each E strength and its chosen k only
  local s k out="" seen=" "
  [ -f "$(host "$CHOICE")" ] || { echo "no choice file $CHOICE: run STAGE=gen1 to the end first" >&2; return 3; }
  for s in $ES; do
    k=$(chosen_k "$s") || return 3
    out="$out $(two "E$EL.rand.p$s" "C$CL.w.$CSIGN$k") $(two "E$EL.rand.m$s" "C$CL.w.$CSIGN$k")"
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
  step "selfcheck $ARM (whole $SPLIT split)"
  pym "${XE[@]}" selfcheck "${COMMON[@]}" --arms "$ARM" ;;
gen1)
  need_gpu
  ARMS=$(arms1)
  step "generate $ARMS"
  pym "${XE[@]}" generate "${COMMON[@]}" --arms $ARMS
  step "checks and the choice of k"
  pyt scripts/review_checks/ff_explore_ec_check.py --run-dir "$E_RUN" --c-run-dir "$C_RUN" --el "$EL" --cl "$CL" \
    --es $ES --ks $KS --split "$SPLIT" --c-sign "$CSIGN" ${EA[@]+"${EA[@]}"} --out "$CHOICE" ;;
gen2)
  need_gpu
  ARMS=$(arms2)
  step "generate $ARMS"
  pym "${XE[@]}" generate "${COMMON[@]}" --arms $ARMS ;;
judge)
  # JUDGES (added with SPLIT; default "allenai pro" = as before): allenai = export + AllenAI only (GPU); pro = the cache
  # merge + DeepSeek Pro only (no GPU), on a work dir whose AllenAI pass is complete
  JUDGES=${JUDGES:-allenai pro}
  for j in $JUDGES; do case "$j" in allenai|pro) ;; *) echo "JUDGES: unknown judge '$j' (allenai / pro)" >&2; exit 2 ;; esac; done
  [ -n "$JUDGES" ] || { echo "JUDGES is empty" >&2; exit 2; }
  case " $JUDGES " in *" allenai "*) DO_A=1; need_gpu ;; *) DO_A=0 ;; esac
  case " $JUDGES " in *" pro "*) DO_P=1 ;; *) DO_P=0 ;; esac
  if [ "$SPLIT" = tune ] && [ $DO_P = 1 ]; then
    : "${CACHE_FROM:?a Pro pass needs CACHE_FROM (reply caches that judged baseline x and the single writes of this model)}"
  fi
  for d in ${CACHE_FROM:-}; do [ -f "$(host "$d")/judge2_cache_deepseek-v4-pro.jsonl" ] || { echo "no Pro reply cache in $d" >&2; exit 3; }; done
  if [ $DO_A = 1 ]; then
  A1=$(arms1_export); A2=$(arms2)
  GEN=""
  for a in $A1 $A2; do
    f="$(token "$a")_$SPLIT.json"
    [ -f "$(host "$E_RUN")/eval_extra/gen/$f" ] || { echo "missing generation file eval_extra/gen/$f" >&2; exit 3; }
    GEN="$GEN $f"
  done
  step "export$GEN + baseline x"
  pym "${XE[@]}" export "${COMMON[@]}" --gen $GEN --name "$NAME"
  mkdir -p "$(host "$WORK")/judge2/cache"
  [ -f "$(host "$WORK")/judge_input.json" ] || cp "$(host "$E_RUN")/eval_extra/judge_$SPLIT/$NAME.json" "$(host "$WORK")/judge_input.json"
  cmp "$(host "$E_RUN")/eval_extra/judge_$SPLIT/$NAME.json" "$(host "$WORK")/judge_input.json"
  step "cells + allenai + single"
  pyj scripts/review_checks/judge_two_axis_run.py cells "$WORK" --generation judge_input.json | grep -E "n_cells|n_unique_question_answer"
  "${DX[@]}" python3 scripts/allenai_tqa_judge.py --cells "$WORK/judge2/cells.jsonl" --out "$WORK/judge2/allenai.jsonl" \
    --truth-model "$MODELS/allenai__truthfulqa-truth-judge-llama2-7B" --info-model "$MODELS/allenai__truthfulqa-info-judge-llama2-7B" | tail -n 4
  pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge allenai > "$(host "$WORK")/single_allenai_stdout.md"
  else
    [ -f "$(host "$WORK")/judge2/single_allenai.jsonl" ] || { echo "no AllenAI pass in $WORK yet: run JUDGES=allenai first" >&2; exit 3; }
  fi
  verdicts_complete "$WORK" allenai
  if [ $DO_P = 0 ]; then step "pro skipped (JUDGES=$JUDGES)"; step "$STAGE done"; exit 0; fi
  if [ -n "${CACHE_FROM:-}" ]; then
  step "merge the Pro reply caches $CACHE_FROM into the work dir's cache"
  pyj scripts/review_checks/judge2_cache_merge.py "$WORK/judge2/cache" $CACHE_FROM
  pyj -c "
import json, sys
k = json.load(open(sys.argv[1])).get('keys', {})
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop (never pick one by convenience)' % n if n else 0)" \
    "$WORK/judge2/cache/conflicts.json"
  else
  step "no CACHE_FROM ($SPLIT split): the Pro pass starts from an empty reply cache"
  fi
  step "pro"
  pyj scripts/review_checks/judge_two_axis_run.py llm "$WORK" --judge pro
  pyj scripts/review_checks/judge_two_axis_run.py single "$WORK" --judge pro > "$(host "$WORK")/single_pro_stdout.md"
  verdicts_complete "$WORK" pro ;;
describe)
  for J in allenai pro; do
    step "describe, judge $J"
    pyt scripts/review_checks/ff_explore_ec_describe.py "$WORK" --judge "$J" --el "$EL" --cl "$CL" --choice "$CHOICE" \
      > "$(host "$WORK")/describe_$J.md"
    cat "$(host "$WORK")/describe_$J.md"
  done ;;
*) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
step "$STAGE done"
