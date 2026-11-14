#!/usr/bin/env bash
# Chain r of 2026-11-14 (Claude, 11:1x; rules v1.1 of from_claude/45 section 3.4, written before any v1.1 run): Llama's E+C
# with C at layer 19 (E2 layer 11, the R6 C layer) again under v1.1, because v1's choice missed baseline x by more than the
# tolerance (tune: 33 / 32 rule abstentions against 23, tolerance 6; test, E strength 2: 34 against 60, tolerance 16), the
# closest grid points bracketing the baseline being 0.5 and 1. Chain q's program with Llama's parameters only:
#   - KS = 0.25 0.5 0.6 0.7 0.8 0.9 1 2 (v1: 0.25 0.5 1 2; refined between the two points that bracket the baseline);
#   - abstentions = rule abstention or empty answer (ABSTAIN_EMPTY=1); a candidate with more empty answers than the
#     tolerance is not eligible; the C sign per (split, s) from E s alone's abstentions against baseline x with the
#     tolerance (Llama: E alone abstains more on both splits, so m, as in v1);
#   - one TAG per (split, s): ecll19q<s><sign> on tune, tll19q<s><sign> on test; the judge stage exports only the chosen k.
# Runs: E = results_e44_factorial_das/llama3.1-8b_side_E2 (E2), C = results_e44_factorial_das/llama3.1-8b (E1), code copy
# code_e45_20261112 (the 11-11 runs' identity), tables / check / describe from code_ff_explore_20261113b (as chain test_b).
# Order: chain test_b (the v1 E+C of the same pair) must be over; tune: selfcheck, per s gen1 / gen2 / AllenAI; then test.
# The selfcheck record of the pair (named by the two directions only) found at start is copied to LOG/selfcheck_before_r/.
# Card: CHAIN_GPU (default 2). Mechanics as chain q. Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
EC=${CHAIN_EC_RUNNER:-$HERE/ff_explore_ec.sh}                     # overrides: dry runs only
LOGS=${CHAIN_LOGS:-$R/results_newmodels_ff_20261112/logs}
LOG=$LOGS/explore_r
TBLOG=$LOGS/test_b
NLOG=$LOGS/explore_n
PLOG=$LOGS/explore_p
POLL=${CHAIN_POLL:-60}
G=${CHAIN_GPU:-2}
KSQ=${CHAIN_KS:-0.25 0.5 0.6 0.7 0.8 0.9 1 2}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain r holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u MODEL -u STAGE -u CODE_PATH -u TABLE_CODE -u SOURCE_DIR -u E_RUN -u C_RUN -u E_ON -u C_E_ON -u EL -u CL -u CLS
       -u ECLS -u ES -u KS -u TAG -u LAYER -u JUDGES -u SPLIT -u CACHE_FROM -u FF_ROOT -u CSIGN -u EXPORT -u ABSTAIN_EMPTY)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {  # job NAME VAR=value ...: one ff_explore_ec.sh call in a clean environment; skipped when NAME.DONE exists
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  [ -e "$LOG/$name.log" ] && mv -f "$LOG/$name.log" "$LOG/$name.log.prev"
  if env "${CLEAN[@]}" "$@" bash "$EC" > "$LOG/$name.log" 2>&1; then
    touch "$LOG/$name.DONE"; note "done  $name"
  else
    echo "failed, see $name.log" > "$LOG/$name.FAIL"; note "FAIL  $name"; return 1
  fi
}
give_up() {
  local n
  for n in "$@"; do
    [ -e "$LOG/$n.DONE" ] || [ -e "$LOG/$n.FAIL" ] || { echo "not tried: an earlier job of its group failed" > "$LOG/$n.FAIL"; note "skip  $n"; }
  done
}
upstream_over() {  # upstream_over DIR NAME...: each DONE or FAIL, or that chain has ended
  local d=$1 f; shift
  for f in "$@"; do
    until [ -e "$d/$f.DONE" ] || [ -e "$d/$f.FAIL" ] || [ ! -d "$d/.lock_chain" ]; do sleep "$POLL"; done
  done
}

CODE=$W/code_e45_20261112                                        # Llama's 11-11 code copy (its runs' identity)
TCODE=$W/code_ff_explore_20261113b                                # tables, check, describe, cache pick (as chain test_b)
RUN=$W/results_e44_factorial_das/llama3.1-8b_side_E2              # the E run (E2); eval_extra/ of the pair lives here
CRUN=$W/results_e44_factorial_das/llama3.1-8b                     # the C run (E1)
SRC=$W/results_qa_das_v1u/llama3.1-8b/full
HRUN=${CHAIN_HRUN:-$R/${RUN#$W/}}                                  # override: dry runs only
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$TCODE -w $TCODE truth_dec python3)
if [ -z "${CHAIN_EC_RUNNER:-}" ]; then
  for f in ff_explore_ec.sh ff_explore_ec_check.py ff_explore_ec_describe.py ff_cache_pick.py e44_round1_tables.py; do
    cmp -s "$HERE/$f" "$R/${TCODE#$W/}/scripts/review_checks/$f" \
      || { echo "$f next to this chain is missing or differs from the 13b copy" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
  done
  cmp -s "$HERE/e44_split_test.py" "$R/${CODE#$W/}/scripts/review_checks/e44_split_test.py" \
    || { echo "e44_split_test.py differs between this chain and $CODE" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
  grep -q 'ABSTAIN_EMPTY=${ABSTAIN_EMPTY:-0}' "$EC" || { echo "$EC is not the v1.1 E+C runner" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
fi
EL=11; CL=19                                                      # Llama's R3 E2 layer and R6 C layer (from_claude/45 section 1)
X=(MODEL=llama3.1-8b CODE_PATH=$CODE TABLE_CODE=$TCODE SOURCE_DIR=$SRC E_RUN=$RUN C_RUN=$CRUN E_ON=E2 C_E_ON=E1 EL=$EL CL=$CL
   "KS=$KSQ" ABSTAIN_EMPTY=1)
sign_of() {  # sign_of SPLIT S -> "m|p|none N BASE TAU": E S alone's abstentions (rule or empty) against baseline x
  python3 - "$HRUN" "$1" "$EL" "$2" <<'EOF'
import json, sys
run, sp, el, s = sys.argv[1:5]
abst = lambda a: bool(a["refusal_rule"]) or not a["text"].strip()
fz = json.load(open("%s/frozen_%s.json" % (run, sp), encoding="utf-8"))
base = sum(1 for q in fz["questions"] if abst(q["answers"]["x"]))
tau = round(0.03 * len(fz["questions"]))
g = json.load(open("%s/eval_extra/gen/x--E%s-w-p%s_%s.json" % (run, el, s.replace(".", "-"), sp), encoding="utf-8"))
if g.get("completed") is not True or g.get("partial") not in (None, False) or g.get("split") != sp or len(g["rows"]) != len(fz["questions"]):
    sys.exit("E %s alone on %s is not a complete generation of the same questions" % (s, sp))
n = sum(1 for r in g["rows"] if abst(r))
print("none" if abs(n - base) <= tau else ("m" if n > base else "p"), n, base, tau)
EOF
}
mkdir -p "$LOG/selfcheck_before_r"
for f in "$HRUN"/eval_extra/selfcheck_E${EL}-*_C${CL}-*.json; do [ -e "$f" ] && { [ -e "$LOG/selfcheck_before_r/${f##*/}" ] || cp -p "$f" "$LOG/selfcheck_before_r/"; }; done
backup_selfcheck() {  # copy the pair's selfcheck record as just made for SPLIT
  local d=$LOG/selfcheck_$1_backup f
  mkdir -p "$d"
  for f in "$HRUN"/eval_extra/selfcheck_E${EL}-*_C${CL}-*.json; do [ -e "$f" ] && cp -p "$f" "$d/"; done
}
TAGS_tune=""; TAGS_test=""
block() {  # block SPLIT PREFIX: signs, selfcheck, then per E strength gen1, gen2, AllenAI (EXPORT=chosen); sets TAGS_<SPLIT>
  local sp=$1 pre=$2 s sign n base tau t p first plan="" tags="" pending=0 sc ran
  for s in 1 2; do
    read -r sign n base tau <<< "$(sign_of "$sp" "$s")"
    case "$sign" in m|p|none) ;; *) note "cannot read E $s alone on $sp: block $sp stops"; return 0 ;; esac
    note "$sp: E $s alone has $n abstentions (rule or empty), baseline x $base, tolerance $tau -> C sign $sign"
    [ "$sign" = none ] && continue
    t=${pre}ll${CL}q${s}${sign}
    plan="$plan $t:$s:$sign"
    { [ -e "$LOG/${t}_g1.DONE" ] && [ -e "$LOG/${t}_g2.DONE" ]; } || pending=1
  done
  [ -n "$plan" ] || { note "$sp: E alone is within tolerance of baseline x at every strength: no E+C needed"; return 0; }
  sc=${pre}q_s_$sp
  [ $pending = 1 ] && rm -f "$LOG/$sc.DONE"                     # a pending generation always gets a fresh record
  ran=0; [ -e "$LOG/$sc.DONE" ] || ran=1
  first=${plan# }; first=${first%% *}                           # the first planned TAG:S:SIGN (names the selfcheck run)
  IFS=: read -r t s sign <<< "$first"
  job $sc "${X[@]}" SPLIT=$sp ES=$s TAG=$t STAGE=selfcheck GPU=$G || { note "selfcheck on $sp failed: block $sp stops"; return 0; }
  [ $ran = 1 ] && backup_selfcheck "$sp"
  for p in $plan; do
    IFS=: read -r t s sign <<< "$p"
    { job ${t}_g1 "${X[@]}" SPLIT=$sp ES=$s CSIGN=$sign TAG=$t STAGE=gen1 GPU=$G &&
      job ${t}_g2 "${X[@]}" SPLIT=$sp ES=$s CSIGN=$sign TAG=$t STAGE=gen2 GPU=$G &&
      job ${t}_a  "${X[@]}" SPLIT=$sp ES=$s CSIGN=$sign TAG=$t STAGE=judge JUDGES=allenai EXPORT=chosen GPU=$G &&
      tags="$tags $p"; } || give_up ${t}_g2 ${t}_a ${t}_p ${t}_d
  done
  eval "TAGS_$sp=\"\$tags\""
}
pro_lane() {  # pro_lane SPLIT TAG S SIGN SEED_DIR...: DeepSeek Pro (seeded) and describe, in the background
  local sp=$1 t=$2 s=$3 sign=$4 d have=() seeds=""; shift 4
  for d in "$@"; do [ -f "$R/${d#$W/}/judge2_cache_deepseek-v4-pro.jsonl" ] && have+=("$d"); done
  if [ ${#have[@]} -gt 0 ]; then
    seeds=$("${TX[@]}" scripts/review_checks/ff_cache_pick.py "${have[@]}" 2>> "$LOG/cache_pick.log" | tee -a "$LOG/cache_pick.log" | sed -n 's/^use: *//p')
  fi
  note "$t Pro seeded from: ${seeds:-nothing}"
  if [ -z "$seeds" ] && [ "$sp" = tune ]; then
    note "no conflict-free Pro reply cache for $t (the tune Pro pass needs one): Pro not tried"; give_up ${t}_p ${t}_d; return 0
  fi
  { job ${t}_p "${X[@]}" SPLIT=$sp ES=$s CSIGN=$sign TAG=$t STAGE=judge JUDGES=pro ${seeds:+"CACHE_FROM=$seeds"} &&
    job ${t}_d "${X[@]}" SPLIT=$sp ES=$s CSIGN=$sign TAG=$t STAGE=describe; } || give_up ${t}_p ${t}_d
}

note "chain start (pid $$): Llama E$EL + C$CL, rules v1.1, KS [$KSQ], card $G"
( cd "$HERE" && sha256sum ff_explore_ec.sh ff_explore_ec_check.py ff_explore_ec_describe.py ff_cache_pick.py ff_explore_20261114r_chain.sh ) \
  >> "$LOG/chain_jobs.log" 2>&1 || true
[ -d "$TBLOG/.lock_chain" ] && { note "chain test_b still runs (the v1 E+C of the same pair): stop"; exit 3; }
for f in ll_ecp_C$CL ll_ec2_C$CL; do [ -e "$TBLOG/$f.DONE" ] || { note "chain test_b has no $f.DONE: stop"; exit 3; }; done
block tune ec
for p in $TAGS_tune; do
  IFS=: read -r t s sign <<< "$p"
  pro_lane tune "$t" "$s" "$sign" "$RUN/eval_extra/judge_tune_ecll${CL}_tune/judge2/cache" \
    "$CRUN/judge_tune_proS_C_L$CL/judge2/cache" &
done
block test t
for p in $TAGS_test; do
  IFS=: read -r t s sign <<< "$p"
  pro_lane test "$t" "$s" "$sign" "$RUN/eval_extra/judge_test_tll${CL}_test/judge2/cache" \
    "$CRUN/judge_test_testS_C_L$CL/judge2/cache" &
done
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"
