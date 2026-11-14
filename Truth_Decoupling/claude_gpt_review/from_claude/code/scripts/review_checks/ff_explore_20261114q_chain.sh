#!/usr/bin/env bash
# Chain q of 2026-11-14 (Claude, 10:xx; rules v1.1 of from_claude/45 section 3.4, written before this chain ran; 赵磊 10:1x
# "继续调小C啊，谁说0.25就是最小档，只是我们约定的，我说过，我们现在是在探索方法，不是在做法律约定"): the 27B's E+C again
# (E = the R3 layer, C = the R6 layer; read from chains n / p), with
#   - the C grid extended below 0.25: KS = 0.03 0.06 0.1 0.15 0.25 0.5 1 2 (v1: 0.25 0.5 1 2; on the 27B even 0.25 left
#     40 more answers than baseline x on tune);
#   - abstentions counted as both judges count them: rule abstention OR empty answer (ABSTAIN_EMPTY=1; red team 10:49:
#     the rule detector alone does not flag an empty answer, which could pick a broken large k toward caution);
#   - the C sign set per E strength s by E s alone's abstentions on the same split (never correctness): within the
#     tolerance of baseline x (3% of the questions: tune 6, test 16) -> no C needed (only E alone is reported); more ->
#     C toward confident (m, as in v1); fewer -> toward caution (p);
#   - the choice of k unchanged: |abstentions of E s + C k - baseline x| smallest, ties -> smaller k.
# One TAG per (split, s): ec27<CL>q<s><sign> on tune, t27<CL>q<s><sign> on test ("27" = the model, as chain p's ec27<CL>).
# The judge stage exports only the chosen k (EXPORT=chosen; the other grid points served only the choice). Pro seeds: the
# conflict-free set (ff_cache_pick.py) of the same split's v1 caches (tune: v1 E+C, C screen and E screen work dirs; test:
# v1 E+C and the C set), once the v1 E+C Pro pass of that split is over; tune Pro is not tried without a seed.
# Order: the engine names a selfcheck record by the two directions only, so generation on tune and on test never overlaps
# (nor with chain p's test generation): wait for chain p's t27_ec2 to be over; tune: selfcheck, then per s gen1, gen2,
# AllenAI; test: selfcheck, then the same. Records: the one found at start is copied to LOG/selfcheck_before_q/ (chain p's
# test record; never overwritten there), and each selfcheck this chain makes is copied to LOG/selfcheck_<split>_backup/
# right after it RAN (not when skipped). A split's selfcheck DONE flag is removed at the block start unless all of that
# split's gen1 / gen2 jobs are DONE, so a rerun after a failed generation makes a fresh record first (as chain p does).
# Do not restart chain p while this chain runs (its lock counts as "over" for the waits below).
# Cards: CHAIN_GPU (default 0,1; the 27B config splits one model over two cards). Mechanics as chains b-p (DONE flags
# skip, FAIL marks the rest "not tried", lock directory). Never edit this file while it runs.
set -uo pipefail
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
HERE=$(cd "$(dirname "$0")" && pwd)
EC=${CHAIN_EC_RUNNER:-$HERE/ff_explore_ec.sh}                     # overrides: dry runs only
LOGS=${CHAIN_LOGS:-$R/results_newmodels_ff_20261112/logs}
LOG=$LOGS/explore_q
NLOG=$LOGS/explore_n
PLOG=$LOGS/explore_p
POLL=${CHAIN_POLL:-60}
G=${CHAIN_GPU:-0,1}
KSQ=${CHAIN_KS:-0.03 0.06 0.1 0.15 0.25 0.5 1 2}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain q holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
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

CODE=$W/code_ff_explore_20261113b
RUN=$W/results_newmodels_ff_20261112/e44/qwen3.8-27b-dm
SRC=$W/results_newmodels_ff_20261112/source/qwen3.8-27b-dm
HRUN=${CHAIN_HRUN:-$R/${RUN#$W/}}                                  # override: dry runs only
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec python3)
if [ -z "${CHAIN_EC_RUNNER:-}" ]; then
  for f in ff_explore_ec.sh ff_explore_ec_check.py ff_explore_ec_describe.py ff_cache_pick.py e44_split_test.py e44_round1_tables.py; do
    cmp -s "$HERE/$f" "$R/${CODE#$W/}/scripts/review_checks/$f" \
      || { echo "$f next to this chain is missing or differs from the 13b copy" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
  done
  grep -q 'ABSTAIN_EMPTY=${ABSTAIN_EMPTY:-0}' "$EC" || { echo "$EC is not the v1.1 E+C runner" | tee -a "$LOG/chain_jobs.log" >&2; exit 3; }
fi
EL=$(sed -n 's/^chosen: //p' "$NLOG/r3_choice.md" 2>/dev/null)
CL=$(sed -n 's/^r6: //p' "$PLOG/r5_r6.md" 2>/dev/null)
for v in "$EL" "$CL"; do
  case "$v" in ""|*[!0-9]*) echo "cannot read the R3 layer ($EL) or the R6 layer ($CL)" | tee -a "$LOG/chain_jobs.log" >&2; exit 3 ;; esac
done
X=(MODEL=qwen3.8-27b-dm CODE_PATH=$CODE SOURCE_DIR=$SRC E_RUN=$RUN C_RUN=$RUN E_ON=E2 C_E_ON=E2 EL=$EL CL=$CL "KS=$KSQ" ABSTAIN_EMPTY=1)
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
mkdir -p "$LOG/selfcheck_before_q"
for f in "$HRUN"/eval_extra/selfcheck_E${EL}-*_C${CL}-*.json; do [ -e "$f" ] && cp -pn "$f" "$LOG/selfcheck_before_q/"; done
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
    t=${pre}27${CL}q${s}${sign}
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

note "chain start (pid $$): 27B E$EL + C$CL, rules v1.1, KS [$KSQ], cards $G"
( cd "$HERE" && sha256sum ff_explore_ec.sh ff_explore_ec_check.py ff_explore_ec_describe.py ff_cache_pick.py ff_explore_20261114q_chain.sh ) \
  >> "$LOG/chain_jobs.log" 2>&1 || true
upstream_over "$PLOG" t27_ec2                 # chain p's test generation of the same pair is over (its selfcheck record)
block tune ec
upstream_over "$PLOG" ec27${CL}_p             # the v1 tune E+C Pro pass is over: its reply cache is complete
for p in $TAGS_tune; do
  IFS=: read -r t s sign <<< "$p"
  pro_lane tune "$t" "$s" "$sign" "$RUN/eval_extra/judge_tune_ec27${CL}_tune/judge2/cache" \
    "$RUN/judge_tune_proS_C_L$CL/judge2/cache" "$RUN/judge_tune_proX_E_L$EL/judge2/cache" &
done
block test t
upstream_over "$PLOG" t27_ecp                 # the v1 test E+C Pro pass is over: its reply cache is complete
for p in $TAGS_test; do
  IFS=: read -r t s sign <<< "$p"
  pro_lane test "$t" "$s" "$sign" "$RUN/eval_extra/judge_test_t27${CL}_test/judge2/cache" \
    "$RUN/judge_test_testS_C_L$CL/judge2/cache" &
done
wait
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"
