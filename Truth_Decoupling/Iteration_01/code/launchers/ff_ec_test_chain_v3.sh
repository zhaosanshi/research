#!/usr/bin/env bash
# RULES v3 (Claude Fable 5.2, 2026-11-15 23:xx): this is ff_ec_test_chain.sh (v1, sha256 340317c0, red-teamed 11-15) with
# two rule changes; everything else, text and mechanics, is v1's and is described below as v1 wrote it.
#   R5 v3  the C strength KC = the smallest of 0.25, 0.5, 1 at which at least one shortlisted C layer's toward-caution arm
#          x.w.p<KC> meets criterion v4, column A, under AllenAI; R5 = those layers. 0.25 (and 0.5 only when nothing passes at
#          0.25) from the lines chain's C tables; 1 (only when nothing passes at 0.5; the frozen strength of criterion v4)
#          from the lines chain's round-1 work dir judge_tune_round1_C, as tables c1_tables_L<L> remade on every run. Where v1
#          found R5 layers, v3 reads the same files and gives the same layers. The C test arms use KC (ff_test_eval.sh CK).
#   R6 v3  the E+C E layer is ALWAYS R3. C = the lowest R5 layer above R3 (v1); when no R5 layer lies above R3, C = the
#          deepest R5 layer below R3 (C below E; v1: no E+C); never R3 itself (two writes need two layers), so R5 = {R3}
#          alone means no E+C. C is still added only as v1 / v1.1 say (v1.1: only when E alone moves the abstentions
#          beyond the tolerance).
# Why (赵磊 11-15): v1 found no R5 layer on Gemma-4-12B (C needs strength 1 there; no R5 layer above E24). Rules v2 of
#   21:1x ("选A，改规则补跑E+C") kept "C deeper than E" by moving the E+C E layer below C (E18), which gave a null E+C;
#   the exploration of 22:1x-22:3x (logs/explore_e1) showed the cause: E18 is a weak layer, while E alone at R3 = E24 on the
#   plain prompt gives +10 / +28 right answers at strengths 1 / 2 with the answer count unchanged (tune, AllenAI), as the
#   27B. v2's E-layer move is withdrawn; ff_ec_test_chain_v2.sh is superseded by this file.
# Logs in <FF_ROOT on the host>/logs/chain_ec_v3.
# ---- v1's header follows ----
# Iteration_01 (Claude Opus 5.5, 2026-11-15; red-team fixes of 11-15 15:xx applied): E+C and the 529-question test for ONE
# model, AllenAI only, after the layer choices of ff_ec_lines_chain.sh. 赵磊 11-15: "用gemma 7-9B 和20几B的模型，复现TF和自由问题，
# 以及加unknow的自由问题的E+C比单独的E和C 评测时，先不用deepseek Pro了，先用allenAI看看是否成立。" The AllenAI-only equivalent of what
# the Qwen3.8-27B got after its layer choices (chains p, q, s, test_sel27 of 11-14 in code/scripts/review_checks). No DeepSeek
# Pro step, no wait for one.
# Rules of from_claude/45 -> this chain [job names]; "45: both judges" means AllenAI alone here:
#   R1     E2 sentence: E_ON=E2 C_E_ON=E2 (the lines chain trained both branches with E2, as the 27B)
#   R3     E layer = first layer of the lines chain's r3_top1.md (AllenAI top 1; 45: largest sum of both judges)
#   R5     C layers = the shortlisted C layers whose toward-caution arm x.w.p0.25 meets criterion v4, column A, under AllenAI
#          (45: both judges), read from the KEY lines of the lines chain's C tables at 0.25 with ff_rule_c_pick.py's pattern
#          and accepted values (that script itself needs a Pro KEY line) [r5_r6]
#   R6     E+C C layer = the lowest R5 layer above R3 [r5_r6]; none -> no E+C and no equal-N comparison
#   R6 v1  plain prompt x, E at R3, strengths 1 2, C toward confident at k = 0.25 0.5 1 2, k per E strength by rule
#          abstentions only (ff_explore_ec_check.py, before judging); E alone, C alone, E->random, C->random; every grid point
#          judged (chain p) [tune ec<R6>_*, test t<R6>_*]
#   v1.1   45 section 3.4 (chain q), per split and E strength s: the C sign from E s alone's abstentions (rule or empty)
#          against baseline x, tolerance 3% of the questions: within -> no C (E alone is in the v1 work dir); more -> m; fewer
#          -> p. First the v1 grid with empty answers counted [<pre><R6>v<s><sign>_g1]; if its closest point misses the
#          tolerance, the grid is refined where the count crosses baseline x [<...>d_*]: below the smallest point the 27B's
#          0.03 0.06 0.1 0.15 (scaled to it), between two points four evenly spaced ones (Llama's 0.6-0.9 between 0.5 and 1);
#          only the chosen k is judged (EXPORT=chosen). The GRIDS match chains q / r, the PROCEDURE does not: q / r set one
#          grid by hand after seeing v1; here each cell is refined by program, and only when its own closest point misses the
#          tolerance, never looking at correctness. This procedure NEEDS 赵磊's APPROVAL (NOTES D6).
#   R2 R4  test: freeze the 11 prompts; E at R3 on the must-answer prompt, strengths 1 2: w, md, rand both signs, prompt
#          control conf_E2; cells tables (conf base) and the dose table [t_freeze t_egen t_ea t_et]
#   R5     test: C at every R5 layer, plain prompt, +-0.25: w, md, rand both signs; cells tables at 0.25 [t_cgen t_ca_L t_ct_L]
#   R6     equal N (from_claude/39; chains s, test_sel27), on the v1 E+C of each split: E+C against "E alone on the
#          must-answer prompt, keep the N most confident answers" and the other selections [sn_tune_* sn_test_*]
#   R7     AllenAI only; the tables print column A (tune reading) and column B (test reading)
# Where a runner stage hard-codes both judges (ff_test_eval.sh etables / ctables, ff_explore_ec.sh describe,
# ff_explore_ec_select.sh judgeconf / compare), the same read-only table script runs directly with --judge allenai; the two
# scripts whose judge loop is a literal ("pro", "allenai") run through AO below. Details: ff_ec_test_chain_NOTES.md.
# Order on the one GPU set: wait for the end of the lines chain -> R5 / R6 -> tune (E+C v1, v1.1, equal N) -> test (E, C,
# E+C v1, v1.1, equal N). Mechanics as the lines chain: clean environment per job, DONE skips, a FAIL marks its dependents
# "not tried", lock directory, logs and tables in <FF_ROOT on the host>/logs/chain_ec_test. The lines chain's files are only read.
# Safety (red team 11-15):
#   * start only when the lines chain's LAST "chain start" is followed by its "chain end:" and its lock is gone (its EXIT trap
#     frees the lock on TERM while its job runs on); its early exits ("C pilot failed / step 0 failed: nothing else tried")
#     end this chain with a note and no job; its params.txt must name this MODEL and FF_ROOT
#   * the code-base checks run at the start and again after the wait; the hash of the code, the model config and the
#     prompts taken after the wait must hold before every job; before every GPU job the lines chain's lock must be absent
#     and nvidia-smi must show no compute process on the cards of GPU (this chain's previous job has ended, so anything
#     there is not ours). Any of these: the job is marked FAIL and the chain stops (rerun it once the cause is gone)
#   * LOG/params.txt holds model, GPU, root and code (first start) and R3 / R5 / R6 (first time known); a rerun that
#     differs is refused, so DONE flags of other layers are never reused
#   * LOG/<job>.pid ("runner-PID runner") exists while a job runs; a start that finds a live one refuses to run
# To stop it completely: (1) kill the chain (PID in chain_jobs.log, "chain start (pid N)"); its lock goes, its job does
#   not. (2) kill the runner PID in LOG/<job>.pid, so it starts no further step. (3) kill the job's python in the
#   container by its host PID: nvidia-smi --query-compute-apps=pid,process_name on the chain's cards, or ps -ef | grep
#   'scripts/' on the host; its docker exec exits with it. Then remove LOG/<job>.pid; a rerun skips DONE jobs.
#   usage (host): MODEL=gemma2-9b GPU=1 FF_ROOT=/workspace/Truth_Decoupling/Iteration_01/results/ff/gemma2/9b bash ff_ec_test_chain.sh
#   (CODE_PATH defaults to /workspace/Truth_Decoupling/Iteration_01/code). Dry runs only, all four or none: CHAIN_TE_RUNNER,
#   CHAIN_EC_RUNNER, CHAIN_SEL_RUNNER (stub runners) and CHAIN_TX_DRY=1 (print the table commands, as the lines chain); with
#   them CHAIN_NVSMI (a stub of nvidia-smi) and CHAIN_POLL (seconds).
set -uo pipefail
: "${MODEL:?}" "${GPU:?}" "${FF_ROOT:?}"
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
CODE=${CODE_PATH:-$W/Iteration_01/code}
for p in "$FF_ROOT" "$CODE"; do
  case "$p" in "$W"/*) ;; *) echo "FF_ROOT and CODE_PATH must be absolute paths under $W" >&2; exit 2 ;; esac
  case "$p" in *results_newmodels_ff_20261112*|*code_ff_explore_20261113b*) echo "$p is an old root" >&2; exit 2 ;; esac
done
case "$GPU" in ""|*[!0-9,]*) echo "GPU must be card numbers such as 1 or 2,3" >&2; exit 2 ;; esac
DRYSET=""; for v in CHAIN_TE_RUNNER CHAIN_EC_RUNNER CHAIN_SEL_RUNNER CHAIN_TX_DRY CHAIN_NVSMI; do [ -n "${!v:-}" ] && DRYSET="$DRYSET $v"; done
if [ -n "$DRYSET" ]; then
  for v in CHAIN_TE_RUNNER CHAIN_EC_RUNNER CHAIN_SEL_RUNNER CHAIN_TX_DRY; do
    [ -n "${!v:-}" ] || { echo "dry-run switch(es)$DRYSET without $v: a dry run sets all of CHAIN_TE_RUNNER CHAIN_EC_RUNNER CHAIN_SEL_RUNNER CHAIN_TX_DRY" >&2; exit 2; }
  done
fi
CODEH=$R/${CODE#"$W"/}
HERE=$CODEH/scripts/review_checks
TE=${CHAIN_TE_RUNNER:-$HERE/ff_test_eval.sh}
EC=${CHAIN_EC_RUNNER:-$HERE/ff_explore_ec.sh}
SEL=${CHAIN_SEL_RUNNER:-$HERE/ff_explore_ec_select.sh}
NVSMI=${CHAIN_NVSMI:-nvidia-smi}
SCRIPTS="ff_test_eval.sh ff_explore_ec.sh ff_explore_ec_select.sh e44_split_test.py ff_explore_ec_check.py ff_explore_ec_describe.py
  ff_explore_ec_confidence.py ff_test_confidence.py ff_explore_ec_compare.py ff_explore_dose_table.py e44_round1_tables.py ff_rule_c_pick.py"
HASHED="core/*.py scripts/common.py scripts/e43_qa_prompt_das.py scripts/e44_factorial_das.py scripts/e44_eval_extra.py
  scripts/e44_export_judge.py scripts/allenai_tqa_judge.py scripts/review_checks/judge_two_axis_run.py configs/models/$MODEL.json
  configs/prompts/default_v1.json configs/prompts/e44_factorial_v1.json configs/prompts/e44_eval_extra_v1.json"
# AO: a table script whose judge loop is ("pro", "allenai"), run with AllenAI alone. Exact-count source replacements in
# memory, as e44_split_test.py does; the file stays unchanged; any count off stops before the script runs. "--check DIR"
# only verifies the counts and compiles the patched source (nothing is executed).
AO=$(cat <<'PY'
import sys
EDITS = {"ff_explore_ec_compare.py": [('for judge in ("pro", "allenai")', 'for judge in ("allenai",)', 2),
                                      ("if len(per) == 2:", "if len(per) == 1:", 1),
                                      ("(both judges, intervals", "(AllenAI only, intervals", 2),
                                      ("both point estimates", "the AllenAI point estimate", 2),
                                      ("both judges together", "AllenAI only (DeepSeek Pro not run)", 1)],
         "ff_explore_dose_table.py": [('for judge in ("pro", "allenai"):', 'for judge in ("allenai",):', 1)]}
def patched(path):
    src = open(path, encoding="utf-8").read()
    for old, new, n in EDITS[path.rsplit("/", 1)[-1]]:
        if src.count(old) != n:
            sys.exit("%s: %r occurs %d times, expected %d: not run" % (path, old, src.count(old), n))
        src = src.replace(old, new)
    return src
if sys.argv[1] == "--check":
    for name in EDITS:
        compile(patched(sys.argv[2] + "/" + name), name, "exec")
    sys.exit(0)
script = sys.argv[1]
src = patched(script)
print("AllenAI only: %s with its judge loop cut to AllenAI in memory (file unchanged; DeepSeek Pro not run)\n" % script)
sys.argv = [script] + sys.argv[2:]
exec(compile(src, script, "exec"), {"__name__": "__main__", "__file__": script})
PY
)
guards() {  # the code-base checks; at the start and again after the wait (red team 11-15: the wait can last a day)
  local f
  [ -f "$CODEH/configs/models/$MODEL.json" ] || { echo "no model config $MODEL" >&2; return 2; }
  for f in $SCRIPTS; do [ -f "$HERE/$f" ] || { echo "the code base has no scripts/review_checks/$f" >&2; return 3; }; done
  [ -n "${CHAIN_EC_RUNNER:-}" ] || { grep -q 'SPLIT=${SPLIT:-tune}' "$EC" && grep -q 'ABSTAIN_EMPTY=${ABSTAIN_EMPTY:-0}' "$EC"; } \
    || { echo "$EC is not the SPLIT / v1.1 E+C runner" >&2; return 3; }
  [ -n "${CHAIN_TE_RUNNER:-}" ] || grep -q 'CK=${CK:-0.25}' "$TE" || { echo "$TE has no CK parameter (rules v3)" >&2; return 3; }
  python3 -B -c "$AO" --check "$HERE" || return 3
}
code_sum() {  # one hash over the code, the model config and the prompts the jobs read (as tf_lane.sh)
  ( cd "$CODEH" && sha256sum $HASHED $(printf 'scripts/review_checks/%s ' $SCRIPTS) ) | sha256sum | cut -c1-16
}
guards || exit $?
SUM_START=$(code_sum) && [ -n "$SUM_START" ] || { echo "cannot hash the code base" >&2; exit 3; }
HFF=$R/${FF_ROOT#"$W"/}
ULOG=$HFF/logs/chain                                                # the lines chain: read only
LOG=$HFF/logs/chain_ec_v3
CLOG=$FF_ROOT/logs/chain_ec_v3                                      # the same directory, container path
POLL=${CHAIN_POLL:-60}
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
P1="model=$MODEL gpu=$GPU root=$FF_ROOT code=$CODE"                 # line 2 ("layers: ...") is added once R3 / R5 / R6 are known
if [ -f "$LOG/params.txt" ]; then
  [ "$(head -n 1 "$LOG/params.txt")" = "$P1" ] || { echo "this log dir was started with other parameters: $(head -n 1 "$LOG/params.txt")" >&2; exit 2; }
else
  echo "$P1" > "$LOG/params.txt"
fi
CLEAN=(-u GPU -u MODEL -u STAGE -u CODE_PATH -u TABLE_CODE -u SOURCE_DIR -u E_RUN -u C_RUN -u E_ON -u C_E_ON -u EL -u CL -u CLS
       -u ECLS -u ES -u KS -u TAG -u LAYER -u LAYERS -u JUDGES -u SPLIT -u CACHE_FROM -u FF_ROOT -u CSIGN -u EXPORT -u ABSTAIN_EMPTY
       -u SCORE_BATCH -u CONF_WORK -u BRANCH -u VARIANT -u RUN_DIR -u STRENGTH -u ARMS -u BASES -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CK)
stop() {  # stop NAME REASON: NAME is marked FAIL and the chain ends here
  echo "not started: $2" > "$LOG/$1.FAIL"; note "STOP  before $1: $2"; exit 3
}
gpu_procs() {  # the compute processes on the cards of GPU ("card N pid P name"); non-zero when nvidia-smi cannot be read
  local map
  map=$("$NVSMI" --query-gpu=index,uuid --format=csv,noheader | tr '\n' ';') || return 2
  "$NVSMI" --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | awk -F', *' -v cards=",$GPU," -v map="$map" '
    BEGIN { n = split(map, m, ";"); for (i = 1; i <= n; i++) if (split(m[i], f, ", *") == 2) card[f[2]] = f[1] }
    ($1 in card) && index(cards, "," card[$1] ",") { print "card " card[$1] " pid " $2 " " $3 }'
}
preflight() {  # preflight NAME GPU|CPU: the checks before a job; a failed one stops the chain
  local busy
  [ "$(code_sum)" = "$SUM0" ] || stop "$1" "the code, the model config or the prompts changed after the wait (hash $SUM0)"
  [ "$2" = GPU ] || return 0
  [ ! -d "$ULOG/.lock_chain" ] || stop "$1" "the lines chain holds its lock again ($ULOG/.lock_chain)"
  if busy=$(gpu_procs 2> /dev/null); then
    if [ -n "$busy" ]; then sleep 10; busy=$(gpu_procs 2> /dev/null) || busy="nvidia-smi unreadable on the recheck; first: $busy"; fi
    [ -z "$busy" ] || stop "$1" "GPU $GPU is in use by a process that is not this chain's: $(echo $busy)"
  else
    note "nvidia-smi could not be read before $1: the cards were not checked"
  fi
}
job() {  # job NAME RUNNER VAR=value ...: one GPU job in a clean environment; skipped when NAME.DONE exists
  local name=$1 runner=$2 pid rc; shift 2
  [ -e "$LOG/$name.DONE" ] && return 0
  preflight "$name" GPU
  rm -f "$LOG/$name.FAIL"; note "start $name"
  [ -e "$LOG/$name.log" ] && mv -f "$LOG/$name.log" "$LOG/$name.log.prev"   # keep the log of a failed earlier try
  env "${CLEAN[@]}" "$@" bash "$runner" > "$LOG/$name.log" 2>&1 < /dev/null &
  pid=$!; echo "$pid $runner" > "$LOG/$name.pid"
  wait "$pid"; rc=$?
  rm -f "$LOG/$name.pid"
  if [ $rc = 0 ]; then
    touch "$LOG/$name.DONE"; note "done  $name"
  else
    echo "failed, see $name.log" > "$LOG/$name.FAIL"; note "FAIL  $name"; return 1
  fi
}
tjob() {  # tjob NAME COMMAND...: a read-only table command with the same flags; stdout to NAME.md
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  preflight "$name" CPU
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if "$@" > "$LOG/$name.md" 2> "$LOG/$name.log" < /dev/null; then
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
up() { [ -e "$ULOG/$1.DONE" ]; }                                    # a DONE flag of the lines chain
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec python3)
tx() { if [ -n "${CHAIN_TX_DRY:-}" ]; then echo "tx $*"; else "${TX[@]}" "$@"; fi; }   # read-only tables in the container
ao() { if [ -n "${CHAIN_TX_DRY:-}" ]; then echo "tx -c <AO> $*"; else "${TX[@]}" -c "$AO" "$@"; fi; }
pre_of() { [ "$1" = tune ] && echo ec || echo t; }                  # TAG prefix of a split, as chain p's ec27 / t27
log_sums() { ( cd "$HERE" && sha256sum $SCRIPTS ) >> "$LOG/chain_jobs.log" 2>&1; sha256sum "$0" >> "$LOG/chain_jobs.log" 2>&1; note "code hash $1"; }

note "chain start (pid $$): model $MODEL, GPU $GPU, root $FF_ROOT, code $CODE (rules v3)"
for f in "$LOG"/*.pid; do                                           # a job of an earlier run whose chain was killed
  [ -e "$f" ] || continue
  read -r pid runner < "$f"
  if [ -n "$pid" ] && { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2> /dev/null | grep -qF -- "$runner"; then
    note "${f##*/}: a job of an earlier run still runs (runner PID $pid): stop"
    echo "${f##*/}: a job of an earlier run still runs (runner PID $pid): stop it first (header: to stop it completely)" >&2; exit 1
  fi
  rm -f "$f"; note "removed ${f##*/}: its runner (PID ${pid:-?}) has ended"
done
rm -f "$LOG"/*.FAIL
log_sums "$SUM_START"

# ---- the lines chain: wait for the END of its last run; a free lock alone proves nothing (red team 11-15)
lines_state() {  # running | ended | stopped (an early exit), nothing when no run is logged: the lines chain's last run
  awk '/ chain start \(pid /{s = 1; st = "running"; next}
       s && / chain end: /{st = "ended"} s && /(C pilot|step 0) failed: nothing else tried/{st = "stopped"}
       END{if (s) print st}' "$ULOG/chain_jobs.log" 2> /dev/null
}
note "waiting for the lines chain ($ULOG): its last \"chain start\" must be followed by its \"chain end:\" and its lock gone"
while :; do
  st=$(lines_state)
  if [ ! -d "$ULOG/.lock_chain" ]; then
    [ "$st" = ended ] && break
    [ "$st" = stopped ] && { note "the lines chain stopped early (C pilot or step 0 failed): nothing to do"; exit 0; }
  fi
  sleep "$POLL"
done
case "$(cat "$ULOG/params.txt" 2> /dev/null)" in
  "model=$MODEL root=$FF_ROOT "*) ;;
  *) note "the lines chain's params.txt does not name model $MODEL and root $FF_ROOT: stop"; exit 2 ;;
esac
guards >> "$LOG/chain_jobs.log" 2>&1 || { note "the code-base checks failed after the wait: stop"; exit 3; }
SUM0=$(code_sum) && [ -n "$SUM0" ] || { note "cannot hash the code base after the wait: stop"; exit 3; }
[ "$SUM0" = "$SUM_START" ] || note "the code base changed while this chain waited; its checks passed again"
log_sums "$SUM0 (after the wait; every job checks it)"
up r3_top1 || { note "the lines chain has no R3 (r3_top1 not DONE): nothing to do"; exit 0; }
R3=$(sed -n 's/^top: //p' "$ULOG/r3_top1.md" | awk '{print $1}')
case "$R3" in ""|*[!0-9]*) note "cannot read the R3 layer from $ULOG/r3_top1.md: stop"; exit 3 ;; esac
PREV=$(sed -n 's/^layers: //p' "$LOG/params.txt")
case "$PREV" in ""|"r3=$R3 "*) ;; *) note "R3 is E$R3 now, the first run had [$PREV]: refuse (DONE flags of another layer)"; exit 2 ;; esac

RUN=$FF_ROOT/e44/$MODEL; HRUN=$HFF/e44/$MODEL; SRC=$FF_ROOT/source/$MODEL

# ---- R5 / R6 v3 with AllenAI alone, from the lines chain's tables on every run; r5_r6.FAIL = undetermined (the E half of
# the test still runs; C and E+C need a rerun); a result that differs from the first run's is refused
pick_k() {  # pick_k STRENGTH LAYER:LOG ...: the table rows of that strength and "r5 at STRENGTH: <layers>|none"
  python3 -B - "$HERE" "$@" <<'PY'
import re, sys
sys.path.insert(0, sys.argv[1])
from ff_rule_c_pick import KEY         # the KEY pattern of the 27B's rule (v1 imports it too)
k, status = sys.argv[2], {}
for spec in sys.argv[3:]:
    L, log = spec.split(":", 1)
    L = int(L)
    keys = [m for m in map(KEY.match, open(log, encoding="utf-8", errors="replace").read().splitlines()) if m]
    if [m.group(1) for m in keys] != ["allenai"] or int(keys[0].group(2)) != L:
        sys.exit("%s: exactly one AllenAI KEY line of layer %d is required" % (log, L))
    line = keys[0].string
    if (k == "1" and "EXPLORATORY" in line) or (k != "1" and not line.rstrip().endswith("EXPLORATORY strength=%s" % k)):
        sys.exit("%s: the KEY line is not one of strength %s" % (log, k))
    A, B = keys[0].group(3), keys[0].group(4)
    if not (A in ("TWO-WAY", "not_met") or re.fullmatch(r"ONE-WAY\(x\.w\.[pm]%s\)" % re.escape(k), A)):   # "incomplete" never reads as "no"
        sys.exit("%s: column A = %s, outside TWO-WAY / ONE-WAY(arm at %s) / not_met" % (log, A, k))
    if L in status:
        sys.exit("layer %d: two tables at strength %s" % (L, k))
    status[L] = (A, B)
r5 = sorted(L for L in status if status[L][0] in ("TWO-WAY", "ONE-WAY(x.w.p%s)" % k))
print("| C layer | strength | AllenAI A / B | R5 (x.w.p%s meets criterion v4, column A, under AllenAI alone) |\n|---|---|---|---|" % k)
for L in sorted(status):
    print("| %d | %s | %s / %s | %s |" % (L, k, status[L][0], status[L][1], "yes" if L in r5 else "no"))
print("\nr5 at %s: %s\n" % (k, " ".join(map(str, r5)) or "none"))
PY
}
record() {  # record: r5_r6.json from this run's values, written only when absent; an existing different file is refused (exit 3)
  python3 -B - "$LOG/r5_r6.json" "$R3" "$KC" "$R5" "$R6" "$MODE" <<'PY'
import json, sys
out, r3, kc, r5, r6, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4].split(), sys.argv[5], sys.argv[6]
text = json.dumps({"rules": "v3 (2026-11-15, AllenAI alone): R5 = the shortlisted C layers whose x.w.p<kc> meets criterion v4, column A, "
                            "kc the smallest of 0.25 / 0.5 / 1 at which one does; the E+C E layer is R3; R6 = the lowest R5 layer "
                            "above R3, else the deepest R5 layer below R3 (never R3 itself)",
                   "r3": r3, "kc": kc, "r5": r5, "r6": r6, "mode": mode}, indent=1, ensure_ascii=False) + "\n"
try:
    old = open(out, encoding="utf-8").read()
except FileNotFoundError:
    old = None
if old is None:
    open(out, "w", encoding="utf-8").write(text)
elif old != text:
    sys.stderr.write("%s exists and differs: not overwritten\n" % out)
    sys.exit(3)
PY
}
why="" KC=none R5="" R6=none MODE=none
rm -f "$LOG/r5_r6.DONE" "$LOG/r5_r6.md.new"; : > "$LOG/r5_r6.log"
if ! up c_screen; then
  why="the lines chain has no C screen"
elif ! grep -q '^shortlist layers:' "$ULOG/c_screen.log"; then
  why="the lines chain's c_screen.log has no \"shortlist layers:\" line"
else
  SHORT=$(sed -n 's/^shortlist layers: *//p' "$ULOG/c_screen.log")
  for k in 0.25 0.5 1; do                                           # the smallest strength at which a layer passes
    t=()
    for L in $SHORT; do
      if [ "$k" = 1 ]; then
        up c_judge || { why="the lines chain has no round-1 C judging (c_judge), which the tables at strength 1 read"; break 2; }
        if [ -n "${CHAIN_TX_DRY:-}" ] && [ -e "$LOG/c1_tables_L$L.dry" ]; then                 # dry runs only: tx prints, it makes no table
          cp "$LOG/c1_tables_L$L.dry" "$LOG/c1_tables_L$L.md"
        else
          rm -f "$LOG/c1_tables_L$L.DONE"                           # remade on every run: a stale table is never read
          tjob c1_tables_L$L tx scripts/review_checks/e44_round1_tables.py cells "$RUN" C "$RUN/judge_tune_round1_C" --judge allenai \
            --layers "$L" --strength 1 || { why="the C table at strength 1 of layer $L failed (c1_tables_L$L.log)"; break 2; }
        fi
        t+=("$L:$LOG/c1_tables_L$L.md")
      else
        up c_tables_L${L}_s$k || { why="the lines chain has no C tables at $k for layer $L"; break 2; }
        t+=("$L:$ULOG/c_tables_L${L}_s$k.log")
      fi
    done
    out=$(pick_k "$k" ${t[@]+"${t[@]}"} 2>> "$LOG/r5_r6.log") || { why="the R5 pick at strength $k failed, see r5_r6.log"; break; }
    printf '%s\n' "$out" >> "$LOG/r5_r6.md.new"
    r=$(printf '%s\n' "$out" | sed -n "s/^r5 at $k: //p")
    [ -n "$r" ] || { why="the R5 pick at strength $k printed no result"; break; }
    if [ "$r" != none ]; then KC=$k; R5=$r; break; fi
  done
fi
if [ -z "$why" ]; then
  if [ -n "$R5" ]; then
    R6=$(for L in $R5; do [ "$L" -gt "$R3" ] && echo "$L"; done | sort -n | head -n 1); MODE=v1
    [ -n "$R6" ] || { R6=$(for L in $R5; do [ "$L" -lt "$R3" ] && echo "$L"; done | sort -n | tail -n 1); MODE=deepest; }
    [ -n "$R6" ] || { R6=none; MODE=none; }                         # R5 holds only the E layer's own number: no E+C (red team 23:45)
  fi
  printf 'R6 v3: the lowest R5 layer above the R3 E layer E%s; none above -> the deepest R5 layer below it (the E+C E layer stays E%s); R5 = that layer alone -> no E+C\n\nkc: %s\nr5: %s\nr6: %s\nmode: %s\n' \
    "$R3" "$R3" "$KC" "${R5:-none}" "$R6" "$MODE" >> "$LOG/r5_r6.md.new"
  record 2>> "$LOG/r5_r6.log"; rc=$?
  case $rc in
    0) mv -f "$LOG/r5_r6.md.new" "$LOG/r5_r6.md"; touch "$LOG/r5_r6.DONE" ;;
    3) note "R5 / R6 from the lines chain's tables now (r5_r6.md.new) differ from the first run's r5_r6.json: refuse"; exit 2 ;;
    *) why="writing r5_r6.json failed, see r5_r6.log" ;;
  esac
fi
if [ -e "$LOG/r5_r6.DONE" ]; then
  NOW="r3=$R3 kc=$KC r5=${R5:-none} r6=$R6"
else
  echo "$why" > "$LOG/r5_r6.FAIL"; KC=none; R5=""; R6=none; NOW="r3=$R3 kc=undetermined r5=undetermined r6=undetermined"
  note "R5 / R6 undetermined: $why (the E half of the test goes on; C and E+C need a rerun)"
fi
case "$PREV" in
  "") echo "layers: $NOW" >> "$LOG/params.txt" ;;
  "$NOW") ;;
  "r3=$R3 kc=undetermined r5=undetermined r6=undetermined") sed -i "s/^layers: .*/layers: $NOW/" "$LOG/params.txt"; note "R5 / R6 are determined now: $NOW" ;;
  *) note "the layers are [$NOW] now, the first run had [$PREV]: refuse (DONE flags of other layers are never reused)"; exit 2 ;;
esac
note "R3 E$R3 (lines chain, AllenAI top 1); R5 v3 [${R5:-none}] at strength $KC; R6 v3 C$R6 ($MODE; the E+C E layer is E$R3)"

KS0="0.25 0.5 1 2"
T0=(MODEL=$MODEL CODE_PATH=$CODE FF_ROOT=$FF_ROOT SOURCE_DIR=$SRC E_RUN=$RUN C_RUN=$RUN E_ON=E2 C_E_ON=E2 EL=$R3 "ES=1 2" "KS=$KS0" GPU=$GPU)
X=("${T0[@]}" CL=$R6)

# ---- E+C helpers (used only when R6 is a layer)
FRESH=" "
gen() {  # gen SPLIT NAME VAR=value ...: an E+C generation job. Before the first one of SPLIT that still has to run in this
  # chain run, a fresh selfcheck of SPLIT: the engine keeps one record per direction pair whatever the split, so the other
  # split's selfcheck replaces it (chains p / q); each record made is copied to selfcheck_<SPLIT>_backup/
  local sp=$1 name=$2 f; shift 2
  [ -e "$LOG/$name.DONE" ] && return 0
  case "$FRESH" in *" $sp "*) ;; *)
    rm -f "$LOG/sc_$sp.DONE"
    job sc_$sp "$EC" "${X[@]}" SPLIT=$sp TAG=$(pre_of $sp)$R6 STAGE=selfcheck || { give_up "$name"; return 1; }
    mkdir -p "$LOG/selfcheck_${sp}_backup"
    for f in "$HRUN"/eval_extra/selfcheck_E$R3-*_C$R6-*.json "$HRUN"/eval_extra/selfcheck_C$R6-*_E$R3-*.json; do
      [ -e "$f" ] && cp -p "$f" "$LOG/selfcheck_${sp}_backup/"; done
    [ -n "$(ls -A "$LOG/selfcheck_${sp}_backup" 2> /dev/null)" ] || note "WARNING: no selfcheck record of E$R3 / C$R6 found to back up after sc_$sp"
    FRESH="$FRESH$sp " ;;
  esac
  job "$name" "$EC" "$@"
}
describe() {  # describe NAME SPLIT TAG: ff_explore_ec_describe.py, AllenAI (the runner's describe stage runs both judges)
  tjob "$1" tx scripts/review_checks/ff_explore_ec_describe.py "$RUN/eval_extra/judge_$2_$3_$2" --judge allenai \
    --el "$R3" --cl "$R6" --choice "$RUN/eval_extra/ec_choice_$3.json"
}
ec_v1() {  # ec_v1 SPLIT: v1 E+C (chain p): gen1 (with the choice of k), gen2, AllenAI, description
  local sp=$1 t; t=$(pre_of $1)$R6
  { gen $sp ${t}_g1 "${X[@]}" SPLIT=$sp TAG=$t STAGE=gen1 &&
    gen $sp ${t}_g2 "${X[@]}" SPLIT=$sp TAG=$t STAGE=gen2 &&
    job ${t}_a "$EC" "${X[@]}" SPLIT=$sp TAG=$t STAGE=judge JUDGES=allenai &&
    describe ${t}_d $sp $t; } || give_up ${t}_g2 ${t}_a ${t}_d
}
sign_of() {  # sign_of SPLIT S -> "m|p|none N BASE TAU": E S alone's abstentions (rule or empty) against baseline x (chain q)
  python3 -B - "$HRUN" "$1" "$R3" "$2" <<'PY'
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
PY
}
dense_of() {  # dense_of CHOICE S SIGN N_E TAU -> the refined C strengths of v1.1 rule 1, or nothing when none apply
  python3 -B - "$@" <<'PY'
import json, sys
f, s, sign, ne, tau = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
c = json.load(open(f, encoding="utf-8"))
if c.get("c_sign", "m") != sign or c.get("empty_counts_as_abstention") is not True:
    sys.exit("%s is not a v1.1 choice file of sign %s" % (f, sign))
base, cand, k0 = c["baseline_rule_abstentions"], c["candidates"][s], c["choice"][s]
if abs(cand[k0] - base) > tau:          # the closest eligible point misses the tolerance: refine where the count crosses x
    pts = [(0.0, ne)] + sorted((float(k), v) for k, v in cand.items())      # k = 0: E alone
    for (a, na), (b, nb) in zip(pts, pts[1:]):
        if (na - base) * (nb - base) < 0:
            new = [b * r for r in (0.12, 0.24, 0.4, 0.6)] if a == 0 else [a + (b - a) * i / 5 for i in (1, 2, 3, 4)]
            print(" ".join("%g" % round(v, 4) for v in new))
            break
PY
}
v11() {  # v11 SPLIT S: rules v1.1 (45 section 3.4, chain q) for E strength S on SPLIT; needs the v1 gen1 of SPLIT
  local sp=$1 s=$2 sign n base tau t new ks x
  read -r sign n base tau <<< "$(sign_of "$sp" "$s" 2>> "$LOG/v11.log")"
  case "$sign" in m|p|none) ;; *) note "v1.1 $sp E $s: E $s alone cannot be read (v11.log): not tried"; return 0 ;; esac
  note "v1.1 $sp: E $s alone has $n abstentions (rule or empty), baseline x $base, tolerance $tau -> C sign $sign"
  [ "$sign" = none ] && return 0
  t=$(pre_of $sp)${R6}v$s$sign
  x=("${X[@]}" SPLIT=$sp "ES=$s" CSIGN=$sign ABSTAIN_EMPTY=1)
  gen $sp ${t}_g1 "${x[@]}" TAG=$t STAGE=gen1 || { give_up ${t}_g2 ${t}_a ${t}_d; return 0; }
  new=$(dense_of "$HRUN/eval_extra/ec_choice_$t.json" "$s" "$sign" "$n" "$tau" 2>> "$LOG/v11.log") \
    || { note "v1.1 $sp E $s: cannot read the choice file of $t (v11.log)"; give_up ${t}_g2 ${t}_a ${t}_d; return 0; }
  ks=$KS0
  if [ -n "$new" ]; then
    ks=$(printf '%s\n' $KS0 $new | sort -g | tr '\n' ' '); ks=${ks% }; t=${t}d
    note "v1.1 $sp E $s: the closest point of [$KS0] misses the tolerance -> grid [$ks], TAG $t"
    gen $sp ${t}_g1 "${x[@]}" "KS=$ks" TAG=$t STAGE=gen1 || { give_up ${t}_g2 ${t}_a ${t}_d; return 0; }
  fi
  { gen $sp ${t}_g2 "${x[@]}" "KS=$ks" TAG=$t STAGE=gen2 &&
    job ${t}_a "$EC" "${x[@]}" "KS=$ks" TAG=$t STAGE=judge JUDGES=allenai EXPORT=chosen &&
    describe ${t}_d $sp $t; } || give_up ${t}_g2 ${t}_a ${t}_d
}
same_n() {  # same_n SPLIT CONF_WORK: equal-N comparison on the v1 E+C of SPLIT (chains s / test_sel27), AllenAI alone
  local sp=$1 cw=$2 t n=sn_$1 v; t=$(pre_of $1)$R6
  v=("${X[@]}" SPLIT=$sp TAG=$t CONF_WORK=$cw)                      # CONF_WORK is read on test only
  { job ${n}_gen "$SEL" "${v[@]}" STAGE=genconf &&
    { [ -e "$LOG/${n}_conf_b1.DONE" ] || job ${n}_conf "$SEL" "${v[@]}" STAGE=confidence ||
      { grep -q "teacher-forced arg-max reproduces only" "$LOG/${n}_conf.log" &&   # batch 1 only on that check (chain s)
        job ${n}_conf_b1 "$SEL" "${v[@]}" STAGE=confidence SCORE_BATCH=1; }; } &&
    tjob $n ao scripts/review_checks/ff_explore_ec_compare.py "$RUN/eval_extra/confidence_$t.json" \
      --ec-work "$RUN/eval_extra/judge_${sp}_${t}_$sp" --conf-work "$cw" --choice "$RUN/eval_extra/ec_choice_$t.json" \
      --json-out "$CLOG/$n.json"; } || give_up ${n}_gen ${n}_conf $n
}

# ---- tune: E+C under R6 (v1, then v1.1), then the equal-N comparison
tune_lane() {
  local t=ec$R6
  ec_v1 tune
  if [ -e "$LOG/${t}_g1.DONE" ]; then v11 tune 1; v11 tune 2; else note "v1.1 on tune not tried: no v1 generation"; fi
  # the must-answer E answers and baseline conf: the lines chain's TAG=D work dir of R3 judged exactly these answers with
  # AllenAI (genconf must reproduce them); the runner's judgeconf would add a Pro pass (chain s seeded it from that dir)
  if [ -e "$LOG/${t}_a.DONE" ] && up e_dallenai; then same_n tune "$RUN/judge_tune_proD_E_L$R3"
  else note "tune equal N not tried (needs ${t}_a and the lines chain's e_dallenai)"; give_up sn_tune_gen sn_tune_conf sn_tune; fi
}
# ---- test: stage B on the 529 questions (chain p's test lane, q's test block, test_sel27)
etables() {  # E at R3 on the must-answer prompt: cells at strengths 1 2 and the dose table, AllenAI
  local s w=$RUN/judge_test_testD_E_L$R3
  for s in 1 2; do
    tx scripts/review_checks/e44_round1_tables.py cells "$RUN" E "$w" --judge allenai --layers "$R3" --base conf --strength $s || return 1
  done
  ao scripts/review_checks/ff_explore_dose_table.py --e-on E2 --work "$MODEL:$R3:$w" --strengths 1 2
}
test_lane() {
  local L tc="" tec="" T=("${T0[@]}" "CLS=$R5" "CK=$KC")
  for L in $R5; do tc="$tc t_ca_L$L t_ct_L$L"; done
  [ "$R6" = none ] || { T+=("ECLS=$R6"); tec="t${R6}_g1 t${R6}_g2 t${R6}_a t${R6}_d sn_test_gen sn_test_conf sn_test"; }
  [ -n "$R5" ] && tc="t_cgen $tc"                                   # the C names (and the E+C names) behind t_cgen
  job t_freeze "$TE" "${T0[@]}" STAGE=freeze || { give_up t_egen t_ea t_et $tc $tec; return 0; }
  job t_egen "$TE" "${T0[@]}" STAGE=egen || { give_up t_ea t_et $tc $tec; return 0; }
  { job t_ea "$TE" "${T0[@]}" STAGE=ejudge JUDGES=allenai && tjob t_et etables; } || give_up t_et
  [ -n "$R5" ] || { note "R5 is empty or undetermined: no C test arms, no E+C"; return 0; }
  job t_cgen "$TE" "${T[@]}" STAGE=cgen || { give_up $tc $tec; return 0; }
  for L in $R5; do
    { job t_ca_L$L "$TE" "${T[@]}" STAGE=cjudge LAYER=$L JUDGES=allenai &&
      tjob t_ct_L$L tx scripts/review_checks/e44_round1_tables.py cells "$RUN" C "$RUN/judge_test_testS_C_L$L" --judge allenai \
        --layers $L --strength "$KC"; } || give_up t_ct_L$L
  done
  [ "$R6" = none ] && { note "no R6 layer: no E+C"; return 0; }
  ec_v1 test
  if [ -e "$LOG/t${R6}_g1.DONE" ]; then v11 test 1; v11 test 2; else note "v1.1 on test not tried: no v1 generation"; fi
  if [ -e "$LOG/t${R6}_a.DONE" ] && [ -e "$LOG/t_ea.DONE" ]; then same_n test "$RUN/judge_test_testD_E_L$R3"
  else note "test equal N not tried (needs t${R6}_a and t_ea)"; give_up sn_test_gen sn_test_conf sn_test; fi
}

if [ "$R6" = none ]; then note "no tune E+C (R6 none or undetermined)"; else tune_lane; fi
test_lane
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"
