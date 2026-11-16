#!/usr/bin/env bash
# Iteration_01 (Claude Opus 5.5, 2026-11-15): free-form E44 for ONE model, step 0 + E line + C line, AllenAI only.
# 赵磊 11-15: "评测时，先不用deepseek Pro了，先用allenAI看看是否成立。"
# The stages, arm lists and rules are those of the 27B run of 11-13/14 (from_claude/45 sections 3 and 3.1; chains
# ff_explore_20261113n_chain.sh = E line and ff_explore_20261113o_chain.sh = C line). What changes (charter 10_04: periphery):
#   * MODEL, GPU, FF_ROOT (layered results root), E_LAYERS and C_LAYERS are parameters;
#   * every judging step is AllenAI only (JUDGES=allenai); no DeepSeek Pro step is run;
#   * R3 = the E layer with the largest AllenAI change in correct answers (must-answer, strength 1), ties to the lower
#     layer (ff_rule_pick.py top --k 1: the first half of R3); the second half of R3 (the sum with Pro) needs Pro;
#   * the strength-2/3 arms (dose) run on that one layer.
# Order on the given GPU(s): C pilot (three questions) -> step 0 -> E line -> C line. E+C and the 529-question test are a
# later chain. Start it only after any hand-run step 0 of the same model has exited (red team 11-15: two step 0 would race).
# Red team 11-15 fixes: jobs read /dev/null, not the layer loop; one-shot DONE checks (the chain is sequential); the layer lists
# are validated and recorded in LOG/params.txt (a rerun with other lists is refused); dry-run hooks CHAIN_RUNNER / CHAIN_LOG /
# CHAIN_TX_DRY as in chain n.
# usage (host):
#   MODEL=gemma2-9b GPU=1 FF_ROOT=/workspace/Truth_Decoupling/Iteration_01/results/ff/gemma2/9b \
#   E_LAYERS="10 12 14 16 18 20" C_LAYERS="13 15 17 19 21 23 25" bash ff_ec_lines_chain.sh
# Logs, .DONE / .FAIL flags and chain_jobs.log in <FF_ROOT on the host>/logs/chain; a rerun skips jobs that are .DONE.
set -uo pipefail
: "${MODEL:?}" "${GPU:?}" "${FF_ROOT:?}" "${E_LAYERS:?}" "${C_LAYERS:?}"
R=/data/sanshizhao/Truth_Decoupling
W=/workspace/Truth_Decoupling
CODE=$W/Iteration_01/code
HERE=$R/Iteration_01/code/scripts/review_checks
RUNNER=${CHAIN_RUNNER:-$HERE/newmodel_e44.sh}                    # override: dry runs only
case "$FF_ROOT" in "$W"/*) ;; *) echo "FF_ROOT must be an absolute path under $W" >&2; exit 2 ;; esac
[ -f "$R/Iteration_01/code/configs/models/$MODEL.json" ] || { echo "no model config $MODEL" >&2; exit 2; }
NL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['num_layers'])" "$R/Iteration_01/code/configs/models/$MODEL.json") \
  || { echo "cannot read num_layers of $MODEL" >&2; exit 2; }
check_layers() {  # check_layers NAME "L ...": digits only, distinct, below the model's layer count
  local name=$1 list=$2 L
  [ -n "$(echo $list)" ] || { echo "$name is empty" >&2; exit 2; }
  for L in $list; do case "$L" in ""|*[!0-9]*) echo "$name: '$L' is not a layer number" >&2; exit 2 ;; esac
    [ "$L" -lt "$NL" ] || { echo "$name: layer $L >= $NL layers" >&2; exit 2; }; done
  [ "$(echo $list | tr ' ' '\n' | sort -n | uniq -d | wc -l)" = 0 ] || { echo "$name repeats a layer" >&2; exit 2; }
}
check_layers E_LAYERS "$E_LAYERS"; check_layers C_LAYERS "$C_LAYERS"
for f in newmodel_e44.sh e44_round1_tables.py ff_rule_pick.py; do
  [ -f "$HERE/$f" ] || { echo "the code base has no scripts/review_checks/$f" >&2; exit 3; }
done
LOG=${CHAIN_LOG:-$R/${FF_ROOT#"$W"/}/logs/chain}
POLL=30
mkdir -p "$LOG"
mkdir "$LOG/.lock_chain" 2>/dev/null || { echo "another chain holds $LOG/.lock_chain (remove it only if none runs)" >&2; exit 1; }
trap 'rmdir "$LOG/.lock_chain" 2>/dev/null' EXIT
PARAMS="model=$MODEL root=$FF_ROOT e_layers=$(echo $E_LAYERS) c_layers=$(echo $C_LAYERS)"
if [ -f "$LOG/params.txt" ]; then
  [ "$(cat "$LOG/params.txt")" = "$PARAMS" ] || { echo "this log dir was started with other parameters: $(cat "$LOG/params.txt")" >&2; exit 2; }
else
  echo "$PARAMS" > "$LOG/params.txt"
fi
rm -f "$LOG"/*.FAIL
CLEAN=(-u GPU -u LAYERS -u LAYER -u ARMS -u BASES -u TAG -u JUDGES -u STRENGTH -u VARIANT -u E_ON -u PURPOSE -u DOSE_LAYERS
       -u DIAG_PROMPTS -u CACHE_FROM -u FF_ROOT -u BRANCH -u MODEL -u STAGE -u CODE_PATH -u RUN_DIR -u SOURCE_DIR)
note() { echo "$(date '+%F %T') $*" >> "$LOG/chain_jobs.log"; }
job() {  # job NAME VAR=value ...: one runner call in a clean environment; skipped when NAME.DONE exists
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  [ -e "$LOG/$name.log" ] && mv -f "$LOG/$name.log" "$LOG/$name.log.prev"   # keep the log of a failed earlier try
  if env "${CLEAN[@]}" "$@" bash "$RUNNER" > "$LOG/$name.log" 2>&1 < /dev/null; then
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
wait_done() {  # one-shot: the chain is sequential, so a job without DONE at this point will not finish later
  local f
  for f in "$@"; do [ -e "$LOG/$f.DONE" ] || return 1; done
}
tjob() {  # tjob NAME COMMAND...: a read-only table command (tx) with the same DONE / FAIL flags; stdout to NAME.md
  local name=$1; shift
  [ -e "$LOG/$name.DONE" ] && return 0
  rm -f "$LOG/$name.FAIL"; note "start $name"
  if "$@" > "$LOG/$name.md" 2> "$LOG/$name.log" < /dev/null; then
    touch "$LOG/$name.DONE"; note "done  $name"
  else
    echo "failed, see $name.log" > "$LOG/$name.FAIL"; note "FAIL  $name"; return 1
  fi
}
groups_of3() {  # groups_of3 "L1 L2 ...": print the layers three per line
  echo "$1" | tr ' ' '\n' | grep -v '^$' | paste -d ' ' - - -
}
RUN=$FF_ROOT/e44/$MODEL
VB=(MODEL=$MODEL E_ON=E2 CODE_PATH=$CODE FF_ROOT=$FF_ROOT)
VE=("${VB[@]}" BRANCH=E)
VC=("${VB[@]}" BRANCH=C)
# arm lists: copied from ff_explore_20261113n_chain.sh (E) and ff_explore_20261113o_chain.sh (C)
E_EXTRA_A="x.rand.m1 x_F.w.p1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_PROA="x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 conf.wq.p1 conf_E2.wq.m1 conf.iso.p1 caut.wq.p1 caut_E2.wq.m1 conf.w.p1 conf.md.p1 conf.rand.p1 conf.rand.m1"
E2_BASES="x x_F conf conf_E2 caut caut_E2"
NEWARMS="conf.w.p2 conf.w.p3 conf.rand.p2 conf.rand.m2 conf.rand.p3 conf.rand.m3 conf.md.p2 conf.md.p3"
DOSE="conf.w.p1 conf.rand.p1 conf.rand.m1 conf.md.p1 $NEWARMS"
DBASES="conf conf_E2"
C_S="x.w.p0.5 x.w.m0.5 x.md.p0.5 x.md.m0.5 x.rand.p0.5 x.rand.m0.5 x.w.p0.25 x.w.m0.25 x.md.p0.25 x.md.m0.25 x.rand.p0.25 x.rand.m0.25"
TX=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec python3)
tx() {  # read-only table scripts (CHAIN_TX_DRY=1: print the command; for the R3 pick also a "top:" line, dry runs only)
  if [ -n "${CHAIN_TX_DRY:-}" ]; then
    echo "tx $*"
    case "$*" in *ff_rule_pick.py\ top*) echo "top: $(echo $E_LAYERS | awk '{print $2}')" ;; esac
  else
    "${TX[@]}" "$@"
  fi
}

note "chain start (pid $$): model $MODEL, GPU $GPU, E layers [$E_LAYERS], C layers [$C_LAYERS], root $FF_ROOT"
( cd "$HERE" && sha256sum newmodel_e44.sh e44_round1_tables.py ff_rule_pick.py ) >> "$LOG/chain_jobs.log" 2>&1 || true
sha256sum "$0" >> "$LOG/chain_jobs.log" 2>&1 || true

# ---- C pilot: the three-question execution check of the C branch (the E pilot ran by hand before step 0)
job pilot_C "${VB[@]}" BRANCH=C STAGE=pilot GPU=$GPU || { note "C pilot failed: nothing else tried"; exit 1; }

# ---- step 0: freeze the 11 prompts on train + tune, reference readout, export, AllenAI
job s0 "${VB[@]}" STAGE=step0 GPU=$GPU || { note "step 0 failed: nothing else tried"; exit 1; }

# ---- E line
e_line() {
  local grp k=0 L names
  while read -r grp; do
    [ -n "$grp" ] || continue
    k=$((k + 1)); names="e_extra_$k"
    for L in $grp; do names="$names e_allenai_L$L e_atables_L$L"; done
    job e_run_$k "${VE[@]}" STAGE=run GPU=$GPU LAYERS="$grp" || { give_up $names; continue; }
    job e_extra_$k "${VE[@]}" STAGE=extra GPU=$GPU LAYERS="$grp" ARMS="$E_EXTRA_A" || { give_up $names; continue; }
    for L in $grp; do
      { job e_allenai_L$L "${VE[@]}" STAGE=pro GPU=$GPU LAYER=$L TAG=X JUDGES=allenai ARMS="$E2_PROA" BASES="$E2_BASES" &&
        job e_atables_L$L "${VE[@]}" STAGE=tables LAYERS=$L TAG=X STRENGTH=1 JUDGES=allenai; } || give_up e_atables_L$L
    done
  done <<< "$(groups_of3 "$E_LAYERS")"
  local f=""
  for L in $E_LAYERS; do f="$f e_allenai_L$L"; done
  wait_done $f || { note "R3 not possible: an AllenAI pass of the E line failed"; give_up r3_top1 e_dextra e_dallenai e_dtables; return 1; }
  tjob r3_top1 tx scripts/review_checks/ff_rule_pick.py top --run "$RUN" --layers $E_LAYERS --k 1 || { give_up e_dextra e_dallenai e_dtables; return 1; }
  R3=$(sed -n 's/^top: //p' "$LOG/r3_top1.md" | awk '{print $1}')
  case "$R3" in ""|*[!0-9]*) echo "r3_top1.md names no layer" > "$LOG/r3_top1.FAIL"; rm -f "$LOG/r3_top1.DONE"; give_up e_dextra e_dallenai e_dtables; return 1 ;; esac
  note "R3 (AllenAI only): layer $R3"
  job e_dextra "${VE[@]}" STAGE=extra GPU=$GPU LAYERS="$R3" ARMS="$NEWARMS" || { give_up e_dallenai e_dtables; return 0; }
  job e_dallenai "${VE[@]}" STAGE=pro GPU=$GPU LAYER=$R3 TAG=D JUDGES=allenai ARMS="$DOSE" BASES="$DBASES" || { give_up e_dtables; return 0; }
  e_dtables() {
    local S
    for S in 1 2 3; do
      tx scripts/review_checks/e44_round1_tables.py cells "$RUN" E "$RUN/judge_tune_proD_E_L$R3" --judge allenai --layers "$R3" --base conf --strength "$S" || return 1
    done
  }
  tjob e_dtables e_dtables || true
}

# ---- C line
c_line() {
  local grp k=0 ok=1 L SHORT S
  while read -r grp; do
    [ -n "$grp" ] || continue
    k=$((k + 1))
    job c_run_$k "${VC[@]}" STAGE=run GPU=$GPU LAYERS="$grp" || ok=0
  done <<< "$(groups_of3 "$C_LAYERS")"
  if [ $ok = 1 ] && job c_judge "${VC[@]}" STAGE=judge GPU=$GPU LAYERS="$C_LAYERS" && job c_screen "${VC[@]}" STAGE=screen; then
    SHORT=$(sed -n 's/^shortlist layers: *//p' "$LOG/c_screen.log")
    note "C shortlist (round 1's screen, AllenAI): [$SHORT]"
    for L in $SHORT; do
      { job c_extra_L$L "${VC[@]}" STAGE=extra GPU=$GPU LAYERS=$L ARMS="$C_S" &&
        job c_allenai_L$L "${VC[@]}" STAGE=pro GPU=$GPU LAYER=$L TAG=S JUDGES=allenai ARMS="$C_S" BASES=x &&
        for S in 0.5 0.25; do job c_tables_L${L}_s$S "${VC[@]}" STAGE=tables LAYERS=$L TAG=S STRENGTH=$S JUDGES=allenai || true; done; } \
        || give_up c_allenai_L$L c_tables_L${L}_s0.5 c_tables_L${L}_s0.25
    done
  else
    [ $ok = 1 ] || note "a C training part failed: the screen needs every C layer; judge and screen not tried"
    give_up c_judge c_screen
  fi
}

e_line || true
c_line
note "chain end: $(ls "$LOG" | grep -c '\.DONE$') done, $(ls "$LOG" | grep -c '\.FAIL$') failed or not tried"
