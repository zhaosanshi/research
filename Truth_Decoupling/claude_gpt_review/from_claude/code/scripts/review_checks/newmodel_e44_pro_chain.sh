#!/usr/bin/env bash
# from_claude/43 section 5, steps 6-8 for one model (Claude, 2026-11-12): the extra arms, the Pro + AllenAI judging of the routed
# layers, and the tables. Host side; every step is a STAGE of newmodel_e44.sh (fixed copy). The layer lists are NOT chosen here:
# they come from the frozen screens by the routing rule written in 43 before round 1 and are passed in, e.g.
#   E_PRO="9 11 12 13 14 15 18" E_DOSE="13 15" C_PRO="..." C_DOSE="..." newmodel_e44_pro_chain.sh qwen3.5-9b
# phase A (generation, in parallel): E pre-Pro arms for E_PRO (gpu0); E strength-sweep arms for E_DOSE (gpu1); C strength-sweep
#   arms for C_DOSE (gpu2). C proA jobs need no extra arms and start at once on gpu3.
# phase B (judging): every remaining Pro job (E proA per E_PRO layer, E proB per E_DOSE layer, C proB per C_DOSE layer) is taken
#   from a queue (atomic mkdir claim) by whichever of the four GPUs is free, each with AllenAI on the same cells.
# phase C: the frozen and exploratory tables (CPU).
set -u
MODEL=${1:?model}
: "${E_PRO:?}" "${E_DOSE:?}" "${C_PRO:?}" "${C_DOSE?}"      # C_DOSE may be empty (red team 5)
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$HOST/code_newmodels_ff_20261112
LOG=$HOST/results_newmodels_ff_20261112/logs
RUNNER=$CODE/scripts/review_checks/newmodel_e44.sh
CH=$LOG/prochain_$MODEL.chain.log
mkdir -p "$LOG"
mkdir "$LOG/.lock_prochain_$MODEL" 2>/dev/null || { echo "pro chain of $MODEL already started" >&2; exit 3; }
Q=$LOG/.queue_${MODEL}_$$; mkdir "$Q"                 # a fresh claim queue per launch: a relaunch is not silently skipped (red team 5)
echo "== $(date '+%F %T') routing: E_PRO=[$E_PRO] E_DOSE=[$E_DOSE] C_PRO=[$C_PRO] C_DOSE=[$C_DOSE]" >> "$CH"
step() {  # step NAME ENV...   (the chain's own layer lists and any stray arm settings are not passed to the runner)
  local name=$1; shift
  rm -f "$LOG/${MODEL}_$name.DONE" "$LOG/${MODEL}_$name.FAIL"
  echo "== $(date '+%F %T') start $name" >> "$CH"
  if env -u E_PRO -u E_DOSE -u C_PRO -u C_DOSE -u ARMS -u BASES -u PURPOSE -u TAG -u LAYER -u LAYERS -u DOSE_LAYERS -u CODE_PATH -u FF_ROOT \
      MODEL="$MODEL" "$@" bash "$RUNNER" > "$LOG/${MODEL}_$name.log" 2>&1; then
    echo "== $(date '+%F %T') done $name" >> "$CH"; touch "$LOG/${MODEL}_$name.DONE"
  else
    echo "== $(date '+%F %T') FAILED $name" >> "$CH"; touch "$LOG/${MODEL}_$name.FAIL"; return 1
  fi
}
# ---- phase A ---- #
step extra_E_proA STAGE=extra BRANCH=E LAYERS="$E_PRO" GPU=0 & pa1=$!
step extra_E_dose STAGE=extra BRANCH=E PURPOSE=dose LAYERS="$E_DOSE" GPU=1 & pa2=$!
if [ -n "$C_DOSE" ]; then step extra_C_dose STAGE=extra BRANCH=C PURPOSE=dose LAYERS="$C_DOSE" GPU=2 & pa3=$!; else pa3=; fi
( for L in $C_PRO; do step proA_C_L$L STAGE=pro BRANCH=C TAG=A LAYER=$L GPU=3; done ) & pc=$!
wait $pa1; ra1=$?; wait $pa2; ra2=$?; ra3=0; [ -n "$pa3" ] && { wait $pa3; ra3=$?; }
# ---- phase B ---- #
JOBS=()
[ $ra1 = 0 ] && for L in $E_PRO; do JOBS+=("proA_E_L$L|BRANCH=E TAG=A LAYER=$L"); done
[ $ra2 = 0 ] && for L in $E_DOSE; do JOBS+=("proB_E_L$L|BRANCH=E TAG=B LAYER=$L"); done
[ $ra3 = 0 ] && for L in $C_DOSE; do JOBS+=("proB_C_L$L|BRANCH=C TAG=B LAYER=$L"); done
[ $ra1 = 0 ] && [ $ra2 = 0 ] && [ $ra3 = 0 ] || echo "== $(date '+%F %T') some extra generation failed; its Pro jobs are skipped" >> "$CH"
lane() {  # lane GPU: take jobs until none is left
  local gpu=$1 j name spec
  for j in "${JOBS[@]}"; do
    name=${j%%|*}; spec=${j#*|}
    [ -e "$LOG/${MODEL}_$name.DONE" ] && continue      # already judged in an earlier launch
    mkdir "$Q/$name" 2>/dev/null || continue
    # shellcheck disable=SC2086
    step "$name" STAGE=pro $spec GPU=$gpu || true          # a failed layer is flagged and reported; the others go on
  done
}
lane 0 & l0=$!; lane 1 & l1=$!; lane 2 & l2=$!
wait $pc; lane 3 & l3=$!
wait $l0 $l1 $l2 $l3
# ---- phase C ---- #
done_of() {  # done_of PREFIX LAYERS... : the layers whose Pro job finished (one failed layer must not cost the others' tables)
  local pre=$1 L out=""; shift
  for L in "$@"; do [ -e "$LOG/${MODEL}_${pre}$L.DONE" ] && out="$out $L"; done
  echo $out
}
EA=$(done_of proA_E_L $E_PRO); EB=$(done_of proB_E_L $E_DOSE); CA=$(done_of proA_C_L $C_PRO)
[ -n "$EA" ] && { step tables_E STAGE=tables BRANCH=E LAYERS="$EA" DOSE_LAYERS="$EB" || true; }
[ -n "$CA" ] && { step tables_C STAGE=tables BRANCH=C LAYERS="$CA" || true; }
ls "$LOG" | grep -E "^${MODEL}_(extra|pro|tables).*\.FAIL$" >> "$CH" 2>/dev/null
echo "== $(date '+%F %T') pro chain finished" >> "$CH"
