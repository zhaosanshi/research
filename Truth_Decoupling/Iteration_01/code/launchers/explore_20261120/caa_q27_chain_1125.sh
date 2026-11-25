#!/bin/bash
# Iteration_01 (main session, Claude Fable 5.2, 2026-11-25 14 h; plan discussion/work_queue/2026-11-22_queue_f.md,
# sections 1, 7 and 8): starts the CAA chain of Qwen3.8-27B. Its nine-layer grid ended on 2026-11-25 04:35
# (baselines/caa/qwen3.8-27b/CAA_grid_g1_DONE). The chain's judging does not start in peak hours (weekdays 9-12 and
# 14-18 Beijing time), so this script waits until START and then calls caa_chain.sh unchanged, with the layer list,
# multipliers, cache tags and GSM8K token cap of code/configs/newsets/caa_grid_20261122.json, on card CARD (the grid's
# card). Nothing is computed here. Markers and tables are those of caa_chain.sh (CAA_CHAIN_qwen3.8-27b_DONE / _FAILED,
# Iteration_01/diag/out_caa_qwen3.8-27b_*_20261122.txt); status lines go to caa_all.log.
# Switches: CAA_Q27_START ("YYYY-MM-DD HH:MM:SS", Beijing), CAA_Q27_CARD, CAA_Q27_DRY=1 (print the call, do not run it).
# usage (host): ( nohup setsid bash Iteration_01/logs/newsets_20261119/caa_q27_chain_1125.sh \
#                 > Iteration_01/logs/newsets_20261119/caa_q27_chain_1125.out 2>&1 < /dev/null & )
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
L=Iteration_01/logs/newsets_20261119
MODEL=qwen3.8-27b
START=${CAA_Q27_START:-2026-11-25 18:00:00}
CARD=${CAA_Q27_CARD:-3}
DRY=${CAA_Q27_DRY:-0}
say() { echo "$(date '+%F %T') caa start $MODEL: $*" >> $L/caa_all.log; }
case "$CARD" in 0|2|3) ;; *) echo "card 0, 2 or 3 only (caa_grid_20261122.json)"; exit 2 ;; esac
t=$(date -d "$START" +%s) || { echo "bad start time"; exit 2; }
[ -e Iteration_01/baselines/caa/$MODEL/CAA_grid_g1_DONE ] || { echo "no finished grid for $MODEL"; exit 2; }
[ -s $L/caa_chain.sh ] || { echo "missing $L/caa_chain.sh"; exit 2; }
CALL=(bash $L/caa_chain.sh $MODEL $CARD "16 20 24 28 32 36 40 44 48" "0.5 1 2 4" g1 pospromptl28 pospromptl28 $CARD)
if [ "$DRY" = 1 ]; then
  echo "dry: would wait until $START (epoch $t, now $(date +%s)) and then run:"; printf ' %q' "${CALL[@]}"; echo; exit 0
fi
mkdir $L/.caa_q27_chain_running 2>/dev/null || { echo "already running ($L/.caa_q27_chain_running)"; exit 3; }
trap 'rmdir $L/.caa_q27_chain_running 2>/dev/null' EXIT
say "waiting until $START, then caa_chain.sh on card $CARD"
while [ "$(date +%s)" -lt "$t" ]; do sleep 30; done
say "starting the chain"
"${CALL[@]}"
rc=$?
say "chain ended (exit $rc)"
exit $rc
