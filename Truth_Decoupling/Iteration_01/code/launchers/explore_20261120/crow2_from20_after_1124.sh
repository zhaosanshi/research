#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor gui, 2026-11-24 21:4x; plan discussion/work_queue/2026-11-24_queue_m.md, progress
# entry 21:35 point 1): what crow2_queue_1124.sh does after a finished generation (its "lane"), for the PARTIAL run of
# gemma3-27b by crow2_gen_from_1124.sh with CROW2_MINL=20 (layers 0-18 not trained: one card cannot hold that). No GPU.
#   1. wait for CROW2_gemma3-27b_FROM20_GEN_DONE (polled every 5 minutes). _FROM20_GEN_FAILED, _FROM20_TIME_STOP, or the
#      generation process gone without a marker -> _FROM20_TABLE_FAILED with the reason, stop.
#   2. judge preparation: the unchanged crow2_judge.sh prep g27 with CROW2_NAME=crow2_C_from20 and CROW2_LAYERS = the
#      layers that have a direction file (19-61 expected). Work dir <run>/judge_tune_crow2_C_from20, its markers
#      CROW2_crow2_C_from20_gemma3-27b_PREP_DONE / _COST_STOP / _prep_FAILED, log crow2_judge_crow2_C_from20_gemma3-27b.log.
#      (crow2_judge.sh's prep always lists all layers of the model under the default name, and fails on the absent arms
#      of layers 0-18; a non-default name is the script's own way to restrict the layers. Seed cache, cost stop at 6 CNY,
#      conflict check, Pro settings are the same.)
#   3. Pro (non-thinking) off-peak, as the queue's lane: waits while it is peak time (weekdays 08:30-12:00, 14:00-18:00
#      Beijing), one Pro pass at a time through .crow2_pro.lock (shared with crow2_queue_1124.sh's lanes), retried every
#      10 minutes after exit 3 (guard) / 4 (stopped itself before a peak). Marker CROW2_crow2_C_from20_gemma3-27b_PRO_DONE.
#   4. table: c_row_layers2_from.py 20 crow2_C_from20 tune g27 (c_row_layers2.py with this work dir; layers 0-18 print
#      "NOT TRAINED (one card cannot hold the training)", also said in the header) ->
#      Iteration_01/diag/out_c_row_layers2_from20_gemma3-27b_<yyyymmdd>.txt, and its check against the stage-one archive
#      diag/ref_c_row_layers_five_models_20261124.txt -> ..._check.txt. Check rc != 0 -> _FROM20_TABLE_FAILED.
# Markers Iteration_01/logs/newsets_20261119/CROW2_gemma3-27b_FROM20_TABLE_DONE / _FROM20_TABLE_FAILED; own log
# crow2_from20_after.log; status lines "crow2 from20 after: ..." in crow_all.log.
# usage (host): (nohup setsid bash Iteration_01/logs/newsets_20261119/crow2_from20_after_1124.sh \
#                 > Iteration_01/logs/newsets_20261119/crow2_from20_after.out 2>&1 < /dev/null &)
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
MODEL=gemma3-27b; KEY=g27; MINL=20; NAME=crow2_C_from20
RUN=Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b
NL=62
G=$L/CROW2_${MODEL}_FROM${MINL}
J=$L/CROW2_${NAME}_${MODEL}
REFT=Iteration_01/diag/ref_c_row_layers_five_models_20261124.txt
LOG=$L/crow2_from20_after.log
say() { echo "$(date '+%F %T') crow2 from20 after: $*" | tee -a $L/crow_all.log >> $LOG; }
fail() { say "FAILED: $*"; touch ${G}_TABLE_FAILED; exit 1; }
mkdir $L/.crow2_from20_after_running 2>/dev/null || { echo "already running ($L/.crow2_from20_after_running)"; exit 3; }
trap 'rmdir $L/.crow2_from20_after_running 2>/dev/null' EXIT
[ -e ${G}_TABLE_DONE ] && { say "already done (${G}_TABLE_DONE)"; exit 0; }
rm -f ${G}_TABLE_FAILED
for f in $L/crow2_judge.sh $L/balance.sh Iteration_01/diag/c_row_layers2_from.py Iteration_01/diag/c_row_layers2.py \
         Iteration_01/diag/c_row_layers.py $REFT; do
  [ -s $f ] || fail "missing $f"
done
offpeak_ok() {  # the guard of crow2_judge.sh pro: weekdays refuse 08:30-12:00 and 14:00-18:00
  local t wd
  t=$((10#$(date +%H) * 60 + 10#$(date +%M))); wd=$(date +%u)
  [ $wd -le 5 ] || return 0
  [ $t -ge 510 ] && [ $t -lt 720 ] && return 1
  [ $t -ge 840 ] && [ $t -lt 1080 ] && return 1
  return 0
}
say "start (pid $$); waiting for ${G}_GEN_DONE"
gone=0
until [ -e ${G}_GEN_DONE ]; do
  [ -e ${G}_GEN_FAILED ] && fail "the generation failed (${G}_GEN_FAILED; see $L/crow2_gen_from${MINL}_$MODEL.log)"
  [ -e ${G}_TIME_STOP ] && fail "the generation stopped at its time limit (${G}_TIME_STOP): rerun crow2_gen_from_1124.sh, then this"
  if pgrep -f "[c]row2_gen_from_1124.sh $KEY" > /dev/null; then gone=0; else gone=$((gone + 1)); fi
  [ $gone -ge 3 ] && fail "no crow2_gen_from_1124.sh $KEY process for 3 checks and no marker"
  sleep 300
done
say "seen ${G}_GEN_DONE"
LAYERS=""
for l in $(seq 0 $((NL - 1))); do [ -s $RUN/dirs/C_L$l.pt ] && LAYERS="$LAYERS $l"; done
LAYERS=${LAYERS# }
for l in $(seq $MINL $((NL - 1))); do [ -s $RUN/dirs/C_L$l.pt ] || fail "layer $l has no direction file after GEN_DONE"; done
say "layers with a direction: [$LAYERS]"
if [ ! -e ${J}_PREP_DONE ]; then
  CROW2_NAME=$NAME CROW2_LAYERS="$LAYERS" bash $L/crow2_judge.sh prep $KEY >> $L/crow2_prep_from20_$KEY.out 2>&1 \
    || fail "prep failed (see $L/crow2_judge_${NAME}_$MODEL.log; cost stop marker ${J}_COST_STOP if the batch is above 6 CNY)"
fi
[ -e ${J}_PREP_DONE ] || fail "no ${J}_PREP_DONE after prep"
say "prep done: $(tr '\n' ' ' < $L/crow2_cost_${NAME}_$MODEL.txt)"
while [ ! -e ${J}_PRO_DONE ]; do
  until offpeak_ok; do sleep 600; done
  exec 7>$L/.crow2_pro.lock
  flock 7
  if offpeak_ok; then
    CROW2_NAME=$NAME bash $L/crow2_judge.sh pro $KEY >> $L/crow2_pro_from20_$KEY.out 2>&1; rc=$?
  else
    rc=3
  fi
  flock -u 7; exec 7>&-
  case $rc in
    0) ;;
    3|4) say "pro not finished (rc $rc), retry off-peak"; sleep 600 ;;
    *) fail "pro failed (rc $rc, see $L/crow2_judge_${NAME}_$MODEL.log)" ;;
  esac
done
say "pro done"
out=Iteration_01/diag/out_c_row_layers2_from20_${MODEL}_$(date +%Y%m%d)
docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec \
  python3 -B Iteration_01/diag/c_row_layers2_from.py $MINL $NAME tune $KEY > $out.txt 2>&1 || fail "reader failed (see $out.txt)"
docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec \
  python3 -B Iteration_01/diag/c_row_layers2_from.py $MINL $NAME check $REFT $KEY > ${out}_check.txt 2>&1
rc=$?
say "table $out.txt; stage-one row check rc $rc: $(sed -n 2p ${out}_check.txt)"
[ $rc = 0 ] || fail "the stage-one row check differs (rc $rc, see ${out}_check.txt)"
touch ${G}_TABLE_DONE
say "done: ${G}_TABLE_DONE"
