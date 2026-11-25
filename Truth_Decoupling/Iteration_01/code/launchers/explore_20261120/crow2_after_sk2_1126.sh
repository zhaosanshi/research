#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor yin, 2026-11-25 22 h; plan discussion/work_queue/2026-11-24_queue_m.md, progress
# entry 2026-11-25 22:11 part one, rules v4): judging and the table of gemma3-27b layers 0-17 trained with the cause part and
# the isolate part back-propagated separately (results root Iteration_01/results/ff/gemma3/27b_splitkinds). No GPU.
# A copy of crow2_from_after_1125.sh (fcd73c3066bd53ee) with these differences, all of them:
#   (1) run dir Iteration_01/results/ff/gemma3/27b_splitkinds/e44/gemma3-27b; the generation it waits for is
#       crow2_gen_sk_1125.sh g27sk (marker CROW2_gemma3-27b_SK_GEN_DONE; _SK_GEN_FAILED / _SK_TIME_STOP / no process for
#       3 polls -> _SK_TABLE_FAILED); AFTER_MINL is not used.
#   (2) layers judged: those with a direction file in the new run dir; layers 0-17 and 18, 20, 31 must all be there.
#   (3) judge preparation and Pro by crow2_judge_sk_1125.sh (key g27sk, export name crow2_C_sk, work dir
#       judge_tune_crow2_C_sk, seed cache from the main root's judge_tune_crow2_C_from18, cost stop 5 CNY), markers
#       CROW2_crow2_C_sk_gemma3-27b_PREP_DONE / _COST_STOP / _prep_FAILED / _PRO_DONE.
#   (4) balance floor: before every Pro start the balance (balance.sh, total_balance) minus the expected cost of the
#       batch (the cost file of prep) must stay >= 10 CNY, else _SK_TABLE_FAILED (cost cap 5 CNY is (3)).
#   (5) table: c_row_layers_sk_1125.py tune on the new run dir, work dir crow2_C_sk, layers 0-17,18,20,31, with the
#       results file of the four checks in the header -> Iteration_01/diag/out_c_row_layers_sk_gemma3-27b_<yyyymmdd>.txt.
#       Instead of the stage-one archive check: the same reader pointed at the MAIN root (work dir
#       judge_tune_crow2_C_from18, layers 18,20,31) must reproduce the rows of diag/out_c_row_layers2_from18_gemma3-27b_20261125.txt
#       character by character (-> ..._check.txt; rc != 0 -> _SK_TABLE_FAILED). The main root's table is not touched.
#   (6) markers CROW2_gemma3-27b_SK_TABLE_DONE / _SK_TABLE_FAILED; log crow2_after_sk.log; status lines "crow2 sk after: ..."
# The off-peak guard (weekdays 08:30-12:00 and 14:00-18:00 no Pro start), the shared .crow2_pro.lock, the retry after
# exit 3 / 4 of the judge script are those of the source.
# usage (host): ( nohup setsid bash Iteration_01/logs/newsets_20261119/crow2_after_sk_1125.sh \
#                 > Iteration_01/logs/newsets_20261119/crow2_after_sk_1125.out 2>&1 < /dev/null & )
#
# crow2_after_sk2_1126.sh (Claude Opus 5.5, executor chen, 2026-11-26 01 h; plan discussion/work_queue/2026-11-24_queue_m.md, entries 2026-11-25 22:11 part one and 2026-11-26 00:48 (second
# version of the patch, split_kinds2_1126.patch: both halves left-padded to the width of the whole 8-row batch;
# training contract "backward": "by_kind_full_width")):
# a copy of crow2_after_sk_1125.sh (7cdc1b162976d501) made by program for the second version. The text above describes the
# source; the differences, all of them (names only):
#   run dir Iteration_01/results/ff/gemma3/27b_splitkinds2/e44/gemma3-27b; key g27sk2; export name crow2_C_sk2;
#   waits for CROW2_gemma3-27b_SK2_GEN_DONE (_SK2_GEN_FAILED / _SK2_TIME_STOP / no crow2_gen_sk2_1126.sh g27sk2 process
#   for 3 polls -> _SK2_TABLE_FAILED); judge script crow2_judge_sk2_1126.sh; reader c_row_layers_sk2_1126.py (a copy of
#   c_row_layers_sk_1125.py whose printed header names the second code copy); results file of the checks
#   out_split_kinds2_checks_gemma3-27b_*.txt; table Iteration_01/diag/out_c_row_layers_sk2_gemma3-27b_<date>.txt (+ _check.txt,
#   the reader check against the main root, unchanged); markers CROW2_gemma3-27b_SK2_TABLE_DONE / _FAILED; log
#   crow2_after_sk2.log, lock-guard dir .crow2_after_sk2_running, outputs crow2_prep_sk2_g27sk2.out / crow2_pro_sk2_g27sk2.out,
#   status lines "crow2 sk2 after". Balance floor 10 CNY, cost cap 5 CNY, off-peak guard, .crow2_pro.lock as in the source.
# usage (host): ( nohup setsid bash Iteration_01/logs/newsets_20261119/crow2_after_sk2_1126.sh \
#                 > Iteration_01/logs/newsets_20261119/crow2_after_sk2_1126.out 2>&1 < /dev/null & )
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
MODEL=gemma3-27b; KEY=g27sk2; NAME=crow2_C_sk2
RUN=Iteration_01/results/ff/gemma3/27b_splitkinds2/e44/gemma3-27b
RUN0=Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b
NL=62
G=$L/CROW2_${MODEL}_SK2
J=$L/CROW2_${NAME}_${MODEL}
REFT=Iteration_01/diag/out_c_row_layers2_from18_gemma3-27b_20261125.txt
LOG=$L/crow2_after_sk2.log
JUDGE=$L/crow2_judge_sk2_1126.sh
READER=Iteration_01/diag/c_row_layers_sk2_1126.py
say() { echo "$(date '+%F %T') crow2 sk2 after: $*" | tee -a $L/crow_all.log >> $LOG; }
fail() { say "FAILED: $*"; touch ${G}_TABLE_FAILED; exit 1; }
mkdir $L/.crow2_after_sk2_running 2>/dev/null || { echo "already running ($L/.crow2_after_sk2_running)"; exit 3; }
trap 'rmdir $L/.crow2_after_sk2_running 2>/dev/null' EXIT
[ -e ${G}_TABLE_DONE ] && { say "already done (${G}_TABLE_DONE)"; exit 0; }
rm -f ${G}_TABLE_FAILED
for f in $JUDGE $L/balance.sh $READER Iteration_01/diag/c_row_layers.py $REFT; do
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
  [ -e ${G}_GEN_FAILED ] && fail "the generation failed (${G}_GEN_FAILED; see $L/crow2_gen_sk2_$MODEL.log)"
  [ -e ${G}_TIME_STOP ] && fail "the generation stopped at its time limit (${G}_TIME_STOP)"
  if pgrep -f "[c]row2_gen_sk2_1126.sh $KEY" > /dev/null; then gone=0; else gone=$((gone + 1)); fi
  [ $gone -ge 3 ] && fail "no crow2_gen_sk2_1126.sh $KEY process for 3 checks and no marker"
  sleep 300
done
say "seen ${G}_GEN_DONE"
CHECKS=$(ls Iteration_01/diag/out_split_kinds2_checks_${MODEL}_*.txt 2>/dev/null)
[ $(echo $CHECKS | wc -w) = 1 ] || fail "need exactly one results file of the checks, found [$CHECKS]"
grep -q "^ALL FOUR CHECKS PASSED" $CHECKS || fail "$CHECKS does not say the checks passed"
LAYERS=""
for l in $(seq 0 $((NL - 1))); do [ -s $RUN/dirs/C_L$l.pt ] && LAYERS="$LAYERS $l"; done
LAYERS=${LAYERS# }
for l in $(seq 0 17) 18 20 31; do [ -s $RUN/dirs/C_L$l.pt ] || fail "layer $l has no direction file after GEN_DONE"; done
say "layers with a direction: [$LAYERS]"
if [ ! -e ${J}_PREP_DONE ]; then
  CROW2_NAME=$NAME CROW2_LAYERS="$LAYERS" bash $JUDGE prep $KEY >> $L/crow2_prep_sk2_$KEY.out 2>&1 \
    || fail "prep failed (see $L/crow2_judge_${NAME}_$MODEL.log; cost stop marker ${J}_COST_STOP if the batch is above 5 CNY)"
fi
[ -e ${J}_PREP_DONE ] || fail "no ${J}_PREP_DONE after prep"
COSTF=$L/crow2_cost_${NAME}_$MODEL.txt
say "prep done: $(tr '\n' ' ' < $COSTF)"
EXP=$(sed -n 's/^expected cost \([0-9.]*\) CNY.*/\1/p' $COSTF)
[[ $EXP =~ ^[0-9]+\.[0-9]+$ ]] || fail "cannot read the expected cost from $COSTF"
while [ ! -e ${J}_PRO_DONE ]; do
  until offpeak_ok; do sleep 600; done
  BAL=$(bash $L/balance.sh | python3 -c 'import json,sys; d=json.load(sys.stdin); print([x for x in d if x["currency"] == "CNY"][0]["total_balance"])' 2>/dev/null)
  if ! [[ $BAL =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    nobal=$(( ${nobal:-0} + 1 ))
    [ $nobal -ge 6 ] && fail "cannot read the balance (6 tries, 10 minutes apart)"
    say "cannot read the balance (try $nobal), retry in 10 minutes"; sleep 600; continue
  fi
  awk -v b=$BAL -v e=$EXP 'BEGIN {exit !(b - e >= 10)}' || fail "balance $BAL CNY minus the expected cost $EXP CNY is below the floor of 10 CNY: Pro not started"
  say "balance $BAL CNY, expected cost $EXP CNY: above the floor"
  exec 7>$L/.crow2_pro.lock
  flock 7
  if offpeak_ok; then
    CROW2_NAME=$NAME bash $JUDGE pro $KEY >> $L/crow2_pro_sk2_$KEY.out 2>&1; rc=$?
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
TL=$(echo $(seq 0 17) 18 20 31 | tr ' ' ',')
out=Iteration_01/diag/out_c_row_layers_sk2_${MODEL}_$(date +%Y%m%d)
docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec \
  python3 -B $READER tune $RUN $NAME $TL $CHECKS > $out.txt 2>&1 || fail "reader failed (see $out.txt)"
docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec \
  python3 -B $READER check $RUN0 crow2_C_from18 18,20,31 $REFT > ${out}_check.txt 2>&1
rc=$?
say "table $out.txt; reader check on the main root rc $rc: $(head -n 1 ${out}_check.txt)"
[ $rc = 0 ] || fail "the reader does not reproduce the main root's rows (rc $rc, see ${out}_check.txt)"
touch ${G}_TABLE_DONE
say "done: ${G}_TABLE_DONE"
