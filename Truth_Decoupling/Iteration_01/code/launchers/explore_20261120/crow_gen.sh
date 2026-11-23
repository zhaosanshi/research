#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor F, 2026-11-24 02 h; plan discussion/work_queue/2026-11-24_queue_m.md section 1,
# first stage): the C row of the free-form 2x2, layer by layer, on the tune split (192 questions). For one model, the two
# C arms of the frozen protocol (P2 / from_claude/45 R5: plain prompt x, C at 0.25 toward caution x.w.p0.25 and toward
# answering x.w.m0.25) on every layer that has a C direction record and no such arm yet. Generation only: it calls the
# unchanged runner newmodel_e44.sh (STAGE=extra; for gemma3-27b the expandable-segments copy of 11-16) with each run's own
# code copy, run dir and truth-sentence setting, so the engine settings are the frozen ones (greedy, batch 8, 48 new
# tokens, target cap 24, seed 42, positions content). No judging here (crow_judge.sh).
# Order: the first layer alone, then the sign check crow_sign.py on its first 16 tune questions (caution must not answer
# more, answering must not answer fewer; a violation stops the model), then the other layers in one call.
# Cards 1 or 2 only, through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock, retried every minute); ours plus others
# at most 66560 MiB on each card used (checked before the start, as falseqa_model.sh). qwen3.8-27b-dm (device_map auto,
# as its runs of 11-13/14) takes both cards 1,2 and both locks.
# Markers: Iteration_01/logs/newsets_20261119/CROW_<model>_GEN_DONE / _GEN_FAILED; log crow_gen_<model>.log there.
# usage (host): crow_gen.sh KEY        KEY = llama | q9 | q27 | g12 | g27   (card: CROW_CARD=1|2, default 1; q27: 1,2)
cd /data/sanshizhao/Truth_Decoupling || exit 2
W=/workspace/Truth_Decoupling
R=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=${1:-}
CARD=${CROW_CARD:-1}
RUNNER=$R/Iteration_01/code/scripts/review_checks/newmodel_e44.sh
ARMS="x.w.p0.25 x.w.m0.25"
G3=$W/Iteration_01/results/ff/gemma3
case $KEY in
  llama) MODEL=llama3.1-8b; NEED=20000
         V=(MODEL=llama3.1-8b RUN_DIR=$W/results_e44_factorial_das/llama3.1-8b SOURCE_DIR=$W/results_qa_das_v1u/llama3.1-8b/full
            CODE_PATH=$W/code_e45_20261112 E_ON=E1 BRANCH=C)
         GEN=results_e44_factorial_das/llama3.1-8b/gen
         LAYERS="$(seq 0 18 | tr '\n' ' ')$(seq 20 30 | tr '\n' ' ')" ;;
  q9)    MODEL=qwen3.5-9b; NEED=22000
         V=(MODEL=qwen3.5-9b CODE_PATH=$W/code_ff_explore_20261113 E_ON=E1 BRANCH=C)
         GEN=results_newmodels_ff_20261112/e44/qwen3.5-9b/gen
         LAYERS="$(seq 0 11 | tr '\n' ' ')13 14 15 $(seq 18 31 | tr '\n' ' ')" ;;
  q27)   MODEL=qwen3.8-27b-dm; NEED=34000; CARD=1,2
         V=(MODEL=qwen3.8-27b-dm CODE_PATH=$W/code_ff_explore_20261113b E_ON=E2 BRANCH=C)
         GEN=results_newmodels_ff_20261112/e44/qwen3.8-27b-dm/gen
         LAYERS="19 21 23 25 27 29 31" ;;
  g12)   MODEL=gemma3-12b; NEED=30000
         V=(MODEL=gemma3-12b CODE_PATH=$W/Iteration_01/code FF_ROOT=$G3/12b E_ON=E2 BRANCH=C)
         GEN=Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b/gen
         LAYERS="15 17 19 23" ;;
  g27)   MODEL=gemma3-27b; NEED=62000; RUNNER=$R/Iteration_01/launchers/newmodel_e44_expseg.sh
         V=(MODEL=gemma3-27b CODE_PATH=$W/Iteration_01/code FF_ROOT=$G3/27b E_ON=E2 BRANCH=C)
         GEN=Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b/gen
         LAYERS="19 21 23 27 29 35 37" ;;
  *) echo "usage: crow_gen.sh llama|q9|q27|g12|g27"; exit 2 ;;
esac
case $CARD in 1|2|1,2) ;; *) echo "cards 1 or 2 only"; exit 2 ;; esac
MK=$L/CROW_${MODEL}_GEN; log=$L/crow_gen_$MODEL.log
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') crow gen $MODEL: $*" | tee -a $L/crow_all.log >> $log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
[ -e ${MK}_DONE ] && { say "already done (${MK}_DONE)"; exit 0; }
rm -f ${MK}_FAILED
[ -f "$RUNNER" ] || fail "no runner $RUNNER"
[ -f Iteration_01/diag/crow_sign.py ] || fail "no Iteration_01/diag/crow_sign.py"
for l in $LAYERS; do
  d=$(dirname $GEN)/dirs/C_L$l.pt
  [ -f $d ] || fail "no C direction record $d"
done
FIRST=${LAYERS%% *}; REST=${LAYERS#* }
say "layers [$LAYERS] arms [$ARMS] cards $CARD; runner $(sha256sum $RUNNER | cut -c1-16)"
fd=8
for g in ${CARD//,/ }; do
  LOCK=Iteration_01/newsets/selfjudge/.gpu$g.lock
  eval "exec $fd>\"$LOCK\"" || fail "cannot open $LOCK"
  until flock -n $fd; do sleep 60; done
  say "holds the lock of card $g"
  fd=$((fd + 1))
done
for g in ${CARD//,/ }; do
  while true; do
    USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$g '$1 == g {print $2}')
    [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $g"
    [ $((USED + NEED)) -le 66560 ] && break
    say "GPU $g holds $USED MiB, $NEED more would exceed 66560; waiting"; sleep 60
  done
  say "GPU $g: $USED MiB in use at the start"
done
run() {  # run LAYERS
  env "${V[@]}" STAGE=extra GPU=$CARD LAYERS="$1" ARMS="$ARMS" bash $RUNNER >> $log 2>&1
}
say "first layer $FIRST"
run "$FIRST" || fail "generation of layer $FIRST (see $log)"
python3 -B Iteration_01/diag/crow_sign.py $GEN $FIRST 16 >> $log 2>&1
rc=$?
tail -n 2 $log | sed "s/^/  /" >> $L/crow_all.log
[ $rc = 0 ] || fail "sign check of layer $FIRST (rc $rc)"
if [ "$REST" != "$LAYERS" ] && [ -n "$REST" ]; then
  say "other layers [$REST]"
  run "$REST" || fail "generation of layers $REST (see $log)"
fi
for l in $LAYERS; do
  for a in p0-25 m0-25; do [ -s $GEN/C_L${l}_x-w-${a}_tune.json ] || fail "missing $GEN/C_L${l}_x-w-${a}_tune.json"; done
done
say "done; cards released"
touch ${MK}_DONE
