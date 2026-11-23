#!/usr/bin/env bash
# 2026-11-23 23:4x, queue L section 11 (executor E, Claude Opus 5.5): batch-size-1 self-judge margins of every FalseQA
# answer a system can hand in, for one model on one card. fillctl_score.sh with three changes: the item lists come from
# diag/falseqa_read.py export-margins MODEL PHASE (CPU, run here before the card is taken; sources fqa0_* = the smoke
# rows of the 16-question check files, fqa1_* = the 700-question files), the scoring entry is diag/falseqa_score.py
# (selfjudge_two_answers.score1 unchanged), and falseqa_read.py check-margins (the 16 check statements scored twice in
# order and once shuffled must agree exactly, and with the main pass) runs after scoring. Code copy, interpreter, E45
# regression, model configs, cards (1 or 2, ours plus others at most 66560 MiB) and the lock as fillctl_score.sh.
# Output only under Iteration_01/newsets/selfjudge/<MODEL>/fqa*; log Iteration_01/logs/newsets_20261119/fqa_<MODEL>.log.
#   usage (host): falseqa_score.sh MODEL GPU smoke|full
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
MODEL=${1:?model}; GPU=${2:?gpu}; PHASE=${3:?phase}
case "$PHASE" in smoke) P=fqa0 ;; full) P=fqa1 ;; *) echo "phase smoke or full" >&2; exit 2 ;; esac
SRCS=${P}_items,${P}_chk,${P}_chka,${P}_chkb
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=$HOST/Iteration_01/logs/newsets_20261119
MK=$L/FQA_${MODEL}_$PHASE; LOG=$L/fqa_$MODEL.log
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') fqa $MODEL $PHASE: $*" | tee -a $LOG; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED ${MK}_DONE
CODE=code_newmodels_20261112
case "$MODEL" in
  llama3.1-8b) CODE=code_e45_20261112; NEED=17000; CFG=$CODE/configs/models/$MODEL.json; REG=1 ;;
  qwen3.5-9b)  NEED=20000; CFG=$CODE/configs/models/$MODEL.json; REG=1 ;;
  qwen3.8-27b) NEED=60000; CFG=$CODE/configs/models/$MODEL.json; REG=1 ;;
  gemma3-12b)  NEED=28000; CFG=$CODE/configs/models/$MODEL.json; REG=0 ;;
  gemma3-27b)  NEED=60000; CFG=Iteration_01/newsets/selfjudge/_configs/$MODEL.json; REG=0 ;;
  *) fail "unknown model" ;;
esac
case "$GPU" in
  1|2) LIM=66560 ;;
  *) fail "cards 1 or 2 only" ;;
esac
[ -f "$HOST/$CFG" ] || fail "no model config $CFG"
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B)
if [ ! -s "$HOST/Iteration_01/newsets/selfjudge/$MODEL/${P}_items_items.json" ]; then
  "${RD[@]}" Iteration_01/diag/falseqa_read.py export-margins $MODEL $PHASE >> $LOG 2>&1 || fail "export-margins"
fi
for s in ${SRCS//,/ }; do [ -s "$HOST/Iteration_01/newsets/selfjudge/$MODEL/${s}_items.json" ] || fail "no ${s}_items.json"; done
command -v flock >/dev/null || fail "flock unavailable"
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu$GPU.lock
say "waiting for card $GPU (needs $NEED MiB, limit $LIM)"
while true; do
  exec 9>"$LOCK" || fail "cannot open $LOCK"
  flock 9 || fail "flock"
  USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1 == g {print $2}')
  [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of card $GPU"
  [ $((USED + NEED)) -le $LIM ] && break
  flock -u 9; exec 9>&-
  sleep 60
done
say "start on card $GPU ($USED MiB in use), code $CODE, config $CFG, sources $SRCS"
PY=$(docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$ROOT/$CFG") || fail "interpreter"
DX=(docker exec -u 1000:1001 -e "HOME=$ROOT/hf_cache" -e "HF_HOME=$ROOT/hf_cache" -e "TD_DATA_DIR=$ROOT/code/data/processed"
    -e "TD_MODEL_ROOT=$ROOT/models" -e "TD_RESULTS_DIR=$ROOT/Iteration_01/newsets/selfjudge/_e45_unused" -e CUDA_VISIBLE_DEVICES=$GPU
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=16
    -e "PYTHONPATH=$ROOT/$CODE" -w "$ROOT" truth_dec "$PY" -B)
if [ "$REG" = 1 ]; then
  RL=$L/fqa_${MODEL}_${PHASE}_regression.log
  "${DX[@]}" Iteration_01/diag/selfjudge_probe.py score "$MODEL" --regress-only > $RL 2>&1; rc=$?
  cat $RL >> $LOG
  [ $rc = 0 ] || fail "E45 regression"
  grep -q "REGRESSION OK" $RL || fail "no REGRESSION OK line"
  say "E45 regression passed"
fi
"${DX[@]}" Iteration_01/diag/falseqa_score.py score "$MODEL" $PHASE >> $LOG 2>&1 || fail "score $SRCS"
flock -u 9; exec 9>&-
"${RD[@]}" Iteration_01/diag/falseqa_read.py check-margins $MODEL $PHASE >> $LOG 2>&1 || fail "check-margins"
say "done; card $GPU released"
touch ${MK}_DONE
