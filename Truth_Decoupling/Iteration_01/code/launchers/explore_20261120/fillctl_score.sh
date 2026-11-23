#!/usr/bin/env bash
# 2026-11-23, queue L section 9 (executor A): the GPU step of diag/fill_only2_score.py ("score": selfjudge_two_answers.score1,
# batch size 1, on the sources fo_items / fo_chk / fo_chka / fo_chkb exported by diag/fill_only2.py export) for one model on
# one card. selfjudge_stack_pick_1122.sh with three changes: the scoring entry (fill_only2_score.py, sources fo_*), cards 1
# or 2 only with ours plus others at most 66560 MiB on the card, and no wait on the old card markers of 11-22 (the lock and
# the memory rule only). Code copy, interpreter, E45 regression and model configs as that launcher.
# Output only under Iteration_01/newsets/selfjudge/<MODEL>/fo_*; log Iteration_01/logs/newsets_20261119/fo_<MODEL>.log.
#   usage (host): fillctl_score.sh MODEL GPU TAG
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
MODEL=${1:?model}; GPU=${2:?gpu}; TAG=${3:?tag}; SRCS=fo_items,fo_chk,fo_chka,fo_chkb
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=$HOST/Iteration_01/logs/newsets_20261119
MK=$L/FO_${MODEL}_$TAG; LOG=$L/fo_$MODEL.log
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') fo $MODEL $TAG: $*" | tee -a $LOG; }
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
  RL=$L/fo_${MODEL}_${TAG}_regression.log
  "${DX[@]}" Iteration_01/diag/selfjudge_probe.py score "$MODEL" --regress-only > $RL 2>&1; rc=$?
  cat $RL >> $LOG
  [ $rc = 0 ] || fail "E45 regression"
  grep -q "REGRESSION OK" $RL || fail "no REGRESSION OK line"
  say "E45 regression passed"
fi
"${DX[@]}" Iteration_01/diag/fill_only2_score.py score "$MODEL" >> $LOG 2>&1 || fail "score $SRCS"
flock -u 9; exec 9>&-
say "done; card $GPU released"
touch ${MK}_DONE
