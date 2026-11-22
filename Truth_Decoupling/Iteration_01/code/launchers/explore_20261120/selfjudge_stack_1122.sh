#!/usr/bin/env bash
# Work queue 2026-11-22_queue_j section 6 (Claude Opus 5.5): the GPU step of diag/selfjudge_stack.py ("score":
# selfjudge_two_answers.score1, batch size 1, on the sources st1_items / st1_chk / st1_chka / st1_chkb) for one model on one
# card, in the container, with the code copy and interpreter E45 ran with (as selfjudge_1122.sh / selfjudge_gpu0_1122.sh):
#   llama3.1-8b code_e45_20261112; qwen3.5-9b, qwen3.8-27b, gemma3-12b code_newmodels_20261112 (the copy's model config);
#   gemma3-27b code_newmodels_20261112 with Iteration_01/newsets/selfjudge/_configs/gemma3-27b.json
#   (repository: code/configs/selfjudge/).
# Steps (any failure: marker TJ_<MODEL>_<TAG>_FAILED, exit 1):
#   1. models with a true/false E: the full E45 regression (selfjudge_probe.py score MODEL --regress-only, every
#      |margin difference| <= 0.02 on E45's 1,442 statements, four conditions); the Gemma models: the stored first batch
#      of the no-write check is re-read inside step 2 (selfjudge_probe.score_model, as before);
#   2. selfjudge_stack.py score MODEL.
# Cards: waits (never takes a card from anyone): card 0 only after CAA_GRID_gemma3-12b_DONE or _FAILED; card 2 only after
# ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE; card 1 only llama3.1-8b / qwen3.5-9b and at most 65 GB in use with this
# model; then the lock Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock, queued) and the memory rule; when the memory
# does not fit, the lock is released again and the check repeats every 60 s.
# Output only under Iteration_01/newsets/selfjudge/<MODEL>/st1_*; log Iteration_01/logs/newsets_20261119/st_<MODEL>.log.
#   usage (host): selfjudge_stack_1122.sh MODEL GPU TAG
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
MODEL=${1:?model}; GPU=${2:?gpu}; TAG=${3:?tag}; SRCS=st1_items,st1_chk,st1_chka,st1_chkb
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=$HOST/Iteration_01/logs/newsets_20261119
MK=$L/ST_${MODEL}_$TAG; LOG=$L/st_$MODEL.log
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') st $MODEL $TAG: $*" | tee -a $LOG; }
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
  0|2|3) LIM=73000 ;;
  1) LIM=66560; case "$MODEL" in llama3.1-8b|qwen3.5-9b) ;; *) fail "card 1 takes only the two small models" ;; esac ;;
  *) fail "unknown card $GPU" ;;
esac
[ -f "$HOST/$CFG" ] || fail "no model config $CFG"
for s in ${SRCS//,/ }; do [ -s "$HOST/Iteration_01/newsets/selfjudge/$MODEL/${s}_items.json" ] || fail "no ${s}_items.json"; done
command -v flock >/dev/null || fail "flock unavailable"
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu$GPU.lock
say "waiting for card $GPU (needs $NEED MiB, limit $LIM)"
while true; do
  if [ "$GPU" = 0 ] && [ ! -e $L/CAA_GRID_gemma3-12b_DONE ] && [ ! -e $L/CAA_GRID_gemma3-12b_FAILED ]; then sleep 60; continue; fi
  if [ "$GPU" = 2 ] && [ ! -e $L/ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE ]; then sleep 60; continue; fi
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
  RL=$L/st_${MODEL}_${TAG}_regression.log
  "${DX[@]}" Iteration_01/diag/selfjudge_probe.py score "$MODEL" --regress-only > $RL 2>&1; rc=$?
  cat $RL >> $LOG
  [ $rc = 0 ] || fail "E45 regression"
  grep -q "REGRESSION OK" $RL || fail "no REGRESSION OK line"
  say "E45 regression passed"
fi
"${DX[@]}" Iteration_01/diag/selfjudge_stack.py score "$MODEL" >> $LOG 2>&1 || fail "score $SRCS"
flock -u 9; exec 9>&-
say "done; card $GPU released"
touch ${MK}_DONE
