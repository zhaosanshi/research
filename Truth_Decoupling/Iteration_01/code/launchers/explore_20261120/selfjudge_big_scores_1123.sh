#!/usr/bin/env bash
# Work queue 2026-11-22_queue_j sections 6, 7, 7a (Claude Opus 5.5; coordinator's plan of 01:46): every batch-1
# self-judge score of the three larger models in ONE hold of one card's lock. Waits until the lock of card 2 or card 3
# is free (flock -n every 30 s; never takes a held lock, never stops anyone), keeps it until all three models are done,
# then touches TJ_BIG_SCORES_DONE. Per model, in the container on that card (code copy / config / interpreter as
# selfjudge_stack_1122.sh): the full E45 regression (Qwen3.8-27B; the Gemma models re-read their stored first batch
# inside the scoring), selfjudge_stack.py score (section 6), selfjudge_stack_pick.py export (no GPU), selfjudge_stack_pick.py
# score (section 7). Before each model the card must have room (the memory rule; checked every 60 s while holding the
# lock). Any failure: TJ_BIG_SCORES_FAILED, exit 1.
#   usage (host): selfjudge_big_scores_1123.sh
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=$HOST/Iteration_01/logs/newsets_20261119; LK=$HOST/Iteration_01/newsets/selfjudge
LOG=$L/tj_big.log
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') tj big: $*" | tee -a $LOG; }
fail() { say "FAILED: $*"; touch $L/TJ_BIG_SCORES_FAILED; exit 1; }
rm -f $L/TJ_BIG_SCORES_FAILED
say "waiting for the lock of card 2 or 3"
G=""
while [ -z "$G" ]; do
  for g in 2 3; do
    exec 9>"$LK/.gpu$g.lock" || fail "cannot open lock $g"
    if flock -n 9; then G=$g; break; fi
    exec 9>&-
  done
  [ -z "$G" ] && sleep 30
done
say "holding the lock of card $G"
CODE=code_newmodels_20261112
for MODEL in qwen3.8-27b gemma3-27b gemma3-12b; do
  case $MODEL in
    qwen3.8-27b) NEED=60000; CFG=$CODE/configs/models/$MODEL.json; REG=1 ;;
    gemma3-27b)  NEED=60000; CFG=Iteration_01/newsets/selfjudge/_configs/$MODEL.json; REG=0 ;;
    gemma3-12b)  NEED=28000; CFG=$CODE/configs/models/$MODEL.json; REG=0 ;;
  esac
  while true; do
    USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$G '$1 == g {print $2}')
    [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of card $G"
    [ $((USED + NEED)) -le 73000 ] && break
    say "card $G holds $USED MiB; $MODEL needs $NEED more; waiting"
    sleep 60
  done
  say "$MODEL start on card $G ($USED MiB in use)"
  PY=$(docker exec truth_dec python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$ROOT/$CFG") || fail "interpreter"
  DX=(docker exec -u 1000:1001 -e "HOME=$ROOT/hf_cache" -e "HF_HOME=$ROOT/hf_cache" -e "TD_DATA_DIR=$ROOT/code/data/processed"
      -e "TD_MODEL_ROOT=$ROOT/models" -e "TD_RESULTS_DIR=$ROOT/Iteration_01/newsets/selfjudge/_e45_unused" -e CUDA_VISIBLE_DEVICES=$G
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=16
      -e "PYTHONPATH=$ROOT/$CODE" -w "$ROOT" truth_dec "$PY" -B)
  DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w "$ROOT" truth_dec python3 -B)
  if [ "$REG" = 1 ]; then
    RL=$L/tj_big_${MODEL}_regression.log
    "${DX[@]}" Iteration_01/diag/selfjudge_probe.py score "$MODEL" --regress-only > $RL 2>&1 || fail "$MODEL E45 regression"
    grep -q "REGRESSION OK" $RL || fail "$MODEL no REGRESSION OK line"
    say "$MODEL E45 regression passed"
  fi
  "${DX[@]}" Iteration_01/diag/selfjudge_stack.py score "$MODEL" >> $LOG 2>&1 || fail "$MODEL section-6 score"
  say "$MODEL section 6 scored"
  if [ ! -s $LK/$MODEL/sp_items_items.json ]; then
    "${DC[@]}" Iteration_01/diag/selfjudge_stack_pick.py export "$MODEL" >> $LOG 2>&1 || fail "$MODEL section-7 export"
  fi
  "${DX[@]}" Iteration_01/diag/selfjudge_stack_pick.py score "$MODEL" >> $LOG 2>&1 || fail "$MODEL section-7 score"
  say "$MODEL section 7 scored"
done
flock -u 9; exec 9>&-
say "all three models done; lock of card $G released"
touch $L/TJ_BIG_SCORES_DONE
