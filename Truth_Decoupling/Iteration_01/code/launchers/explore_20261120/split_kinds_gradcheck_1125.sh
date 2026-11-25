#!/bin/bash
# Iteration_01 (main session, Claude Fable 5.2, 2026-11-25 23 h; plan discussion/work_queue/2026-11-24_queue_m.md,
# progress entry of 2026-11-25 23:5x, whose reading rule was committed before this script ran): gradient agreement of
# the two ways of back-propagating a training question of gemma3-27b (all rows in one pass / cause rows and isolate rows
# one after the other, split_kinds_1125.patch), measured on the real model at a fixed direction.
# For each layer of LAYERS (default "31 18"; both fit on one card either way) the patched trainer is run twice for one
# epoch with a learning rate of 1e-12 (the direction stays at its seeded random start), once without and once with
# --split-kinds, each into a scratch run directory of the split-kinds results root that holds a copy of that root's
# frozen train answers. Iteration_01/diag/split_kinds_grad_check.py listens to the gradient of every step and compares
# the two runs step by step. Nothing existing is written to: the scratch directories are
# results/ff/gemma3/27b_splitkinds/e44/gradcheck_{single,split}_L<layer>.
# Card CARD (default 0) through its lock file; starts only when ours plus others stay within 66560 MiB.
# Result: Iteration_01/diag/out_split_kinds_gradcheck_gemma3-27b_<date>.txt; markers
# CROW2_gemma3-27b_SK_GRADCHECK_DONE / _FAILED; status lines "sk gradcheck: ..." in crow_all.log.
# SK_DRY=1: every card command runs once without a card under timeout 90 (expected: "No CUDA GPUs are available").
# SK_TAG=<suffix> (added 2026-11-26 00 h): a second run under other names (scratch directories gradcheck<suffix>_*,
# result file and markers with the suffix); after each layer the batch widths recorded by the listening program are
# read too (split_kinds_grad_check.py widths).
# usage (host): ( nohup setsid bash Iteration_01/logs/newsets_20261119/split_kinds_gradcheck_1125.sh \
#                 > Iteration_01/logs/newsets_20261119/split_kinds_gradcheck_1125.out 2>&1 < /dev/null & )
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
export TZ=Asia/Shanghai
W=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
MODEL=gemma3-27b
CARD=${SK_CARD:-0}
DRY=${SK_DRY:-0}
LAYERS=${SK_LAYERS:-31 18}
TAG=${SK_TAG:-}
case "$TAG" in ""|_[a-z0-9]*) ;; *) echo "SK_TAG must be empty or _name"; exit 2 ;; esac
CODE=Iteration_01/code_splitkinds_20261125
NEWR=Iteration_01/results/ff/gemma3/27b_splitkinds
SRC=$NEWR/source/$MODEL
RUN=$NEWR/e44/$MODEL
PROG=Iteration_01/diag/split_kinds_grad_check.py
MK=$L/CROW2_${MODEL}_SK_GRADCHECK$TAG
OUT=Iteration_01/diag/out_split_kinds_gradcheck${TAG}_${MODEL}_$(date +%Y%m%d).txt
log=$L/split_kinds_gradcheck${TAG}_$MODEL.log
say() { echo "$(date '+%F %T') sk gradcheck: $*" >> $L/crow_all.log; }
fail() { say "FAILED: $*"; [ "$DRY" = 1 ] || touch ${MK}_FAILED; exit 1; }
case "$CARD" in 0|1|2|3) ;; *) echo "SK_CARD must be one card"; exit 2 ;; esac
for f in $PROG $CODE/scripts/e44_factorial_das.py $SRC/bank.json $RUN/frozen_train.json; do [ -s $f ] || { echo "missing $f"; exit 2; }; done
[ "$(sha256sum $CODE/core/qa_factorial_das.py | cut -c1-16)" = 77a6ed37a0ff2478 ] || { echo "the code copy is not the patched one"; exit 2; }
if [ "$DRY" = 1 ]; then CVD=""; TO=(timeout 90); else
  mkdir $L/.sk_gradcheck${TAG}_running 2>/dev/null || { echo "already running ($L/.sk_gradcheck${TAG}_running)"; exit 3; }
  trap 'rmdir $L/.sk_gradcheck${TAG}_running 2>/dev/null' EXIT
  rm -f ${MK}_DONE ${MK}_FAILED
  exec 8>Iteration_01/newsets/selfjudge/.gpu$CARD.lock || fail "cannot open the lock of card $CARD"
  until flock -n 8; do sleep 60; done
  USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$CARD '$1 == g {print $2}')
  [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $CARD"
  [ $((USED + 66560)) -le 66560 ] || fail "GPU $CARD holds $USED MiB: not an empty card"
  say "holds the lock of card $CARD ($USED MiB in use); layers $LAYERS"
  CVD=$CARD; TO=()
  : > $OUT
fi
DX=(docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models
    -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES="$CVD" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$W/$CODE -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -w $W/$CODE truth_dec)
COMMON=(--source-run $W/$SRC --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
for l in $LAYERS; do
  for mode in single split; do
    D=$NEWR/e44/gradcheck${TAG}_${mode}_L$l
    G=$D/step_gradients.pt
    if [ "$DRY" != 1 ]; then
      [ -e $D ] && fail "scratch directory $D exists already"
      mkdir -p $D && cp -p $RUN/frozen_train.json $D/frozen_train.json || fail "cannot prepare $D"
    fi
    SW=(); [ $mode = split ] && SW=(--split-kinds)
    echo "=== $(date '+%F %T') layer $l $mode ===" >> $log
    "${TO[@]}" "${DX[@]}" python3 -B $W/$PROG record $W/$G -- train "${COMMON[@]}" --e-on E2 --run-dir $W/$D --family C \
        --layers $l --val-n 16 --epochs 1 --lr 1e-12 "${SW[@]}" >> $log 2>&1
    rc=$?
    if [ "$DRY" = 1 ]; then echo "dry layer $l $mode rc $rc: $(tail -n 1 $log | cut -c1-160)"; continue; fi
    [ $rc = 0 ] && [ -s $G ] || fail "layer $l $mode: rc $rc (see $log)"
    say "layer $l $mode: $(tail -n 1 $log | cut -c1-80)"
  done
  [ "$DRY" = 1 ] && continue
  docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec python3 -B $W/$PROG compare \
      $W/$NEWR/e44/gradcheck${TAG}_single_L$l/step_gradients.pt $W/$NEWR/e44/gradcheck${TAG}_split_L$l/step_gradients.pt \
      "layer $l, first = one pass on all rows, second = by kind" >> $OUT 2>> $log || fail "comparison of layer $l"
  if [ -s $NEWR/e44/gradcheck${TAG}_single_L$l/step_gradients.pt.packs.json ]; then
    docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec python3 -B $W/$PROG widths \
        $W/$NEWR/e44/gradcheck${TAG}_single_L$l/step_gradients.pt $W/$NEWR/e44/gradcheck${TAG}_split_L$l/step_gradients.pt \
        "layer $l, batch widths" >> $OUT 2>> $log || fail "batch widths of layer $l"
  fi
done
[ "$DRY" = 1 ] && exit 0
say "done: $OUT"
touch ${MK}_DONE
