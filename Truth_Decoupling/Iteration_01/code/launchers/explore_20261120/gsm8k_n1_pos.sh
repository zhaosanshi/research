#!/bin/bash
# 2026-11-21 23:3x, exploration: the re-asks that the full N1 needs on GSM8K at the write rule "prompt". For each model:
# the items blank under single E at "prompt" (diag/gsm8k_n1_prompt.py ids), re-asked with E + C at 0.25 / 0.5 / 1 / 2,
# both writes at "prompt" (scripts/newsets/ns_positions.py with NS_POS_TWO_WRITE=1; only those items, so their batches
# differ from a full pass, as recorded in the files). At most a handful of items per model (GPT's count: 2, 4, 1, 0, 1).
# usage: gsm8k_n1_pos.sh GPUS MODEL [GPUS MODEL ...]      table: diag/gsm8k_n1_prompt.py table (in the container)
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
NSP=$ROOT/Iteration_01/code/scripts/newsets/ns_positions.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
(( $# > 0 && $# % 2 == 0 )) || exit 2
while [ $# -ge 2 ]; do
  GPUS=$1; MODEL=$2; shift 2
  plan=$(python3 -B Iteration_01/diag/gsm8k_n1_prompt.py ids "$MODEL") || exit 1
  read -r N EL S CL CAP DIR <<< "$plan"
  [[ "$N" =~ ^[0-9]+$ && -n "$DIR" ]] || exit 1
  echo "$(date '+%F %T') gsm8k n1 $MODEL: $N items blank under E at prompt (cap $CAP)" >> $L/gsm8k_all.log
  [ "$N" -gt 0 ] || continue
  PY=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)
  arms=""; for k in 0.25 0.5 1 2; do arms="$arms neutral:E$EL.w.p$S+C$CL.w.m$k"; done
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec \
    $PY -I $NSP prompt gen --models $M --model $MODEL --task gsm8k --items $ROOT/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet \
    --ids $ROOT/Iteration_01/capability/sets/gsm8k_1319_20261119.json --only-ids $ROOT/$DIR/n1_blank_ids_at-prompt.json \
    --max-new-tokens $CAP --batch-size 8 --out $ROOT/$DIR --arms $arms >> $L/gsm8k_n1_$MODEL.log 2>&1
  rc=$?
  echo "gsm8k n1 $MODEL exit=$rc $(date '+%T')" >> $L/gsm8k_all.log
  [ "$rc" -eq 0 ] || exit "$rc"
done
