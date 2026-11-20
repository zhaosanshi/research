#!/bin/bash
# 2026-11-21 06:3x, exploration: GSM8K (1,319 test items, the frozen list of the capability design) under the settings of
# the main table - each model with the E write at its own layer and strength (diag/main_table_layers.py, budget 1) - as a
# second capability check next to MMLU. Full passes with the generator of the capability design (newsets_gen.py, system
# prompt "neutral", 512 new tokens, batch 8, as fixed there): no write, and single E at the given layer and strength.
# N1's re-asks are not run: they only touch items left blank under E, and the scorer reports how many there are.
# usage: gsm8k_main.sh GPUS MODEL ARM [MODEL ARM ...]     e.g. gsm8k_main.sh 0 llama3.1-8b neutral:E16.w.p3.5
#        (models run one after the other on the same card; answers in capability/gen_main/gsm8k/<model>/b8)
# table: python3 -B Iteration_01/diag/gsm8k_main_table.py
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
GPUS=$1; shift
[ -n "$GPUS" ] && [ $# -ge 2 ] || { echo "usage: gsm8k_main.sh GPUS MODEL ARM [MODEL ARM ...]"; exit 2; }
while [ $# -ge 2 ]; do
  MODEL=$1; ARM=$2; shift 2
  PY=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || { echo "gsm8k $MODEL: not in the config" >> $L/gsm8k_all.log; continue; }
  echo "$(date '+%F %T') gsm8k $MODEL start on GPU $GPUS: neutral $ARM" >> $L/gsm8k_all.log
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec \
    $PY -I $G gen --models $M --model $MODEL --task gsm8k --items $ROOT/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet \
    --ids $ROOT/Iteration_01/capability/sets/gsm8k_1319_20261119.json --max-new-tokens 512 --batch-size 8 \
    --out $ROOT/Iteration_01/capability/gen_main/gsm8k/$MODEL/b8 --arms neutral $ARM >> $L/gsm8k_$MODEL.log 2>&1
  echo "gsm8k $MODEL exit=$? $(date '+%T')" >> $L/gsm8k_all.log
done
touch $L/GSM8K_GPU${GPUS//,/_}_DONE
