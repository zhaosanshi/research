#!/bin/bash
# 2026-11-21 12:0x, exploration: GSM8K (the 1,319 frozen test items, the generator settings of the capability design)
# with the E write at another position rule than the training rule, through scripts/newsets/ns_positions.py
# (newsets_gen.py unchanged; the output file name carries "-at-<POS>", the contract records "positions").
# Why: at the main-table settings the training rule (every generated token written) costs 19 points on Llama-3.1-8B
# and 14 on Gemma-3-12B; the rule "prompt" writes the question tokens and the assistant header only.
# The no-write pass is the one already in the directory (gsm8k_main.sh / gsm8k_cap.sh).
# usage: [NEWTOK=1024] gsm8k_pos.sh GPUS POS MODEL ARM [MODEL ARM ...]    e.g. gsm8k_pos.sh 1 prompt llama3.1-8b neutral:E16.w.p3.5
# table: python3 -B Iteration_01/diag/gsm8k_main_table.py [--cap N]   (a pass at another rule is shown as ARM@POS)
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
NSP=$ROOT/Iteration_01/code/scripts/newsets/ns_positions.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
GPUS=$1; POS=$2; shift 2
[ -n "$GPUS" ] && [ -n "$POS" ] && [ $# -ge 2 ] || { echo "usage: [NEWTOK=1024] gsm8k_pos.sh GPUS POS MODEL ARM [MODEL ARM ...]"; exit 2; }
NEWTOK=${NEWTOK:-512}; OUTTAG=gen_main; [ "$NEWTOK" = 512 ] || OUTTAG=gen_main$NEWTOK
while [ $# -ge 2 ]; do
  MODEL=$1; ARM=$2; shift 2
  PY=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || { echo "gsm8k $MODEL: not in the config" >> $L/gsm8k_all.log; continue; }
  [ -e $HOST/Iteration_01/capability/$OUTTAG/gsm8k/$MODEL/b8/neutral.json ] || { echo "gsm8k $MODEL: no no-write pass at cap $NEWTOK" >> $L/gsm8k_all.log; continue; }
  echo "$(date '+%F %T') gsm8k $MODEL start on GPU $GPUS (cap $NEWTOK, write at $POS): $ARM" >> $L/gsm8k_all.log
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec \
    $PY -I $NSP $POS gen --models $M --model $MODEL --task gsm8k --items $ROOT/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet \
    --ids $ROOT/Iteration_01/capability/sets/gsm8k_1319_20261119.json --max-new-tokens $NEWTOK --batch-size 8 \
    --out $ROOT/Iteration_01/capability/$OUTTAG/gsm8k/$MODEL/b8 --arms $ARM >> $L/gsm8k_${MODEL}_${NEWTOK}_$POS.log 2>&1
  echo "gsm8k $MODEL cap $NEWTOK at $POS exit=$? $(date '+%T')" >> $L/gsm8k_all.log
done
touch $L/GSM8K_POS_GPU${GPUS//,/_}_${NEWTOK}_DONE
