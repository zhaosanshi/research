#!/bin/bash
# 2026-11-21 03:0x, exploration: the MMLU layer map of Gemma-3-12B on the 1,000-item diagnostic list, E at layers
# 18 / 20 / 22 / 24, strengths 1 and 2 (capability/layermap/gemma3-12b.json), as done for Llama and Gemma-3-27B: is
# there a layer near the one chosen for TruthfulQA (22) where the E write costs less MMLU? On the tune split (AllenAI,
# strength 1) single E scores +55 at layer 18, +48 at 20, +54 at 22, +49 at 24 (no write +29).
# usage: g12map.sh GPUS
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
GPUS=$1
[ -n "$GPUS" ] || { echo "usage: g12map.sh GPUS"; exit 2; }
PY=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["models"]["gemma3-12b"]["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json) || exit 2
mkdir -p Iteration_01/capability/layermap; rm -f Iteration_01/capability/layermap/gemma3-12b.json
docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_VISIBLE_DEVICES=$GPUS -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $ROOT truth_dec \
  $PY -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
  --model gemma3-12b --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_1000_20261119.json \
  --out Iteration_01/capability/layermap/gemma3-12b.json \
  --arms x x:E22.w.p1 x:E22.w.p2 x:E24.w.p1 x:E24.w.p2 x:E20.w.p1 x:E20.w.p2 x:E18.w.p1 x:E18.w.p2 > $L/layermap_g12.log 2>&1
echo "layer map gemma3-12b exit=$? $(date '+%T')" >> $L/ns2_all.log
