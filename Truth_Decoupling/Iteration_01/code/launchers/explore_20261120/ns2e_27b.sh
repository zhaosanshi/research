#!/bin/bash
# 2026-11-21 01:2x, replaces the second lane of ns2b_all.sh (stopped while it was waiting; the first lane, Llama and
# Qwen3.5-9B on GPU 1, keeps running): the two 27B models get a shorter list. Reason: a 500-question pass of a 27B model
# takes several minutes and the full list is 47 passes per task; on Gemma-3-12B (six strengths) and on all five models
# at strength 2 the release step adds nothing, so for the 27B models only what the table needs is generated:
#   tune: single E and the release ladder at the five strengths (the choice of every setting needs them);
#   test: single E + ladder at the strengths named by diag/ns_choose.py from tune (function three(), unchanged).
# The cautious-C arms are left out for the 27B models (they can be added if the three smaller models call for it).
# The Gemma-3-12B step of the stopped lane (three gemma3-12b 3) is kept as it was.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
STRENGTHS="0.5 1 1.5 3 4"
one27() {  # one27 MODEL GPUS : tune only, single E + ladder at the five strengths
  local MODEL=$1 GPUS=$2 PY EL CL t s k arms
  read -r PY EL CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || { echo "ns2c $MODEL: not in the config" >> $L/ns2_all.log; return 1; }
  local DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
  arms=""
  for s in $STRENGTHS; do arms="$arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
  for t in popqa triviaqa; do
    "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_bank_20261119.jsonl --split tune \
        --out $NS/gen/$t/$MODEL/tune --arms $arms >> $L/ns2_$MODEL.log 2>&1
    echo "ns2c $MODEL $t tune exit=$? $(date '+%T')" >> $L/ns2_all.log
  done
  touch $L/NS2_${MODEL}_DONE
}
three() {  # three MODEL GPUS : test ladders at the tune-chosen strengths
  local MODEL=$1 GPUS=$2 PY EL CL t s k ss arms
  read -r PY EL CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || return 1
  local DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
  for t in popqa triviaqa; do
    ss=$(python3 -B Iteration_01/diag/ns_choose.py $t $MODEL 2>> $L/ns3_$MODEL.log | tail -1)
    case "$ss" in ""|*[!0-9.\ ]*) echo "ns3 $MODEL $t choose FAILED ($ss)" >> $L/ns2_all.log; continue ;; esac
    arms=""
    for s in $ss; do arms="$arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
    "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_bank_20261119.jsonl --split test \
        --out $NS/gen/$t/$MODEL/test --arms $arms >> $L/ns3_$MODEL.log 2>&1
    echo "ns3 $MODEL $t test ladders at E $ss exit=$? $(date '+%T')" >> $L/ns2_all.log
  done
  touch $L/NS3_${MODEL}_DONE
}
# 2026-11-21 01:4x, ns2d = ns2c_27b.sh with one step added before the 27B new-dataset passes (ns2c was stopped while
# waiting): the MMLU layer map of Gemma-3-27B on the 1,000-item diagnostic list, E at layers 24 / 26 / 28 / 30,
# strengths 1 and 2 (capability/layermap/gemma3-27b.json). Reason: on Llama the E write costs far less MMLU at layer 16
# than at the layer chosen for TruthfulQA; on the tune split (AllenAI, strength 1) layers 28 and 30 of Gemma-3-27B
# give about the same TruthfulQA gain as layer 26, so the map says whether one of them is cheaper.
g27map() {
  local PY
  PY=$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1]))["models"]["gemma3-27b"]["python"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json) || return 1
  mkdir -p Iteration_01/capability/layermap; rm -f Iteration_01/capability/layermap/gemma3-27b.json
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=2 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $ROOT truth_dec \
    $PY -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model gemma3-27b --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_1000_20261119.json \
    --out Iteration_01/capability/layermap/gemma3-27b.json \
    --arms x x:E26.w.p1 x:E26.w.p2 x:E28.w.p1 x:E28.w.p2 x:E30.w.p1 x:E30.w.p2 x:E24.w.p1 x:E24.w.p2 > $L/layermap_g27.log 2>&1
  echo "layer map gemma3-27b exit=$? $(date '+%T')" >> $L/ns2_all.log
}
# 2026-11-21 02:2x, ns2e = ns2d_27b.sh with one more step for Gemma-3-27B before the 27B new-dataset passes (ns2d was
# stopped while waiting; its Gemma-3-12B step had already finished): after the MMLU layer map, the same-strength
# strong comparator at E 0.5 / 0.75 (low_strong.sh g27 2; its GPU part runs here, its judging in the background).
echo "$(date '+%F %T') 27B lane (ns2e: layer map, low-strength strong comparator, then the short list) start" >> $L/ns2_all.log
until [ -e $L/LOW_GPU_g27_DONE ] && [ -e $L/NSCAUT_llama3.1-8b_DONE ]; do sleep 60; done
g27map
bash $L/low_strong.sh g27 2 > $L/low_strong_g27.out 2>&1
echo "low-strength strong comparator gemma3-27b (GPU part) exit=$? $(date '+%T')" >> $L/ns2_all.log
one27 qwen3.8-27b 2,3; three qwen3.8-27b 2,3; one27 gemma3-27b 2; three gemma3-27b 2
touch $L/NS2C_DONE
