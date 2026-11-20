#!/bin/bash
# MMLU option-probability sweep of write variants (dose, reversed sign, random axis, mean difference, the two parts of
# the learned write, write position), five models. GPU 0 is not used (another user's process sits there).
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
TF=/workspace/Truth_Decoupling/envs/tf519/bin/python
one() {
  rm -f Iteration_01/capability/sweep/$1.json
  docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e HF_HOME=/workspace/Truth_Decoupling/hf_cache \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=$3 -w /workspace/Truth_Decoupling truth_dec \
    $2 -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model $1 --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_1000_20261119.json \
    --out Iteration_01/capability/sweep/$1.json --ref Iteration_01/capability/mcq/$1.json \
    --arms x "x:E#.w.p2" "x:C#.w.m2" "x:E#.w.p0.5" "x:E#.w.p1" "x:E#.w.p1.5" "x:E#.w.m2" "x:E#.rand.p2" "x:E#.mdg.p2" \
           "x:E#.wpar.p2" "x:E#.wperp.p2" "x:E#.w.p2@gen" "x:E#.w.p1@gen" > $L/sweep_$1.log 2>&1
  echo "sweep $1 exit=$? $(date '+%T')" >> $L/sweep_all.log
}
rm -f $L/sweep_all.log $L/SWEEP_DONE
mkdir -p Iteration_01/capability/sweep
( one gemma3-12b python3 1; one llama3.1-8b python3 1; one qwen3.5-9b $TF 1; one gemma3-27b python3 1 ) &
( one qwen3.8-27b $TF 2,3 ) &
wait
touch $L/SWEEP_DONE
