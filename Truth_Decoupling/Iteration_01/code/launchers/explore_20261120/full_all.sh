#!/bin/bash
# MMLU full test split (14,042 items) by option probability: no write, E 1, E 2, C = 2; five models. Confirmation of
# section 8.4 of the capability design (predictions committed before this run). GPU 0 carries another user's idle
# process (4.7 GB): only the three small models go there; the two 27B models use GPU 1 and GPUs 2,3.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
TF=/workspace/Truth_Decoupling/envs/tf519/bin/python
[ -s Iteration_01/capability/FULL_REGISTERED ] || { echo "predictions not registered"; exit 1; }
one() {
  rm -f Iteration_01/capability/full/$1.json
  docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e HF_HOME=/workspace/Truth_Decoupling/hf_cache \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=$3 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -w /workspace/Truth_Decoupling truth_dec \
    $2 -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model $1 --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_14042_20261120.json \
    --out Iteration_01/capability/full/$1.json --arms x "x:E#.w.p1" "x:E#.w.p2" "x:C#.w.m2" > $L/full_$1.log 2>&1
  echo "full $1 exit=$? $(date '+%T')" >> $L/full_all.log
}
rm -f $L/full_all.log $L/FULL_DONE
mkdir -p Iteration_01/capability/full
( one llama3.1-8b python3 0; one gemma3-12b python3 0; one qwen3.5-9b $TF 0 ) &
( one qwen3.8-27b $TF 2,3 ) &
( until [ -e $L/SWEEP_G27_DONE ]; do sleep 30; done; one gemma3-27b python3 1 ) &
wait
touch $L/FULL_DONE
