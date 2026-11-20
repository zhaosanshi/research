#!/bin/bash
# Rerun of the Gemma-3-27B sweep (06:56 out of memory on one 72 GB card at the fp32 logits of a long batch, 67 GB in use
# with 5.5 GB reserved but unallocated): same call, with the allocator option the error message names.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
rm -f Iteration_01/capability/sweep/gemma3-27b.json $L/SWEEP_G27_DONE
docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e HF_HOME=/workspace/Truth_Decoupling/hf_cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -w /workspace/Truth_Decoupling truth_dec \
  python3 -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
  --model gemma3-27b --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_1000_20261119.json \
  --out Iteration_01/capability/sweep/gemma3-27b.json --ref Iteration_01/capability/mcq/gemma3-27b.json \
  --arms x "x:E#.w.p2" "x:C#.w.m2" "x:E#.w.p0.5" "x:E#.w.p1" "x:E#.w.p1.5" "x:E#.w.m2" "x:E#.rand.p2" "x:E#.mdg.p2" \
         "x:E#.wpar.p2" "x:E#.wperp.p2" "x:E#.w.p2@gen" "x:E#.w.p1@gen" > $L/sweep_gemma3-27b.log 2>&1
echo "sweep gemma3-27b (rerun) exit=$? $(date '+%T')" >> $L/sweep_all.log
touch $L/SWEEP_G27_DONE
