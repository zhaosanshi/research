#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
S=code/experiments/scripts/run_e14_model.sh
for spec in "llama3.1-8b 14 15 16 14 18" "qwen3-8b 20 21 22 21 24" "mistral-7b 15 16 17 18 19" "qwen3-4b 20 21 22 21 24" "qwen3-1.7b 15 16 17 16 19" "qwen3-0.6b 14 15 16 15 18"; do
  set -- $spec
  bash $S 1 $1 $2 $3 $4 $5 $6 bbq_mc bbq_tf E2b_bbq E1_directions_bbq results_bbq > logs/e14bbq_$1.log 2>&1
  stamp "e14 bbq $1 rc=$?"
done
bash $S 1 qwen3-0.6b 14 15 16 15 18 > logs/e14_qwen3-0.6b.log 2>&1; stamp "e14 tqa 0.6b rc=$?"
dex 1 results_tqa python3 scripts/e14_mc_eval.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b --e-layers 12 13 14 --probe-layer 14 --write-layer 16 --conditions transfer --tag transfer > logs/e14_llama_transfer.log 2>&1; stamp "e14 llama transfer rc=$?"
stamp "QUEUE GPU1 DONE"
