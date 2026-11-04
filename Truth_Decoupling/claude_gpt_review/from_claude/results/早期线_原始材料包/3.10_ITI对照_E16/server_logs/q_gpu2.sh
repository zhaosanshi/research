#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
until grep -q "^wrote" logs/e16_tqa_llama3.1-8b.log; do sleep 30; done
for m in qwen3-8b mistral-7b qwen3-4b qwen3-1.7b qwen3-0.6b; do
  stamp "e16 tqa $m start"
  dex 2 results_tqa python3 scripts/e16_iti_baseline.py --exp E2b_tqa --e1-exp E1_directions_tqa --model $m --k 48 --alphas 5 10 15 20 --mc > logs/e16_tqa_$m.log 2>&1
  stamp "e16 tqa $m rc=$?"
done
dex 2 results_tqa python3 scripts/e16_iti_baseline.py --exp E2b_tqa --e1-exp E1_directions_tqa --model llama3.1-8b --k 48 --alphas 5 10 15 20 --mc --train-on-half-a --tag halfA > logs/e16_tqa_llama_halfA.log 2>&1; stamp "e16 llama halfA rc=$?"
for m in llama3.1-8b qwen3-8b; do
  dex 2 results_bbq python3 scripts/e16_iti_baseline.py --exp E2b_bbq --e1-exp E1_directions_bbq --model $m --k 48 --alphas 5 10 15 20 --mc --mc-dataset bbq_mc --exclude-dataset bbq_tf > logs/e16_bbq_$m.log 2>&1; stamp "e16 bbq $m rc=$?"
done
stamp "QUEUE GPU2 DONE"
