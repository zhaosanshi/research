#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
stamp "e4b llama tqa start"
dex 3 results_tqa python3 scripts/e4b_qa_gated.py --phase generate --regime tqa --model llama3.1-8b --force --batch-size 16 \
  --conditions baseline cautious_prompt E_a1 E_a2 E_a3 E_p5 C_abst_a1 E_a2+C_abst_a1 gate_disagree_a3 gate_disagree_abst_a3 gate_probeneg_a3 gate_disagree_p5 > logs/e4b_tqa_llama3.1-8b.log 2>&1
stamp "e4b llama tqa rc=$?"
until [ -f results_tqa/E16/llama3.1-8b/iti_heads.npz ]; do sleep 30; done
dex 3 results_tqa python3 scripts/e4b_qa_gated.py --phase generate --regime tqa --model llama3.1-8b --batch-size 16 --conditions iti_k48_a5 iti_k48_a15 >> logs/e4b_tqa_llama3.1-8b.log 2>&1
stamp "e4b llama tqa iti rc=$?"
stamp "e4b llama bbq start"
dex 3 results_bbq python3 scripts/e4b_qa_gated.py --phase generate --regime bbq --model llama3.1-8b --force --batch-size 16 \
  --conditions baseline cautious_prompt E_a1 E_a2 E_a3 E_p5 C_abst_a1 E_a2+C_abst_a1 gate_disagree_a3 gate_disagree_abst_a3 gate_probeneg_a3 gate_disagree_p5 > logs/e4b_bbq_llama3.1-8b.log 2>&1
stamp "e4b llama bbq rc=$?"
stamp "e4b qwen3-8b bbq start"
dex 3 results_bbq python3 scripts/e4b_qa_gated.py --phase generate --regime bbq --model qwen3-8b --force --batch-size 16 \
  --conditions baseline cautious_prompt E_a1 E_a2 E_a3 E_p5 C_abst_a1 E_a2+C_abst_a1 gate_disagree_a3 gate_disagree_abst_a3 gate_probeneg_a3 gate_disagree_p5 > logs/e4b_bbq_qwen3-8b.log 2>&1
stamp "e4b qwen3-8b bbq rc=$?"
stamp "QUEUE GPU3 DONE"
