#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
until grep -q "e14 rc=" logs/e14_qwen3-8b.log; do sleep 20; done
stamp "e4b qwen3-8b tqa start"
dex 0 results_tqa python3 scripts/e4b_qa_gated.py --phase generate --regime tqa --model qwen3-8b --force --batch-size 16 \
  --conditions baseline cautious_prompt E_a1 E_a2 E_a3 E_p5 C_abst_a1 E_a2+C_abst_a1 gate_disagree_a3 gate_disagree_abst_a3 gate_probeneg_a3 gate_disagree_p5 > logs/e4b_tqa_qwen3-8b.log 2>&1
stamp "e4b qwen3-8b tqa rc=$?"
until [ -f results_tqa/E16/qwen3-8b/iti_heads.npz ]; do sleep 30; done
dex 0 results_tqa python3 scripts/e4b_qa_gated.py --phase generate --regime tqa --model qwen3-8b --batch-size 16 --conditions iti_k48_a5 iti_k48_a15 >> logs/e4b_tqa_qwen3-8b.log 2>&1
stamp "e4b qwen3-8b tqa iti rc=$?"
stamp "QUEUE GPU0 DONE"
