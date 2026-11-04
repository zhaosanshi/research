#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
J="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e CUDA_VISIBLE_DEVICES= -e PYTHONUNBUFFERED=1 -e DEEPSEEK_API_KEY_FILE=/workspace/Truth_Decoupling/logs/.deepseek_key -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed -w /workspace/Truth_Decoupling/code/experiments"
stamp "judge tqa llama start"
$J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_tqa truth_dec python3 scripts/e4b_qa_gated.py --phase judge --regime tqa --model llama3.1-8b --workers 12 > logs/judge_tqa_llama.log 2>&1; stamp "judge tqa llama rc=$?"
$J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_tqa truth_dec python3 scripts/e4b_qa_gated.py --phase judge --regime tqa --model qwen3-8b --workers 12 > logs/judge_tqa_qwen.log 2>&1; stamp "judge tqa qwen rc=$?"
$J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_bbq truth_dec python3 scripts/e4b_qa_gated.py --phase judge --regime bbq --model llama3.1-8b --workers 12 > logs/judge_bbq_llama.log 2>&1; stamp "judge bbq llama rc=$?"
until grep -q "e4b qwen3-8b bbq rc=" logs/queue_gpu3.log; do sleep 30; done
$J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_bbq truth_dec python3 scripts/e4b_qa_gated.py --phase judge --regime bbq --model qwen3-8b --workers 12 > logs/judge_bbq_qwen.log 2>&1; stamp "judge bbq qwen rc=$?"
until grep -q "QUEUE GPU0b DONE" logs/queue_gpu0b.log; do sleep 60; done
for m in qwen3-8b llama3.1-8b; do
  $J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_sqa truth_dec python3 scripts/e20_thinking_rescue.py --phase judge --model $m --datasets sqa_self tqa --workers 12 > logs/judge_e20_$m.log 2>&1; stamp "judge e20 $m rc=$?"
done
stamp "QUEUE JUDGE DONE"
