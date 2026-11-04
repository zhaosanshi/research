#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
stamp "e4b mistral tqa start"
dex 3 results_tqa python3 scripts/e4b_qa_gated.py --phase generate --regime tqa --model mistral-7b --force --batch-size 16 > logs/e4b_tqa_mistral-7b.log 2>&1
stamp "e4b mistral tqa rc=$?"
J="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e CUDA_VISIBLE_DEVICES= -e PYTHONUNBUFFERED=1 -e DEEPSEEK_API_KEY_FILE=/workspace/Truth_Decoupling/logs/.deepseek_key -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed -w /workspace/Truth_Decoupling/code/experiments"
until grep -q "judge bbq llama rc=" logs/queue_judge.log; do sleep 60; done
$J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_tqa truth_dec python3 scripts/e4b_qa_gated.py --phase judge --regime tqa --model mistral-7b --workers 12 > logs/judge_tqa_mistral.log 2>&1; stamp "judge tqa mistral rc=$?"
stamp "QUEUE GPU3b DONE"
