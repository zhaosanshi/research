#!/usr/bin/env bash
source /data/sanshizhao/Truth_Decoupling/logs/common.sh
J="docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e CUDA_VISIBLE_DEVICES= -e PYTHONUNBUFFERED=1 -e DEEPSEEK_API_KEY_FILE=/workspace/Truth_Decoupling/logs/.deepseek_key -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed -w /workspace/Truth_Decoupling/code/experiments"
# wait for the first judge chain (tqa qwen, bbq llama, bbq qwen) so the two chains do not hammer the API together
until grep -q "judge bbq qwen rc=" logs/queue_judge.log; do sleep 60; done
# v2 judge (adds the selective-answering rows; all calls cached) for the cells the first chain graded with the old code
$J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_tqa truth_dec python3 scripts/e4b_qa_gated.py --phase judge --regime tqa --model qwen3-8b --workers 12 > logs/judge_tqa_qwen_v2.log 2>&1; stamp "judge v2 tqa qwen rc=$?"
$J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_bbq truth_dec python3 scripts/e4b_qa_gated.py --phase judge --regime bbq --model llama3.1-8b --workers 12 > logs/judge_bbq_llama_v2.log 2>&1; stamp "judge v2 bbq llama rc=$?"
$J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/results_bbq truth_dec python3 scripts/e4b_qa_gated.py --phase judge --regime bbq --model qwen3-8b --workers 12 > logs/judge_bbq_qwen_v2.log 2>&1; stamp "judge v2 bbq qwen rc=$?"
# e4c rerank judges as each generation lands
for cell in "tqa llama3.1-8b results_tqa" "tqa qwen3-8b results_tqa" "bbq llama3.1-8b results_bbq" "bbq qwen3-8b results_bbq"; do
  set -- $cell
  until grep -q "e4c $1 $2 rc=" logs/queue_gpu3c.log; do sleep 60; done
  $J -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/$3 truth_dec python3 scripts/e4c_rerank.py --phase judge --regime $1 --model $2 --workers 12 > logs/judge_e4c_$1_$2.log 2>&1; stamp "judge e4c $1 $2 rc=$?"
done
stamp "QUEUE JUDGE2 DONE"
