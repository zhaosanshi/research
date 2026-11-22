#!/bin/bash
# 2026-11-22 21:5x, queue I: Qwen3.8-27B on GPU 3, which is free now, in place of lane D of n1s_1122.sh (that lane waited
# for GPU 2, still held by the ITI run; it was stopped before it started anything). Same steps: tune, its judge, the
# choice (diag/n1_on_sentence.py choose), test.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/n1s_model.sh
say() { echo "$(date '+%F %T') n1s lanes: $*" >> $L/n1s_all.log; }
bash $P q27 3 tune > $L/n1s_q27_tune.out 2>&1
until [ -e $L/N1S_qwen3.8-27b_tune_JUDGE_DONE ]; do
  { [ -e $L/N1S_qwen3.8-27b_tune_JUDGE_FAILED ] || [ -e $L/N1S_qwen3.8-27b_tune_FAILED ]; } && { say "qwen3.8-27b: tune failed, no test"; exit 1; }
  sleep 30
done
docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec \
    python3 -B Iteration_01/diag/n1_on_sentence.py choose qwen3.8-27b >> $L/n1s_choose.log 2>&1 || { say "qwen3.8-27b: choice failed"; exit 1; }
say "qwen3.8-27b: chosen ($(tail -1 $L/n1s_choose.log))"
bash $P q27 3 test > $L/n1s_q27_test.out 2>&1
say "lane q27 (GPU 3) end"
