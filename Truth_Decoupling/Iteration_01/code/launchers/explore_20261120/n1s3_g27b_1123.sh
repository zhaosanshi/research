#!/bin/bash
# 2026-11-23 07:1x, queue I: GPU 3 became free while n1s3_g27_1123.sh (GPU 0) was waiting for Gemma-3-27B's (QA)^2 judge,
# with Qwen3.8-27B's section 9 queued after it. That script was stopped while it only waited (nothing of its own running;
# the judge runs on its own). Now: Qwen3.8-27B section 9 on GPU 3 (this script, in the background), and Gemma-3-27B's
# section 8 continues here: wait for the judge, the choice, the test pass on GPU 0 if s_half is chosen. Gemma-3-27B's
# section 9 is left to cap9_1123.sh, whose lane waits for exactly this.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
say() { echo "$(date '+%F %T') n1s3 lanes: $*" >> $L/n1s_all.log; }
( bash $L/cap9_model.sh q27 3 mmlu > $L/cap9_q27_mmlu.out 2>&1; bash $L/cap9_model.sh q27 3 gsm > $L/cap9_q27_gsm.out 2>&1
  say "qwen3.8-27b section 9 end (GPU 3)" ) &
m=gemma3-27b
c=Iteration_01/newsets/h22/n1s3_choice_$m.json
until [ -e $L/N1S3_${m}_qaqa_JUDGE_DONE ] || [ -e $L/N1S3_${m}_qaqa_JUDGE_FAILED ]; do sleep 30; done
if [ -e $L/N1S3_${m}_qaqa_JUDGE_DONE ]; then
  docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec \
      python3 -B Iteration_01/diag/n1_on_sentence3.py choose $m >> $L/n1s3_choose.log 2>&1 && say "$m: chosen E $(tail -1 $L/n1s3_choose.log)"
  if [ -e $c ] && [ "$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["chosen"] != d["s_sel"]))' $c)" = 1 ]; then
    bash $L/n1s3_model.sh g27 0 test > $L/n1s3_g27_test.out 2>&1
  fi
fi
say "$m section 8 end"
wait
