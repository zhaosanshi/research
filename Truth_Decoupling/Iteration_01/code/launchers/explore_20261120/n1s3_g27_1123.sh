#!/bin/bash
# 2026-11-23 06:3x, queue I: section 8 comes before section 9, and section 8's judges must end before 08:55. Gemma-3-27B's
# section 8 waited for a card while Qwen3.8-27B's section 9 had just taken GPU 0 (the section-8 lane does not use GPU 0:
# about 3.7 GB of another process sits there and that lane wants it near empty). Qwen3.8-27B's section-9 run was stopped
# in its MMLU check (nothing written), the waiting Gemma-3-27B lane was stopped, and this script runs on GPU 0:
# Gemma-3-27B section 8 (gen; its (QA)^2 judge; choice; test if s_half), then Qwen3.8-27B section 9 (MMLU, GSM8K).
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
say() { echo "$(date '+%F %T') n1s3 lanes: $*" >> $L/n1s_all.log; }
m=gemma3-27b
c=Iteration_01/newsets/h22/n1s3_choice_$m.json
bash $L/n1s3_model.sh g27 0 gen > $L/n1s3_g27_gen.out 2>&1
if [ -e $L/N1S3_${m}_gen_GPU_DONE ]; then
  until [ -e $L/N1S3_${m}_qaqa_JUDGE_DONE ] || [ -e $L/N1S3_${m}_qaqa_JUDGE_FAILED ]; do sleep 30; done
  if [ -e $L/N1S3_${m}_qaqa_JUDGE_DONE ]; then
    docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec \
        python3 -B Iteration_01/diag/n1_on_sentence3.py choose $m >> $L/n1s3_choose.log 2>&1 && say "$m: chosen E $(tail -1 $L/n1s3_choose.log)"
  fi
  if [ -e $c ] && [ "$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["chosen"] != d["s_sel"]))' $c)" = 1 ]; then
    bash $L/n1s3_model.sh g27 0 test > $L/n1s3_g27_test.out 2>&1
  fi
fi
say "$m section 8 end (GPU 0)"
bash $L/cap9_model.sh q27 0 mmlu > $L/cap9_q27_mmlu.out 2>&1
bash $L/cap9_model.sh q27 0 gsm > $L/cap9_q27_gsm.out 2>&1
say "qwen3.8-27b section 9 end (GPU 0)"
