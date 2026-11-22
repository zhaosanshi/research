#!/bin/bash
# 2026-11-23 01:5x, queue I section 8: n1s3_model.sh for the five models, each after its 7.3 run has left its card.
#   small models (GPU 1): after CAPH22_qwen3.5-9b_gsm_DONE / _FAILED (the last 7.3 run on GPU 1): Llama-3.1-8B, Qwen3.5-9B;
#   large models: after CAPH22_<MODEL>_DONE / _FAILED, on whichever of GPU 3, 2, 0 (0 only after CAA_GRID_gemma3-12b_DONE
#   / _FAILED) is free first.
# Then, per model: wait for its judges, the choice (diag/n1_on_sentence3.py choose), and the test phase if s_half is chosen.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/n1s3_model.sh
LK=Iteration_01/newsets/selfjudge
say() { echo "$(date '+%F %T') n1s3 lanes: $*" >> $L/n1s_all.log; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}'; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
pick() {
  while true; do
    if [ "$(mem 3)" -lt 2000 ] && lockfree 3; then echo 3; return; fi
    if [ "$(mem 2)" -lt 2000 ] && lockfree 2; then echo 2; return; fi
    if { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; } && [ "$(mem 0)" -lt 2000 ] && lockfree 0; then echo 0; return; fi
    sleep 60
  done
}
waitj() {  # waitj MODEL TAG: 0 when the judge is done
  until [ -e $L/N1S3_$1_$2_JUDGE_DONE ]; do [ -e $L/N1S3_$1_$2_JUDGE_FAILED ] && return 1; sleep 30; done
}
after_gen() {  # after_gen KEY MODEL SMALL(1|0)
  local k=$1 m=$2 small=$3 g pick_s
  [ -e $L/N1S3_${m}_gen_GPU_DONE ] || { say "$m: gen failed"; return 1; }
  waitj $m qaqa || { say "$m: qaqa judge failed"; return 1; }
  case $m in qwen3.8-27b|gemma3-12b) waitj $m tune || { say "$m: TruthfulQA tune judge failed"; return 1; } ;; esac
  docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec \
      python3 -B Iteration_01/diag/n1_on_sentence3.py choose $m >> $L/n1s3_choose.log 2>&1 || { say "$m: choice failed"; return 1; }
  pick_s=$(tail -1 $L/n1s3_choose.log)
  say "$m: chosen E $pick_s"
  if [ "$pick_s" != "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["strength_dot"])' Iteration_01/newsets/h22/n1s_choice_$m.json)" ]; then
    if [ $small = 1 ]; then g=1; else g=$(pick); fi
    bash $P $k $g test > $L/n1s3_${k}_test.out 2>&1
  fi
}
( until [ -e $L/CAPH22_qwen3.5-9b_gsm_DONE ] || [ -e $L/CAPH22_qwen3.5-9b_gsm_FAILED ]; do sleep 30; done
  bash $P llama 1 gen > $L/n1s3_llama_gen.out 2>&1; bash $P q9 1 gen > $L/n1s3_q9_gen.out 2>&1
  after_gen llama llama3.1-8b 1; after_gen q9 qwen3.5-9b 1; say "small models end" ) &
for km in g27:gemma3-27b q27:qwen3.8-27b g12:gemma3-12b; do
  k=${km%%:*}; m=${km#*:}
  ( until [ -e $L/CAPH22_${m}_DONE ] || [ -e $L/CAPH22_${m}_FAILED ]; do sleep 30; done
    sleep $((RANDOM % 20)); g=$(pick); say "$m gen on GPU $g"
    bash $P $k $g gen > $L/n1s3_${k}_gen.out 2>&1
    after_gen $k $m 0; say "$m end" ) &
done
wait
say "all lanes ended"
