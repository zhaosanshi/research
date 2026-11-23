#!/bin/bash
# 2026-11-23 05:3x, queue I section 9: cap9_model.sh for the five models, after the model's 7.3 run and its section-8
# GPU work (gen, and test when s_half was chosen) have ended.
#   small models: GSM8K on GPU 1 (Llama-3.1-8B, then Qwen3.5-9B); MMLU on whichever of GPU 3, 2, 0 is free first (their
#   MMLU does not fit the 65 GB rule of GPU 1), after their 7.3 MMLU;
#   large models: MMLU then GSM8K on whichever of GPU 3, 2, 0 is free first (0 only after CAA_GRID_gemma3-12b_DONE / _FAILED;
#   about 3.7 GB of another process sits on GPU 0, so its threshold is 8 GB).
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/cap9_model.sh
LK=Iteration_01/newsets/selfjudge
say() { echo "$(date '+%F %T') cap9 lanes: $*" >> $L/n1s_all.log; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}'; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
pick() {
  while true; do
    if [ "$(mem 3)" -lt 2000 ] && lockfree 3; then echo 3; return; fi
    if [ "$(mem 2)" -lt 2000 ] && lockfree 2; then echo 2; return; fi
    if { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; } && [ "$(mem 0)" -lt 8000 ] && lockfree 0; then echo 0; return; fi
    sleep 60
  done
}
sec8_done() {  # sec8_done MODEL: section-8 GPU work of the model has ended
  local m=$1 c=Iteration_01/newsets/h22/n1s3_choice_$1.json
  until [ -e $L/N1S3_${m}_gen_GPU_DONE ] || [ -e $L/N1S3_${m}_gen_FAILED ]; do sleep 30; done
  until [ -e $c ] || [ -e $L/N1S3_${m}_qaqa_JUDGE_FAILED ]; do sleep 30; done
  [ -e $c ] || return 0
  if [ "$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(int(d["chosen"] != d["s_sel"]))' $c)" = 1 ]; then
    until [ -e $L/N1S3_${m}_test_GPU_DONE ] || [ -e $L/N1S3_${m}_test_FAILED ]; do sleep 30; done
  fi
}
( sec8_done llama3.1-8b; sec8_done qwen3.5-9b
  bash $P llama 1 gsm > $L/cap9_llama_gsm.out 2>&1; bash $P q9 1 gsm > $L/cap9_q9_gsm.out 2>&1; say "small GSM8K end" ) &
( until [ -e $L/CAPH22_qwen3.5-9b_mmlu_DONE ] || [ -e $L/CAPH22_qwen3.5-9b_mmlu_FAILED ]; do sleep 30; done
  sec8_done llama3.1-8b; sec8_done qwen3.5-9b
  for k in llama q9; do g=$(pick); say "$k MMLU on GPU $g"; bash $P $k $g mmlu > $L/cap9_${k}_mmlu.out 2>&1; done; say "small MMLU end" ) &
for km in g12:gemma3-12b q27:qwen3.8-27b g27:gemma3-27b; do
  k=${km%%:*}; m=${km#*:}
  ( until [ -e $L/CAPH22_${m}_DONE ] || [ -e $L/CAPH22_${m}_FAILED ]; do sleep 30; done
    sec8_done $m; sleep $((RANDOM % 20))
    g=$(pick); say "$m section 9 on GPU $g"
    bash $P $k $g mmlu > $L/cap9_${k}_mmlu.out 2>&1; bash $P $k $g gsm > $L/cap9_${k}_gsm.out 2>&1; say "$m section 9 end" ) &
done
wait
say "all lanes ended"
