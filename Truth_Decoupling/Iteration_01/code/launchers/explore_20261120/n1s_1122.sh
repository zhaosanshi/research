#!/bin/bash
# 2026-11-22 21:4x, queue I: n1s_model.sh for the five models, tune first, then (after that model's tune judge and the
# choice by diag/n1_on_sentence.py choose) the chosen strength on test.
#   lane B, GPU 1: Llama-3.1-8B tune, Qwen3.5-9B tune, then their test passes;
#   lane C, GPU 3: Gemma-3-27B;
#   lane D, GPU 2: Qwen3.8-27B, after ITI_FREE_gemma3-27b_faithful_seed42_endq_DONE (or _FAILED) and GPU 2 near empty;
#   lane E: Gemma-3-12B on whichever of GPU 0 (after CAA_GRID_gemma3-12b_DONE / _FAILED), 3, 2 is free first (lock free,
#           memory near 0), picked again for the test pass.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
P=$L/n1s_model.sh
LK=Iteration_01/newsets/selfjudge
ITI=$L/ITI_FREE_gemma3-27b_faithful_seed42_endq
say() { echo "$(date '+%F %T') n1s lanes: $*" >> $L/n1s_all.log; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}'; }
lockfree() { ( exec 8>$LK/.gpu$1.lock && flock -n 8 ) ; }
choose() {  # choose MODEL: wait for its tune judge, then write the choice
  local m=$1
  until [ -e $L/N1S_${m}_tune_JUDGE_DONE ]; do
    [ -e $L/N1S_${m}_tune_JUDGE_FAILED ] || [ -e $L/N1S_${m}_tune_FAILED ] && { say "$m: tune failed, no test"; return 1; }
    sleep 30
  done
  docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec \
      python3 -B Iteration_01/diag/n1_on_sentence.py choose $m >> $L/n1s_choose.log 2>&1 || { say "$m: choice failed"; return 1; }
  say "$m: chosen ($(tail -1 $L/n1s_choose.log))"
}
pick() {  # a free card for Gemma-3-12B
  while true; do
    if { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; } && [ "$(mem 0)" -lt 2000 ] && lockfree 0; then echo 0; return; fi
    if [ "$(mem 3)" -lt 2000 ] && lockfree 3; then echo 3; return; fi
    if { [ -e ${ITI}_DONE ] || [ -e ${ITI}_FAILED ]; } && [ "$(mem 2)" -lt 2000 ] && lockfree 2; then echo 2; return; fi
    sleep 60
  done
}
( bash $P llama 1 tune > $L/n1s_llama_tune.out 2>&1
  bash $P q9 1 tune > $L/n1s_q9_tune.out 2>&1
  choose llama3.1-8b && bash $P llama 1 test > $L/n1s_llama_test.out 2>&1
  choose qwen3.5-9b && bash $P q9 1 test > $L/n1s_q9_test.out 2>&1
  say "lane B end" ) &
( bash $P g27 3 tune > $L/n1s_g27_tune.out 2>&1
  choose gemma3-27b && bash $P g27 3 test > $L/n1s_g27_test.out 2>&1
  say "lane C end" ) &
( until { [ -e ${ITI}_DONE ] || [ -e ${ITI}_FAILED ]; } && [ "$(mem 2)" -lt 2000 ]; do sleep 60; done
  say "GPU 2 free for Qwen3.8-27B"
  bash $P q27 2 tune > $L/n1s_q27_tune.out 2>&1
  choose qwen3.8-27b && bash $P q27 2 test > $L/n1s_q27_test.out 2>&1
  say "lane D end" ) &
( sleep 120
  g=$(pick); say "Gemma-3-12B tune on GPU $g"
  bash $P g12 $g tune > $L/n1s_g12_tune.out 2>&1
  if choose gemma3-12b; then g=$(pick); say "Gemma-3-12B test on GPU $g"; bash $P g12 $g test > $L/n1s_g12_test.out 2>&1; fi
  say "lane E end" ) &
wait
say "all lanes ended"
