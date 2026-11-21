#!/bin/bash
# 2026-11-22 02:0x: the (QA)^2 tune split (the 100 pilot questions) for the five models (qaqa_tune.sh): single C at four
# strengths, and single E and E + C at four strengths for every E strength of each model's sweep; the judge of each
# model runs in the background. Reason and reading rule: qaqa_tune.sh and the work queue, section 2.8.
#   GPU 1: Llama-3.1-8B, then Qwen3.5-9B;   GPU 2 (next to the GSM8K job of Qwen3.8-27B): Gemma-3-12B;
#   GPU 0, after the full-run generation of Qwen3.8-27B has left it: Gemma-3-27B, then Qwen3.8-27B.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
T=$L/qaqa_tune.sh
pids=()
( rc=0
  bash $T llama3.1-8b 1 16 "1 2 3.5 5" > $L/qaqatune_llama.out 2>&1 || rc=1
  bash $T qwen3.5-9b 1 15 "0.5 1 2 3" > $L/qaqatune_q9.out 2>&1 || rc=1
  exit $rc ) &
pids+=("$!")
( bash $T gemma3-12b 2 24 "1 2.5 4" > $L/qaqatune_g12.out 2>&1 ) &
pids+=("$!")
( until [ -e $L/QAQA_qaqafull_qwen3.8-27b_GEN_DONE ] || [ -e $L/QAQA_qaqafull_qwen3.8-27b_FAILED ]; do sleep 20; done
  rc=0
  bash $T gemma3-27b 0 28 "0.5 1 2" > $L/qaqatune_g27.out 2>&1 || rc=1
  bash $T qwen3.8-27b 0 28 "0.5 1 1.75 3" > $L/qaqatune_q27.out 2>&1 || rc=1
  exit $rc ) &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then touch $L/QAQA_TUNE_1122_GEN_FAILED; exit 1; fi
touch $L/QAQA_TUNE_1122_GEN_DONE
