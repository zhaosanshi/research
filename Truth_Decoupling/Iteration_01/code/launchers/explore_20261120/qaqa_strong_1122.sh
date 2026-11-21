#!/bin/bash
# 2026-11-22 05:1x: the equal-answer-count strong comparator on (QA)^2 (qaqa_strong.sh) for the four models after
# Llama-3.1-8B, which was run first on its own as the check of the new entry scripts/newsets/ns_conf_pos.py (greedy
# reproduction 0.989, first-difference check 481 of 481 toward the target).
#   lane A, GPU 1:    Qwen3.5-9B;
#   lane B, GPU 2:    Gemma-3-12B, then Qwen3.8-27B on GPUs 2,3 (next to its own fact-dataset pilot, about 30 GB a card);
#   lane C, GPU 0:    Gemma-3-27B, after its TriviaQA / PopQA run at "prompt" (ns_prompt.sh) has left the card.
# A failed model does not stop the next. Markers: per model as in qaqa_strong.sh; QAQA_STRONG_1122_GPU_DONE / _FAILED
# when every lane has ended (the judges may still be running: wait for the five QAQA_STRONG_qaqafull_<MODEL>_DONE).
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
R=$L/qaqa_strong.sh; T=qaqafull; B=qaqa_full_bank_20261122.jsonl; W=test_full570_strong
rm -f $L/QAQA_STRONG_1122_GPU_DONE $L/QAQA_STRONG_1122_FAILED
pids=()
( bash $R $T $B $W qwen3.5-9b 1 15 2 > $L/qaqastrong_q9.out 2>&1 ) &
pids+=("$!")
(
  rc=0
  bash $R $T $B $W gemma3-12b 2 24 2.5 > $L/qaqastrong_g12.out 2>&1 || rc=1
  bash $R $T $B $W qwen3.8-27b 2,3 28 1.75 > $L/qaqastrong_q27.out 2>&1 || rc=1
  exit $rc
) &
pids+=("$!")
(
  until [ -e $L/NSP_gemma3-27b_DONE ] || [ -e $L/NSP_gemma3-27b_FAILED ]; do sleep 30; done
  bash $R $T $B $W gemma3-27b 0 28 1 > $L/qaqastrong_g27.out 2>&1
) &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then touch $L/QAQA_STRONG_1122_FAILED; exit 1; fi
touch $L/QAQA_STRONG_1122_GPU_DONE
