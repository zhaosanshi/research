#!/bin/bash
# 2026-11-22 01:4x: (QA)^2, the whole evaluation set (570 questions), five models, every setting frozen from TruthfulQA
# at the write rule "prompt" (qaqa_run.sh; settings as in diag/qaqa_pilot_table.py SET). Reading rule and predictions:
# discussion/work_queue/2026-11-22_queue.md section 2.8, written before this was started.
#   lane A, GPU 0: Gemma-3-12B, Gemma-3-27B, Qwen3.8-27B (one card each, one after the other);
#   lane B, GPU 1: Llama-3.1-8B, Qwen3.5-9B.
# Each model's judging runs in the background while the next model generates. A failed model does not stop the next.
# Table: docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B Iteration_01/diag/qaqa_pilot_table.py \
#          --task qaqafull --bank Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl --work test_full570 \
#          llama3.1-8b qwen3.5-9b qwen3.8-27b gemma3-12b gemma3-27b
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
R=$L/qaqa_run.sh; T=qaqafull; B=qaqa_full_bank_20261122.jsonl; W=test_full570
pids=()
(
  rc=0
  bash $R $T $B $W gemma3-12b 0 24 2.5 0.5 > $L/qaqafull_g12.out 2>&1 || rc=1
  bash $R $T $B $W gemma3-27b 0 28 1 0.25 > $L/qaqafull_g27.out 2>&1 || rc=1
  bash $R $T $B $W qwen3.8-27b 0 28 1.75 0.5 > $L/qaqafull_q27.out 2>&1 || rc=1
  exit $rc
) &
pids+=("$!")
(
  rc=0
  bash $R $T $B $W llama3.1-8b 1 16 3.5 0.25 > $L/qaqafull_llama.out 2>&1 || rc=1
  bash $R $T $B $W qwen3.5-9b 1 15 2 0.25 > $L/qaqafull_q9.out 2>&1 || rc=1
  exit $rc
) &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then touch $L/QAQA_FULL_1122_GEN_FAILED; exit 1; fi
touch $L/QAQA_FULL_1122_GEN_DONE
