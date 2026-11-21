#!/bin/bash
# 2026-11-22 00:0x, third night queue: the caution pilot of nsc_prompt.sh (Llama-3.1-8B and Gemma-3-12B run first, on
# GPU 0) for the other three models, each on the card(s) its earlier jobs free. No judge call (alias match), so the
# only cost is idle GPU time; if the two-model pilot shows nothing, these runs are simply not used.
#   GPU 0:    Gemma-3-27B, after the two pilot runs;
#   GPU 1:    Qwen3.5-9B, after lane 1 of night2b_1121.sh;
#   GPUs 2,3: Qwen3.8-27B, after lane 2 of night2b_1121.sh.
# Table afterwards: python3 -B Iteration_01/diag/nsc_prompt_table.py
set -e
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
N=$L/nsc_prompt.sh
rm -f "$L/NIGHT3_1121_DONE" "$L/NIGHT3_1121_FAILED"
pids=()
wait_for() {
  local prefix=$1
  until [ -f "${prefix}_DONE" ]; do
    [ ! -f "${prefix}_FAILED" ] || return 1
    sleep 30
  done
}
(
  wait_for "$L/NSC_llama3.1-8b"
  wait_for "$L/NSC_gemma3-12b"
  bash $N gemma3-27b 0 28 "0.5 1 2" > $L/nsc_g27.out 2>&1
) &
pids+=("$!")
(
  wait_for "$L/NIGHT2B_LANE1"
  bash $N qwen3.5-9b 1 15 "0.5 1 2 3" > $L/nsc_q9.out 2>&1
) &
pids+=("$!")
(
  wait_for "$L/NIGHT2B_LANE2"
  bash $N qwen3.8-27b 2,3 28 "0.5 1 1.75 3" > $L/nsc_q27.out 2>&1
) &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then touch "$L/NIGHT3_1121_FAILED"; exit 1; fi
touch "$L/NIGHT3_1121_DONE"
