#!/bin/bash
# 2026-11-22 02:4x: confidence of the prompt-only answers on the full (QA)^2 set for the five models (qaqa_conf.sh).
#   GPU 2 (next to the GSM8K job of Qwen3.8-27B): Llama-3.1-8B, then Qwen3.5-9B;   GPU 3 (same): Gemma-3-12B;
#   GPU 0, after qaqa_tuned_chain.sh has finished generating on it: Gemma-3-27B, then Qwen3.8-27B.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
C=$L/qaqa_conf.sh; T=qaqafull; B=qaqa_full_bank_20261122.jsonl
pids=()
( rc=0; bash $C $T $B llama3.1-8b 2 > $L/qaqaconf_llama.out 2>&1 || rc=1; bash $C $T $B qwen3.5-9b 2 > $L/qaqaconf_q9.out 2>&1 || rc=1; exit $rc ) &
pids+=("$!")
( bash $C $T $B gemma3-12b 3 > $L/qaqaconf_g12.out 2>&1 ) &
pids+=("$!")
( until grep -q "qaqa tuned chain: generation of the chosen arms done\|qaqa tuned chain: FAILED" $L/qaqa_all.log 2>/dev/null; do sleep 30; done
  rc=0; bash $C $T $B gemma3-27b 0 > $L/qaqaconf_g27.out 2>&1 || rc=1; bash $C $T $B qwen3.8-27b 0 > $L/qaqaconf_q27.out 2>&1 || rc=1; exit $rc ) &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then touch $L/QAQA_CONF_1122_FAILED; exit 1; fi
touch $L/QAQA_CONF_1122_DONE
