#!/bin/bash
# 2026-11-22 09:2x (Opus 5.5), exploration, work queue discussion/work_queue/2026-11-22_queue_b.md task 3. The
# project's single E on Llama-3.1-8B at layer 16 has test nets of +69 (strength 5) and +70 (strength 7) at the write
# rule "prompt" (summary section 14.3), but no MMLU and no GSM8K reading at those strengths, so the comparison with ITI
# "both without a budget" had one row missing. This measures:
#   MMLU by option probability on the 14,042 items (scripts/capability/cap_mcq_sweep.py, the program of the main
#   table's full-set files; it writes at the training rule, and for MMLU the read position is the last prompt token,
#   where the training rule and "prompt" have written the same positions);
#   GSM8K at "prompt" (launchers/explore_20261120/gsm8k_pos.sh, 512 tokens, batch 8).
# No judge call. Tables: python3 -B Iteration_01/diag/iti_table.py ... is not involved; read the MMLU file with the
# halves of diag/per_model_table.py, and GSM8K with diag/gsm8k_main_table.py llama3.1-8b (container).
# usage: e_hi_llama.sh GPU          Markers: E_HI_LLAMA_DONE / _FAILED (gsm8k_pos.sh's own marker is not removed at
# its start, so it is not used here).
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
GPU=$1
[ -n "$GPU" ] || { echo "usage: e_hi_llama.sh GPU"; exit 2; }
MK=$L/E_HI_LLAMA
say() { echo "$(date '+%F %T') e_hi_llama: $*" >> $L/iti_all.log; }
rm -f ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
OUT=Iteration_01/capability/full_l16hi/llama3.1-8b.json
G=Iteration_01/capability/gen_main/gsm8k/llama3.1-8b/b8
mkdir -p Iteration_01/capability/full_l16hi
say "start on GPU $GPU"
if [ ! -e $OUT ]; then
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$GPU -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -w $ROOT truth_dec \
    python3 -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model llama3.1-8b --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet \
    --ids Iteration_01/capability/sets/mmlu_14042_20261120.json --out $OUT --arms x "x:E16.w.p5" "x:E16.w.p7" \
    > $L/e_hi_llama_mmlu.log 2>&1 || fail "mmlu"
fi
[ -s $OUT ] || fail "no MMLU file"
say "mmlu done"
bash $L/gsm8k_pos.sh $GPU prompt llama3.1-8b neutral:E16.w.p5 llama3.1-8b neutral:E16.w.p7 > $L/e_hi_llama_gsm8k.out 2>&1 || fail "gsm8k"
[ -s $G/neutral--E16-w-p5-at-prompt.json ] && [ -s $G/neutral--E16-w-p7-at-prompt.json ] || fail "a GSM8K file is missing"
say "done"
touch ${MK}_DONE
