#!/bin/bash
# 2026-11-21 23:2x, the night queue after GPT's review of the prompt write rule (four cards idle):
#   GPU 0:    TriviaQA / PopQA at "prompt" (ns_prompt.sh): Gemma-3-12B, then Gemma-3-27B;
#   GPU 1:    the same for Llama-3.1-8B, then Qwen3.5-9B (the card has about 28 GB free next to another user's job);
#   GPUs 2,3: the same for Qwen3.8-27B, and side by side GSM8K for Qwen3.8-27B at a cap of 1,024 tokens (no write, then
#             E at "prompt"): at 512, 177 of its 1,319 no-write solutions are cut off, so its change of -1.0 is mostly
#             length; about 30 GB per card for each of the two jobs.
# E layers and strengths are the main-table ones (budget 1); the sweeps contain them.
set -e
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
N=$L/ns_prompt.sh
pids=()
rm -f "$L"/NIGHT_1121_{FAILED,DONE} "$L/GSM8K_POS_GPU2_3_1024_FAILED"
( bash $N gemma3-12b 0 24 2.5 "1 2.5 4" > $L/nsp_g12.out 2>&1; bash $N gemma3-27b 0 28 1 "0.5 1 2" > $L/nsp_g27.out 2>&1 ) &
pids+=("$!")
( bash $N llama3.1-8b 1 16 3.5 "1 2 3.5 5" > $L/nsp_llama.out 2>&1; bash $N qwen3.5-9b 1 15 2 "0.5 1 2 3" > $L/nsp_q9.out 2>&1 ) &
pids+=("$!")
( bash $N qwen3.8-27b 2,3 28 1.75 "0.5 1 1.75 3" > $L/nsp_q27.out 2>&1 ) &
pids+=("$!")
(
  PY=$(python3 -c 'import json; print(json.load(open("Iteration_01/code/configs/newsets/frozen_models_20261119.json"))["models"]["qwen3.8-27b"]["python"])')
  echo "$(date '+%F %T') gsm8k qwen3.8-27b start on GPU 2,3 (cap 1024): neutral" >> $L/gsm8k_all.log
  trap 'rc=$?; if [ "$rc" -ne 0 ]; then touch "$L/GSM8K_POS_GPU2_3_1024_FAILED"; fi' EXIT
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=2,3 -w $ROOT truth_dec \
    $PY -I $ROOT/Iteration_01/code/scripts/newsets/newsets_gen.py gen --models $ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model qwen3.8-27b --task gsm8k --items $ROOT/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet \
    --ids $ROOT/Iteration_01/capability/sets/gsm8k_1319_20261119.json --max-new-tokens 1024 --batch-size 8 \
    --out $ROOT/Iteration_01/capability/gen_main1024/gsm8k/qwen3.8-27b/b8 --arms neutral >> $L/gsm8k_qwen3.8-27b_1024.log 2>&1
  echo "gsm8k qwen3.8-27b cap 1024 no write exit=$? $(date '+%T')" >> $L/gsm8k_all.log
  NEWTOK=1024 bash $L/gsm8k_pos.sh 2,3 prompt qwen3.8-27b neutral:E28.w.p1.75
) &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then touch "$L/NIGHT_1121_FAILED"; exit 1; fi
touch "$L/NIGHT_1121_DONE"
