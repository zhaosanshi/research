#!/usr/bin/env bash
# Llama-3.1-8B extras for the teacher's steps, run on one GPU (shares the GPU with a model chain):
#   TQA robustness (step 8): two paraphrased prompts x E11 subset; three direction-estimation seeds x E11 subset
#   E14: TruthfulQA question-level MC1/MC2 accuracy under the same writes (the "answer accuracy" the reviewers read)
#   E6 transfer (clamp to probe score) on HaluEval and SimpleQA-tf, so the transfer write has curves on all four regimes
# usage: run_llama_extra.sh <gpu>
set -u
GPU="${1:-1}"
MODEL=llama3.1-8b
BASE=/data/sanshizhao/Truth_Decoupling
cd "$BASE"
mkdir -p logs
dex() { local root="$1"; shift; docker exec -u 1000:1001 -e HOME=/workspace/Truth_Decoupling/hf_cache -e CUDA_VISIBLE_DEVICES=$GPU -e PYTHONUNBUFFERED=1 \
  -e TD_RESULTS_DIR=/workspace/Truth_Decoupling/$root -e TD_DATA_DIR=/workspace/Truth_Decoupling/code/data/processed \
  -w /workspace/Truth_Decoupling/code/experiments truth_dec "$@"; }
run() { local root="$1" name="$2"; shift 2; echo "[$(date +%H:%M:%S)] $name start"; dex "$root" "$@" > "logs/extra_${name}_$MODEL.log" 2>&1; echo "[$(date +%H:%M:%S)] $name rc=$?"; }
SUB='E_a1$|E_a2$|E_a3$|gate_disagree_a5|C_abst_a1$|prompt_only'
for P in default_v1_para1 default_v1_para2; do
  run results_tqa "tqa_e11_$P" python3 scripts/e11_dose_rates.py --exp E2b_tqa --e1-exp E1_directions_tqa --model "$MODEL" --e-layers 12 13 14 --prompts "$P" --conditions "$SUB" --random-seeds 0 --tag "prompt_$P"
done
for S in 1 2 3; do
  run results_tqa "tqa_e11_estseed$S" python3 scripts/e11_dose_rates.py --exp E2b_tqa --e1-exp E1_directions_tqa --model "$MODEL" --e-layers 12 13 14 --reestimate-seed $S --conditions "$SUB" --random-seeds 0 --tag "estseed$S"
done
run results_tqa tqa_e14 python3 scripts/e14_mc_eval.py --exp E2b_tqa --e1-exp E1_directions_tqa --model "$MODEL" --e-layers 12 13 14 --probe-layer 14
run results_halu halu_e6 python3 scripts/e6_transfer.py --exp E2b_halu --e1-exp E1_directions_halu --model "$MODEL" --read-layers 14 13 --write-layers 16 14
run results_sqa sqa_e6 python3 scripts/e6_transfer.py --exp E2b_sqa --e1-exp E1_directions_sqa --model "$MODEL" --read-layers 12 13 --write-layers 15 14
echo "[$(date +%H:%M:%S)] LLAMA EXTRA DONE"
