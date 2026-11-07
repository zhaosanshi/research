#!/bin/bash
# E41 chain (Claude, 2026-11-07): fixed 2x2 C x E at every layer, true/false judgement.
#   usage: e41_chain.sh MODEL GPU [pilot]      (pilot = first 48 statements, reference check only)
# Runs the DAS-E sweep first, then the mean-difference-E sweep, on one GPU; the C is the 2x2's C in both (orthogonalised against
# the DAS-E of its layer). Reference NPZs are the 2x2 sweeps; baseline / E-only / C-only conditions must reproduce them.
set -u
MODEL=$1; GPU=$2; MODE=${3:-full}
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/review_r22_claude/claude_gpt_review/from_claude/code
RES=$ROOT/results_e41_20261107
TFCE=$ROOT/results_das_tfce
case $MODEL in
  llama3.1-8b) CA=Cp_L25_abst_a2; CN=Cp_L26_ans_a2;;      # 2x2 protocol: prompt-contrast C for Llama
  qwen3-8b)    CA=Cs_L33_abst_a2; CN=Cs_L24_ans_a2;;      # self-report C for Qwen
  gemma3-4b)   CA=Cp_L25_abst_a2; CN=Cp_L23_ans_a2;;      # prompt-contrast C for Gemma
  *) echo "unknown model $MODEL"; exit 2;;
esac
LIMIT=""; SUFFIX=""
[ "$MODE" = pilot ] && { LIMIT="--limit 48"; SUFFIX="_pilot48"; }
mkdir -p $HOST/results_e41_20261107/logs
run() {  # LOGNAME args...
  local log=$1; shift
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e TD_RESULTS_DIR=$RES -e TD_DATA_DIR=$ROOT/code/data/processed -e TD_MODEL_ROOT=$ROOT/models \
    -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -w $CODE truth_dec python3 "$@" \
    > $HOST/results_e41_20261107/logs/$log.log 2>&1
  echo $? > $HOST/results_e41_20261107/logs/$log.exit
}
for EST in das meandiff; do
  run e41_${MODEL}_${EST}${SUFFIX} scripts/e41_fixed_c_all_e.py --exp E2b_tqa --model $MODEL --tune-pairs 192 --e-estimator $EST \
    --das-dir $TFCE/E1/$MODEL --das-variant tf-ce-bidir --c-orth das --c-abst $CA --c-ans $CN --tag ${EST}${SUFFIX} $LIMIT \
    --ref-e-npz $TFCE/E23/$MODEL/knob_sweep_das_tfce_full.npz --ref-c-npz $TFCE/E23/$MODEL/knob_sweep_das_tfce_C.npz
  [ "$MODE" = pilot ] && break
done
date > $HOST/results_e41_20261107/logs/${MODEL}${SUFFIX}.DONE
