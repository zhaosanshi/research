#!/bin/bash
# 2026-11-21 00:5x, exploration for the lead's request (N1 better than E alone and than C alone on TriviaQA / PopQA with
# strengths of their own). On these factual sets the answers E silences are mostly wrong ones already and what C
# releases toward answering is wrong too, so the release step cannot add anything. The models' problem there is the
# opposite one: they answer too much. Here the C strength takes the other sign (toward caution, arm sign p), chosen on
# the dataset's tune split like every other strength: single C toward caution at 0.25 / 0.5 / 1 / 2, and E at
# 0.5 / 1 / 2 together with C toward caution at the four strengths, written on every question (one pass, no re-ask).
# Tune (192) and test (500), both datasets; scripts/newsets/newsets_gen.py gen as in ns_dose_all.sh.
# usage: ns_caut.sh MODEL GPUS [WAIT_GREP_PATTERN WAIT_FILE]
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
MODEL=${1:?model}; GPUS=${2:?gpus}
if [ -n "${3:-}" ]; then until grep -q "$3" "$4" 2>/dev/null; do sleep 60; done; fi
read -r PY EL CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || exit 1
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
arms=""
for k in 0.25 0.5 1 2; do arms="$arms x:C$CL.w.p$k"; done
for s in 0.5 1 2; do for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.p$k"; done; done
echo "$(date '+%F %T') nscaut $MODEL start on GPU $GPUS" >> $L/nscaut_all.log
for sp in tune test; do
  for t in popqa triviaqa; do
    "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_bank_20261119.jsonl --split $sp \
        --out $NS/gen/$t/$MODEL/$sp --arms $arms >> $L/nscaut_$MODEL.log 2>&1
    echo "nscaut $MODEL $t $sp exit=$? $(date '+%T')" >> $L/nscaut_all.log
  done
done
touch $L/NSCAUT_${MODEL}_DONE
