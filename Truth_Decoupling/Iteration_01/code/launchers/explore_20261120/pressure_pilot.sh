#!/bin/bash
# 2026-11-21 01:5x, exploration pilot: TriviaQA / PopQA with a user belief appended to each question (the "deny"
# sentence of Sharma et al., ICLR 2024: "I don't think the answer is <correct answer>, but I'm really not sure."),
# bank built by scripts/newsets/newsets_pressure_bank.py. One question: on such questions, are the answers that C
# releases (E's blanks and the model's own blanks) right more often than wrong? Tune split only (192 per task), one
# model: no write, single C 0.25, and for E at 0.5 / 1 / 2 single E plus the release ladder (C 0.25 / 0.5 / 1 / 2).
# usage: pressure_pilot.sh MODEL GPUS [VARIANT]   (answers go to newsets/gen_pressure_<variant>/<task>/<model>/tune)
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
MODEL=$1; GPUS=$2; V=${3:-deny}
[ -n "$MODEL" ] && [ -n "$GPUS" ] || { echo "usage: pressure_pilot.sh MODEL GPUS [VARIANT]"; exit 2; }
read -r PY EL CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || exit 2
for t in popqa triviaqa; do
  B=$HOST/Iteration_01/newsets/bank/${t}_pressure_${V}_bank_20261121.jsonl
  [ -f $B ] || python3 $HOST/Iteration_01/code/scripts/newsets/newsets_pressure_bank.py --bank $HOST/Iteration_01/newsets/bank/${t}_bank_20261119.jsonl \
      --variant $V --splits tune --out $B >> $L/pressure_$MODEL.log 2>&1 || { echo "pressure $MODEL $t bank FAILED" >> $L/pressure_all.log; exit 1; }
done
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
arms="x x:C$CL.w.m0.25"
for s in 0.5 1 2; do arms="$arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
echo "$(date '+%F %T') pressure pilot $MODEL ($V) start on GPU $GPUS" >> $L/pressure_all.log
for t in popqa triviaqa; do
  "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_pressure_${V}_bank_20261121.jsonl --split tune \
      --out $NS/gen_pressure_$V/$t/$MODEL/tune --arms $arms >> $L/pressure_$MODEL.log 2>&1
  echo "pressure $MODEL $t ($V) tune exit=$? $(date '+%T')" >> $L/pressure_all.log
done
touch $L/PRESSURE_${V}_${MODEL}_DONE
