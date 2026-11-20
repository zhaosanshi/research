#!/bin/bash
# 2026-11-21 02:1x, exploration: TriviaQA / PopQA with a user belief appended to each question (sentences of Sharma et
# al., ICLR 2024; bank scripts/newsets/newsets_pressure_bank.py), the method's systems with every setting chosen on tune
# and read on test. Follows the 192-question pilot (pressure_pilot.sh) in which, on TriviaQA, the answers C released
# under E were right more often than wrong.
# Per model, variant and task: tune = no write, the must-answer prompt, the prompt-only baseline, single C at
# 0.25 / 0.5 / 1 / 2, and for E at 0.5 / 1 / 2 / 3 single E plus the release ladder; then the arms named by
# diag/ns_pressure_table.py --choose (from tune only) on the test split. Passes that exist are skipped by the generator.
# usage: pressure2.sh MODEL GPUS "VARIANT ..." ['SHELL CONDITION TO WAIT FOR']   table: diag/ns_pressure_table.py VARIANT MODEL ...
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets
MODEL=$1; GPUS=$2; VARIANTS=${3:-deny}; COND=${4:-true}
[ -n "$MODEL" ] && [ -n "$GPUS" ] || { echo "usage: pressure2.sh MODEL GPUS VARIANTS [CONDITION]"; exit 2; }
read -r PY EL CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || exit 2
until eval "$COND"; do sleep 60; done
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
tune_arms="x conf conf_E2"
for k in 0.25 0.5 1 2; do tune_arms="$tune_arms x:C$CL.w.m$k"; done
for s in 0.5 1 2 3; do tune_arms="$tune_arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do tune_arms="$tune_arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
echo "$(date '+%F %T') pressure2 $MODEL ($VARIANTS) start on GPU $GPUS" >> $L/pressure_all.log
for V in $VARIANTS; do
  for t in triviaqa popqa; do
    for sp in tune test; do
      B=$HOST/Iteration_01/newsets/bank/${t}_pressure_${V}_bank_$([ $sp = tune ] || echo ${sp}_)20261121.jsonl
      [ -f $B ] || python3 $HOST/Iteration_01/code/scripts/newsets/newsets_pressure_bank.py --bank $HOST/Iteration_01/newsets/bank/${t}_bank_20261119.jsonl \
          --variant $V --splits $sp --out $B >> $L/pressure_$MODEL.log 2>&1 || { echo "pressure2 $MODEL $t $V $sp bank FAILED" >> $L/pressure_all.log; exit 1; }
    done
    "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_pressure_${V}_bank_20261121.jsonl --split tune \
        --out $NS/gen_pressure_$V/$t/$MODEL/tune --arms $tune_arms >> $L/pressure_$MODEL.log 2>&1
    echo "pressure2 $MODEL $t ($V) tune exit=$? $(date '+%T')" >> $L/pressure_all.log
    arms=$(python3 -B Iteration_01/diag/ns_pressure_table.py --choose $V $t $MODEL 2>> $L/pressure_$MODEL.log | tail -1)
    case "$arms" in "x conf conf_E2 "*) ;; *) echo "pressure2 $MODEL $t ($V) choose FAILED ($arms) $(date '+%T')" >> $L/pressure_all.log; continue ;; esac
    echo "pressure2 $MODEL $t ($V) chosen for test: $arms" >> $L/pressure_all.log
    "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $t --items $NS/bank/${t}_pressure_${V}_bank_test_20261121.jsonl --split test \
        --out $NS/gen_pressure_$V/$t/$MODEL/test --arms $arms >> $L/pressure_$MODEL.log 2>&1
    echo "pressure2 $MODEL $t ($V) test exit=$? $(date '+%T')" >> $L/pressure_all.log
  done
done
touch $L/PRESSURE2_${MODEL}_DONE
