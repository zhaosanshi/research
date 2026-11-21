#!/bin/bash
# 2026-11-21 23:1x, exploration: TriviaQA and PopQA with every write at the core rule "prompt" (question tokens and
# assistant header; no decode step written), E at the model's MAIN-TABLE layer. The earlier TriviaQA / PopQA results
# (summary section 12) were made at the training rule and at the layer chosen for TruthfulQA alone; if the prompt rule
# becomes the method's setting, the out-of-domain numbers have to be at that rule too.
# One more reason to look again: at the training rule the answers that the C re-ask released on these fact questions
# were mostly denials of the question's premise ("None", "No one"), the same E bias that put zeros into GSM8K
# derivations; at "prompt" the generated tokens are no longer pushed.
# Written before any answer of this run was read:
#   F (frozen transfer, the main reading): E strength = the model's main-table strength; N1's two caps and the strength
#     of single C = the ones chosen on the TruthfulQA tune split at "prompt"; nothing is chosen on the new datasets;
#     test (500 questions) is read once, test2 (500) is the repeat.
#   D (dataset-tuned, the lead's 2026-11-21 00:15 request): E strength and the two caps chosen on each dataset's own
#     tune split (192) among the strengths below; test read at that choice; test2 is the repeat.
#   "N1 better than single E" on a dataset = the five-model total of N1 minus single E (same strength) has an interval
#     above 0 on test AND on test2. Alias match throughout (no judge call).
# Arms per split (tune, test, test2), all at "prompt": single E and E + C at 0.25 / 0.5 / 1 / 2 for every strength of
# the sweep; single C at 0.25 / 0.5 / 1 / 2. The no-write pass is the existing one (copied; no write, so no rule).
# Generation: scripts/newsets/ns_positions.py (newsets_gen.py unchanged) with NS_POS_TWO_WRITE=1 and
# NS_POS_PLAIN_NAMES=1 into newsets/gen_prompt/<task>/<model>/<split>/. Table: diag/ns_prompt_table.py.
# usage: ns_prompt.sh MODEL GPUS E_LAYER MAIN_STRENGTH "SWEEP STRENGTHS"
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
CODE=$ROOT/Iteration_01/code
NSP=$CODE/scripts/newsets/ns_positions.py
M=$CODE/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
MODEL=$1; GPUS=$2; EL=$3; MAIN=$4; SWEEP=$5
[ -n "$SWEEP" ] || { echo "usage: ns_prompt.sh MODEL GPUS E_LAYER MAIN_STRENGTH \"SWEEP STRENGTHS\""; exit 2; }
case " $SWEEP " in *" $MAIN "*) ;; *) echo "the sweep must contain the main strength"; exit 2 ;; esac
read -r PY CL <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["cl"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL)" || { echo "nsp $MODEL: not in the config" >> $L/nsp_all.log; exit 1; }
[ "$EL" -lt "$CL" ] || { echo "nsp $MODEL: the E layer must be below the C layer" >> $L/nsp_all.log; exit 1; }
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
arms=""
for k in 0.25 0.5 1 2; do arms="$arms x:C$CL.w.m$k"; done
for s in $SWEEP; do arms="$arms x:E$EL.w.p$s"; for k in 0.25 0.5 1 2; do arms="$arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
echo "$(date '+%F %T') nsp $MODEL start on GPU $GPUS: E$EL main $MAIN sweep $SWEEP, C$CL" >> $L/nsp_all.log
for sp in tune test test2; do
  for t in popqa triviaqa; do
    items=$NS/bank/${t}_bank_20261119.jsonl; [ $sp = test2 ] && items=$NS/bank/${t}_bank_test2_20261120.jsonl
    d=$HNS/gen_prompt/$t/$MODEL/$sp
    mkdir -p $d
    [ -e $d/x.json ] || cp -p $HNS/gen/$t/$MODEL/$sp/x.json $d/x.json || { echo "nsp $MODEL $t $sp: no no-write pass to copy" >> $L/nsp_all.log; continue; }
    "${DX[@]}" $PY -I $NSP prompt gen --models $M --model $MODEL --task $t --items $items --split $sp \
        --out $NS/gen_prompt/$t/$MODEL/$sp --arms $arms >> $L/nsp_$MODEL.log 2>&1
    echo "nsp $MODEL $t $sp exit=$? $(date '+%T')" >> $L/nsp_all.log
  done
done
touch $L/NSP_${MODEL}_DONE
