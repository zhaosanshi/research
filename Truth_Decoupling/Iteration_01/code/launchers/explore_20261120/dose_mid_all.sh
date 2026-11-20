#!/bin/bash
# E strengths 1.25 / 1.5 / 1.75 (the lead, 2026-11-20 08:33: strengths between 1 and 2 are to be tried), five models.
# 1. TruthfulQA: single E and E + C at 0.25 / 0.5 / 1 / 2 (the arms N1 needs) on the test split first (the stored
#    selfcheck records are for test), then selfcheck (ff_explore_ec.sh, unchanged) and the same arms on tune (the caps
#    are chosen there). Generation, export and the judge cells call e44_eval_extra.py / judge_two_axis_run.py with the
#    arguments of ff_explore_ec.sh; only the arm list differs (no random controls; the single C arms exist already).
#    Judging is a separate script (DeepSeek Pro, off-peak hours only).
# 2. MMLU, full test split, option probability, at the same three strengths (cap_mcq_sweep.py).
# Starts when the full-MMLU confirmation run has released the cards. GPU 0 carries another user's idle process: only
# the three small models go there.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
RUNNER=$HOST/Iteration_01/code/scripts/review_checks/ff_explore_ec.sh
TF=$ROOT/envs/tf519/bin/python
ES="1.25 1.5 1.75"; KS="0.25 0.5 1 2"; TAG=dosemid
tq() {  # tq KEY MODEL CODE PYM E_RUN C_RUN E_ON C_E_ON EL CL SRC GPU
  local key=$1 MODEL=$2 CODE=$ROOT/$3 PYM=$4 E_RUN=$ROOT/$5 C_RUN=$ROOT/$6 E_ON=$7 C_E_ON=$8 EL=$9 CL=${10} SRC=$ROOT/${11} GPU=${12}
  local DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
            -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
            -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
  local arms="" s k sp XE gen a f NAME WORK hw rc=0
  for s in $ES; do arms="$arms x:E$EL.w.p$s"; for k in $KS; do arms="$arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
  for sp in test tune; do
    if [ $sp = tune ]; then
      XE="scripts/e44_eval_extra.py"
      MODEL=$MODEL CODE_PATH=$CODE TABLE_CODE=$ROOT/Iteration_01/code SOURCE_DIR=$SRC E_RUN=$E_RUN C_RUN=$C_RUN E_ON=$E_ON \
        C_E_ON=$C_E_ON EL=$EL CL=$CL ES="$ES" KS="$KS" GPU=$GPU SPLIT=tune TAG=$TAG STAGE=selfcheck bash $RUNNER >> $L/dosemid_$key.log 2>&1 || { rc=11; break; }
    else
      XE="scripts/review_checks/e44_split_test.py extra"
    fi
    local COMMON=(--source-run $SRC --run-dir $E_RUN --e-on $E_ON --c-run-dir $C_RUN --c-e-on $C_E_ON --split $sp
                  --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
    echo "=== $(date '+%F %T') $key $sp generate$arms" >> $L/dosemid_$key.log
    "${DX[@]}" $PYM $XE generate "${COMMON[@]}" --arms $arms >> $L/dosemid_$key.log 2>&1 || { rc=12; break; }
    gen=""
    for a in $arms; do f=${a//:/--}; f=${f//+/-and-}; f=${f//./-}; gen="$gen ${f}_$sp.json"; done
    NAME=${TAG}_$sp; WORK=$E_RUN/eval_extra/judge_${sp}_$NAME; hw=$HOST/${WORK#$ROOT/}
    echo "=== $(date '+%F %T') $key $sp export + cells" >> $L/dosemid_$key.log
    "${DX[@]}" $PYM $XE export "${COMMON[@]}" --gen $gen --name $NAME >> $L/dosemid_$key.log 2>&1 || { rc=13; break; }
    mkdir -p $hw/judge2/cache
    [ -f $hw/judge_input.json ] || cp $HOST/${E_RUN#$ROOT/}/eval_extra/judge_$sp/$NAME.json $hw/judge_input.json
    cmp $HOST/${E_RUN#$ROOT/}/eval_extra/judge_$sp/$NAME.json $hw/judge_input.json || { rc=14; break; }
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $WORK --generation judge_input.json >> $L/dosemid_$key.log 2>&1 || { rc=15; break; }
  done
  echo "dosemid tq $key exit=$rc $(date '+%T')" >> $L/dosemid_all.log
}
mm() {  # mm KEY PYTHON GPU
  rm -f Iteration_01/capability/full_mid/$1.json
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$3 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $ROOT truth_dec \
    $2 -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model $1 --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_14042_20261120.json \
    --out Iteration_01/capability/full_mid/$1.json --arms x "x:E#.w.p1.25" "x:E#.w.p1.5" "x:E#.w.p1.75" > $L/fullmid_$1.log 2>&1
  echo "fullmid $1 exit=$? $(date '+%T')" >> $L/dosemid_all.log
}
rm -f $L/dosemid_all.log $L/DOSEMID_DONE $L/DOSEMID_TQ_DONE
mkdir -p Iteration_01/capability/full_mid
until [ -e $L/FULL_DONE ]; do sleep 30; done
echo "$(date '+%F %T') cards free, start" >> $L/dosemid_all.log
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
( tq llama llama3.1-8b code_e45_20261112 python3 results_e44_factorial_das/llama3.1-8b_side_E2 results_e44_factorial_das/llama3.1-8b E2 E1 11 19 results_qa_das_v1u/llama3.1-8b/full 0
  tq q9 qwen3.5-9b code_ff_explore_20261113b $TF $E44/qwen3.5-9b_E2 $E44/qwen3.5-9b E2 E1 15 16 results_newmodels_ff_20261112/source/qwen3.5-9b 0
  tq g12 gemma3-12b Iteration_01/code python3 $G3/12b/e44/gemma3-12b $G3/12b/e44/gemma3-12b E2 E2 22 25 $G3/12b/source/gemma3-12b 0 ) &
( tq g27 gemma3-27b Iteration_01/code python3 $G3/27b/e44/gemma3-27b $G3/27b/e44/gemma3-27b E2 E2 26 31 $G3/27b/source/gemma3-27b 1 ) &
( tq q27 qwen3.8-27b-dm code_ff_explore_20261113b $TF $E44/qwen3.8-27b-dm $E44/qwen3.8-27b-dm E2 E2 28 33 results_newmodels_ff_20261112/source/qwen3.8-27b-dm 2,3 ) &
wait
touch $L/DOSEMID_TQ_DONE
( mm llama3.1-8b python3 0; mm qwen3.5-9b $TF 0; mm gemma3-12b python3 0 ) &
( mm gemma3-27b python3 1 ) &
( mm qwen3.8-27b $TF 2,3 ) &
wait
touch $L/DOSEMID_DONE
