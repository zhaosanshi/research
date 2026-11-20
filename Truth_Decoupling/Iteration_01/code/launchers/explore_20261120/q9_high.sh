#!/bin/bash
# Qwen3.5-9B at E strengths 2.5 and 3 (up to strength 2 its MMLU does not move and N1 is only +21 over the prompt-only
# baseline): the same arms as dose_mid_all.sh (single E, E + C at 0.25 / 0.5 / 1 / 2), tune first (after that chain's
# Qwen3.5-9B lane the selfcheck record is for tune), then selfcheck + test; then MMLU (1,000 items) at the two strengths.
# Waits for that lane and for the small-model pilot on GPU 0 (card memory).
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
TF=$ROOT/envs/tf519/bin/python
RUNNER=$HOST/Iteration_01/code/scripts/review_checks/ff_explore_ec.sh
until grep -q "dosemid tq q9 exit=0" $L/dosemid_all.log 2>/dev/null && [ -e $L/IDEAA_llama_q9_g12_DONE ]; do sleep 30; done
E44=results_newmodels_ff_20261112/e44
MODEL=qwen3.5-9b; CODE=$ROOT/code_ff_explore_20261113b; run=$E44/qwen3.5-9b_E2; E_RUN=$ROOT/$run; C_RUN=$ROOT/$E44/qwen3.5-9b
SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.5-9b; EL=15; CL=16; GPU=0; ES="2.5 3"; KS="0.25 0.5 1 2"; TAG=dosehigh
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
arms=""; for s in $ES; do arms="$arms x:E$EL.w.p$s"; for k in $KS; do arms="$arms x:E$EL.w.p$s+C$CL.w.m$k"; done; done
rc=0; log=$L/q9high.log
for sp in tune test; do
  if [ $sp = test ]; then
    XE="scripts/review_checks/e44_split_test.py extra"
    MODEL=$MODEL CODE_PATH=$CODE TABLE_CODE=$ROOT/Iteration_01/code SOURCE_DIR=$SRC E_RUN=$E_RUN C_RUN=$C_RUN E_ON=E2 C_E_ON=E1 \
      EL=$EL CL=$CL ES="$ES" KS="$KS" GPU=$GPU SPLIT=test TAG=$TAG STAGE=selfcheck bash $RUNNER >> $log 2>&1 || { rc=11; break; }
  else
    XE="scripts/e44_eval_extra.py"
  fi
  COMMON=(--source-run $SRC --run-dir $E_RUN --e-on E2 --c-run-dir $C_RUN --c-e-on E1 --split $sp
          --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
  echo "=== $(date '+%F %T') q9 $sp generate$arms" >> $log
  "${DX[@]}" $TF $XE generate "${COMMON[@]}" --arms $arms >> $log 2>&1 || { rc=12; break; }
  gen=""; for a in $arms; do f=${a//:/--}; f=${f//+/-and-}; f=${f//./-}; gen="$gen ${f}_$sp.json"; done
  NAME=${TAG}_$sp; W=$run/eval_extra/judge_${sp}_$NAME
  "${DX[@]}" $TF $XE export "${COMMON[@]}" --gen $gen --name $NAME >> $log 2>&1 || { rc=13; break; }
  mkdir -p $HOST/$W/judge2/cache
  cp $HOST/$run/eval_extra/judge_$sp/$NAME.json $HOST/$W/judge_input.json
  "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || { rc=15; break; }
done
echo "q9high tq exit=$rc $(date '+%T')" >> $L/q9high_all.log
mkdir -p Iteration_01/capability/q9high; rm -f Iteration_01/capability/q9high/qwen3.5-9b.json
docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec $TF -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run \
  --models Iteration_01/code/configs/newsets/frozen_models_20261119.json --model qwen3.5-9b \
  --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_1000_20261119.json \
  --out Iteration_01/capability/q9high/qwen3.5-9b.json --arms x "x:E#.w.p2" "x:E#.w.p2.5" "x:E#.w.p3" >> $log 2>&1
echo "q9high mmlu exit=$? $(date '+%T')" >> $L/q9high_all.log
touch $L/Q9HIGH_DONE
