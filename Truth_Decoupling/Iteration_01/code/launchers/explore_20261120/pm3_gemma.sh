#!/bin/bash
# 2026-11-20 22:3x: the Gemma-3 steps of pm_all.sh / pm2_all.sh that ran out of memory on GPU 1 (another user's process
# grew there from 3 GB to 43 GB during the afternoon), run again on GPUs 2 and 3 once pm2_all.sh has released them.
# Same functions and arguments as pm2_all.sh / pm_all.sh; only the card of the two Gemma-3 models differs.
#   Gemma-3-27B (GPU 2): must-answer + E at 1.25 / 1.5 / 1.75 on test (dmust_test), its confidence file, then MMLU on
#                        the full test split for the on-prompt arms (capability/ideaA_full).
#   Gemma-3-12B (GPU 3): the confidence file (its dmust_test generation succeeded).
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
RUNNER=$HOST/Iteration_01/code/scripts/review_checks/ff_explore_ec.sh
TF=$ROOT/envs/tf519/bin/python
KS="0.25 0.5 1 2"
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
setm() {
  case $1 in
    llama) MODEL=llama3.1-8b; NG=llama3.1-8b; CODE=$ROOT/code_e45_20261112; PYM=python3; run=results_e44_factorial_das/llama3.1-8b_side_E2
           crun=results_e44_factorial_das/llama3.1-8b; E_ON=E2; C_E_ON=E1; EL=11; CL=19; SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full; GPU=0 ;;
    q9)    MODEL=qwen3.5-9b; NG=qwen3.5-9b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.5-9b_E2
           crun=$E44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; EL=15; CL=16; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.5-9b; GPU=0 ;;
    g12)   MODEL=gemma3-12b; NG=gemma3-12b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/12b/e44/gemma3-12b
           crun=$run; E_ON=E2; C_E_ON=E2; EL=22; CL=25; SRC=$ROOT/$G3/12b/source/gemma3-12b; GPU=3 ;;
    g27)   MODEL=gemma3-27b; NG=gemma3-27b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/27b/e44/gemma3-27b
           crun=$run; E_ON=E2; C_E_ON=E2; EL=26; CL=31; SRC=$ROOT/$G3/27b/source/gemma3-27b; GPU=2 ;;
    q27)   MODEL=qwen3.8-27b-dm; NG=qwen3.8-27b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.8-27b-dm
           crun=$run; E_ON=E2; C_E_ON=E2; EL=28; CL=33; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.8-27b-dm; GPU=2,3 ;;
    *) echo "unknown key $1" >> $L/pm3_all.log; exit 2 ;;
  esac
  E_RUN=$ROOT/$run; C_RUN=$ROOT/$crun
  DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
}
tq() {  # tq KEY SPLIT NAME SELFCHECK_ES(- for none) TAG ARMS...
  local key=$1 sp=$2 NAME=$3 es=$4 tag=$5; shift 5
  local arms="$*" XE gen="" a f rc=0 log=$L/pm3_$key.log
  setm $key
  if [ $sp = tune ]; then XE="scripts/e44_eval_extra.py"; else XE="scripts/review_checks/e44_split_test.py extra"; fi
  if [ "$es" != "-" ]; then
    MODEL=$MODEL CODE_PATH=$CODE TABLE_CODE=$ROOT/Iteration_01/code SOURCE_DIR=$SRC E_RUN=$E_RUN C_RUN=$C_RUN E_ON=$E_ON C_E_ON=$C_E_ON \
      EL=$EL CL=$CL ES="$es" KS="$KS" GPU=$GPU SPLIT=$sp TAG=$tag STAGE=selfcheck bash $RUNNER >> $log 2>&1 ||
      { echo "pm3 tq $key $NAME selfcheck FAILED $(date '+%T')" >> $L/pm3_all.log; return 11; }
  fi
  local COMMON=(--source-run $SRC --run-dir $E_RUN --e-on $E_ON --c-run-dir $C_RUN --c-e-on $C_E_ON --split $sp
                --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
  for a in $arms; do f=${a//:/--}; f=${f//+/-and-}; f=${f//./-}; gen="$gen ${f}_$sp.json"; done
  local W=$run/eval_extra/judge_${sp}_$NAME
  { echo "=== $(date '+%F %T') $key $sp $NAME generate $arms"
    "${DX[@]}" $PYM $XE generate "${COMMON[@]}" --arms $arms &&
    "${DX[@]}" $PYM $XE export "${COMMON[@]}" --gen $gen --name $NAME &&
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_$sp/$NAME.json $HOST/$W/judge_input.json &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json; } >> $log 2>&1 || rc=1
  echo "pm3 tq $key $NAME exit=$rc $(date '+%T')" >> $L/pm3_all.log
  return $rc
}
mm() {  # mm KEY OUTDIR ARMS...
  local key=$1 out=$2; shift 2
  setm $key
  mkdir -p Iteration_01/capability/$out; rm -f Iteration_01/capability/$out/$NG.json
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$GPU -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $ROOT truth_dec \
    $PYM -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model $NG --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_14042_20261120.json \
    --out Iteration_01/capability/$out/$NG.json --arms "$@" > $L/pm3_mm_${out}_$NG.log 2>&1
  echo "pm3 mmlu $out $key exit=$? $(date '+%T')" >> $L/pm3_all.log
}
must() { local d o=""; for d in $1; do o="$o conf:E$EL.w.p$d"; done; echo $o; }
cf() {  # cf KEY OUT ES...: confidence on the test split for the targets of these E strengths
  local key=$1 out=$2; shift 2
  setm $key
  "${DX[@]}" $PYM scripts/review_checks/ff_test_confidence.py $out --split test --source-run $SRC --run-dir $E_RUN --e-on $E_ON \
    --c-run-dir $C_RUN --c-e-on $C_E_ON --el $EL --cl $CL --es "$@" --ks $KS >> $L/pm3_$key.log 2>&1
  echo "pm3 conf $key $out exit=$? $(date '+%T')" >> $L/pm3_all.log
}
MID="1.25 1.5 1.75"
rm -f $L/pm3_all.log $L/PM3_DONE
until [ -e $L/PM2_DONE ]; do sleep 30; done
echo "$(date '+%F %T') start" >> $L/pm3_all.log
( setm g27; tq g27 test dmust_test - - $(must "$MID")
  cf g27 pmmid_test $MID
  mm g27 ideaA_full x conf_E2 "conf_E2:E#.w.p0.25" "conf_E2:E#.w.p0.5" "conf_E2:E#.w.p0.75" "conf_E2:E#.w.p1" ) &
( cf g12 pmmid_test $MID ) &
wait
touch $L/PM3_DONE
