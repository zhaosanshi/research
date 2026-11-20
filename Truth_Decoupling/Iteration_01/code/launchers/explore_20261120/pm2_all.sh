#!/bin/bash
# 2026-11-20 second chain (after pm_all.sh): what the same-answer-count comparisons need at E strengths other than 1 / 2.
#  1. must-answer prompt + E at the strength (base conf), TEST split: generation, export, judge cells (work dir
#     judge_test_dmust_test). These are the answers "prompt B" re-asks with and the pool of the strong form.
#  2. the model's confidence in its own answers (ff_test_confidence.py, unchanged) for the targets of these strengths,
#     one file per strength group (a group that fails a check fails alone): confidence_pmmid / pmhigh / pmhigh2.
# Same scripts and arguments as pm_all.sh; waits for its PM_DONE (cards).
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
           crun=$run; E_ON=E2; C_E_ON=E2; EL=22; CL=25; SRC=$ROOT/$G3/12b/source/gemma3-12b; GPU=1 ;;
    g27)   MODEL=gemma3-27b; NG=gemma3-27b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/27b/e44/gemma3-27b
           crun=$run; E_ON=E2; C_E_ON=E2; EL=26; CL=31; SRC=$ROOT/$G3/27b/source/gemma3-27b; GPU=1 ;;
    q27)   MODEL=qwen3.8-27b-dm; NG=qwen3.8-27b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.8-27b-dm
           crun=$run; E_ON=E2; C_E_ON=E2; EL=28; CL=33; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.8-27b-dm; GPU=2,3 ;;
    *) echo "unknown key $1" >> $L/pm2_all.log; exit 2 ;;
  esac
  E_RUN=$ROOT/$run; C_RUN=$ROOT/$crun
  DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
}
tq() {  # tq KEY SPLIT NAME SELFCHECK_ES(- for none) TAG ARMS...
  local key=$1 sp=$2 NAME=$3 es=$4 tag=$5; shift 5
  local arms="$*" XE gen="" a f rc=0 log=$L/pm2_$key.log
  setm $key
  if [ $sp = tune ]; then XE="scripts/e44_eval_extra.py"; else XE="scripts/review_checks/e44_split_test.py extra"; fi
  if [ "$es" != "-" ]; then
    MODEL=$MODEL CODE_PATH=$CODE TABLE_CODE=$ROOT/Iteration_01/code SOURCE_DIR=$SRC E_RUN=$E_RUN C_RUN=$C_RUN E_ON=$E_ON C_E_ON=$C_E_ON \
      EL=$EL CL=$CL ES="$es" KS="$KS" GPU=$GPU SPLIT=$sp TAG=$tag STAGE=selfcheck bash $RUNNER >> $log 2>&1 ||
      { echo "pm2 tq $key $NAME selfcheck FAILED $(date '+%T')" >> $L/pm2_all.log; return 11; }
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
  echo "pm2 tq $key $NAME exit=$rc $(date '+%T')" >> $L/pm2_all.log
  return $rc
}
must() { local d o=""; for d in $1; do o="$o conf:E$EL.w.p$d"; done; echo $o; }
cf() {  # cf KEY OUT ES...: confidence on the test split for the targets of these E strengths
  local key=$1 out=$2; shift 2
  setm $key
  "${DX[@]}" $PYM scripts/review_checks/ff_test_confidence.py $out --split test --source-run $SRC --run-dir $E_RUN --e-on $E_ON \
    --c-run-dir $C_RUN --c-e-on $C_E_ON --el $EL --cl $CL --es "$@" --ks $KS >> $L/pm2_$key.log 2>&1
  echo "pm2 conf $key $out exit=$? $(date '+%T')" >> $L/pm2_all.log
}
MID="1.25 1.5 1.75"
rm -f $L/pm2_all.log $L/PM2_TQ_DONE $L/PM2_DONE
until [ -e $L/PM_DONE ]; do sleep 60; done
echo "$(date '+%F %T') start" >> $L/pm2_all.log
( setm llama; tq llama test dmust_test - - $(must "$MID")
  setm q9;    tq q9 test dmust_test - - $(must "$MID 2.5 3 3.5 4")
  touch $L/PM2_TQ_LANE0
  cf llama pmmid_test $MID
  cf q9 pmmid_test $MID; cf q9 pmhigh_test 2.5 3; cf q9 pmhigh2_test 3.5 4 ) &
( setm g27;   tq g27 test dmust_test - - $(must "$MID")
  setm g12;   tq g12 test dmust_test - - $(must "$MID")
  touch $L/PM2_TQ_LANE1
  cf g27 pmmid_test $MID; cf g12 pmmid_test $MID ) &
( setm q27;   tq q27 test dmust_test - - $(must "$MID 2.5 3")
  touch $L/PM2_TQ_LANE2
  cf q27 pmmid_test $MID; cf q27 pmhigh_test 2.5 3 ) &
wait
touch $L/PM2_DONE
