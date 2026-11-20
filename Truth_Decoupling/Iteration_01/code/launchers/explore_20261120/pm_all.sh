#!/bin/bash
# 2026-11-20 afternoon chain: data for per-model E strengths (the lead, 15:24: the method is shared, the strength may
# differ by model). Same scripts and arguments as dose_mid_all.sh / idea_a.sh / q9_high.sh; only arms, splits, names.
#  A. E written on top of the prompt-only baseline (prompt conf_E2 = must-answer + the honest sentence), TUNE split
#     (the test split exists: ideaA4_test, ideaA4b_test), so that a strength can be chosen per model without the test set.
#  C. higher E strengths with the N1 arms (single E, E + C at 0.25 / 0.5 / 1 / 2), tune + test, selfcheck before each
#     split: Qwen3.8-27B at 2.5 / 3, Qwen3.5-9B at 3.5 / 4 (their MMLU cost up to 2 is small).
#  D. the mean-difference direction in place of the learned E, normal prompt, test split, strengths 1 and 2 (kinds mdg
#     and md in separate calls: a kind that cannot be built fails alone).
#  M. MMLU on the full test split (option probability) for the arms of A (and the test-split strengths 1.5 / 2 of the
#     two Qwens) and of C.
# All TruthfulQA generation comes first (the judge starts at 18:00), MMLU last. A failing step does not stop the lane.
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
    *) echo "unknown key $1" >> $L/pm_all.log; exit 2 ;;
  esac
  E_RUN=$ROOT/$run; C_RUN=$ROOT/$crun
  DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
}
onp() { local d o=""; for d in $1; do o="$o conf_E2:E$EL.w.p$d"; done; echo $o; }
n1arms() { local s k o=""; for s in $1; do o="$o x:E$EL.w.p$s"; for k in $KS; do o="$o x:E$EL.w.p$s+C$CL.w.m$k"; done; done; echo $o; }
tq() {  # tq KEY SPLIT NAME SELFCHECK_ES(- for none) TAG ARMS...
  local key=$1 sp=$2 NAME=$3 es=$4 tag=$5; shift 5
  local arms="$*" XE gen="" a f rc=0 log=$L/pm_$key.log
  setm $key
  if [ $sp = tune ]; then XE="scripts/e44_eval_extra.py"; else XE="scripts/review_checks/e44_split_test.py extra"; fi
  if [ "$es" != "-" ]; then
    MODEL=$MODEL CODE_PATH=$CODE TABLE_CODE=$ROOT/Iteration_01/code SOURCE_DIR=$SRC E_RUN=$E_RUN C_RUN=$C_RUN E_ON=$E_ON C_E_ON=$C_E_ON \
      EL=$EL CL=$CL ES="$es" KS="$KS" GPU=$GPU SPLIT=$sp TAG=$tag STAGE=selfcheck bash $RUNNER >> $log 2>&1 ||
      { echo "pm tq $key $NAME selfcheck FAILED $(date '+%T')" >> $L/pm_all.log; return 11; }
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
  echo "pm tq $key $NAME exit=$rc $(date '+%T')" >> $L/pm_all.log
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
    --out Iteration_01/capability/$out/$NG.json --arms "$@" > $L/pm_mm_${out}_$NG.log 2>&1
  echo "pm mmlu $out $key exit=$? $(date '+%T')" >> $L/pm_all.log
}
md() {  # md KEY: the mean-difference direction in place of E, test split
  setm $1
  tq $1 test mdg_test - - x:E$EL.mdg.p1 x:E$EL.mdg.p2
  tq $1 test md_test - - x:E$EL.md.p1 x:E$EL.md.p2
}
SMALL="0.25 0.5 0.75 1"; WIDE="0.25 0.5 0.75 1 1.5 2"
rm -f $L/pm_all.log $L/PM_TQ_DONE $L/PM_DONE
echo "$(date '+%F %T') start" >> $L/pm_all.log
( setm llama; tq llama tune ideaA6_tune - - $(onp "$SMALL")
  setm q9;    tq q9 tune ideaA6_tune - - $(onp "$WIDE")
  setm q9;    A=$(n1arms "3.5 4"); for sp in tune test; do tq q9 $sp dosehigh2_$sp "3.5 4" dosehigh2 $A; done
  md llama; md q9
  touch $L/PM_TQ_LANE0
  mm llama ideaA_full x conf_E2 "conf_E2:E#.w.p0.25" "conf_E2:E#.w.p0.5" "conf_E2:E#.w.p0.75" "conf_E2:E#.w.p1"
  mm q9 ideaA_full x conf_E2 "conf_E2:E#.w.p0.25" "conf_E2:E#.w.p0.5" "conf_E2:E#.w.p0.75" "conf_E2:E#.w.p1" "conf_E2:E#.w.p1.5" "conf_E2:E#.w.p2"
  mm q9 full_high x "x:E#.w.p2.5" "x:E#.w.p3" "x:E#.w.p3.5" "x:E#.w.p4" ) &
( setm g27;   tq g27 tune ideaA6_tune - - $(onp "$SMALL")
  setm g12;   tq g12 tune ideaA6_tune - - $(onp "$SMALL")
  md g27; md g12
  touch $L/PM_TQ_LANE1
  mm g27 ideaA_full x conf_E2 "conf_E2:E#.w.p0.25" "conf_E2:E#.w.p0.5" "conf_E2:E#.w.p0.75" "conf_E2:E#.w.p1"
  mm g12 ideaA_full x conf_E2 "conf_E2:E#.w.p0.25" "conf_E2:E#.w.p0.5" "conf_E2:E#.w.p0.75" "conf_E2:E#.w.p1" ) &
( setm q27;   tq q27 tune ideaA6_tune - - $(onp "$WIDE")
  setm q27;   A=$(n1arms "2.5 3"); for sp in tune test; do tq q27 $sp dosehigh_$sp "2.5 3" dosehigh $A; done
  md q27
  touch $L/PM_TQ_LANE2
  mm q27 ideaA_full x conf_E2 "conf_E2:E#.w.p0.25" "conf_E2:E#.w.p0.5" "conf_E2:E#.w.p0.75" "conf_E2:E#.w.p1" "conf_E2:E#.w.p1.5" "conf_E2:E#.w.p2"
  mm q27 full_high x "x:E#.w.p2.5" "x:E#.w.p3" ) &
wait
touch $L/PM_DONE
