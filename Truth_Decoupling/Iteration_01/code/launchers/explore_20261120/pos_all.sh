#!/bin/bash
# 2026-11-20 night, exploration: where does the E write have to act? The trained E written only from the last prompt
# token on (core rule "gen": the question tokens stay untouched) instead of the training rule "content". On MMLU this
# rule barely costs accuracy (capability design, section 8.2); the TruthfulQA side was never measured.
# Per model, test split, through the evaluation-only entry scripts/review_checks/e44_positions.py (single writes only):
#   normal prompt:                     x:E.w.p{1,2,3,4}          at "gen"
#   must-answer + the honest sentence: conf_E2:E.w.p{0.5,1,2}    at "gen"
# then export (name posgen_test), judge cells, and MMLU on the 1,000-item list with the same arms (@gen).
# Small models on GPU 0 now; the two 27B models after pm3_gemma.sh has released GPUs 2 and 3.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
TF=$ROOT/envs/tf519/bin/python
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
setm() {
  case $1 in
    llama) MODEL=llama3.1-8b; NG=llama3.1-8b; CODE=$ROOT/code_e45_20261112; PYM=python3; run=results_e44_factorial_das/llama3.1-8b_side_E2
           crun=results_e44_factorial_das/llama3.1-8b; E_ON=E2; C_E_ON=E1; EL=11; CL=19; SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full; GPU=0 ;;
    q9)    MODEL=qwen3.5-9b; NG=qwen3.5-9b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.5-9b_E2
           crun=$E44/qwen3.5-9b; E_ON=E2; C_E_ON=E1; EL=15; CL=16; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.5-9b; GPU=0 ;;
    g12)   MODEL=gemma3-12b; NG=gemma3-12b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/12b/e44/gemma3-12b
           crun=$run; E_ON=E2; C_E_ON=E2; EL=22; CL=25; SRC=$ROOT/$G3/12b/source/gemma3-12b; GPU=0 ;;
    g27)   MODEL=gemma3-27b; NG=gemma3-27b; CODE=$ROOT/Iteration_01/code; PYM=python3; run=$G3/27b/e44/gemma3-27b
           crun=$run; E_ON=E2; C_E_ON=E2; EL=26; CL=31; SRC=$ROOT/$G3/27b/source/gemma3-27b; GPU=2 ;;
    q27)   MODEL=qwen3.8-27b-dm; NG=qwen3.8-27b; CODE=$ROOT/code_ff_explore_20261113b; PYM=$TF; run=$E44/qwen3.8-27b-dm
           crun=$run; E_ON=E2; C_E_ON=E2; EL=28; CL=33; SRC=$ROOT/results_newmodels_ff_20261112/source/qwen3.8-27b-dm; GPU=2,3 ;;
    *) echo "unknown key $1" >> $L/pos_all.log; exit 2 ;;
  esac
  E_RUN=$ROOT/$run; C_RUN=$ROOT/$crun
  DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
      -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
}
one() {  # one KEY: TruthfulQA at "gen", export, cells; then MMLU (1,000 items)
  local key=$1 arms="" gen="" a f s rc=0 log=$L/pos_$1.log
  setm $key
  local PX="scripts/review_checks/e44_positions.py gen"
  local COMMON=(--source-run $SRC --run-dir $E_RUN --e-on $E_ON --c-run-dir $C_RUN --c-e-on $C_E_ON --split test
                --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
  for s in 1 2 3 4; do arms="$arms x:E$EL.w.p$s"; done
  for s in 0.5 1 2; do arms="$arms conf_E2:E$EL.w.p$s"; done
  for a in $arms; do f=${a//:/--}; f=${f//./-}; gen="$gen ${f}-at-gen_test.json"; done
  local NAME=posgen_test W=$run/eval_extra/judge_test_posgen_test
  { echo "=== $(date '+%F %T') $key generate at gen:$arms"
    "${DX[@]}" $PYM $PX generate "${COMMON[@]}" --arms $arms &&
    "${DX[@]}" $PYM $PX export "${COMMON[@]}" --gen $gen --name $NAME &&
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_test/$NAME.json $HOST/$W/judge_input.json &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json; } >> $log 2>&1 || rc=1
  echo "pos tq $key exit=$rc $(date '+%T')" >> $L/pos_all.log
  mkdir -p Iteration_01/capability/posgen; rm -f Iteration_01/capability/posgen/$NG.json
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$GPU -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $ROOT truth_dec \
    $PYM -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model $NG --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_1000_20261119.json \
    --out Iteration_01/capability/posgen/$NG.json --arms x "x:E#.w.p1@gen" "x:E#.w.p2@gen" "x:E#.w.p3@gen" "x:E#.w.p4@gen" \
      conf_E2 "conf_E2:E#.w.p0.5@gen" "conf_E2:E#.w.p1@gen" "conf_E2:E#.w.p2@gen" > $L/pos_mm_$NG.log 2>&1
  echo "pos mmlu $key exit=$? $(date '+%T')" >> $L/pos_all.log
}
rm -f $L/pos_all.log $L/POS_SMALL_DONE $L/POS_DONE
echo "$(date '+%F %T') start" >> $L/pos_all.log
( one llama; one q9; one g12; touch $L/POS_SMALL_DONE ) &
( until [ -e $L/PM3_DONE ]; do sleep 60; done; one q27; one g27 ) &
wait
touch $L/POS_DONE
