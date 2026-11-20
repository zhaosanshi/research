#!/bin/bash
# Pilot (2026-11-20): the honest sentence stays in the prompt (must-answer + sentence, the run's prompt conf_E2) and E is
# written on top at small strengths (0.25 / 0.5 / 0.75 / 1). Per model: TruthfulQA test split (generation, export,
# cells, AllenAI as a free early read; DeepSeek Pro later, off-peak) and MMLU (the 1,000-item list, option probability).
# Generation / export / cells / AllenAI call the same scripts with the same arguments as ff_explore_ec.sh.
# usage: idea_a.sh KEY...   (llama q9 g12 on GPU 0; g27 on GPU 1; q27 on GPUs 2,3); WAIT=<marker file> waits for it first
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
TF=$ROOT/envs/tf519/bin/python
DOSES="0.25 0.5 0.75 1"; NAME=ideaA4_test
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3
one() {  # one KEY NGKEY CODE PYM E_RUN C_RUN E_ON C_E_ON EL SRC GPU
  local key=$1 ng=$2 CODE=$ROOT/$3 PYM=$4 run=$5 E_RUN=$ROOT/$5 C_RUN=$ROOT/$6 E_ON=$7 C_E_ON=$8 EL=$9 SRC=$ROOT/${10} GPU=${11}
  local DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
            -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
            -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
  local XE="scripts/review_checks/e44_split_test.py extra" arms="" gen="" s f rc=0 log=$L/ideaA_$key.log
  local COMMON=(--source-run $SRC --run-dir $E_RUN --e-on $E_ON --c-run-dir $C_RUN --c-e-on $C_E_ON --split test
                --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42)
  for s in $DOSES; do arms="$arms conf_E2:E$EL.w.p$s"; f="conf_E2--E$EL-w-p$s"; gen="$gen ${f//./-}_test.json"; done
  local W=$run/eval_extra/judge_test_$NAME
  { echo "=== $(date '+%F %T') $key generate$arms"
    "${DX[@]}" $PYM $XE generate "${COMMON[@]}" --arms $arms &&
    "${DX[@]}" $PYM $XE export "${COMMON[@]}" --gen $gen --name $NAME &&
    mkdir -p $HOST/$W/judge2/cache && cp $HOST/$run/eval_extra/judge_test/$NAME.json $HOST/$W/judge_input.json &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json &&
    "${DX[@]}" python3 scripts/allenai_tqa_judge.py --cells $ROOT/$W/judge2/cells.jsonl --out $ROOT/$W/judge2/allenai.jsonl \
        --truth-model $ROOT/models/allenai__truthfulqa-truth-judge-llama2-7B --info-model $ROOT/models/allenai__truthfulqa-info-judge-llama2-7B &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge allenai > $HOST/$W/single_allenai_stdout.md; } >> $log 2>&1 || rc=1
  echo "ideaA tq $key exit=$rc $(date '+%T')" >> $L/ideaA_all.log
  mkdir -p Iteration_01/capability/ideaA4; rm -f Iteration_01/capability/ideaA4/$ng.json
  docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CUDA_VISIBLE_DEVICES=$GPU -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w $ROOT truth_dec \
    $PYM -I Iteration_01/code/scripts/capability/cap_mcq_sweep.py run --models Iteration_01/code/configs/newsets/frozen_models_20261119.json \
    --model $ng --items data_external/mmlu_20261119/all/test-00000-of-00001.parquet --ids Iteration_01/capability/sets/mmlu_1000_20261119.json \
    --out Iteration_01/capability/ideaA4/$ng.json --arms x conf_E2 "conf_E2:E#.w.p0.25" "conf_E2:E#.w.p0.5" "conf_E2:E#.w.p0.75" "conf_E2:E#.w.p1" >> $log 2>&1
  echo "ideaA mmlu $key exit=$? $(date '+%T')" >> $L/ideaA_all.log
}
[ -z "${WAIT:-}" ] || until [ -e "$WAIT" ]; do sleep 30; done
for k in "$@"; do
  case $k in
    llama) one llama llama3.1-8b code_e45_20261112 python3 results_e44_factorial_das/llama3.1-8b_side_E2 results_e44_factorial_das/llama3.1-8b E2 E1 11 results_qa_das_v1u/llama3.1-8b/full 0 ;;
    q9)    one q9 qwen3.5-9b code_ff_explore_20261113b $TF $E44/qwen3.5-9b_E2 $E44/qwen3.5-9b E2 E1 15 results_newmodels_ff_20261112/source/qwen3.5-9b 0 ;;
    g12)   one g12 gemma3-12b Iteration_01/code python3 $G3/12b/e44/gemma3-12b $G3/12b/e44/gemma3-12b E2 E2 22 $G3/12b/source/gemma3-12b 0 ;;
    g27)   one g27 gemma3-27b Iteration_01/code python3 $G3/27b/e44/gemma3-27b $G3/27b/e44/gemma3-27b E2 E2 26 $G3/27b/source/gemma3-27b 1 ;;
    q27)   one q27 qwen3.8-27b code_ff_explore_20261113b $TF $E44/qwen3.8-27b-dm $E44/qwen3.8-27b-dm E2 E2 28 results_newmodels_ff_20261112/source/qwen3.8-27b-dm 2,3 ;;
    *) echo "unknown key $k" >> $L/ideaA_all.log ;;
  esac
done
touch $L/IDEAA_$(echo "$@" | tr ' ' '_')_DONE
