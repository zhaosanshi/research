#!/bin/bash
# Informational smoke of the prompt ladder, Claude Opus 5.5, 2026-11-18 (Beijing). GPT's re-check of 0dadab5
# (discussion/prompt_ladder/2026-11-18_gpt_recheck.md, section 1, row 1) allows one 8-question smoke per model in a new
# directory, run serially. The helper will change after that re-check, so these directories are for checking the device
# layout only; the real smoke and generation follow in a fresh OUT_ROOT once the helper is fixed and re-checked.
# Serial, GPU 1 only; Qwen3.8-27B falls back to GPUs 2,3 in its own OUT_ROOT if its one-card control does not reproduce.
set -u
PL=/data/sanshizhao/Truth_Decoupling/Iteration_01/promptladder
LAD=/workspace/Truth_Decoupling/Iteration_01/promptladder/ladder_prompts_draft_20261118.json
OR=/workspace/Truth_Decoupling/Iteration_01/promptladder/smoke_r2_20261118
cd "$PL" || exit 2
for k in llama qwen9 g12 g27 qwen27; do
  echo "##### $k dryrun $(date '+%F %T')"
  KEY=$k LADDER=$LAD STAGE=dryrun OUT_ROOT=$OR bash ladder_run.sh > ops/smoke_r2_${k}_dryrun.log 2>&1
  echo "dryrun $k exit=$?"
  echo "##### $k smoke $(date '+%F %T')"
  KEY=$k LADDER=$LAD STAGE=smoke GPU=1 OUT_ROOT=$OR bash ladder_run.sh > ops/smoke_r2_${k}.log 2>&1
  rc=$?
  echo "smoke $k exit=$rc"
  if [ "$k" = qwen27 ] && [ "$rc" -ne 0 ]; then
    echo "##### $k smoke on GPUs 2,3 $(date '+%F %T')"
    KEY=$k LADDER=$LAD STAGE=smoke GPU=2,3 OUT_ROOT=${OR}_2gpu bash ladder_run.sh > ops/smoke_r2_${k}_2gpu.log 2>&1
    echo "smoke $k 2gpu exit=$?"
  fi
done
python3 -B ops/smoke_r2_check.py > ops/smoke_r2_check.txt 2>&1
echo "check exit=$?"
date '+%F %T' > ops/smoke_r2_20261118.DONE
