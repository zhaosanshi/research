#!/bin/bash
# DeepSeek Pro on the extra work dirs of 2026-11-20 (the prompt-plus-small-write pilot, five models; Qwen3.5-9B at E 2.5 /
# 3), off-peak only: waits for START (default 12:00:30), every lane stops at END (default 13:57); a rerun resumes from the
# reply caches. A lane whose cells do not exist yet waits for them (until 60 minutes before END). Same calls as
# dose_mid_judge.sh.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
START=${START:-120030}; END=${END:-13:57}
lane() {  # lane KEY CODE WORK OLD_CACHE_WORK
  local key=$1 CODE=$ROOT/$2 W=$3 oc=$4/judge2/cache log=$L/extra_judge_$1.log
  local DX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
  until [ -f $HOST/$W/judge2/cells.jsonl ] || [ $(( $(date -d "$END" +%s) - $(date +%s) )) -lt 3600 ]; do sleep 60; done
  local secs=$(( $(date -d "$END" +%s) - $(date +%s) ))
  if [ ! -f $HOST/$W/judge2/cells.jsonl ] || [ $secs -lt 120 ]; then echo "extra judge $key: no cells or outside the window" >> $L/extra_judge2.log; return; fi
  { echo "=== $(date '+%F %T') $key: merge $oc, then pro (at most $secs s)"
    { [ ! -f $HOST/$oc/judge2_cache_deepseek-v4-pro.jsonl ] || "${DX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$W/judge2/cache $ROOT/$oc; } &&
    "${DX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$W/judge2/cache/conflicts.json &&
    "${DX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$W --judge pro &&
    "${DX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge pro > $HOST/$W/single_pro_stdout.md; } >> $log 2>&1
  echo "extra judge $key exit=$? $(date '+%T')" >> $L/extra_judge2.log
}
rm -f $L/EXTRA_JUDGE2_DONE
until [ "$(date +%H%M%S)" -ge "$START" ]; do sleep 20; done
E44=results_newmodels_ff_20261112/e44; G3=Iteration_01/results/ff/gemma3; LR=results_e44_factorial_das/llama3.1-8b_side_E2


lane ideaA2_q9 code_ff_explore_20261113b $E44/qwen3.5-9b_E2/eval_extra/judge_test_ideaA4b_test $E44/qwen3.5-9b_E2/judge_test_testD_E_L15 &
lane ideaA2_q27 code_ff_explore_20261113b $E44/qwen3.8-27b-dm/eval_extra/judge_test_ideaA4b_test $E44/qwen3.8-27b-dm/judge_test_testD_E_L28 &




wait
touch $L/EXTRA_JUDGE2_DONE
