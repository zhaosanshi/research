#!/bin/bash
# 2026-11-22 05:4x, exploration: judge ITI answers (scripts/newsets/iti_baseline.py gen) with the same judge inputs as
# the project's own arms: the judge input is built by iti_baseline.py export from an existing judge input of the same
# model and split (question, gold and incorrect lists copied), the verdict cache of that existing work dir is merged
# first (an answer that an earlier arm already gave gets the verdict it already has; the merge stops on conflicting
# replies, as launchers/explore_20261120/posm.sh does), then DeepSeek Pro (two-axis rubric, not started in peak hours).
# usage: iti_judge.sh MODEL SPLIT NAME LIKE_NAME GEN_FILE [GEN_FILE ...]
#   NAME       tag of the new work dir   <run>/eval_extra/judge_<SPLIT>_<NAME>_<SPLIT>
#   LIKE_NAME  tag of the existing one   <run>/eval_extra/judge_<SPLIT>_<LIKE_NAME>_<SPLIT>   (e.g. pospromptl16)
#   GEN_FILE   ITI generation files, paths relative to the project root
# Markers: ITI_JUDGE_<MODEL>_<SPLIT>_<NAME>_DONE / _FAILED. Rules: discussion/work_queue/2026-11-22_queue.md section 4.
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
IT=$ROOT/Iteration_01/code/scripts/newsets/iti_baseline.py
MODEL=$1; SPLIT=$2; NAME=$3; LIKE=$4; shift 4
[ -n "$1" ] || { echo "usage: iti_judge.sh MODEL SPLIT NAME LIKE_NAME GEN_FILE [GEN_FILE ...]"; exit 2; }
MK=$L/ITI_JUDGE_${MODEL}_${SPLIT}_${NAME}
log=$L/iti_judge_${MODEL}_${SPLIT}_$NAME.log
say() { echo "$(date '+%F %T') iti judge $MODEL $SPLIT $NAME: $*" >> $L/iti_all.log; }
rm -f ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
plan=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["run_dir"], m["code"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || fail "model not in the config"
read -r RUN MCODE <<< "$plan"
[ -n "$RUN" ] && [ -n "$MCODE" ] || fail "model not in the config"
CODE=$ROOT/$MCODE
W=$RUN/eval_extra/judge_${SPLIT}_${NAME}_$SPLIT
OLD=$RUN/eval_extra/judge_${SPLIT}_${LIKE}_$SPLIT
[ -s $HOST/$OLD/judge_input.json ] || fail "no judge input in $OLD"
GENS=""
for g in "$@"; do [ -s $HOST/$g ] || fail "no generation file $g"; GENS="$GENS $ROOT/$g"; done
J=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
   -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
say "start: $# generation files, like $LIKE"
if [ ! -e $HOST/$W/judge_input.json ]; then
  "${J[@]}" python3 -I $IT export --like $ROOT/$OLD/judge_input.json --gen $GENS --out $ROOT/$W/judge_input.json >> $log 2>&1 || fail "export"
fi
if [ ! -e $HOST/$W/judge2/cells.jsonl ]; then
  "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "cells"
fi
mkdir -p $HOST/$W/judge2/cache
if [ -f $HOST/$OLD/judge2/cache/judge2_cache_deepseek-v4-pro.jsonl ]; then
  "${J[@]}" python3 scripts/review_checks/judge2_cache_merge.py $ROOT/$W/judge2/cache $ROOT/$OLD/judge2/cache >> $log 2>&1 || fail "cache merge"
  "${J[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $ROOT/$W/judge2/cache/conflicts.json >> $log 2>&1 || fail "conflicting cached replies"
fi
secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))')
[ "$secs" -ge 900 ] || fail "peak hours or too close to them (${secs}s left), judge not started"
"${J[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $ROOT/$W --judge pro >> $log 2>&1 || fail "judge"
"${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $ROOT/$W --judge pro > $HOST/$W/single_pro_stdout.md 2>> $log || fail "single"
say "judged"
touch ${MK}_DONE
