#!/bin/bash
# 2026-11-22 20 h, exploration (Claude Opus 5.5): judge the CAA answers of one work dir with DeepSeek Pro, the way
# iti_judge.sh judges the ITI answers. The judge input and its cells are made before (caa_grid.sh for tune, caa_chain.sh
# for test: scripts/review_checks/e44_positions.py gen export, the engine's own export, then judge_two_axis_run.py cells);
# here: the verdict cache of an existing work dir of the model's usual run is merged first (an answer that an earlier arm
# already gave, e.g. every no-write answer, gets the verdict it already has; the merge stops on conflicting replies),
# then DeepSeek Pro (two-axis rubric), not started in peak hours or too close to them, and the single-judge verdicts.
# usage: caa_judge.sh MODEL WORK LIKE
#   WORK  the CAA work dir, relative to the project root (run_caa/eval_extra/judge_<split>_<name>_<split>)
#   LIKE  tag of the usual run's work dir <run>/eval_extra/judge_<split>_<LIKE>_<split> (e.g. pospromptl16)
# Markers: Iteration_01/baselines/caa/<MODEL>/CAA_JUDGE_<basename of WORK>_DONE / _FAILED; log lines in caa_all.log.
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
MODEL=$1; W=$2; LIKE=$3
[ -n "$LIKE" ] || { echo "usage: caa_judge.sh MODEL WORK LIKE"; exit 2; }
B=Iteration_01/baselines/caa/$MODEL
NAME=$(basename "$W")
MK=$HOST/$B/CAA_JUDGE_$NAME
log=$HOST/$B/caa_judge_$NAME.log
say() { echo "$(date '+%F %T') caa judge $MODEL $NAME: $*" >> $L/caa_all.log; }
rm -f ${MK}_DONE ${MK}_FAILED
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
case "$NAME" in judge_tune_*_tune) SPLIT=tune ;; judge_test_*_test) SPLIT=test ;; *) fail "work dir name is not judge_<split>_<name>_<split>" ;; esac
plan=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["run_dir"], m["code"])' $HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json $MODEL) || fail "model not in the config"
read -r RUN MCODE <<< "$plan"
[ -n "$RUN" ] && [ -n "$MCODE" ] || fail "model not in the config"
case "$W" in $B/run_caa/eval_extra/*) ;; *) fail "the work dir is not inside the model's CAA mirror" ;; esac
CODE=$ROOT/$MCODE
OLD=$RUN/eval_extra/judge_${SPLIT}_${LIKE}_$SPLIT
[ -s $HOST/$W/judge_input.json ] && [ -s $HOST/$W/judge2/cells.jsonl ] || fail "no judge input or cells in $W"
[ -s $HOST/$OLD/judge_input.json ] || fail "no judge input in $OLD"
J=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
   -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
say "start, like $LIKE"
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
