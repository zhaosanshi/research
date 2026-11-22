#!/bin/bash
# 2026-11-23 00:0x, queue I 7.1, decision of the lead session (00:02): the (QA)^2 judge of Llama-3.1-8B and Gemma-3-27B
# stopped at the cache-merge conflict check (one request key with different replies in two earlier work dirs). As the
# project does since 11-20 (one old cache per new work dir), the judge is rerun in a NEW work dir test_full570_h22n1 with
# the reply cache of the S0 pass only (test_full570_h22z, which already holds the test_full570 cache merged then). Same
# judge input and cells as the stopped dir test_full570_h22n (copied; that dir is kept). The conflict check stays on.
# usage: n1s2_qaqa_rejudge.sh MODEL
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
MODEL=$1; [ -n "$MODEL" ] || exit 2
MK=$L/N1S2_${MODEL}_qaqa1; log=$L/n1s2_${MODEL}_rejudge.log
say() { echo "$(date '+%F %T') n1s2 $MODEL: $*" >> $L/n1s_all.log; }
rm -f ${MK}_JUDGE_DONE ${MK}_JUDGE_FAILED
J0=Iteration_01/newsets/judge/qaqafull/$MODEL
OLD=$HOST/$J0/test_full570_h22n; HW=$HOST/$J0/test_full570_h22n1; W=$ROOT/$J0/test_full570_h22n1
[ -s $OLD/judge_input.json ] && [ -s $OLD/judge2/cells.jsonl ] || { say "rejudge: no judge input in the stopped dir"; touch ${MK}_JUDGE_FAILED; exit 1; }
[ -e $HW ] && { say "rejudge: $HW exists"; touch ${MK}_JUDGE_FAILED; exit 1; }
mkdir -p $HW/judge2/cache && cp $OLD/judge_input.json $HW/judge_input.json
JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$ROOT/Iteration_01/code
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $ROOT/Iteration_01/code truth_dec)
{ "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation judge_input.json &&
  [ "$(wc -l < $HW/judge2/cells.jsonl)" = "$(wc -l < $OLD/judge2/cells.jsonl)" ] &&
  "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $W/judge2/cache $ROOT/$J0/test_full570_h22z/judge2/cache &&
  "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $W/judge2/cache/conflicts.json; } >> $log 2>&1 || { say "rejudge: cells or single-cache merge refused (see $log)"; touch ${MK}_JUDGE_FAILED; exit 1; }
secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))')
[ "$secs" -ge 900 ] || { say "rejudge not started: peak hours"; touch ${MK}_JUDGE_FAILED; exit 1; }
if "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $W --judge pro >> $log 2>&1 &&
   "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $W --judge pro > $HW/single_pro_stdout.md 2>> $log; then
  say "rejudged qaqa (single cache test_full570_h22z)"; touch ${MK}_JUDGE_DONE
else
  say "FAILED: rejudge qaqa"; touch ${MK}_JUDGE_FAILED; exit 1
fi
