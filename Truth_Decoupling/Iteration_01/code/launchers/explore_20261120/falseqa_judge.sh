#!/bin/bash
# 2026-11-23 23:4x, exploration, queue L section 11 (executor E, Claude Opus 5.5): judge export and DeepSeek Pro judging
# of one model's 22 FalseQA arms, one set at a time: the development set (100 questions) first, then the held-out set
# (600). The same route and prompt as (QA)^2 (qaqa_run.sh / h22z_model.sh): the unchanged export of newsets_gen.py
# through scripts/newsets/ns_h22.py (task falseqa, all 700 bank questions, the 22 arms), the unchanged two-axis judge
# scripts/review_checks/judge_two_axis_run.py (cells, llm --judge pro, single --judge pro; DeepSeek Pro with the flow's
# own config, non-thinking), reference = the bank's gold list (the export's first GOLD_CAP). The export is split by
# scripts/newsets/falseqa_judge_split.py into the work dirs newsets/judge/falseqa/<model>/test_full700_dev and
# test_full700_heldout (judge_input.json each); diag/falseqa_read.py reads both. No reply cache to merge (new questions).
# Steps: 1 falseqa_read.py check MODEL (the 22 full files); 2 export (once) to newsets/judge/falseqa/<model>/export_full700.json;
# 3 split into the work dir of SET; 4 cells; 5 SET heldout only after the dev set is judged (marker); 6 the clock guard of
# qaqa_run.sh (no start in peak hours or within 15 minutes of them; stopped by timeout at 5 minutes before the next
# peak), then llm and single in the background. Markers FALSEQA_JUDGE_<MODEL>_<SET>_DONE / _FAILED.
# dry: steps 1-4 and the guard only; prints the cell counts and the commands it would run, and stops before any API call.
# usage (host): falseqa_judge.sh MODEL dev|heldout [dry]
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
MODEL=$1; SET=$2; DRY=$3
case $MODEL in llama3.1-8b|qwen3.5-9b|qwen3.8-27b|gemma3-12b|gemma3-27b) ;; *) echo "usage: falseqa_judge.sh MODEL dev|heldout [dry]"; exit 2 ;; esac
case $SET in dev|heldout) ;; *) echo "usage: falseqa_judge.sh MODEL dev|heldout [dry]"; exit 2 ;; esac
case "$DRY" in ""|dry) ;; *) echo "third argument: dry or nothing"; exit 2 ;; esac
MK=$L/FALSEQA_JUDGE_${MODEL}_$SET; log=$L/falseqa_judge_$MODEL.log
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') falseqa judge $MODEL $SET${DRY:+ (dry)}: $*" | tee -a $L/falseqa_all.log; }
fail() { say "FAILED: $*"; [ -z "$DRY" ] && touch ${MK}_FAILED; exit 1; }
[ -z "$DRY" ] && rm -f ${MK}_DONE ${MK}_FAILED
CODE=$ROOT/Iteration_01/code
NH=$CODE/scripts/newsets/ns_h22.py
SPL=$CODE/scripts/newsets/falseqa_judge_split.py
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
BANK=$NS/falseqa/falseqa_full_bank_20261123.jsonl
GEN=$NS/gen_prompt/falseqa/$MODEL/test
EXP=$NS/judge/falseqa/$MODEL/export_full700.json; HEXP=$HNS/judge/falseqa/$MODEL/export_full700.json
W=$NS/judge/falseqa/$MODEL/test_full700_$SET; HW=$HNS/judge/falseqa/$MODEL/test_full700_$SET
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B)
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
JK=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE
    -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $CODE truth_dec)
# ---- 1. the 22 full generation files
"${RD[@]}" Iteration_01/diag/falseqa_read.py check $MODEL >> $log 2>&1 || fail "generation incomplete (falseqa_read.py check $MODEL; see $log)"
ARMS=$("${RD[@]}" -c '
import sys
sys.path[:0] = ["Iteration_01/diag"]
import falseqa_read as F
print(" ".join(F.roster(sys.argv[1]).values()))' $MODEL 2>> $log)
[ $(echo $ARMS | wc -w) = 22 ] || fail "roster"
# ---- 2. export (once)
if [ ! -e $HEXP ]; then
  "${J[@]}" python3 -I $NH falseqa none export --task falseqa --model $MODEL --items $BANK --split test \
      --gen-dir $GEN --only-arms $ARMS --out $EXP >> $log 2>&1 || fail "export"
fi
# ---- 3. split, 4. cells
"${J[@]}" python3 -I $SPL $EXP $BANK $SET $W/judge_input.json >> $log 2>&1 || fail "split"
[ -e $HW/judge2/cells.jsonl ] || "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $W --generation judge_input.json >> $log 2>&1 || fail "cells"
CNT=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
print("questions %d, conditions %d, cells %d, empty outputs %d, distinct question-answer requests %d" % (
    m["n_questions"], len(m["conditions"]), m["n_cells"], m["n_empty_outputs"], m["n_unique_semantic_requests"]))' $HW/judge2/meta.json) || fail "meta"
say "work dir ready: $CNT"
# ---- 5. order: the held-out set only after the dev set is judged
if [ $SET = heldout ] && [ ! -e $L/FALSEQA_JUDGE_${MODEL}_dev_DONE ]; then
  if [ -n "$DRY" ]; then say "would stop here: the dev set is not judged yet (no FALSEQA_JUDGE_${MODEL}_dev_DONE)"; else fail "the dev set is not judged yet"; fi
fi
# ---- 6. clock guard, then the judge in the background
secs=$(TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))')
if [ -n "$DRY" ]; then
  say "dry run: clock guard gives ${secs}s (a start needs at least 900); would run: judge_two_axis_run.py llm $W --judge pro (timeout ${secs}s), then single $W --judge pro; stopped before any API call"
  exit 0
fi
if [ "$secs" -lt 900 ]; then fail "peak hours or too close to them (${secs}s left), judge not started"; fi
( if "${JK[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $W --judge pro >> $log 2>&1 &&
     "${JK[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $W --judge pro > $HW/single_pro_stdout.md 2>> $log; then
    say "judged"; touch ${MK}_DONE
  else
    say "FAILED: judge"; touch ${MK}_FAILED; exit 1
  fi ) > /dev/null 2>&1 < /dev/null &
say "judge started in the background (guard ${secs}s)"
