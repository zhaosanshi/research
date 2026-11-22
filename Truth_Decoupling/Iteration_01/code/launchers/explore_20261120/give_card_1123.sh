#!/bin/bash
# 2026-11-23 01:5x, queue I, lead session 01:46: give GPU 2 or 3 to the queue-J self-judge scores once, at the end of the
# capability pass that is running there (never in the middle of one). The two running cap_h22_model.sh scripts
# (Qwen3.8-27B on GPU 3, Gemma-3-27B on GPU 2) are paused (SIGSTOP; their current docker exec keeps running); the first one
# whose current pass has ended is ended (its lock on the card is released) and the other one continues (SIGCONT). After
# TJ_BIG_SCORES_DONE (or 75 minutes) the ended model's cap_h22_model.sh is started again on the same card: the passes
# already written are skipped by the programs themselves (existing outputs), the rest is generated.
cd /data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
say() { echo "$(date '+%F %T') give card: $*" >> $L/n1s_all.log; }
QB=$(pgrep -f "bash Iteration_01/logs/newsets_20261119/cap_h22_model.sh q27 3" | head -1)
GB=$(pgrep -f "bash .*cap_h22_model.sh g27 2" | head -1)
[ -n "$QB" ] && [ -n "$GB" ] || { say "did not find both scripts (q27 $QB, g27 $GB)"; exit 1; }
kill -STOP $QB $GB
say "paused cap_h22_model.sh q27 (GPU 3, pid $QB) and g27 (GPU 2, pid $GB) at their running pass"
busy() { ps --ppid $1 -o stat= | grep -qv '^Z'; }
while true; do
  if ! busy $QB; then key=q27; gpu=3; out=$QB; keep=$GB; break; fi
  if ! busy $GB; then key=g27; gpu=2; out=$GB; keep=$QB; break; fi
  sleep 10
done
kill -KILL $out; kill -CONT $keep
say "the pass of $key on GPU $gpu has ended: its script is ended and GPU $gpu is free for the self-judge scores; the other model continues"
t0=$(date +%s)
until [ -e $L/TJ_BIG_SCORES_DONE ] || [ $(( $(date +%s) - t0 )) -ge 4500 ]; do sleep 30; done
say "GPU $gpu back to $key ($( [ -e $L/TJ_BIG_SCORES_DONE ] && echo TJ_BIG_SCORES_DONE || echo 75 minutes))"
bash $L/cap_h22_model.sh $key $gpu > $L/caph22_${key}_resume.out 2>&1
