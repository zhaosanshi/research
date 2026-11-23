#!/bin/bash
# 2026-11-23 23:3x, exploration, queue L section 11 (discussion/work_queue/2026-11-23_queue_l.md; executor E, Claude Opus 5.5):
# FalseQA fresh confirmation, the 22 arms of one model on one card (lock), through scripts/newsets/ns_h22.py (the h22
# prompt file configs/prompts/e44_eval_extra_h22_v1.json; ns_task.py takes the new task name "falseqa" as it is, no
# registry edit), writes at the rule "prompt" (ns_positions.py, NS_POS_TWO_WRITE=1, NS_POS_PLAIN_NAMES=1).
# Settings (none chosen here): EL, S (main), CL from diag/n1_on_sentence.py plan (the main table's records, as
# n1s2_model.sh); s_sel from newsets/h22/n1s_choice_<model>.json. The 22 arms:
#   plain x:            x | x:E<EL>.w.p<S> | x:E<EL>.w.p<S>+C<CL>.w.m<c> | x:C<CL>.w.m<c>
#   plain + sentence:   x_E2 | x_E2:E<EL>.w.p<s_sel> | x_E2:E<EL>.w.p<s_sel>+C<CL>.w.m<c> | x_E2:C<CL>.w.m<c>
#   must answer + sent: conf_E2 | conf_E2:E<EL>.w.p<s_sel> (TE of section 9)            c = 0.25 0.5 1 2
# Bank Iteration_01/newsets/falseqa/falseqa_full_bank_20261123.jsonl (700 rows: 100 dev first, then 600 held-out; all
# split "test"). Gen dir newsets/gen_prompt/falseqa/<model>/test (the qaqafull pattern); check dir
# newsets/gen_prompt/h22chk/falseqa/<model>/test.
# Steps (any failure stops: marker FALSEQA_<MODEL>_FAILED):
#   0. CPU: diag/falseqa_read.py check (bank, pair ids, the id file) before the card is taken;
#   1. the entry changes nothing: three existing (QA)^2 arms regenerated through the same entry on the first 16
#      (QA)^2 questions (x:E<EL>.w.p<S>+C<CL>.w.m0.25 and conf_E2 = gen_prompt/qaqafull/...; x_E2:E<EL>.w.p<s_sel>+C<CL>.w.m0.25
#      = gen_prompt/h22/qaqafull/...) must equal the archived files (ns_h22.py compare, token ids and end state);
#   2. the 22 arms on the first 16 FalseQA questions (first 2 batches; all 16 are development questions);
#   3. write records = the (QA)^2 run's (scripts/newsets/n1s_check.py parts): every E write against the (QA)^2 file of
#      the same base row and strength, every C write against the C part of x:E<EL>.w.p<S>+C<CL>.w.m<c> of (QA)^2;
#   MODE smoke stops here (marker FALSEQA_<MODEL>_SMOKE_DONE);
#   4. MODE full: the 22 arms on all 700 questions (newsets_gen.py skips a complete file of the same inputs);
#   5. the same setting twice: the first 16 rows of every full file = the step-2 file (ns_h22.py compare); step 3 on the
#      full files; then the card is released (marker FALSEQA_<MODEL>_GPU_DONE). No judge call here (falseqa_judge.sh).
# Cards 1 or 2 only, through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock, retried every minute) and ours plus
# others at most 66560 MiB on the card (checked before the start, as n1s2_model.sh).
# usage (host): falseqa_model.sh KEY GPU smoke|full      KEY = llama | q9 | q27 | g12 | g27
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=$1; GPU=$2; MODE=$3
[ -n "$MODE" ] || { echo "usage: falseqa_model.sh KEY GPU smoke|full"; exit 2; }
case $MODE in smoke|full) ;; *) echo "MODE must be smoke or full"; exit 2 ;; esac
case $GPU in 1|2) LIM=66560 ;; *) echo "cards 1 or 2 only"; exit 2 ;; esac
TF=$ROOT/envs/tf519/bin/python
case $KEY in
  llama) MODEL=llama3.1-8b; PYM=python3; NEED=20000 ;;
  q9)    MODEL=qwen3.5-9b; PYM=$TF; NEED=22000 ;;
  q27)   MODEL=qwen3.8-27b; PYM=$TF; NEED=62000 ;;
  g12)   MODEL=gemma3-12b; PYM=python3; NEED=30000 ;;
  g27)   MODEL=gemma3-27b; PYM=python3; NEED=62000 ;;
  *) echo "unknown key $KEY"; exit 2 ;;
esac
MK=$L/FALSEQA_$MODEL; log=$L/falseqa_$MODEL.log
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') falseqa $MODEL $MODE: $*" >> $L/falseqa_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
rm -f ${MK}_FAILED
[ $MODE = full ] && rm -f ${MK}_GPU_DONE
[ $MODE = smoke ] && rm -f ${MK}_SMOKE_DONE
NH=$ROOT/Iteration_01/code/scripts/newsets/ns_h22.py
NC=$ROOT/Iteration_01/code/scripts/newsets/n1s_check.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
BANK=$NS/falseqa/falseqa_full_bank_20261123.jsonl
QBANK=$NS/bank/qaqa_full_bank_20261122.jsonl
for f in code/scripts/newsets/ns_h22.py code/scripts/newsets/n1s_check.py code/configs/prompts/e44_eval_extra_h22_v1.json \
         newsets/falseqa/falseqa_full_bank_20261123.jsonl diag/falseqa_read.py; do
  [ -s $HOST/Iteration_01/$f ] || fail "missing $f"
done
RD=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -B)
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
read -r EL S SH CL <<< "$("${RD[@]}" Iteration_01/diag/n1_on_sentence.py plan $MODEL 2>> $log)"
[ -n "$CL" ] || fail "plan"
SS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["strength_dot"])' $HNS/h22/n1s_choice_$MODEL.json) || fail "no s_sel"
[ -n "$SS" ] || fail "no s_sel"
D() { echo "$1" | tr . -; }
tok() { echo "$1" | sed -e 's/:/--/g' -e 's/+/-and-/g' -e 's/\./-/g'; }
NONE="x x_E2 conf_E2"
WRITE="x:E$EL.w.p$S"
for c in 0.25 0.5 1 2; do WRITE="$WRITE x:E$EL.w.p$S+C$CL.w.m$c"; done
for c in 0.25 0.5 1 2; do WRITE="$WRITE x:C$CL.w.m$c"; done
WRITE="$WRITE x_E2:E$EL.w.p$SS"
for c in 0.25 0.5 1 2; do WRITE="$WRITE x_E2:E$EL.w.p$SS+C$CL.w.m$c"; done
for c in 0.25 0.5 1 2; do WRITE="$WRITE x_E2:C$CL.w.m$c"; done
WRITE="$WRITE conf_E2:E$EL.w.p$SS"
n=$(echo $NONE $WRITE | wc -w); [ $n = 22 ] || fail "expected 22 arms, got $n"
say "plan E$EL main $S, s_sel $SS, C$CL; 22 arms"
# ---- 0. bank checks (CPU, before the card)
"${RD[@]}" Iteration_01/diag/falseqa_read.py check >> $log 2>&1 || fail "bank check"
# ---- the card: lock (retry every minute), then the memory rule
LOCK=$HNS/selfjudge/.gpu$GPU.lock
exec 9>"$LOCK" || fail "cannot open $LOCK"
until flock -n 9; do sleep 60; done
while true; do
  USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1 == g {print $2}')
  [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $GPU"
  [ $((USED + NEED)) -le $LIM ] && break
  say "GPU $GPU holds $USED MiB, $NEED more would exceed $LIM; waiting"; sleep 60
done
say "start on GPU $GPU ($USED MiB in use)"
DQ=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
gen() {  # gen TASK ITEMS OUT none|prompt ARMS... [--first-batches 2]
  local task=$1 items=$2 out=$3 pos=$4; shift 4
  "${DQ[@]}" $PYM -I $NH $task $pos gen --models $M --model $MODEL --task $task --items $items --split test --out $out --arms "$@" >> $log 2>&1
}
first() { ls $HOST/${1#$ROOT/}/s????????????.$2.json 2>/dev/null | head -1 | sed "s#^$HOST#$ROOT#"; }
QA=$NS/gen_prompt/qaqafull/$MODEL/test; QO=$NS/gen_prompt/h22/qaqafull/$MODEL/test
# ---- 1. the entry changes nothing ((QA)^2, first 16 questions)
EC=$NS/gen_prompt/h22chk/falseqa_entry/qaqafull/$MODEL/test
A1=x:E$EL.w.p$S+C$CL.w.m0.25; A2=x_E2:E$EL.w.p$SS+C$CL.w.m0.25
gen qaqafull $QBANK $EC none conf_E2 --first-batches 2 || fail "entry check generation (no write)"
gen qaqafull $QBANK $EC prompt $A1 $A2 --first-batches 2 || fail "entry check generation (writes)"
for pr in "conf_E2 $QA" "$A1 $QA" "$A2 $QO"; do
  set -- $pr
  s=$(first $EC $(tok $1)); [ -n "$s" ] || fail "entry check file $1"
  "${DC[@]}" $NH compare $s $2/$(tok $1).json >> $log 2>&1 || fail "entry check $1: differs from the archived (QA)^2 file"
done
say "entry checks passed (3 archived (QA)^2 arms reproduced on 16 questions)"
# ---- write records against the (QA)^2 run's
parts_of() {  # parts_of FILE ARM
  local f=$1 a=$2 base=${2%%:*} body=${2#*:} c
  case $body in
    E*+C*) c=${body##*.w.m}
           if [ $base = x ]; then "${DC[@]}" $NC parts $f $QA/x--E$EL-w-p$(D $S).json $S $S E >> $log 2>&1 || return 1
           else "${DC[@]}" $NC parts $f $QO/x_E2--E$EL-w-p$(D $SS).json $SS $SS E >> $log 2>&1 || return 1; fi
           "${DC[@]}" $NC parts $f $QA/x--E$EL-w-p$(D $S)-and-C$CL-w-m$(D $c).json 1 1 C >> $log 2>&1 ;;
    E*)    if [ $base = x ]; then "${DC[@]}" $NC parts $f $QA/x--E$EL-w-p$(D $S).json $S $S E >> $log 2>&1
           else "${DC[@]}" $NC parts $f $QO/x_E2--E$EL-w-p$(D $SS).json $SS $SS E >> $log 2>&1; fi ;;
    C*)    c=${body##*.w.m}
           "${DC[@]}" $NC parts $f $QA/x--E$EL-w-p$(D $S)-and-C$CL-w-m$(D $c).json 1 1 C >> $log 2>&1 ;;
    *) return 1 ;;
  esac
}
# ---- 2. the 22 arms on the first 16 FalseQA questions
FC=$NS/gen_prompt/h22chk/falseqa/$MODEL/test
gen falseqa $BANK $FC none $NONE --first-batches 2 || fail "16-question generation (no write)"
gen falseqa $BANK $FC prompt $WRITE --first-batches 2 || fail "16-question generation (writes)"
for a in $NONE $WRITE; do [ -n "$(first $FC $(tok $a))" ] || fail "16-question file of $a missing"; done
# ---- 3. write records
for a in $WRITE; do parts_of $(first $FC $(tok $a)) $a || fail "write record of $a (16 questions)"; done
say "16-question pass of the 22 arms done; write records = (QA)^2's"
if [ $MODE = smoke ]; then
  flock -u 9; exec 9>&-
  say "smoke done; card released"
  touch ${MK}_SMOKE_DONE
  exit 0
fi
# ---- 4. the 22 arms on 700 questions
FO=$NS/gen_prompt/falseqa/$MODEL/test
gen falseqa $BANK $FO none $NONE || fail "generation (no write)"
gen falseqa $BANK $FO prompt $WRITE || fail "generation (writes)"
# ---- 5. twice, write records on the full files
for a in $NONE $WRITE; do
  "${DC[@]}" $NH compare $(first $FC $(tok $a)) $FO/$(tok $a).json >> $log 2>&1 || fail "twice check $a"
done
for a in $WRITE; do parts_of $FO/$(tok $a).json $a || fail "write record of $a"; done
"${RD[@]}" Iteration_01/diag/falseqa_read.py check $MODEL >> $log 2>&1 || fail "generation check"
flock -u 9; exec 9>&-
say "22 arms on 700 questions generated and checked; card released"
touch ${MK}_GPU_DONE
