#!/bin/bash
# 2026-11-23 03:4x, exploration, queue K section 6 (discussion/work_queue/2026-11-23_queue_k.md; the last one): the keep
# term started from the main E's saved direction (e44_keepfix.py --keep-init-from), lambda 0.03 / 0.1, and lambda 0 from
# the same start as a reference (never chosen); everything else as section 1 (seed and question order, keep set and draws,
# A = 5.0264, layer 16, 10 epochs, 16 held out). First the entry's own reproduction check after the change: lambda 0
# without a start must equal the main E element for element (run dir llama3.1-8b_keepfix_l0r), else stop.
# keepfix_llama.sh and keepfix_llama_lam.sh stay as they ran; this launcher is keepfix_llama_lam.sh with a run dir per lambda
# (Iteration_01/results/keepfix/llama3.1-8b/e44/llama3.1-8b_keepfix_<tag>, frozen answers copied from the main run, sha256
# checked) and the newsets model config of that run dir (written here into Iteration_01/newsets/keepfix/).
# Per lambda: train; usable = signcheck passes with --min-gap 0.718 (half of the main E's gap; --l0 = the lambda 0 run from
# the same start) and the cosine with the main E is at least 0.5; an unusable lambda gets no generation and no judge. The
# reference (lambda 0 from the main E) is generated after the two others.
# Then, if any lambda is usable: the main E at half strength (1.75) on (QA)^2 570 and PopQA / TriviaQA tune and P alone on
# the fact tune splits (keepfix_main dirs, from the main run, read only); per usable lambda: TruthfulQA tune (x_E2 + E_k at
# 3.5 and at the aligned s', each with C19 at 0.25 0.5 1 2; x + E_k at 3.5 and s'), PopQA / TriviaQA tune (x_E2 + E_k at
# 3.5 and s'), (QA)^2 570 (the ten x_E2 arms; judged: the 100 pilot questions); checks as keepfix_llama.sh; exports; cells;
# DeepSeek Pro in the background (one chain per lambda, one old cache per work dir, off peak only).
# Markers KEEPFIX_<MODEL>_w0_<tag>_USABLE / _UNUSABLE / _GEN_DONE, KEEPFIX_<MODEL>_w0_<tag>_<judge>_DONE / _FAILED,
# KEEPFIX_<MODEL>_w0_DONE / _FAILED.   usage: keepfix_llama_w0.sh GPU TRAIN_NEED_MIB
cd /data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
GPU=$1; TNEED=$2
[ -n "$TNEED" ] || { echo "usage: keepfix_llama_w0.sh GPU TRAIN_NEED_MIB"; exit 2; }
LAMS="0:w0 0.03:w0p03 0.1:w0p1"; REF=w0
MODEL=llama3.1-8b; CODE=$ROOT/code_keepfix_20261123; PYM=python3
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
MAINR=results_e44_factorial_das/llama3.1-8b_side_E2; CRUN=results_e44_factorial_das/llama3.1-8b
KR=Iteration_01/results/keepfix/llama3.1-8b/e44; R0R=$KR/llama3.1-8b_keepfix_l0r; RREF=$KR/llama3.1-8b_keepfix_w0
EL=16; CL=19; S=3.5; SH=1.75
KF=$ROOT/Iteration_01/code/keepfix_20261123
NSK=Iteration_01/newsets/keepfix
KS=$NSK/keep_set_$MODEL.json
AMT=$MAINR/eval_extra/gen/x_E2--E16-w-p3-5-at-prompt_tune.json
NEWP=$ROOT/Iteration_01/code/configs/prompts/e44_eval_extra_h22_v1.json
WR=$ROOT/Iteration_01/code/scripts/review_checks/e44_positions_h22.py
NH=$ROOT/Iteration_01/code/scripts/newsets/ns_h22.py
NC=$ROOT/Iteration_01/code/scripts/newsets/n1s_check.py
M=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
NS=$ROOT/Iteration_01/newsets; HNS=$HOST/Iteration_01/newsets
BANKQ=$NS/bank/qaqa_full_bank_20261122.jsonl
MK=$L/KEEPFIX_${MODEL}_w0; log=$L/keepfix_${MODEL}_w0.log
say() { echo "$(date '+%F %T') keepfix $MODEL w0: $*" >> $L/keepfix_all.log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
tok() { echo "$1" | sed -e 's/:/--/g' -e 's/+/-and-/g' -e 's/\./-/g'; }
first() { ls $HOST/${1#$ROOT/}/s*.$2.json 2>/dev/null | head -1 | sed "s#^$HOST#$ROOT#"; }
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1 == g {print $2}'; }
DC=(docker exec -u 1000:1001 -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= -w $ROOT truth_dec python3 -I)
DT=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES=$GPU -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
DQ=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NS_POS_TWO_WRITE=1 -e NS_POS_PLAIN_NAMES=1
    -e CUDA_VISIBLE_DEVICES=$GPU -w $ROOT truth_dec)
J=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e CUDA_VISIBLE_DEVICES=
   -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$ROOT/Iteration_01/code -w $ROOT/Iteration_01/code truth_dec)
KEEPA=(--keep-set $ROOT/$KS --keep-amount-from $ROOT/$AMT --keep-prompts-extra $NEWP)
TA=(--source-run $SRC --e-on E2 --family E --layers $EL --val-n 16 --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
passed() { "${DC[@]}" -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('passed') is True else 1)" $1; }
budget() {  # seconds until 5 minutes before the next weekday peak (Beijing time), at most 6 h; 0 inside a peak
  TZ=Asia/Shanghai python3 -c '
import time
def peak(t):
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    return lt.tm_wday < 5 and (9 * 60 <= m < 12 * 60 or 14 * 60 <= m < 18 * 60)
now = int(time.time()); t = now
while not peak(t) and t - now < 8 * 86400:
    t += 60
print(0 if peak(now) else max(0, min(t - now - 300, 6 * 3600)))'
}
judge_one() {  # judge_one MARKER CODE WORK(container) OLD_WORK(container): merge one old cache, Pro off peak, single
  local mk=$1 code=$2 w=$3 old=$4 hw=$HOST/${3#$ROOT/} secs rc jlog=$L/$(basename $1).log
  local JX=(docker exec -u 1000:1001 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$code
            -e DEEPSEEK_API_KEY_FILE=$ROOT/logs/.deepseek_key -w $code truth_dec)
  rm -f ${mk}_DONE ${mk}_FAILED
  [ -e $hw/judge2/cells.jsonl ] || { say "judge $(basename $mk): no cells"; touch ${mk}_FAILED; return 1; }
  if [ ! -e ${mk}_MERGED ]; then
    "${JX[@]}" python3 scripts/review_checks/judge2_cache_merge.py $w/judge2/cache $old/judge2/cache >> $jlog 2>&1 || { touch ${mk}_FAILED; return 1; }
    "${JX[@]}" python3 -c "
import json, os, sys
p = sys.argv[1]
k = json.load(open(p)).get('keys', {}) if os.path.exists(p) else {}
n = sum(len(v) for v in k.values()) if isinstance(k, dict) else len(k)
sys.exit('the merged caches give %d request keys different replies: stop' % n if n else 0)" $w/judge2/cache/conflicts.json >> $jlog 2>&1 || { say "judge $(basename $mk): cache conflict"; touch ${mk}_FAILED; return 1; }
    echo "$old" > ${mk}_MERGED
  fi
  while true; do
    secs=$(budget)
    if [ "$secs" -lt 900 ]; then sleep 300; continue; fi
    "${JX[@]}" timeout -k 60 --signal=INT $secs python3 scripts/review_checks/judge_two_axis_run.py llm $w --judge pro >> $jlog 2>&1
    rc=$?
    if [ $rc = 0 ]; then break; fi
    if [ $rc != 124 ]; then say "FAILED: judge $(basename $mk) exit $rc"; touch ${mk}_FAILED; return 1; fi
    say "judge $(basename $mk): stopped before the peak; resumes after it"
  done
  "${JX[@]}" python3 scripts/review_checks/judge_two_axis_run.py single $w --judge pro > $hw/single_pro_stdout.md 2>> $jlog || { touch ${mk}_FAILED; return 1; }
  say "judged $(basename $mk)"; touch ${mk}_DONE
}
rm -f ${MK}_FAILED ${MK}_DONE
# ---------------------------------------------------------------------------- the card
if [ "$GPU" = 0 ] && ! { [ -e $L/CAA_GRID_gemma3-12b_DONE ] || [ -e $L/CAA_GRID_gemma3-12b_FAILED ]; }; then fail "GPU 0 only after the CAA grid"; fi
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu$GPU.lock
exec 9>"$LOCK" || fail "cannot open $LOCK"
flock 9 || fail "cannot lock $LOCK"            # blocks in the lock's queue until the holder releases it
if [ "$GPU" = 1 ]; then LIM=66560; else LIM=73000; fi
while true; do
  USED=$(mem $GPU)
  [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $GPU"
  [ $((USED + TNEED)) -le $LIM ] && break
  say "GPU $GPU holds $USED MiB, $TNEED more would exceed $LIM; waiting"; sleep 60
done
CAP=$((LIM - USED - 1024))
say "start on GPU $GPU ($USED MiB in use, limit $LIM, cap for PyTorch allocations $CAP MiB)"
# ---------------------------------------------------------------------------- the entry changed: lambda 0 without a start again
mkdir -p $HOST/$R0R || fail "mkdir $R0R"
[ -s $HOST/$R0R/frozen_train.json ] || cp -p $HOST/$MAINR/frozen_train.json $HOST/$R0R/frozen_train.json || fail "copy frozen_train to l0r"
[ "$(sha256sum $HOST/$R0R/frozen_train.json | cut -c1-64)" = "$(sha256sum $HOST/$MAINR/frozen_train.json | cut -c1-64)" ] || fail "copied frozen_train of l0r differs"
if [ ! -s $HOST/$R0R/dirs/E_L$EL.pt ]; then
  "${DT[@]}" $PYM scripts/e44_keepfix.py train "${TA[@]}" --run-dir $ROOT/$R0R "${KEEPA[@]}" --keep-lambda 0 --keep-mem-cap-mib $CAP >> $log 2>&1 || fail "lambda 0 (no start) training"
fi
"${DT[@]}" $PYM scripts/e44_keepfix.py compare $ROOT/$R0R/dirs/E_L$EL.pt $ROOT/$MAINR/dirs/E_L$EL.pt --min-cos 0.9999 > $HOST/$NSK/compare_l0r_$MODEL.json 2>> $log \
    || fail "after the change, lambda 0 without a start does not reproduce the main E: stop"
say "after the change, lambda 0 without a start reproduces the main E: $(cut -c300-520 $HOST/$NSK/compare_l0r_$MODEL.json)"
# ---------------------------------------------------------------------------- train the three runs from the main E
USABLE=""
for lt in $LAMS; do
  LAM=${lt%%:*}; TAG=${lt#*:}; R=$KR/llama3.1-8b_keepfix_$TAG
  mkdir -p $HOST/$R || fail "mkdir $R"
  for f in frozen_train.json frozen_tune.json frozen_test.json; do
    [ -s $HOST/$R/$f ] || cp -p $HOST/$MAINR/$f $HOST/$R/$f || fail "copy $f to $TAG"
    [ "$(sha256sum $HOST/$R/$f | cut -c1-64)" = "$(sha256sum $HOST/$MAINR/$f | cut -c1-64)" ] || fail "copied $f of $TAG differs from the main run's"
  done
  if [ ! -s $HOST/$R/dirs/E_L$EL.pt ]; then
    "${DT[@]}" $PYM scripts/e44_keepfix.py train "${TA[@]}" --run-dir $ROOT/$R "${KEEPA[@]}" --keep-lambda $LAM --keep-mem-cap-mib $CAP \
        --keep-init-from $ROOT/$MAINR/dirs/E_L$EL.pt >> $log 2>&1 || fail "lambda $LAM training"
  fi
  "${DT[@]}" $PYM scripts/e44_keepfix.py compare $ROOT/$R/dirs/E_L$EL.pt $ROOT/$MAINR/dirs/E_L$EL.pt --min-cos 0.5 > $HOST/$NSK/compare_${TAG}_$MODEL.json 2>> $log; c_ok=$?
  "${DT[@]}" $PYM scripts/e44_keepfix.py signcheck $ROOT/$R/dirs/E_L$EL.pt --l0 $ROOT/$RREF/dirs/E_L$EL.pt --min-gap 0.718 > $HOST/$NSK/signcheck_${MODEL}_$TAG.json 2>> $log; s_ok=$?
  rm -f ${MK}_${TAG}_USABLE ${MK}_${TAG}_UNUSABLE
  if [ $c_ok = 0 ] && [ $s_ok = 0 ]; then
    SPX=$("${DT[@]}" $PYM scripts/e44_keepfix.py aligned $ROOT/$R/dirs/E_L$EL.pt --keep-amount-from $ROOT/$AMT 2>> $log)
    [[ $SPX =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "aligned strength $TAG"
    echo "$SPX" > $HOST/$NSK/aligned_s_${MODEL}_$TAG.txt
    touch ${MK}_${TAG}_USABLE
    if [ $TAG = $REF ]; then REFLT=$lt; else USABLE="$USABLE $lt"; fi
    say "lambda $LAM trained: usable, aligned s' $SPX; $(cut -c1-160 $HOST/$NSK/signcheck_${MODEL}_$TAG.json)"
  else
    touch ${MK}_${TAG}_UNUSABLE
    say "lambda $LAM trained: NOT usable (compare exit $c_ok, signcheck exit $s_ok); $(cut -c1-160 $HOST/$NSK/signcheck_${MODEL}_$TAG.json)"
  fi
done
if [ -z "$USABLE" ]; then
  flock -u 9; exec 9>&-
  say "neither lambda 0.03 nor 0.1 from the main E is usable: nothing generated, nothing judged; card released"; touch ${MK}_DONE; exit 0
fi
say "usable:$USABLE; reference: ${REFLT:-not usable}"
USABLE="$USABLE $REFLT"
gf() { local task=$1 pos=$2 models=$3; shift 3
       "${DQ[@]}" $PYM -I $NH $task $pos gen --models $models --model $MODEL --task $task --items $NS/bank/${task}_bank_20261119.jsonl --split tune "$@" >> $log 2>&1; }
gq() { local models=$1; shift
       "${DQ[@]}" $PYM -I $NH qaqafull prompt gen --models $models --model $MODEL --task qaqafull --items $BANKQ --split test "$@" >> $log 2>&1; }
qexport() {  # qexport GEN_DIR WORK ARMS...: export the 570, keep the 100 pilot questions, make the cells
  local gd=$1 w=$2; shift 2
  if [ ! -e $HOST/${w#$ROOT/}/judge_input.json ]; then
    "${J[@]}" python3 -I $NH qaqafull none export --task qaqafull --model $MODEL --items $BANKQ --split test \
        --gen-dir $gd --only-arms "$@" --out $w/judge_input_full570.json >> $log 2>&1 || return 1
    "${DC[@]}" $KF/keepfix_qaqa_pilot.py $w/judge_input_full570.json $w/judge_input.json >> $log 2>&1 || return 1
  fi
  [ -e $HOST/${w#$ROOT/}/judge2/cells.jsonl ] || "${J[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $w --generation judge_input.json >> $log 2>&1
}
ES=x_E2:E$EL.w.p$S; EH=x_E2:E$EL.w.p$SH
QM=$NS/gen_prompt/h22/qaqafull/$MODEL/test
O_kf=$ROOT/$MAINR/eval_extra/judge_tune_n1s_tune; O_kfx=$ROOT/$MAINR/eval_extra/judge_tune_pospromptl16_tune
O_q=$NS/judge/qaqafull/$MODEL/test_full570_h22n1
say "balance before: $(bash $L/balance.sh)"
# ---------------------------------------------------------------------------- the main E at half strength, P alone (once)
QH=$NS/gen_prompt/keepfix_main/qaqafull/$MODEL/test; QHC=$NS/gen_prompt/keepfix_main_chk/qaqafull/$MODEL/test
QWH=$NS/judge/qaqafull/$MODEL/keepfix_main_pilot100
HARMS="$EH"; for c in 0.25 0.5 1 2; do HARMS="$HARMS $EH+C$CL.w.m$c"; done
gq $M --out $QHC --arms $EH+C$CL.w.m0.25 --first-batches 2 || fail "qaqa half check generation"
gq $M --out $QH --arms $HARMS || fail "qaqa half generation"
"${DC[@]}" $NH compare $(first $QHC $(tok $EH+C$CL.w.m0.25)) $QH/$(tok $EH+C$CL.w.m0.25).json >> $log 2>&1 || fail "qaqa half twice"
"${DC[@]}" $NC parts $QH/$(tok $EH).json $QM/$(tok $ES).json $SH $S E >> $log 2>&1 || fail "qaqa half E part"
for c in 0.25 0.5 1 2; do
  f=$QH/$(tok $EH+C$CL.w.m$c).json; mf=$QM/$(tok $ES+C$CL.w.m$c).json
  "${DC[@]}" $NC parts $f $mf $SH $S E >> $log 2>&1 && "${DC[@]}" $NC parts $f $mf $SH $S C >> $log 2>&1 || fail "qaqa half parts $c"
done
qexport $QH $QWH $HARMS || fail "qaqa half export / cells"
( exec 9>&-; judge_one ${MK}_main_qaqah $ROOT/Iteration_01/code $QWH $O_q ) > /dev/null 2>&1 < /dev/null &
for task in popqa triviaqa; do
  FM=$NS/gen_prompt/keepfix_main/$task/$MODEL/tune
  gf $task prompt $M --out $FM --arms $ES $EH || fail "fact main E $task"
  gf $task none $M --out $FM --arms x_E2 || fail "fact P $task"
  for s in $S $SH; do
    "${DC[@]}" $NC parts $FM/$(tok x_E2:E$EL.w.p$s).json $NS/gen_prompt/$task/$MODEL/tune/x--E$EL-w-p$(echo $S | tr . -).json $s $S E >> $log 2>&1 || fail "fact main E part $task $s"
  done
done
say "main E at half strength and P alone generated, checked; (QA)^2 pilot judge started"
# ---------------------------------------------------------------------------- per usable lambda
for lt in $USABLE; do
  LAM=${lt%%:*}; TAG=${lt#*:}; R=$KR/llama3.1-8b_keepfix_$TAG
  SP=$(cat $HOST/$NSK/aligned_s_${MODEL}_$TAG.txt); EA=x_E2:E$EL.w.p$SP
  KM=$ROOT/$NSK/models_${MODEL}_$TAG.json
  python3 - $HOST/$NSK/models_${MODEL}_$TAG.json $R <<'PYEOF' || fail "models config $TAG"
import json, sys
out, run = sys.argv[1], sys.argv[2]
cfg = {"note": "queue K section 5: Llama-3.1-8B with the keep-term E of run dir %s (code copy code_keepfix_20261123); everything else as configs/newsets/frozen_models_20261119.json" % run,
       "models": {"llama3.1-8b": {"code": "code_keepfix_20261123", "python": "python3", "gpus": 1, "source_run": "results_qa_das_v1u/llama3.1-8b/full",
                                  "run_dir": run, "e_on": "E2", "c_run_dir": "results_e44_factorial_das/llama3.1-8b", "c_e_on": "E1", "el": 16, "cl": 19}}}
open(out + ".new", "w").write(json.dumps(cfg, indent=1) + "\n")
import os
os.replace(out + ".new", out)
PYEOF
  XARMS="$ES $EA"; for s in $S $SP; do for c in 0.25 0.5 1 2; do XARMS="$XARMS x_E2:E$EL.w.p$s+C$CL.w.m$c"; done; done
  TX="x:E$EL.w.p$S x:E$EL.w.p$SP"
  COMMON=(--source-run $SRC --run-dir $ROOT/$R --e-on E2 --c-run-dir $ROOT/$CRUN --c-e-on E1
          --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --prompts-extra $NEWP)
  G1=$ROOT/$R/eval_extra/gen; GM=$ROOT/$MAINR/eval_extra/gen
  # ---- TruthfulQA tune
  CK=$ROOT/$NSK/n1s_check_${MODEL}_$TAG.json
  [ -s $HOST/${CK#$ROOT/} ] || "${DQ[@]}" $PYM -I $NC gpu --models $KM --model $MODEL --el $EL --cl $CL --s $S \
      --base-file $GM/../base_x_E2_tune.json --out $CK >> $log 2>&1 || fail "$TAG TruthfulQA checks"
  passed $CK || fail "$TAG TruthfulQA check report says not passed"
  "${DT[@]}" $PYM $WR prompt selfcheck "${COMMON[@]}" --split tune --arms $ES+C$CL.w.m0.25 >> $log 2>&1 || fail "$TAG selfcheck"
  "${DT[@]}" $PYM $WR prompt generate "${COMMON[@]}" --split tune --arms $XARMS $TX >> $log 2>&1 || fail "$TAG TruthfulQA generation"
  "${DC[@]}" $NC cross $CK $G1/$(tok $ES)-at-prompt_tune.json $G1/$(tok $ES+C$CL.w.m0.25)-at-prompt_tune.json >> $log 2>&1 || fail "$TAG TruthfulQA check across runs"
  for c in 0.25 0.5 1 2; do
    for s in $S $SP; do
      "${DC[@]}" $NC parts $G1/$(tok x_E2:E$EL.w.p$s+C$CL.w.m$c)-at-prompt_tune.json $GM/$(tok $ES+C$CL.w.m$c)-at-prompt_tune.json 1 1 C >> $log 2>&1 || fail "$TAG C part $s $c"
    done
  done
  "${DC[@]}" $KF/keepfix_table.py check-tq --run $ROOT/$R --main $ROOT/$MAINR --arms $XARMS $TX >> $log 2>&1 || fail "$TAG TruthfulQA write records / P"
  GEN1=""; for a in $XARMS; do GEN1="$GEN1 $(tok $a)-at-prompt_tune.json"; done
  GENX=""; for a in $TX; do GENX="$GENX $(tok $a)-at-prompt_tune.json"; done
  for pair in "kf_tune:$GEN1" "kfx_tune:$GENX"; do
    NAME=${pair%%:*}; GEN=${pair#*:}; W=$R/eval_extra/judge_tune_$NAME
    if [ ! -f $HOST/$W/judge2/cells.jsonl ]; then
      "${DT[@]}" $PYM $WR prompt export "${COMMON[@]}" --split tune --gen $GEN --name $NAME >> $log 2>&1 || fail "$TAG export $NAME"
      mkdir -p $HOST/$W/judge2/cache && cp $HOST/$R/eval_extra/judge_tune/$NAME.json $HOST/$W/judge_input.json || fail "$TAG judge input $NAME"
      "${DT[@]}" python3 scripts/review_checks/judge_two_axis_run.py cells $ROOT/$W --generation judge_input.json >> $log 2>&1 || fail "$TAG cells $NAME"
    fi
  done
  say "lambda $LAM: TruthfulQA tune generated, checked, exported"
  # ---- PopQA / TriviaQA tune (alias match, no judge)
  FC=$NS/gen_prompt/keepfix_${TAG}_chk/popqa/$MODEL/tune
  gf popqa prompt $KM --out $FC --arms $ES --first-batches 2 || fail "$TAG fact check generation"
  for task in popqa triviaqa; do
    gf $task prompt $KM --out $NS/gen_prompt/keepfix_$TAG/$task/$MODEL/tune --arms $ES $EA || fail "$TAG fact E_k $task"
  done
  "${DC[@]}" $NH compare $(first $FC $(tok $ES)) $NS/gen_prompt/keepfix_$TAG/popqa/$MODEL/tune/$(tok $ES).json >> $log 2>&1 || fail "$TAG fact twice"
  "${DC[@]}" $KF/keepfix_table.py check-fact --run $ROOT/$R --gen-name keepfix_$TAG --arms $ES $EA >> $log 2>&1 || fail "$TAG fact write records"
  say "lambda $LAM: fact sets generated and checked"
  # ---- (QA)^2 570 (judged: the 100 pilot questions)
  QO=$NS/gen_prompt/keepfix_$TAG/qaqafull/$MODEL/test; QC=$NS/gen_prompt/keepfix_${TAG}_chk/qaqafull/$MODEL/test
  QW=$NS/judge/qaqafull/$MODEL/keepfix_${TAG}_pilot100
  gq $KM --out $QC --arms $ES $ES+C$CL.w.m0.25 --first-batches 2 || fail "$TAG qaqa check generation"
  gq $KM --out $QO --arms $XARMS || fail "$TAG qaqa generation"
  for a in $ES $ES+C$CL.w.m0.25; do "${DC[@]}" $NH compare $(first $QC $(tok $a)) $QO/$(tok $a).json >> $log 2>&1 || fail "$TAG qaqa twice $a"; done
  for c in 0.25 0.5 1 2; do
    for s in $S $SP; do
      "${DC[@]}" $NC parts $QO/$(tok x_E2:E$EL.w.p$s+C$CL.w.m$c).json $QM/$(tok $ES+C$CL.w.m$c).json 1 1 C >> $log 2>&1 || fail "$TAG qaqa C part $s $c"
    done
  done
  qexport $QO $QW $XARMS || fail "$TAG qaqa export / cells"
  touch ${MK}_${TAG}_GEN_DONE
  say "lambda $LAM: (QA)^2 generated, checked, exported; judges started"
  ( exec 9>&-; judge_one ${MK}_${TAG}_kf $CODE $ROOT/$R/eval_extra/judge_tune_kf_tune $O_kf &&
    judge_one ${MK}_${TAG}_kfx $CODE $ROOT/$R/eval_extra/judge_tune_kfx_tune $O_kfx &&
    judge_one ${MK}_${TAG}_qaqa $ROOT/Iteration_01/code $QW $O_q ) > /dev/null 2>&1 < /dev/null &
done
flock -u 9; exec 9>&-
say "card released; waiting for the judges"
wait
say "balance after: $(bash $L/balance.sh)"
ok=1; [ -e ${MK}_main_qaqah_DONE ] || ok=0
for lt in $USABLE; do TAG=${lt#*:}; for j in kf kfx qaqa; do [ -e ${MK}_${TAG}_${j}_DONE ] || ok=0; done; done
if [ $ok = 1 ]; then touch ${MK}_DONE; say "all judged"; else fail "a judge did not finish"; fi
