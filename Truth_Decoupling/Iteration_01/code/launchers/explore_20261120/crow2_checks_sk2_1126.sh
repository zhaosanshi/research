#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor yin, 2026-11-25 22 h; plan discussion/work_queue/2026-11-24_queue_m.md, progress
# entry 2026-11-25 22:11 part one, rules v4): gemma3-27b layers 0-17 with the cause rows and the isolate rows of a question
# back-propagated one after the other (patch split_kinds_1125.patch, patched code copy Iteration_01/code_splitkinds_20261125,
# switch --split-kinds of e44_factorial_das.py train). This script sets up the separate results root and runs the FOUR
# pre-run checks in the order 1, 2, 3, 4 on card 0. Nothing in the main root is written.
#   setup  results root Iteration_01/results/ff/gemma3/27b_splitkinds/: source/gemma3-27b/{bank,questions}.json copied
#          byte for byte from the main root (cp -p, cmp); run dir e44/gemma3-27b.
#   1.     freeze train + tune (the runner's step0 freeze command, newmodel_e44.sh line 201, with the patched copy) and
#          the no-write readout of prompt x on tune (runner line 204 with --split tune --bases x); compare_sk_1125.py
#          frozen / mc against the main root: every (question, prompt) record and every baseline row identical, else stop.
#   2.     layer 0 trained WITH --split-kinds into e44/gemma3-27b (the generator's training command + the switch); out of
#          memory or any failure: stop. The train log's training_contract against the main root's layer 19 is recorded;
#          keys other than layer, backward, frozen_record_sha256, frozen_file_sha256 differing: stop.
#   3.     layer 31 trained WITHOUT the switch into its own run dir e44/gemma3-27b_noswitch (frozen answers and readout
#          copied there byte for byte from e44/gemma3-27b); cosine with the main root's layer-31 direction and the final
#          losses of the two train logs: recorded only.
#   4.     layers 18, 20, 31 trained WITH the switch into e44/gemma3-27b; cosine with the main root's directions, each >= 0.99;
#          arms x.w.p0.25 and x.w.m0.25 on those layers (unchanged expseg runner, STAGE=extra, batch 8); the "all 192
#          questions" answered count that crow_sign.py prints, new vs main root, within 3 for all six; identical-answer
#          counts recorded only. Anything outside: stop.
# Results (plain text, numbers only): Iteration_01/diag/out_split_kinds_checks_gemma3-27b_<yyyymmdd of the start>.txt.
# Markers Iteration_01/logs/newsets_20261119/CROW2_gemma3-27b_SK_CHECK_DONE / _SK_CHECK_FAILED; log crow2_checks_sk_gemma3-27b.log;
# status lines "crow2 sk checks gemma3-27b: ..." in crow_all.log.
# Card 0 only, through Iteration_01/newsets/selfjudge/.gpu0.lock (flock, retried every minute) held for the whole run;
# starts when ours plus others stay within 66560 MiB (an empty card). Resumable: finished steps are found on disk
# (the engine skips verified files); rerun after a failure only once the cause is known.
# SK_DRY=1: no lock, no marker, CUDA_VISIBLE_DEVICES empty, every card command run once with timeout 90 into
#   crow2_checks_sk_dry.log (each must end in "No CUDA GPUs are available"); setup is done (copies only).
# usage (host): ( nohup setsid bash Iteration_01/logs/newsets_20261119/crow2_checks_sk_1125.sh \
#                 > Iteration_01/logs/newsets_20261119/crow2_checks_sk_1125.out 2>&1 < /dev/null & )
#
# crow2_checks_sk2_1126.sh (Claude Opus 5.5, executor chen, 2026-11-26 01 h; plan discussion/work_queue/2026-11-24_queue_m.md, entries 2026-11-25 22:11 part one and 2026-11-26 00:48 (second
# version of the patch, split_kinds2_1126.patch: both halves left-padded to the width of the whole 8-row batch;
# training contract "backward": "by_kind_full_width")):
# a copy of crow2_checks_sk_1125.sh (4f00d7a8c02143ec) made by program for the second version. The text above describes the
# source; the differences, all of them:
#   (1) names: code copy Iteration_01/code_splitkinds2_20261126, results root Iteration_01/results/ff/gemma3/27b_splitkinds2,
#       markers CROW2_gemma3-27b_SK2_CHECK_DONE / _FAILED, results file Iteration_01/diag/out_split_kinds2_checks_gemma3-27b_<date>.txt,
#       logs crow2_checks_sk2_gemma3-27b.log / crow2_checks_sk2_dry.log / crow2_checks_sk2_dry_out.txt, scratch files
#       .sk2_cpu.txt / .sk2_sign_new.txt / .sk2_sign_old.txt, status lines "crow2 sk2 checks", log headers "[sk2 checks".
#   (2) card 3 (lock Iteration_01/newsets/selfjudge/.gpu3.lock) instead of card 0.
#   (3) start condition: before anything else is run (after the DONE / FAILED marker test) it waits, polling every minute
#       and holding no lock, for CROW2_gemma3-27b_SK2_GRADGATE_DONE (written by the main session after gate 1);
#       CROW2_gemma3-27b_SK2_GRADGATE_FAILED -> CROW2_gemma3-27b_SK2_CHECK_FAILED, exit 1, no card taken. In SK_DRY=1
#       the same wait applies (no marker is written). Test overrides: SK2_GATE=<marker base> and SK2_MK=<marker base>
#       (defaults the real names above); used only for the dry and gate tests with TEST_chen names.
#   (4) the four patched files of the code copy must have the sha256 of the second version (5f0cd7b55befa139,
#       295b3a2bc69d71ac, 366c6da05a46df18, 6bbd9ec52d668102), else FAILED; the results file records all four.
#   (5) check 2 requires "backward": "by_kind_full_width" (source: "by_kind").
#   (6) the first line of the results file names the 00:48 entry too.
# Thresholds unchanged (cosine >= 0.99 on 18, 20, 31; answered counts within 3). compare_sk_1125.py is used unchanged.
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
W=/workspace/Truth_Decoupling
R=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
MODEL=gemma3-27b
CARD=3
DRY=${SK_DRY:-0}
CODE=Iteration_01/code_splitkinds2_20261126
MAINR=Iteration_01/results/ff/gemma3/27b
NEWR=Iteration_01/results/ff/gemma3/27b_splitkinds2
SRC0=$MAINR/source/$MODEL; SRC=$NEWR/source/$MODEL
RUN0=$MAINR/e44/$MODEL; RUN=$NEWR/e44/$MODEL; RUNNS=$NEWR/e44/${MODEL}_noswitch
RUNNER=$R/Iteration_01/launchers/newmodel_e44_expseg.sh
CMP=Iteration_01/diag/compare_sk_1125.py
MK=${SK2_MK:-$L/CROW2_${MODEL}_SK2_CHECK}
if [ "$DRY" = 1 ]; then CVD=""; GPUV=99; log=$L/crow2_checks_sk2_dry.log; else CVD=$CARD; GPUV=$CARD; log=$L/crow2_checks_sk2_$MODEL.log; fi
OUT=Iteration_01/diag/out_split_kinds2_checks_${MODEL}_$(TZ=Asia/Shanghai date +%Y%m%d).txt
[ "$DRY" = 1 ] && OUT=$L/crow2_checks_sk2_dry_out.txt
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') crow2 sk2 checks $MODEL: $*" | tee -a $L/crow_all.log >> $log; }
rec() { echo "$*" >> $OUT; say "$*"; }
fail() { say "FAILED: $*"; echo "FAILED: $*" >> $OUT; [ "$DRY" = 1 ] || touch ${MK}_FAILED; exit 1; }
if [ "$DRY" != 1 ]; then
  [ -e ${MK}_DONE ] && { say "already done (${MK}_DONE)"; exit 0; }
  [ -e ${MK}_FAILED ] && { echo "${MK}_FAILED exists: remove it by hand once the cause is known"; exit 2; }
fi
# chen 2026-11-26: start condition of the second version (plan 2026-11-26 00:48, gate 1 belongs to the main session).
# Polled every minute; no lock is held and nothing is written to the results file while waiting.
GATE=${SK2_GATE:-$L/CROW2_${MODEL}_SK2_GRADGATE}
say "waiting for ${GATE}_DONE (polled every minute, no lock held)"
while true; do
  if [ -e ${GATE}_FAILED ]; then
    say "FAILED: gate 1 failed (${GATE}_FAILED): nothing run, no card taken"; [ "$DRY" = 1 ] || touch ${MK}_FAILED; exit 1
  fi
  [ -e ${GATE}_DONE ] && break
  sleep 60
done
say "seen ${GATE}_DONE"
for p in core/qa_context.py:5f0cd7b55befa139 core/qa_prompt_das.py:295b3a2bc69d71ac core/qa_factorial_das.py:366c6da05a46df18 \
         scripts/e44_factorial_das.py:6bbd9ec52d668102; do
  [ "$(sha256sum $CODE/${p%%:*} | cut -c1-16)" = "${p##*:}" ] || fail "$CODE/${p%%:*} is not the second version (sha256 ${p##*:} expected)"
done
for f in $RUNNER $CMP Iteration_01/diag/crow_sign.py $CODE/scripts/e44_factorial_das.py $CODE/core/qa_factorial_das.py; do
  [ -s $f ] || fail "missing $f"
done
grep -q -- '--split-kinds' $CODE/scripts/e44_factorial_das.py || fail "$CODE has no --split-kinds switch"
for l in 18 19 20 31; do [ -s $RUN0/dirs/C_L$l.pt ] && [ -s $RUN0/train_logs/C_L$l.json ] || fail "main root lacks layer $l"; done
for l in 18 20 31; do for a in p0-25 m0-25; do [ -s $RUN0/gen/C_L${l}_x-w-${a}_tune.json ] || fail "main root lacks arm $l $a"; done; done

DX=(docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models
    -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES="$CVD" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$W/$CODE -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e DEEPSEEK_API_KEY_FILE=$W/logs/.deepseek_key -w $W/$CODE truth_dec)
COMMON=(--source-run $W/$SRC --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
TO=(); [ "$DRY" = 1 ] && TO=(timeout 90)
model() {  # model ARGS...: the model interpreter of the runner (gemma3-27b's config names no "python": python3)
  echo "=== $(TZ=Asia/Shanghai date '+%F %T') [sk2 checks gpu${CVD:-none}] $* ===" >> $log
  "${TO[@]}" "${DX[@]}" python3 "$@" >> $log 2>&1
}
cpu() {  # cpu ARGS...: compare_sk_1125.py in the scoring container without a card; stdout to OUT and the log
  docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -w $W truth_dec python3 -B $CMP "$@" > $L/.sk2_cpu.txt 2>&1
  local rc=$?
  cat $L/.sk2_cpu.txt >> $OUT; cat $L/.sk2_cpu.txt >> $log; sed "s/^/  /" $L/.sk2_cpu.txt >> $L/crow_all.log
  return $rc
}
train() {  # train RUNDIR "LAYERS" [--split-kinds]: the training command of crow2_gen.sh / runner line 226
  model scripts/e44_factorial_das.py train "${COMMON[@]}" --e-on E2 --run-dir $W/$1 --family C --layers $2 --val-n 16 "${@:3}"
}
oom() { grep -m 1 -o 'CUDA out of memory.*' $log | cut -c1-400; }
copy_same() {  # copy_same FROM TO: byte copy once, then cmp
  mkdir -p "$(dirname $2)"
  [ -e $2 ] || cp -p $1 $2 || return 1
  cmp -s $1 $2
}

# ---------------------------------------------------------------- setup (copies only)
for f in bank.json questions.json; do copy_same $SRC0/$f $SRC/$f || fail "source copy $f differs"; done
mkdir -p $RUN $RUNNS
if [ "$DRY" != 1 ]; then
  : > $OUT
  rec "# four pre-run checks, gemma3-27b, split kinds (plan queue_m 2026-11-25 22:11, rules v4; second version, entry 2026-11-26 00:48); start $(TZ=Asia/Shanghai date '+%F %T') Beijing"
  rec "code copy $CODE: $(for f in core/qa_context.py core/qa_prompt_das.py core/qa_factorial_das.py scripts/e44_factorial_das.py; do echo -n "$f $(sha256sum $CODE/$f | cut -c1-16) "; done)"
  rec "source copied from $SRC0: bank.json $(sha256sum $SRC/bank.json | cut -c1-16), questions.json $(sha256sum $SRC/questions.json | cut -c1-16)"
  exec 8>Iteration_01/newsets/selfjudge/.gpu$CARD.lock || fail "cannot open the lock of card $CARD"
  until flock -n 8; do sleep 60; done
  say "holds the lock of card $CARD"
  while true; do
    USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$CARD '$1 == g {print $2}')
    [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $CARD"
    [ $((USED + 66560)) -le 66560 ] && break
    say "GPU $CARD holds $USED MiB, 66560 more would exceed 66560; waiting"; sleep 60
  done
  rec "card $CARD obtained $(TZ=Asia/Shanghai date '+%F %T') Beijing, $USED MiB in use"
else
  : > $OUT; : > $log
fi

# ---------------------------------------------------------------- check 1
say "check 1: freeze train + tune, no-write readout x tune"
t0=$(date +%s)
model scripts/e44_factorial_das.py freeze "${COMMON[@]}" --run-dir $W/$RUN --splits train tune
rc=$?
if [ "$DRY" = 1 ]; then say "dry freeze rc $rc: $(tail -n 1 $log)"; else
  [ $rc = 0 ] || fail "check 1: freeze (rc $rc, see $log)"; fi
model scripts/e44_factorial_das.py mc "${COMMON[@]}" --run-dir $W/$RUN --split tune --bases x
rc=$?
if [ "$DRY" = 1 ]; then say "dry mc rc $rc: $(tail -n 1 $log)"; else
  [ $rc = 0 ] || fail "check 1: no-write readout (rc $rc, see $log)"
  rec "check 1: freeze + readout $(( $(date +%s) - t0 )) s"
  ok=0
  cpu frozen $RUN/frozen_train.json $RUN0/frozen_train.json || ok=1
  cpu frozen $RUN/frozen_tune.json $RUN0/frozen_tune.json || ok=1
  cpu mc $RUN/mc/base_x_tune.json $RUN0/mc/base_x_tune.json || ok=1
  [ $ok = 0 ] || fail "check 1: the regenerated frozen answers or baseline differ from the main root"
  rec "check 1: passed (frozen train, frozen tune, baseline x tune identical)"
fi

# ---------------------------------------------------------------- check 2
say "check 2: layer 0 with --split-kinds"
t0=$(date +%s)
train $RUN 0 --split-kinds
rc=$?
secs=$(( $(date +%s) - t0 ))
if [ "$DRY" = 1 ]; then say "dry train L0 rc $rc: $(tail -n 1 $log)"; else
  if [ $rc != 0 ]; then
    rec "check 2: layer 0 failed after $secs s (rc $rc); out-of-memory line: $(oom)"
    fail "check 2: layer 0 with the switch did not train (see $log)"
  fi
  rec "check 2: layer 0 trained with the switch in $secs s (call incl. model load); train log seconds $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("%.0f" % d["seconds"])' $RUN/train_logs/C_L0.json)"
  cpu contract $RUN/train_logs/C_L0.json $RUN0/train_logs/C_L19.json
  python3 - $RUN/train_logs/C_L0.json $RUN0/train_logs/C_L19.json $RUN/frozen_train.json <<'EOF' >> $OUT 2>&1 || fail "check 2: the training contract of layer 0 differs beyond layer / backward / frozen hashes"
import json, sys
a, b, f = (json.load(open(p)) for p in sys.argv[1:4])
ta, tb = a["training_contract"], b["training_contract"]
diff = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
allowed = {"layer", "backward", "frozen_record_sha256", "frozen_file_sha256"}
ok = (set(diff) <= allowed and ta.get("backward") == "by_kind_full_width" and ta.get("layer") == 0 and a.get("layer") == 0
      and a.get("family") == "C" and a.get("pilot") is False and a.get("identity_sha256") == f.get("identity_sha256")
      and a.get("e_on") == b.get("e_on") and a.get("protocol") == b.get("protocol"))
print("check 2 contract: differing keys vs main root layer 19 %s (allowed %s); backward %s; identity = new root frozen: %s" % (
    diff, sorted(allowed), ta.get("backward"), a.get("identity_sha256") == f.get("identity_sha256")))
sys.exit(0 if ok else 1)
EOF
  rec "check 2: passed"
fi

# ---------------------------------------------------------------- check 3
say "check 3: layer 31 without the switch (own run dir)"
for f in frozen_train.json frozen_tune.json mc/base_x_tune.json; do
  [ "$DRY" = 1 ] && [ ! -e $RUN/$f ] && continue
  copy_same $RUN/$f $RUNNS/$f || fail "check 3: copy of $f into $RUNNS differs"
done
t0=$(date +%s)
train $RUNNS 31
rc=$?
secs=$(( $(date +%s) - t0 ))
if [ "$DRY" = 1 ]; then say "dry train L31 no switch rc $rc: $(tail -n 1 $log)"; else
  [ $rc = 0 ] || { rec "check 3: failed (rc $rc); out-of-memory line: $(oom)"; fail "check 3: layer 31 without the switch did not train (see $log)"; }
  rec "check 3: layer 31 without the switch trained in $secs s (call incl. model load)"
  cpu cos $RUNNS/dirs/C_L31.pt $RUN0/dirs/C_L31.pt
  cpu losses $RUNNS/train_logs/C_L31.json $RUN0/train_logs/C_L31.json
  cpu contract $RUNNS/train_logs/C_L31.json $RUN0/train_logs/C_L31.json
  rec "check 3: recorded (no threshold)"
fi

# ---------------------------------------------------------------- check 4
say "check 4: layers 18 20 31 with the switch, two C arms each"
t0=$(date +%s)
train $RUN "18 20 31" --split-kinds
rc=$?
secs=$(( $(date +%s) - t0 ))
if [ "$DRY" = 1 ]; then say "dry train 18 20 31 rc $rc: $(tail -n 1 $log)"; else
  [ $rc = 0 ] || { rec "check 4: training failed (rc $rc); out-of-memory line: $(oom)"; fail "check 4: training of 18 20 31 (see $log)"; }
  rec "check 4: layers 18 20 31 trained with the switch in $secs s (one call incl. model load)"
fi
echo "=== $(TZ=Asia/Shanghai date '+%F %T') [sk2 checks] runner extra 18 20 31 x.w.p0.25 x.w.m0.25 ===" >> $log
if [ "$DRY" = 1 ]; then
  env MODEL=$MODEL CODE_PATH=$W/$CODE FF_ROOT=$W/$NEWR E_ON=E2 BRANCH=C STAGE=extra GPU=$GPUV LAYERS="18 20 31" \
      ARMS="x.w.p0.25 x.w.m0.25" timeout 90 bash $RUNNER >> $log 2>&1
  say "dry runner extra rc $?: $(tail -n 1 $log)"
  say "dry run done; nothing trained or generated"
  exit 0
fi
env MODEL=$MODEL CODE_PATH=$W/$CODE FF_ROOT=$W/$NEWR E_ON=E2 BRANCH=C STAGE=extra GPU=$GPUV LAYERS="18 20 31" \
    ARMS="x.w.p0.25 x.w.m0.25" bash $RUNNER >> $log 2>&1 || fail "check 4: generation of the arms (see $log)"
ok=0
for l in 18 20 31; do
  cpu contract $RUN/train_logs/C_L$l.json $RUN/train_logs/C_L0.json
  cpu cos $RUN/dirs/C_L$l.pt $RUN0/dirs/C_L$l.pt
  c=$(docker exec -u 1000:1001 -e CUDA_VISIBLE_DEVICES= -w $W truth_dec python3 -B -c '
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["w"].double()
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["w"].double()
print("%.6f" % float(torch.dot(a, b) / (a.norm() * b.norm())))' $RUN/dirs/C_L$l.pt $RUN0/dirs/C_L$l.pt)
  if awk -v c="$c" 'BEGIN {exit !(c >= 0.99)}'; then rec "check 4: L$l cosine $c >= 0.99"; else rec "check 4: L$l cosine $c BELOW 0.99"; ok=1; fi
  python3 -B Iteration_01/diag/crow_sign.py $RUN/gen $l 16 > $L/.sk2_sign_new.txt 2>&1
  python3 -B Iteration_01/diag/crow_sign.py $RUN0/gen $l 16 > $L/.sk2_sign_old.txt 2>&1
  sed 's/^/new:    /' $L/.sk2_sign_new.txt >> $OUT; sed 's/^/stored: /' $L/.sk2_sign_old.txt >> $OUT
  for a in p0.25 m0.25; do
    n=$(sed -n "s/^L$l x\.w\.$a: .*; all [0-9]* questions [0-9]* -> \([0-9]*\) .*/\1/p" $L/.sk2_sign_new.txt)
    o=$(sed -n "s/^L$l x\.w\.$a: .*; all [0-9]* questions [0-9]* -> \([0-9]*\) .*/\1/p" $L/.sk2_sign_old.txt)
    [[ $n =~ ^[0-9]+$ ]] && [[ $o =~ ^[0-9]+$ ]] || { rec "check 4: L$l x.w.$a: cannot read the answered counts"; ok=1; continue; }
    d=$((n - o)); ad=${d#-}
    if [ $ad -le 3 ]; then rec "check 4: L$l x.w.$a answered stored $o new $n (diff $d, within 3)"
    else rec "check 4: L$l x.w.$a answered stored $o new $n (diff $d, OUTSIDE 3)"; ok=1; fi
    cpu answers $RUN/gen/C_L${l}_x-w-${a/./-}_tune.json $RUN0/gen/C_L${l}_x-w-${a/./-}_tune.json || ok=1
  done
done
[ $ok = 0 ] || fail "check 4: a cosine below 0.99 or an answered count outside 3 (numbers in $OUT)"
rec "check 4: passed"
rec "ALL FOUR CHECKS PASSED $(TZ=Asia/Shanghai date '+%F %T') Beijing; card $CARD released"
touch ${MK}_DONE
