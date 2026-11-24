#!/bin/bash
# Iteration_01 (Claude Opus 5.5, executor H, 2026-11-24 08 h; plan discussion/work_queue/2026-11-24_queue_m.md section 4,
# second stage of the per-layer C row): for one model, every layer gets on the tune split (192 questions) the two C arms
# at 0.25 (x.w.p0.25 toward caution, x.w.m0.25 toward answering) and the two random-direction arms (x.rand.p0.25,
# x.rand.m0.25). What is missing is decided from the files on the server at every start (resumable):
#   * a layer without dirs/C_L<l>.pt is trained first. The command is the first step of STAGE=run of the unchanged runner
#     code/scripts/review_checks/newmodel_e44.sh (line 210): e44_factorial_das.py train with the runner's COMMON flags
#     (--source-run, batch 8, 48 new tokens, target cap 24), --e-on, --run-dir, --family C, --layers, --val-n 16, in the
#     same container, environment, code copy and interpreter (model config "python") as the runner; for gemma3-27b with
#     the one difference of launchers/newmodel_e44_expseg.sh (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True).
#     The runner's other steps of STAGE=run (the 11 first-round arms, the reference readout) are not run (plan 4.2).
#   * a missing arm file gen/C_L<l>_<arm>_tune.json is generated through the unchanged runner, STAGE=extra (as crow_gen.sh);
#     existing answer files are never redone (the engine skips a verified file anyway).
# Order: the first layer to train alone, then its acceptance (train log: identity_sha256, protocol, e_on, frozen_source
# equal to those of a stage-one layer of the same model, training_contract equal except its "layer"; direction file
# present), its four arms, 192 rows each, crow_sign.py on its first 16 questions; any difference stops the model. Then
# the other layers in one training call, the same acceptance on each, then the arms (one runner call per set of missing
# arms). Models without layers to train: the first layer's arms alone first, then the rest.
# CROW2_SMOKE=1: stop after the first layer's checks (no DONE marker).
# Cards 1 or 2 only, through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock, retried every minute) held for the whole
# run; ours plus others at most 66560 MiB on each card used (checked after the lock, as crow_gen.sh; a run that trains
# asks for the training memory NEEDT where it is known). qwen3.8-27b-dm takes cards 1,2 and both locks.
# Markers Iteration_01/logs/newsets_20261119/CROW2_<model>_GEN_DONE / _GEN_FAILED; log crow2_gen_<model>.log there;
# status lines "crow2 gen <model>: ..." appended to crow_all.log.
# usage (host): crow2_gen.sh KEY        KEY = g12 | g27 | q27 | llama | q9   (card: CROW2_CARD=1|2, default 1; q27: 1,2)
#
# crow2_gen_from_1124.sh (Claude Opus 5.5, executor gui, 2026-11-24 21:4x; plan 2026-11-24_queue_m.md, progress entry
# 21:35 point 1): a copy of crow2_gen.sh (1de5e9f59fbb5e5c) with these additions; with none of them set it behaves
# exactly as crow2_gen.sh (same lists, commands, markers, log):
#   CROW2_MINL=<n>   no layer below n is trained (gemma3-27b: one card cannot hold the training of layers 0-18). Arms
#                    are generated on layers >= n and on lower layers that already have their direction file (for
#                    gemma3-27b at 20: layer 19, trained on 11-16). The end check covers the same layers. Markers
#                    CROW2_<model>_FROM<n>_GEN_DONE / _FROM<n>_GEN_FAILED, log crow2_gen_from<n>_<model>.log; the
#                    complete run's markers (CROW2_<model>_GEN_*) are neither read nor touched.
#   CROW2_STOPAT="YYYY-MM-DD HH:MM" (Beijing): after the first layer, the other layers are trained one call per layer
#                    and their arms generated one call per layer (same command, fewer layers per call); before each
#                    layer the clock is read, and from that time on no new layer is started: marker
#                    CROW2_<model>[_FROM<n>]_TIME_STOP, exit 0, card released. Rerun to resume.
#   CROW2_LISTONLY=1 print the layer lists and the marker base to stdout and exit (no log, no marker, no lock).
cd /data/sanshizhao/Truth_Decoupling || exit 2
W=/workspace/Truth_Decoupling
R=/data/sanshizhao/Truth_Decoupling
L=Iteration_01/logs/newsets_20261119
KEY=${1:-}
CARD=${CROW2_CARD:-1}
SMOKE=${CROW2_SMOKE:-0}
RUNNER=$R/Iteration_01/code/scripts/review_checks/newmodel_e44.sh
ARMS="x.w.p0.25 x.w.m0.25 x.rand.p0.25 x.rand.m0.25"
G3=$W/Iteration_01/results/ff/gemma3
XENV=()
TRAIN=yes
case $KEY in
  llama) MODEL=llama3.1-8b; NEED=20000; TRAIN=no; CODE=code_e45_20261112
         V=(MODEL=llama3.1-8b RUN_DIR=$W/results_e44_factorial_das/llama3.1-8b SOURCE_DIR=$W/results_qa_das_v1u/llama3.1-8b/full
            CODE_PATH=$W/code_e45_20261112 E_ON=E1 BRANCH=C)
         RUNH=results_e44_factorial_das/llama3.1-8b ;;
  q9)    MODEL=qwen3.5-9b; NEED=22000; TRAIN=no; CODE=code_ff_explore_20261113
         V=(MODEL=qwen3.5-9b CODE_PATH=$W/code_ff_explore_20261113 E_ON=E1 BRANCH=C)
         RUNH=results_newmodels_ff_20261112/e44/qwen3.5-9b ;;
  q27)   MODEL=qwen3.8-27b-dm; NEED=34000; CARD=1,2; CODE=code_ff_explore_20261113b; E_ON=E2; REF=19
         V=(MODEL=qwen3.8-27b-dm CODE_PATH=$W/code_ff_explore_20261113b E_ON=E2 BRANCH=C)
         RUNH=results_newmodels_ff_20261112/e44/qwen3.8-27b-dm; SRCH=results_newmodels_ff_20261112/source/qwen3.8-27b-dm ;;
  g12)   MODEL=gemma3-12b; NEED=30000; NEEDT=42000; CODE=Iteration_01/code; E_ON=E2; REF=15
         V=(MODEL=gemma3-12b CODE_PATH=$W/Iteration_01/code FF_ROOT=$G3/12b E_ON=E2 BRANCH=C)
         RUNH=Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b; SRCH=Iteration_01/results/ff/gemma3/12b/source/gemma3-12b ;;
  g27)   MODEL=gemma3-27b; NEED=62000; NEEDT=66560; CODE=Iteration_01/code; E_ON=E2; REF=19
         RUNNER=$R/Iteration_01/launchers/newmodel_e44_expseg.sh; XENV=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
         V=(MODEL=gemma3-27b CODE_PATH=$W/Iteration_01/code FF_ROOT=$G3/27b E_ON=E2 BRANCH=C)
         RUNH=Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b; SRCH=Iteration_01/results/ff/gemma3/27b/source/gemma3-27b ;;
  *) echo "usage: crow2_gen.sh g12|g27|q27|llama|q9"; exit 2 ;;
esac
case $CARD in 1|2|1,2) ;; *) echo "cards 1 or 2 only"; exit 2 ;; esac
GEN=$RUNH/gen
MINL=${CROW2_MINL:-}
STOPAT=${CROW2_STOPAT:-}
LISTONLY=${CROW2_LISTONLY:-0}
case $MINL in ""|[0-9]|[0-9][0-9]) ;; *) echo "CROW2_MINL must be a layer number"; exit 2 ;; esac
STOPS=""
if [ -n "$STOPAT" ]; then
  STOPS=$(TZ=Asia/Shanghai date -d "$STOPAT" +%s) || { echo "CROW2_STOPAT: cannot read '$STOPAT'"; exit 2; }
fi
if [ -n "$MINL" ]; then
  MK=$L/CROW2_${MODEL}_FROM${MINL}_GEN; log=$L/crow2_gen_from${MINL}_$MODEL.log; TS=$L/CROW2_${MODEL}_FROM${MINL}_TIME_STOP
  SAYM="$MODEL (from L$MINL)"
else
  MK=$L/CROW2_${MODEL}_GEN; log=$L/crow2_gen_$MODEL.log; TS=$L/CROW2_${MODEL}_TIME_STOP; SAYM=$MODEL
fi
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') crow2 gen $SAYM: $*" | tee -a $L/crow_all.log >> $log; }
fail() { say "FAILED: $*"; touch ${MK}_FAILED; exit 1; }
timeup() {  # timeup WHAT: from CROW2_STOPAT on, start nothing new: TIME_STOP marker, exit 0 (the locks go with the process)
  [ -n "$STOPS" ] && [ $(date +%s) -ge $STOPS ] || return 0
  say "time stop ($STOPAT Beijing) reached before $1; nothing new started; marker $TS; cards released"
  touch $TS; exit 0
}
if [ "$LISTONLY" != 1 ]; then
  [ -e ${MK}_DONE ] && { say "already done (${MK}_DONE)"; exit 0; }
  rm -f ${MK}_FAILED $TS
fi
[ -f "$RUNNER" ] || fail "no runner $RUNNER"
[ -f Iteration_01/diag/crow_sign.py ] || fail "no Iteration_01/diag/crow_sign.py"
CFG=$CODE/configs/models/$MODEL.json
[ -f $CFG ] || fail "no model config $CFG"
NL=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["num_layers"]))' $CFG) || fail "num_layers of $CFG"
PYM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("python", "python3"))' $CFG) || fail "python of $CFG"
tok() { echo "${1//./-}"; }
missing_arms() {  # missing_arms LAYER -> the arms whose answer file is absent or empty
  local a out=""
  for a in $ARMS; do [ -s $GEN/C_L${1}_$(tok $a)_tune.json ] || out="$out $a"; done
  echo $out
}
inscope() {  # inscope LAYER: no CROW2_MINL -> every layer; else layers >= MINL and lower layers with a direction file
  [ -z "$MINL" ] || [ $1 -ge $MINL ] || [ -f $RUNH/dirs/C_L$1.pt ]
}
TRAINL=""; GENL=""; OUTL=""
for l in $(seq 0 $((NL - 1))); do
  inscope $l || { OUTL="$OUTL $l"; continue; }
  [ -f $RUNH/dirs/C_L$l.pt ] || TRAINL="$TRAINL $l"
  [ -n "$(missing_arms $l)" ] && GENL="$GENL $l"
done
TRAINL=${TRAINL# }; GENL=${GENL# }; OUTL=${OUTL# }
if [ "$LISTONLY" = 1 ]; then
  echo "model $MODEL; $NL layers; CROW2_MINL '${MINL}'; CROW2_STOPAT '${STOPAT}'; markers ${MK}_DONE / _FAILED; log $log"
  echo "to train [${TRAINL}]"
  echo "arms missing on [${GENL}]"
  for l in $GENL; do echo "  L$l: $(missing_arms $l)"; done
  echo "left out (below CROW2_MINL, no direction) [${OUTL}]"
  exit 0
fi
say "$NL layers; to train [${TRAINL}]; arms missing on [${GENL}]; cards $CARD; runner $(sha256sum $RUNNER | cut -c1-16)$([ "$SMOKE" = 1 ] && echo '; SMOKE')$([ -n "$MINL" ] && echo "; CROW2_MINL $MINL, left out [$OUTL]")$([ -n "$STOPAT" ] && echo "; time stop $STOPAT")"
if [ -n "$TRAINL" ] && [ $TRAIN = no ]; then fail "layers without a C direction [$TRAINL], none expected for $MODEL"; fi
if [ -z "$GENL" ]; then say "nothing missing"; [ "$SMOKE" = 1 ] || touch ${MK}_DONE; exit 0; fi
if [ -n "$TRAINL" ]; then
  [ -f $RUNH/frozen_tune.json ] && [ -f $RUNH/mc/base_x_tune.json ] || fail "no frozen answers / baseline readout in $RUNH"
  [ -f $SRCH/bank.json ] || fail "no source bank $SRCH/bank.json"
  [ -f $RUNH/train_logs/C_L$REF.json ] || fail "no reference train log C_L$REF.json"
  # main session, 2026-11-24 08:4x: training takes more memory than generation (gemma3-12b 41350 MiB on the smoke;
  # gemma3-27b fills a card, 11-16), so the room asked for before a run that trains is NEEDT: gemma3-27b then starts
  # only on a card nobody else uses.
  [ -n "${NEEDT:-}" ] && NEED=$NEEDT
fi

fd=8
for g in ${CARD//,/ }; do
  LOCK=Iteration_01/newsets/selfjudge/.gpu$g.lock
  eval "exec $fd>\"$LOCK\"" || fail "cannot open $LOCK"
  until flock -n $fd; do sleep 60; done
  say "holds the lock of card $g"
  fd=$((fd + 1))
done
for g in ${CARD//,/ }; do
  while true; do
    USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$g '$1 == g {print $2}')
    [[ $USED =~ ^[0-9]+$ ]] || fail "cannot read the memory of GPU $g"
    [ $((USED + NEED)) -le 66560 ] && break
    say "GPU $g holds $USED MiB, $NEED more would exceed 66560; waiting"; sleep 60
  done
  say "GPU $g: $USED MiB in use at the start"
done

train() {  # train "LAYERS": runner newmodel_e44.sh line 210 (STAGE=run, first step), same container command
  echo "=== $(date '+%F %T') [$MODEL C train (crow2) gpu$CARD e_on=$E_ON] train C layers: $1 ===" >> $log
  docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models \
      -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES="$CARD" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$W/$CODE "${XENV[@]}" \
      -e DEEPSEEK_API_KEY_FILE=$W/logs/.deepseek_key -w $W/$CODE truth_dec \
      "$PYM" scripts/e44_factorial_das.py train --source-run $W/$SRCH --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 \
      --e-on $E_ON --run-dir $W/$RUNH --family C --layers $1 --val-n 16 >> $log 2>&1
}
accept() {  # accept LAYER: train log fields against the stage-one reference layer REF; direction file present
  [ -s $RUNH/dirs/C_L$1.pt ] || { echo "no direction file $RUNH/dirs/C_L$1.pt"; return 1; }
  python3 -c '
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); layer = int(sys.argv[3])
ok = True
for k in ("identity_sha256", "protocol", "e_on", "frozen_source"):
    same = a.get(k) == b.get(k)
    ok &= same
    print("  %s: new %s ref %s [%s]" % (k, str(a.get(k))[:16], str(b.get(k))[:16], "same" if same else "DIFFERENT"))
ta = dict(a["training_contract"]); tb = dict(b["training_contract"])
la, lb = ta.pop("layer"), tb.pop("layer")
diff = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
ok &= not diff and la == layer and a.get("layer") == layer and a.get("pilot") is False and a.get("family") == "C"
print("  training_contract: %d keys besides layer, differing %s; layer new %s ref %s; pilot %s" % (len(ta), diff or "none", la, lb, a.get("pilot")))
sys.exit(0 if ok else 1)' $RUNH/train_logs/C_L$1.json $RUNH/train_logs/C_L$REF.json $1
}
gen() {  # gen "LAYERS" "ARMS": the unchanged runner, STAGE=extra
  env "${V[@]}" STAGE=extra GPU=$CARD LAYERS="$1" ARMS="$2" bash $RUNNER >> $log 2>&1
}
rows() {  # rows LAYER: each of the four arm files complete, not a pilot, 192 rows
  local a
  for a in $ARMS; do
    python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
n = len(d["rows"])
print("  %s: completed %s, pilot %s, rows %d" % (sys.argv[1].rsplit("/", 1)[1], d.get("completed"), d.get("pilot"), n))
sys.exit(0 if d.get("completed") and not d.get("pilot") and n == 192 else 1)' $GEN/C_L${1}_$(tok $a)_tune.json || return 1
  done
}

FIRST=${GENL%% *}
[ -n "$TRAINL" ] && FIRST=${TRAINL%% *}
timeup "layer $FIRST"
if [ -n "$TRAINL" ]; then
  say "train first layer $FIRST"
  t0=$(date +%s)
  train "$FIRST" || fail "training of layer $FIRST (see $log)"
  say "layer $FIRST trained in $(( $(date +%s) - t0 )) s"
  accept $FIRST >> $log 2>&1; rc=$?
  tail -n 6 $log | sed "s/^/  /" >> $L/crow_all.log
  [ $rc = 0 ] || fail "acceptance of the train log of layer $FIRST"
fi
MA=$(missing_arms $FIRST)
if [ -n "$MA" ]; then
  say "arms of layer $FIRST [$MA]"
  gen "$FIRST" "$MA" || fail "generation of layer $FIRST (see $log)"
fi
rows $FIRST >> $log 2>&1 || fail "arm files of layer $FIRST (see $log)"
python3 -B Iteration_01/diag/crow_sign.py $GEN $FIRST 16 >> $log 2>&1
rc=$?
tail -n 2 $log | sed "s/^/  /" >> $L/crow_all.log
[ $rc = 0 ] || fail "sign check of layer $FIRST (rc $rc)"
if [ "$SMOKE" = 1 ]; then say "smoke: layer $FIRST passed; stopping here (no DONE marker); cards released"; exit 0; fi

REST=""
for l in $TRAINL; do [ $l = $FIRST ] || REST="$REST $l"; done
REST=${REST# }
if [ -n "$REST" ] && [ -n "$STOPAT" ]; then  # CROW2_STOPAT: one training call per layer, the clock read before each
  say "train other layers [$REST], one call per layer (time stop $STOPAT)"
  for l in $REST; do
    timeup "training layer $l"
    t0=$(date +%s)
    train "$l" || fail "training of layer $l (see $log)"
    say "layer $l trained in $(( $(date +%s) - t0 )) s"
    echo "acceptance L$l" >> $log
    accept $l >> $log 2>&1 || fail "acceptance of the train log of layer $l (see $log)"
  done
  say "acceptance passed on [$REST]"
elif [ -n "$REST" ]; then
  say "train other layers [$REST]"
  t0=$(date +%s)
  train "$REST" || fail "training of layers $REST (see $log)"
  say "$(echo $REST | wc -w) layers trained in $(( $(date +%s) - t0 )) s"
  for l in $REST; do
    echo "acceptance L$l" >> $log
    accept $l >> $log 2>&1 || fail "acceptance of the train log of layer $l (see $log)"
  done
  say "acceptance passed on [$REST]"
fi
declare -A BY
for l in $GENL; do
  [ $l = $FIRST ] && continue
  MA=$(missing_arms $l)
  [ -n "$MA" ] || continue
  if [ -n "$STOPAT" ]; then  # CROW2_STOPAT: one generation call per layer, the clock read before each
    timeup "the arms of layer $l"
    say "arms [$MA] on layer $l"
    gen "$l" "$MA" || fail "generation of arms [$MA] on layer $l (see $log)"
  else
    BY["$MA"]="${BY["$MA"]:+${BY["$MA"]} }$l"
  fi
done
for MA in "${!BY[@]}"; do
  say "arms [$MA] on layers [${BY["$MA"]}]"
  gen "${BY["$MA"]}" "$MA" || fail "generation of arms [$MA] on layers ${BY["$MA"]} (see $log)"
done
NIN=0
for l in $(seq 0 $((NL - 1))); do
  inscope $l || continue
  NIN=$((NIN + 1))
  [ -s $RUNH/dirs/C_L$l.pt ] || fail "missing $RUNH/dirs/C_L$l.pt"
  [ -z "$(missing_arms $l)" ] || fail "missing arms [$(missing_arms $l)] of layer $l"
done
for l in $GENL; do rows $l >> $log 2>&1 || fail "arm files of layer $l (see $log)"; done
if [ -n "$MINL" ]; then
  say "done: the $NIN layers in scope have a direction and the four arms; left out (not trained, below L$MINL) [$OUTL]; cards released"
else
  say "done: all $NL layers have a direction and the four arms; cards released"
fi
touch ${MK}_DONE
