#!/bin/bash
# Iteration_01 exploration (executor R-bing, Claude Opus 5.5, 2026-11-29 Beijing): work queue R item R6
# (discussion/work_queue/2026-11-28_queue_r.md sections 2, 4, 5). Not reviewed by GPT.
# What: new run folders for the E-layer re-selection on TruthfulQA, and the E2 directions of the window layers that the
# main-table E runs lack, trained with the unchanged program and the main runs' settings.
#   setup  (CPU) per model a NEW run folder Iteration_01/results/relayer_1128/<ng>/run_E2 with byte copies (cp -p, then
#          cmp) of the E run's frozen_train.json and frozen_tune.json and of every window layer's dirs/E_L<l>.pt and
#          train_logs/E_L<l>.json; manifest relayer_1128_manifest.json (sha256 of every copy and of its source). An
#          existing target is never overwritten: it must already be identical. The E runs themselves are only read.
#   queue  three lanes, each holding its cards through Iteration_01/newsets/selfjudge/.gpu<N>.lock (flock -n, retried
#          every 60 s; ours plus others at most 66560 MiB on each card at the start), queued BEHIND the FalseQA relearn
#          chain: no lane asks for a card before every relearn job is taken and the last two (T_q9_falseqa_C / _E) hold
#          their cards (their "card" line in relearn_1128_all.log, or their DONE / FAILED / OOM marker), or
#          RELEARN_FULL_DONE exists. That chain is never touched.
#            lane 1 (2 cards): selection smoke q27, then q27 (25 layers)
#            lane 2 (1 card):  g27 (24 layers)
#            lane 3 (1 card):  selection smoke llama, q9, g12, g27; then g12 (18 layers), q9 (3 layers)
#          per model, on the same cards: (1) pilot = the training command with --n-check 3 --val-n 1 on the deepest
#          missing layer (t.* files), accepted against the model's reference layer of the E run (identity,
#          identity_sha256, protocol, e_on, frozen_source; training_contract equal apart from layer / pilot / train_ids /
#          val_ids / val_n / targets_sha256; pilot true; direction file present with the logged sha256); (2) only then
#          the full training of every missing window layer, deepest first, one call; (3) each new layer accepted against
#          the reference layer (as crow2_gen.sh / depth_queue3_1124.sh: identity, identity_sha256, protocol, e_on,
#          frozen_source, training_contract equal apart from layer; family E, pilot false, direction sha256).
#          Training command = newmodel_e44.sh STAGE=run step 1 as copied in crow2_gen.sh / depth_queue3_1124.sh:
#            e44_factorial_das.py train --source-run <source> --batch-size 8 --max-new-tokens 48 --max-target-tokens 24
#              --e-on E2 --run-dir <new run> --family E --layers ... --val-n 16
#          code copy and interpreter of the E run's own training (q9 code_ff_explore_20261113 as depth_queue3_1124.sh;
#          q27 code_ff_explore_20261113b on cards "a,b" (device_map auto, as crow2_gen.sh); g12 / g27 Iteration_01/code;
#          g27 with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True as newmodel_e44_expseg.sh). The program is never
#          split across cards other than the model config's own device_map (q27).
#          A shallow layer that runs out of memory ends that model's training (marker RELAYER_T_<key>_OOM with the
#          layers done); any other failure writes RELAYER_<step>_<key>_FAILED and stops that lane.
# Markers and logs in Iteration_01/logs/newsets_20261119: RELAYER_SETUP_DONE, RELAYER_SMOKE_<key>_DONE / _FAILED
# (written by relayer_1128.sh smoke), RELAYER_P_<key>_DONE / _FAILED, RELAYER_T_<key>_DONE / _FAILED / _OOM,
# RELAYER_LANE<n>_DONE, RELAYER_TRAIN_QUEUE_DONE; status lines relayer_1128_all.log; per model relayer_1128_train_<key>.log.
# usage (host):  bash Iteration_01/logs/newsets_20261119/relayer_1128_train.sh setup
#                setsid -f bash Iteration_01/logs/newsets_20261119/relayer_1128_train.sh queue \
#                  > Iteration_01/logs/newsets_20261119/relayer_1128_train_queue.out 2>&1 < /dev/null
set -uo pipefail
HOST=/data/sanshizhao/Truth_Decoupling; W=/workspace/Truth_Decoupling
cd $HOST || exit 2
export TZ=Asia/Shanghai
L=Iteration_01/logs/newsets_20261119
ALL=$L/relayer_1128_all.log
HNS=$HOST/Iteration_01/newsets
RR=Iteration_01/results/relayer_1128
RLOG=$L/relearn_1128_all.log
MAXMIB=66560
MODE=${1:-}
say() { echo "$(date '+%F %T') relayer train: $*" >> $ALL; }

setm() {  # setm KEY: model settings
  KEY=$1; XENV=()
  case $1 in
    llama) NG=llama3.1-8b; MCFG=llama3.1-8b; ORIG=results_e44_factorial_das/llama3.1-8b_side_E2; NL=32
           SRC=results_qa_das_v1u/llama3.1-8b/full; TCODE=code_e45_20261112; NCARD=1; NEEDT=20000; REF=16 ;;
    q9)    NG=qwen3.5-9b; MCFG=qwen3.5-9b; ORIG=results_newmodels_ff_20261112/e44/qwen3.5-9b_E2; NL=32
           SRC=results_newmodels_ff_20261112/source/qwen3.5-9b; TCODE=code_ff_explore_20261113; NCARD=1; NEEDT=44000; REF=15 ;;
    q27)   NG=qwen3.8-27b; MCFG=qwen3.8-27b-dm; ORIG=results_newmodels_ff_20261112/e44/qwen3.8-27b-dm; NL=64
           SRC=results_newmodels_ff_20261112/source/qwen3.8-27b-dm; TCODE=code_ff_explore_20261113b; NCARD=2; NEEDT=34000; REF=28 ;;
    g12)   NG=gemma3-12b; MCFG=gemma3-12b; ORIG=Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b; NL=48
           SRC=Iteration_01/results/ff/gemma3/12b/source/gemma3-12b; TCODE=Iteration_01/code; NCARD=1; NEEDT=42000; REF=24 ;;
    g27)   NG=gemma3-27b; MCFG=gemma3-27b; ORIG=Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b; NL=62
           SRC=Iteration_01/results/ff/gemma3/27b/source/gemma3-27b; TCODE=Iteration_01/code; NCARD=1; NEEDT=66560; REF=28
           XENV=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) ;;
    *) echo "unknown key $1" >&2; return 2 ;;
  esac
  RUN=$RR/$NG/run_E2; LOG=$L/relayer_1128_train_$KEY.log
  PYM=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("python") or "python3")' $TCODE/configs/models/$MCFG.json) || return 2
  return 0
}
window() {  # window NL: the middle-50% layers (section 4), deepest first
  local l
  for l in $(seq $(($1 - 1)) -1 0); do
    [ $((10 * (l + 1))) -gt $((3 * $1)) ] && [ $((10 * (l + 1))) -le $((8 * $1)) ] && echo $l
  done | tr '\n' ' '
}
missing() {  # missing: window layers of the current model without a full direction + train log in the new run
  local l out=""
  for l in $(window $NL); do [ -s $RUN/dirs/E_L$l.pt ] && [ -s $RUN/train_logs/E_L$l.json ] || out="$out $l"; done
  echo $out
}
cpv() {  # cpv SRC DST: byte copy, never overwriting
  if [ -e "$2" ]; then cmp -s "$1" "$2" || { echo "$2 exists and differs from $1" >&2; return 1; }; return 0; fi
  cp -p "$1" "$2" && cmp -s "$1" "$2"
}

setup() {
  local key l n
  for key in llama q9 q27 g12 g27; do
    setm $key || { say "setup: settings of $key"; return 1; }
    [ -s $ORIG/frozen_train.json ] && [ -s $ORIG/frozen_tune.json ] || { say "setup: no frozen files in $ORIG"; return 1; }
    mkdir -p $RUN/dirs $RUN/train_logs || return 1
    cpv $ORIG/frozen_train.json $RUN/frozen_train.json && cpv $ORIG/frozen_tune.json $RUN/frozen_tune.json \
      || { say "setup: frozen copy failed for $key"; return 1; }
    n=0
    for l in $(window $NL); do
      if [ -s $ORIG/dirs/E_L$l.pt ]; then
        [ -s $ORIG/train_logs/E_L$l.json ] || { say "setup: $ORIG has E_L$l.pt without its train log"; return 1; }
        cpv $ORIG/dirs/E_L$l.pt $RUN/dirs/E_L$l.pt && cpv $ORIG/train_logs/E_L$l.json $RUN/train_logs/E_L$l.json \
          || { say "setup: direction copy failed for $key layer $l"; return 1; }
        n=$((n + 1))
      fi
    done
    python3 - $ORIG $RUN <<'EOF' || { say "setup: manifest of $key"; return 1; }
import hashlib, json, os, sys
orig, run = sys.argv[1], sys.argv[2]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
rows = []
for sub in ("", "dirs", "train_logs"):
    d = os.path.join(run, sub)
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p) or name.startswith("t.") or name == "relayer_1128_manifest.json":
            continue
        if sub == "" and name not in ("frozen_train.json", "frozen_tune.json"):
            continue
        q = os.path.join(orig, sub, name)
        if not os.path.exists(q):
            continue          # trained here, not a copy
        a, b = sha(p), sha(q)
        if a != b:
            sys.exit("%s differs from %s" % (p, q))
        rows.append({"copy": p, "source": q, "sha256": a})
man = os.path.join(run, "relayer_1128_manifest.json")
new = {"note": "byte copies from the main-table E run (relayer_1128_train.sh setup); E runs only read", "copies": rows}
if os.path.exists(man):
    old = json.load(open(man))
    if old != new:
        sys.exit("%s exists with other content" % man)
else:
    with open(man, "w") as f:
        json.dump(new, f, indent=1)
print("%s: %d copies verified" % (run, len(rows)))
EOF
    say "setup $key: $RUN, frozen 2 + window directions $n copied and verified; missing [$(missing)]"
  done
  touch $L/RELAYER_SETUP_DONE
}

behind_relearn() {  # true when the FalseQA relearn chain no longer needs a card it does not hold
  [ -e $L/RELEARN_FULL_DONE ] && return 0
  local j
  for j in T_q9_falseqa_C T_q9_falseqa_E; do
    [ -e $L/RELEARN_${j}_TAKEN ] || return 1
    [ -e $L/RELEARN_${j}_DONE ] || [ -e $L/RELEARN_${j}_FAILED ] || [ -e $L/RELEARN_${j}_OOM ] || \
      grep -q "relearn: $j: card " $RLOG 2>/dev/null || return 1
  done
  return 0
}
CARDS=""; FDS=()
getcards() {  # getcards N MIB: N cards whose lock is free and that have room; locks held (FDS) until putcards
  local n=$1 need=$2 g fd used got
  until behind_relearn; do sleep 60; done
  while true; do
    got=(); FDS=()
    for g in 0 1 2 3; do
      [ ${#got[@]} -ge $n ] && break
      exec {fd}>"$HNS/selfjudge/.gpu$g.lock" || return 1
      if flock -n $fd; then
        used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$g '$1 == g {print $2}')
        if [[ $used =~ ^[0-9]+$ ]] && [ $((used + need)) -le $MAXMIB ]; then got+=($g); FDS+=($fd); continue; fi
      fi
      exec {fd}>&-
    done
    if [ ${#got[@]} -ge $n ]; then CARDS=$(IFS=,; echo "${got[*]}"); say "$KEY: cards $CARDS (asking $need MiB each)"; return 0; fi
    for fd in "${FDS[@]}"; do exec {fd}>&-; done
    FDS=(); sleep 60
  done
}
putcards() { local fd; for fd in "${FDS[@]}"; do exec {fd}>&-; done; say "$KEY: released cards $CARDS"; FDS=(); CARDS=""; }
dtrain() {  # dtrain ARGS...: the unchanged training program in the container, on CARDS
  docker exec -u 1000:1001 -e HOME=$W/hf_cache -e HF_HOME=$W/hf_cache -e TD_MODEL_ROOT=$W/models \
    -e TD_DATA_DIR=$W/code/data/processed -e CUDA_VISIBLE_DEVICES="$CARDS" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$W/$TCODE "${XENV[@]}" -w $W/$TCODE truth_dec \
    "$PYM" scripts/e44_factorial_das.py train --source-run $W/$SRC --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 \
    --e-on E2 --run-dir $W/$RUN --family E "$@"
}
accept() {  # accept LAYER pilot|full: the new train log against the reference layer REF of the E run
  python3 - $RUN $1 $ORIG $REF $2 <<'EOF'
import hashlib, json, os, sys
run, l, orig, ref, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
pre = "t." if kind == "pilot" else ""
a = json.load(open("%s/train_logs/%sE_L%d.json" % (run, pre, l)))
b = json.load(open("%s/train_logs/E_L%d.json" % (orig, ref)))
ok = True
for k in ("identity", "identity_sha256", "protocol", "e_on", "frozen_source"):
    same = a.get(k) == b.get(k)
    ok &= same
    print("  L%d %s %s: %s" % (l, kind, k, "same" if same else "DIFFERENT"))
ta, tb = dict(a["training_contract"]), dict(b["training_contract"])
skip = {"layer"} | ({"pilot", "train_ids", "val_ids", "val_n", "targets_sha256"} if kind == "pilot" else set())
diff = sorted(k for k in (set(ta) | set(tb)) - skip if ta.get(k) != tb.get(k))
ok &= not diff and ta.get("layer") == l and a.get("layer") == l and a.get("family") == "E" and a.get("pilot") is (kind == "pilot")
print("  L%d %s training_contract: differing %s (ignored %s); layer %s; family %s; pilot %s" % (
    l, kind, diff or "none", sorted(skip), ta.get("layer"), a.get("family"), a.get("pilot")))
p = "%s/dirs/%s" % (run, a.get("direction_file"))
sha = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
ok &= sha is not None and sha == a.get("direction_file_sha256") and a.get("direction_file") == "%sE_L%d.pt" % (pre, l)
print("  L%d %s direction file %s: %s" % (l, kind, a.get("direction_file"), "matches the log" if sha == a.get("direction_file_sha256") else "MISSING OR DIFFERENT"))
sys.exit(0 if ok else 1)
EOF
}
fail() { say "$KEY FAILED: $1 (see $LOG)"; touch $L/RELAYER_$1_${KEY}_FAILED; }

train_model() {  # train_model KEY: pilot, acceptance, full training, acceptance (cards held by the caller)
  setm $1 || { fail SETTINGS; return 1; }
  [ -e $L/RELAYER_T_${KEY}_DONE ] && { say "$KEY: training already done"; return 0; }
  local miss first l rc
  miss=$(missing)
  [ -z "$miss" ] && { touch $L/RELAYER_T_${KEY}_DONE; say "$KEY: nothing to train"; return 0; }
  first=${miss%% *}
  if [ ! -e $L/RELAYER_P_${KEY}_DONE ]; then
    echo "=== $(date '+%F %T') $KEY pilot on cards $CARDS: layer $first, --n-check 3 --val-n 1" >> $LOG
    dtrain --layers $first --val-n 1 --n-check 3 >> $LOG 2>&1 || { fail P; return 1; }
    accept $first pilot >> $LOG 2>&1 || { fail P; return 1; }
    touch $L/RELAYER_P_${KEY}_DONE; say "$KEY: pilot on layer $first passed"
  fi
  echo "=== $(date '+%F %T') $KEY training on cards $CARDS: layers $miss" >> $LOG
  dtrain --layers $miss --val-n 16 >> $LOG 2>&1; rc=$?
  for l in $miss; do
    if [ -s $RUN/dirs/E_L$l.pt ] && [ -s $RUN/train_logs/E_L$l.json ]; then
      accept $l full >> $LOG 2>&1 || { fail T; return 1; }
    fi
  done
  miss=$(missing)
  if [ $rc = 0 ] && [ -z "$miss" ]; then touch $L/RELAYER_T_${KEY}_DONE; say "$KEY: training done and accepted"; return 0; fi
  if [ $rc != 0 ] && grep -qE "OutOfMemoryError|CUDA out of memory" $LOG; then
    echo "out of memory; not done: $miss" > $L/RELAYER_T_${KEY}_OOM; say "$KEY: OOM, not done: $miss"; return 0
  fi
  fail T; return 1
}
smoke_model() {  # smoke_model KEY: the selection pipeline's GPU smoke (relayer_1128.sh smoke KEY) on the held cards
  [ -e $L/RELAYER_SMOKE_$1_DONE ] && return 0
  CARDS=$CARDS bash $L/relayer_1128.sh smoke $1 >> $L/relayer_1128_smoke_$1.out 2>&1
  [ -e $L/RELAYER_SMOKE_$1_DONE ] && say "$1: selection smoke passed" || say "$1: selection smoke FAILED (see relayer_1128_smoke_$1.out); lane goes on"
  return 0
}
lane() {  # lane N STEP...: STEP = smoke:KEY | train:KEY; one card set per model
  local n=$1 s kind key cur=""; shift
  for s in "$@"; do
    kind=${s%%:*}; key=${s#*:}
    setm $key || return 1
    if [ "$cur" != "$key" ]; then
      [ -n "$cur" ] && putcards
      getcards $NCARD $NEEDT || return 1
      cur=$key
    fi
    if [ $kind = smoke ]; then smoke_model $key; else train_model $key || { putcards; say "lane $n stopped"; return 1; }; fi
  done
  [ -n "$cur" ] && putcards
  touch $L/RELAYER_LANE${n}_DONE; say "lane $n done"
}

case "$MODE" in
setup) setup || exit 1 ;;
queue)
  [ -e $L/RELAYER_SETUP_DONE ] || { echo "run setup first" >&2; exit 2; }
  [ -s $L/relayer_1128.sh ] || { echo "missing $L/relayer_1128.sh" >&2; exit 2; }
  say "queue start (pid $$; sha256 $(sha256sum $0 | cut -c1-16)); waiting behind the relearn chain"
  lane 1 smoke:q27 train:q27 &
  sleep 2; lane 2 train:g27 &
  sleep 2; lane 3 smoke:llama smoke:q9 smoke:g12 smoke:g27 train:g12 train:q9 &
  wait
  touch $L/RELAYER_TRAIN_QUEUE_DONE; say "queue ended" ;;
*) echo "usage: relayer_1128_train.sh setup|queue" >&2; exit 2 ;;
esac
