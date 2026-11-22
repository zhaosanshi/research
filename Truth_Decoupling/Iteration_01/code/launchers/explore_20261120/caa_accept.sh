#!/bin/bash
# 2026-11-22 19 h, exploration (Claude Opus 5.5): acceptance of the CAA adaptation for one model on one card
# (discussion/work_queue/2026-11-22_queue_f.md, sections 4 and 7). No judge is called. Steps:
#   1. scripts/newsets/caa_vector.py fit: the CAA vectors from the TruthfulQA training split (all 96 questions, and the
#      half0 vector for the sign check), baselines/caa/<MODEL>/vectors.pt;
#   2. caa_vector.py mirror: run_caa/ (one direction record per layer carrying the CAA vector) and mirror_e/ (the run's
#      real E record of layer EL copied byte for byte), both with copies of the run's frozen files;
#   3. caa_vector.py check: multiplier 0 = no write (candidate layers, first tune batches), the record's md write equals
#      the vector, the sign check on the held-out half, the question-ID check, the template / position record;
#   4. acceptance 1: the real E record, engine kind md at the rule gen (scripts/review_checks/e44_positions.py gen,
#      engine unchanged), first FB tune batches, twice; the two answer files must agree token for token (there is no
#      earlier md-at-gen answer file of any model to compare with);
#   5. a small look, not a choice: the CAA vector at the middle candidate layer, multipliers 1 / 2 / 4, same questions:
#      answers changed and answer length against no write.
# Holds the card-1 lock of the self-judge workers (Iteration_01/newsets/selfjudge/.gpu1.lock, as in
# discussion/validation_dataset/2026-11-22d_probe.patch) and refuses to start when card GPU would exceed 65 GB.
# usage (host): caa_accept.sh MODEL GPU EL [FB]       e.g. caa_accept.sh llama3.1-8b 1 16 2
# Outputs: Iteration_01/baselines/caa/<MODEL>/ ; log Iteration_01/baselines/caa/<MODEL>/acceptance/acceptance.log
set -uo pipefail
cd /data/sanshizhao/Truth_Decoupling || exit 2
ROOT=/workspace/Truth_Decoupling; HOST=/data/sanshizhao/Truth_Decoupling
MODEL=${1:?model}; GPU=${2:?gpu}; EL=${3:?layer of the real E record for acceptance 1}; FB=${4:-2}
case "$MODEL" in llama3.1-8b) NEED=17000 ;; qwen3.5-9b) NEED=20000 ;; gemma3-12b) NEED=26000 ;; *) echo "unknown model" >&2; exit 2 ;; esac
LOCK=$HOST/Iteration_01/newsets/selfjudge/.gpu1.lock
mkdir -p "$(dirname "$LOCK")" || { echo "cannot prepare GPU 1 reservation" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock unavailable on GPU host" >&2; exit 2; }
exec 9>"$LOCK" || { echo "cannot open GPU 1 reservation" >&2; exit 2; }
flock -n 9 || { echo "another worker holds GPU 1" >&2; exit 3; }
USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$GPU" '$1 == g {print $2}') || { echo "cannot query GPU memory" >&2; exit 2; }
[[ $USED =~ ^[0-9]+$ ]] || { echo "cannot read GPU memory" >&2; exit 2; }
if [ $((USED + NEED)) -gt 66560 ]; then echo "GPU $GPU holds $USED MiB; with $NEED MiB more it exceeds 65 GB, not starting" >&2; exit 3; fi
CFG_HOST=$HOST/Iteration_01/code/configs/newsets/frozen_models_20261119.json
CFG=$ROOT/Iteration_01/code/configs/newsets/frozen_models_20261119.json
read -r CODE PY SRC E_ON < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["code"], m["python"], m["source_run"], m["e_on"])' "$CFG_HOST" "$MODEL") || { echo "model not in the config" >&2; exit 2; }
[ -n "$E_ON" ] || { echo "model not in the config" >&2; exit 2; }
CAA=$ROOT/Iteration_01/code/scripts/newsets/caa_vector.py
B=Iteration_01/baselines/caa/$MODEL
A=$B/acceptance
mkdir -p $HOST/$A
LOG=$HOST/$A/acceptance.log
ENV=(-u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models -e TD_DATA_DIR=$ROOT/code/data/processed
     -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUDA_VISIBLE_DEVICES=$GPU)
DXR=(docker exec "${ENV[@]}" -w $ROOT truth_dec)
DXC=(docker exec "${ENV[@]}" -e PYTHONPATH=$ROOT/$CODE -w $ROOT/$CODE truth_dec)
say() { echo "=== $(date '+%F %T') $*" | tee -a $LOG; }
die() { say "FAILED: $*"; touch $HOST/$A/ACCEPT_FAILED; exit 1; }
rm -f $HOST/$A/ACCEPT_DONE $HOST/$A/ACCEPT_FAILED
say "caa acceptance $MODEL on GPU $GPU ($USED MiB used before start), code copy $CODE, interpreter $PY, E layer $EL, first $FB batches"
say "step 1: fit"
if [ ! -s $HOST/$B/vectors.pt ]; then
  "${DXR[@]}" $PY -I $CAA fit --models $CFG --model $MODEL --out $ROOT/$B/vectors.pt >> $LOG 2>&1 || die "fit"
else say "vectors.pt exists, kept"; fi
say "step 2: mirrors"
[ -e $HOST/$B/run_caa ] || "${DXR[@]}" $PY -I $CAA mirror --models $CFG --model $MODEL --mirror $B/run_caa --kind caa --vectors $ROOT/$B/vectors.pt >> $LOG 2>&1 || die "mirror caa"
[ -e $HOST/$B/mirror_e ] || "${DXR[@]}" $PY -I $CAA mirror --models $CFG --model $MODEL --mirror $B/mirror_e --kind real --layers $EL >> $LOG 2>&1 || die "mirror real"
say "step 3: check (acceptance 2 to 5)"
if [ ! -s $HOST/$A/check.json ]; then
  "${DXR[@]}" $PY -I $CAA check --models $CFG --model $MODEL --mirror $B/run_caa --vectors $ROOT/$B/vectors.pt \
      --out $ROOT/$A/check.json --first-batches $FB >> $LOG 2>&1 || die "check"
else say "check.json exists, kept"; fi
COMMON=(--source-run $ROOT/$SRC --e-on $E_ON --split tune --batch-size 8 --max-new-tokens 48 --max-target-tokens 24 --seed 42 --first-batches $FB)
NQ=$((FB * 8))
say "step 4: acceptance 1, real E$EL record, kind md at gen, twice"
G=$HOST/$B/mirror_e/eval_extra/gen; F=t$NQ.x--E$EL-md-p2-at-gen_tune.json
for run in A B; do
  if [ ! -s $HOST/$A/c1_run$run.json ]; then
    "${DXC[@]}" $PY scripts/review_checks/e44_positions.py gen generate "${COMMON[@]}" --run-dir $ROOT/$B/mirror_e --arms x:E$EL.md.p2 >> $LOG 2>&1 || die "acceptance 1 run $run"
    mv $G/$F $HOST/$A/c1_run$run.json || die "acceptance 1 run $run: no answer file"
  fi
done
python3 - $HOST/$A/c1_runA.json $HOST/$A/c1_runB.json <<'EOF' 2>&1 | tee -a $LOG || die "acceptance 1 compare"
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
keys = ("text", "token_ids", "finished", "stop_id")
same = sum(all(x[k] == y[k] for k in keys) for x, y in zip(a["rows"], b["rows"]))
amt = {round(p["amount"], 9) for d in (a, b) for p in d["parts"]}
pos = {d["positions"] for d in (a, b)}
changed = sum(r["changed_normalized"] for r in a["rows"])
pre = sorted({r["write"]["prefill_positions"] for d in (a, b) for r in d["rows"]})
print("acceptance 1: %s, positions %s, amount %s, prefill positions written per row %s: %d of %d answers token for token the same in the two runs; "
      "%d of %d differ from no write; baseline reproduction verified %s / %s" % (a["arm"], pos, amt, pre, same, len(a["rows"]), changed, len(a["rows"]),
      a["baseline_reproduction_verified"], b["baseline_reproduction_verified"]))
sys.exit(0 if same == len(a["rows"]) == len(b["rows"]) and pos == {"gen"} and pre == [1] else 1)
EOF
MID=$(python3 -c 'import json, sys; c = json.load(open(sys.argv[1]))["candidate_layers"]; print(c[len(c) // 2])' $HOST/$B/vectors.pt.json) || die "no candidate layers"
say "step 5: small look, CAA at layer $MID, multipliers 1 / 2 / 4 (not a choice)"
G=$HOST/$B/run_caa/eval_extra/gen
"${DXC[@]}" $PY scripts/review_checks/e44_positions.py gen generate "${COMMON[@]}" --run-dir $ROOT/$B/run_caa \
    --arms x:E$MID.md.p1 x:E$MID.md.p2 x:E$MID.md.p4 >> $LOG 2>&1 || die "small look"
python3 - $G/t$NQ.x--E$MID-md-p1-at-gen_tune.json $G/t$NQ.x--E$MID-md-p2-at-gen_tune.json $G/t$NQ.x--E$MID-md-p4-at-gen_tune.json <<'EOF' 2>&1 | tee -a $LOG || die "small look summary"
import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p))
    r = d["rows"]
    tok_changed = sum(1 for x in r if x["changed"])
    words = lambda t: len(t.split())
    print("small look %s: amount %.3f | changed %d of %d (normalized text %d) | mean words %.1f -> %.1f | mean tokens -> %.1f | rule abstain %d -> %d | unfinished %d" % (
        d["arm"], d["parts"][0]["amount"], tok_changed, len(r), d["summary"]["changed_normalized"],
        sum(words(x["baseline_text"]) for x in r) / len(r), sum(words(x["text"]) for x in r) / len(r),
        sum(x["n_tokens"] for x in r) / len(r), sum(x["baseline_refusal_rule"] for x in r), sum(x["refusal_rule"] for x in r),
        d["summary"]["unfinished"]))
EOF
say "done"
touch $HOST/$A/ACCEPT_DONE
