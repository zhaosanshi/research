#!/usr/bin/env bash
# Iteration_01 (Claude Opus 5.5, 2026-11-19): work queue F1, capability check on MMLU (stratified 1,000) and GSM8K (1,319)
# (novelty/ours/2026-11-19_capability_preservation_design.md). Thin wrapper around scripts/newsets/newsets_gen.py with the
# frozen TruthfulQA setup (configs/newsets/frozen_models_20261119.json); scoring is rule-based and done offline (no DeepSeek).
# usage (host side, one model per call):  capability_f1_20261119.sh TASK MODEL GPUS [BATCH]
#   TASK = mmlu (16 new tokens, system prompt = the plain prompt x) or gsm8k (512 new tokens, system prompt "neutral").
#   1. full passes: no write, E at 2, C alone at its largest strength 2 toward answering;
#   2. N1 re-asks: the items the rule detector marks blank (or that are empty) under E get E + C at 0.25 / 0.5 / 1 / 2
#      (only those items; their batches differ from a full pass, recorded in the files).
#   BATCH (8, 16 or 32; default 8) must be the same for every condition of a model.
set -uo pipefail
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=$ROOT/Iteration_01/code
G=$CODE/scripts/newsets/newsets_gen.py
M=$CODE/configs/newsets/frozen_models_20261119.json
TASK=${1:?task}; MODEL=${2:?model}; GPUS=${3:?gpus}; BATCH=${4:-8}
host() { echo "$HOST/${1#$ROOT/}"; }
say() { echo "$(TZ=Asia/Shanghai date '+%F %T') [$TASK $MODEL] $*"; }
fail() { say "FAILED: $*"; exit 1; }
read -r PY EL CL NG <<< "$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["models"][sys.argv[2]]
print(m["python"], m["el"], m["cl"], m["gpus"])' "$(host "$M")" "$MODEL")" || fail "model $MODEL not in the config"
[ "$(echo "$GPUS" | tr ',' '\n' | grep -c .)" = "$NG" ] || fail "$MODEL needs $NG GPU(s), got $GPUS"
case "$TASK" in
  mmlu)  BASE=x;       NEW=16;  ITEMS=$ROOT/data_external/mmlu_20261119/all/test-00000-of-00001.parquet;  IDS=$ROOT/Iteration_01/capability/sets/mmlu_1000_20261119.json ;;
  gsm8k) BASE=neutral; NEW=512; ITEMS=$ROOT/data_external/gsm8k_20261119/main/test-00000-of-00001.parquet; IDS=$ROOT/Iteration_01/capability/sets/gsm8k_1319_20261119.json ;;
  *) fail "unknown task $TASK" ;;
esac
OUT=$ROOT/Iteration_01/capability/gen/$TASK/$MODEL/b$BATCH
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1
    -e CUDA_VISIBLE_DEVICES=$GPUS -w $ROOT truth_dec)
E="E$EL.w.p2"; C="C$CL.w"
run() { "${DX[@]}" $PY -I $G gen --models $M --model $MODEL --task $TASK --items $ITEMS --ids $IDS --max-new-tokens $NEW \
          --batch-size $BATCH --out $OUT "$@" || fail "gen $*"; }
say "full passes (batch $BATCH, $NEW new tokens)"
run --arms "$BASE" "$BASE:$E" "$BASE:$C.m2"
EF=$(host "$OUT")/$(echo "$BASE:$E" | sed -e 's/:/--/' -e 's/\./-/g').json
BL=$(host "$OUT")/n1_blank_ids.json
python3 -c '
import json, sys
g = json.load(open(sys.argv[1]))
ids = [r["id"] for r in g["rows"] if r["rule_abstain"] or not r["text"].strip()]
json.dump(ids, open(sys.argv[2], "w"))
print(len(ids), "of", len(g["rows"]), "blank under E")' "$EF" "$BL" || fail "blank list"
if [ "$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$BL")" -gt 0 ]; then
  say "N1 re-asks"
  run --only-ids "${BL/$HOST/$ROOT}" --arms "$BASE:$E+$C.m0.25" "$BASE:$E+$C.m0.5" "$BASE:$E+$C.m1" "$BASE:$E+$C.m2"
fi
say "done"
