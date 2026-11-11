#!/usr/bin/env bash
# Free-form E44 (two-factor interchange DAS) for ANY model config (Claude, 2026-11-12): the round-1 main-hypothesis steps of
# results/09_.../scripts/e44_round1.sh with nothing model-specific left in the script. The model enters only through
#   * its source directory (scripts/review_checks/make_e44_source.py: questions.json byte-identical to the old models',
#     bank.json carrying this model's config), and
#   * the interpreter named in its model config ("python").
# Arms, prompts, caps and batch size are e44_round1.sh's, unchanged.
#   MODEL=qwen3.5-9b GPU=0 BRANCH=E|C STAGE=pilot|run  [LAYERS="..."]  newmodel_e44.sh
#   STAGE=pilot : three-question execution check on two layers (L/8 and L/2), own pilot run dir, no judging
#   STAGE=run   : train -> generate (tune arms) -> reference readout for LAYERS (default: every layer)
# Judging (AllenAI / DeepSeek Pro) and layer selection are separate steps with their own plan.
set -euo pipefail
: "${MODEL:?}" "${GPU:?}" "${BRANCH:?}" "${STAGE:?}"
ROOT=/workspace/Truth_Decoupling
HOST=/data/sanshizhao/Truth_Decoupling
CODE=${CODE_PATH:-$ROOT/code_newmodels_20261112}
FF=${FF_ROOT:-$ROOT/results_newmodels_ff_20261112}
SRC=$FF/source/$MODEL
MAIN=$FF/e44/$MODEL
LIKE=$ROOT/results_qa_das_v1u/llama3.1-8b/full
DX=(docker exec -u 1000:1001 -e HOME=$ROOT/hf_cache -e HF_HOME=$ROOT/hf_cache -e TD_MODEL_ROOT=$ROOT/models
    -e TD_DATA_DIR=$ROOT/code/data/processed -e CUDA_VISIBLE_DEVICES="$GPU" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=$CODE -w $CODE truth_dec)
PY=$("${DX[@]}" python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('python', 'python3'))" "$CODE/configs/models/$MODEL.json")
NL=$("${DX[@]}" python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['num_layers'])" "$CODE/configs/models/$MODEL.json")
py() { "${DX[@]}" "$PY" "$@"; }
step() { echo "=== $(date '+%F %T') [$MODEL $BRANCH gpu$GPU $STAGE] $* ==="; }

if [ ! -f "$HOST/${SRC#$ROOT/}/bank.json" ]; then
  step "source directory"
  "${DX[@]}" python3 scripts/review_checks/make_e44_source.py --model "$MODEL" --like "$LIKE" --out "$SRC"
fi
case "$BRANCH" in
  E) FAMILY=E; GEN_ARMS="conf.wq.p1 conf_E1.wq.m1 caut.wq.p1 caut_E1.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.rand.p1"
     MC_ARMS="x.w.p1 x.w.m1 x.md.p1 x.rand.p1" ;;
  C) FAMILY=C; GEN_ARMS="conf.wq.p1 caut.wq.m1 conf_E1.wq.p1 caut_E1.wq.m1 conf.iso.p1 x.w.p1 x.w.m1 x.md.p1 x.md.m1 x.rand.p1 x.rand.m1"
     MC_ARMS="x.w.p1 x.w.m1" ;;
  *) echo "BRANCH must be E or C" >&2; exit 2 ;;
esac
RUN=$MAIN; PIL=(); VAL=16; LAYERS=${LAYERS:-$(seq 0 $((NL - 1)) | tr '\n' ' ')}
if [ "$STAGE" = pilot ]; then
  RUN=${MAIN}_pilot_$BRANCH; PIL=(--n-check 3); VAL=1; LAYERS="$((NL / 8)) $((NL / 2))"
elif [ "$STAGE" != run ]; then
  echo "unknown STAGE $STAGE" >&2; exit 2
fi
COMMON=(--source-run "$SRC" --run-dir "$RUN" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
mkdir -p "$HOST/${RUN#$ROOT/}"
if [ "$STAGE" = pilot ] || [ ! -f "$HOST/${RUN#$ROOT/}/frozen_tune.json" ]; then
  step "freeze train + tune"
  py scripts/e44_factorial_das.py freeze "${COMMON[@]}" --splits train tune "${PIL[@]}"
fi
step "train $FAMILY layers: $LAYERS"
py scripts/e44_factorial_das.py train "${COMMON[@]}" --family "$FAMILY" --layers $LAYERS --val-n "$VAL" "${PIL[@]}"
step "generate tune arms: $GEN_ARMS"
py scripts/e44_factorial_das.py generate "${COMMON[@]}" --family "$FAMILY" --layers $LAYERS --split tune --arms $GEN_ARMS "${PIL[@]}"
step "reference readout: $MC_ARMS"
py scripts/e44_factorial_das.py mc "${COMMON[@]}" --family "$FAMILY" --layers $LAYERS --split tune --arms $MC_ARMS "${PIL[@]}"
if [ "$STAGE" = pilot ]; then
  step "pilot summary"
  py scripts/e44_factorial_das.py summarize --source-run "$SRC" --run-dir "$RUN" --include-pilot | tail -n 70
fi
step "$STAGE $BRANCH done (layers: $LAYERS)"
