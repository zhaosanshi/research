#!/usr/bin/env bash
# Robustness of the main must-answer E result (layer 10, strength 2, base conf) on the tuning questions.
# No code change: the reviewed driver only (same pattern as e44_side_M_robust.sh, main branch, truth levels none / E1).
#   MODE=seed SEED=n GPU=k   retrain E at layer 10 with another optimisation seed in its own run dir (frozen files byte for
#                            byte, same holdout partition), generate the must-answer arms with the first round's random axis
#   MODE=axis SEED=n GPU=k   the main run's own direction file byte for byte; regenerate conf.w.p2 (must reproduce the main
#                            run's answers exactly) and random axis n at the same strength
set -euo pipefail
: "${MODE:?}" "${SEED:?}" "${GPU:?}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
ROOT=/workspace/Truth_Decoupling
MAIN=$ROOT/results_e44_factorial_das/llama3.1-8b
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
CODE=$ROOT/review_e44_claude/claude_gpt_review/from_claude/code
RUN=${MAIN}_mainE_${MODE}${SEED}
COMMON=(--source-run "$SRC" --run-dir "$RUN" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
step() { echo "=== $(date '+%F %T') [main E $MODE $SEED gpu$GPU] $* ==="; }
mkdir -p "$RUN/dirs"
for s in train tune; do
  [ -f "$RUN/frozen_$s.json" ] || cp "$MAIN/frozen_$s.json" "$RUN/frozen_$s.json"
  cmp "$MAIN/frozen_$s.json" "$RUN/frozen_$s.json"
done
cd "$CODE"
if [ "$MODE" = seed ]; then
  step "train E layer 10, optimisation seed $SEED"
  python scripts/e44_factorial_das.py train "${COMMON[@]}" --family E --layers 10 --val-n 16 --seed "$SEED"
  step "generate the must-answer arms (random axis of the first round: --seed 42)"
  python scripts/e44_factorial_das.py generate "${COMMON[@]}" --family E --layers 10 --split tune \
    --arms conf.w.p1 conf.w.p2 conf.md.p2 conf.rand.p2 conf.rand.m2 conf_E1.w.m2 --seed 42
elif [ "$MODE" = axis ]; then
  [ -f "$RUN/dirs/E_L10.pt" ] || cp "$MAIN/dirs/E_L10.pt" "$RUN/dirs/E_L10.pt"
  cmp "$MAIN/dirs/E_L10.pt" "$RUN/dirs/E_L10.pt"
  step "generate conf.w.p2 and random axis $SEED"
  python scripts/e44_factorial_das.py generate "${COMMON[@]}" --family E --layers 10 --split tune \
    --arms conf.w.p2 conf.rand.p2 conf.rand.m2 --seed "$SEED"
  step "the learned arm must reproduce the main run's answers exactly"
  python - "$MAIN" "$RUN" <<'PY'
import json, sys
a = json.load(open("%s/gen/E_L10_conf-w-p2_tune.json" % sys.argv[1]))["rows"]
b = json.load(open("%s/gen/E_L10_conf-w-p2_tune.json" % sys.argv[2]))["rows"]
bad = [x["pair_id"] for x, y in zip(a, b) if (x["text"], x["token_ids"]) != (y["text"], y["token_ids"])]
if len(a) != len(b) or bad:
    raise SystemExit("the learned arm differs on %d questions" % len(bad))
print("the learned arm is identical on %d questions" % len(a))
PY
else
  echo "unknown MODE $MODE" >&2; exit 2
fi
step "done"
