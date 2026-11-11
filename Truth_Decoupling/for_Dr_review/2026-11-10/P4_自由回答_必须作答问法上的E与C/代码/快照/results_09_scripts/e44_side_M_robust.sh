#!/usr/bin/env bash
# Robustness of the side-branch M candidate layers on the tuning questions. No code change: the reviewed driver only.
#   MODE=seed SEED=n LAYERS="11 13" GPU=k   retrain with another optimisation seed in its own run dir (the side branch's
#                                           frozen files byte for byte; same holdout partition, --split-seed default 42),
#                                           then generate the frozen-criteria arms with the first round's random axis (--seed 42)
#   MODE=axis SEED=n LAYERS="11 13" GPU=k   the side branch's own direction files byte for byte in their own run dir;
#                                           generate x.w.p1 (must reproduce the side branch's answers exactly) and random axis n
set -euo pipefail
: "${MODE:?}" "${SEED:?}" "${LAYERS:?}" "${GPU:?}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16          # shared server: do not spawn hundreds of CPU threads
ROOT=/workspace/Truth_Decoupling
MAIN=$ROOT/results_e44_factorial_das/llama3.1-8b
SIDE=${MAIN}_side_M
SRC=$ROOT/results_qa_das_v1u/llama3.1-8b/full
CODE=$ROOT/review_e44_claude/claude_gpt_review/from_claude/code
RUN=${SIDE}_${MODE}${SEED}
COMMON=(--source-run "$SRC" --run-dir "$RUN" --batch-size 8 --max-new-tokens 48 --max-target-tokens 24)
step() { echo "=== $(date '+%F %T') [side M $MODE $SEED gpu$GPU] $* ==="; }
mkdir -p "$RUN/dirs"
for s in train tune; do
  [ -f "$RUN/frozen_$s.json" ] || cp "$SIDE/frozen_$s.json" "$RUN/frozen_$s.json"
  cmp "$SIDE/frozen_$s.json" "$RUN/frozen_$s.json"
done
cd "$CODE"
if [ "$MODE" = seed ]; then
  step "train E (low level M) layers $LAYERS, optimisation seed $SEED"
  python scripts/e44_factorial_das.py train "${COMMON[@]}" --family E --e-off M --layers $LAYERS --val-n 16 --seed "$SEED"
  step "generate the frozen-criteria arms with the first round's random axis (--seed 42)"
  python scripts/e44_factorial_das.py generate "${COMMON[@]}" --family E --e-off M --layers $LAYERS --split tune \
    --arms x.w.p1 x.w.m1 x.md.p1 x.rand.p1 x.rand.m1 x_F.w.p1 --seed 42
elif [ "$MODE" = axis ]; then
  for l in $LAYERS; do
    [ -f "$RUN/dirs/E_L$l.pt" ] || cp "$SIDE/dirs/E_L$l.pt" "$RUN/dirs/E_L$l.pt"
    cmp "$SIDE/dirs/E_L$l.pt" "$RUN/dirs/E_L$l.pt"
  done
  step "generate x.w.p1 and random axis $SEED"
  python scripts/e44_factorial_das.py generate "${COMMON[@]}" --family E --e-off M --layers $LAYERS --split tune \
    --arms x.w.p1 x.rand.p1 x.rand.m1 --seed "$SEED"
  step "the learned arm must reproduce the side branch's answers exactly"
  python - "$SIDE" "$RUN" $LAYERS <<'PY'
import json, sys
side, run, layers = sys.argv[1], sys.argv[2], sys.argv[3:]
for l in layers:
    a = json.load(open("%s/gen/E_L%s_x-w-p1_tune.json" % (side, l)))["rows"]
    b = json.load(open("%s/gen/E_L%s_x-w-p1_tune.json" % (run, l)))["rows"]
    bad = [x["pair_id"] for x, y in zip(a, b) if (x["text"], x["token_ids"]) != (y["text"], y["token_ids"])]
    if len(a) != len(b) or bad:
        raise SystemExit("L%s: the learned arm differs on %d questions" % (l, len(bad)))
    print("L%s: the learned arm is identical on %d questions" % (l, len(a)))
PY
else
  echo "unknown MODE $MODE" >&2; exit 2
fi
step "done"
