#!/usr/bin/env bash
# Same layer, alpha, objective, training data, budget for DAS/MD/random starts.
# QA_SOURCE OUTPUT_ROOT [ALPHA=1] [EPOCHS=3] [RANKING_WEIGHT=0]
# Executes training + TUNING generation/judging only. Test is a separate command.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 QA_SOURCE OUTPUT_ROOT [ALPHA] [EPOCHS] [RANKING_WEIGHT]" >&2
  exit 2
fi
QA_SOURCE=$1
OUTPUT_ROOT=$2
ALPHA=${3:-1}
EPOCHS=${4:-3}
RANKING_WEIGHT=${5:-0}
CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CODE_DIR"
for INIT in das md random; do
  RUN="$OUTPUT_ROOT/$INIT"
  python3 scripts/e27_qa_generation.py train --source-run "$QA_SOURCE" --run-dir "$RUN" \
    --init "$INIT" --alpha "$ALPHA" --epochs "$EPOCHS" --ranking-weight "$RANKING_WEIGHT"
  python3 scripts/e27_qa_generation.py generate-tune --source-run "$QA_SOURCE" --run-dir "$RUN"
  python3 scripts/e27_qa_generation.py judge-tune --source-run "$QA_SOURCE" --run-dir "$RUN" --judge-exp E4_qa
done
echo "GENERATION_ADAPTER_CONTROLS_TUNE_DONE $OUTPUT_ROOT"
