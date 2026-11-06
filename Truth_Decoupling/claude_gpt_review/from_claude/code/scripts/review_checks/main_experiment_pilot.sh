#!/usr/bin/env bash
# Run inside the model environment. Only three-question wiring acceptance.
# MODEL QA_SOURCE TF_DAS_DIR TF_DAS_SWEEP TF_MD_SWEEP OUTPUT_ROOT
# Model/data roots, GPU and judge credentials come from existing environment.
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "usage: $0 MODEL QA_SOURCE TF_DAS_DIR TF_DAS_SWEEP TF_MD_SWEEP OUTPUT_ROOT" >&2
  exit 2
fi
MODEL=$1
QA_SOURCE=$2
TF_DAS_DIR=$3
TF_DAS_SWEEP=$4
TF_MD_SWEEP=$5
OUTPUT_ROOT=$6
CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CODE_DIR"
MAIN_RUN="$OUTPUT_ROOT/e26/$MODEL/pilot3"
ADAPTER_RUN="$OUTPUT_ROOT/e27/$MODEL/pilot3_das"
python3 scripts/e26_matched_coverage.py prepare --model "$MODEL" --run-dir "$MAIN_RUN" \
  --qa-run "$QA_SOURCE" --tf-das-dir "$TF_DAS_DIR" --tf-das-sweep "$TF_DAS_SWEEP" \
  --tf-md-sweep "$TF_MD_SWEEP" --pilot 3
for PHASE in collect-tune judge-tune freeze collect-test judge-test report; do
  python3 scripts/e26_matched_coverage.py "$PHASE" --model "$MODEL" --run-dir "$MAIN_RUN" --judge-exp E4_qa
done
python3 scripts/e27_qa_generation.py train --source-run "$QA_SOURCE" --run-dir "$ADAPTER_RUN" \
  --init das --alpha 1 --epochs 1 --pilot 3
python3 scripts/e27_qa_generation.py check --source-run "$QA_SOURCE" --run-dir "$ADAPTER_RUN"
python3 scripts/e27_qa_generation.py judge-check --source-run "$QA_SOURCE" --run-dir "$ADAPTER_RUN" --judge-exp E4_qa
echo "MAIN_EXPERIMENT_PILOT_DONE $MODEL"
