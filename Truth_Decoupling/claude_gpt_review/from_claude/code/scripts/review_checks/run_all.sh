#!/usr/bin/env bash
# Review checks for GPT round 12 (E path). Runs on the Pro 5000 HOST; python inside the truth_dec container, CPU only.
set -u
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
REV=$ROOT/review_e25
LOGD=$ROOT_HOST/review_e25/checks
CODE=$REV/claude_gpt_review/from_claude/code
rm -f "$LOGD/ALL.DONE"
mkdir -p "$ROOT_HOST/review_e25/results"
run() {   # run <workdir> <python args...>
  local wd="$1"; shift
  docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$REV/results" -e TD_DATA_DIR="$ROOT/code/data/processed" \
    -e TD_MODEL_ROOT="$ROOT/models" -e CUDA_VISIBLE_DEVICES= -w "$wd" truth_dec python3 "$@"
}
echo "== $(date '+%F %T') GPT offline tests (55)"
run "$REV" -m unittest discover -s claude_gpt_review/from_gpt -p 'test_*20261103.py' -v > "$LOGD/gpt_tests.log" 2>&1
echo "exit $?"
echo "== $(date '+%F %T') intervene.py old-vs-new regression"
run "$CODE" "$REV/checks/chk_intervene_regress.py" "$ROOT/code/experiments/core/intervene.py" "$CODE/core/intervene.py" \
  "$REV/checks/intervene_regress.json" > "$LOGD/intervene_regress.log" 2>&1
echo "exit $?"
echo "== $(date '+%F %T') E25 data + tokenizer checks"
run "$CODE" "$REV/checks/chk_e25_data_tokens.py" "$CODE" "$REV/checks/e25_data_tokens.json" > "$LOGD/e25_data_tokens.log" 2>&1
echo "exit $?"
echo "== $(date '+%F %T') done"
touch "$LOGD/ALL.DONE"
