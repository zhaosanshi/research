#!/usr/bin/env bash
# Positive acceptance checks, CPU only. Historical reproduce_* scripts assert
# that bugs EXIST and are not acceptance gates after their fixes. Their archived
# logs remain evidence; every command run here must exit zero.
set -euo pipefail
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
REV=$ROOT/review_e25
CODE=$REV/claude_gpt_review/from_claude/code
CHECKS=$CODE/scripts/review_checks
G15=$REV/checks/gpt15
# The historical causal/token checks are already staged here on the GPU host;
# a caller can supply another container directory without copying stale checks.
AUDIT=${TD_ROUND15_AUDIT_DIR:-$G15}
LOGD=$ROOT_HOST/review_e25/checks/gpt15
mkdir -p "$LOGD"
rm -f "$LOGD/ROUND15.DONE"
fail=0
run() {   # run <workdir> <python args...>
  local wd="$1"; shift
  docker exec -u 1000:1001 -e TD_RESULTS_DIR="$REV/results" -e TD_DATA_DIR="$ROOT/code/data/processed" \
    -e TD_MODEL_ROOT="$ROOT/models" -e CUDA_VISIBLE_DEVICES= -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -w "$wd" truth_dec python3 "$@"
}
step() {  # step <name> <log> <run args...>
  local name="$1" log="$2"; shift 2
  echo "== $(date '+%F %T') $name"
  if run "$@" > "$log" 2>&1; then
    echo "exit 0"
  else
    local rc=$?
    echo "exit $rc (FAILED: $name)"
    fail=1
  fi
}
step "py_compile (current files)" "$LOGD/py_compile.log" "$CODE" -m py_compile \
  core/directions.py core/model_io.py core/qa_precision.py scripts/e23_merge.py \
  "$CHECKS/chk_das_loader.py" "$CHECKS/chk_e25_data_tokens.py" "$CHECKS/chk_merge_guard.py"
step "offline unit tests (including QA precision)" "$LOGD/offline_tests.log" "$REV" \
  -m unittest discover -s claude_gpt_review/from_gpt -p 'test_*2026110*.py' -v
# The fixed checker also works without a manually staged old loader.
step "DAS loader positive acceptance" "$LOGD/das_loader15.log" "$CODE" \
  "$CHECKS/chk_das_loader.py" "$CODE" "$G15/das_loader15.json"
# verify_answer_end_causal_path.py refuses an existing --out (argparse exit 2); without this the driver cannot be re-run
# (round 16: the second run failed only here). Keep the previous result under a dated name instead of losing it.
[ -f "$LOGD/last_layer_causal_claude.json" ] && mv -f "$LOGD/last_layer_causal_claude.json" "$LOGD/last_layer_causal_claude.prev_$(date '+%Y%m%d%H%M%S').json"
step "answer-end causal path (random small model)" "$LOGD/gpt_causal_path.log" "$CODE" "$AUDIT/verify_answer_end_causal_path.py" \
  --code-root "$CODE" --out "$G15/last_layer_causal_claude.json"
step "three local tokenizers (fixed suffix and labels)" "$LOGD/gpt_selfcheck_tokens.log" "$CODE" "$AUDIT/verify_selfcheck_tokens.py" \
  --code-root "$CODE" --model-config-dir "$CODE/configs/models"
step "E25 data, tokens and prefix equality" "$LOGD/e25_data_tokens15.log" "$CODE" "$CHECKS/chk_e25_data_tokens.py" "$CODE" \
  "$G15/e25_data_tokens15.json"
step "Qwen merge guard (cached data)" "$LOGD/merge_guard_qwen_md288.log" "$CODE" \
  "$CHECKS/chk_merge_guard.py" "$CODE" "$ROOT/results_das_tf_r288/E23/qwen3-8b" \
  meandiff_full qwen3-8b "$G15/merge_guard_qwen_md288"
step "Llama merge guard (cached data)" "$LOGD/merge_guard_llama_das288.log" "$CODE" \
  "$CHECKS/chk_merge_guard.py" "$CODE" "$ROOT/results_das_tf_r288/E23/llama3.1-8b" \
  das_tf_full llama3.1-8b "$G15/merge_guard_llama_das288"
if [ "$fail" -ne 0 ]; then
  echo "SOME CHECKS FAILED (no ROUND15.DONE written)"
  exit 1
fi
touch "$LOGD/ROUND15.DONE"
echo "ALL POSITIVE CHECKS PASSED"
