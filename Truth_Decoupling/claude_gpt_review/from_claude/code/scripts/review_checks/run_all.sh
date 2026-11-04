#!/usr/bin/env bash
# Review checks (GPT round 12 E path; round 14 DAS loader). Runs on the Pro 5000 HOST; python inside the truth_dec container, CPU only.
# Exit status 0 and ALL.DONE only if EVERY check exits 0 (review round 13: the first version wrote ALL.DONE unconditionally).
set -uo pipefail
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
REV=$ROOT/review_e25
LOGD=$ROOT_HOST/review_e25/checks
CODE=$REV/claude_gpt_review/from_claude/code
rm -f "$LOGD/ALL.DONE"
mkdir -p "$ROOT_HOST/review_e25/results"
fail=0
run() {   # run <workdir> <python args...>
  local wd="$1"; shift
  docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$REV/results" -e TD_DATA_DIR="$ROOT/code/data/processed" \
    -e TD_MODEL_ROOT="$ROOT/models" -e CUDA_VISIBLE_DEVICES= -w "$wd" truth_dec python3 "$@"
}
step() {  # step <name> <log> <run args...>: a non-zero exit is recorded, never hidden behind a completion flag
  local name="$1" log="$2"; shift 2
  echo "== $(date '+%F %T') $name"
  run "$@" > "$log" 2>&1; local rc=$?
  echo "exit $rc"
  [ "$rc" -eq 0 ] || fail=1
}
step "GPT offline tests (55)" "$LOGD/gpt_tests.log" "$REV" -m unittest discover -s claude_gpt_review/from_gpt -p 'test_*20261103.py' -v
step "intervene.py old-vs-new regression" "$LOGD/intervene_regress.log" "$CODE" "$REV/checks/chk_intervene_regress.py" \
  "$ROOT/code/experiments/core/intervene.py" "$CODE/core/intervene.py" "$REV/checks/intervene_regress.json"
step "E25 data + tokenizer checks" "$LOGD/e25_data_tokens.log" "$CODE" "$REV/checks/chk_e25_data_tokens.py" "$CODE" "$REV/checks/e25_data_tokens.json"
# directions_old.py = the pre-fix loader (git show cb3f67b:Truth_Decoupling/claude_gpt_review/from_claude/code/core/directions.py), staged on the server by hand
step "DAS loader + teacher cache (round 14 D1-D5, round 15 3.1-3.5 / 4)" "$LOGD/das_loader.log" "$CODE" "$REV/checks/chk_das_loader.py" "$CODE" \
  "$REV/checks/das_loader.json" "$REV/checks/gpt14/directions_old.py"
# round 15 (2.2): merge guard on the real Qwen MD-288 cache split into two layer chunks (merged in reversed order) + synthetic conflicts
rm -rf "$LOGD/merge_guard"
step "e23_merge guard (round 15, real cache + conflicts)" "$LOGD/merge_guard.log" "$CODE" "$REV/checks/chk_merge_guard.py" "$CODE" \
  "$ROOT/results_das_tf_r288/E23/qwen3-8b" meandiff_full qwen3-8b "$REV/checks/merge_guard"
echo "== $(date '+%F %T') done"
if [ "$fail" -eq 0 ]; then
  touch "$LOGD/ALL.DONE"; echo "ALL CHECKS PASSED"
else
  echo "SOME CHECKS FAILED (no ALL.DONE written)"; exit 1
fi
