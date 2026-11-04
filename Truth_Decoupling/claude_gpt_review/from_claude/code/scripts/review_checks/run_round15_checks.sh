#!/usr/bin/env bash
# Round 15 checks (Claude, 2026-11-04). Runs on the Pro 5000 HOST; python inside the truth_dec container, CPU only.
# Logs/JSON under review_e25/checks/gpt15/. GPT's two "reproduce" scripts assert the OLD (buggy) behaviour, so on the fixed
# code they are EXPECTED to stop at their first "accepted" assertion; their logs are kept as evidence of what changed.
set -uo pipefail
ROOT_HOST=/data/sanshizhao/Truth_Decoupling
ROOT=/workspace/Truth_Decoupling
REV=$ROOT/review_e25
CODE=$REV/claude_gpt_review/from_claude/code
G15=$REV/checks/gpt15
LOGD=$ROOT_HOST/review_e25/checks/gpt15
mkdir -p "$LOGD"
run() {   # run <workdir> <python args...>
  local wd="$1"; shift
  docker exec -u 1000:1001 -e HOME="$ROOT/hf_cache" -e TD_RESULTS_DIR="$REV/results" -e TD_DATA_DIR="$ROOT/code/data/processed" \
    -e TD_MODEL_ROOT="$ROOT/models" -e CUDA_VISIBLE_DEVICES= -w "$wd" truth_dec python3 "$@"
}
step() {  # step <name> <log> <run args...>
  local name="$1" log="$2"; shift 2
  echo "== $(date '+%F %T') $name"
  run "$@" > "$log" 2>&1; local rc=$?
  echo "exit $rc"
}
step "py_compile (fixed files)" "$LOGD/py_compile.log" "$CODE" -m py_compile core/directions.py scripts/e23_merge.py \
  "$REV/checks/chk_das_loader.py" "$REV/checks/chk_e25_data_tokens.py"
step "chk_das_loader round 15 (vs cb3f67b loader)" "$LOGD/das_loader15.log" "$CODE" "$REV/checks/chk_das_loader.py" "$CODE" \
  "$REV/checks/das_loader15.json" "$REV/checks/gpt14/directions_old.py"
rm -f "$LOGD/loader_edges_on_fixed.json"
step "GPT reproduce_loader_edges on the FIXED code (expected to stop at its first 'accepted' assert)" "$LOGD/gpt_reproduce_loader_edges.log" \
  "$G15" "$G15/reproduce_loader_edges.py" --code-root "$CODE" --old-directions "$REV/checks/gpt14/directions_old.py" --out "$G15/loader_edges_on_fixed.json"
rm -f "$LOGD/last_layer_causal_claude.json"
step "GPT verify_answer_end_causal_path (random small Llama)" "$LOGD/gpt_causal_path.log" "$G15" "$G15/verify_answer_end_causal_path.py" \
  --code-root "$CODE" --out "$G15/last_layer_causal_claude.json"
step "GPT verify_selfcheck_tokens (3 tokenizers, suffix + labels)" "$LOGD/gpt_selfcheck_tokens.log" "$G15" "$G15/verify_selfcheck_tokens.py" \
  --code-root "$CODE" --model-config-dir "$ROOT/code/experiments/configs/models"
step "GPT reproduce_selection_edges on the FIXED merge (expected to stop at the bad-merge assert)" "$LOGD/gpt_reproduce_selection_edges.log" \
  "$G15" "$G15/reproduce_selection_edges.py" --code-root "$CODE" --data-root "$ROOT"
step "chk_e25_data_tokens (prefix assertion added)" "$LOGD/e25_data_tokens15.log" "$CODE" "$REV/checks/chk_e25_data_tokens.py" "$CODE" \
  "$REV/checks/e25_data_tokens15.json"
echo "== $(date '+%F %T') host /tmp fixtures left by GPT (read-only reuse for the real-cache merge check)"
ls -d /tmp/td_review15* 2>/dev/null
echo "== $(date '+%F %T') done"
