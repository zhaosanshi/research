#!/usr/bin/env bash
# Shared future-run driver. Scientific commands stay in the three launchers.
set -euo pipefail
ROOT=${TD_PROJECT_ROOT:-/workspace/Truth_Decoupling}
HOST=${TD_HOST_PROJECT_ROOT:-/data/sanshizhao/Truth_Decoupling}
: "${CODE_PATH:?Set CODE_PATH to the reviewed container code directory}"
: "${RESULTS_PATH:?Set RESULTS_PATH to a fresh container result directory}"
CODE=$CODE_PATH
OUT=$RESULTS_PATH
CONTAINER=${TD_CONTAINER:-truth_dec}
case "$OUT" in "$ROOT"/*) ;; *) echo 'RESULTS_PATH must be inside TD_PROJECT_ROOT' >&2; exit 2 ;; esac
HOST_OUT=$HOST/${OUT#"$ROOT"/}
R19_PIDS=()
R19_NAMES=()

r19_cleanup() {
  local status=$?
  trap - EXIT
  if [[ $status -ne 0 && ! -e "$R19_FAILED" ]]; then
    printf 'driver_exit=%s\n' "$status" > "$R19_FAILED" || true
  fi
  rm -f "$R19_DONE.tmp"
  rmdir "$R19_RUNNING" 2>/dev/null || true
  exit "$status"
}

r19_init() {
  local group=$1; shift
  R19_EXPECTED=("$@")
  [[ ${#R19_EXPECTED[@]} -gt 0 ]] || { echo 'No declared jobs' >&2; exit 2; }
  R19_LOGDIR=$HOST_OUT/logs
  mkdir -p "$R19_LOGDIR"
  R19_DONE=$R19_LOGDIR/$group.DONE
  R19_FAILED=$R19_LOGDIR/$group.FAILED
  R19_RUNNING=$R19_LOGDIR/$group.RUNNING
  local path name
  for path in "$R19_DONE" "$R19_DONE.tmp" "$R19_FAILED" "$R19_RUNNING"; do
    [[ ! -e "$path" && ! -L "$path" ]] || { echo "Preserve prior run artifact: $path" >&2; exit 2; }
  done
  for name in "${R19_EXPECTED[@]}"; do
    for path in "$R19_LOGDIR/$name.log" "$R19_LOGDIR/$name.exit"; do
      [[ ! -e "$path" && ! -L "$path" ]] || { echo "Preserve prior job artifact: $path" >&2; exit 2; }
    done
  done
  mkdir "$R19_RUNNING"  # Atomic reservation; two same-group invocations cannot both launch.
  trap r19_cleanup EXIT
}

r19_run() {
  trap - EXIT  # A background worker must not execute the parent's cleanup trap.
  local gpu=$1 log=$2 status=0; shift 2
  if docker exec -u 1000:1001 -e HF_HOME="$ROOT/hf_cache" -e TD_DATA_DIR="$ROOT/code/data/processed" -e TD_MODEL_ROOT="$ROOT/models" \
    -e TD_RESULTS_DIR="$OUT" -e CUDA_VISIBLE_DEVICES="$gpu" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -w "$CODE" "$CONTAINER" python3 "$@" > "$R19_LOGDIR/$log.log" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "$status" > "$R19_LOGDIR/$log.exit" || return 125
  return "$status"
}

r19_launch() {
  local name=$2 found=0 prior
  for prior in "${R19_EXPECTED[@]}"; do [[ $prior != "$name" ]] || found=1; done
  [[ $found -eq 1 ]] || { echo "Undeclared job: $name" >&2; return 2; }
  for prior in "${R19_NAMES[@]-}"; do [[ $prior != "$name" ]] || { echo "Duplicate job: $name" >&2; return 2; }; done
  r19_run "$@" &
  R19_PIDS+=("$!")
  R19_NAMES+=("$name")
}

r19_wait_all() {
  local failed=0 status i
  [[ ${#R19_PIDS[@]} -eq ${#R19_EXPECTED[@]} ]] || failed=1
  for i in "${!R19_PIDS[@]}"; do
    if wait "${R19_PIDS[$i]}"; then status=0; else status=$?; failed=1; fi
    if [[ $status -ne 0 ]]; then printf 'FAILED %s exit=%s\n' "${R19_NAMES[$i]}" "$status" >&2; fi
  done
  if [[ $failed -ne 0 ]]; then
    printf 'one_or_more_declared_jobs_failed\n' > "$R19_FAILED"
    return 1
  fi
  date > "$R19_DONE.tmp"
  mv "$R19_DONE.tmp" "$R19_DONE"
}
