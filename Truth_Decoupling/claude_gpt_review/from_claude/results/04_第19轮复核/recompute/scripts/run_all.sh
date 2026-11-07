#!/bin/bash
# Re-run every independent recomputation; outputs land in ../outputs/<script>.out
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
O="$S/../outputs"
mkdir -p "$O"
for f in c0_integrity c1_random c2_fffd c3_release c4_trace c5_current_dose c6_generation_dose c7_models c8_pure_das c9_real_traces c10_misc c11_tensors c12_release_llama_cache c13_crossfile c14_small; do
  python3 -I "$S/$f.py" > "$O/$f.out" 2>&1 || { echo "FAILED: $f"; exit 1; }
  echo "ok $f ($(wc -l < "$O/$f.out") lines)"
done
python3 -I "$S/np_pcg64.py" > "$O/np_pcg64_selfcheck.out"
