# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# Llama-3.1-8B with the E write at LAYER 16 instead of 11. The table itself now lives in diag/alt_layer.py, which also
# knows the second such case (Gemma-3-27B at layer 28); this file keeps the first version's entry point and names.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/l16_table.py
import sys

sys.path.insert(0, "Iteration_01/diag")
import alt_layer as AL

A = AL.ALTS[0]
MODEL, NG, LAYER, DOSES = A.model, A.ng, A.layer, A.doses
load, mmlu16 = A.load, A.mmlu

if __name__ == "__main__":
    AL.report(A)
