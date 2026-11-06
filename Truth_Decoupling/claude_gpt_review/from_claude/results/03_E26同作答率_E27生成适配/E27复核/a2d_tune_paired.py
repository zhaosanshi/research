"""Qwen alpha-3 tuning: DAS adapter paired comparisons vs baseline / original DAS / MD / random (own code)."""
import json, sys
sys.path.append("/Users/airsanshi/Library/Python/3.9/lib/python/site-packages")
sys.path.insert(0, "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad/audit_results")
import a2_recompute_lib as L
G = L.GEN + "/qwen3-8b/ce_alpha3_ep3/"
runs = {i: L.table_rows(G + i + "/tune_scored_items.json")[2] for i in ("das", "md", "random")}
ids = [p for p, r in runs["das"].items() if r["subst"]]
lab = {"baseline": [runs["das"][p]["lab"]["baseline"] for p in ids],
       "original_das": [runs["das"][p]["lab"]["original_das"] for p in ids]}
for i in ("das", "md", "random"):
    lab[i] = [runs[i][p]["lab"][i + "_generation_adapter"] for p in ids]
for other in ("baseline", "original_das", "md", "random"):
    r = L.paired(lab["das"], lab[other])
    print("tune DAS vs %-12s net %+d (%d/%d) p=%.6g CI %s" % (other, r["net_correct"], r["a_correct_b_not"], r["b_correct_a_not"],
          r["mcnemar_exact_p"], [round(x, 4) for x in r["ci95_correct_rate"]]))
