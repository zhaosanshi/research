# -*- coding: utf-8 -*-
"""Read-only: per-layer training table of an E44 run dir (losses per answer token, held-out part separately)."""
import json, glob, os, sys
R = sys.argv[1].rstrip("/") + "/train_logs/"
rows = {}
for p in sorted(glob.glob(R + "*.json")):
    if os.path.basename(p).startswith("t."):
        continue
    d = json.load(open(p)); tr, va = d["losses"]["train"], d["losses"]["val"]
    rows[(d["family"], d["layer"])] = (d["gap"], d["md_norm"], d["cosine_w_md"], tr["no_write"]["cause"], tr["trained"]["cause"], tr["mean_difference_direction"]["cause"],
                                      va["no_write"]["cause"], va["trained"]["cause"], va["mean_difference_direction"]["cause"], va["no_write"]["isolate"], va["trained"]["isolate"], d["seconds"])
print("# E44 training losses per answer token (JSD to the next-token distribution of the counterfactual prompt); 80 training questions, 16 held out (never used for a gradient)")
print()
print("| family | layer | gap | cos(w, mean difference) | train cause: no write -> trained (mean-difference direction) | held-out cause: no write -> trained (mean-difference direction) | held-out change | held-out isolate: no write -> trained | seconds |")
print("|---|---|---|---|---|---|---|---|---|")
for (f, l), r in sorted(rows.items()):
    print("| %s | %d | %.3f | %.2f | %.4f -> %.4f (%.4f) | %.4f -> %.4f (%.4f) | %+.0f%% | %.4f -> %.4f | %.0f |" % (
        f, l, r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8], 100 * (r[7] - r[6]) / r[6], r[9], r[10], r[11]))
