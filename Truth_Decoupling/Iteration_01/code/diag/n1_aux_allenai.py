# Iteration_01 analysis (Claude Fable 5.2, 2026-11-18), read-only, no model, no judge call.
# Auxiliary check of scheme N1 (release_origin.py) with the second judge: the settings chosen on the tune split with DeepSeek Pro,
# scored on the FULL test split (523 questions) with AllenAI verdicts. AllenAI is auxiliary; conclusions rest on Pro.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/n1_aux_allenai.py
import os
import sys
sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import two_caps as tc
import release_origin as ro
CAPS = ro.CAPS
for name, m in cf.MODELS.items():
    tune = ps.Split(m, 0)
    ttx = ro.texts(m["ec"][0])
    work = tc.AUX.get(name) or m["ec"][1]
    if not os.path.exists(work + "/judge2/single_allenai.jsonl"):
        print("%-30s no AllenAI verdicts at %s" % (name, work))
        continue
    m2 = dict(m, ec=(m["ec"][0], work), d=None, judge=(m["judge"][0], "allenai"))
    test = ps.Split(m2, 1)
    etx = ro.texts(work)
    fam = {"single E": [("E", "1"), ("E", "2")],
           "method": [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in ps.LAD],
           "N1": [("E", s) for s in "12"] + [("N1", s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]}
    row = {}
    for f, c in fam.items():
        pick = max(c, key=lambda x: (ps.score(ro.states(tune, ttx, x), 1), -c.index(x)))
        row[f] = ro.states(test, etx, pick)
    print("%-30s AllenAI, %d test questions: single E %4d | method %4d | N1 %4d | N1 minus single E %-18s | N1 minus method %s" % (
        name, len(test.qs), ps.score(row["single E"], 1), ps.score(row["method"], 1), ps.score(row["N1"], 1),
        ps.ci([x - y for x, y in zip(cf.val(row["N1"]), cf.val(row["single E"]))]),
        ps.ci([x - y for x, y in zip(cf.val(row["N1"]), cf.val(row["method"]))])))
