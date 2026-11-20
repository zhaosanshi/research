# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# True/false with an Unknown option in the same pass (the E45 runs: results_e45_20261112 for the 11-12 models,
# results_e45_newmodels_20261119 for Qwen3.5-9B / Qwen3.8-27B), scored the way the free-form answers are scored
# (the lead, 2026-11-20 15:24: true/false in line with free-form): right +1, Unknown 0, wrong -1. The answer of a
# statement is the stored same-pass choice (Unknown when its probability is the largest; otherwise True / False by the
# larger of the two). Test = the 529 test pairs of the free-form split (1,058 statements); the counts must reproduce the
# stored summary of every run. E = the true/false DAS direction of that run at strengths 1 / 2 / 3 (not the free-form E).
# Intervals: question pairs resampled.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/tf_net_score.py
import json
import random

import numpy as np

FILES = {"Llama-3.1-8B": "results_e45_20261112/E45/llama3.1-8b/e45_v1",
         "Qwen3.5-9B": "results_e45_newmodels_20261119/E45/qwen3.5-9b/e45_nm_Cp",
         "Qwen3.8-27B": "results_e45_newmodels_20261119/E45/qwen3.8-27b/e45_nm_Cp",
         "Qwen3-8B (11-12 run, not a current model)": "results_e45_20261112/E45/qwen3-8b/e45_v1"}
FROZEN_TEST = "results_e44_factorial_das/llama3.1-8b_side_E2/frozen_test.json"


def ci(v, n_boot=4000, seed=20261120):
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return "%+d [%+d, %+d]" % (sum(v), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1])


def main():
    with open(FROZEN_TEST, encoding="utf-8") as f:
        test_ids = {q["pair_id"] for q in json.load(f)["questions"]}
    print("True/false with Unknown in the same pass, scored right - wrong (Unknown = 0); %d test pairs" % len(test_ids))
    for name, p in FILES.items():
        z = np.load(p + ".npz", allow_pickle=True)
        with open(p + ".json", encoding="utf-8") as f:
            d = json.load(f)
        ids, labels = [str(x) for x in z["ids"]], z["labels"]

        def val(c):
            ans = z[c + "__answer"]
            pred = z[c + "__lpT"] > z[c + "__lpF"]
            return np.where(ans, np.where(pred == labels, 1, -1), 0), ans

        v0, a0 = val("baseline")
        rows = [i for i, x in enumerate(ids) if x.rsplit(":", 1)[0] in test_ids]
        t = d["conds"]["baseline"]["test"]
        if not (len(rows) == t["n"] and int(a0[rows].sum()) == t["n_answered"] and int((v0[rows] == 1).sum()) == t["correct_answered"]):
            raise SystemExit("%s: the rows do not reproduce the stored test summary" % name)
        pairs = {}
        for i in rows:
            pairs.setdefault(ids[i].rsplit(":", 1)[0], []).append(i)
        print("\n== %s: %d test pairs, %d statements | no write: answered %d, right %d, net %+d" % (
            name, len(pairs), len(rows), int(a0[rows].sum()), int((v0[rows] == 1).sum()), int(v0[rows].sum())))
        for c in ("E_a1", "E_a2", "E_a3"):
            v, a = val(c)
            dv = [int(sum(v[i] - v0[i] for i in idx)) for idx in pairs.values()]
            print("    E strength %s: answered %d, right %d, net %+d | minus no write %s" % (
                c[-1], int(a[rows].sum()), int((v[rows] == 1).sum()), int(v[rows].sum()), ci(dv)))
        rest = [c for c in d["conds"] if c not in ("baseline", "E_a1", "E_a2", "E_a3") and "forcedcentre" not in c]
        sc = lambda c, s: 2 * d["conds"][c][s]["correct_answered"] - d["conds"][c][s]["n_answered"]
        best = max(rest, key=lambda c: sc(c, "tune"))
        print("    of the %d conditions with C: best on tune %s (tune net %+d, E strength 3 alone %+d) | its test: answered %d, net %+d" % (
            len(rest), best, sc(best, "tune"), sc("E_a3", "tune"), d["conds"][best]["test"]["n_answered"], sc(best, "test")))


if __name__ == "__main__":
    main()
