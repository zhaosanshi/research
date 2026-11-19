# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# MMLU by option probability on the full test split (scripts/capability/cap_mcq_sweep.py), reported separately for the
# items that were NOT in the 1,000-item list the dose was looked at on (the confirmation), for those 1,000, and for all.
# Per model and arm: accuracy, change vs no write in points [95%, items resampled], picks changed; then the label of
# novelty/ours/2026-11-19_capability_preservation_design.md section 6 for the accuracy part: "kept" only if the interval
# excludes a drop of more than 1 point, "damage" if the interval lies below -1, otherwise "cannot exclude damage".
# usage: python3 -I Iteration_01/diag/cap_mcq_fresh.py SEEN_IDS_JSON FULL_JSON...
import json
import random
import sys


def ci(v, seed=20261119, n_boot=2000):
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(v) / n, bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]


def label(lo, hi):
    if lo > -0.01:
        return "kept"
    if hi < -0.01:
        return "damage"
    return "cannot exclude damage"


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        seen = set(json.load(f)["ids"])
    print("MMLU by option probability, full test split; change vs no write in points [95%]; label by the 1-point rule")
    for p in sys.argv[2:]:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        base = {r["id"]: r for r in d["arms"]["x"]["rows"]}
        for part, keep in (("fresh items (not in the 1,000)", lambda i: i not in seen), ("the 1,000", lambda i: i in seen), ("all", lambda i: True)):
            ids = [i for i in base if keep(i)]
            acc0 = sum(base[i]["pick"] == base[i]["gold"] for i in ids) / float(len(ids))
            print("\n== %s, %s: %d items, no write %.3f" % (d["model"], part, len(ids), acc0))
            for arm in d["arms"]:
                if arm == "x":
                    continue
                a = {r["id"]: r for r in d["arms"][arm]["rows"]}
                diff = [(a[i]["pick"] == a[i]["gold"]) - (base[i]["pick"] == base[i]["gold"]) for i in ids]
                m_, lo, hi = ci(diff)
                moved = sum(a[i]["pick"] != base[i]["pick"] for i in ids)
                print("    %-16s %.3f (%+.2f [%+.2f, %+.2f]) | picks changed %5d (%.1f%%) | %s" % (
                    arm.split(":", 1)[1], acc0 + m_, 100 * m_, 100 * lo, 100 * hi, moved, 100.0 * moved / len(ids), label(lo, hi)))


if __name__ == "__main__":
    main()
