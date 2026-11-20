# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# TriviaQA / PopQA with the C strength allowed to take the cautious sign (launcher explore_20261120/ns_caut.sh): single
# C toward caution at 0.25 / 0.5 / 1 / 2, and E at 0.5 / 1 / 2 together with C toward caution, written on every question
# (one pass). Alias-match verdicts (no judge call), score = right - wrong.
# Per model and task, every setting is chosen on the tune split (192) and read on test (500):
#   single E at its best strength (0.5 / 1 / 1.5 / 2 / 3 / 4 as far as the files exist);
#   single C toward caution at its best strength;
#   E + C toward caution at its best pair;
# then the combination against no write, against single E (same E strength, and E's own best), and against single C.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/ns_caut_table.py [TASK ...]
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_score as S     # noqa: E402
import alias_match as AM      # noqa: E402

NS = S.NS
ES = ("0.5", "1", "1.5", "2", "3", "4")
KS = ("0.25", "0.5", "1", "2")


def passes(task, model, split):
    """{arm: (states, answered)} for every arm file of the split that is a complete full pass."""
    p = "%s/bank/%s_bank_20261119.jsonl" % (NS, task)
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    b = {r["id"]: r for r in rows if r["split"] == split}
    want = list(b)

    def take(arm):
        f = "%s/gen/%s/%s/%s/%s.json" % (NS, task, model, split, S.tok(arm))
        if not os.path.exists(f):
            return None
        with open(f, encoding="utf-8") as h:
            g = json.load(h)
        k = g.get("contract") or {}
        if not (g.get("completed") and k.get("full_pass") and k.get("split") == split and k.get("model") == model
                and k.get("items_file_sha256") == S.sha_file(p) and [r["id"] for r in g["rows"]] == want):
            raise SystemExit("%s is not a complete %s pass of %s" % (f, split, model))
        st, ans = [], []
        for r in g["rows"]:
            blank = bool(r["rule_abstain"]) or not r["text"].strip()
            st.append(AM.verdict(r["text"], b[r["id"]]["gold"], not blank))
            ans.append(not blank)
        return st, ans

    cache = {}

    def cached(arm):
        if arm not in cache:
            cache[arm] = take(arm)
        return cache[arm]

    return cached, want


def net(x):
    return sum((s == "R") - (s == "W") for s in x[0])


def desc(x):
    return "net %+4d (answers %3d, right %3d, wrong %3d)" % (net(x), sum(x[1]), x[0].count("R"), x[0].count("W"))


def diff(a, b):
    return S.ps.ci([((x == "R") - (x == "W")) - ((y == "R") - (y == "W")) for x, y in zip(a[0], b[0])])


def main():
    M = S.models()
    for task in sys.argv[1:] or list(S.TASKS):
        print("\n# %s (alias match; tune 192, test 500)" % task)
        tot = {}
        for name in S.ORDER:
            m = M[name]
            tt, _ = passes(task, name, "tune")
            tx, ids = passes(task, name, "test")
            e, c = "E%d.w.p" % m["el"], "C%d.w.p" % m["cl"]
            arms_e = [s for s in ES if tt("x:" + e + s) and tx("x:" + e + s)]
            arms_c = [k for k in KS if tt("x:" + c + k) and tx("x:" + c + k)]
            arms_ec = [(s, k) for s in ("0.5", "1", "2") for k in KS if tt("x:%s%s+%s%s" % (e, s, c, k)) and tx("x:%s%s+%s%s" % (e, s, c, k))]
            if not arms_c or not arms_ec:
                print("\n## %s: the cautious-C passes are not there yet" % name)
                continue
            b_t, b_x = tt("x"), tx("x")
            print("\n## %s: no write: tune %s | test %s" % (name, desc(b_t), desc(b_x)))
            for s in arms_e:
                print("    single E %-4s tune %s | test %s" % (s, desc(tt("x:" + e + s)), desc(tx("x:" + e + s))))
            for k in arms_c:
                print("    single C toward caution %-4s tune %s | test %s" % (k, desc(tt("x:" + c + k)), desc(tx("x:" + c + k))))
            for s, k in arms_ec:
                a = "x:%s%s+%s%s" % (e, s, c, k)
                print("    E %-3s + C toward caution %-4s tune %s | test %s" % (s, k, desc(tt(a)), desc(tx(a))))
            se = max(arms_e, key=lambda s: (net(tt("x:" + e + s)), -arms_e.index(s)))
            kc = max(arms_c, key=lambda k: (net(tt("x:" + c + k)), -arms_c.index(k)))
            sk = max(arms_ec, key=lambda p: (net(tt("x:%s%s+%s%s" % (e, p[0], c, p[1]))), -arms_ec.index(p)))
            combo = tx("x:%s%s+%s%s" % (e, sk[0], c, sk[1]))
            e_best, e_same, c_best = tx("x:" + e + se), tx("x:" + e + sk[0]), tx("x:" + c + kc)
            print("  chosen on tune: combination E %s + C toward caution %s | single E %s | single C toward caution %s" % (sk[0], sk[1], se, kc))
            print("  test: combination %s | minus no write %s | minus single E at the same strength %s | minus single E at its own best %s | minus single C %s" % (
                desc(combo), diff(combo, b_x), diff(combo, e_same), diff(combo, e_best), diff(combo, c_best)))
            for key, other in (("combination minus no write", b_x), ("combination minus single E (its own best)", e_best),
                               ("combination minus single C toward caution", c_best)):
                tot.setdefault(key, []).append([((x == "R") - (x == "W")) - ((y == "R") - (y == "W")) for x, y in zip(combo[0], other[0])])
        if tot and all(len(v) == len(S.ORDER) for v in tot.values()):
            print("\n## %s, five models together (questions resampled jointly)" % task)
            for key, per in tot.items():
                t_, lo, hi = S.joint_ci(per)
                print("    %-46s %+d [%+d, %+d], %d of 5 models positive" % (key, t_, lo, hi, sum(sum(v) > 0 for v in per)))


if __name__ == "__main__":
    main()
