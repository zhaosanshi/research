# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# The main table with ONE E STRENGTH PER MODEL (the lead, 2026-11-21 00:15: what is fixed is the method, not the number;
# each model gets the strength at which MMLU barely moves).
# Rule for the strength (stated before looking at any TruthfulQA test number of this table): split the 14,042 MMLU test
# items into a selection half (the 1,000 diagnostic items + a seeded random draw of the others, 7,021 in all) and a
# report half (7,021, never used for anything); for a budget of B points, a model's strength is the LARGEST strength on
# its grid whose accuracy change on the selection half is not below -B. The MMLU change is then reported on the other
# half. The grid of a model = the strengths for which it has both the N1 arms on TruthfulQA (tune and test, judged) and
# full-set MMLU answers. Budgets 1, 1.5 and 2 points are printed; which one to use is not decided here.
# TruthfulQA (test, DeepSeek Pro, score = right - wrong) at that strength: no write, single C (strength chosen on tune),
# single E, N1 (caps chosen on tune at that strength), the strong form at the same strength (must-answer + E, the N most
# confident answers, N = N1's answers), and the prompt-only baseline; differences with paired intervals. Several models
# together: per question the differences are summed and questions are resampled.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/per_model_table.py [BUDGET ...]
import json
import os
import random
import sys

sys.path.insert(0, "Iteration_01/diag")
import dose_mid as DM
import strong_mid as SM

MT, S = DM.MT, DM.S
ps, ro = MT.ps, MT.ro
CAPS = MT.CAPS
NG = {"Llama-3.1-8B": "llama3.1-8b", "Qwen3.5-9B": "qwen3.5-9b", "Qwen3.8-27B": "qwen3.8-27b", "Gemma-3-12B": "gemma3-12b", "Gemma-3-27B": "gemma3-27b"}
CAP = "Iteration_01/capability"
SEEN = CAP + "/sets/mmlu_1000_20261119.json"
SPLIT_SEED = 20261121


def val(x):
    return (x == "R") - (x == "W")


def mmlu(model, el):
    """{strength: {item id: correct}} with "0" for no write, from every full-set file of the model."""
    out, base = {}, None
    for d in ("full", "full_mid", "full_high", "full_low"):
        p = "%s/%s/%s.json" % (CAP, d, NG[model])
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        if j["n"] != 14042:
            raise SystemExit("%s is not a full-set file" % p)
        x = {r["id"]: r["pick"] == r["gold"] for r in j["arms"]["x"]["rows"]}
        if base is None:
            base = x
        elif x != base:
            raise SystemExit("%s: the no-write picks differ from the first file's" % p)
        for arm, v in j["arms"].items():
            pre = "x:E%d.w.p" % el
            if arm.startswith(pre):
                out[arm[len(pre):].replace(".", "-")] = {r["id"]: r["pick"] == r["gold"] for r in v["rows"]}
    out["0"] = base
    return out


def halves(ids):
    with open(SEEN, encoding="utf-8") as f:
        seen = set(json.load(f)["ids"])
    rest = sorted(i for i in ids if i not in seen)
    random.Random(SPLIT_SEED).shuffle(rest)
    k = len(ids) // 2 - len(seen)
    return sorted(seen) + rest[:k], rest[k:]


def boot(v, n_boot=2000, seed=SPLIT_SEED):
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return 100.0 * sum(v) / n, 100.0 * bs[int(0.025 * n_boot)], 100.0 * bs[int(0.975 * n_boot) - 1]


def row(label, st, ans):
    return "    %-44s answers %3d | right %3d wrong %3d | net %+4d" % (label, sum(ans), st.count("R"), st.count("W"), sum(map(val, st)))


def model_block(model, budgets, tot):
    Dt, tsp, ttx, tdoses, _, _ = DM.merged_all(model, 0)
    Dx, xsp, xtx, doses, _, _ = DM.merged_all(model, 1)
    qs = xsp.qs
    el = Dx.m["e"]
    acc = mmlu(model, el)
    dev, rep = halves(list(acc["0"]))
    grid = [s for s in doses if s in acc]
    chg = {s: 100.0 * sum(acc[s][i] - acc["0"][i] for i in dev) / len(dev) for s in grid}
    print("\n## %s: grid %s | MMLU change on the selection half: %s" % (
        model, " ".join(DM.show(s) for s in grid), ", ".join("%s: %+.2f" % (DM.show(s), chg[s]) for s in grid)))
    run = Dx.m["ec"][1].split("/eval_extra/")[0]
    must_new, conf_new, _ = SM.new_must(run, xsp, doses)
    da0 = MT.handed_in(Dx, ("DA",))
    da = dict(zip(Dx.sp.qs, da0[0]))
    da_s = [da[q] for q in qs]
    base = S.system(xsp, xtx, ("base",))
    cks = [k for k in ps.LAD if k in Dt.sp.cks and k in Dx.sp.cks]
    ck = max(cks, key=lambda k: (ps.score(ro.states(Dt.sp, Dt.tx, ("C", k)), 1), -cks.index(k))) if cks else None
    c0 = MT.plain(Dx, ("C", ck)) if ck else None
    cby = dict(zip(Dx.sp.qs, zip(c0[0], c0[1]))) if c0 else None
    for B in budgets:
        ok = [s for s in grid if chg[s] >= -B]
        if not ok:
            print("  budget %g: no strength on the grid stays within it (smallest strength %s: %+.2f)" % (B, DM.show(grid[0]), chg[grid[0]]))
            continue
        s = max(ok, key=lambda x: float(DM.show(x)))
        m_, lo, hi = boot([acc[s][i] - acc["0"][i] for i in rep])
        cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        pick = max(cands, key=lambda x: (ps.score(S.system(tsp, ttx, x)[0], 1), -cands.index(x)))
        e, n1 = S.system(xsp, xtx, ("E", s)), S.system(xsp, xtx, pick)
        print("  budget %g -> E strength %s | MMLU on the report half %+.2f [%+.2f, %+.2f] | N1 on tune: %s" % (B, DM.show(s), m_, lo, hi, ro.name_of(pick)))
        print(row("no write", base[0], base[1]))
        if cby:
            cs, ca = [cby[q][0] for q in qs], [cby[q][1] for q in qs]
            print(row("single C (strength %s, chosen on tune)" % ps.NAME[ck], cs, ca))
        print(row("single E", e[0], e[1]))
        print(row("N1", n1[0], n1[1]))
        N = sum(n1[1])
        strong = None
        if s in ("1", "2"):
            de0 = MT.must(Dx, Dx.sp.dE(s))
            by = dict(zip(Dx.sp.qs, zip(de0[0], de0[1])))
            de = ([by[q][0] for q in qs], [by[q][1] for q in qs])
            rows = Dx.rows("conf_E%s" % s, Dx.m["d"][1], Dx.sp.dE(s))
        else:
            de, rows = must_new.get(s), conf_new.get(s)
        if de is not None and rows is not None:
            order = sorted([i for i, a in enumerate(de[1]) if a], key=lambda i: (-rows[qs[i]]["mean_logp"], i))
            if len(order) >= N:
                strong = MT.keep(de[0], order, N)
                print(row("strong form: must-answer + E, top %d" % N, strong[0], strong[1]))
        print("    %-44s net %+4d" % ("prompt only (must-answer + sentence)", sum(map(val, da_s))))
        diffs = [("no write", base[0]), ("single E", e[0]), ("prompt only", da_s)]
        if cby:
            diffs.insert(1, ("single C", cs))
        if strong:
            diffs.insert(len(diffs) - 1, ("the strong form", strong[0]))
        print("    N1 minus: " + " | ".join("%s %s" % (name, ps.ci([val(x) - val(y) for x, y in zip(n1[0], other)])) for name, other in diffs))
        for name, other in diffs:
            t = tot.setdefault((B, name), [0, {}])
            t[0] += 1
            for q, x, y in zip(qs, n1[0], other):
                t[1][q] = t[1].get(q, 0) + val(x) - val(y)
        tot.setdefault((B, "strengths"), []).append("%s %s" % (model, DM.show(s)))


def main():
    budgets = [float(a) for a in sys.argv[1:]] or [1.0, 1.5, 2.0]
    tot = {}
    for model in ro.FIVE:
        model_block(model, budgets, tot)
    for title, names in (("the five models together", ro.FIVE),):
        print("\n## %s, each model at its own strength (per question the differences are summed; questions resampled)" % title)
        for B in budgets:
            got = tot.get((B, "strengths"), [])
            print("  budget %g: %s" % (B, "; ".join(got) if len(got) == len(names) else "not every model has a strength within the budget (%s)" % "; ".join(got)))
            if len(got) != len(names):
                continue
            for (b, name), v in tot.items():
                if b == B and name != "strengths" and v[0] == len(names):
                    print("    N1 minus %-16s %s" % (name, ps.ci([v[1][q] for q in sorted(v[1])])))


if __name__ == "__main__":
    main()
