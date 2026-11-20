# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# The per-model main table (diag/per_model_table.py) with the E LAYER allowed to differ from the one chosen for
# TruthfulQA alone, where a second layer has been run (the registry diag/alt_layer.py: Llama-3.1-8B layer 16 next to
# 11, Gemma-3-27B layer 28 next to 26; both found on the MMLU layer map, where they cost far less).
# Rule, stated before reading any TruthfulQA test number of this table and using tune + the MMLU selection half only:
#   candidates of a model = every (layer, strength) that has the N1 arms judged on tune and test and full-set MMLU answers;
#   1. within a layer, as in per_model_table.py: for a budget of B points the strength is the LARGEST one whose MMLU
#      accuracy change on the selection half (the 7,021 items of per_model_table.halves) is not below -B;
#   2. across layers: the layer whose N1 at that strength has the highest TUNE score (N1's caps chosen on tune; ties
#      to the earlier layer = the one chosen for TruthfulQA).
# With one layer per model this is per_model_table.py exactly, so the rows of the other four models do not change.
# The tune score of every candidate is printed, so a reader can see what "best tune score within the budget" (a rule
# that also lets the tune split choose the strength) would have picked instead.
# Reported at the pick, as in per_model_table.py: MMLU change on the report half; TruthfulQA test (DeepSeek Pro): no
# write, single C (strength chosen on tune), single E, N1, the strong form at the same layer and strength (must-answer
# + E, the N most confident answers, N = N1's answers) where its files exist, the prompt-only baseline; differences with
# paired intervals; models together with the per-question differences summed.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/main_table_layers.py [BUDGET ...]
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import dose_mid as DM
import strong_mid as SM
import per_model_table as PM
import alt_layer as AL

MT, S = DM.MT, DM.S
ps, ro = MT.ps, MT.ro
CAPS = MT.CAPS


def val(x):
    return (x == "R") - (x == "W")


def n1_pick(tsp, ttx, s):
    cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
    return max(cands, key=lambda x: (ps.score(S.system(tsp, ttx, x)[0], 1), -cands.index(x)))


def sources(model):
    """[(layer, tune split, tune texts, test split, test texts, strengths, {strength: MMLU correctness}, strong loader)]."""
    out = []
    Dt, tsp, ttx, tdoses, _, _ = DM.merged_all(model, 0)
    Dx, xsp, xtx, doses, _, _ = DM.merged_all(model, 1)
    el = Dx.m["e"]
    run = Dx.m["ec"][1].split("/eval_extra/")[0]

    def strong11(s, qs):
        if s in ("1", "2"):
            de0 = MT.must(Dx, Dx.sp.dE(s))
            by = dict(zip(Dx.sp.qs, zip(de0[0], de0[1])))
            return ([by[q][0] for q in qs], [by[q][1] for q in qs]), Dx.rows("conf_E%s" % s, Dx.m["d"][1], Dx.sp.dE(s))
        must, conf, _ = SM.new_must(run, xsp, doses)
        return must.get(s), conf.get(s)

    out.append((el, tsp, ttx, xsp, xtx, [s for s in doses if s in tdoses], PM.mmlu(model, el), strong11))
    for alt in AL.ALTS:
        if alt.model != model:
            continue
        t, x, acc = alt.load(0), alt.load(1), alt.mmlu()
        if not (t and x and acc):
            continue
        _, tsp2, ttx2 = t
        _, xsp2, xtx2 = x

        def strong2(s, qs, alt=alt, xsp2=xsp2):
            must, conf, _ = alt.strong(run, xsp2)
            return must.get(s), conf.get(s)

        out.append((alt.layer, tsp2, ttx2, xsp2, xtx2, list(alt.doses), acc, strong2))
    return Dt, Dx, out


def model_block(model, budgets, tot):
    Dt, Dx, srcs = sources(model)
    base_acc = srcs[0][6]["0"]
    dev, rep = PM.halves(list(base_acc))
    for layer, tsp, ttx, xsp, xtx, doses, acc, strong in srcs[1:]:
        if tsp.qs != srcs[0][1].qs or xsp.qs != srcs[0][3].qs:
            raise SystemExit("%s layer %d: its tune or test question list differs from layer %d's (%d / %d against %d / %d); the layers would be compared on different questions" % (
                model, layer, srcs[0][0], len(tsp.qs), len(xsp.qs), len(srcs[0][1].qs), len(srcs[0][3].qs)))
    cands = []
    for layer, tsp, ttx, xsp, xtx, doses, acc, strong in srcs:
        if acc["0"] != base_acc:
            raise SystemExit("%s layer %d: the no-write MMLU picks differ from the first source" % (model, layer))
        for s in doses:
            if s not in acc:
                continue
            pick = n1_pick(tsp, ttx, s)
            cands.append(dict(layer=layer, s=s, pick=pick, tune=ps.score(S.system(tsp, ttx, pick)[0], 1),
                              chg=100.0 * sum(acc[s][i] - base_acc[i] for i in dev) / len(dev), src=(tsp, ttx, xsp, xtx, acc, strong)))
    print("\n## %s: candidates (layer, strength: MMLU change on the selection half, N1 tune score)" % model)
    for layer in sorted({c["layer"] for c in cands}):
        print("    layer %d: %s" % (layer, " | ".join("%s: %+.2f, %+d" % (DM.show(c["s"]), c["chg"], c["tune"]) for c in cands if c["layer"] == layer)))
    cks = [k for k in ps.LAD if k in Dt.sp.cks and k in Dx.sp.cks]
    ck = max(cks, key=lambda k: (ps.score(ro.states(Dt.sp, Dt.tx, ("C", k)), 1), -cks.index(k))) if cks else None
    c0 = MT.plain(Dx, ("C", ck)) if ck else None
    cby = dict(zip(Dx.sp.qs, zip(c0[0], c0[1]))) if c0 else None
    da0 = MT.handed_in(Dx, ("DA",))
    da = dict(zip(Dx.sp.qs, da0[0]))
    for B in budgets:
        ok = [c for c in cands if c["chg"] >= -B]
        if not ok:
            print("  budget %g: no candidate stays within it" % B)
            continue
        layers = list(dict.fromkeys(c["layer"] for c in cands))
        finalists = [max((c for c in ok if c["layer"] == l), key=lambda c: float(DM.show(c["s"]))) for l in layers if any(c["layer"] == l for c in ok)]
        best = max(finalists, key=lambda c: (c["tune"], -finalists.index(c)))
        alt = max(ok, key=lambda c: (c["tune"], -cands.index(c)))
        tsp, ttx, xsp, xtx, acc, strong = best["src"]
        qs, s = xsp.qs, best["s"]
        m_, lo, hi = PM.boot([acc[s][i] - base_acc[i] for i in rep])
        print("  budget %g -> layer %d, E strength %s (N1 tune score %+d; N1 on tune: %s) | MMLU on the report half %+.2f [%+.2f, %+.2f] | per layer: %s | best tune score within the budget would be layer %d, %s (%+d)" % (
            B, best["layer"], DM.show(s), best["tune"], ro.name_of(best["pick"]), m_, lo, hi,
            "; ".join("layer %d -> %s (tune %+d)" % (c["layer"], DM.show(c["s"]), c["tune"]) for c in finalists),
            alt["layer"], DM.show(alt["s"]), alt["tune"]))
        base = S.system(xsp, xtx, ("base",))
        e, n1 = S.system(xsp, xtx, ("E", s)), S.system(xsp, xtx, best["pick"])
        da_s = [da[q] for q in qs]
        print(PM.row("no write", base[0], base[1]))
        diffs = [("no write", base[0])]
        if cby:
            cs, ca = [cby[q][0] for q in qs], [cby[q][1] for q in qs]
            print(PM.row("single C (strength %s, chosen on tune)" % ps.NAME[ck], cs, ca))
            diffs.append(("single C", cs))
        print(PM.row("single E", e[0], e[1]))
        print(PM.row("N1", n1[0], n1[1]))
        diffs.append(("single E", e[0]))
        N = sum(n1[1])
        de, rows = strong(s, qs)
        if de is not None and rows is not None:
            order = sorted([i for i, a in enumerate(de[1]) if a], key=lambda i: (-rows[qs[i]]["mean_logp"], i))
            if len(order) >= N:
                st = MT.keep(de[0], order, N)
                print(PM.row("strong form: must-answer + E, top %d" % N, st[0], st[1]))
                diffs.append(("the strong form", st[0]))
        print("    %-44s net %+4d" % ("prompt only (must-answer + sentence)", sum(map(val, da_s))))
        diffs.append(("prompt only", da_s))
        print("    N1 minus: " + " | ".join("%s %s" % (name, ps.ci([val(x) - val(y) for x, y in zip(n1[0], other)])) for name, other in diffs))
        for name, other in diffs:
            t = tot.setdefault((B, name), [0, {}])
            t[0] += 1
            for q, x, y in zip(qs, n1[0], other):
                t[1][q] = t[1].get(q, 0) + val(x) - val(y)
        tot.setdefault((B, "picks"), []).append("%s L%d %s" % (model, best["layer"], DM.show(s)))
        tot.setdefault((B, "questions"), []).append(set(qs))


def main():
    budgets = [float(a) for a in sys.argv[1:]] or [1.0, 1.5, 2.0]
    tot = {}
    for model in ro.FIVE:
        model_block(model, budgets, tot)
    print("\n## the five models together, each at its own layer and strength (per question the differences are summed; questions resampled)")
    for B in budgets:
        got = tot.get((B, "picks"), [])
        print("  budget %g: %s" % (B, "; ".join(got) if len(got) == len(ro.FIVE) else "not every model has a candidate within the budget (%s)" % "; ".join(got)))
        if len(got) != len(ro.FIVE):
            continue
        common = sorted(set.intersection(*tot[(B, "questions")]))
        print("    on the %d questions that all five models have (of %d in any):" % (len(common), len(set.union(*tot[(B, "questions")]))))
        for (b, name), v in tot.items():
            if b == B and name not in ("picks", "questions"):
                if v[0] == len(ro.FIVE):
                    print("    N1 minus %-16s %s" % (name, ps.ci([v[1][q] for q in common])))
                else:
                    print("    N1 minus %-16s only %d of %d models have it" % (name, v[0], len(ro.FIVE)))


if __name__ == "__main__":
    main()
