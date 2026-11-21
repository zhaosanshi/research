# Iteration_01 analysis (Claude Opus 5.5, 2026-11-21), read-only, no model, no judge call.
# The per-model main table (diag/main_table_layers.py) with every write at the core rule "prompt" (question tokens and
# assistant header; no decode step written) instead of the training rule "content" (every generated token written as
# well). Exploration: at the training rule the main-table settings cost 19 / 14 GSM8K points on Llama-3.1-8B /
# Gemma-3-12B; at "prompt" Llama's cost is gone.
# Rule, the main table's, unchanged: per layer the largest strength whose MMLU change on the selection half is not
# below -B (the MMLU read is identical under the two rules, so these candidates are the main table's); across layers
# the one whose N1 has the higher TUNE score (here: N1 at "prompt"; ties to the earlier layer = the one chosen for
# TruthfulQA). N1's caps and the strength of single C are chosen on tune, as in the main table.
# Inputs per (model, layer): the usual no-write cells; single E at "prompt" (work dir judge_<split>_posprompt<tag>_<split>),
# E + C at "prompt" (judge_<split>_posprompt2<tag>_<split>, conditions with the suffix "-at-prompt"), single C at
# "prompt" (judge_<split>_pospromptc_<split>, one per run), tag = l<layer>; launchers/explore_20261120/pospall.sh.
# Not here yet: the strong comparator at "prompt" (it needs the must-answer answers with E at "prompt" and their
# confidence). For reference the training-rule N1 of the main table is printed at the same layer and strength.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/main_table_prompt.py [BUDGET ...]
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_layers as ML
import prompt_sweep_table as PW

DM, MT, S, PM = ML.DM, ML.MT, ML.S, ML.PM
ps, ro = ML.ps, ML.ro
fr = MT.fr
SUF = "-at-prompt"


def val(x):
    return (x == "R") - (x == "W")


def work(run, split, name):
    return "%s/eval_extra/judge_%s_%s_%s" % (run, split, name, split)


def has(w):
    return os.path.exists(w + "/judge2/single_pro.jsonl")


def usual(model, i):
    D = MT.Data(model, i)
    run = D.m["ec"][i].split("/eval_extra/")[0]
    return D, run, fr.raw_texts(D.m["ec"][i])


def prompt_split(model, layer, i, s):
    """(Split at the prompt rule for E at this layer and strength, texts, single-C states by strength, questions) or
    (None, why)."""
    D, run, utx = usual(model, i)
    split = ("tune", "test")[i]
    tag = "l%d" % layer
    w1, w2, wc = work(run, split, "posprompt" + tag), work(run, split, "posprompt2" + tag), work(run, split, "pospromptc")
    for w in (w1, w2, wc):
        if not has(w):
            return None, "no Pro verdicts yet in %s" % os.path.basename(w)
    st, bl, tx = {}, {}, {}
    for q in D.sp.qs:
        k = (q, "baseline_x")
        st[k], bl[k], tx[k] = D.sp.st[k], D.sp.blank[k], utx[k]
    for w, two in ((w1, False), (w2, True), (wc, False)):
        st1, bl1 = ps.load(w, "pro")
        t1 = fr.raw_texts(w)
        for (q, c), v in st1.items():
            if c == "baseline_x":
                if (q, c) in st and (v != st[(q, c)] or bl1[(q, c)] != bl[(q, c)] or t1.get((q, c)) != tx[(q, c)]):
                    raise SystemExit("%s: the no-write cell of %s differs from the usual one (%s)" % (model, os.path.basename(w), q))
                continue
            c0 = c
            if two:
                if not c.endswith(SUF):
                    raise SystemExit("%s: condition %s in %s lacks %s" % (model, c, os.path.basename(w), SUF))
                c = c[:-len(SUF)]
            elif c.endswith(SUF):
                raise SystemExit("%s: a two-write condition %s in a single-write work dir" % (model, c))
            k = (q, c)
            if k in st:
                raise SystemExit("%s: condition %s appears in two work dirs" % (model, c))
            st[k], bl[k], tx[k] = v, bl1[(q, c0)], t1[(q, c0)]
    sh = S.shell(layer, D.m["c"])
    need = ["baseline_x", sh.cE(s)] + [sh.cEC(s, k) for k in ps.LAD]
    sp = S.NewSplit(layer, D.m["c"], st, bl, need, qs=D.sp.qs)
    singles = {}
    for k in ps.LAD:
        c = "C_L%d_x-w-m%s" % (D.m["c"], k)
        if all(st.get((q, c), "?") != "?" for q in sp.qs):
            singles[k] = ([st[(q, c)] for q in sp.qs], [not bl[(q, c)] for q in sp.qs])
    return (sp, fr.Texts(tx, tx), singles, list(sp.qs)), None


def model_block(model, B, tot):
    Dt, Dx, srcs = ML.sources(model)
    base_acc = srcs[0][6]["0"]
    dev, rep = PM.halves(list(base_acc))
    finalists, notes = [], []
    for layer, tsp, ttx, xsp, xtx, doses, acc, strong in srcs:
        ok = [s for s in doses if s in acc and 100.0 * sum(acc[s][i] - base_acc[i] for i in dev) / len(dev) >= -B]
        if not ok:
            continue
        s = max(ok, key=lambda x: float(DM.show(x)))
        T, why_t = prompt_split(model, layer, 0, s)
        X, why_x = prompt_split(model, layer, 1, s)
        if T is None or X is None:
            notes.append("layer %d / %s: %s" % (layer, DM.show(s), why_t or why_x))
            continue
        tsp_p, ttx_p, tsing, tqs = T
        pick = ML.n1_pick(tsp_p, ttx_p, s)
        finalists.append(dict(layer=layer, s=s, pick=pick, tune=ps.score(S.system(tsp_p, ttx_p, pick)[0], 1), T=T, X=X,
                              acc=acc, content=(tsp, ttx, xsp, xtx)))
    print("\n## %s, budget %g: candidates at the prompt rule: %s%s" % (
        model, B, "; ".join("layer %d / %s (N1 tune %+d)" % (c["layer"], DM.show(c["s"]), c["tune"]) for c in finalists) or "none",
        "" if not notes else " | missing: " + "; ".join(notes)))
    if not finalists or notes:
        return
    best = max(finalists, key=lambda c: (c["tune"], -finalists.index(c)))
    s, acc = best["s"], best["acc"]
    tsp_p, ttx_p, tsing, tqs = best["T"]
    xsp_p, xtx_p, xsing, qs = best["X"]
    m_, lo, hi = PM.boot([acc[s][i] - base_acc[i] for i in rep])
    print("  -> layer %d, E strength %s (N1 on tune: %s) | MMLU on the report half %+.2f [%+.2f, %+.2f] (the same as at the training rule)" % (
        best["layer"], DM.show(s), ro.name_of(best["pick"]), m_, lo, hi))
    base = S.system(xsp_p, xtx_p, ("base",))
    e, n1 = S.system(xsp_p, xtx_p, ("E", s)), S.system(xsp_p, xtx_p, best["pick"])
    rows = [("no write", base[0], base[1])]
    diffs = [("no write", base[0])]
    if tsing and all(k in xsing for k in tsing):
        ks = [k for k in ps.LAD if k in tsing]
        ck = max(ks, key=lambda k: (ps.score(tsing[k][0], 1), -ks.index(k)))
        cst, cans = xsing[ck]
        rows.append(("single C (strength %s, chosen on tune)" % ps.NAME[ck], cst, cans))
        diffs.append(("single C", cst))
    rows.append(("single E", e[0], e[1]))
    rows.append(("N1", n1[0], n1[1]))
    diffs.append(("single E", e[0]))
    for label, st, ans in rows:
        print(PM.row(label + ", prompt rule" if label != "no write" else label, st, ans))
    da0 = MT.handed_in(Dx, ("DA",))
    da = dict(zip(Dx.sp.qs, da0[0]))
    da_s = [da[q] for q in qs]
    print("    %-44s net %+4d" % ("prompt only (must-answer + sentence)", sum(map(val, da_s))))
    diffs.append(("prompt only", da_s))
    run = Dx.m["ec"][1].split("/eval_extra/")[0]
    ptune, ptest = PW.prompts_of(run, "tune", Dt.sp.qs), PW.prompts_of(run, "test", Dx.sp.qs)
    if ptune and ptest:
        order = sorted(ptune, key=lambda p: (-sum(map(val, ptune[p][0])), p != "conf_E2", p))
        if order[0] in ptest:
            by = dict(zip(Dx.sp.qs, ptest[order[0]][0]))
            bp = [by[q] for q in qs]
            print("    %-44s net %+4d" % ("best prompt on tune: %s" % order[0], sum(map(val, bp))))
            diffs.append(("the best prompt on tune", bp))
    tsp, ttx, xsp, xtx = best["content"]
    cpick = ML.n1_pick(tsp, ttx, s)
    cn1 = S.system(xsp, xtx, cpick)
    cby = dict(zip(xsp.qs, cn1[0]))
    if all(q in cby for q in qs):
        cs = [cby[q] for q in qs]
        print("    %-44s answers %3d | net %+4d (reference: the main table's row at this layer and strength)" % (
            "N1, training rule", sum(cn1[1]), sum(map(val, cn1[0]))))
        diffs.append(("N1 at the training rule", cs))
    print("    N1 (prompt rule) minus: " + " | ".join("%s %s" % (name, ps.ci([val(x) - val(y) for x, y in zip(n1[0], other)])) for name, other in diffs))
    for name, other in diffs:
        t = tot.setdefault((B, name), [0, {}])
        t[0] += 1
        for q, x, y in zip(qs, n1[0], other):
            t[1][q] = t[1].get(q, 0) + val(x) - val(y)
    tot.setdefault((B, "picks"), []).append("%s L%d %s" % (model, best["layer"], DM.show(s)))
    tot.setdefault((B, "questions"), []).append(set(qs))


def main():
    budgets = [float(a) for a in sys.argv[1:]] or [1.0]
    tot = {}
    for B in budgets:
        for model in ro.FIVE:
            model_block(model, B, tot)
        got = tot.get((B, "picks"), [])
        print("\n## the five models together at the prompt rule, budget %g: %s" % (B, "; ".join(got)))
        if len(got) != len(ro.FIVE):
            print("    only %d of %d models are complete" % (len(got), len(ro.FIVE)))
            continue
        common = sorted(set.intersection(*tot[(B, "questions")]))
        print("    on the %d questions that all five models have:" % len(common))
        for (b, name), v in tot.items():
            if b == B and name not in ("picks", "questions"):
                if v[0] == len(ro.FIVE):
                    print("    N1 minus %-26s %s" % (name, ps.ci([v[1][q] for q in common])))
                else:
                    print("    N1 minus %-26s only %d of %d models have it" % (name, v[0], len(ro.FIVE)))


if __name__ == "__main__":
    main()
