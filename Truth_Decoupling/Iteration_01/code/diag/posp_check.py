# Iteration_01 analysis (Claude Opus 5.5, 2026-11-21), read-only, no model, no judge call.
# One extra (model, layer, strength) at the write rule "prompt", outside the budget-1 table: N1 at "prompt" (caps
# chosen on tune) against single E at "prompt", no write, single C at "prompt", the prompt-only baseline, and N1 at the
# training rule at the same layer and strength (diag/main_table_layers.sources). Work dirs as in main_table_prompt.py
# with the given tag (launchers/explore_20261120/pospall.sh with that tag).
# usage (host): python3 -B Iteration_01/diag/posp_check.py MODEL LAYER STRENGTH TAG     e.g. Llama-3.1-8B 16 4 l16s4
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_prompt as MP

ML, DM, MT, S, ps = MP.ML, MP.DM, MP.MT, MP.S, MP.ps


def main():
    model, layer, s, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3].replace(".", "-"), sys.argv[4]
    Dt, Dx, srcs = ML.sources(model)
    src = next((x for x in srcs if x[0] == layer), None)
    if src is None or s not in src[5]:
        raise SystemExit("%s: no training-rule source at layer %d strength %s" % (model, layer, DM.show(s)))
    _, tsp, ttx, xsp, xtx, doses, acc, _ = src
    T, why_t = MP.prompt_split(model, layer, 0, s, qs=list(tsp.qs), tag=tag)
    X, why_x = MP.prompt_split(model, layer, 1, s, qs=list(xsp.qs), tag=tag)
    if T is None or X is None:
        raise SystemExit("not complete: %s %s" % (why_t, why_x))
    tsp_p, ttx_p, tsing, tqs = T
    xsp_p, xtx_p, xsing, qs = X
    if qs != list(xsp.qs) or tqs != list(tsp.qs):
        raise SystemExit("question lists differ between the rules (test %d / %d, tune %d / %d)" % (len(qs), len(xsp.qs), len(tqs), len(tsp.qs)))
    base_acc = srcs[0][6]["0"]
    dev, rep = ML.PM.halves(list(base_acc))
    m_, lo, hi = ML.PM.boot([acc[s][i] - base_acc[i] for i in rep]) if s in acc else (float("nan"),) * 3
    pick = ML.n1_pick(tsp_p, ttx_p, s)
    cpick = ML.n1_pick(tsp, ttx, s)
    print("## %s, layer %d, E strength %s at the prompt rule (tag %s) | MMLU on the report half %+.2f [%+.2f, %+.2f] | N1 on tune: %s (tune %+d); training rule N1 on tune %+d" % (
        model, layer, DM.show(s), tag, m_, lo, hi, ML.ro.name_of(pick), ps.score(S.system(tsp_p, ttx_p, pick)[0], 1), ps.score(S.system(tsp, ttx, cpick)[0], 1)))
    base = S.system(xsp_p, xtx_p, ("base",))
    e, n1 = S.system(xsp_p, xtx_p, ("E", s)), S.system(xsp_p, xtx_p, pick)
    cn1, ce = S.system(xsp, xtx, cpick), S.system(xsp, xtx, ("E", s))
    da = dict(zip(Dx.sp.qs, MT.handed_in(Dx, ("DA",))[0]))
    rows = [("no write", base), ("single E, prompt rule", e), ("N1, prompt rule", n1), ("single E, training rule", ce), ("N1, training rule", cn1)]
    for label, (st, ans, _) in rows:
        print(ML.PM.row(label, st, ans))
    d = lambda a, b: ps.ci([MP.val(x) - MP.val(y) for x, y in zip(a, b)])  # noqa: E731
    print("    N1 (prompt rule) minus: single E (prompt rule) %s | no write %s | prompt only %s | N1 at the training rule %s" % (
        d(n1[0], e[0]), d(n1[0], base[0]), d(n1[0], [da[q] for q in qs]), d(n1[0], cn1[0])))


if __name__ == "__main__":
    main()
