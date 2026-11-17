# Iteration_01 analysis (Claude Fable 5.2, 2026-11-17), read-only, no model, no judge call.
# Levels of the abstention readout log p("I do not know.") on tune: without a write (a0), with E at strength 1 (aE), and the change d = aE - a0,
# by what the model does with E (Pro labels; blanks by the rule detector). For novelty/ours/2026-11-17_silence_and_release.md, section 4.7.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/doubt_levels.py
import os, sys, statistics as S
sys.path.insert(0, "Iteration_01/diag")
import doubt_signal as ds
def med(v):
    return "%6.2f" % S.median(v) if v else "   -  "
for name, run, el, grid, _ in ds.MODELS:
    gw = os.path.join(run, grid)
    base = ds.read_mc(os.path.join(run, "mc", "base_x_tune.json"))[1]
    ew = ds.read_mc(os.path.join(run, "mc", "E_L%d_x-w-p1_tune.json" % el))[1]
    st, bl = ds.load_states(gw, "pro")
    e = "E_L%d_x-w-p1" % el
    qs = sorted(p for p in base if base[p].get("abstain_scored") and ew[p].get("abstain_scored") and (p, e) in st and (p, "baseline_x") in st)
    a0 = {p: float(base[p]["abstain_logp"]) for p in qs}
    aE = {p: float(ew[p]["abstain_logp"]) for p in qs}
    d = {p: aE[p] - a0[p] for p in qs}
    ntok = sorted({len(base[p]["abstain_record"].get("token_ids", base[p]["abstain_record"].get("tokens", []))) for p in qs})
    print("\n## %s (n %d; tokens of the scored string: %s; record keys: %s)" % (name, len(qs), ntok, sorted(base[qs[0]]["abstain_record"])[:12]))
    print("  all questions: median a0 %s | aE %s | d %s | share with d > 0: %.0f%%" % (med(list(a0.values())), med(list(aE.values())), med(list(d.values())), 100 * sum(v > 0 for v in d.values()) / len(qs)))
    ans0 = [p for p in qs if not bl[(p, "baseline_x")]]
    groups = (("answered without a write, blank with E", [p for p in ans0 if bl[(p, e)]]),
              ("answered with E, wrong", [p for p in qs if not bl[(p, e)] and st[(p, e)] == "W"]),
              ("answered with E, right", [p for p in qs if not bl[(p, e)] and st[(p, e)] == "R"]),
              ("blank without a write", [p for p in qs if bl[(p, "baseline_x")]]))
    for label, g in groups:
        print("  %-42s n %3d | median a0 %s | aE %s | d %s" % (label, len(g), med([a0[p] for p in g]), med([aE[p] for p in g]), med([d[p] for p in g])))
    blankE = [aE[p] for p in qs if bl[(p, e)]]
    ansE = [aE[p] for p in qs if not bl[(p, e)]]
    if blankE and ansE:
        print("  aE of the blank ones: min %.2f, median %s | aE of the answered ones: median %s, max %.2f, 90th pct %.2f" % (
            min(blankE), med(blankE), med(ansE), max(ansE), sorted(ansE)[int(0.9 * (len(ansE) - 1))]))
