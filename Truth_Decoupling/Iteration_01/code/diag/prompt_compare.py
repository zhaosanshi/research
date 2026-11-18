# Iteration_01 analysis (Claude Fable 5.2, 2026-11-18), read-only, no model, no judge call.
# 赵磊 2026-11-18: is "stronger-caution prompt, then the new scheme" better than "plain prompt, E alone" (Gemma-4), and better than
# the plain prompt with the present method / the new scheme (the other models)? If so the base prompt should be the stronger one.
# For every model that has BOTH a plain run and a stronger-caution run (the plain prompt followed by the sentence U1; nothing
# else differs), on the questions the two runs share, paired by question. Each run chooses its own settings on its own tune
# split (DeepSeek Pro); scores are right - wrong.
#   plain run:    no write | single E | method | N1 | must-answer prompt + E (where the must-answer test split is judged)
#   caution run:  no write | single E | N1
# Main: tune (192) and test (the Dr's 98), DeepSeek Pro. Auxiliary: the full test split with AllenAI, same settings.
# A model without a stronger-caution run is listed as such. usage (host, from /data/sanshizhao/Truth_Decoupling):
#   python3 -B Iteration_01/diag/prompt_compare.py
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import two_caps as tc
import release_origin as ro

CAPS = ro.CAPS
I1 = "Iteration_01/results"
# the stronger-caution runs: name of the plain entry -> entry of the caution run (as policy_select2.MODELS) and its AllenAI test dir
CAUTION = {
    "Gemma-4-12B": (cf.MODELS["Gemma-4-12B stronger caution"], tc.AUX["Gemma-4-12B stronger caution"]),
}
def entry(root, tag, e, c, tune, test, dtune, dtest, aux):
    d = (root + "/" + dtune, root + "/" + dtest) if os.path.exists(root + "/" + dtest + "/judge2/single_pro.jsonl") else None
    return (dict(ec=(root + "/eval_extra/" + tune, root + "/eval_extra/" + test), d=d, judge=("pro", "pro"), e=e, c=c, order="EC"), root + "/eval_extra/" + aux)
# runs started 2026-11-18 (filled in when their judged dirs exist; layers are read from the dir names at that time)
PENDING = {"Gemma-4-31B": I1 + "/ff_xu1/gemma4/31b/e44/gemma4-31b-dm", "Gemma-3-12B": I1 + "/ff_xu1/gemma3/12b/e44/gemma3-12b",
           "Gemma-3-27B": I1 + "/ff_xu1/gemma3/27b/e44/gemma3-27b"}


def families(sp):
    return {"no write": [("base",)], "single E": [("E", "1"), ("E", "2")],
            "method": [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in ps.LAD],
            "N1": [("E", s) for s in "12"] + [("N1", s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]}


def picks(m):
    tune = ps.Split(m, 0)
    ttx = ro.texts(m["ec"][0])
    fam = families(tune)
    pick = {f: max(c, key=lambda x: (ps.score(ro.states(tune, ttx, x), 1), -c.index(x))) for f, c in fam.items()}
    if tune.dst is not None:
        fd = [("DE", "1"), ("DE", "2")]
        pick["must-answer + E"] = max(fd, key=lambda x: (ps.score(tune.states(x), 1), -fd.index(x)))
    return tune, ttx, pick


def by_id(sp, tx, pick):
    out = {}
    for f, cfg in pick.items():
        if cfg[0] == "DE" and sp.dst is None:
            continue
        st = sp.states(cfg) if cfg[0] == "DE" else ro.states(sp, tx, cfg)
        out[f] = dict(zip(sp.qs, st))
    return out


def show(label, a, b, names):
    common = sorted(set(next(iter(a.values()))) & set(next(iter(b.values()))))
    sc = lambda d: ps.score([d[p] for p in common], 1)
    print("  %s, %d shared questions" % (label, len(common)))
    print("     plain run:   " + " | ".join("%s %d" % (f, sc(a[f])) for f in a))
    print("     caution run: " + " | ".join("%s %d" % (f, sc(b[f])) for f in b))
    for fb, fa in names:
        if fb in b and fa in a:
            d = [((b[fb][p] == "R") - (b[fb][p] == "W")) - ((a[fa][p] == "R") - (a[fa][p] == "W")) for p in common]
            print("     caution %-8s minus plain %-16s %s" % (fb, fa, ps.ci(d)))


def main():
    names = (("N1", "single E"), ("N1", "N1"), ("N1", "method"), ("single E", "single E"), ("no write", "no write"), ("N1", "must-answer + E"))
    print("# Part 0: within the plain run, the must-answer prompt against the plain prompt (test split, DeepSeek Pro, paired)")
    for name, m in cf.MODELS.items():
        if "stronger caution" in name or not m.get("d"):
            continue
        tune_a, ttx_a, pick_a = picks(m)
        test_a = ps.Split(m, 1)
        a = by_id(test_a, ro.texts(m["ec"][1]), pick_a)
        qs = test_a.qs
        sc = lambda f: ps.score([a[f][p] for p in qs], 1)
        dd = lambda x, y: ps.ci([((a[x][p] == "R") - (a[x][p] == "W")) - ((a[y][p] == "R") - (a[y][p] == "W")) for p in qs])
        mn = dict(zip(qs, test_a.states(("Dnone",))))
        a["must-answer, no write"] = mn
        print("  %-14s (%d): must-answer no write %4d | must-answer + E %4d | plain no write %4d | plain single E %4d | plain N1 %4d || plain single E minus must-answer + E %-18s | plain N1 minus must-answer + E %s" % (
            name, len(qs), sc("must-answer, no write"), sc("must-answer + E"), sc("no write"), sc("single E"), sc("N1"), dd("single E", "must-answer + E"), dd("N1", "must-answer + E")))
    print("\n# Part 1: the stronger-caution run against the plain run")
    for name, m in cf.MODELS.items():
        if "stronger caution" in name:
            continue
        if name not in CAUTION:
            root = PENDING.get(name)
            print("\n## %s: no stronger-caution run%s" % (name, (" yet (expected under %s)" % root) if root else ""))
            continue
        mc, aux_c = CAUTION[name]
        print("\n## %s" % name)
        tune_a, ttx_a, pick_a = picks(m)
        tune_b, ttx_b, pick_b = picks(mc)
        pick_b = {f: pick_b[f] for f in ("no write", "single E", "N1")}
        print("  settings, plain run:   " + "; ".join("%s = %s" % (f, ro.name_of(c) if c[0] != "DE" else ps.describe(c)) for f, c in pick_a.items()))
        print("  settings, caution run: " + "; ".join("%s = %s" % (f, ro.name_of(c)) for f, c in pick_b.items()))
        show("tune, DeepSeek Pro", by_id(tune_a, ttx_a, pick_a), by_id(tune_b, ttx_b, pick_b), names)
        test_a, test_b = ps.Split(m, 1), ps.Split(mc, 1)
        show("test, DeepSeek Pro", by_id(test_a, ro.texts(m["ec"][1]), pick_a), by_id(test_b, ro.texts(mc["ec"][1]), pick_b), names)
        aux_a = tc.AUX.get(name) or m["ec"][1]
        if aux_c and all(os.path.exists(w + "/judge2/single_allenai.jsonl") for w in (aux_a, aux_c)):
            fa = ps.Split(dict(m, ec=(m["ec"][0], aux_a), d=None, judge=(m["judge"][0], "allenai")), 1)
            fb = ps.Split(dict(mc, ec=(mc["ec"][0], aux_c), d=None, judge=(mc["judge"][0], "allenai")), 1)
            pa = {f: c for f, c in pick_a.items() if c[0] != "DE"}
            show("auxiliary: full test split, AllenAI", by_id(fa, ro.texts(aux_a), pa), by_id(fb, ro.texts(aux_c), pick_b), names)


if __name__ == "__main__":
    main()
