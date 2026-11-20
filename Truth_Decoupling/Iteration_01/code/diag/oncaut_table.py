# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# The N1 scheme with the best prompt-only baseline as its base prompt: every arm of
# logs/newsets_20261119/oncaut_all.sh runs on caut_E2 (cautious base + the honest sentence), which still lets the model
# decline. TruthfulQA, DeepSeek Pro, score = right - wrong. Work dirs judge_<split>_oncaut_<split>.
# The records are built by diag/final_records.py unchanged; for that the base prompt's name in the condition names is
# read as "x" (the name plays no part in the logic; the same device as diag/cap_score.py), so "no write" below is the
# prompt caut_E2 with nothing written, i.e. the prompt-only baseline itself.
# Per model and strength: the caps of N1 are chosen on tune (single E is a candidate too); on test: the prompt alone,
# single E on it, N1 on it, and the differences with paired intervals.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/oncaut_table.py [MODEL ...]
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_same_n as MT
import newsets_score as S

ps, ro, fr = MT.ps, MT.ro, MT.fr
CAPS = MT.CAPS
BASE = "caut_E2"


def rename(c):
    if c == "baseline_" + BASE:
        return "baseline_x"
    return c.replace("_%s-" % BASE, "_x-").replace("X_%s--" % BASE, "X_x--")


def load(D, split):
    run = D.m["ec"][0].split("/eval_extra/")[0]
    w = "%s/eval_extra/judge_%s_oncaut_%s" % (run, split, split)
    if not os.path.exists(w + "/judge2/single_pro.jsonl"):
        return None
    st0, bl0 = ps.load(w, "pro")
    st = {(p, rename(c)): v for (p, c), v in st0.items()}
    bl = {(p, rename(c)): v for (p, c), v in bl0.items()}
    tx = {(p, rename(c)): v for (p, c), v in fr.raw_texts(w).items()}
    shell = S.shell(D.m["e"], D.m["c"])
    strengths = sorted({c.split("-p")[-1] for _, c in st if c.startswith("E_L")}, key=lambda s: float(s.replace("-", ".")))
    strengths = [s for s in strengths if all((shell.cEC(s, k) in {c for _, c in st}) for k in ps.LAD)]
    need = ["baseline_x"] + [c for s in strengths for c in [shell.cE(s)] + [shell.cEC(s, k) for k in ps.LAD]]
    sp = S.NewSplit(D.m["e"], D.m["c"], st, bl, need)
    return sp, fr.Texts(tx, tx), strengths


def val(x):
    return (x == "R") - (x == "W")


def main():
    for model in sys.argv[1:] or list(ro.FIVE):
        Dt, Dx = MT.Data(model, 0), MT.Data(model, 1)
        t, x = load(Dt, "tune"), load(Dx, "test")
        if t is None or x is None:
            print("\n## %s: no Pro verdicts for the %s base yet (tune %s, test %s)" % (model, BASE, t is not None, x is not None))
            continue
        tsp, ttx, ts = t
        xsp, xtx, xs = x
        base = S.system(xsp, xtx, ("base",))
        print("\n## %s, base prompt %s: test %d questions | the prompt alone: answers %d, right %d, wrong %d, net %+d (tune net %+d)" % (
            model, BASE, len(xsp.qs), sum(base[1]), base[0].count("R"), base[0].count("W"), sum(map(val, base[0])),
            sum(map(val, S.system(tsp, ttx, ("base",))[0]))))
        for s in [s for s in ts if s in xs]:
            cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
            pick = max(cands, key=lambda c: (ps.score(S.system(tsp, ttx, c)[0], 1), -cands.index(c)))
            e, n1 = S.system(xsp, xtx, ("E", s)), S.system(xsp, xtx, pick)
            d = lambda a, b: ps.ci([val(p) - val(q) for p, q in zip(a[0], b[0])])
            print("    E %-4s single E: answers %3d net %+4d | N1 (on tune: %s; tune net %+d): answers %3d net %+4d || single E minus the prompt %s | N1 minus the prompt %s | N1 minus single E %s" % (
                s.replace("-", "."), sum(e[1]), sum(map(val, e[0])), ro.name_of(pick), ps.score(S.system(tsp, ttx, pick)[0], 1),
                sum(n1[1]), sum(map(val, n1[0])), d(e, base), d(n1, base), d(n1, e)))


if __name__ == "__main__":
    main()
