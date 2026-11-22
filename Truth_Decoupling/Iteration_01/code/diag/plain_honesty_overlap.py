# Read-only. TruthfulQA, five models, existing DeepSeek Pro verdicts only.
# S0 = the plain prompt with the honesty sentence, nothing written (x_E2). How its verdicts overlap with single E and
# N1 of the main table (cross-table, oracle best-of-two), and T2 = a prompt-only two-pass comparator: S0 first, its
# blanks filled with the answers of the must-answer + sentence prompt (conf_E2).
# Usage: python3 -B Iteration_01/diag/plain_honesty_overlap.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_table as A
MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
V = {"R": 1, "W": -1, "X": 0, "H": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
def cls(v): return {1: "R", -1: "W", 0: "0"}[v]
for split, name in ((1, "test"), (0, "tune")):
    print("=====", name)
    per = {}
    for model in ORDER:
        display, layer, s = A.KEY[model]
        T, err = MP.prompt_split(display, layer, split, s)
        src = [x for x in ML.sources(display)[2] if x[0] == layer]
        want = set(src[0][3 if split else 1].qs)
        ids = [q for q in T[3] if q in want]
        if ids != T[3]:
            T, err = MP.prompt_split(display, layer, split, s, qs=ids)
        sp, tx, _, ids = T
        tT, _ = MP.prompt_split(display, layer, 0, s)
        e = S.system(sp, tx, ("E", s)); n1 = S.system(sp, tx, ML.n1_pick(tT[0], tT[1], s))
        D = MP.MT.Data(display, split)
        run = D.m["ec"][split].split("/eval_extra/")[0]
        w = MP.work(run, name, "h22c4")
        st, bl = ps.load(w, "pro")
        st1, bl1 = ps.load(D.m["d"][split], "pro")
        ok = [q for q in ids if st.get((q, "baseline_x_E2"), "?") != "?" and st1.get((q, "baseline_conf_E2"), "?") != "?"]
        ix = {q: k for k, q in enumerate(ids)}
        row = {}
        for q in ok:
            k = ix[q]
            s0 = (V[st[(q, "baseline_x_E2")]], not bl[(q, "baseline_x_E2")])
            c1 = (V[st1[(q, "baseline_conf_E2")]], not bl1[(q, "baseline_conf_E2")])
            row[q] = {"S0": s0, "E": (V[e[0][k]], e[1][k]), "N1": (V[n1[0][k]], n1[1][k]), "conf": c1,
                      "T2": s0 if s0[1] else c1}
        per[model] = row
        def tab(a, b):
            c = {}
            for q in row:
                key = cls(row[q][a][0]) + ("a" if row[q][a][1] else "n") + ">" + cls(row[q][b][0]) + ("a" if row[q][b][1] else "n")
                c[key] = c.get(key, 0) + 1
            return c
        t = tab("S0", "N1"); t2 = tab("S0", "E")
        def g(c, a, b): return sum(v for k, v in c.items() if k[0] == a and k[3] == b)
        print("%-12s %d q | S0 net %+d ans %d | T2 (S0 + blanks from must-answer+sentence) net %+d ans %d | N1 net %+d ans %d" % (
            model, len(row), sum(r["S0"][0] for r in row.values()), sum(r["S0"][1] for r in row.values()),
            sum(r["T2"][0] for r in row.values()), sum(r["T2"][1] for r in row.values()),
            sum(r["N1"][0] for r in row.values()), sum(r["N1"][1] for r in row.values())))
        print("     S0 wrong -> N1 right %d / zero %d / wrong %d | S0 right -> N1 right %d / zero %d / wrong %d | S0 zero -> N1 right %d / zero %d / wrong %d" % (
            g(t, "W", "R"), g(t, "W", "0"), g(t, "W", "W"), g(t, "R", "R"), g(t, "R", "0"), g(t, "R", "W"), g(t, "0", "R"), g(t, "0", "0"), g(t, "0", "W")))
        print("     S0 wrong -> E  right %d / zero %d / wrong %d | S0 right -> E  right %d / zero %d / wrong %d | S0 zero -> E  right %d / zero %d / wrong %d" % (
            g(t2, "W", "R"), g(t2, "W", "0"), g(t2, "W", "W"), g(t2, "R", "R"), g(t2, "R", "0"), g(t2, "R", "W"), g(t2, "0", "R"), g(t2, "0", "0"), g(t2, "0", "W")))
        print("     oracle best of (S0, N1) net %+d | best of (S0, E) net %+d" % (
            sum(max(r["S0"][0], r["N1"][0]) for r in row.values()), sum(max(r["S0"][0], r["E"][0]) for r in row.values())))
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    def tot(k): return sum(per[m][q][k][0] for m in ORDER for q in common), sum(per[m][q][k][1] for m in ORDER for q in common)
    def diff(a, b): return ps.ci([sum(per[m][q][a][0] - per[m][q][b][0] for m in ORDER) for q in common])
    print("five models, %d common questions: " % len(common) + " | ".join("%s net %+d ans %d" % ((k,) + tot(k)) for k in ("S0", "T2", "conf", "E", "N1")))
    print("   N1 minus S0 %s | N1 minus T2 %s | T2 minus S0 %s | oracle best of (S0, N1) %+d" % (
        diff("N1", "S0"), diff("N1", "T2"), diff("T2", "S0"), sum(max(per[m][q]["S0"][0], per[m][q]["N1"][0]) for m in ORDER for q in common)))
