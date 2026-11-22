# Read-only. TruthfulQA, five models, existing DeepSeek Pro verdicts only; nothing is chosen.
# S0 = the plain prompt with the honesty sentence, nothing written (x_E2). Fill systems:
#   F1 = S0 first; its blanks take N1's answer (main table N1).   F2 = N1 first; its blanks take S0's answer.
#   FE = S0 first; its blanks take single E's answer.            T2 = S0 first; its blanks take the answer of the
#   must-answer + sentence prompt (conf_E2), i.e. a prompt-only two-pass comparator.
# Also: on the S0-blank questions that N1 and the must-answer prompt both answer, the two nets side by side.
# Usage: python3 -B Iteration_01/diag/plain_honesty_fill.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_table as A
MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
V = {"R": 1, "W": -1, "X": 0, "H": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
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
        st, bl = ps.load(MP.work(run, name, "h22c4"), "pro")
        st1, bl1 = ps.load(D.m["d"][split], "pro")
        ix = {q: k for k, q in enumerate(ids)}
        row = {}
        for q in ids:
            if st.get((q, "baseline_x_E2"), "?") == "?" or st1.get((q, "baseline_conf_E2"), "?") == "?":
                continue
            k = ix[q]
            s0 = (V[st[(q, "baseline_x_E2")]], not bl[(q, "baseline_x_E2")])
            c1 = (V[st1[(q, "baseline_conf_E2")]], not bl1[(q, "baseline_conf_E2")])
            E = (V[e[0][k]], e[1][k]); N = (V[n1[0][k]], n1[1][k])
            row[q] = {"S0": s0, "E": E, "N1": N, "conf": c1, "T2": s0 if s0[1] else c1, "F1": s0 if s0[1] else N,
                      "F2": N if N[1] else s0, "FE": s0 if s0[1] else E}
        per[model] = row
        def tot(k): return sum(r[k][0] for r in row.values()), sum(r[k][1] for r in row.values())
        def d(a, b): return ps.ci([row[q][a][0] - row[q][b][0] for q in row])
        both = [q for q in row if (not row[q]["S0"][1]) and row[q]["N1"][1] and row[q]["conf"][1]]
        print("%-12s %d q | " % (model, len(row)) + " | ".join("%s %+d (%d)" % ((k,) + tot(k)) for k in ("S0", "N1", "F1", "F2", "FE", "T2")))
        print("     F1 minus S0 %s | F1 minus N1 %s | F1 minus T2 %s | on the %d S0-blank questions N1 and the must-answer prompt both answer: N1 %+d, must-answer %+d" % (
            d("F1", "S0"), d("F1", "N1"), d("F1", "T2"), len(both), sum(row[q]["N1"][0] for q in both), sum(row[q]["conf"][0] for q in both)))
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    def tot(k): return sum(per[m][q][k][0] for m in ORDER for q in common), sum(per[m][q][k][1] for m in ORDER for q in common)
    def diff(a, b): return ps.ci([sum(per[m][q][a][0] - per[m][q][b][0] for m in ORDER) for q in common])
    print("five models, %d common questions: " % len(common) + " | ".join("%s %+d (%d)" % ((k,) + tot(k)) for k in ("S0", "N1", "F1", "F2", "FE", "T2")))
    print("   F1 minus S0 %s | F1 minus N1 %s | F1 minus T2 %s | F2 minus S0 %s | FE minus S0 %s" % (
        diff("F1", "S0"), diff("F1", "N1"), diff("F1", "T2"), diff("F2", "S0"), diff("FE", "S0")))
    pair = [(m, q) for m in ORDER for q in common if (not per[m][q]["S0"][1]) and per[m][q]["N1"][1] and per[m][q]["conf"][1]]
    byq = {}
    for m, q in pair:
        byq[q] = byq.get(q, 0) + per[m][q]["N1"][0] - per[m][q]["conf"][0]
    print("   S0-blank questions answered by both N1 and the must-answer prompt: %d model x question | N1 %+d | must-answer %+d | difference %s" % (
        len(pair), sum(per[m][q]["N1"][0] for m, q in pair), sum(per[m][q]["conf"][0] for m, q in pair), ps.ci([byq.get(q, 0) for q in common])))
