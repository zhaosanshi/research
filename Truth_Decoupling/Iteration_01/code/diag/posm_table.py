# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# TruthfulQA test, DeepSeek Pro, score = right - wrong, everything written at the core rule "prompt": the learned E
# direction against the mean-difference direction at the SAME layer with the SAME write amount (kind mdg), single
# write, at each model's main-table setting (budget 1). The comparison of summary section 3 at the rule the method
# would use. Work dirs: single E judge_test_posprompt<tag>_test (pospall.sh), mean difference
# judge_test_pospromptm<tag>_test (posm.sh); the no-write cells of both must equal the usual ones.
# Printed per model: no write, mean difference, learned E, and N1 (from diag/main_table_prompt.py's split) with paired
# differences; then the five models together on the questions all have.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/posm_table.py
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_prompt as MP

ML, DM, MT, S, ps, ro, fr = MP.ML, MP.DM, MP.MT, MP.S, MP.ps, MP.ro, MP.fr
SETS = (("Llama-3.1-8B", 16, "3-5"), ("Qwen3.5-9B", 15, "2"), ("Qwen3.8-27B", 28, "1-75"), ("Gemma-3-12B", 24, "2-5"), ("Gemma-3-27B", 28, "1"))


def val(x):
    return (x == "R") - (x == "W")


def main():
    tot, sets = {}, []
    for model, layer, s in SETS:
        X, why = MP.prompt_split(model, layer, 1, s)
        T, why_t = MP.prompt_split(model, layer, 0, s)
        if X is None or T is None:
            print("\n## %s: the prompt-rule split is not complete (%s)" % (model, why or why_t))
            continue
        xsp, xtx, _, qs = X
        tsp, ttx, _, _ = T
        D, run, utx = MP.usual(model, 1)
        w = MP.work(run, "test", "pospromptml%d" % layer)
        if not MP.has(w):
            print("\n## %s: no Pro verdicts yet in %s" % (model, os.path.basename(w)))
            continue
        st, bl = ps.load(w, "pro")
        tx = fr.raw_texts(w)
        bad = [q for q in qs if (q, "baseline_x") in st and (st[(q, "baseline_x")] != xsp.st[(q, "baseline_x")] or tx.get((q, "baseline_x")) != utx[(q, "baseline_x")])]
        if bad:
            raise SystemExit("%s: the no-write cells of %s differ from the usual ones (%d)" % (model, os.path.basename(w), len(bad)))
        cond = "E_L%d_x-mdg-p%s" % (layer, s)
        if any(st.get((q, cond), "?") == "?" for q in qs):
            print("\n## %s: incomplete verdicts for %s" % (model, cond))
            continue
        md = [st[(q, cond)] for q in qs]
        base = S.system(xsp, xtx, ("base",))[0]
        e = S.system(xsp, xtx, ("E", s))
        n1 = S.system(xsp, xtx, ML.n1_pick(tsp, ttx, s))
        d = lambda a, b: [val(x) - val(y) for x, y in zip(a, b)]  # noqa: E731
        print("\n## %s, layer %d, strength %s, prompt rule, test %d questions" % (model, layer, DM.show(s), len(qs)))
        print("    no write net %+d | mean difference (same layer, same amount) net %+d (answers %d) | learned E net %+d (answers %d) | N1 net %+d (answers %d)" % (
            sum(map(val, base)), sum(map(val, md)), sum(not bl[(q, cond)] for q in qs), sum(map(val, e[0])), sum(e[1]), sum(map(val, n1[0])), sum(n1[1])))
        print("    learned E minus mean difference %s | N1 minus mean difference %s | mean difference minus no write %s" % (
            ps.ci(d(e[0], md)), ps.ci(d(n1[0], md)), ps.ci(d(md, base))))
        for key, v in (("learned E minus mean difference", d(e[0], md)), ("N1 minus mean difference", d(n1[0], md)), ("mean difference minus no write", d(md, base))):
            t = tot.setdefault(key, {})
            for q, x in zip(qs, v):
                t[q] = t.get(q, 0) + x
        sets.append(set(qs))
    if len(sets) == len(SETS):
        common = sorted(set.intersection(*sets))
        print("\n## the five models together on the %d questions all have" % len(common))
        for key, t in tot.items():
            print("    %-34s %s" % (key, ps.ci([t[q] for q in common])))


if __name__ == "__main__":
    main()
