# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22, queue I section 7.6), read-only, no model, no judge call.
# TruthfulQA, DeepSeek Pro, the main table's question lists (diag/n1_on_sentence.restricted). Every no-write prompt with
# verdicts: the prompt sweep of the run (eval_extra/judge_<split>_promptsweep_<split>: x, x_E1, x_F, conf, caut, conf_E1,
# caut_E1, conf_E2, caut_E2, conf_M, caut_M), x_E2 (judge_<split>_h22c4_<split>) and the three of section 7.6
# (judge_<split>_pfam_<split>: x_H, x_E2_H, x_E1_E2).
#   1. net and answers of every prompt on tune and test;
#   2. PB = per model the prompt with the highest tune net (ties: more answers, then the name); on test SN1 - PB and
#      SE - PB (SN1 / SE of step one, diag/n1_on_sentence.py, natural answer counts);
#   3. equal answer counts: both sides keep n = the smaller of the two answer counts, the rest scored 0; the side with more
#      answers can keep any n of its answers, so SN1 - PB ranges from (its worst n) to (its best n), per model.
# Intervals: policy_select2.ci per model and on the per-question sums of the common questions.
# usage (scoring container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/prompt_family.py
import json
import sys

sys.path.insert(0, "Iteration_01/diag")
import n1_on_sentence as N  # noqa: E402

A, MP, ps, V, ORDER = N.A, N.MP, N.ps, N.V, N.ORDER


def prompts(model, i, qs):
    display, layer, s = A.KEY[model]
    D = MP.MT.Data(display, i)
    split = ("tune", "test")[i]
    run = D.m["ec"][i].split("/eval_extra/")[0]
    out = {}
    for name in ("promptsweep", "h22c4", "pfam"):
        w = MP.work(run, split, name)
        if not MP.has(w):
            raise FileNotFoundError("%s: no Pro verdicts in %s" % (model, w))
        st, bl = ps.load(w, "pro")
        for c in sorted({c for _, c in st if c.startswith("baseline_")}):
            p = c[len("baseline_"):]
            if all(st.get((q, c), "?") != "?" for q in qs):
                d = {q: (V[st[(q, c)]], not bl[(q, c)]) for q in qs}
                if p in out and out[p] != d:
                    raise SystemExit("%s: prompt %s has different verdicts in two work dirs" % (model, p))
                out[p] = d
    return out


def tot(d, qs):
    return sum(d[q][0] for q in qs), sum(d[q][1] for q in qs)


def main():
    per, qsm, pbs = {}, {}, {}
    for model in ORDER:
        ch = json.load(open(N.CHOICE % model))
        ct, cx = N.restricted(model, 0), N.restricted(model, 1)
        tp, xp = prompts(model, 0, ct), prompts(model, 1, cx)
        names = sorted(set(tp) & set(xp))
        print("\n## %s: %d tune / %d test questions" % (model, len(ct), len(cx)))
        for p in sorted(names, key=lambda p: -tot(tp[p], ct)[0]):
            print("    %-9s tune %+4d (%3d) | test %+4d (%3d)" % ((p,) + tot(tp[p], ct) + tot(xp[p], cx)))
        pb = max(names, key=lambda p: (tot(tp[p], ct)[0], tot(tp[p], ct)[1], p))
        sp, tx = N.build(model, 1, "x_E2", ch["strength"], cx)
        se, sn = N.run_sys(sp, tx, ("E", ch["strength"])), N.run_sys(sp, tx, tuple(ch["cfg"]))
        qs = [q for q in sp.qs if q in xp[pb]]
        per[model] = {"PB": xp[pb], "SE": se, "SN1": sn}
        qsm[model], pbs[model] = qs, pb
        nS, nP = sum(sn[q][1] for q in qs), sum(xp[pb][q][1] for q in qs)
        vs = sorted(sn[q][0] for q in qs if sn[q][1])
        vp = sorted(xp[pb][q][0] for q in qs if xp[pb][q][1])
        netS, netP = sum(vs), sum(vp)
        n = min(nS, nP)
        if nS >= nP:
            lo, hi = sum(vs[:n]) - netP, sum(vs[len(vs) - n:]) - netP
        else:
            lo, hi = netS - sum(vp[len(vp) - n:]), netS - sum(vp[:n])
        print("    PB = %s (tune %+d) | test %d q: PB %+d (%d) | SE %+d (%d) | SN1 %+d (%d) | SN1 - PB %s | SE - PB %s | equal answer counts (n = %d): SN1 - PB from %+d to %+d" % (
            (pb, tot(tp[pb], ct)[0], len(qs)) + tot(xp[pb], qs) + tot(se, qs) + tot(sn, qs)
            + (ps.ci([sn[q][0] - xp[pb][q][0] for q in qs]), ps.ci([se[q][0] - xp[pb][q][0] for q in qs]), n, lo, hi)))
    common = sorted(set.intersection(*(set(v) for v in qsm.values())))
    print("\n## the five models together, %d common test questions (PB per model: %s)" % (len(common), ", ".join("%s %s" % (m, pbs[m]) for m in ORDER)))
    print("    " + " | ".join("%s %+d (%d)" % (k, sum(per[m][k][q][0] for m in ORDER for q in common),
                                             sum(per[m][k][q][1] for m in ORDER for q in common)) for k in ("PB", "SE", "SN1")))
    print("    SN1 - PB %s | SE - PB %s" % tuple(ps.ci([sum(per[m][a][q][0] - per[m]["PB"][q][0] for m in ORDER) for q in common]) for a in ("SN1", "SE")))


if __name__ == "__main__":
    main()
