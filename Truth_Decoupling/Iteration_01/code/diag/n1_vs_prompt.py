# Iteration_01 analysis (Claude Fable 5.2, 2026-11-17), read-only, no model, no judge call.
# N1 (release_origin.py: two caps, and one of E's own blanks is released only with a new answer) against the two prompt
# baselines and against the method, test split, DeepSeek Pro, right - wrong; settings chosen on tune.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/n1_vs_prompt.py
import sys
sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import two_caps as tc
import release_origin as ro
CAPS = ro.CAPS
def pb_own(sp, s):
    e = sp.cE(s)
    return [sp.dst[(p, sp.dE(s))] if (sp.blank[(p, "baseline_x")] and sp.blank[(p, e)]) else sp.st[(p, e)] for p in sp.qs]
for name, m in cf.MODELS.items():
    tune, test = ps.Split(m, 0), ps.Split(m, 1)
    ttx, etx = ro.texts(m["ec"][0]), ro.texts(m["ec"][1])
    f1 = [("E", s) for s in "12"] + [("N1", s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
    p1 = max(f1, key=lambda x: (ps.score(ro.states(tune, ttx, x), 1), -f1.index(x)))
    a = ro.states(test, etx, p1)
    fm = [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in ps.LAD]
    pm = max(fm, key=lambda x: (ps.score(tune.states(x), 1), -fm.index(x)))
    mm = test.states(pm)
    fc = [("C", k) for k in ps.LAD if k in tune.cks and k in test.cks]
    pc = max(fc, key=lambda x: (ps.score(tune.states(x), 1), -fc.index(x)))
    cc = test.states(pc)
    line = "%-30s N1 %4d (%s) | minus method %-16s | minus single C %-22s" % (name, ps.score(a, 1), ro.name_of(p1),
        ps.ci([x - y for x, y in zip(cf.val(a), cf.val(mm))]), ps.ci([x - y for x, y in zip(cf.val(a), cf.val(cc))]))
    if test.dst is not None:
        fB = [("B", "1"), ("B", "2")]
        pB = max(fB, key=lambda x: (ps.score(tune.states(x), 1), -fB.index(x)))
        sO = max("12", key=lambda s: (ps.score(pb_own(tune, s), 1), -int(s)))
        b, o = test.states(pB), pb_own(test, sO)
        line += " | prompt B %d: N1 minus it %-16s | prompt B on own blanks %d: N1 minus it %s" % (
            ps.score(b, 1), ps.ci([x - y for x, y in zip(cf.val(a), cf.val(b))]), ps.score(o, 1), ps.ci([x - y for x, y in zip(cf.val(a), cf.val(o))]))
    print(line)
