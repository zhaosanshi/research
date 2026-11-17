# Iteration_01 analysis (Claude Fable 5.2, 2026-11-17), read-only, no model, no judge call.
# The two-caps scheme (two_caps.py) against the two prompt baselines on the test split (DeepSeek Pro; right - wrong):
#   prompt B     E at s; every question E leaves blank is asked again with the must-answer prompt (same E write)
#   prompt B own E at s; only the questions blank with no write (and still blank with E) are asked again with the must-answer prompt
# Settings chosen on the tune split. usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/two_caps_vs_prompt.py
import sys
sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import two_caps as tc

def pb_own(sp, s):
    e = sp.cE(s)
    return [sp.dst[(p, sp.dE(s))] if (sp.blank[(p, "baseline_x")] and sp.blank[(p, e)]) else sp.st[(p, e)] for p in sp.qs]

for name, m in cf.MODELS.items():
    if not m.get("d"):
        print("%-30s no must-answer judging on the test split" % name)
        continue
    tune, test = ps.Split(m, 0), ps.Split(m, 1)
    f2 = tc.FAM["two caps"]
    p2 = max(f2, key=lambda x: (ps.score(tc.states(tune, x), 1), -f2.index(x)))
    fB = [("B", "1"), ("B", "2")]
    pB = max(fB, key=lambda x: (ps.score(tune.states(x), 1), -fB.index(x)))
    sO = max("12", key=lambda s: (ps.score(pb_own(tune, s), 1), -int(s)))
    a, b, o = tc.states(test, p2), test.states(pB), pb_own(test, sO)
    print("%-30s two caps %4d | prompt B %4d | prompt B on own blanks %4d | two caps minus prompt B: %-18s | minus prompt B on own blanks: %s" % (
        name, ps.score(a, 1), ps.score(b, 1), ps.score(o, 1),
        ps.ci([x - y for x, y in zip(cf.val(a), cf.val(b))]), ps.ci([x - y for x, y in zip(cf.val(a), cf.val(o))])))
