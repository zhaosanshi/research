# -*- coding: utf-8 -*-
"""Read-only: chance that the frozen B column (criteria v4, section 4.1) is met on N new questions IF the per-question
changes seen on the tuning questions were the true distribution. Plug-in simulation: draw N questions with replacement
from the tuning questions (all arms of one question move together), apply the B checks with normal-approximation 95%
intervals. usage: e44_stageB_power.py JUDGE N SIMS MAIN_WORK AXIS_WORK [AXIS_WORK ...]   (layer 11, family E)"""
import sys, math, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t

judge, N, sims, main_work, axis_works = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5:]
L = 11
_, _, ids, st, _, _ = t.load(main_work, judge)
R = lambda s, c, i: 1 if s[(i, c)] == "R" else 0                      # noqa: E731
A = lambda s, c, i: 1 if s[(i, c)] in ("R", "W") else 0               # noqa: E731
w, xf, xfw = t.cond("E", L, "x.w.p1"), "baseline_x_F", t.cond("E", L, "x_F.w.p1")
rows = {i: {"d": R(st, w, i) - R(st, "baseline_x", i), "a": A(st, w, i) - A(st, "baseline_x", i),
            "dF": R(st, xfw, i) - R(st, xf, i), "aF": A(st, xfw, i) - A(st, xf, i), "rand": []} for i in ids}
for work in [main_work] + axis_works:
    _, _, ids2, s2, _, _ = t.load(work, judge)
    assert ids2 == ids
    for arm in ("x.rand.p1", "x.rand.m1"):
        c = t.cond("E", L, arm)
        for i in ids:
            rows[i]["rand"].append(R(s2, c, i) - R(s2, "baseline_x", i))
K = len(rows[ids[0]]["rand"])
pool = [rows[i] for i in ids]
rng = random.Random(20261111)


def ci(values):
    n = len(values); m = sum(values) / n
    v = sum((x - m) ** 2 for x in values) / (n - 1)
    h = 1.96 * math.sqrt(v / n)
    return m - h, m, m + h


tol = 0.03
counts = {"main": 0, "random_all": 0, "F": 0, "retention": 0, "all": 0}
for _ in range(sims):
    draw = [pool[rng.randrange(len(pool))] for _ in range(N)]
    lo, m, hi = ci([r["d"] for r in draw])
    main_ok = lo > 0 and m > tol
    rand_ok = all(ci([r["d"] - r["rand"][k] for r in draw])[0] > 0 for k in range(K))
    lo_f, m_f, _ = ci([r["dF"] for r in draw]); lo_a, _, hi_a = ci([r["aF"] for r in draw])
    f_ok = lo_f > 0 and lo_a > -tol and hi_a < tol
    lo_r, _, hi_r = ci([r["a"] for r in draw])
    ret_ok = lo_r > -tol and hi_r < tol
    for k, ok in (("main", main_ok), ("random_all", rand_ok), ("F", f_ok), ("retention", ret_ok), ("all", main_ok and rand_ok and f_ok and ret_ok)):
        counts[k] += ok
mean = lambda key: sum(r[key] for r in pool) / len(pool)  # noqa: E731
print("judge %s, layer %d, tuning means per question: d %+.4f, a %+.4f, dF %+.4f, aF %+.4f, random arms %d" % (
    judge, L, mean("d"), mean("a"), mean("dF"), mean("aF"), K))
print("N = %d, %d simulations: " % (N, sims) + ", ".join("%s %.3f" % (k, v / sims) for k, v in counts.items()))
