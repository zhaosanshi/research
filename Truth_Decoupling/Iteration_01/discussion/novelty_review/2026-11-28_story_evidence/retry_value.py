"""Compare the payoff from the very same E+C retry in the two kinds of E abstentions.
All eligible questions are in the denominator. Use paired resampling across the original test questions/pairs.
"""

import sys
import numpy as np

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
]
import falseqa_read as f
import tqa_gate2b_1127 as t


def run(dataset, model, bank):
    if dataset == "TruthfulQA":
        b = t.base(model)
        ids = b.ids
        groups = ids

        def arm(cond):
            return [
                (not b.sp.blank[q, cond], t.VAL[t.G[b.sp.st[q, cond]]]) for q in ids
            ]

        o, e, ec = [arm(k) for k in ("baseline_x", b.sp.cE(b.sd), b.sp.cEC(b.sd, "1"))]
    else:
        records = [r for r in bank if r["set"] == "heldout"]
        ids = [r["id"] for r in records]
        groups = [r["pair_id"] for r in records]
        raw, _ = f.load_rows(model, set(ids))
        s = f.setting(model)[1]

        def arm(cond):
            return [(not raw.blank(cond, q), raw.cell(cond, q)) for q in ids]

        o, e, ec = [arm(k) for k in ("x", raw.e(s), raw.ec(s, "1"))]
    double = np.array([not a[0] and not b[0] for a, b in zip(o, e)])
    induced = np.array([a[0] and not b[0] for a, b in zip(o, e)])
    v = np.array([int(judge) if submitted else 0 for submitted, judge in ec])
    assert not np.any(double & induced) and double.any() and induced.any()
    group_keys = list(dict.fromkeys(groups))
    pos = {q: j for j, q in enumerate(group_keys)}
    unit = [pos[q] for q in groups]
    by_group = np.zeros((len(group_keys), 4))
    np.add.at(
        by_group, unit, np.column_stack((double, induced, double * v, induced * v))
    )
    nD, nI, vD, vI = by_group.sum(axis=0)
    actual = 100 * (vD / nD - vI / nI)
    rng = np.random.default_rng(20261128)
    replicas = []
    for _ in range(40):
        draws = rng.integers(len(group_keys), size=(100, len(group_keys)))
        dd, ii, vd, vi = by_group[draws].sum(axis=1).T
        valid = (dd > 0) & (ii > 0)
        replicas.extend(100 * (vd[valid] / dd[valid] - vi[valid] / ii[valid]))
    ci = np.quantile(replicas, [0.025, 0.975]).round(2).tolist()
    print(
        dataset,
        model,
        "n",
        len(ids),
        "D eligible,net",
        int(nD),
        int(vD),
        "I eligible,net",
        int(nI),
        int(vI),
        "net per 100 attempts D/I",
        round(100 * vD / nD, 2),
        round(100 * vI / nI, 2),
        "D-I per 100 attempts",
        round(actual, 2),
        "paired 95%",
        ci,
        "D vs uniform same number of EC attempts, expected net difference",
        round(vD - nD / (nD + nI) * (vD + vI), 2),
        flush=True,
    )


if __name__ == "__main__":
    ok, bank = f.check_bank()
    assert ok
    for dataset in ("TruthfulQA", "FalseQA"):
        for model in t.ORDER:
            run(dataset, model, bank)
