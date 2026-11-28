"""Archived TruthfulQA only, main prompt-write E settings; one identical mandatory-answer question, O versus E.
All scoring is conditional on submission by the rule; also report the separate Pro-substantive definition (R/W)."""

import sys
import numpy as np

sys.path.insert(0, "Iteration_01/diag")
import tqa_gate2b_1127 as H
import tq_same_n_decomp as T

for m in H.ORDER:
    b = H.base(m)
    ids = b.ids
    d, run, _ = H.MP.usual(b.display, 1)
    assert run == b.run
    a, ab = H.ps.load(d.m["d"][1], "pro")
    forced = T.forced(b.run, b.el, b.sd, ids)
    ej = H.MP.work(run, "test", "pospromptdl%d" % b.el)
    e, eb = H.ps.load(ej, "pro")
    key = "E_L%d_conf-w-p%s" % (b.el, b.sd)
    aa = [a[q, "baseline_conf"] for q in ids]
    bb = [e[q, key] for q in ids]
    ar = np.array([not ab[q, "baseline_conf"] for q in ids], int)
    br = np.array([not eb[q, key] for q in ids], int)
    assert not any(x == "?" for x in aa + bb)
    assert all(
        (forced[q][0], forced[q][1]) == (H.VAL[H.G[x]], bool(flag))
        for q, x, flag in zip(ids, bb, br)
    )
    qi = np.random.default_rng(20261128).integers(len(ids), size=(4000, len(ids)))

    def fit(x):
        return [
            int(x.sum()),
            *np.quantile(x[qi].sum(1), [0.025, 0.975], method="inverted_cdf")
            .astype(int)
            .tolist(),
        ]

    oa = np.array([H.VAL[H.G[x]] for x in aa]) * ar
    ee = np.array([H.VAL[H.G[x]] for x in bb]) * br
    sa = np.array([x in ("R", "W") for x in aa], int)
    sb = np.array([x in ("R", "W") for x in bb], int)
    print(
        m,
        len(ids),
        "E layer/amp",
        b.el,
        b.s,
        "rule n O/E",
        ar.sum(),
        br.sum(),
        "diff95",
        fit(br - ar),
        "judge-substantive n O/E",
        sa.sum(),
        sb.sum(),
        "diff95",
        fit(sb - sa),
        "issued-only nets O/E & diff95",
        oa.sum(),
        ee.sum(),
        fit(ee - oa),
        "right-only strict-submitted difference95",
        fit((ee == 1).astype(int) - (oa == 1).astype(int)),
    )
