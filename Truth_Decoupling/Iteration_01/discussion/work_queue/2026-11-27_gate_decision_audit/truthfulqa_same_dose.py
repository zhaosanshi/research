"""CPU-only reconstruction from the archived TruthfulQA question answers, judges, gate and fallback confidence."""

import sys
import json
import hashlib
import numpy as np

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/gpt_tools_20261126/gate_audit"]
import tqa_gate2b_1127 as H
import q100_gate2_1127 as Q


def cell(val, answered, text):
    if val not in ("R", "W", "H", "X"):
        raise ValueError(val)
    return ((val == "R") - (val == "W"), bool(answered), text.strip())


def run(m, phase):
    b = H.base(m)
    qs = b.ids if phase == "full" else [q for q in b.ids if q in set(Q.dr100())]
    root = H.GS.OUT / m
    f = json.loads((root / "fit_B_gate_1126.json").read_bytes())
    p = f["probs"]["chosen"]["tq_test"]
    assert (
        f["capture_sha256"]
        == hashlib.sha256((root / "capture_gate_1126.json").read_bytes()).hexdigest()
        and f["chosen_layer"] == f["refit"]["chosen"]["layer"]
        and f["model"] == m
        and f["pool"] == "B"
    )
    assert all(q in p and 0 <= p[q] <= 1 for q in qs)
    on = {q for q in qs if p[q] > 0.5}
    new = H.load_new(b)
    keys = H.conds(m)
    work = Q.work_b(b, phase)
    judged, _ = H.ps.load(work, "pro")
    want = {(q, keys[k]) for k, ids in H.needed(b, new, qs).items() for q in ids}
    rawby = {(q, keys[k]): new[k][q]["text"] for k in keys for q in qs}
    seen = {}
    for ln in open(work + "/judge2/cells.jsonl"):
        r = json.loads(ln)
        ident = (r["pair_id"], r["condition"])
        if ident in want:
            assert r["prediction"] == rawby[ident] and ident not in seen
            seen[ident] = 1
    assert want == set(seen) and all(
        judged.get(k) in ("R", "W", "H", "X") for k in want
    )

    def newcell(n, q):
        a = new[n][q]
        return cell(
            judged[q, keys[n]] if (q, keys[n]) in want else "X",
            not H.blank_row(a),
            a["text"],
        )

    o = {
        q: cell(
            b.sp.st[q, "baseline_x"],
            not b.sp.blank[q, "baseline_x"],
            b.tx.ec[q, "baseline_x"],
        )
        for q in qs
    }
    lo_st, lo_ans, lo_text = H.S.system(b.sp, b.tx, b.pick)[:3]
    low_reg = dict(zip(b.ids, zip(lo_st, lo_ans, lo_text)))
    n1 = {q: cell(low_reg[q][0], low_reg[q][1], low_reg[q][2]["text"]) for q in qs}
    e1 = {
        q: cell(
            b.sp.st[q, b.sp.cE(b.sd)],
            not b.sp.blank[q, b.sp.cE(b.sd)],
            b.tx.ec[q, b.sp.cE(b.sd)],
        )
        for q in qs
    }
    e2 = {q: newcell("E2", q) for q in qs}
    n2 = {}
    for q in qs:
        n2[q] = e2[q]
        if e2[q][1]:
            continue
        cap = b.cap_e if o[q][1] else b.cap_own
        for k in H.LAD[: H.LAD.index(cap) + 1] if cap is not None else []:
            v = newcell("EC2_" + k, q)
            if not v[1]:
                continue
            if not (o[q][1] and H.RO.norm(v[2]) == H.RO.norm(o[q][2])):
                n2[q] = v
            break
    gate = lambda d: {q: d[q] if q in on else o[q] for q in qs}
    n, gn = n2, gate(n2)
    l, gl = n1, gate(n1)
    ge = gate(e2)
    net = lambda d: sum(d[q][0] for q in qs)
    cnt = lambda d: sum(d[q][1] for q in qs)
    print(
        "\nMODEL",
        m,
        phase,
        len(qs),
        "on",
        len(on),
        "net,count",
        {
            name: (net(d), cnt(d))
            for name, d in [
                ("O", o),
                ("E1", e1),
                ("E2", e2),
                ("L", l),
                ("GL", gl),
                ("H", n),
                ("GH", gn),
                ("GE2", ge),
            ]
        },
        flush=True,
    )
    rng = np.random.default_rng(20261127)
    indices = rng.integers(len(qs), size=(15000, len(qs)), dtype=np.int16)

    def ci(x):
        x = np.asarray(x, dtype=int)
        lo, hi = np.quantile(
            x[indices].sum(axis=1), [0.025, 0.975], method="inverted_cdf"
        )
        return (sum(x.tolist()), int(lo), int(hi))

    def comp(k, a, b_, eligible=qs):
        d = [a[q][0] - b_[q][0] for q in qs]
        n_a, n_b = cnt(a), cnt(b_)
        diff = n_a - n_b
        pool = a if diff >= 0 else b_
        drop = sorted(pool[q][0] for q in eligible if pool[q][1])
        rg = []
        if diff:
            if len(drop) < abs(diff):
                raise ValueError("unreachable count")
            dropped = [sum(drop[: abs(diff)]), sum(drop[-abs(diff) :])]
            rg = sorted(sum(d) - (1 if diff > 0 else -1) * x for x in dropped)
        agree = [v if a[q][2] != b_[q][2] else 0 for q, v in zip(qs, d)]
        issuedonly = [
            (a[q][0] if a[q][1] else 0) - (b_[q][0] if b_[q][1] else 0) for q in qs
        ]
        print(
            k,
            "cost,95% / same text unified / issued-only:",
            ci(d),
            ci(agree),
            ci(issuedonly),
            "delta counts",
            diff,
            "any equal-K trim net bounds",
            rg if diff else [sum(d)] * 2,
        )
        return d

    cs = comp("gate only at high (GH-H)", gn, n)
    low = comp("gate only at current (GL-L)", gl, l)
    dh = comp("increase both doses (H-L)", n, l)
    dg = comp("increase dose inside same gate (GH-GL)", gn, gl, on)
    comp("published composite (GH-L)", gn, l)
    assert all(x - y == a - bb for x, y, a, bb in zip(cs, low, dg, dh))
    print("gate by dose interaction", ci([x - y for x, y in zip(cs, low)]))
    comp("combination minus one-pass high E inside gate", gn, ge, on)
    delta_high = [n[q][0] - l[q][0] for q in qs]
    opportunity = [n[q][0] - o[q][0] for q in qs]
    print(
        "all value on OFF current,high; dose effect ON,OFF",
        sum(l[q][0] - o[q][0] for q in qs if q not in on),
        sum(n[q][0] - o[q][0] for q in qs if q not in on),
        sum(n[q][0] - l[q][0] for q in on),
        sum(n[q][0] - l[q][0] for q in qs if q not in on),
    )
    # Counterfactual: what if we selectively allow the identical number of E highs/active expert evaluations at random?
    # N1 and O trajectories held fixed. Redraw the base questions jointly and re-estimate the triggered fraction in each bootstrap.
    active = np.array([q in on for q in qs])
    val = np.array(opportunity)
    observed = (active * val).sum()
    boot = (active * val)[indices].sum(1) - active[indices].mean(1) * val[indices].sum(
        1
    )
    print(
        "high gate vs random at exact same opening count (expected over random subsets)",
        round(observed - active.mean() * val.sum(), 2),
        np.quantile(boot, [0.025, 0.975]).round(1),
    )
    effect = np.array(delta_high)
    weighted = (active * effect).sum() - active.mean() * effect.sum()
    boot = (active * effect)[indices].sum(1) - active[indices].mean(1) * effect[
        indices
    ].sum(1)
    print(
        "raise dose inside real gate vs random same number of stronger writes; low dose elsewhere, additional cost same",
        round(weighted, 2),
        np.quantile(boot, [0.025, 0.975]).round(1),
    )
    print("direct high-gate vs full high, repository paired interval", H.GS.tq_ci(cs))
    for name, base_policy, score in [("O off", o, val), ("low N1 off", l, effect)]:
        counts = np.array([int(n[q][1]) - int(base_policy[q][1]) for q in qs])
        a = active.astype(int)
        adjustment = int((a * counts).sum()) - a.mean() * int(counts.sum())
        expected = 0
        boot_expected = np.zeros(len(indices))
        group_rows = []
        for flag in (-1, 0, 1):
            inside = counts == flag
            number = int(inside.sum())
            if not number:
                continue
            selected = int((inside * a).sum())
            group_rows.append((flag, selected, number))
            expected += selected / number * score[inside].sum()
            denominator = inside[indices].sum(1)
            numerator = (inside * a)[indices].sum(1)
            weight = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=float),
                where=denominator > 0,
            )
            boot_expected += weight * (inside * score)[indices].sum(1)
        constrained = (a * score).sum() - expected
        d = (a * score)[indices].sum(1) - boot_expected
        print(
            name,
            "actual minus exact-size random handed-in counts (expectation)",
            round(adjustment, 2),
            "groups (change of answer flag, high count, all)",
            group_rows,
            "net minus random with SAME #strong and SAME #answers (permute within those groups)",
            round(constrained, 2),
            np.quantile(d, [0.025, 0.975]).round(1),
        )
    print("native paired interval for gate minus always high", H.GS.tq_ci(cs))
    if phase == "full":
        initial = set(Q.dr100())
        remaining = [q for q in qs if q not in initial]
        print(
            "outside the screening subset",
            len(remaining),
            "composite / dose alone / gate alone",
            [
                H.GS.tq_ci([a[q][0] - bb[q][0] for q in remaining])
                for a, bb in [(gn, l), (n, l), (gn, n)]
            ],
        )
        print(
            "composite, submitted-only",
            H.GS.tq_ci(
                [
                    (gn[q][0] if gn[q][1] else 0) - (l[q][0] if l[q][1] else 0)
                    for q in qs
                ]
            ),
        )


if __name__ == "__main__":
    for model, phase in [
        ("gemma3-27b", "full"),
        ("qwen3.8-27b", "q100"),
        ("qwen3.5-9b", "full"),
    ]:
        run(model, phase)
