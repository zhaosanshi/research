"""Change only what happens when E abstains on a question the unmodified model answered.
Read archived ordinary-prompt generations and ratings. Use natural counts; an unsubmitted response scores zero.
"""

import sys
import random
from functools import lru_cache

import numpy as np

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
]
import falseqa_read as fq
import tqa_gate2b_1127 as tq
import simple_ec_1127 as previous


@lru_cache(maxsize=6)
def bootstrap_weights(n, seed):
    rng = random.Random(seed)
    return np.stack(
        [
            np.bincount([rng.randrange(n) for _ in range(n)], minlength=n)
            for _ in range(4000)
        ]
    ).astype(np.int16)


def interval(delta, group, seed):
    units = {g: i for i, g in enumerate(dict.fromkeys(group))}
    values = np.zeros(len(units), dtype=np.int64)
    np.add.at(values, [units[g] for g in group], delta)
    draws = np.sort(bootstrap_weights(len(values), seed) @ values)
    return [int(values.sum()), int(draws[100]), int(draws[3899])]


def load(dataset, model, rows):
    if dataset == "FalseQA":
        held = [r for r in rows if r["set"] == "heldout"]
        ids = [r["id"] for r in held]
        group = [r["pair_id"] for r in held]
        raw, _ = fq.load_rows(model, set(ids))
        dev, dev_other = fq.load_rows(
            model, {r["id"] for r in rows if r["set"] == "dev"}
        )
        st = fq.setting(model)
        caps = fq.choose(dev, dev_other, st)[0]["N1"]
        roster = fq.roster(model)
        n1 = fq.n1_cells(raw, st[1], caps, ids)
        canonical = [n1[q][:3] for q in ids]

        def read(cond):
            arm = roster[cond]
            return [
                (raw.cell(arm, q), not raw.blank(arm, q), raw.text(arm, q)) for q in ids
            ]

        devdata = previous.fqa_split(model, rows, "dev")
    else:
        b = tq.base(model)
        ids, group = b.ids, b.ids
        caps = tuple(tq.DOT[k] if k is not None else None for k in (b.cap_e, b.cap_own))
        states, issued, records = tq.S.system(b.sp, b.tx, b.pick)[:3]
        canonical = [
            (tq.VAL[tq.G[v]], a, r["text"]) for v, a, r in zip(states, issued, records)
        ]

        def read(cond):
            if cond == "O":
                key = "baseline_x"
            elif cond == "E":
                key = b.sp.cE(b.sd)
            elif cond.startswith("EC"):
                key = b.sp.cEC(b.sd, cond[2:].replace(".", "-"))
            else:
                key = b.sp.cC(cond[1:].replace(".", "-"))
            return [
                (tq.VAL[tq.G[b.sp.st[q, key]]], not b.sp.blank[q, key], b.tx.ec[q, key])
                for q in ids
            ]

        devdata = previous.tqa_split(model, 0)
    _, dev_scores, candidates, _ = devdata
    c = previous.pick(
        {k: sum(dev_scores["C" + k]) for k in candidates}, candidates
    ).replace("-", ".")
    return group, read, caps, canonical, c


def summary(cells):
    scores = [int(v) for v, issued, _ in cells if issued]
    return dict(
        submitted=len(scores),
        right=scores.count(1),
        wrong=scores.count(-1),
        other=scores.count(0),
        net=sum(scores),
    )


def analyze(dataset, model, rows):
    groups, read, caps, full, c_dose = load(dataset, model, rows)
    original, e = read("O"), read("E")
    ladder = {k: read("EC" + k) for k in fq.LAD}
    single_c = read("C" + c_dose)
    affected = [
        i for i, (o, first) in enumerate(zip(original, e)) if o[1] and not first[1]
    ]
    simple = [
        first if first[1] or o[1] else candidate
        for first, o, candidate in zip(e, original, ladder["1"])
    ]
    hybrid = list(simple)
    restored = list(simple)
    calls = veto = 0
    active = [] if caps[0] is None else fq.LAD[: fq.LAD.index(caps[0]) + 1]
    for i in affected:
        restored[i] = original[i]
        for dose in active:
            calls += 1
            candidate = ladder[dose][i]
            if candidate[1]:
                if fq.Q.ro.norm(original[i][2]) == fq.Q.ro.norm(candidate[2]):
                    veto += 1
                else:
                    hybrid[i] = candidate
                break
        assert hybrid[i] == full[i], (dataset, model, i)
    print(
        dataset,
        model,
        "n",
        len(groups),
        "affected",
        len(affected),
        "cap",
        caps[0],
        "calls",
        calls,
        "vetoed",
        veto,
        flush=True,
    )
    for name, arm in [
        ("simple", simple),
        ("hybrid", hybrid),
        ("original", restored),
        ("complete_N1", full),
    ]:
        print(
            " ",
            name,
            "all",
            summary(arm),
            "affected",
            summary([arm[i] for i in affected]),
        )

    def vals(arm):
        return np.array([v if a else 0 for v, a, _ in arm], dtype=np.int64)

    seed = 20261116 if dataset == "TruthfulQA" else fq.S.SEED
    for name, a, b in [
        ("hybrid-simple", hybrid, simple),
        ("original-simple", restored, simple),
        ("hybrid-original", hybrid, restored),
        ("hybrid-full", hybrid, full),
        ("hybrid-E", hybrid, e),
        ("hybrid-C", hybrid, single_c),
        ("original-E", restored, e),
        ("original-C", restored, single_c),
    ]:
        d = vals(a) - vals(b)
        if name.endswith("-simple"):
            keep = np.zeros(len(groups), dtype=bool)
            keep[affected] = True
            assert not d[~keep].any()
        print(" ", name, interval(d, groups, seed))
    print(flush=True)


if __name__ == "__main__":
    ok, rows = fq.check_bank()
    assert ok
    for dataset in ("TruthfulQA", "FalseQA"):
        for model in tq.ORDER:
            analyze(dataset, model, rows)
