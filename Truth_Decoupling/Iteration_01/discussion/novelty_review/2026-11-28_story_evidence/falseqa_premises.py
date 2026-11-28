"""Decompose the natural score improvement on actual, labeled FalseQA test pairs.
Premise labels enter only this analysis after all method outputs have been selected.
"""

import sys
import numpy as np

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
]
import falseqa_read as F
import tqa_gate2b_1127 as T


def main():
    ok, bank = F.check_bank()
    assert ok
    bank = [r for r in bank if r["set"] == "heldout"]
    ids = [r["id"] for r in bank]
    prem = {r["id"]: r["premise"] for r in bank}
    pairs = {r["id"]: r["pair_id"] for r in bank}
    assert len(ids) == 600 and len(set(pairs.values())) == 300
    for m in T.ORDER:
        raw, _ = F.load_rows(m, set(ids))
        s = F.setting(m)[1]
        ar = [raw.e(s), raw.ec(s, "1"), raw.c("1"), "x"]

        def record(q, a):
            issued = not raw.blank(a, q)
            return (raw.cell(a, q) if issued else 0), issued

        for tag in ("false", "valid"):
            records = []
            for q in ids:
                if prem[q] != tag:
                    continue
                e, ec, c, o = (record(q, a) for a in ar)
                eligible = not o[1] and not e[1]
                fixed = ec if eligible else e
                records.append(
                    (
                        q,
                        eligible,
                        fixed[0],
                        e[0],
                        o[0],
                        c[0],
                        ec[0],
                        int(eligible and ec[1]),
                    )
                )
            assert len(records) == 300
            d = {q: v - e for q, _, v, e, *_ in records}
            ref = {q: e - o for q, _, v, e, o, *_ in records}
            right = sum(eligible and v == 1 for q, eligible, v, *_ in records)
            wrong = sum(eligible and v == -1 for q, eligible, v, *_ in records)
            assert right - wrong == sum(d.values())
            print(
                m,
                tag,
                "original to E / E to fixed; double blanks / fills / R / W",
                sum(ref.values()),
                F.pair_ci(d, pairs, list(d)),
                sum(r[1] for r in records),
                sum(r[-1] for r in records),
                right,
                wrong,
                flush=True,
            )


if __name__ == "__main__":
    main()
