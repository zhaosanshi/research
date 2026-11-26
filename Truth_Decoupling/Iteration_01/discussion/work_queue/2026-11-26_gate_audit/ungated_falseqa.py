"""Three-way equal-coverage audit, CPU only, Truth_Decoupling data root. No model generation or judging."""

import sys

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
]
import falseqa_read as F
from collections import defaultdict


def main():
    ok, bank = F.check_bank()
    assert ok
    train = {r["id"] for r in bank if r["set"] == "dev"}
    held = [r["id"] for r in bank if r["set"] == "heldout"]
    assert (
        len(train) == 100
        and len(held) == len(set(held)) == 600
        and train.isdisjoint(held)
    )
    qtext = {r["id"]: r["question"] for r in bank}
    pair = {r["id"]: r["pair_id"] for r in bank}
    total = defaultdict(lambda: defaultdict(int))
    all_scores = defaultdict(int)
    for m in F.ORDER:
        st = F.setting(m)
        r, r2 = F.load_rows(m, train)
        caps, _ = F.choose(r, r2, st)
        t, t2 = F.load_rows(m, set(held))
        D = F.systems_of(t, t2, st, caps, t.ids)
        M = F.margins_of(m)
        systems = ("N1", "E", "Conly")
        target = min(sum(D[k][q][1] for q in held) for k in systems)
        scores = {}
        for k in systems:
            A = D[k]
            scored = [
                (F.margin(M, qtext[q], A[q]), i, q)
                for i, q in enumerate(held)
                if A[q][1]
            ]
            assert all(s is not None for s, i, q in scored)
            omit = {q for s, i, q in sorted(scored)[: len(scored) - target]}
            scores[k] = {q: 0 if q in omit else A[q][0] for q in held}
            all_scores[k] += sum(scores[k].values())
        print(
            m,
            "quota",
            target,
            "scores",
            [(k, sum(scores[k].values())) for k in systems],
            flush=True,
        )
        for k in systems[1:]:
            d = {q: scores["N1"][q] - scores[k][q] for q in held}
            for q in held:
                total[k][q] += d[q]
            print(k, F.pair_ci(d, pair, held), flush=True)
    print(
        dict(all_scores),
        [(k, F.pair_ci(v, pair, held)) for k, v in total.items()],
        flush=True,
    )


if __name__ == "__main__":
    main()
