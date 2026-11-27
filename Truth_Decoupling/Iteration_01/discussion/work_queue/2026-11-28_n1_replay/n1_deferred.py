"""Count generation calls for the frozen N1 rule and verify answer text and submission flags item by item."""

"""Audit archived question-level outputs of N1's exact lazy schedules; no generation or judging."""
import sys
from collections import Counter

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
]
import falseqa_read as F
import tqa_gate2b_1127 as T
import release_origin as RO


def audit(ids, baseline, e, arms, caps, canonical):
    ce, co = caps
    grid = list(arms)
    visits = Counter()
    for q in ids:
        ef, et = e(q)
        of, ot = baseline(q)
        exists = []
        dose = None
        n = 0
        emitted = (not ef, et)
        if ef:
            cap = co if of else ce
            for d in grid[: grid.index(cap) + 1] if cap is not None else []:
                n += 1
                b, txt = arms[d](q)
                if not b:
                    dose = d
                    if not of and RO.norm(txt) == RO.norm(ot):
                        visits["veto"] += 1
                    else:
                        emitted = (True, txt)
                    break
        assert emitted == canonical(q), q
        visits["N"] += 1
        visits["E_answered"] += int(not ef)
        visits["E_blank"] += int(ef)
        visits["C_calls"] += n
        visits["lazy_O_saved_extra"] += int(
            ef and (ce == co and (ce is None or dose is None))
        )
        visits["source_skip_induced"] += int(ef and not of and ce is None)
        visits["filled"] += int(ef and emitted[0])
    return visits


def main():
    ok, bank = F.check_bank()
    assert ok
    D = {r["id"] for r in bank if r["set"] == "dev"}
    I = {r["id"] for r in bank if r["set"] == "heldout"}
    for ds in ["TQA", "FQA"]:
        joint = Counter()
        for m in F.ORDER:
            if ds == "TQA":
                b = T.base(m)
                ids = b.ids
                sp, tx = b.sp, b.tx

                def make(c):
                    return lambda q: (sp.blank[(q, c)], tx.ec[(q, c)])

                e = make(sp.cE(b.sd))
                base = make("baseline_x")
                arms = {d: make(sp.cEC(b.sd, d)) for d in T.LAD}
                caps = (b.cap_e, b.cap_own)
                out = T.S.system(sp, tx, b.pick)
                canon = {
                    q: (bool(a), rec["text"]) for q, a, rec in zip(ids, out[1], out[2])
                }
            else:
                dev, dev2 = F.load_rows(m, D)
                raw, _ = F.load_rows(m, I)
                ids = raw.ids
                st = F.setting(m)
                caps = F.choose(dev, dev2, st)[0]["N1"]
                s = st[1]

                def make(c):
                    return lambda q: (raw.blank(c, q), raw.text(c, q))

                base = make("x")
                e = make(raw.e(s))
                arms = {d: make(raw.ec(s, d)) for d in F.LAD}
                canon = {
                    q: (a, t)
                    for q, (_, a, t, _) in F.n1_cells(raw, s, caps, ids).items()
                }
            v = audit(ids, base, e, arms, caps, lambda q: canon[q])
            joint.update(v)
            print(ds, m, caps, dict(v), flush=True)
        old = 2 * joint["N"] + joint["C_calls"]
        fast = (
            joint["N"]
            + joint["E_blank"]
            + joint["C_calls"]
            - joint["lazy_O_saved_extra"]
        )
        print(
            "TOTAL",
            ds,
            dict(joint),
            "queries from all-base-then-E-plus-ladder",
            old,
            "lazy exact",
            fast,
            "queries saved",
            old - fast,
            flush=True,
        )


main()
