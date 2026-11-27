"""Point identification or bounds for the natural-count gate effect at fixed HIGH dose using archived Pro verdicts only. No new ratings. Also bound the original Llama equal-coverage high-minus-main read."""

import sys
import collections
from pathlib import Path
import numpy as np

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
    "Iteration_01/gpt_tools_20261126/gate_audit",
]
import falseqa_gate2_1127 as H
import table_gate2_1127 as T
import q100_gate2_1127 as QH
from judge_two_axis_run import load_verified_single, judge_request_keys, load_cfg


def main():
    bank = H.bank_rows()
    md = {r["id"]: r for r in bank}
    held = {r["id"] for r in bank if r["set"] == "heldout"}
    pair = {r["id"]: r["pair_id"] for r in bank}
    cfg = load_cfg()
    votes = collections.defaultdict(set)
    attrs = {}
    auth = {}
    keys = {}
    count = collections.Counter()
    for m in H.ORDER:
        # Separate full or subset direct scores for high N1 from the supplementary identical-request matches.
        auth[m] = {}
        root = Path(H.NSD) / "judge" / "falseqa" / m
        for w in (
            *[root / n for n in H.F.WORKS],
            H.work(m, "full"),
            Path(QH.work_a(m, "q100")),
        ):
            if not (w / "judge2" / "single_pro.jsonl").exists():
                continue
            high = w.name not in H.F.WORKS
            jin = w / (H.JIN if high else "judge_input.json")
            meta, cells, vals = load_verified_single(
                str(w), "pro", generation_path=str(jin), require_complete=False
            )
            for idx, c in cells.items():
                q = c["pair_id"].split(":", 1)[1]
                txt = c["prediction"]
                if q not in held or not txt.strip():
                    continue
                item = {n: c[n] for n in ["question", "gold_refs", "incorrect"]}
                assert item["question"] == md[q]["question"]
                assert q not in attrs or attrs[q] == item
                attrs[q] = item
                if (q, txt) not in keys:
                    keys[q, txt] = tuple(judge_request_keys(c, "pro", cfg))
                vv = vals[idx]
                if vv["status"] == "scored":
                    v = H.Q.VALUE[vv["label"]]
                elif vv["basis"] == "pro_declined":
                    v = 0
                else:
                    raise ValueError(vv["status"])
                k = keys[q, txt]
                votes[k].add(v)
                count["nonempty"] += 1
                if high:
                    sk = (q, c["condition"])
                    assert sk not in auth[m] or auth[m][sk] == (txt, v)
                    auth[m][sk] = (txt, v)
    assert len(attrs) == 600
    print(
        "archives: nonempty cells",
        count["nonempty"],
        "unique exact texts",
        len(keys),
        "conflicting request-pairs",
        sum(len(v) > 1 for v in votes.values()),
        flush=True,
    )
    for m in H.ORDER:
        B = H.base_state(m, bank)
        on = set(B.on)
        off = set(B.ids) - on
        raw, conditions, want = H.load_new(m, "full", bank)
        assert (
            len(want) == len(set(want)) == 700
            and set(want) == set(md)
            and all(set(rr) == set(md) for rr in raw.values())
        )
        ss = H.F.systems_of(B.R, B.R2, B.st, B.caps, B.ids)

        def cell(arm, q):
            r = raw[arm][q]
            return 0, not H.blank_row(r), r["text"]

        hi = T.n1_rebuild(
            B.ids,
            lambda q: cell("E2", q),
            lambda q: B.R.blank("x", q),
            lambda q: B.R.text("x", q),
            lambda k, q: cell("EC2_" + k, q),
            B.caps["N1"],
        )
        vals = {}
        reason = collections.Counter()
        direct = auth[m]
        for q in B.ids:
            _, issued, text, rung = hi[q]
            arm = "E2" if rung in (None, "0") else "EC2_" + rung
            if not text.strip():
                res = (0, 0)
                src = "empty"
            elif (q, conditions[arm]) in direct:
                t, s = direct[q, conditions[arm]]
                assert t == text
                res = (s, s)
                src = "official"
            else:
                rq = tuple(
                    judge_request_keys(dict(attrs[q], prediction=text), "pro", cfg)
                )
                seen = votes.get(rq, set())
                if len(seen) == 1:
                    s = next(iter(seen))
                    res = (s, s)
                    src = "same request"
                else:
                    res = (-1, 1)
                    src = "conflict" if seen else "absent"
            vals[q] = res
            reason[q in on, src] += 1
        o = ss["O"]
        known = sum(vals[q][0] == vals[q][1] for q in off)
        lo = sum(o[q][0] - vals[q][1] for q in off)
        hi_bound = sum(o[q][0] - vals[q][0] for q in off)
        print(
            m,
            "six sets of ids",
            [len(xx) for xx in raw.values()],
            "off",
            len(off),
            "identified",
            known,
            "missing",
            len(off) - known,
            "natural fixed-dose gated-minus-always HIGH bounds",
            lo,
            hi_bound,
            "source counts",
            dict(reason),
            flush=True,
        )
        # Equalize submissions only among gate-OFF questions. Both arms return the same high response on all gate-ON questions, which are locked.
        mg = T.margins(m)
        a_issued = [q for q in B.ids if q in off and o[q][1]]
        b_issued = [q for q in B.ids if q in off and hi[q][1]]
        diff = len(a_issued) - len(b_issued)
        elig = a_issued if diff > 0 else b_issued
        larger = o if diff > 0 else hi
        sc = {q: H.F.margin(mg, md[q]["question"], larger[q]) for q in elig}
        missing = [q for q in elig if sc[q] is None or not np.isfinite(sc[q])]
        lo_nat = {q: (o[q][0] - vals[q][1] if q in off else 0) for q in B.ids}
        hi_nat = {q: (o[q][0] - vals[q][0] if q in off else 0) for q in B.ids}
        print(
            m,
            "paired-bootstrap lower envelope natural (lower at pessimal; upper at optimal)",
            H.F.pair_ci(lo_nat, pair, B.ids)[1],
            H.F.pair_ci(hi_nat, pair, B.ids)[2],
            flush=True,
        )
        if not missing or diff == 0:
            drop = (
                set(sorted(elig, key=lambda q: sc[q])[: abs(diff)]) if diff else set()
            )
            dl = {}
            du = {}
            for q in B.ids:
                if q not in off:
                    dl[q] = du[q] = 0
                    continue
                s = o[q][0] if not (diff > 0 and q in drop) else 0
                l, h = vals[q] if not (diff < 0 and q in drop) else (0, 0)
                dl[q] = s - h
                du[q] = s - l
            a = H.F.pair_ci(dl, pair, B.ids)
            b = H.F.pair_ci(du, pair, B.ids)
            print(
                m,
                "equal-coverage OFF, natural submissions gate vs always, exact trim",
                len(a_issued),
                len(b_issued),
                "point bounds",
                a[0],
                b[0],
                "95% envelope",
                a[1],
                b[2],
                flush=True,
            )
        else:
            print(
                m,
                "equal-coverage OFF, natural submissions gate vs always, missing ranks",
                len(a_issued),
                len(b_issued),
                len(missing),
                "conservative worst-case trim bounds",
                lo - abs(diff),
                hi_bound + abs(diff),
                flush=True,
            )
        if m == "llama3.1-8b":
            mg = T.margins(m)
            low = ss["N1"]
            live = [q for q in B.ids if q in on]
            c_hi = {q for q in live if hi[q][1]}
            c_lo = {q for q in live if low[q][1]}
            quota = min(len(c_hi), len(c_lo))
            drops = []
            for sysd, issued in [(hi, c_hi), (low, c_lo)]:
                eligible = [q for q in live if q in issued]
                sc = {q: H.F.margin(mg, md[q]["question"], sysd[q]) for q in eligible}
                assert all(np.isfinite(v) for v in sc.values())
                drops.append(
                    set(sorted(eligible, key=lambda q: sc[q])[: len(eligible) - quota])
                )
            # Match returns stored scores of non-issued but non-empty straddling judge labels, just like FM.rankcut.
            low_d = {}
            hi_d = {}
            unk = 0
            for q in B.ids:
                if q not in on:
                    low_d[q] = hi_d[q] = 0
                    continue
                b = low[q][0] if q not in drops[1] else 0
                l, h = vals[q] if q not in drops[0] else (0, 0)
                low_d[q] = l - b
                hi_d[q] = h - b
                unk += int(h != l)
            lower = H.F.pair_ci(low_d, pair, B.ids)
            upper = H.F.pair_ci(hi_d, pair, B.ids)
            print(
                "LLAMA full 600, original primary with reopened archival matches",
                "unscored final",
                unk,
                "full possible point",
                lower[0],
                upper[0],
                "bootstrap lower endpoint at both extreme completions",
                lower[1],
                upper[1],
                flush=True,
            )


if __name__ == "__main__":
    main()
