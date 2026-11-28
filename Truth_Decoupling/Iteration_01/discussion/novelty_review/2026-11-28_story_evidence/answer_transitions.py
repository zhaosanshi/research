"""Archive-only review of answer transitions and conditional fills for the fixed double-abstention EC1 policy.
Run from the original Truth_Decoupling data root in the scoring container. No generation, verifier call, or judgment. Only aggregate counts are emitted; no question or response text.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
    "Iteration_01/gpt_tools_20261126/gate_audit",
]
import falseqa_read as F
import tqa_gate2b_1127 as T
import tq_same_n_decomp as SF
import simple_ec_1127 as SC
from judge_two_axis_run import load_verified_single


def summarize_risk(records, models, dataset):
    import numpy as np
    from collections import defaultdict

    pooled = defaultdict(lambda: np.zeros(4, dtype=np.int32))
    strata = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=np.int32)))

    def unit(q):
        return q if dataset == "TruthfulQA" else q[:-1]

    def get_ci(cluster):
        ordered = sorted(cluster)
        matrix = np.stack([cluster[q] for q in ordered]).astype(np.int32)
        arr = matrix.sum(axis=0)
        n = len(ordered)
        random = np.random.default_rng(20261128)
        draws = random.integers(n, size=(12000, n))
        counts = matrix[draws].sum(axis=1)

        # [suppressed wrong, suppressed right, continued wrong, continued right]
        def rate(x):
            return x[..., 0] / (x[..., 0] + x[..., 1]), x[..., 2] / (
                x[..., 2] + x[..., 3]
            )

        u, v = rate(arr)
        U, V = rate(counts)
        assert np.isfinite(U - V).all()
        ci = np.percentile(100 * (U - V), [2.5, 97.5], method="inverted_cdf")
        return {
            "unit_count": n,
            "counts": arr.tolist(),
            "error_rates_pct": np.round([100 * u, 100 * v], 4).tolist(),
            "gap_pp": round(100 * (u - v), 4),
            "ci95_pp": np.round(ci, 4).tolist(),
        }

    for m in models:
        by_unit = defaultdict(lambda: np.zeros(4, dtype=np.int32))
        for q, row in records[m].items():
            orig, e = row["O"], row["E"]
            uq = unit(q)
            # Sample the entire test question roster as blocks, not each model-question independently.
            x = by_unit[uq]
            p = pooled[uq]
            st = strata[q[-1] if dataset == "FalseQA" else "all"][uq]
            if orig[1] and orig[0] in ("W", "R"):
                j = (2 if e[1] else 0) + (0 if orig[0] == "W" else 1)
                x[j] += 1
                p[j] += 1
                st[j] += 1
        print("RISK", dataset, m, json.dumps(get_ci(by_unit)), flush=True)
    print("POOLED_RISK", dataset, json.dumps(get_ci(pooled)), flush=True)
    if dataset == "FalseQA":
        for t, c in strata.items():
            print(
                "STRATIFIED_POOLED_RISK", dataset, t, json.dumps(get_ci(c)), flush=True
            )


def summarize_control(records, models, dataset):
    import numpy as np
    from collections import defaultdict

    score = {"R": 1, "W": -1, "H": 0, "X": 0, "D": 0}
    summed = Counter()
    for m in models:
        dd = defaultdict(dict)
        pair = {}
        ct = Counter()
        for q, r in records[m].items():
            g = q if dataset == "TruthfulQA" else q[:-1]
            pair[q] = g
            o, e, ec, c, cd, te = [r[k] for k in ("O", "E", "EC", "C1", "Cdev", "TE")]
            dbl = not o[1] and not e[1]
            if dbl:
                ct["BOTH_BLANK"] += 1
            for name, b in [("C1", c), ("Cdev", cd), ("TE", te)]:
                d = 0
                if dbl:
                    d = score[ec[0]] * int(ec[1]) - score[b[0]] * int(b[1])
                    if ec[1] and b[1]:
                        ct[name + "_overlap"] += 1
                        ct[name + "_overlap_net"] += score[ec[0]] - score[b[0]]
                        if ec[0] == "R" and b[0] == "W":
                            ct[name + "_WR"] += 1
                        if ec[0] == "W" and b[0] == "R":
                            ct[name + "_RW"] += 1
                dd[name][q] = d
        summed.update(ct)
        units = list(dict.fromkeys(pair.values()))
        groups = {u: i for i, u in enumerate(units)}
        n = len(units)
        weights = np.random.default_rng(20261128).integers(n, size=(5000, n))
        res = {}
        for k in ("C1", "Cdev", "TE"):
            val = np.zeros(n, dtype=int)
            for q, d in dd[k].items():
                val[groups[pair[q]]] += d
            res[k] = [
                int(val.sum()),
                *np.percentile(
                    val[weights].sum(axis=1), [2.5, 97.5], method="inverted_cdf"
                )
                .astype(int)
                .tolist(),
            ]
        print("COND", dataset, m, dict(ct), res, flush=True)
    print("COND TOTAL", dataset, dict(summed), flush=True)


def count_block(records, models, dataset):
    all_trans = Counter()
    for m in models:
        rows = records[m]
        stats = Counter()
        for q, r in rows.items():
            orig, e, ec, c1, cd, forced = [
                r[k] for k in ("O", "E", "EC", "C1", "Cdev", "TE")
            ]
            stats["N"] += 1
            o, ev = orig[1], e[1]
            if o and ev:
                stats["both_answered"] += 1
                stats["both_answered_%s_%s" % (orig[0], e[0])] += 1
                if orig[0] in ("R", "W") and e[0] in ("R", "W") and orig[0] != e[0]:
                    if orig[2].strip() == e[2].strip():
                        stats[
                            "unchanged_with_verdict_flip_" + orig[0] + "_" + e[0]
                        ] += 1
            if o and not ev:
                stats["O_answered_E_blank"] += 1
                stats["O_to_E_blank_" + orig[0]] += 1
            if not o and not ev:
                stats["both_blank"] += 1
                for arm, d in [("EC", ec), ("C1", c1), ("Cdev", cd), ("TE", forced)]:
                    j, a, txt = d
                    cat = j if a else "not_submitted"
                    stats["fill_" + arm + "_" + cat] += 1
        print("STATS", dataset, m, json.dumps(stats, sort_keys=True), flush=True)
        all_trans.update(stats)
    print("POOLED", dataset, json.dumps(all_trans, sort_keys=True), flush=True)


def parse_falseqa_judges(model, R):
    labels = {}
    for w in F.WORKS:
        work = Path(F.NSD) / "judge" / "falseqa" / model / w
        _, cells, votes = load_verified_single(
            str(work),
            "pro",
            generation_path=str(work / "judge_input.json"),
            require_complete=False,
        )
        for key, item in cells.items():
            v = votes[key]
            q, c = item["pair_id"].split(":", 1)[1], item["condition"]
            label = (
                {"A": "R", "B": "W", "H": "H", "C": "X"}.get(v.get("label"))
                if v["status"] == "scored"
                else ("X" if v.get("basis") == "empty_output_no_judge_vote" else "D")
            )
            if (q, c) in labels:
                assert labels[q, c] == label
            labels[q, c] = label
    return labels


def main():
    ok, bank = F.check_bank()
    assert ok
    for dataset in ("TruthfulQA", "FalseQA"):
        allrows = {}
        for m in F.ORDER:
            if dataset == "TruthfulQA":
                selected = SC.tqa_split(m, 0)
                lad = selected[2]
                dc = SC.pick({k: sum(selected[1]["C" + k]) for k in lad}, lad)
                b = T.base(m)
                ids = b.ids
                v = json.loads((Path(b.run) / "frozen_test.json").read_text())[
                    "questions"
                ]
                questions = {x["pair_id"]: x["question"] for x in v}
                display, layer, s = SF.A.KEY[m]
                r, dat = SF.load_model(m)
                assert ids == r
                run = dat[4]
                work = SF.MP.work(run, "test", f"pospromptdl{layer}")
                st, bl = SF.ps.load(work, "pro")
                ft = SF.MP.fr.raw_texts(work)
                fcond = f"E_L{layer}_conf-w-p{s}"
                cs = {
                    "O": "baseline_x",
                    "E": b.sp.cE(b.sd),
                    "EC": b.sp.cEC(b.sd, "1"),
                    "C1": b.sp.cC("1"),
                    "Cdev": b.sp.cC(dc),
                }

                def get(q, n):
                    if n == "TE":
                        return st[q, fcond], not bl[q, fcond], ft[q, fcond]
                    c = cs[n]
                    return b.sp.st[q, c], not b.sp.blank[q, c], b.tx.ec[q, c]
            else:
                assert all(r["id"][:-1] == r["pair_id"] for r in bank)
                H = {r["id"] for r in bank if r["set"] == "heldout"}
                R, _ = F.load_rows(m, H)
                ids = R.ids
                questions = {x["id"]: x["question"] for x in bank}
                selected = SC.fqa_split(m, bank, "dev")
                lad = selected[2]
                dc = SC.pick({k: sum(selected[1]["C" + k]) for k in lad}, lad)
                roster = F.roster(m)
                labels = parse_falseqa_judges(m, R)
                cs = {
                    "O": roster["O"],
                    "E": roster["E"],
                    "EC": roster["EC1"],
                    "C1": roster["C1"],
                    "Cdev": roster["C" + dc],
                    "TE": roster["TE"],
                }

                def get(q, n):
                    a = cs[n]
                    c = R.cond[a]
                    lab = labels[q, c]
                    assert {"R": 1, "W": -1, "H": 0, "X": 0, "D": 0}[lab] == R.cell(
                        a, q
                    ), (m, q, c)
                    return lab, not R.blank(a, q), R.text(a, q)

            allrows[m] = {
                q: {
                    "question": questions[q],
                    **{k: get(q, k) for k in ("O", "E", "EC", "C1", "Cdev", "TE")},
                }
                for q in ids
            }
            print("LOADED", dataset, m, len(ids), "Cdev", dc, flush=True)
        count_block(allrows, F.ORDER, dataset)
        summarize_risk(allrows, F.ORDER, dataset)
        summarize_control(allrows, F.ORDER, dataset)


if __name__ == "__main__":
    main()
