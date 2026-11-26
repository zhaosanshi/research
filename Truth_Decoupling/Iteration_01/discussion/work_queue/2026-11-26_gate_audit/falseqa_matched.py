"""Archive-only diagnostic of plan P, Pool B / chosen layer frozen. In grading container, from Truth_Decoupling.
Same gate on all competitors; preserve its OFF branch, match counts only within ON. Rank by baseline singleton verifier.
For two-pass E, keep every first E submission and fill the same number of its remaining blanks with available TE answers.
"""

import sys
import json
import hashlib
import math

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
    "Iteration_01/code/scripts/capability",
]
import falseqa_read as F
import gate_sim_1126 as G


def verify_gate(m, task, ids):
    root = G.OUT / m
    fit = json.loads((root / "fit_B_gate_1126.json").read_bytes())
    cap = root / "capture_gate_1126.json"
    if (
        fit["model"] != m
        or fit["pool"] != "B"
        or fit["chosen_layer"] != fit["refit"]["chosen"]["layer"]
        or hashlib.sha256(cap.read_bytes()).hexdigest() != fit["capture_sha256"]
    ):
        raise ValueError("wrong gate source")
    gg = fit["probs"]["chosen"][task]
    if not set(ids).issubset(gg) or any(not 0 <= gg[q] <= 1 for q in ids):
        raise ValueError("invalid probability bank")
    return [q for q in ids if gg[q] > 0.5]


def ci(x, ids, pair):
    return F.pair_ci(dict(zip(ids, x)), pair, ids) if pair else G.tq_ci(x)


def rankcut(ids, sys, k, scores):
    valid = [q for q in ids if sys[q][1]]
    if len(valid) < k:
        raise ValueError("target is unreachable")
    if len(valid) == k:
        return {q: sys[q][0] for q in ids}
    if any(scores[q] is None or not math.isfinite(scores[q]) for q in valid):
        return None
    drop = set(sorted(valid, key=lambda q: scores[q])[: len(valid) - k])
    return {q: (0 if q in drop else sys[q][0]) for q in ids}


def evaluate(m, task, ids, sys, qt, margin, group, info=None, pair=None):
    on = verify_gate(m, task, ids)
    opened = set(on)
    ss = {
        n: {q: (sys[n][q] if q in opened else sys["O"][q]) for q in ids}
        for n in ("N1", "E", "Conly")
    }
    scores = lambda n: {q: F.margin(margin, qt[q], sys[n][q]) for q in on}
    natt = {
        k: (sum(c[0] for c in v.values()), sum(c[1] for c in v.values()))
        for k, v in ss.items()
    }
    print(task, m, "ON", len(on), "natural (net, issued)", natt, flush=True)
    out = {}
    for arms in [("N1", "E", "Conly"), ("N1", "Conly"), ("N1", "E")]:
        k = min(sum(sys[s][q][1] for q in on) for s in arms)
        vals = {s: rankcut(on, sys[s], k, scores(s)) for s in arms}
        if any(r is None for r in vals.values()):
            print("missing scores", arms, flush=True)
            continue
        fullk = k + sum(sys["O"][q][1] for q in ids if q not in opened)
        for n in arms[1:]:
            vec = [vals["N1"][q] - vals[n][q] if q in opened else 0 for q in ids]
            print(" counts", arms, "n", fullk, "vs", n, ci(vec, ids, pair), flush=True)
            out["total_arms_" + str(arms) + "_" + n] = vec
    if info and pair:
        for typ in ["false", "valid"]:
            z = [q for q in ids if info[q] == typ]
            on_typ = set(z) & opened
            base = sum(sys["N1"][q][0] - sys["O"][q][0] for q in z)
            actual = sum(sys["N1"][q][0] - sys["O"][q][0] for q in on_typ)
            print(
                " vs random within",
                typ,
                "at same gate-open count:",
                actual - base * len(on_typ) / len(z),
                flush=True,
            )
    for strat in (False, True):
        base = {q: sys["E"][q][0] for q in on}
        deficit = []
        take = set()
        eb = [q for q in on if not sys["E"][q][1]]
        if any(sys["E"][q][:3] != sys["N1"][q][:3] for q in on if sys["E"][q][1]):
            raise ValueError("changed E")
        for own in [0, 1] if strat else [None]:
            pool = [q for q in eb if own is None or int(not sys["O"][q][1]) == own]
            quota = sum(sys["N1"][q][1] for q in pool)

            def passes(q):
                return sys["TE"][q][1] and (
                    not strat
                    or not sys["O"][q][1]
                    or F.Q.ro.norm(sys["O"][q][2]) != F.Q.ro.norm(sys["TE"][q][2])
                )

            eligible = [q for q in pool if passes(q)]
            sc = {q: F.margin(margin, qt[q], sys["TE"][q]) for q in eligible}
            if len(eligible) < quota or any(v is None for v in sc.values()):
                deficit.append(
                    (
                        own,
                        quota,
                        len(eligible),
                        "missing",
                        sum(v is None for v in sc.values()),
                    )
                )
            else:
                take.update(sorted(eligible, key=lambda q: -sc[q])[:quota])
        if deficit:
            print("two-pass TE", strat, "unavailable", deficit, flush=True)
            continue
        vec = [
            sys["N1"][q][0] - (sys["TE"][q][0] if q in take else base[q])
            if q in opened
            else 0
            for q in ids
        ]
        print(
            " gate-shared two-pass TE",
            strat,
            "kept supplementary",
            len(take),
            "net difference",
            ci(vec, ids, pair),
            flush=True,
        )
        out["two_" + str(strat)] = vec
        raw = [
            (q, sys["N1"][q], sys["TE"][q] if q in take else sys["E"][q]) for q in on
        ]
        conflicts = {
            q for q, a, b in raw if a[0] != b[0] and a[2].strip() == b[2].strip()
        }
        vec0 = [0 if q in conflicts else v for q, v in zip(ids, vec)]
        print(
            "identical-text conflicts count, delta:",
            len(conflicts),
            sum(vec) - sum(vec0),
            flush=True,
        )
        out["two_exact_text_" + str(strat)] = vec0
    return out


def main():
    _, bank = F.check_bank()
    dev = {r["id"] for r in bank if r["set"] == "dev"}
    held = {r["id"] for r in bank if r["set"] == "heldout"}
    qt = {r["id"]: r["question"] for r in bank}
    pair = {r["id"]: r["pair_id"] for r in bank}
    info = {r["id"]: r["premise"] for r in bank}
    allvals = []
    for m in G.ORDER:
        st = F.setting(m)
        tr, tr2 = F.load_rows(m, dev)
        caps, _ = F.choose(tr, tr2, st)
        r, r2 = F.load_rows(m, held)
        ids = r.ids
        sys = F.systems_of(r, r2, st, caps, ids)
        if (
            tuple(sum(sys[k][q][0] for q in ids) for k in ("O", "E", "N1", "Conly"))
            != G.K_FQA[m]
        ):
            raise ValueError("judgments differ")
        vals = evaluate(m, "fqa_held", ids, sys, qt, F.margins_of(m), None, info, pair)
        allvals.append(vals)
    for k in set.intersection(*(set(v) for v in allvals)):
        v = [sum(allvals[j][k][i] for j in range(5)) for i in range(len(ids))]
        print("FalseQA five models", k, ci(v, ids, pair))


if __name__ == "__main__":
    main()
