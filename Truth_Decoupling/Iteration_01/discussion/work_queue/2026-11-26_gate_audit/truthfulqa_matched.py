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
import selfjudge_probe as SP
import selfjudge_stack_pick as SK
import tq_same_n_decomp as TQ


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
    out = {}
    getv = lambda s, q: sys[s][q][0] if q in opened else sys["O"][q][0]
    getn = lambda s, q: sys[s][q][1] if q in opened else sys["O"][q][1]
    single_count = min(sum(sys[arm][q][1] for q in on) for arm in ("E", "N1"))
    matched = {
        arm: rankcut(
            on,
            sys[arm],
            single_count,
            {q: F.margin(margin, qt[q], sys[arm][q]) for q in on},
        )
        for arm in ("E", "N1")
    }
    if any(v is None for v in matched.values()):
        raise ValueError("missing ranking for single E comparison")
    out["single_E"] = [
        matched["N1"][q] - matched["E"][q] if q in opened else 0 for q in ids
    ]
    for arms in [("N1", "Conly"), ("N1", "E", "Conly")]:
        k = min(sum(getn(s, q) for q in ids) for s in arms)
        kn = k - sum(getn("O", q) for q in ids if q not in opened)
        n1 = rankcut(
            on, sys["N1"], kn, {q: F.margin(margin, qt[q], sys["N1"][q]) for q in on}
        )
        assert n1 is not None
        nvalid = sum(sys["Conly"][q][1] for q in on)
        ndrop = nvalid - kn
        # if C margins unavailable, all retention policies that leave the same number in ON; one worst case each way.
        for label, sgn in [("lower_bound", 1), ("upper_bound", -1)]:
            cand = [q for q in on if sys["Conly"][q][1]]
            actual = all(
                F.margin(margin, qt[q], sys["Conly"][q]) is not None for q in cand
            )
            # for truly missing C scores, use the gold labels ONLY for the pessimistic/optimistic bound; never a method.
            drop = set(
                sorted(
                    cand,
                    key=lambda q: (
                        F.margin(margin, qt[q], sys["Conly"][q])
                        if actual
                        else sgn * sys["Conly"][q][0]
                    ),
                )[:ndrop]
            )
            vec = [
                n1[q] - (0 if q in drop else sys["Conly"][q][0]) if q in opened else 0
                for q in ids
            ]
            out[str(arms) + "_" + label] = vec
            print(m, "C", arms, k, ndrop, actual, label, sum(vec), flush=True)
    base = {q: sys["E"][q][0] for q in on}
    eb = [q for q in on if not sys["E"][q][1]]
    if any(sys["E"][q][:3] != sys["N1"][q][:3] for q in on if sys["E"][q][1]):
        raise ValueError("changed E")
    for selector in ["singleton", "worst_drop", "best_drop"]:
        keep = set()
        take = set()
        for own in [False, True]:
            pool = [q for q in eb if (not sys["O"][q][1]) == own]
            rel = [q for q in pool if sys["N1"][q][1]]
            eligible = [
                q
                for q in pool
                if sys["TE"][q][1]
                and (own or F.Q.ro.norm(sys["O"][q][2]) != F.Q.ro.norm(sys["TE"][q][2]))
            ]
            quota = min(len(eligible), len(rel))
            raw = {q: F.margin(margin, qt[q], sys["N1"][q]) for q in rel}
            if len(rel) > quota and any(v is None for v in raw.values()):
                raise ValueError("missing")
            if len(rel) > quota:
                rel = sorted(
                    rel,
                    key=lambda q: (
                        -raw[q]
                        if selector == "singleton"
                        else (1 if selector == "worst_drop" else -1) * sys["N1"][q][0]
                    ),
                )[:quota]
            take.update(sorted(eligible, key=lambda q: -sys["TE"][q][3])[:quota])
            keep.update(rel)

        def score(q):
            if q not in opened:
                return 0
            a = (
                sys["N1"][q][0]
                if (sys["E"][q][1] or q in keep or not sys["N1"][q][1])
                else 0
            )
            b = sys["TE"][q][0] if q in take else base[q]
            return a - b

        vec = [score(q) for q in ids]
        out["strict_TE_" + selector] = vec

        def literal(q):
            if q not in opened:
                return False
            n1 = (
                sys["N1"][q]
                if (sys["E"][q][1] or q in keep or not sys["N1"][q][1])
                else (0, False, "", None)
            )
            te = sys["TE"][q] if q in take else sys["E"][q]
            return n1[0] != te[0] and n1[2].strip() == te[2].strip()

        out["strict_TE_exact_" + selector] = [
            0 if literal(q) else v for q, v in zip(ids, vec)
        ]
        print(
            "Gated 2passE",
            m,
            selector,
            "full K=",
            len(keep) + sum(getn("E", q) for q in ids),
            "kept",
            len(keep),
            sum(vec),
            flush=True,
        )
    return out


def truth_data(m):
    display, layer, s = TQ.A.KEY[m]
    ids, dat = TQ.load_model(m)
    _, _, _, cfg, run, e, ea, n1, issued, forced = dat
    (sp, tx, singles, ids_x), err = TQ.MP.prompt_split(display, layer, 1, s, qs=ids)
    assert not err and list(sp.qs) == ids
    (tsp, tt, tsing, id_t), err = TQ.MP.prompt_split(display, layer, 0, s)
    pick = TQ.ML.n1_pick(tsp, tt, s)
    sts, ans, recs = TQ.S.system(sp, tx, pick)[:3]
    if tuple(pick) != tuple(cfg) or any(r["p"] != q for q, r in zip(ids, recs)):
        raise ValueError("N1 choice")
    cref = TQ.MP.usual(display, 1)[0].m["c"]
    ks = [k for k in TQ.ps.LAD if k in tsing]
    ck = max(ks, key=lambda k: (TQ.ps.score(tsing[k][0], 1), -ks.index(k)))
    cs, ca = singles[ck]
    step = json.load(open(SP.OUT + "/" + m + "/items.json"))
    qt = {x["id"]: x["question"] for x in step["items"] if x["split"] == "test"}
    forced_text = TQ.MP.fr.raw_texts(TQ.MP.work(run, "test", "pospromptdl%d" % layer))
    out = {k: {} for k in ("E", "N1", "O", "TE", "Conly")}
    for i, q in enumerate(ids):
        out["E"][q] = (e[q], ea[q], tx.ec[q, sp.cE(s)], None)
        out["N1"][q] = (n1[q], issued[q], recs[i]["text"], None)
        out["O"][q] = (
            TQ.V[sp.st[q, "baseline_x"]],
            not sp.blank[q, "baseline_x"],
            tx.ec[q, "baseline_x"],
            None,
        )
        out["TE"][q] = (
            forced[q][0],
            forced[q][1],
            forced_text[q, "E_L%d_conf-w-p%s" % (layer, s)],
            forced[q][2],
        )
        out["Conly"][q] = (
            TQ.V[cs[i]],
            ca[i],
            tx.ec[q, "C_L%d_x-w-m%s" % (cref, ck)],
            None,
        )
    if (
        tuple([len(ids)] + [sum(out[k][q][0] for q in ids) for k in ("O", "E", "N1")])
        != G.K_TQ[m]
    ):
        raise ValueError("judgments do not match")
    if not set(ids) <= set(qt):
        raise ValueError("incomplete question bank")
    return ids, out, qt, SK.lookup(m)


def tqmain():
    allm = {m: truth_data(m) for m in G.ORDER}
    ids = sorted(set.intersection(*(set(t[0]) for t in allm.values())))
    assert len(ids) == 520
    vals = []
    for m in G.ORDER:
        _, sy, qt, M = allm[m]
        vals.append(evaluate(m, "tq_test", ids, sy, qt, M, None))
    for k in sorted(set.intersection(*(set(v) for v in vals))):
        v = [sum(vals[j][k][i] for j in range(5)) for i in range(len(ids))]
        print("TruthfulQA common520 5", k, ci(v, ids, None), flush=True)


if __name__ == "__main__":
    tqmain()
