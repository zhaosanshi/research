"""Exploratory matched-count check of directional signals on the SAME no-write answers.
Run in the experiment's scoring container, from Truth_Decoupling: python3 -B PATH.
No generation or new judge requests. For each bootstrap, share question IDs
across models and recalculate K. Average over uniform tie-breaking at the boundary.
"""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/review_checks"]
import nsc_prompt_table as pilot
from judge_two_axis_run import load_verified_single

MODELS = ("llama3.1-8b", "gemma3-12b")
VALUE = {"A": 1, "B": -1, "C": 0, "H": 0}


def matched_value(weights, values, eligible, tiers, target):
    """Expected net from selecting exactly target responses in descending tiers."""
    groups = defaultdict(list)
    for i in np.flatnonzero(eligible):
        groups[tiers[i]].append(i)
    left = target.copy()
    net = np.zeros(len(weights))
    for tier in sorted(groups, reverse=True):
        ids = groups[tier]
        w = weights[:, ids]
        size = w.sum(axis=1)
        keep = np.minimum(size, left)
        net += np.divide((w @ values[ids]) * keep, size, out=np.zeros(len(w)), where=size > 0)
        left -= keep
    if np.any(left):
        raise ValueError("The requested answer count is unreachable")
    return net


def pro_original(task, model, split, bank, orig):
    work = Path(pilot.S.NS) / "judge" / task / model / (split + "_pro100")
    _, cells, votes = load_verified_single(str(work), "pro", generation_path=str(work / "judge_input.json"),
                                          require_complete=False)
    scores = {}
    for key, c in cells.items():
        if c["condition"] != "baseline_x":
            continue
        q = c["pair_id"].removeprefix(task + ":")
        if q in scores or c["prediction"] != orig[q] or c["question"] != bank[q]["question"]:
            raise ValueError("Original-answer Pro vote is duplicated or refers to another input")
        vote = votes[key]
        if vote["status"] == "scored":
            scores[q] = VALUE[vote["label"]]
        elif vote.get("basis") == "empty_output_no_judge_vote" and not orig[q].strip():
            scores[q] = 0
        else:
            raise ValueError("Unresolved vote")
    if len(scores) != 100 or set(scores) != {q for q, b in bank.items() if b["pro100"]}:
        raise ValueError("Expected the 100 frozen Pro IDs")
    return scores


def load(task, model, split):
    pick, _, _ = pilot.choose(task, model, pilot.S.models()[model]["cl"])
    if not isinstance(pick, dict):
        raise ValueError(pick)
    take, ids = pilot.passes(task, model, split)
    if len(ids) != 500 or len(set(ids)) != len(ids):
        raise ValueError("Incomplete test split")
    e = pick["single E"][1]
    c = pick["single C, either sign"][1]
    layer, _, grid, _, _ = pilot.T.MAIN[model]
    grid = list(grid)
    i = grid.index(e.rsplit(".p", 1)[1])
    e2 = f"x:E{layer}.w.p{grid[i-1 if i else 1]}"
    c2 = c.replace(".w.p", ".w.m") if ".w.p" in c else c.replace(".w.m", ".w.p")
    base, *probes = [take(a) for a in ("x", e, c, e2, c2)]
    if base is None or any(p is None for p in probes):
        raise ValueError("The fixed main or equal-pass control arm is unavailable")
    bank, _ = pilot.T.bank(task, split)
    original = dict(zip(ids, base[2]))
    pro = pro_original(task, model, split, bank, original)
    confidence = None
    if split == "test":
        generated, _ = pilot.S.gen_file(task, model, split, "conf")
        texts = {task + ":" + r["id"]: r["text"] for r in generated["rows"]}
        rr = pilot.S.conf_rows(task, model, split, "conf", texts)
        confidence = {q: rr[task + ":" + q]["mean_logp"] for q in ids}
    bits = []
    for probe in probes:
        bits.append(np.array([int(a and b and pilot.ro.norm(t) == pilot.ro.norm(u))
                              for a, b, t, u in zip(base[1], probe[1], base[2], probe[2])]))
    return ids, base, probes, bits, pro, confidence


def run(task, split):
    sources = [load(task, m, split) for m in MODELS]
    for judge in ("alias", "pro"):
        for tie_conf in ((False, True) if split == "test" else (False,)):
            ids = None
            totals = defaultdict(lambda: 0)
            for seq, base, probes, bits, pro, conf in sources:
                ix = [i for i, q in enumerate(seq) if judge == "alias" or q in pro]
                current = [seq[i] for i in ix]
                if ids is None:
                    ids = current
                    n = len(ids)
                    weights = np.vstack([np.ones(n, dtype=int), np.random.default_rng(20261122)
                                         .multinomial(n, np.full(n, 1/n), size=4000)])
                if current != ids:
                    raise ValueError("Model question lists differ")
                x, e, c = (np.asarray(b[1], dtype=int)[ix] for b in (base, *probes[:2]))
                k = np.minimum(np.minimum(weights@e, weights@c), weights@x)
                totals["N"] += k
                vals = np.array([base[0][i] if judge == "alias" else pro[seq[i]] for i in ix])
                be, bc, be2, bc2 = [b[ix] for b in bits]
                tie = [conf[q] if tie_conf else 0 for q in ids]
                if any(not np.isfinite(v) for v, ok in zip(tie, x) if ok):
                    raise ValueError("Missing or non-finite tie-break confidence for an eligible original")
                # All filtering variants score exactly the same original text. The two
                # same-axis controls use the adjacent E strength or the opposite C sign.
                for label, score in (("E", be), ("C", bc), ("EC", be+bc), ("EE", be+be2),
                                     ("CC", bc+bc2), ("no directional check", be*0)):
                    totals[label] += matched_value(weights, vals, x, list(zip(score, tie)), k)
            print(task, split, judge, "confidence ties" if tie_conf else "uniform ties", "K", totals["N"][0])
            for label in ("E", "C", "EE", "CC", "no directional check"):
                d = totals["EC"] - totals[label]
                lo, hi = np.percentile(d[1:], (2.5, 97.5))
                print("  E+C minus %-22s %+.1f [%+.1f, %+.1f]" % (label, d[0], lo, hi))



def popqa_u_originals():
    """Only scored where both frozen U winners hand in the known no-write text."""
    for split in ("test", "test2"):
        totals = defaultdict(lambda: 0)
        shared = None
        for model in MODELS:
            task = "popqa"
            pick, _, _ = pilot.choose(task, model, pilot.S.models()[model]["cl"])
            take, ids = pilot.passes(task, model, split)
            x = take("x")
            gold = pro_original(task, model, split, pilot.T.bank(task, split)[0], dict(zip(ids, x[2])))
            keys = [pick[fam] for fam in (pilot.ALL_FAM, "E as a check")]
            if any(key[0] != "check" for key in keys):
                raise ValueError("A selected method can submit a different answer; original votes alone do not suffice")
            ix = [i for i, q in enumerate(ids) if q in gold]
            qs = [ids[i] for i in ix]
            if shared is None:
                shared = qs
                n = len(qs)
                w = np.vstack([np.ones(n, dtype=int), np.random.default_rng(20261122)
                               .multinomial(n, np.full(n, 1/n), size=4000)])
            if qs != shared:
                raise ValueError("Incompatible common Pro sample")
            v = np.array([gold[q] for q in qs])
            u, b = [np.array(pilot.system(take, key)[1], dtype=int)[ix] for key in keys]
            target = w @ u
            totals["N"] += target
            totals["U"] += w @ (u*v)
            totals["E-check"] += w @ (b*v)
            if split == "test":
                generated, _ = pilot.S.gen_file(task, model, split, "conf")
                conf = pilot.S.conf_rows(task, model, split, "conf",
                                         {task + ":" + r["id"]: r["text"] for r in generated["rows"]})
                rank = [conf[task + ":" + q]["mean_logp"] for q in qs]
                elig = np.array(x[1])[ix]
                if any(not np.isfinite(vv) for vv, ok in zip(rank, elig) if ok):
                    raise ValueError("Missing confidence for an eligible question")
                totals["no-write ranking"] += matched_value(w, v, elig, rank, target)
        for label in (("E-check", "no-write ranking") if split=="test" else ("E-check",)):
            delta = totals["U"] - totals[label]
            lo, hi = np.percentile(delta[1:], (2.5,97.5))
            print("popqa",split,"Pro 100: U N",int(totals["N"][0]),"U",int(totals["U"][0]),
                  "control",label,"net",int(totals[label][0]),"difference %+.1f [%+.1f,%+.1f]" % (delta[0],lo,hi))



def recovery_ceiling(task, model, split):
    """A generous upper bound: keep every already-answered E response and let an oracle
    choose both the newly answered IDs and any of the four observed re-ask texts.
    Alias scoring is binary on answered rows. A new answer must replace an E blank.
    """
    m = pilot.S.models()[model]
    el, e_strength, _, (cap_new, cap_own), _ = pilot.T.MAIN[model]
    take, ids = pilot.passes(task, model, split)
    base, e = take("x"), take(f"x:E{el}.w.p{e_strength}")
    rounds = [take(f"x:E{el}.w.p{e_strength}+C{m['cl']}.w.m{k}") for k in pilot.KS]
    if not all(rounds):
        raise ValueError("Incomplete C candidate pool")
    k = right_available = any_available = realized_gain = 0
    for j in range(len(ids)):
        if e[1][j]:
            continue
        candidates = [r[0][j] for r in rounds if r[1][j]]
        if e[0][j] != 0 or any(v not in (-1,1) for v in candidates):
            raise ValueError("Expected binary alias scores on actual answers and zero on blanks")
        right_available += 1 in candidates
        any_available += bool(candidates)
        own = not base[1][j]
        cap = float((cap_own if own else cap_new).replace("-", "."))
        for strength, candidate in zip(pilot.KS, rounds):
            if float(strength) > cap:
                break
            if candidate[1][j]:
                if own or pilot.ro.norm(candidate[2][j]) != pilot.ro.norm(base[2][j]):
                    k += 1
                    realized_gain += candidate[0][j]
                break
    upper = 2*min(right_available,k) - k
    if not (k <= any_available and realized_gain <= upper):
        raise ValueError("inconsistent oracle bound")
    print("recovery", task, model, split, "actual_extra", k, "IDs_with_some_answer", any_available,
          "IDs_with_any_correct", right_available, "actual_gain", realized_gain, "maximum_gain_at_actual_extra", upper)

if __name__ == "__main__":
    popqa_u_originals()
    for task in ("triviaqa", "popqa"):
        for split in ("test", "test2"):
            run(task, split)
            for model in MODELS:
                recovery_ceiling(task, model, split)
