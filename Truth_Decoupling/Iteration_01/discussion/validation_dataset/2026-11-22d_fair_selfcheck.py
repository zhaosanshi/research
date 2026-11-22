"""Read-only. Run from Truth_Decoupling in the scoring container. Uses the two models whose
ordinary, forced and released answers have already been verified. Does not produce new answers.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import selfjudge_probe as sp
import tq_same_n_decomp as tq
import qaqa_pilot_table as qa

MODELS = ("llama3.1-8b", "qwen3.5-9b")
SCORE = {"R": 1, "W": -1, "H": 0, "X": 0}


def load(root, model, source):
    stem = source + "_" if source else ""
    path = root / (stem + "items.json")
    doc = json.loads(path.read_bytes())
    meta = json.loads((root / (stem + "probe.json")).read_bytes())
    if not (doc["model"] == meta["model"] == model and meta["items_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError((model, source, "foreign items"))
    rows = doc["items"]
    active = [r for r in rows if r["answered"]]
    with np.load(root / (stem + "probe.npz"), allow_pickle=False) as raw:
        if (list(raw["ids"]) != [r["id"] for r in active] or list(raw["answer_sha1"]) != [sp.sha1(r["answer"]) for r in active]
                or list(raw["statement_sha1"]) != [sp.sha1(sp.statement(r["question"], r["answer"])) for r in active]
                or ("split" in raw and list(raw["split"]) != [r["split"] for r in active])):
            raise ValueError((source, "identity"))
        margins = raw["baseline__margin"]
        if not np.isfinite(margins).all() or not np.allclose(margins, raw["baseline__lpT"] - raw["baseline__lpF"], rtol=0, atol=2e-5):
            raise ValueError((source, "invalid margins"))
        for r, m, statement_id in zip(active, margins, raw["statement_sha1"]):
            r["margin"] = float(m)
            r["statement_id"] = str(statement_id)
    by_id = {}
    for r in rows:
        if r["answer_sha1"] != sp.sha1(r["answer"]) or r["neg"] != bool(sp.NEG_RE.match(r["answer"])):
            raise ValueError("answer or negative-prefix flag changed")
        key = (r.get("split", ""), r["id"].split("|", 1)[-1])
        if key in by_id:
            raise ValueError("duplicate answer")
        r["net"] = SCORE[r["state"]] if "state" in r else r["value"]
        by_id[key] = r
    return by_id


def sources(model, task):
    root = Path(sp.OUT) / model
    main = load(root, model, "")
    training = [r for (split, _), r in main.items() if split == "tune" and r["answered"] and not r["neg"]]
    cutpoints = thresholds(training)
    gains = [-sum(r["net"] for r in training if r["margin"] < t) for t in cutpoints]
    threshold = cutpoints[gains.index(max(gains))]
    group = "test" if task == "tq" else ""
    if not group:
        main = load(root, model, "qaqa")
    ordinary = {q: r for (s, q), r in main.items() if s == group}
    forced = {q: r for (_, q), r in load(root, model, task + "_forced").items()}
    released = {q: r for (s, q), r in load(root, model, task + "_release").items() if s == group}
    if task == "tq":
        ids, (_, _, _, _, _, ev, ea, nv, na, fv) = tq.load_model(model)
        display, layer, strength = tq.A.KEY[model]
        (s, tx, _, _), error = tq.MP.prompt_split(display, layer, 1, strength)
        if error:
            raise ValueError(error)
        original = {q: (s.blank[q, "baseline_x"], tx.ec[q, "baseline_x"]) for q in ids}
    else:
        base = qa.Raw(model, "qaqafull", sp.QAQA_BANK, ("test_full570", "test_full570_tuned", "test_full570_strong"))
        s = qa.SET[model][1]
        e = base.e(s)
        f = "conf:" + e.split(":")[1]
        for arm in ("x", e, f):
            base.need(arm)
        choice = json.loads((Path(qa.S.NS) / "splits/qaqa_tune_choice_fixedE_20261122.json").read_bytes())[model]
        nv, na, _ = base.n1_system(s, choice["n1_cap_e"], choice["n1_cap_own"])
        ev, ea = base.arm_system(e)
        confidence = base.confidence("qaqafull", f)
        excluded = set(json.loads(Path(sp.QAQA_PILOT).read_bytes())["ids"])
        if len(excluded) != 100 or len(set(base.ids)) != 570 or not excluded.issubset(base.ids):
            raise ValueError("held-out roster")
        ids = [q for q in base.ids if q not in excluded]
        fv = {q: (base.cell(f, q), not base.blank(f, q), confidence[q]) for q in ids}
        original = {q: (base.blank("x", q), base.text("x", q)) for q in ids}
    ids = sorted(ids) if task == "tq" else ids
    if {q for q in ids if na[q] and not ea[q]} != set(ids).intersection(released):
        raise ValueError("incomplete releases")
    for q in ids:
        o, f = ordinary[q], forced[q]
        if (o["net"], o["answered"]) != (ev[q], ea[q]) or (f["net"], f["answered"]) != fv[q][:2]:
            raise ValueError("issued flag or verdict")
        if f["answered"] and (f["conf"] != fv[q][2] or not math.isfinite(f["conf"])):
            raise ValueError("confidence")
        if q in released and (not released[q]["answered"] or released[q]["net"] != nv[q]):
            raise ValueError("release verdict")
        if any(r["question"] != o["question"] for r in [f] + ([released[q]] if q in released else [])):
            raise ValueError("question mismatch")
        own, text = original[q]
        f["same"] = not own and tq.MP.ro.norm(f["answer"]) == tq.MP.ro.norm(text)
        f["own"] = own
    return ids, threshold, ordinary, forced, released


def thresholds(items):
    m = sorted({r["margin"] for r in items if not r["neg"]})
    return [-math.inf] + [(a+b)/2 for a,b in zip(m, m[1:])] + [math.inf]


def passes(r, t):
    return r["answered"] and (r["neg"] or r["margin"] >= t)


def results(ids, t, ordinary, forced, released, all_thresholds=True):
    e = {q for q in ids if ordinary[q]["answered"]}
    first = {q for q in e if passes(ordinary[q], t)}
    releases = {q for q in ids if q in released and passes(released[q], t)}
    ordered = [q for q in ids if forced[q]["answered"]]
    by_conf = sorted(ordered, key=lambda q: -forced[q]["conf"])
    by_margin = sorted(ordered, key=lambda q: -forced[q]["margin"])
    s1 = {q: (ordinary[q]["net"] if q in first or q not in e else 0) for q in ids}
    s3 = {q: (released[q]["net"] if q in releases else (0 if q in released else s1[q])) for q in ids}
    s2 = {q: (released[q]["net"] if q in released else s1[q]) for q in ids}
    ans = {"S1": len(first), "S2": len(first) + sum(q in released for q in ids), "S3": len(first) + len(releases)}
    if max(ans.values()) > len(ordered):
        raise ValueError("strong arm cannot reach target")
    def bdelta(scores, sel):
        return [scores[q] - (forced[q]["net"] if q in sel else 0) for q in ids]
    def cdelta(sel):
        if len(sel) != len(releases) or sel.intersection(e):
            raise ValueError("two-pass count or pool")
        return [s3[q] - (forced[q]["net"] if q in sel else s1[q]) for q in ids]
    curves = {k: [] for k in ("S1-B+", "S2-B+", "S3-B+", "S3-C3")}
    def calibrated(cut):
        accept = {q for q in ordered if passes(forced[q], cut)}
        priority = [q for q in by_conf if q in accept] + [q for q in by_margin if q not in accept]
        answers = {}
        for k, scores in (("S1", s1), ("S2", s2), ("S3", s3)):
            sel = priority[:ans[k]]
            answers[k+"-B+"] = bdelta(scores, set(sel))
        eligible = [q for q in by_conf if q not in e and q in accept]
        answers["S3-C3"] = cdelta(set(eligible[:len(releases)])) if len(eligible) >= len(releases) else None
        return answers
    for cut in (thresholds([forced[q] for q in ordered]) if all_thresholds else [t]):
        for k, vs in calibrated(cut).items():
            if vs is not None:
                curves[k].append(sum(vs))
    diag = calibrated(t)
    passed = {q for q in ordered if passes(forced[q], t)}
    refill = {kind: max(0, n - len(passed)) for kind, n in ans.items()}
    refill["C3 need / eligible"] = (len(releases), len(passed - e))
    vetted = set()
    deficits = []
    excess = []
    for own in (True, False):
        need = sum(forced[q]["own"] == own for q in releases)
        pool = [q for q in by_conf if q not in e and q in passed and forced[q]["own"] == own and not forced[q]["same"]]
        short = max(need - len(pool), 0)
        deficits.append(short)
        if short:
            excess.append((short, [(i, s3[q] - s1[q]) for i, q in enumerate(ids) if q in releases and forced[q]["own"] == own]))
        vetted.update(pool[:need])
    refill["C3 veto and origin shortfalls (own, induced)"] = deficits
    relaxed = [s3[q] - (forced[q]["net"] if q in vetted else s1[q]) for q in ids]
    diag["S3-C3-veto-quotas"] = None if any(deficits) else cdelta(vetted)
    for k, scores in (("S1", s1), ("S2", s2), ("S3", s3)):
        diag[k+"-margin"] = bdelta(scores, set(by_margin[:ans[k]]))
    eligible = [q for q in by_margin if q not in e]
    diag["S3-two-pass-margin"] = cdelta(set(eligible[:len(releases)]))
    take = set()
    for own in (True, False):
        quota = sum(forced[q]["own"] == own for q in releases)
        pool = [q for q in eligible if forced[q]["own"] == own and not forced[q]["same"]]
        if len(pool) < quota:
            break
        take.update(pool[:quota])
    else:
        diag["S3-two-pass-margin-veto-quotas"] = cdelta(take)
    diag["S3-S2"] = [-released[q]["net"] if q in released and q not in releases else 0 for q in ids]
    return ans, len(releases), {k:(min(v), max(v), len(v)) for k,v in curves.items() if v}, diag, refill, (relaxed, excess, ans["S3"] - sum(deficits))


def share_identical_inputs(ids, collections):
    # Called per model and condition. The question is part of the statement digest.
    cache, changed = {}, 0
    for rows in collections:
        for q in ids:
            r = rows.get(q)
            if r is None or not r["answered"]:
                continue
            key = r["statement_id"]
            value = cache.setdefault(key, r["margin"])
            changed += value != r["margin"]
            r["margin"] = value
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", choices=("original", "ordinary-first", "forced-first"), default="original")
    parser.add_argument("--rule", choices=("P", "Z", "Z-no-exemption"), default="P")
    args = parser.parse_args()
    mode = args.reuse
    print("ordinary rule", args.rule, "duplicate-input margins", mode)
    for task in ("tq", "qq"):
        vectors = []
        bounds = []
        near = []
        for model in MODELS:
            ids, t, o, f, r = sources(model, task)
            if mode != "original":
                order = (o, r, f) if mode == "ordinary-first" else (f, r, o)
                print("  scores replaced", task, model, share_identical_inputs(ids, order))
            if args.rule.startswith("Z"):
                t = 0.
            if args.rule == "Z-no-exemption":
                for pool in (o, r, f):
                    for x in pool.values():
                        x["neg"] = False
            ans, k, curve, diag, refill, trims = results(ids, t, o, f, r, all_thresholds=args.rule == "P")
            print("  matched-count eligibility", refill)
            if vectors and ids != roster:
                raise ValueError("mismatched aggregate roster")
            roster = ids
            vectors.append(diag)
            bounds.append(curve)
            near.append(trims)
            print(task, model, "threshold", t, "issued", ans, "releases", k, "threshold bounds and feasible cuts", curve)
            print("  fixed controls", {k:sum(v) if v is not None else None for k,v in diag.items()}, flush=True)
        print(task, len(roster), "joint threshold bounds (each model free)",
              {k:(sum(d[k][0] for d in bounds), sum(d[k][1] for d in bounds)) for k in sorted(set.intersection(*(set(d) for d in bounds)))})
        # For each draw, every admissible deletion is considered. Report the outer interval,
        # not a confidence interval for a single data-selected pruning policy.
        n = len(roster)
        rng = random.Random(20261116 if task == "tq" else 20261119)
        weights = np.array([np.bincount([rng.randrange(n) for _ in range(n)], minlength=n) for _ in range(4000)])
        weights = np.vstack((np.ones(n, dtype=int), weights))
        upper = lower = weights @ np.sum([x[0] for x in near], axis=0)
        upper, lower = upper.copy(), lower.copy()
        for _, excess, _ in near:
            for short, eligible in excess:
                indices, rewards = zip(*eligible)
                reductions = np.sort(weights[:, indices] * np.array(rewards), axis=1)
                lower -= reductions[:, -short:].sum(axis=1)
                upper -= reductions[:, :short].sum(axis=1)
        print("  nearest origin/veto count, both sides", sum(x[2] for x in near), "net bounds", lower[0], upper[0],
              "paired outer envelope", np.sort(lower[1:])[100], np.sort(upper[1:])[3899])
        for k in vectors[0]:
            if any(v.get(k) is None for v in vectors):
                print(" ", k, "unreachable")
                continue
            totals = [sum(v[k][i] for v in vectors) for i in range(len(roster))]
            print(" ",k,tq.ps.ci(totals) if task == "tq" else qa.S.joint_ci([totals]), flush=True)


if __name__ == "__main__":
    main()
