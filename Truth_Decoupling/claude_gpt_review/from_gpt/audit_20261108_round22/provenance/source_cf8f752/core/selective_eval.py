"""Matched *actual* answer coverage, with no model or judging dependencies.

Rows have ``id``, ``pair_id`` (the independent question/group), ``outcome``
(correct/wrong/abstain) and a finite ``score`` (larger means retain). Correctness
never enters ranking: only score, a stable identity hash, and whether the actual
candidate is a refusal. The caller owns score provenance; explicit reference or
gold-label-use flags are refused here, but a float alone cannot prove provenance.

Two distinct evaluations are deliberately named:
* batch_exact_quota: select floor(q*N) actual answers on this batch without using
  test correctness. Bootstrap resamples whole questions, then reselects the quota
  on every shared replicate, including repeated instances of the same question.
* frozen_threshold: learn a score/hash threshold on tuning rows and apply it
  unchanged. Its test coverage is measured, never asserted to equal tuning q.

All cohorts stay fixed, including natural refusals. Insufficient candidates make
a quota infeasible; no row is removed and no incomplete bootstrap is silently
conditioned on the successful replicates. Returned dictionaries are JSON-safe.
"""
from collections import Counter, defaultdict
from collections.abc import Mapping
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
import math
from numbers import Integral, Real

import numpy as np


OUTCOMES = ("correct", "wrong", "abstain")
METRICS = ("accuracy_answered", "risk_answered", "correct_rate", "wrong_rate")
PROTOCOL = "actual_coverage_v1"


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    return int(value)


def _fraction(value):
    if isinstance(value, bool):
        raise ValueError("coverage must be a finite fraction in [0,1]")
    try:
        out = Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError):
        raise ValueError("coverage must be a finite fraction in [0,1]") from None
    if not 0 <= out <= 1:
        raise ValueError("coverage must be in [0,1]")
    return out


def quota_count(coverage, n):
    """Exact decimal arithmetic, e.g. quota_count(.58, 100) == 58."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    q = _fraction(coverage)
    return (q.numerator * int(n)) // q.denominator


@lru_cache(maxsize=65536)
def _hash(item_id, seed):
    return hashlib.sha256(json.dumps([PROTOCOL, seed, item_id],
                                     ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def random_score(item_id, seed=42):
    """A reproducible random-ranking control independent of arm and outcomes."""
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError("item_id must be a nonempty string")
    digest = _hash(item_id, _seed(seed))
    return (int(digest[:14], 16) >> 3) / float(2**53)


def _rows(rows):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list/tuple")
    out, seen = [], set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each row must be a mapping")
        for flag in ("uses_gold_labels", "uses_reference_answer"):
            if flag in row and type(row[flag]) is not bool:
                raise ValueError("%s must be boolean" % flag)
            if row.get(flag, False):
                raise ValueError("reference/gold-informed scores cannot select test answers")
        item_id, group = row.get("id"), row.get("pair_id")
        if not isinstance(item_id, str) or not item_id.strip() or item_id in seen:
            raise ValueError("row ids must be unique nonempty strings")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("pair_id must be a nonempty string")
        seen.add(item_id)
        outcome = row.get("outcome")
        if outcome not in OUTCOMES:
            raise ValueError("unresolved or unknown outcome for %s: %r" % (item_id, outcome))
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
            raise ValueError("every score must be finite, including scores for refusals")
        out.append({"id": item_id, "pair_id": group, "outcome": outcome, "score": float(score)})
    return sorted(out, key=lambda row: row["id"])


def validate_arms(arms):
    """Return copied rows in canonical ID order; every arm has the same cohort/groups."""
    if not isinstance(arms, Mapping) or not arms:
        raise ValueError("arms must be a nonempty mapping")
    if any(not isinstance(name, str) or not name.strip() for name in arms):
        raise ValueError("arm names must be nonempty strings")
    result = {name: _rows(arms[name]) for name in sorted(arms)}
    first = next(iter(result.values()))
    identity = [(r["id"], r["pair_id"]) for r in first]
    for name, rows in result.items():
        if [(r["id"], r["pair_id"]) for r in rows] != identity:
            raise ValueError("arm %s has a different fixed cohort or pair_id mapping" % name)
    return result


def _metrics(rows, mask):
    n = len(rows)
    counts = Counter(r["outcome"] if keep else "abstain" for r, keep in zip(rows, mask))
    counts = {key: int(counts[key]) for key in OUTCOMES}
    answered = counts["correct"] + counts["wrong"]
    return {"counts": counts, "answered": answered, **counts,
            "actual_coverage": answered / n if n else None,
            "accuracy_answered": counts["correct"] / answered if answered else None,
            "risk_answered": counts["wrong"] / answered if answered else None,
            "correct_rate": counts["correct"] / n if n else None,
            "wrong_rate": counts["wrong"] / n if n else None}


def _quota(rows, q, seed, details=True):
    n, k = len(rows), quota_count(q, len(rows))
    # Repeated bootstrap instances share an ID; their instance index resolves
    # exact duplicates without merging them into one observation.
    eligible = [i for i, row in enumerate(rows) if row["outcome"] != "abstain"]
    order = sorted(eligible, key=lambda i: (-rows[i]["score"], _hash(rows[i]["id"], seed), rows[i]["id"], i))
    result = {"evaluation": "batch_exact_quota", "coverage_fraction": str(q),
              "target_coverage": float(q), "n": n, "k": k, "n_available": len(eligible),
              "feasible": len(eligible) >= k, "reason": None, "tie_seed": seed}
    if details:
        result.update(ids=[r["id"] for r in rows], pair_ids={r["id"]: r["pair_id"] for r in rows})
    if len(eligible) < k:
        result.update(reason="insufficient_actual_answers", counts=None, answered=None,
                      correct=None, wrong=None, abstain=None, actual_coverage=None,
                      **{key: None for key in METRICS})
        if details:
            result.update(mask=None, selected_ids=None, delivered_outcomes=None)
        return result
    selected = set(order[:k])
    mask = [i in selected for i in range(n)]
    result.update(_metrics(rows, mask))
    assert result["answered"] == k
    if details:
        result.update(mask=mask, selected_ids=[r["id"] for i, r in enumerate(rows) if i in selected],
                      delivered_outcomes={r["id"]: r["outcome"] if mask[i] else "abstain"
                                          for i, r in enumerate(rows)})
    return result


def quota_select(rows, coverage, seed=42):
    """Return an exact-quota delivery result in canonical ID order.

    ``mask`` aligns with ``ids``; ``selected_ids`` and ``delivered_outcomes`` let
    the caller write its actual output texts (replace unselected outputs by its
    fixed refusal). Infeasible results have all three fields set to None rather
    than silently serving fewer than k answers. Empty and k=0 cohorts have
    accuracy/risk None. All input rows and their denominator remain present.
    """
    return _quota(_rows(rows), _fraction(coverage), _seed(seed))


def _delivered_identity(result):
    if not result.get("feasible") or result.get("delivered_outcomes") is None:
        raise ValueError("exact comparison requires feasible delivery results")
    ids, groups, delivered = result.get("ids"), result.get("pair_ids"), result["delivered_outcomes"]
    if (not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != set(delivered)
            or set(ids) != set(groups) or any(v not in OUTCOMES for v in delivered.values())):
        raise ValueError("invalid delivery identities or outcomes")
    answered = sum(v != "abstain" for v in delivered.values())
    if answered != result.get("answered"):
        raise ValueError("stored answered count disagrees with actual deliveries")
    return sorted(ids), groups, delivered


def exact_paired_correctness(a, b):
    """Two-sided exact group-swap test on the *fixed observed delivery masks*.

    This is not the reselected-quota bootstrap estimator. Each row contributes
    I(delivered correct by a)-I(delivered correct by b); differences are summed
    within pair_id before independent sign flips. Singleton groups reduce to
    the exact discordant-pair binomial test (4 versus 0 gives p=.125). For TF,
    swapping both statements together respects their shared question.
    """
    ids, groups, da = _delivered_identity(a)
    ids_b, groups_b, db = _delivered_identity(b)
    if ids != ids_b or groups != groups_b or a["answered"] != b["answered"]:
        raise ValueError("exact comparison requires identical cohort/groups and equal actual answer counts")
    diff = {i: int(da[i] == "correct") - int(db[i] == "correct") for i in ids}
    group_diff = defaultdict(int)
    for item_id, value in diff.items():
        group_diff[groups[item_id]] += value
    weights = [abs(value) for value in group_diff.values() if value]
    observed = abs(sum(diff.values()))
    if observed == 0:
        p = 1.0
    else:
        distribution = {0: 1}
        for weight in weights:
            nxt = defaultdict(int)
            for total, count in distribution.items():
                nxt[total - weight] += count
                nxt[total + weight] += count
            distribution = dict(nxt)
        p = sum(count for total, count in distribution.items() if abs(total) >= observed) / (2**len(weights))
    return {"test": "exact_two_sided_group_swap_fixed_deliveries", "p_value": float(p),
            "positive_discordant_rows": sum(v > 0 for v in diff.values()),
            "negative_discordant_rows": sum(v < 0 for v in diff.values()),
            "n_groups": len(group_diff), "informative_groups": len(weights),
            "delta_correct_count": sum(diff.values()), "answered_each": a["answered"],
            "scope": "fixed observed delivery masks; group-wise arm exchangeability under the null"}


def _bootstrap_summary(values, failures, n_boot, empty, zero_answers, coverages, counts):
    status = ("not_requested" if n_boot == 0 else "empty_cohort" if empty else
              "infeasible_resamples" if failures else "undefined_zero_answers" if zero_answers else "ok")
    intervals = None
    if status == "ok":
        intervals = {key: np.percentile(values[key], [2.5, 97.5]).tolist() for key in METRICS}
    return {"status": status, "n_resamples": n_boot, "infeasible_resamples": failures,
            "ci95": intervals,
            "actual_coverage_range": [min(coverages), max(coverages)] if coverages else None,
            "answered_count_range": [min(counts), max(counts)] if counts else None}


def evaluate_coverage(arms, coverages=(.25, .5, .75), n_boot=2000, seed=42):
    """Evaluate all arms on each exact quota and paired, group-reselected CIs.

    Return ``{protocol, arm_names, cohort_ids, n_groups, bootstrap_unit,
    n_boot, seed, coverages: [entry, ...]}``. Each entry has target_coverage,
    coverage_fraction, n, k, ``arms[name]`` (quota_select result plus a bootstrap
    object), and ``comparisons`` (one record for each unordered arm pair).
    Comparison records contain arm_a/arm_b, point_delta, ci95, bootstrap_status,
    infeasible_resamples and exact_fixed_deliveries. Positive deltas mean a-b.

    Every replicate draws len(unique pair_id) groups with replacement, keeps all
    their rows and reselects floor(q*N_replicate) per arm. Unequal group sizes
    may change N across replicates; within each replicate N and K remain shared.
    Any infeasible replicate suppresses that arm/comparison's entire CI instead
    of computing an interval conditional on success. Point results remain shown.
    """
    canonical, seed = validate_arms(arms), _seed(seed)
    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 0 or n_boot == 1:
        raise ValueError("n_boot must be 0 or an integer >= 2")
    n_boot = int(n_boot)
    if not isinstance(coverages, (list, tuple)) or not coverages:
        raise ValueError("coverages must be a nonempty list/tuple")
    qs = [_fraction(q) for q in coverages]
    if len(set(qs)) != len(qs):
        raise ValueError("duplicate target coverages")
    names = list(canonical)
    first = canonical[names[0]]
    grouped = defaultdict(list)
    for i, row in enumerate(first):
        grouped[row["pair_id"]].append(i)
    group_indices = [grouped[g] for g in sorted(grouped)]
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(group_indices), len(group_indices)).tolist()
             for _ in range(n_boot)] if group_indices else []
    pairs = list(itertools.combinations(names, 2))
    entries = []
    for q in qs:
        points = {name: _quota(canonical[name], q, seed) for name in names}
        values = {name: {key: [] for key in METRICS} for name in names}
        failure = Counter()
        coverage_values, answer_counts = defaultdict(list), defaultdict(list)
        pair_values = {pair: {key: [] for key in METRICS} for pair in pairs}
        pair_failures = Counter()
        zero_answers = False
        for draw in draws:
            idx = [i for group in draw for i in group_indices[group]]
            rep = {name: _quota([canonical[name][i] for i in idx], q, seed, details=False) for name in names}
            if quota_count(q, len(idx)) == 0:
                zero_answers = True
            for name, row in rep.items():
                if not row["feasible"]:
                    failure[name] += 1
                    continue
                coverage_values[name].append(row["actual_coverage"])
                answer_counts[name].append(row["answered"])
                if row["answered"]:
                    for key in METRICS:
                        values[name][key].append(row[key])
            for pair in pairs:
                a, b = (rep[name] for name in pair)
                if not a["feasible"] or not b["feasible"]:
                    pair_failures[pair] += 1
                elif a["answered"]:
                    for key in METRICS:
                        pair_values[pair][key].append(a[key] - b[key])
        for name in names:
            points[name]["bootstrap"] = _bootstrap_summary(values[name], failure[name], n_boot,
                                                          not first, zero_answers,
                                                          coverage_values[name], answer_counts[name])
            if not points[name]["feasible"]:
                points[name]["bootstrap"].update(status="infeasible_point", ci95=None)
        comparisons = []
        for pair in pairs:
            a, b = (points[name] for name in pair)
            available = a["feasible"] and b["feasible"]
            bs = _bootstrap_summary(pair_values[pair], pair_failures[pair], n_boot,
                                    not first, zero_answers, [], [])
            if not available:
                bs.update(status="infeasible_point", ci95=None)
            comparisons.append({"arm_a": pair[0], "arm_b": pair[1],
                                "point_delta": {key: a[key] - b[key] if available and a[key] is not None else None
                                                for key in METRICS},
                                "ci95": bs["ci95"], "bootstrap_status": bs["status"],
                                "infeasible_resamples": bs["infeasible_resamples"],
                                "exact_fixed_deliveries": exact_paired_correctness(a, b) if available else None})
        entries.append({"target_coverage": float(q), "coverage_fraction": str(q), "n": len(first),
                        "k": quota_count(q, len(first)), "arms": points, "comparisons": comparisons})
    return {"protocol": PROTOCOL, "evaluation": "batch_exact_quota", "arm_names": names,
            "cohort_ids": [r["id"] for r in first], "n_groups": len(group_indices),
            "bootstrap_unit": "pair_id", "bootstrap_rule": "resample whole groups then reselect quota",
            "n_boot": n_boot, "seed": seed, "coverages": entries}


def calibrate_threshold(tune_rows, coverage, seed=42):
    """Freeze a label-free score/hash decision boundary on tuning rows only.

    The rule may be all/none/cutoff. A cutoff accepts score > threshold, or score
    == threshold with identity hash <= tie_cutoff. Hash ordering makes score ties
    reproducible without using outcome labels or current test-batch quantiles.
    """
    rows, q, seed = _rows(tune_rows), _fraction(coverage), _seed(seed)
    point = _quota(rows, q, seed)
    policy = {"protocol": PROTOCOL, "evaluation": "frozen_threshold", "feasible": point["feasible"],
              "target_coverage": float(q), "coverage_fraction": str(q), "seed": seed,
              "n_tune": len(rows), "k_tune": point["k"], "n_available_tune": point["n_available"],
              "tune_ids_sha256": hashlib.sha256(json.dumps([r["id"] for r in rows],
                                                           ensure_ascii=False).encode()).hexdigest(),
              "mode": None, "threshold": None, "tie_cutoff": None,
              "reason": point["reason"]}
    if not point["feasible"]:
        return policy
    if point["k"] == 0:
        policy["mode"] = "none"
    elif point["k"] == point["n_available"]:
        policy["mode"] = "all"
    else:
        by_id = {r["id"]: r for r in rows}
        selected = [by_id[i] for i in point["selected_ids"]]
        boundary = max(selected, key=lambda r: (-r["score"], _hash(r["id"], seed), r["id"]))
        policy.update(mode="cutoff", threshold=boundary["score"],
                      tie_cutoff=[_hash(boundary["id"], seed), boundary["id"]])
    return policy


def apply_threshold(test_rows, policy):
    """Apply the frozen rule unchanged and report its measured actual coverage."""
    rows = _rows(test_rows)
    if (not isinstance(policy, Mapping) or policy.get("protocol") != PROTOCOL
            or policy.get("evaluation") != "frozen_threshold" or policy.get("feasible") is not True):
        raise ValueError("a feasible frozen_threshold policy is required")
    seed = _seed(policy.get("seed"))
    q = _fraction(policy.get("coverage_fraction"))
    mode = policy.get("mode")
    if mode not in ("all", "none", "cutoff"):
        raise ValueError("unknown threshold mode")
    threshold, cutoff = policy.get("threshold"), policy.get("tie_cutoff")
    if mode == "cutoff" and (isinstance(threshold, bool) or not isinstance(threshold, Real)
                             or not math.isfinite(float(threshold)) or not isinstance(cutoff, (list, tuple))
                             or len(cutoff) != 2 or not all(isinstance(x, str) for x in cutoff)):
        raise ValueError("invalid frozen score/hash threshold")
    if mode == "cutoff" and (not cutoff[1].strip() or cutoff[0] != _hash(cutoff[1], seed)):
        raise ValueError("frozen tie cutoff does not match its identity and seed")
    mask = []
    for r in rows:
        keep = mode == "all" or (mode == "cutoff" and (r["score"] > threshold or
                (r["score"] == threshold and (_hash(r["id"], seed), r["id"]) <= tuple(cutoff))))
        mask.append(r["outcome"] != "abstain" and keep)
    return {"evaluation": "frozen_threshold", "feasible": True, "n": len(rows),
            "target_coverage": float(q), "coverage_fraction": str(q), "tie_seed": seed,
            "ids": [r["id"] for r in rows], "pair_ids": {r["id"]: r["pair_id"] for r in rows},
            "mask": mask, "selected_ids": [r["id"] for r, keep in zip(rows, mask) if keep],
            "delivered_outcomes": {r["id"]: r["outcome"] if keep else "abstain" for r, keep in zip(rows, mask)},
            **_metrics(rows, mask)}
