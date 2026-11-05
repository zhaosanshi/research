"""Pure candidate selection for source-free QA-DAS and its QA mean-difference control.

The proxy ranks aligned paired-answer correctness arrays; it never certifies
free-generation accuracy. Final selection uses resolved semantic QA outcomes on
the exact same question IDs. Reference-answer diagnostics must be separated by
the caller: passing one here is an error, not a silently ignored candidate.
"""
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import math
from numbers import Integral, Real


ESTIMATORS = ("qa_das", "qa_md")
OUTCOMES = ("correct", "wrong", "abstain")
WRITE_MODES = ("prefill", "continuous")


def _boolean_flag(spec, key, default):
    value = spec.get(key, default)
    if type(value) is not bool:
        raise ValueError("%s.%s must be boolean" % (spec.get("name"), key))
    return value


def _validated_specs(condition_specs):
    if not isinstance(condition_specs, (list, tuple)):
        raise ValueError("condition_specs must be an ordered list/tuple")
    specs, names = [], set()
    for spec in condition_specs:
        if not isinstance(spec, Mapping):
            raise ValueError("every condition spec must be a mapping")
        name = spec.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("condition names must be nonempty and unique")
        names.add(name)
        reference = _boolean_flag(spec, "uses_reference_answer", False)
        eligible = _boolean_flag(spec, "eligible_for_selection", True)
        if reference:
            raise ValueError("diagnostic %s uses a reference answer and cannot enter selection" % name)
        if name == "baseline":
            if spec.get("kind") != "baseline":
                raise ValueError("baseline must have kind='baseline'")
            # A baseline may explicitly say it is not itself selectable.
        else:
            if not eligible:
                raise ValueError("diagnostic %s is ineligible for selection" % name)
            if spec.get("kind") != "E" or spec.get("context") not in ESTIMATORS:
                raise ValueError("%s must be an E candidate with context qa_das or qa_md" % name)
            if spec.get("mode") not in WRITE_MODES:
                raise ValueError("%s has an unsupported write mode" % name)
            layer, alpha = spec.get("layer"), spec.get("alpha")
            if isinstance(layer, bool) or not isinstance(layer, Integral) or layer < 0:
                raise ValueError("%s.layer must be a nonnegative integer" % name)
            if (isinstance(alpha, bool) or not isinstance(alpha, Real)
                    or not math.isfinite(float(alpha)) or alpha < 0):
                raise ValueError("%s.alpha must be a finite nonnegative number" % name)
        specs.append(spec)
    if "baseline" not in names:
        raise ValueError("baseline condition is required")
    return specs, names


def _aligned_keys(values, names, label):
    if not isinstance(values, Mapping) or set(values) != names:
        actual = set(values) if isinstance(values, Mapping) else set()
        raise ValueError("%s condition names differ: missing=%s extra=%s" % (
            label, sorted(names - actual), sorted(actual - names, key=str)))


def rank_proxy_candidates(condition_specs, conditions, top_k=3, main_mode="continuous"):
    """Return {qa_das: [spec, ...], qa_md: [spec, ...]}, at most top_k layers each.

    All ``correct`` arrays must be nonempty boolean lists/tuples, in the same
    question order as baseline. Counts and paired harms are recomputed from
    these arrays; cached ``accuracy``/``harmed`` summaries do not select a row.
    Ranking is: more correct, fewer baseline-correct items lost, smaller alpha,
    shallower layer, lexical name. Select one alpha per layer, then the best
    distinct layers within each estimator. A proxy need not beat baseline to
    enter the shortlist. Modes other than main_mode are validated but not ranked.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if main_mode not in WRITE_MODES:
        raise ValueError("main_mode must be prefill or continuous")
    specs, names = _validated_specs(condition_specs)
    _aligned_keys(conditions, names, "proxy")
    vectors = {}
    for name in names:
        row = conditions[name]
        vector = row.get("correct") if isinstance(row, Mapping) else None
        if (not isinstance(vector, (list, tuple)) or not vector
                or any(type(value) is not bool for value in vector)):
            raise ValueError("%s.correct must be a nonempty boolean list/tuple" % name)
        vectors[name] = vector
    base = vectors["baseline"]
    if any(len(vector) != len(base) for vector in vectors.values()):
        raise ValueError("proxy correctness arrays must have the same length as baseline")

    def key(spec):
        correct = vectors[spec["name"]]
        harm = sum(before and not after for before, after in zip(base, correct))
        return (-sum(correct), harm, float(spec["alpha"]), int(spec["layer"]), spec["name"])

    result = {estimator: [] for estimator in ESTIMATORS}
    for estimator in ESTIMATORS:
        by_layer = {}
        for spec in specs:
            if spec.get("context") != estimator or spec.get("mode") != main_mode:
                continue
            layer = int(spec["layer"])
            if layer not in by_layer or key(spec) < key(by_layer[layer]):
                by_layer[layer] = spec
        result[estimator] = [deepcopy(spec) for spec in sorted(by_layer.values(), key=key)[:top_k]]
    return result


def select_real_candidates(condition_specs, outcomes, question_ids, tolerance=.03):
    """Select each estimator independently from resolved, actual QA outcomes.

    ``outcomes[name][question_id]`` is exactly correct/wrong/abstain. All specs
    and all outcome maps must cover exactly the same unique question IDs; any
    unresolved/unknown label, missing/extra question or diagnostic is rejected.

    Feasible: strictly more correct answers than baseline AND absolute change
    in actual answered count / all questions <= tolerance. Among feasible
    candidates: maximize correct count, then minimize original-correct answers
    lost (correct->wrong plus correct->abstain), then smaller alpha, shallower
    layer, lexical name. Return full specs or None, plus auditable reports.
    """
    specs, names = _validated_specs(condition_specs)
    _aligned_keys(outcomes, names, "semantic outcomes")
    if (not isinstance(question_ids, (list, tuple)) or not question_ids
            or any(not isinstance(pid, str) or not pid.strip() for pid in question_ids)
            or len(set(question_ids)) != len(question_ids)):
        raise ValueError("question_ids must be a nonempty ordered list/tuple of unique string IDs")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, Real)
            or not math.isfinite(float(tolerance)) or not 0 <= tolerance <= 1):
        raise ValueError("tolerance must be a finite number in [0,1]")
    # Compare integer count changes to the stated decimal tolerance, avoiding
    # e.g. 0.53 - 0.50 > 0.03 due solely to floating-point subtraction.
    limit = Fraction(str(tolerance))
    ids, id_set, n = list(question_ids), set(question_ids), len(question_ids)
    max_answer_change = (limit.numerator * n) // limit.denominator
    ordered = {}
    for name in names:
        values = outcomes[name]
        if not isinstance(values, Mapping) or set(values) != id_set:
            actual = set(values) if isinstance(values, Mapping) else set()
            raise ValueError("%s question IDs differ: missing=%s extra=%s" % (
                name, sorted(id_set - actual), sorted(actual - id_set, key=str)))
        for pid in ids:
            if values[pid] not in OUTCOMES:
                raise ValueError("%s/%s has unresolved or unknown outcome %r" % (name, pid, values[pid]))
        ordered[name] = [values[pid] for pid in ids]
    base = ordered["baseline"]
    base_counts = Counter(base)
    base_answered = base_counts["correct"] + base_counts["wrong"]
    reports = {}
    for spec in specs:
        name = spec["name"]
        values = ordered[name]
        counter = Counter(values)
        counts = {status: int(counter[status]) for status in OUTCOMES}
        answered = counts["correct"] + counts["wrong"]
        transitions = {before + "->" + after: 0 for before in OUTCOMES for after in OUTCOMES}
        for before, after in zip(base, values):
            transitions[before + "->" + after] += 1
        gained = transitions["wrong->correct"] + transitions["abstain->correct"]
        lost = transitions["correct->wrong"] + transitions["correct->abstain"]
        net_correct = counts["correct"] - base_counts["correct"]
        net_answered = answered - base_answered
        assert gained - lost == net_correct
        reasons = []
        if name == "baseline":
            reasons.append("baseline_reference_only")
        else:
            if net_correct <= 0:
                reasons.append("no_positive_accuracy_gain")
            if abs(net_answered) > max_answer_change:
                reasons.append("answer_rate_change_exceeds_tolerance")
        reports[name] = {
            "condition": deepcopy(spec), "n": n, "counts": counts, "answered": answered,
            "Acc": counts["correct"] / n, "R": answered / n,
            "precision_answered": counts["correct"] / answered if answered else None,
            "baseline_correct": int(base_counts["correct"]), "baseline_answered": int(base_answered),
            "delta_Acc": net_correct / n, "delta_R": net_answered / n,
            "net_correct": int(net_correct), "net_answered": int(net_answered),
            "corrected": transitions["wrong->correct"], "damaged": transitions["correct->wrong"],
            "correct_to_abstain": transitions["correct->abstain"],
            "abstain_to_correct": transitions["abstain->correct"],
            "gained_correct": gained, "lost_correct": lost,
            "original_correct_loss_rate": lost / base_counts["correct"] if base_counts["correct"] else None,
            "transitions": transitions,
            "denominators": {"accuracy": n, "answer_rate": n, "precision_answered": answered,
                             "original_correct_loss_rate": int(base_counts["correct"])},
            "eligible": not reasons, "ineligibility_reasons": reasons,
        }

    selected = {estimator: None for estimator in ESTIMATORS}
    for estimator in ESTIMATORS:
        eligible = [spec for spec in specs if spec.get("context") == estimator and reports[spec["name"]]["eligible"]]
        if eligible:
            best = min(eligible, key=lambda spec: (-reports[spec["name"]]["net_correct"],
                                                   reports[spec["name"]]["lost_correct"],
                                                   float(spec["alpha"]), int(spec["layer"]), spec["name"]))
            selected[estimator] = deepcopy(best)
    return {
        "selected": selected, "reports": reports, "question_ids": ids, "n_questions": n,
        "rule": {"metric": "actual_semantic_QA_outcomes", "baseline": "baseline",
                 "accuracy_definition": "correct / all requested questions",
                 "answer_rate_definition": "(correct + wrong) / all requested questions",
                 "requires_positive_accuracy_gain": True, "max_abs_answer_rate_change": float(tolerance),
                 "max_abs_answer_count_change": max_answer_change,
                 "primary_rank": "maximum integer net_correct",
                 "tie_break": ["fewer correct->wrong plus correct->abstain", "smaller alpha", "shallower layer", "lexical name"],
                 "diagnostics": "rejected", "unresolved": "rejected", "missing_or_extra_questions": "rejected"},
    }
