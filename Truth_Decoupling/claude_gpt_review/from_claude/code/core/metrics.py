"""Evaluation metrics (numpy only).

Statement level (balanced true/false sets):
  TPR = P(say True | statement true), TNR = P(say False | statement false)
  J   = TPR + TNR - 1           (Youden's J: discrimination)
  B   = P(say True)             (label bias; a pure bias shift moves B, not J)
  Rans = 1 - P(abstain)         (answer rate)

QA level:
  three-way decomposition of an intervention relative to baseline:
    corrected  : wrong -> right
    damaged    : right -> wrong
    to_abstain : wrong -> abstain  (calibration, not new knowledge)
    plus right -> abstain (over-abstention) and abstain -> * transitions.
  known / unknown split by baseline correctness.

Also: Wilson CI, bootstrap CI, paired bootstrap of differences, McNemar exact test,
Spearman rho, ECE, AUROC, risk-coverage / AURC, selective accuracy.
"""
import math

import numpy as np

# --------------------------------------------------------------------------- #
# Statement level
# --------------------------------------------------------------------------- #

LABEL_TRUE, LABEL_FALSE, LABEL_ABSTAIN = "True", "False", "Abstain"


def confusion_tf(y_true, y_pred):
    """y_true: iterable of bool; y_pred: iterable of {"True","False","Abstain"}.

    Returns dict with counts and rates. Abstentions are excluded from TPR/TNR/J
    (they are reported through answer rate), matching the plan: J is measured
    on answered items.
    """
    y_true = np.asarray(list(y_true), dtype=bool)
    y_pred = np.asarray(list(y_pred), dtype=object)
    n = len(y_true)
    answered = y_pred != LABEL_ABSTAIN
    say_true = y_pred == LABEL_TRUE
    say_false = y_pred == LABEL_FALSE

    pos = y_true & answered
    neg = (~y_true) & answered
    tp = int((pos & say_true).sum())
    fn = int((pos & say_false).sum())
    tn = int((neg & say_false).sum())
    fp = int((neg & say_true).sum())

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    n_ans = int(answered.sum())
    return {
        "n": n, "n_answered": n_ans, "answer_rate": n_ans / n if n else float("nan"),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "tpr": tpr, "tnr": tnr,
        "J": (tpr + tnr - 1.0) if not (math.isnan(tpr) or math.isnan(tnr)) else float("nan"),
        "B": (tp + fp) / n_ans if n_ans else float("nan"),
        "acc_answered": (tp + tn) / n_ans if n_ans else float("nan"),
        "acc_all": (tp + tn) / n if n else float("nan"),
    }


def dissociation_matrix(base, e_only, c_only, both=None, rand=None):
    """Build the 2x2 (direction x measure) matrix from confusion dicts.

    Rows: intervention (E, C); columns: delta J, delta answer rate (points, x100).
    Ideal: E changes J not Rans; C changes Rans not J -> anti-diagonal ~ 0.
    """
    def d(a, b, k):
        return 100.0 * (a[k] - b[k])
    out = {
        "E": {"dJ": d(e_only, base, "J"), "dRans": d(e_only, base, "answer_rate"), "dB": d(e_only, base, "B")},
        "C": {"dJ": d(c_only, base, "J"), "dRans": d(c_only, base, "answer_rate"), "dB": d(c_only, base, "B")},
    }
    if both is not None:
        out["Both"] = {"dJ": d(both, base, "J"), "dRans": d(both, base, "answer_rate"), "dB": d(both, base, "B")}
    if rand is not None:
        out["Random"] = {"dJ": d(rand, base, "J"), "dRans": d(rand, base, "answer_rate"), "dB": d(rand, base, "B")}
    out["anti_diagonal_max_abs"] = max(abs(out["E"]["dRans"]), abs(out["C"]["dJ"]))
    return out


# --------------------------------------------------------------------------- #
# QA level
# --------------------------------------------------------------------------- #

QA_CORRECT, QA_WRONG, QA_ABSTAIN = "correct", "wrong", "abstain"


def qa_rates(outcomes):
    outcomes = list(outcomes)
    n = len(outcomes)
    c = sum(o == QA_CORRECT for o in outcomes)
    w = sum(o == QA_WRONG for o in outcomes)
    a = sum(o == QA_ABSTAIN for o in outcomes)
    return {"n": n, "correct": c / n if n else float("nan"), "wrong": w / n if n else float("nan"),
            "abstain": a / n if n else float("nan"), "answer_rate": (c + w) / n if n else float("nan"),
            "precision_answered": c / (c + w) if (c + w) else float("nan")}


def three_way_decomposition(base_outcomes, new_outcomes):
    """Transition counts between baseline and intervened QA outcomes."""
    base_outcomes, new_outcomes = list(base_outcomes), list(new_outcomes)
    assert len(base_outcomes) == len(new_outcomes)
    n = len(base_outcomes)
    trans = {}
    for b, a in zip(base_outcomes, new_outcomes):
        trans[(b, a)] = trans.get((b, a), 0) + 1
    g = lambda b, a: trans.get((b, a), 0)
    out = {
        "n": n,
        "corrected": g(QA_WRONG, QA_CORRECT),
        "damaged": g(QA_CORRECT, QA_WRONG),
        "wrong_to_abstain": g(QA_WRONG, QA_ABSTAIN),
        "correct_to_abstain": g(QA_CORRECT, QA_ABSTAIN),
        "abstain_to_correct": g(QA_ABSTAIN, QA_CORRECT),
        "abstain_to_wrong": g(QA_ABSTAIN, QA_WRONG),
        "unchanged": g(QA_CORRECT, QA_CORRECT) + g(QA_WRONG, QA_WRONG) + g(QA_ABSTAIN, QA_ABSTAIN),
    }
    for k in list(out.keys()):
        if k != "n":
            out[k + "_pct"] = 100.0 * out[k] / n if n else float("nan")
    return out


def known_unknown_split(base_outcomes):
    """Indices of items the baseline answered correctly (known) vs not (unknown)."""
    known = [i for i, o in enumerate(base_outcomes) if o == QA_CORRECT]
    unknown = [i for i, o in enumerate(base_outcomes) if o != QA_CORRECT]
    return known, unknown


def stratified_report(base_outcomes, new_outcomes):
    known, unknown = known_unknown_split(base_outcomes)
    sub = lambda idx, arr: [arr[i] for i in idx]
    return {
        "known": {"n": len(known), **three_way_decomposition(sub(known, base_outcomes), sub(known, new_outcomes)),
                  "rates": qa_rates(sub(known, new_outcomes))},
        "unknown": {"n": len(unknown), **three_way_decomposition(sub(unknown, base_outcomes), sub(unknown, new_outcomes)),
                    "rates": qa_rates(sub(unknown, new_outcomes))},
        "all": {"n": len(base_outcomes), **three_way_decomposition(base_outcomes, new_outcomes),
                "rates": qa_rates(new_outcomes)},
    }


# --------------------------------------------------------------------------- #
# Uncertainty / statistics
# --------------------------------------------------------------------------- #

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values, stat=np.mean, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        stats[b] = stat(values[idx])
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def paired_bootstrap_delta(a, b, stat=np.mean, n_boot=2000, seed=0, alpha=0.05):
    """CI for stat(a) - stat(b) with paired resampling (same indices)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    assert len(a) == len(b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        deltas[i] = stat(a[idx]) - stat(b[idx])
    return {"delta": float(stat(a) - stat(b)),
            "ci": (float(np.quantile(deltas, alpha / 2)), float(np.quantile(deltas, 1 - alpha / 2)))}


def mcnemar_exact(b, c):
    """Exact two-sided McNemar test on discordant counts b (A right, B wrong) and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def _rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rho(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(scores, labels):
    """Rank-based AUROC with tie handling. labels: 1 = positive."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sign_invariant_auroc(scores, labels):
    a = auroc(scores, labels)
    return max(a, 1 - a) if not math.isnan(a) else a


def ece(confidences, correct, n_bins=10):
    confidences, correct = np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if m.any():
            total += m.mean() * abs(confidences[m].mean() - correct[m].mean())
    return float(total)


def risk_coverage(confidences, correct):
    """Risk-coverage curve: sort by confidence desc; risk = error rate among covered.

    Returns coverage array, risk array and AURC (area under risk-coverage).
    """
    confidences, correct = np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float)
    order = np.argsort(-confidences, kind="mergesort")
    err = 1.0 - correct[order]
    n = len(err)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    aurc = float(np.sum((coverage[1:] - coverage[:-1]) * (risk[1:] + risk[:-1]) / 2.0)) if n > 1 else float("nan")
    return coverage, risk, aurc


def selective_accuracy(confidences, correct, coverage_level):
    confidences, correct = np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float)
    k = max(1, int(round(coverage_level * len(correct))))
    order = np.argsort(-confidences, kind="mergesort")[:k]
    return float(correct[order].mean())


def summarize_bool(flags):
    flags = [bool(f) for f in flags]
    k, n = sum(flags), len(flags)
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "wilson95": [lo, hi]}
