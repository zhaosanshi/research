"""Replay the fixed 9B and PopQA matched-answer controls (read only).

Run from the Truth_Decoupling data root with Python 3 and NumPy:
  python3 -B PATH/2026-11-20_replay_main_controls.py
The runtime data tree supplies Iteration_01/diag and the archived generations,
Pro verdicts and confidence scores. All output goes to stdout.
"""
from collections import Counter
from pathlib import Path
import math
import sys

sys.dont_write_bytecode = True
DIAG = Path("Iteration_01/diag").resolve()
if not (DIAG / "main_table_same_n.py").is_file():
    raise SystemExit("Run from the Truth_Decoupling data root containing Iteration_01/diag/")
sys.path.insert(0, str(DIAG))

import numpy as np
import main_table_same_n as matched
import newsets_score as newsets


def value(state):
    if state not in ("R", "W", "H", "X"):
        raise ValueError("Unresolved or unknown verdict: %r" % state)
    return int(state == "R") - int(state == "W")


def qwen9_control():
    data = matched.Data("Qwen3.5-9B", 1)
    sp = data.sp
    fixed = matched.fr.records(sp, ("ALL", "2", "0-5"), data.texts)
    forced = matched.fr.records(sp, ("DE", "2"), data.texts)
    if [r["p"] for r in fixed] != sp.qs or [r["p"] for r in forced] != sp.qs or len(sp.qs) != len(set(sp.qs)):
        raise ValueError("Unaligned or repeated 9B question IDs")
    answered = matched.fr.answered(fixed, sp, data.dblank)
    forced_answered = matched.fr.answered(forced, sp, data.dblank)
    confidence = data.rows("conf_E2", data.m["d"][1], sp.dE("2"))
    order = matched.ranked(data, forced_answered, confidence)
    if any(not math.isfinite(confidence[sp.qs[i]]["mean_logp"]) for i in order):
        raise ValueError("Non-finite 9B ranking score")
    target = sum(answered)
    if len(order) < target:
        raise ValueError("9B comparator cannot reach the actual answer count")
    retained = set(order[:target])
    groups = {(a, b): Counter() for a in (True, False) for b in (True, False)}
    for i, (a, b) in enumerate(zip(fixed, forced)):
        selected = i in retained
        b_state = b["state"] if selected else "X"
        count = groups[(answered[i], selected)]
        count["n"] += 1
        count["fixed right"] += a["state"] == "R"
        count["fixed wrong"] += a["state"] == "W"
        count["control right"] += b_state == "R"
        count["control wrong"] += b_state == "W"
        delta = value(a["state"]) - value(b_state)
        count["net difference"] += delta
        if selected and a["text"].strip() == b["text"].strip():
            count["same text"] += 1
            count["different score on same text"] += delta != 0
            count["net from same text"] += delta
    print("9B TruthfulQA fixed E2+C0.5 vs forced E2; actual answers %d vs %d" % (target, len(retained)))
    fields = ("n", "fixed right", "fixed wrong", "control right", "control wrong", "net difference",
              "same text", "different score on same text", "net from same text")
    print("fixed answered, control retained, " + ", ".join(fields))
    for key, counts in groups.items():
        print(", ".join(map(str, [*key, *(counts[field] for field in fields)])))
    total = sum(count["net difference"] for count in groups.values())
    same = sum(count["net from same text"] for count in groups.values())
    print("Net difference: %d; using the control vote on identical returned text: %d" % (total, total - same))


def top_scores(weights, eligible, confidence, target, verdicts):
    """Each bootstrap sample has its own E answer count and therefore its own top N."""
    indices = sorted(np.flatnonzero(eligible), key=lambda i: (-confidence[i], i))
    ranked_weights = weights[:, indices]
    if np.any(ranked_weights.sum(axis=1) < target):
        raise ValueError("Matched N cannot be reached in the observed or a resampled cohort")
    before = ranked_weights.cumsum(axis=1) - ranked_weights
    retained = np.clip(target[:, None] - before, 0, ranked_weights)
    return retained @ verdicts[indices]


def summary(samples):
    low, high = np.quantile(samples[1:], [0.025, 0.975], method="nearest")
    return "%+d [%+d, %+d]" % (samples[0], low, high)


def popqa_controls():
    task = "popqa"
    ids = newsets.expected(task, "test", pro100=True)
    if len(ids) != 100 or len(set(ids)) != len(ids):
        raise ValueError("Expected the fixed first-batch Pro 100 distinct IDs")
    # Same bank order for every model and every draw; row zero gives the original scores.
    rng = np.random.default_rng(20261120)
    draws = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=4000)
    weights = np.vstack([np.ones(len(ids), dtype=int), draws])
    pooled = np.zeros((3, len(weights)), dtype=int)
    total_n = 0
    print("\nPopQA first Pro 100; A ranks ordinary no-write answers by forced no-write confidence; "
          "B ranks and submits forced no-write answers")
    print("model, actual E answers, E net, A net, B net, A minus E, B minus E")
    for model, config in newsets.models().items():
        split = newsets.shell(config["el"], config["cl"])
        st, blank, text, excluded, _ = newsets.pro_tables(task, model, "test_pro100", newsets.need_test(split), ids)
        if excluded:
            raise ValueError("First-batch cohort changed for %s; excluded: %s" % (model, excluded))
        base, e, forced = "baseline_x", split.cE("2"), "baseline_conf"
        checked = newsets.conf_rows(task, model, "test", "conf", {q: text[q, forced] for q in ids})
        confidence = np.array([float("nan") if checked[q]["mean_logp"] is None else checked[q]["mean_logp"] for q in ids])
        base_answers = np.array([not blank[q, base] for q in ids])
        forced_answers = np.array([not blank[q, forced] for q in ids]) & np.isfinite(confidence)
        target = weights @ np.array([int(not blank[q, e]) for q in ids])
        e_net = weights @ np.array([value(st[q, e]) for q in ids])
        base_vals = np.array([value(st[q, base]) for q in ids])
        forced_vals = np.array([value(st[q, forced]) for q in ids])
        a_net = top_scores(weights, base_answers & forced_answers, confidence, target, base_vals)
        b_net = top_scores(weights, forced_answers, confidence, target, forced_vals)
        print("%s, %d, %d, %d, %d, %s, %s" % (
            model, target[0], e_net[0], a_net[0], b_net[0], summary(a_net - e_net), summary(b_net - e_net)))
        pooled += np.array([e_net, a_net, b_net])
        total_n += target[0]
    e_net, a_net, b_net = pooled
    print("five-model total, %d, %d, %d, %d, %s, %s" % (
        total_n, e_net[0], a_net[0], b_net[0], summary(a_net - e_net), summary(b_net - e_net)))


if __name__ == "__main__":
    qwen9_control()
    popqa_controls()
