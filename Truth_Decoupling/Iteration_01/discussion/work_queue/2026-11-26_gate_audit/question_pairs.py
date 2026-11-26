"""CPU-only, held-out FalseQA paired checks; feature and text content are not printed."""

import json
import sys
import re
from collections import Counter

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/capability",
]
import gate_capture_1126 as cap
import gate_fit_1126 as fit


def main():
    m = fit.MODELS[0]
    cfg = json.load(open(cap.FROZEN_MODELS))["models"][m]
    S = cap.items_of(m, cfg)
    train = [x[1] for x in S["fqa_train"]["items"]]
    held = S["fqa_held"]["items"]
    groups = {}
    for q, text, label, pair in held:
        groups.setdefault(pair, {})[label] = q
    assert len(groups) == 300 and all(set(g) == {0, 1} for g in groups.values())
    # Surface near-duplicate screen: exact multiset Jaccard of lowercase word unigrams.
    token = lambda s: set(re.findall(r"\w+", s.casefold()))
    trains = [token(s) for s in train]
    near = {}
    for q, s, _, _ in held:
        a = token(s)
        near[q] = max(len(a & b) / len(a | b) for b in trains)
    print(
        "held >=.8 token jaccard to any FalseQA training question",
        sum(t >= 0.8 for t in near.values()),
        "out of",
        len(near),
    )
    for m in fit.MODELS:
        db = json.load(open(fit.OUT / m / "fit_B_gate_1126.json"))
        prob = db["probs"]["chosen"]["fqa_held"]
        matches = Counter()
        ranked = 0
        nearok = farok = 0
        for g in groups.values():
            ft, vt = prob[g[1]], prob[g[0]]
            ranked += int(ft > vt)
            matches[(int(ft > 0.5), int(vt > 0.5))] += 1
        for q, _, t, _ in held:
            v = int((prob[q] > 0.5) == bool(t))
            if near[q] >= 0.8:
                nearok += v
            else:
                farok += v
        print(
            m,
            "pairs correctly ordered",
            ranked,
            "/300",
            "correct-both, reversed, both-on, both-off",
            [matches[k] for k in [(1, 0), (0, 1), (1, 1), (0, 0)]],
            "questions correct on near, remaining",
            nearok,
            farok,
        )


if __name__ == "__main__":
    main()
