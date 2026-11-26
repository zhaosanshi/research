"""Replay the frozen gate on existing (QA)^2 generations. Run from the server project root."""

import hashlib
import json
import math
import sys

import numpy as np

sys.path.insert(0, "Iteration_01/diag")
import gate_sim_1126 as gate
import qaqa_pilot_table as qa


def gated_difference(qids, opened, low, high):
    # Outside the gate both policies return the identical original answer.
    score = [high[0][q] - low[0][q] if q in opened else 0 for q in qids]
    count = sum(int(high[1][q]) - int(low[1][q]) for q in opened)
    rng = np.random.default_rng(20261126)
    idx = rng.integers(len(score), size=(10000, len(score)))
    ci = np.quantile(
        np.array(score)[idx].sum(axis=1), [0.025, 0.975], method="inverted_cdf"
    )
    return (sum(score), int(ci[0]), int(ci[1]), count)


def main():
    with open(gate.QAQA_PILOT, encoding="utf8") as f:
        pilot = {json.loads(line)["id"] for line in f if line.strip()}
    for model in gate.ORDER:
        r = qa.Raw(
            model,
            "qaqafull",
            gate.QAQA_BANK,
            ["test_full570", "test_full570_tuned", "test_full570_strong"],
        )
        ids = [q for q in r.ids if q not in pilot]
        root = gate.OUT / model
        fit = json.loads((root / "fit_B_gate_1126.json").read_bytes())
        assert (
            fit["model"] == model
            and fit["pool"] == "B"
            and fit["chosen_layer"] == fit["refit"]["chosen"]["layer"]
        )
        assert (
            hashlib.sha256((root / "capture_gate_1126.json").read_bytes()).hexdigest()
            == fit["capture_sha256"]
        )
        probs = fit["probs"]["chosen"]["qaqa:held"]
        assert len(set(ids)) == len(ids) == 470 and set(probs) == set(ids)
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
        opened = {q for q in ids if probs[q] > 0.5}
        main_e, high_e = gate.STRENGTH[model]
        _, strength, _, cap_e, cap_o, _ = qa.SET[model]
        assert strength == main_e
        original = r.arm_system("x")
        single = r.arm_system(r.e(main_e))
        n1 = r.n1_system(main_e, cap_e, cap_o)
        assert (
            tuple(sum(s[0][q] for q in ids) for s in [original, single, n1])
            == gate.K_QQ[model]
        )
        print(model, "open", len(opened), "of", len(ids), "E", main_e, "->", high_e)
        if not r.has(r.e(high_e)):
            print("  no complete high-E arm")
            continue
        strong = r.arm_system(r.e(high_e))
        print(
            "  high-E minus main-E (net, lo, hi, delta submissions)",
            gated_difference(ids, opened, single, strong),
        )
        caps = [qa.LAD.index(c) for c in [cap_e, cap_o] if c is not None]
        missing = [k for k in qa.LAD[: max(caps) + 1] if not r.has(r.ec(high_e, k))]
        if missing:
            print("  missing judged high-E+C arms", missing)
        else:
            strong_n1 = r.n1_system(high_e, cap_e, cap_o)
            print(
                "  high-N1 minus main-N1 (net, lo, hi, delta submissions)",
                gated_difference(ids, opened, n1, strong_n1),
            )


if __name__ == "__main__":
    main()
