"""Compare policies at fixed intervention strength on the archived (QA)^2 data.
Run from the Truth_Decoupling server data root, using CPU only.
"""

import json
import sys
import hashlib

import numpy as np

sys.path.insert(0, "Iteration_01/diag")
import gate_sim_1126 as g
import qaqa_pilot_table as qa
import falseqa_read as falseqa
import selfjudge_stack_pick as ranker


def bootstrap(delta, rows):
    ci = np.quantile(delta[rows].sum(axis=1), [0.025, 0.975], method="inverted_cdf")
    return (int(delta.sum()), int(ci[0]), int(ci[1]))


def main():
    with open(g.QAQA_PILOT, encoding="utf8") as stream:
        pilot = {json.loads(line)["id"] for line in stream}
    with open(g.QAQA_BANK, encoding="utf8") as stream:
        question = {r["id"]: r["question"] for r in map(json.loads, stream)}
    for model in ("qwen3.5-9b", "gemma3-27b"):
        bank = qa.Raw(
            model,
            "qaqafull",
            g.QAQA_BANK,
            ["test_full570", "test_full570_tuned", "test_full570_strong"],
        )
        ids = [q for q in bank.ids if q not in pilot]
        fit_path = g.OUT / model
        fit = json.loads((fit_path / "fit_B_gate_1126.json").read_bytes())
        assert fit["model"] == model and fit["pool"] == "B"
        assert (
            hashlib.sha256(
                (fit_path / "capture_gate_1126.json").read_bytes()
            ).hexdigest()
            == fit["capture_sha256"]
        )
        probs = fit["probs"]["chosen"]["qaqa:held"]
        assert set(probs) == set(ids) and len(ids) == 470
        p = np.array([probs[q] for q in ids])
        assert np.all((0 <= p) & (p <= 1))
        on = p > 0.5
        low, high = g.STRENGTH[model]
        _, strength, _, cap_e, cap_o, _ = qa.SET[model]
        assert strength == low
        original = bank.arm_system("x")
        lower = bank.n1_system(low, cap_e, cap_o)
        higher = bank.n1_system(high, cap_e, cap_o)

        def array(system, k):
            return np.array([system[k][q] for q in ids], dtype=int)

        orig, lo, hi = (array(s, 0) for s in (original, lower, higher))
        assert (
            sum(orig),
            sum(array(bank.arm_system(bank.e(low)), 0)),
            sum(lo),
        ) == g.K_QQ[model]
        inds = np.random.default_rng(20261127).integers(
            len(ids), size=(10000, len(ids))
        )
        gated = np.where(on, hi, orig)
        policy_delta = gated - hi
        d = hi - lo

        def targeting(mask, gains, axis):
            return (mask * gains).sum(axis=axis) - mask.mean(axis=axis) * gains.sum(
                axis=axis
            )

        resampled = targeting(on[inds], d[inds], 1)
        print(model, "n", len(ids), "open", on.sum(), "dose", low, "->", high)
        print(
            "natural net ungated low/high, gated low/high:",
            sum(lo),
            sum(hi),
            sum(np.where(on, lo, orig)),
            sum(gated),
        )
        print(
            "high fixed policies gate - always (point,lo,hi):",
            bootstrap(policy_delta, inds),
        )
        print(
            "issued high gated/always:",
            np.where(on, array(higher, 1), array(original, 1)).sum(),
            array(higher, 1).sum(),
        )
        print(
            "higher - lower behind gate:",
            bootstrap(d * on, inds),
            "outside gate:",
            bootstrap(d * (~on), inds),
        )
        print(
            "same upgrade budget, gate - expectation of random upgrades, low on rest:",
            round(targeting(on, d, 0), 1),
            np.quantile(resampled, [0.025, 0.975]).round(1).tolist(),
        )
        for kind in sorted({bank.qtype[q] for q in ids}):
            selected = np.array([bank.qtype[q] == kind for q in ids])
            group_gate = policy_delta[selected]
            rows = np.random.default_rng(20261127).integers(
                len(group_gate), size=(10000, len(group_gate))
            )
            print(
                "premise",
                kind,
                "n",
                sum(selected),
                "off",
                sum((~on) & selected),
                "high fixed gate - always",
                bootstrap(group_gate, rows),
                "always high minus original, all in type",
                bootstrap((hi - orig)[selected], rows),
            )
        # Both policies use the same answer on open items. Match only among the rest.
        off = [q for q, active in zip(ids, on) if not active]
        gap = sum(original[1][q] - higher[1][q] for q in off)
        larger = original if gap > 0 else higher
        eligible = [q for q in off if larger[1][q]]
        margins = ranker.lookup(model)
        values = {}
        for q in eligible:
            arm = "x"
            if gap < 0:
                arm = bank.e(high)
                if bank.blank(arm, q):
                    cap = cap_o if bank.blank("x", q) else cap_e
                    arm = next(
                        bank.ec(high, k)
                        for k in qa.LAD[: qa.LAD.index(cap) + 1]
                        if not bank.blank(bank.ec(high, k), q)
                    )
            assert bank.cell(arm, q) == larger[0][q]
            values[q] = falseqa.margin(
                margins, question[q], (larger[0][q], True, bank.text(arm, q))
            )
        known = sorted((q for q in eligible if values[q] is not None), key=values.get)
        assert all(np.isfinite(values[q]) for q in known)
        missing = sorted(larger[0][q] for q in eligible if values[q] is None)
        possible = []
        for n_unknown in range(
            max(0, abs(gap) - len(known)), min(len(missing), abs(gap)) + 1
        ):
            base = sum(larger[0][q] for q in known[: abs(gap) - n_unknown])
            for dropped in (missing[:n_unknown], list(reversed(missing))[:n_unknown]):
                possible.append(
                    int(policy_delta.sum()) - int(np.sign(gap)) * (base + sum(dropped))
                )
        print(
            "equal submissions, gate minus always high, bounds over all missing confidence ranks",
            min(possible),
            max(possible),
            "extra submissions gate minus always",
            gap,
            "missing ranks",
            len(missing),
            "of",
            len(eligible),
        )
        if not missing:
            trimmed = set(known[: abs(gap)])
            matched = policy_delta - np.array(
                [int(np.sign(gap)) * larger[0][q] if q in trimmed else 0 for q in ids]
            )
            print(
                "equal submissions fixed rankings (point,lo,hi)",
                bootstrap(matched, inds),
            )
        delivered_delta = np.where(on, hi, orig) * np.where(
            on, array(higher, 1), array(original, 1)
        ) - hi * array(higher, 1)
        print(
            "high fixed policies gate - always, score only submissions:",
            bootstrap(delivered_delta, inds),
        )


if __name__ == "__main__":
    main()
