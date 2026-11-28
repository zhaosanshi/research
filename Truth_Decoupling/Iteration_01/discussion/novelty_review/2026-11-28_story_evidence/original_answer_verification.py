"""Do archived self-ratings of original answers differ between answers E suppresses and retains?
The rating is a separate, unmodified verifier on the same model; no scores are produced here.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "work_queue"
        / "2026-11-27_simple_ec_replay"
    ),
)
import compare as inputs


def summary(removed, kept):
    x = np.array([m for m in removed if m is not None], dtype=float)
    y = np.array([m for m in kept if m is not None], dtype=float)
    if not len(x) or not len(y):
        return [len(x), len(y), None, None, None]
    assert np.isfinite(x).all() and np.isfinite(y).all()
    ranks = (x[:, None] < y).sum() + 0.5 * (x[:, None] == y).sum()
    total = len(removed) * len(kept)
    return [
        len(x),
        len(y),
        round(float(ranks / (len(x) * len(y))), 3),
        round(float(ranks / total), 3),
        round(float((ranks + total - len(x) * len(y)) / total), 3),
    ]


def boot_interval(left, right, groups):
    """Paired item (TruthfulQA) or pair (FalseQA) resampling, retaining both types together."""
    units = list(dict.fromkeys(groups))
    mapper = {g: i for i, g in enumerate(units)}
    x = np.array([m for _, m in left], dtype=float)
    y = np.array([m for _, m in right], dtype=float)
    gx = np.array([mapper[groups[i]] for i, _ in left])
    gy = np.array([mapper[groups[i]] for i, _ in right])
    ix = np.argsort(y)
    y = y[ix]
    gy = gy[ix]
    a = np.searchsorted(y, x, "left")
    b = np.searchsorted(y, x, "right")
    rng = np.random.default_rng(20261128)
    outcomes = []
    for _ in range(20):
        picks = rng.integers(len(units), size=(100, len(units)))
        w = np.array([np.bincount(k, minlength=len(units)) for k in picks])
        wx = w[:, gx]
        wy = w[:, gy]
        n_y = wy.sum(axis=1)
        prefix = np.column_stack((np.zeros(len(wy)), np.cumsum(wy, axis=1)))
        credits = ((n_y[:, None] - 0.5 * (prefix[:, a] + prefix[:, b])) * wx).sum(
            axis=1
        )
        denom = n_y * wx.sum(axis=1)
        outcomes.extend(credits[denom > 0] / denom[denom > 0])
    return np.quantile(outcomes, [0.025, 0.975]).round(3).tolist()


def main():
    ok, bank = inputs.fq.check_bank()
    assert ok
    for ds in ("TruthfulQA", "FalseQA"):
        for model in inputs.tq.ORDER:
            ids, groups, questions, read, margins = inputs.get_inputs(model, ds, bank)
            if ds == "TruthfulQA":
                b = inputs.tq.base(model)
                original = [
                    (
                        inputs.tq.VAL[inputs.tq.G[b.sp.st[q, "baseline_x"]]],
                        not b.sp.blank[q, "baseline_x"],
                        b.tx.ec[q, "baseline_x"],
                    )
                    for q in ids
                ]
            else:
                original = read("O")
            e = read("E")
            vals = {k: [[], []] for k in ("all", "wrong", "right")}
            valid = [[], []]
            for idx, (orig, revised, prompt) in enumerate(zip(original, e, questions)):
                val, issued, _ = orig
                if not issued:
                    continue
                rating = inputs.fq.margin(margins, prompt, orig)
                bucket = int(revised[1])
                vals["all"][bucket].append(rating)
                if rating is not None:
                    valid[bucket].append((idx, rating))
                if val == 1:
                    vals["right"][bucket].append(rating)
                if val == -1:
                    vals["wrong"][bucket].append(rating)
            print(
                ds,
                model,
                "P(suppressed original rated lower than retained); n_measured / all, probability, bounds if missing",
            )
            print("all paired interval:", boot_interval(*valid, groups))
            for key, (suppressed, retained) in vals.items():
                print(
                    key,
                    len(suppressed),
                    len(retained),
                    summary(suppressed, retained),
                    flush=True,
                )


if __name__ == "__main__":
    main()
