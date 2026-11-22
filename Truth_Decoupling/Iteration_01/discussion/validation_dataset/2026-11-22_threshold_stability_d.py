"""CPU-only. Run in the scoring environment from Truth_Decoupling. No new generations or verdicts.
P chooses the best tune gain; P' chooses the least restrictive threshold within 2 net points of that gain.
For each, refit after leaving out each single calibration answer and record the range of deployment decisions.
The refit range is descriptive, not a confidence interval. For Z=0, a conditional randomization
control drops the same number in each answer-length bin. The original negation exemption is shared.
"""
import sys

import numpy as np

sys.path.insert(0, "Iteration_01/diag")
import selfjudge_probe as sj


def choose(margin, value, slack):
    distinct = np.unique(margin)
    thresholds = np.r_[-np.inf, (distinct[:-1] + distinct[1:]) / 2, np.inf]
    gains = -((margin[None, :] < thresholds[:, None]) @ value)
    return thresholds[np.flatnonzero(gains >= gains.max() - slack)[0]]


def samples(model, condition):
    data = {}
    for external in (False, True):
        doc, _, arrays = sj.read_probe(model, qaqa=external)
        answered = [r for r in doc["items"] if r["answered"]]
        records = [(r, v) for r, v in zip(answered, arrays[condition + "__margin"]) if not r["neg"]]
        for label in (("heldout470",) if external else ("tune", "test")):
            chosen = [(m, r["value"] if external else sj.V[r["state"]], len(r["answer"].split())) for r, m in records
                      if (not r["pilot"] if external else r["split"] == label)]
            data[label] = tuple(np.array(col) for col in zip(*chosen))
    return data


def length_control(margins, values, lengths):
    # Whitespace word-count bins: 1, 2, 3-4, 5-8, 9-16 and 17+. Negative openings already excluded.
    bins = np.digitize(lengths, [1, 2, 4, 8, 16], right=True)
    dropped = margins < 0
    observed = int(-values[dropped].sum())
    rng = np.random.default_rng(20261122)
    random_nets = np.zeros(20000, dtype=int)
    for group in np.unique(bins):
        mask = bins == group
        k = int(dropped[mask].sum())
        populations = np.bincount(values[mask] + 1, minlength=3)
        sampled = rng.multivariate_hypergeometric(populations, k, size=len(random_nets))
        random_nets += sampled[:, 0] - sampled[:, 2]
    tail = (1 + int((random_nets >= observed).sum())) / (1 + len(random_nets))
    return int(dropped.sum()), observed, round(random_nets.mean(), 2), round(tail, 5)


def main():
    for model in sj.ORDER:
        for cond in ("baseline", "E_a3"):
            data = samples(model, cond)
            margin, value, _ = data["tune"]
            for rule, slack in (("P", 0), ("Pprime", 2)):
                t = choose(margin, value, slack)
                fits = [choose(np.delete(margin, i), np.delete(value, i), slack) for i in range(len(margin))]
                print(model, cond, rule, "eligible tune", len(margin), "threshold", f"{t:.6f}",
                      "tune drops", int((margin < t).sum()))
                for dataset in ("test", "heldout470"):
                    margins, values, _ = data[dataset]
                    counts = [int((margins < v).sum()) for v in fits]
                    nets = [int(-values[margins < v].sum()) for v in fits]
                    print(" ", dataset, "dropped", int((margins < t).sum()), "net gain vs E",
                          int(-values[margins < t].sum()), "refit drop range", min(counts), max(counts),
                          "refit net range", min(nets), max(nets))
            for dataset in ("test", "heldout470"):
                print(model, cond, dataset, "Z drop / net / length-control mean / one-sided randomization p",
                      *length_control(*data[dataset]))


if __name__ == "__main__":
    main()
