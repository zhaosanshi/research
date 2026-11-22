"""CPU replay in the scoring container, from the Truth_Decoupling root. No calls to a model or judge."""
import json
import random
import sys

import numpy as np

sys.path.insert(0, "Iteration_01/diag")
import n1_on_sentence as N


def prompt_arms(model, index, ids):
    display, _, _ = N.A.KEY[model]
    data = N.MP.MT.Data(display, index)
    split = ("tune", "test")[index]
    run = data.m["ec"][index].split("/eval_extra/")[0]
    arms = {}
    for directory in ("promptsweep", "h22c4", "pfam"):
        path = N.MP.work(run, split, directory)
        if not N.MP.has(path):
            raise ValueError(path)
        verdicts, blanks = N.ps.load(path, "pro")
        for cond in sorted({c for _, c in verdicts if c.startswith("baseline_")}):
            name = cond[len("baseline_"):]
            if not all(verdicts.get((q, cond), "?") != "?" for q in ids):
                continue
            rows = {q: (N.V[verdicts[q, cond]], not blanks[q, cond]) for q in ids}
            if name in arms and arms[name] != rows:
                raise ValueError((model, split, name))
            arms[name] = rows
    return arms


def load(model):
    tune_ids, test_ids = (N.restricted(model, i) for i in (0, 1))
    tune = prompt_arms(model, 0, tune_ids)
    test = prompt_arms(model, 1, test_ids)
    names = sorted(tune)
    chosen = max(names, key=lambda c: (*N.net(tune[c], tune_ids), c))
    if chosen not in test:
        raise ValueError("chosen arm missing on test")
    recs = N.tune_model(model)
    rec = max(recs, key=lambda r: (r["SN1"][0], -float(N.dot(r["s"]))))
    stored = json.load(open(N.CHOICE % model, encoding="utf-8"))
    if (stored["strength"], tuple(stored["cfg"])) != (rec["s"], tuple(rec["cfg"])):
        raise ValueError("choice differs from the tune-only replay")
    sp, tx = N.build(model, 1, "x_E2", rec["s"], test_ids)
    legacy = N.A.KEY[model][2]
    native_sp, native_tx = N.build(model, 1, "x", legacy, test_ids)
    old_t, old_tx = N.build(model, 0, "x", legacy, tune_ids)
    n1 = N.run_sys(native_sp, native_tx, N.ML.n1_pick(old_t, old_tx, legacy))
    print(model, "chosen prompt", chosen, "stack strength", rec["s"], flush=True)
    return dict(PB=test[chosen], P=test["x_E2"], SN1=N.run_sys(sp, tx, tuple(rec["cfg"])),
                SE=N.run_sys(sp, tx, ("E",rec["s"])), N1=n1)


def limits(rows, n):
    wrong, neutral, right = rows
    worst = -np.minimum(n, wrong) + np.maximum(0, n-wrong-neutral)
    best = np.minimum(n, right) - np.maximum(0, n-right-neutral)
    return worst, best


def bounds(methods, a, b, ids, weights):
    low = np.zeros(len(weights), dtype=int)
    high = low.copy()
    total = low.copy()
    signed = low.copy()
    for arms in methods:
        aggregates = []
        for key in (a, b):
            vals = np.array([arms[key][q] for q in ids], dtype=int)
            counts = np.array([weights @ ((vals[:,0]==v) & (vals[:,1]==1)) for v in (-1,0,1)])
            aggregates.append(counts)
            signed += (1 if key == a else -1) * (weights @ vals[:,0])
        n = np.minimum(aggregates[0].sum(axis=0), aggregates[1].sum(axis=0))
        la, ua = limits(aggregates[0],n)
        lb, ub = limits(aggregates[1],n)
        low += la-ub
        high += ua-lb
        total += n
    def ci(x):
        y = np.sort(x[1:])
        return [int(y[100]), int(y[3899])]
    print(a, "minus", b, "natural net and interval", signed[0], ci(signed),
          "equal count per side", total[0], "all-trimming sample range", [int(low[0]),int(high[0])],
          "paired outer interval", [ci(low)[0],ci(high)[1]], flush=True)


def main():
    sets = [load(m) for m in N.ORDER]
    for k in (3,5):
        data = sets[:k]
        ids = sorted(set.intersection(*(set(d) for ms in data for d in ms.values())))
        if len(ids) != (521 if k == 3 else 520):
            raise ValueError(("common roster", len(ids)))
        rng = random.Random(20261116)
        n = len(ids)
        weights = np.array([np.ones(n, dtype=int)] +
                           [np.bincount([rng.randrange(n) for _ in range(n)], minlength=n) for _ in range(4000)])
        print(k,"models",n,"questions")
        for a,b in (("N1","P"),("SN1","P"),("SN1","PB"),("SE","PB"),("SN1","SE")):
            bounds(data,a,b,ids,weights)


if __name__ == "__main__":
    main()
