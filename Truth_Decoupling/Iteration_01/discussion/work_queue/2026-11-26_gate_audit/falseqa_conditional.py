"""Compare the frozen router to uniform selection with the same number of open gates.
Repeat within the two gold premise classes as a diagnostic.
"""

import sys
import json
import hashlib
import numpy as np

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
    "Iteration_01/code/scripts/capability",
]
import falseqa_read as F
import gate_sim_1126 as G


def verify_gate(m, task, ids):
    root = G.OUT / m
    fit = json.loads((root / "fit_B_gate_1126.json").read_bytes())
    cap = root / "capture_gate_1126.json"
    if (
        fit["model"] != m
        or fit["pool"] != "B"
        or fit["chosen_layer"] != fit["refit"]["chosen"]["layer"]
        or hashlib.sha256(cap.read_bytes()).hexdigest() != fit["capture_sha256"]
    ):
        raise ValueError("wrong gate source")
    gg = fit["probs"]["chosen"][task]
    if not set(ids).issubset(gg) or any(not 0 <= gg[q] <= 1 for q in ids):
        raise ValueError("invalid probability bank")
    return [q for q in ids if gg[q] > 0.5]


def main():
    _, bank = F.check_bank()
    dev = {r["id"] for r in bank if r["set"] == "dev"}
    held = {r["id"] for r in bank if r["set"] == "heldout"}
    bank = [r for r in bank if r["id"] in held]
    ids = [r["id"] for r in bank]
    mids = {r["id"]: i for i, r in enumerate(bank)}
    pr = {r["id"]: r["pair_id"] for r in bank}
    units = sorted(set(pr.values()))
    pairs = [[mids[r["id"]] for r in bank if r["pair_id"] == u] for u in units]
    assert all(
        len(u) == 2 and len(set(bank[i]["premise"] for i in u)) == 2 for u in pairs
    )
    ii = np.array(pairs)
    labels = np.array([r["premise"] == "false" for r in bank])
    X = []
    A = []
    for m in G.ORDER:
        st = F.setting(m)
        tr, tr2 = F.load_rows(m, dev)
        caps, _ = F.choose(tr, tr2, st)
        r, r2 = F.load_rows(m, held)
        sy = F.systems_of(r, r2, st, caps, ids)
        on = set(verify_gate(m, "fqa_held", ids))
        X.append([sy["N1"][q][0] - sy["O"][q][0] for q in ids])
        A.append([q in on for q in ids])
    X = np.array(X)
    A = np.array(A)

    def calc(idx):
        margins = []
        z = labels[idx]
        a = A[:, idx]
        x = X[:, idx]
        obs = (a * x).sum(1)
        plain = x.sum(1) * a.mean(1)
        bytyp = sum(
            x[:, z == sub].sum(1) * a[:, z == sub].mean(1) for sub in (True, False)
        )
        return obs - plain, obs - bytyp

    actual = calc(np.arange(600))
    rng = np.random.default_rng(20261126)
    st = np.array(
        [
            calc(ii[rng.choice(len(units), len(units), replace=True)].flatten())
            for _ in range(4000)
        ]
    )
    for j, ref in enumerate(("random on all 600", "random within each premise type")):
        print(
            ref,
            "models",
            actual[j].round(1),
            "pooled gate gain",
            actual[j].sum(),
            np.quantile(st[:, j, :].sum(1), [0.025, 0.975]),
        )


if __name__ == "__main__":
    main()
