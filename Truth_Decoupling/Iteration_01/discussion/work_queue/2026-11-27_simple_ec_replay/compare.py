"""Replay three fixed treatments on ordinary prompts. Compare a direct write with a single retry after E abstains."""

from pathlib import Path
import json
import sys

import numpy as np

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
]
import falseqa_read as fq
import tqa_gate2b_1127 as tq
import selfjudge_probe as probe
import selfjudge_stack_pick as verifier
import falseqa_gate2_1127 as stored_falseqa
import fill_only2  # noqa: F401 - registers the identifiers of archived verification sets.


def all_margins(model):
    probe.SOURCES.update(
        fq.SRC["smoke"] + fq.SRC["full"] + stored_falseqa.sources_all()
    )
    extras = []
    for p in sorted((Path(probe.OUT) / model).glob("*_probe.json")):
        name = p.name.removesuffix("_probe.json")
        if name in verifier.REUSE or name == "sp_items":
            continue
        if json.loads(p.read_bytes()).get("batch_size") == 1:
            if name not in probe.SOURCES:
                raise ValueError(f"unknown verifier source {name}")
            extras.append(name)
    return verifier.lookup(model, extra_sources=tuple(extras))


def get_inputs(model, dataset, questions):
    if dataset == "FalseQA":
        held = [r for r in questions if r["set"] == "heldout"]
        ids = [r["id"] for r in held]
        raw, _ = fq.load_rows(model, set(ids))
        assert ids == raw.ids
        arms = fq.roster(model)
        groups = [r["pair_id"] for r in held]

        def read(name):
            arm = arms[name]
            return [
                (raw.cell(arm, q), not raw.blank(arm, q), raw.text(arm, q)) for q in ids
            ]

        return ids, groups, [r["question"] for r in held], read, all_margins(model)
    b = tq.base(model)
    ids, sp, tx = b.ids, b.sp, b.tx
    with open(probe.OUT + "/" + model + "/items.json") as f:
        items = {
            r["id"]: r["question"]
            for r in json.load(f)["items"]
            if r["split"] == "test"
        }

    def read(name):
        if name == "E":
            cond = sp.cE(b.sd)
        elif name.startswith("EC"):
            cond = sp.cEC(b.sd, name[2:].replace(".", "-"))
        else:
            cond = sp.cC(name[1:].replace(".", "-"))
        return [
            (tq.VAL[tq.G[sp.st[q, cond]]], not sp.blank[q, cond], tx.ec[q, cond])
            for q in ids
        ]

    return ids, ids, [items[q] for q in ids], read, all_margins(model)


def extrema_net(score, issued, ranks, target):
    """Exact sample or bounds over all orderings of missing scores, with a fixed tie order."""
    live = np.flatnonzero(issued)
    n_drop = len(live) - target
    if not n_drop:
        return int(score.sum()), int(score.sum()), 0, score.copy()
    valid = [i for i in live if ranks[i] is not None]
    missing = np.sort([score[i] for i in live if ranks[i] is None])
    ranked = sorted(valid, key=lambda i: ranks[i])
    if not len(missing):
        v = score.copy()
        v[ranked[:n_drop]] = 0
        return int(v.sum()), int(v.sum()), 0, v
    known_prefix = np.r_[0, np.cumsum(score[ranked])]
    candidates = []
    for m in range(max(0, n_drop - len(ranked)), min(n_drop, len(missing)) + 1):
        known = known_prefix[n_drop - m]
        for unknown in [missing[:m], missing[::-1][:m]]:
            candidates.append(int(score.sum() - known - unknown.sum()))
    return min(candidates), max(candidates), len(missing), None


def run(model, dataset, questions):
    ids, group, texts, read, margins = get_inputs(model, dataset, questions)
    names = ("E", "C", "EC", "retry")
    doses = ("0.5", "0.25", "1", "2")
    keys = list(dict.fromkeys(group))
    positions = [np.array([i for i, g in enumerate(group) if g == key]) for key in keys]
    assert len(set(map(len, positions))) == 1 and len(set(ids)) == len(ids)
    if dataset == "FalseQA":
        assert len(ids) == 600 and len(positions) == 300
    samples = np.random.default_rng(20261127).integers(
        len(keys), size=(3000, len(keys)), dtype=np.int16
    )
    cache = {}
    for dose in doses:
        e, c, ec = read("E"), read("C" + dose), read("EC" + dose)
        retry = [first if first[1] else second for first, second in zip(e, ec)]
        all_scores = []
        counts = []
        rank = []
        for arm in [e, c, ec, retry]:
            scored = np.array([s if a else 0 for s, a, _ in arm])
            delivered = np.array([a for _, a, _ in arm], dtype=int)
            assert all(not a or t.strip() for (_, a, t) in arm)
            all_scores.append(scored)
            counts.append(delivered)
            rank.append(
                [
                    fq.margin(margins, question, response)
                    for question, response in zip(texts, arm)
                ]
            )
            assert all(v is None or np.isfinite(v) for v in rank[-1])
        print(
            dataset,
            model,
            "C",
            dose,
            "questions",
            len(ids),
            "natural (issued, net) "
            + " ".join(
                str((int(a.sum()), int(s.sum()))) for a, s in zip(counts, all_scores)
            ),
            flush=True,
        )
        cache[dose] = (all_scores, counts, rank)
        for combo in [2, 3]:
            arms = [0, 1, combo]
            target = min(int(counts[i].sum()) for i in arms)
            picked = [
                extrema_net(all_scores[i], counts[i], rank[i], target) for i in arms
            ]
            print(
                " ",
                names[combo],
                "K",
                target,
                "net bounds",
                [r[:3] for r in picked],
                "minus E,C bounds",
                [
                    (picked[-1][0] - other[1], picked[-1][1] - other[0])
                    for other in picked[:2]
                ],
            )
            pos = np.stack(positions)
            if picked[0][3] is not None and picked[-1][3] is not None:
                d = (picked[-1][3] - picked[0][3])[pos].sum(axis=1)
                print(
                    "  exact difference vs E, CI",
                    int(d.sum()),
                    np.quantile(
                        d[samples].sum(axis=1), [0.025, 0.975], method="inverted_cdf"
                    )
                    .astype(int)
                    .tolist(),
                )
            if (
                combo == 3
                and dose == "0.25"
                and picked[-1][2] == 0
                and target == sum(counts[0])
            ):
                live = np.flatnonzero(counts[3])
                eligible = sorted(live, key=lambda i: rank[3][i])
                dropped = set(eligible[: len(live) - target])
                added = [i for i in live if not counts[0][i] and i not in dropped]
                replaced = [i for i in dropped if counts[0][i]]
                assert len(added) == len(replaced)
                print(
                    "  one-retry vs E: matched changes (new/removed counts, new/removed net)",
                    len(added),
                    len(replaced),
                    int(all_scores[3][added].sum()),
                    int(all_scores[0][replaced].sum()),
                )
            if all(p[3] is not None for p in picked):
                groupscores = [p[3][np.stack(positions)].sum(axis=1) for p in picked]

                def paired(base):
                    d = groupscores[-1] - groupscores[base]
                    return [int(d.sum())] + np.quantile(
                        d[samples].sum(axis=1), [0.025, 0.975], method="inverted_cdf"
                    ).astype(int).tolist()

                print(
                    "  matched margin differences and intervals:", paired(0), paired(1)
                )
            common = np.all(np.stack([counts[i] for i in arms]), axis=0)

            def within(base):
                v = (all_scores[combo] - all_scores[base]) * common
                groupsum = v[np.stack(positions)].sum(axis=1)
                return [int(v.sum())] + np.quantile(
                    groupsum[samples].sum(axis=1), [0.025, 0.975], method="inverted_cdf"
                ).astype(int).tolist()

            print(
                "  same answered questions, count and paired differences:",
                int(common.sum()),
                within(0),
                within(1),
            )
            # No verifier calls: expected net when each arm uniformly retains exactly K of its actual submissions.
            sc = np.stack(
                [all_scores[i][np.stack(positions)].sum(axis=1) for i in arms]
            )
            ct = np.stack([counts[i][np.stack(positions)].sum(axis=1) for i in arms])
            nets = sc.sum(axis=1) * target / ct.sum(axis=1)
            bct = ct[:, samples].sum(axis=2)
            bsc = sc[:, samples].sum(axis=2)
            rep = bsc * np.min(bct, axis=0) / bct
            print(
                "  uniform K, expected score differences [95%] vs E,C:",
                [
                    (
                        round(float(nets[-1] - nets[j]), 1),
                        np.quantile(rep[-1] - rep[j], [0.025, 0.975]).round(1).tolist(),
                    )
                    for j in [0, 1]
                ],
                flush=True,
            )

    return ids, group, cache


def draw_bounds(score, issued, ranks, target, weights):
    """For each paired resample bound the fixed-quota policy over all missing-rank orderings."""
    k = int(issued.sum()) - target
    total = weights @ score
    if k == 0:
        return total, total
    valid = [i for i, a in enumerate(issued) if a and ranks[i] is not None]
    ranked = sorted(valid, key=lambda i: ranks[i])
    absent = [i for i, a in enumerate(issued) if a and ranks[i] is None]
    if not absent:
        exact = total - weights[:, ranked[:k]] @ score[ranked[:k]]
        return exact, exact
    wk = weights[:, ranked] * score[ranked]
    wu = np.sort(weights[:, absent] * score[absent], axis=1)

    def pref(rows):
        return np.column_stack(
            (np.zeros(len(rows), dtype=int), np.cumsum(rows, axis=1))
        )

    known = pref(wk)
    small = pref(wu)
    large = pref(wu[:, ::-1])
    low, high = (
        np.full(len(weights), np.iinfo(np.int64).max),
        np.full(len(weights), np.iinfo(np.int64).min),
    )
    for n in range(max(0, k - len(ranked)), min(len(absent), k) + 1):
        rest = total - known[:, k - n]
        low = np.minimum(low, rest - large[:, n])
        high = np.maximum(high, rest - small[:, n])
    return low, high


def pool(dataset, models):
    shared = set.intersection(*(set(ids) for ids, _, _ in models))
    ids = [q for q in models[0][0] if q in shared]
    labels = dict(zip(models[0][0], models[0][1]))
    keys = list(dict.fromkeys(labels[q] for q in ids))
    lookup = {key: i for i, key in enumerate(keys)}
    rng = np.random.default_rng(20261127)
    ix = rng.integers(len(keys), size=(4000, len(keys)))
    base = np.array(
        [np.bincount(row, minlength=len(keys)) for row in ix], dtype=np.int16
    )
    weights = np.vstack((np.ones(len(keys), dtype=np.int16), base))[
        :, [lookup[labels[q]] for q in ids]
    ]
    for dose in ("0.5", "0.25"):
        for arm in [2, 3]:
            lo = np.zeros((2, len(weights)), dtype=int)
            hi = lo.copy()
            total_k = 0
            random_difference = np.zeros(2)
            for original, _, cache in models:
                positions = [original.index(q) for q in ids]
                scores, counts, ranks = cache[dose]
                k = min(int(counts[i][positions].sum()) for i in [0, 1, arm])
                total_k += k
                draw = {}
                rand = {}
                for i in [0, 1, arm]:
                    s = scores[i][positions]
                    a = counts[i][positions]
                    r = [ranks[i][j] for j in positions]
                    draw[i] = draw_bounds(s, a, r, k, weights)
                    expected = extrema_net(s, a, r, k)[:2]
                    assert (draw[i][0][0], draw[i][1][0]) == expected
                    rand[i] = s.sum() * k / a.sum()
                for i in [0, 1]:
                    lo[i] += draw[arm][0] - draw[i][1]
                    hi[i] += draw[arm][1] - draw[i][0]
                    random_difference[i] += rand[arm] - rand[i]
            print(
                "POOLED",
                dataset,
                "common",
                len(ids),
                "C",
                dose,
                ["EC", "retry"][arm - 2],
                "total K",
                total_k,
                flush=True,
            )
            for i, name in enumerate(["E", "C"]):
                print(
                    "  minus",
                    name,
                    "sample range",
                    [int(lo[i, 0]), int(hi[i, 0])],
                    "outer paired 95%",
                    [
                        int(np.quantile(lo[i, 1:], 0.025, method="inverted_cdf")),
                        int(np.quantile(hi[i, 1:], 0.975, method="inverted_cdf")),
                    ],
                    "uniform quota expected difference",
                    round(float(random_difference[i]), 1),
                    flush=True,
                )


if __name__ == "__main__":
    ok, questions = fq.check_bank()
    assert ok
    for dataset in ("FalseQA", "TruthfulQA"):
        results = [run(model, dataset, questions) for model in tq.ORDER]
        pool(dataset, results)
