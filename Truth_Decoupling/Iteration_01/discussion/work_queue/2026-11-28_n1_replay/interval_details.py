"""Explain the matched Gemma12 gap and check the natural-count table with unsubmitted scores zeroed."""

from policies import inputs, previous, selected_c, verification_receipt


def partition_gap(ids, questions, scores, initial, original_flags, simple, full):
    """Reapply the frozen top-323 rule to the two methods and assign each item to its origin."""
    bank = verification_receipt("TruthfulQA", "gemma3-12b")
    margins = {k: v["baseline"] for k, v in scores.items() if "baseline" in v}
    for k, v in bank["scores"].items():
        assert k not in margins or margins[k] == v
        margins[k] = v

    def retained(cells):
        eligible = [i for i, cell in enumerate(cells) if cell[1]]
        eligible.sort(
            key=lambda i: (
                margins[inputs.fq.statement_sha(questions[i], cells[i][2])],
                i,
            ),
            reverse=True,
        )
        return set(eligible[:323])

    keep_s, keep_n = retained(simple), retained(full)
    assert len(ids) == 522 and len(keep_s) == len(keep_n) == 323
    labels = [
        "E answered" if r[1] else ("original answered" if o else "both abstained")
        for r, o in zip(initial, original_flags)
    ]

    def score(cells, keep, group):
        vals = [
            v
            for i, (v, a, _) in enumerate(cells)
            if a and i in keep and labels[i] == group
        ]
        return len(vals), vals.count(1), vals.count(-1), vals.count(0), sum(vals)

    totals = []
    for group in ("E answered", "original answered", "both abstained"):
        a, b = score(simple, keep_s, group), score(full, keep_n, group)
        print(
            "Gemma12 matched:",
            group,
            labels.count(group),
            "simple/N1 (issued,right,wrong,other,net)",
            a,
            b,
            "difference",
            a[-1] - b[-1],
        )
        totals.append(a[-1] - b[-1])
    assert totals == [6, -33, 2]
    print(
        "Gemma12 natural/matched simple submissions",
        sum(r[1] for r in simple),
        len(keep_s),
        "N1",
        sum(r[1] for r in full),
        len(keep_n),
    )


def main():
    ok, bank = inputs.fq.check_bank()
    assert ok
    pairs = {r["id"]: r["pair_id"] for r in bank}
    for dataset in ("TruthfulQA", "FalseQA"):
        for model in inputs.tq.ORDER:
            ids, _, questions, read, margins = inputs.get_inputs(model, dataset, bank)
            if dataset == "TruthfulQA":
                b = inputs.tq.base(model)
                original = [not b.sp.blank[q, "baseline_x"] for q in ids]
                states, issued, records = inputs.tq.S.system(b.sp, b.tx, b.pick)[:3]
                full = [
                    (inputs.tq.VAL[inputs.tq.G[state]], a, r["text"])
                    for state, a, r in zip(states, issued, records)
                ]
                interval = previous.GS.tq_ci
            else:
                raw, _ = inputs.fq.load_rows(model, set(ids))
                dev, other = inputs.fq.load_rows(
                    model, {r["id"] for r in bank if r["set"] == "dev"}
                )
                st = inputs.fq.setting(model)
                caps = inputs.fq.choose(dev, other, st)[0]["N1"]
                saved = inputs.fq.n1_cells(raw, st[1], caps, ids)
                full = [saved[q][:3] for q in ids]
                original = [r[1] for r in read("O")]

                def interval(delta):
                    return inputs.fq.pair_ci(dict(zip(ids, delta)), pairs, ids)

            e = read("E")
            c = read("C" + selected_c(dataset, model, bank))
            combined = read("EC1")
            simple = [r if r[1] or o else ec for r, o, ec in zip(e, original, combined)]
            for name, arm in [("E", e), ("C", c), ("N1", full)]:
                legacy = interval([a[0] - b[0] for a, b in zip(simple, arm)])
                submitted = interval(
                    [a[0] * int(a[1]) - b[0] * int(b[1]) for a, b in zip(simple, arm)]
                )
                print(
                    "natural",
                    dataset,
                    model,
                    name,
                    "previous",
                    tuple(legacy),
                    "submitted",
                    tuple(submitted),
                    flush=True,
                )
            if dataset == "TruthfulQA" and model == "gemma3-12b":
                partition_gap(ids, questions, margins, e, original, simple, full)


if __name__ == "__main__":
    main()
