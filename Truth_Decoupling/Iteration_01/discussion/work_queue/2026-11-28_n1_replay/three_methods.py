"""Compare the cd558a9 two-step method, full N1 and the fixed-dose double-abstention rule."""

import hashlib
import json

from policies import (
    completed_falseqa_scores,
    difference,
    draw_weights,
    inputs,
    merge_scores,
    previous,
    score_ranges,
    submitted_score,
    verification_receipt,
)


def run(dataset, model, questions):
    ids, groups, prompts, read, margins = inputs.get_inputs(model, dataset, questions)
    if dataset == "TruthfulQA":
        tune = previous.tqa_split(model, 0)
        bank = inputs.tq.base(model)
        assert bank.ids == ids
        original_answered = [not bank.sp.blank[q, "baseline_x"] for q in ids]
        states, issued, rec = inputs.tq.S.system(bank.sp, bank.tx, bank.pick)[:3]
        full = [
            (inputs.tq.VAL[inputs.tq.G[v]], a, r["text"])
            for v, a, r in zip(states, issued, rec)
        ]
        reference = inputs.tq.GS.K_TQ[model][3]
    else:
        tune = previous.fqa_split(model, questions, "dev")
        dev, other = inputs.fq.load_rows(
            model, {r["id"] for r in questions if r["set"] == "dev"}
        )
        raw, _ = inputs.fq.load_rows(model, set(ids))
        settings = inputs.fq.setting(model)
        caps = inputs.fq.choose(dev, other, settings)[0]["N1"]
        rows = inputs.fq.n1_cells(raw, settings[1], caps, ids)
        full = [rows[q][:3] for q in ids]
        original_answered = [x[1] for x in read("O")]
        reference = inputs.tq.GS.K_FQA[model][2]
        if model in ("llama3.1-8b", "gemma3-12b"):
            margins = completed_falseqa_scores(model, ids, prompts, read, margins)
    assert sum(x[0] for x in full) == reference
    tids, dev, lad, _ = tune
    two_dose = previous.pick(
        {
            k: sum(previous.two_step(tids, dev["E"], dev["EC" + k], dev["_eblank"]))
            for k in lad
        },
        lad,
    ).replace("-", ".")
    single_c_dose = previous.pick({k: sum(dev["C" + k]) for k in lad}, lad).replace(
        "-", "."
    )
    e, c = read("E"), read("C" + single_c_dose)
    ec_two, ec_one = read("EC" + two_dose), read("EC1")
    two = [a if a[1] else b for a, b in zip(e, ec_two)]
    simple = [
        a if a[1] or orig else b for a, orig, b in zip(e, original_answered, ec_one)
    ]
    if dataset == "TruthfulQA" and model == "gemma3-12b":
        receipt = verification_receipt(dataset, model)
        assert (
            receipt["frozen_roster_sha256"]
            == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        )
        source = {"C1": read("C1"), "C05": read("C0.5"), "both_blank_EC1": simple}
        for name, arm in source.items():
            roster = [
                (
                    q,
                    x[1],
                    inputs.probe.sha1(t),
                    inputs.probe.sha1(x[2]),
                    inputs.fq.statement_sha(t, x[2]),
                )
                for q, t, x in zip(ids, prompts, arm)
            ]
            assert (
                hashlib.sha256(
                    json.dumps(
                        roster, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                == receipt["arms_manifest_sha256"][name]
            )
        margins = merge_scores(receipt, source.values(), prompts, margins)
    arms = {"E": e, "C": c, "two_step": two, "N1": full, "fixed": simple}
    quota = min(submitted_score(a)[1] for a in arms.values())
    assert quota == min(submitted_score(e)[1], submitted_score(c)[1])
    weights = draw_weights(groups)
    matched = {
        name: score_ranges(a, quota, weights, prompts, margins)
        for name, a in arms.items()
    }
    print(
        dataset,
        model,
        "quota",
        quota,
        "chosen_two",
        two_dose,
        "chosen_C",
        single_c_dose,
        flush=True,
    )
    for name, arm in arms.items():
        print(
            " ",
            name,
            "matched net",
            (int(matched[name][0][0]), int(matched[name][1][0])),
            "natural (submitted net,count)",
            submitted_score(arm),
            "legacy natural net",
            sum(x[0] for x in arm),
        )
    # Use the same resampling function as cd558a9 for the natural-count check.
    pair = dict(zip(ids, groups))
    for name, comparator in (("E", e), ("C", c)):
        vals = [x[0] * int(x[1]) - y[0] * int(y[1]) for x, y in zip(two, comparator)]
        if dataset == "TruthfulQA":
            natural = previous.GS.tq_ci(vals)
        else:
            natural = inputs.fq.pair_ci(dict(zip(ids, vals)), pair, ids)
        print("  two_step natural minus", name, "(submitted only)", list(natural))
    for name, comparator in (
        ("two_step", "E"),
        ("two_step", "C"),
        ("N1", "E"),
        ("N1", "C"),
        ("fixed", "E"),
        ("fixed", "C"),
        ("two_step", "N1"),
        ("two_step", "fixed"),
        ("N1", "fixed"),
    ):
        print(
            " ",
            name,
            "minus",
            comparator,
            "[sample min,max, interval low,high]",
            difference(matched[name], matched[comparator]),
        )
    print(flush=True)


if __name__ == "__main__":
    ok, bank = inputs.fq.check_bank()
    assert ok
    for dataset in ("TruthfulQA", "FalseQA"):
        for model in inputs.tq.ORDER:
            run(dataset, model, bank)
