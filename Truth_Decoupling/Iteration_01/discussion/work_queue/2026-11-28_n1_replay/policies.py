"""Replay fixed simplified N1 rules on archived generations, from the Truth_Decoupling server data root.

The sibling compare.py supplies the verified original answer-bank and verifier loaders.
Each method uses the same pre-existing E layer and dose. An unsubmitted answer scores zero.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "2026-11-27_simple_ec_replay")
)
import compare as inputs
import simple_ec_1127 as previous


def submitted_score(cells):
    return sum(v for v, answered, _ in cells if answered), sum(
        answered for _, answered, _ in cells
    )


def selected_c(dataset, model, rows):
    if dataset == "TruthfulQA":
        _, development, ladder, _ = previous.tqa_split(model, 0)
        scores = {k: sum(development["C" + k]) for k in ladder}
    else:
        raw, _ = inputs.fq.load_rows(
            model, {r["id"] for r in rows if r["set"] == "dev"}
        )
        ladder = inputs.fq.LAD
        scores = {k: sum(raw.cell(raw.c(k), q) for q in raw.ids) for k in ladder}
    return previous.pick(scores, ladder).replace("-", ".")


def verification_receipt(dataset, model):
    payload = Path(__file__).with_name("verification_scores.json").read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest()
        == "04390cac6f2dc2cbd58fbddbef223e9967dbd2b4b488c86708d701d62ceb66ec"
    )
    return json.loads(payload)[dataset + "/" + model]


def merge_scores(receipt, cells_by_arm, texts, margins):
    extra = receipt["scores"]
    needed = {
        inputs.fq.statement_sha(question, row[2])
        for arm in cells_by_arm
        for question, row in zip(texts, arm)
        if row[1] and inputs.fq.statement_sha(question, row[2]) not in margins
    }
    assert set(extra) == needed and all(np.isfinite(v) for v in extra.values())
    return {**margins, **{key: {"baseline": value} for key, value in extra.items()}}


def completed_falseqa_scores(model, ids, texts, read, margins):
    receipt = verification_receipt("FalseQA", model)
    arm = inputs.fq.roster(model)["C2"]
    generated = inputs.fq.gdir(model) / (inputs.fq.G.arm_token(arm) + ".json")
    assert receipt["arm"] == arm and receipt["bank_sha256"] == inputs.fq.sha_file(
        inputs.fq.BANK
    )
    assert (
        receipt["generation_sha256"] == inputs.fq.sha_file(generated)
        and len(ids) == len(texts) == 600
    )
    return merge_scores(receipt, [read("C2")], texts, margins)


def draw_weights(groups):
    units = list(dict.fromkeys(groups))
    indices = {g: i for i, g in enumerate(units)}
    rng = np.random.default_rng(20261128)
    draws = rng.integers(len(units), size=(3000, len(units)))
    weights = np.vstack(
        [np.ones(len(units), dtype=int)]
        + [np.bincount(row, minlength=len(units)) for row in draws]
    )
    return weights[:, [indices[g] for g in groups]]


def score_ranges(arm, quota, weights, prompts, margins):
    issued = np.array([int(a) for _, a, _ in arm])
    net = np.array([v if a else 0 for v, a, _ in arm])
    ranks = [inputs.fq.margin(margins, p, t) for p, t in zip(prompts, arm)]
    assert all(x is None or np.isfinite(x) for x in ranks)
    lo, hi = inputs.draw_bounds(net, issued, ranks, quota, weights)
    assert (int(lo[0]), int(hi[0])) == inputs.extrema_net(net, issued, ranks, quota)[:2]
    return lo, hi


def difference(scores, comparator):
    low = scores[0] - comparator[1]
    high = scores[1] - comparator[0]
    return [
        int(low[0]),
        int(high[0]),
        int(np.quantile(low[1:], 0.025, method="inverted_cdf")),
        int(np.quantile(high[1:], 0.975, method="inverted_cdf")),
    ]


def run(dataset, model, questions, supplement):
    ids, groups, prompts, read, margins = inputs.get_inputs(model, dataset, questions)
    if dataset == "FalseQA":
        base = read("O")
        raw, _ = inputs.fq.load_rows(model, set(ids))
        dev_raw, dev_sent = inputs.fq.load_rows(
            model, {r["id"] for r in questions if r["set"] == "dev"}
        )
        settings = inputs.fq.setting(model)
        caps = inputs.fq.choose(dev_raw, dev_sent, settings)[0]["N1"]
        canonical = inputs.fq.n1_cells(raw, settings[1], caps, ids)
        n1 = [canonical[q][:3] for q in ids]
        if model in ("llama3.1-8b", "gemma3-12b") and supplement:
            margins = completed_falseqa_scores(model, ids, prompts, read, margins)
    else:
        bank = inputs.tq.base(model)
        assert ids == bank.ids
        states, issued, records = inputs.tq.S.system(bank.sp, bank.tx, bank.pick)[:3]
        n1 = [
            (inputs.tq.VAL[inputs.tq.G[state]], flag, row["text"])
            for state, flag, row in zip(states, issued, records)
        ]
        caps = tuple(
            x.replace("-", ".") if x is not None else None
            for x in (bank.cap_e, bank.cap_own)
        )
        base = [
            (0, not bank.sp.blank[(q, "baseline_x")], bank.tx.ec[(q, "baseline_x")])
            for q in ids
        ]
    e = read("E")
    stages = {k: read("EC" + k) for k in ("0.25", "0.5", "1", "2")}
    methods = {"N1": n1}
    at_caps = []
    for i, first in enumerate(e):
        original = base[i]
        cap = caps[0 if original[1] else 1]
        picked = first
        if not first[1] and cap is not None:
            candidate = stages[cap][i]
            equal_original = original[1] and (
                inputs.fq.Q.ro.norm(original[2]) == inputs.fq.Q.ro.norm(candidate[2])
            )
            if candidate[1] and not equal_original:
                picked = candidate
        at_caps.append(picked)
    methods["one_at_old_cap"] = at_caps
    for k, arm in stages.items():
        # The third pass uses the original question unchanged. The old answers are never added to its prompt.
        methods["both_blank_" + k] = [
            second if not first[1] and not original[1] else first
            for first, original, second in zip(e, base, arm)
        ]
        methods["veto_" + k] = [
            first
            if (
                first[1]
                or not second[1]
                or (
                    original[1]
                    and inputs.fq.Q.ro.norm(original[2])
                    == inputs.fq.Q.ro.norm(second[2])
                )
            )
            else second
            for first, original, second in zip(e, base, arm)
        ]
    methods["two_light_rungs"] = [
        a if a[1] else (b if b[1] else c)
        for a, b, c in zip(e, stages["0.25"], stages["0.5"])
    ]
    if dataset == "TruthfulQA" and model == "gemma3-12b" and supplement:
        receipt = verification_receipt(dataset, model)
        assert (
            receipt["frozen_roster_sha256"]
            == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        )
        arms = {
            "C1": read("C1"),
            "C05": read("C0.5"),
            "both_blank_EC1": methods["both_blank_1"],
        }
        for name, arm in arms.items():
            roster = [
                (
                    q,
                    a[1],
                    inputs.probe.sha1(t),
                    inputs.probe.sha1(a[2]),
                    inputs.fq.statement_sha(t, a[2]),
                )
                for q, t, a in zip(ids, prompts, arm)
            ]
            assert (
                hashlib.sha256(
                    json.dumps(
                        roster, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                == receipt["arms_manifest_sha256"][name]
            )
        margins = merge_scores(receipt, arms.values(), prompts, margins)
    chosen = selected_c(dataset, model, questions)
    net_e, n_e = submitted_score(e)
    weights = draw_weights(groups)
    eligible = sum(not a[1] and not b[1] for a, b in zip(e, base))
    for dose in dict.fromkeys((chosen, "1")):
        only_c = read("C" + dose)
        net_c, n_c = submitted_score(only_c)
        quota = min(n_e, n_c)
        range_e = score_ranges(e, quota, weights, prompts, margins)
        range_c = score_ranges(only_c, quota, weights, prompts, margins)
        range_n1 = score_ranges(n1, quota, weights, prompts, margins)
        print(
            dataset,
            model,
            "C_dev",
            chosen,
            "C_test",
            dose,
            "K",
            quota,
            "n",
            len(ids),
            "E_submitted",
            n_e,
            "both_blank",
            eligible,
            flush=True,
        )
        for name, method in methods.items():
            score, n = submitted_score(method)
            assert n >= quota
            matched = score_ranges(method, quota, weights, prompts, margins)
            uniform = [
                round(quota * (score / n - net_e / n_e), 2),
                round(quota * (score / n - net_c / n_c), 2),
            ]
            print(
                " ",
                name,
                "vs_E",
                difference(matched, range_e),
                "vs_C",
                difference(matched, range_c),
                "vs_N1",
                difference(matched, range_n1),
                "natural_delta",
                score - net_e,
                score - net_c,
                "uniform_delta",
                uniform,
                flush=True,
            )


if __name__ == "__main__":
    supplement = "--before-supplement" not in sys.argv
    ok, bank = inputs.fq.check_bank()
    assert ok
    for ds in ("TruthfulQA", "FalseQA"):
        for model in inputs.tq.ORDER:
            run(ds, model, bank, supplement)
