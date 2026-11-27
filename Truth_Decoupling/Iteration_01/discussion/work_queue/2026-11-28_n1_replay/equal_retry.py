"""Use the same retry opportunities for a mandatory-answer E comparator. Run from the data root."""

from policies import (
    inputs,
    draw_weights,
    score_ranges,
    difference,
    verification_receipt,
    completed_falseqa_scores,
)
import tq_same_n_decomp as T

ok, qs = inputs.fq.check_bank()
assert ok
for ds in ("TruthfulQA", "FalseQA"):
    for m in inputs.tq.ORDER:
        ids, groups, prompts, read, margins = inputs.get_inputs(m, ds, qs)
        e = read("E")
        ec = read("EC1")
        if ds == "TruthfulQA":
            bank = inputs.tq.base(m)
            orig = [not bank.sp.blank[q, "baseline_x"] for q in ids]
            r, dat = T.load_model(m)
            assert r == ids
            disp, layer, s = T.A.KEY[m]
            run = dat[4]
            forced = dat[-1]
            tx = T.MP.fr.raw_texts(T.MP.work(run, "test", "pospromptdl%d" % layer))
            te = [
                (forced[q][0], forced[q][1], tx[q, "E_L%d_conf-w-p%s" % (layer, s)])
                for q in ids
            ]
            fs = s
        else:
            F = inputs.fq
            raw, _ = F.load_rows(m, set(ids))
            s = F.Q.SET[m][1]
            fs = F.setting(m)[3]
            te = read("TE")
            orig = [x[1] for x in read("O")]
        if ds == "TruthfulQA" and m == "gemma3-12b":
            R = verification_receipt(ds, m)
            for key, val in R["scores"].items():
                assert key not in margins or margins[key]["baseline"] == val
            margins = {
                **margins,
                **{k: {"baseline": v} for k, v in R["scores"].items()},
            }
        elif ds == "FalseQA" and m in ("llama3.1-8b", "gemma3-12b"):
            margins = completed_falseqa_scores(m, ids, prompts, read, margins)
        selected = [i for i in range(len(ids)) if not e[i][1] and not orig[i]]
        a = list(e)
        b = list(e)
        for i in selected:
            a[i] = ec[i]
            b[i] = te[i]
        quota = sum(v[1] for v in e)
        wg = draw_weights(groups)
        sa = score_ranges(a, quota, wg, prompts, margins)
        sb = score_ranges(b, quota, wg, prompts, margins)
        nat = sum(x[0] for x in a if x[1]) - sum(x[0] for x in b if x[1])
        na = sum(a[i][1] for i in selected)
        nb = sum(b[i][1] for i in selected)
        print(
            "EQUAL-RETRY",
            ds,
            m,
            "E-doses",
            s,
            fs,
            "double-blanks",
            len(selected),
            "fills",
            na,
            nb,
            "natural EC-forcedE",
            nat,
            "same coverage net (point range, 95 range)",
            difference(sa, sb),
            flush=True,
        )
