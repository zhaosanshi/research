"""Read-only cross-bank replay of the fixed double-abstention EC1 rule. Run in archive CPU container from Truth_Decoupling. No answer generation or judge call.
Only issued answers contribute right (+1) or wrong (-1). The same frozen rule and E settings are used throughout.
"""

import sys
import json
import collections
from pathlib import Path
import numpy as np

sys.path[:0] = [
    "Iteration_01/diag",
    "Iteration_01/code/scripts/newsets",
    "Iteration_01/code/scripts/review_checks",
    "Iteration_01/code/scripts/capability",
]
import newsets_votes as NV
import gate_sim_1126 as G
import ns_prompt_table as F
import newsets_score as N
import qaqa_pilot_table as Q
import gsm8k_n1_prompt as M
import cap_score as CS
from judge_two_axis_run import load_verified_single

ORDER = G.ORDER


def triple(v, issued, text):
    return (int(v) if issued else 0, bool(issued), text)


def summarize(a, ids):
    sel = [a[q] for q in ids]
    return (
        sum(c[1] for c in sel),
        sum(c[0] == 1 for c in sel),
        sum(c[0] == -1 for c in sel),
        sum(c[0] for c in sel),
    )


def interval(vals):
    vals = np.array(vals)
    bs = vals[
        np.random.default_rng(20261128).integers(len(vals), size=(3000, len(vals)))
    ].sum(axis=1)
    return [
        int(vals.sum()),
        int(np.quantile(bs, 0.025, method="inverted_cdf")),
        int(np.quantile(bs, 0.975, method="inverted_cdf")),
    ]


def report(tag, m, ids, arms, ec, qual=None):
    double = [q for q in ids if not arms["E"][q][1] and not arms["O"][q][1]]
    assert set(double) <= set(ec)
    dset = set(double)
    combo = {q: ec[q] if q in dset else arms["E"][q] for q in ids}
    arms["rule"] = combo
    ds = {
        n: interval([combo[q][0] - x[q][0] for q in ids])
        for n, x in arms.items()
        if n not in ("rule", "O")
    }
    print(
        "RESULT",
        tag,
        m,
        "n",
        len(ids),
        "double",
        len(double),
        "fill (submitted,R,W,net)",
        summarize(combo, double),
        "natural",
        {n: summarize(x, ids) for n, x in arms.items()},
        "differences (net,95lo,95hi)",
        ds,
        flush=True,
    )
    if qual:
        for k in sorted(set(qual.values())):
            jj = [q for q in double if qual[q] == k]
            print("  branch", k, len(jj), summarize(combo, jj))
    return arms


def formal(task, split, m, b, arms):
    # Pro subset was judged under earlier intervention settings. Reuse a verdict for this model only on exactly matching question + answer text, in the same bank and official subset.
    work = Path(N.NS) / "judge" / task / m / (split + "_pro100")
    NV.load_votes(str(work), str(Path(N.NS) / "gen" / task / m / split))
    _, cells, votes = load_verified_single(
        str(work),
        "pro",
        generation_path=str(work / "judge_input.json"),
        require_complete=False,
    )
    vals = collections.defaultdict(set)
    qids = set()
    badref = set()
    seen_conds = set()
    for i, c in cells.items():
        q = c["pair_id"]
        orig = b[q.split(":", 1)[1]]
        assert c["question"] == orig["question"]
        qids.add(q)
        seen_conds.add(c["condition"])
        if c.get("gold_is_abstention"):
            badref.add(q)
            continue
        x = votes[i]
        if x["status"] == "scored" and c["prediction"].strip():
            vals[q, c["prediction"]].add(Q.VALUE[x["label"]])
    assert (
        len(qids) == 100
        and len({q for q in qids if b[q.split(":", 1)[1]]["pro100"]}) == 100
    )
    # A refused submission is 0 regardless of missing manual verdict; identical texts with joint scores always have zero paired difference.
    eligible = sorted(qids - badref)

    def v(cell, q):
        if not cell[1]:
            return {0}
        known = vals.get((q, cell[2]), set())
        return known if len(known) == 1 else {-1, 0, 1}

    for name in ("E", "C1", "Csrc", "Cdev"):
        point_lo = point_hi = paired = both_scored = pd = 0
        for q in eligible:
            aa, bb = arms["rule"][q], arms[name][q]
            u, w = v(aa, q), v(bb, q)
            dd = (
                {0}
                if aa[1] == bb[1] and (not aa[1] or aa[2] == bb[2])
                else {x - y for x in u for y in w}
            )
            point_lo += min(dd)
            point_hi += max(dd)
            paired += len(dd) == 1
            both_scored += len(u) == len(w) == 1
            if len(dd) == 1:
                pd += next(iter(dd))
        print(
            "FORMAL_SUBSET",
            task,
            split,
            m,
            name,
            "ever-graded subset",
            len(qids),
            "scorable-ref",
            len(eligible),
            "both absolute scores known",
            both_scored,
            "paired difference known incl identical",
            paired,
            "known part",
            pd,
            "whole-subset paired bounds",
            point_lo,
            point_hi,
            flush=True,
        )


def facts():
    mm = N.models()
    dev_choices = {}
    for task in ("popqa", "triviaqa"):
        for m in ORDER:
            sp, _, _, cc, _, _ = F.load(task, m, mm[m], "tune")
            assert len(sp.qs) == 192
            ns = {
                k: sum((sp.st[q, x] == "R") - (sp.st[q, x] == "W") for q in sp.qs)
                for k, x in cc.items()
            }
            dev_choices[task, m] = max(F.LAD, key=lambda k: ns[k])
            print("SOURCE target tune pure C", task, m, dev_choices[task, m])
    for task in ("popqa", "triviaqa"):
        seen = []
        for split in ("test", "test2"):
            b, _ = F.bank(task, split)
            assert len(b) == 500
            seen.append(set(b))
            for m in ORDER:
                args = mm[m]
                el, s, sweep, caps, c = F.MAIN[m]
                ss = F.dash(s)
                sp, tx, have, cs, _, _ = F.load(task, m, args, split)
                assert ss in have and "1" in cs
                ids = sp.qs

                def cell(cond):
                    return {
                        q: triple(
                            (sp.st[q, cond] == "R") - (sp.st[q, cond] == "W"),
                            not sp.blank[q, cond],
                            tx.ec[q, cond],
                        )
                        for q in ids
                    }

                arms = {
                    n: cell(cond)
                    for n, cond in [
                        ("O", "baseline_x"),
                        ("E", sp.cE(ss)),
                        ("C1", cs["1"]),
                        ("Csrc", cs[F.dash(c)]),
                        ("Cdev", cs[dev_choices[task, m]]),
                    ]
                }
                ec = cell(sp.cEC(ss, "1"))
                assert (
                    tuple(summarize(arms[n], ids)[-1] for n in ("O", "E"))
                    == G.K_FACT[(task, split)][m]
                )
                rsp = report(task + "_" + split, m, ids, arms, ec)
                formal(task, split, m, b, rsp)
        assert not seen[0] & seen[1]


def qaqa():
    pilot = {
        json.loads(l)["id"]
        for l in Path(G.QAQA_PILOT).read_text().splitlines()
        if l.strip()
    }
    assert len(pilot) == 100
    qs_diffs = {}
    qtypes = {}
    choice = json.loads(
        (Path(N.NS) / "splits/qaqa_tune_choice_fixedE_20261122.json").read_bytes()
    )
    for m in ORDER:
        workbase = Path(N.NS) / "judge" / "qaqafull" / m
        works = [
            p.name
            for p in sorted(workbase.glob("test_full570*"))
            if (p / "judge2" / "single_pro.jsonl").is_file()
        ]
        rr = Q.Raw(m, "qaqafull", G.QAQA_BANK, works)
        el, s, cl, ce, co, c = Q.SET[m]
        ids = [q for q in rr.ids if q not in pilot]
        assert len(ids) == 470

        def cells(arm):
            rr.need(arm)
            return {
                q: triple(rr.cell(arm, q), not rr.blank(arm, q), rr.text(arm, q))
                for q in ids
            }

        arms = {
            n: cells(name)
            for n, name in [
                ("O", "x"),
                ("E", rr.e(s)),
                ("Csrc", rr.c(c)),
                ("Cdev", rr.c(choice[m]["single_C"])),
            ]
        }
        ec = cells(rr.ec(s, "1"))
        old = rr.n1_system(s, ce, co)
        if rr.has(rr.c("1")):
            arms["C1"] = cells(rr.c("1"))
        else:
            import newsets_gen as NG

            for p in (Path(N.NS) / "gen_prompt").glob(
                "**/qaqafull/" + m + "/test/" + NG.arm_token(rr.c("1")) + ".json"
            ):
                print("QAQA potential supplemental ordinary C1", m, str(p))
            print("QAQA missing C1", m, "complete scored arm unavailable")
        assert (
            tuple(
                sum(sys[0][q] for q in ids)
                for sys in (rr.arm_system("x"), rr.arm_system(rr.e(s)), old)
            )
            == G.K_QQ[m]
        )
        ar = report("qaqa_held_Pro", m, ids, arms, ec, {q: rr.qtype[q] for q in ids})
        print("SOURCE target tune pure C", "qaqa", m, choice[m]["single_C"])
        qs_diffs[m] = {q: ar["rule"][q][0] - ar["E"][q][0] for q in ids}
        qtypes.update({q: rr.qtype[q] for q in ids})
    assert len(qtypes) == 470 and all(set(d) == set(qtypes) for d in qs_diffs.values())
    rng = np.random.default_rng(20261128)
    bootstrap = {}
    mean = {}
    for category in ["failure", "no_failure", "unverifiable", "all"]:
        group = [q for q in qtypes if category == "all" or qtypes[q] == category]
        v = np.array([sum(qs_diffs[m][q] for m in ORDER) for q in group])
        draw = v[rng.integers(len(v), size=(12000, len(v)))].sum(axis=1)
        factor = 100 / (5 * len(v))
        bootstrap[category] = factor * draw
        mean[category] = int(v.sum()) * factor
        print(
            "QAQA aggregate",
            category,
            "distinct questions",
            len(v),
            "five-model net delta",
            int(v.sum()),
            "joint 95%",
            np.quantile(draw, [0.025, 0.975], method="inverted_cdf").tolist(),
            "mean per model per 100 questions",
            round(mean[category], 3),
        )
    print(
        "QAQA exploratory per100 mean-model interaction false-premise less valid-premise",
        round(mean["failure"] - mean["no_failure"], 3),
        "stratified paired-model bootstrap 95%",
        np.round(
            np.quantile(bootstrap["failure"] - bootstrap["no_failure"], [0.025, 0.975]),
            2,
        ).tolist(),
    )


def gsm():
    ids = M.frozen_ids()
    gold = CS.gold_of("gsm8k")
    assert len(ids) == len(set(ids)) == 1319
    for m in ORDER:
        el, s, cl, ce, co, cap = M.SET[m]
        root = Path(M.root(m))
        anchor, o = M.load_arm(m, "neutral", None, ids)
        e_cfg, e = M.load_arm(m, M.e_arm(m), "prompt", ids)
        signature = (
            "identity_sha256",
            "frozen_file_sha256",
            "model_config",
            "system_text",
            "user_template",
        )
        assert all(anchor[k] == e_cfg[k] for k in signature)

        def graded(raw):
            return {
                q: triple((x == "R") - (x == "W"), not M.blank(rr), rr["text"])
                for q, rr in raw.items()
                for x in [CS.judge_row("gsm8k", rr, gold(q))[0]]
            }

        dbl = [q for q in ids if M.blank(o[q]) and M.blank(e[q])]
        bl = [q for q in ids if M.blank(e[q])]
        ec = {}
        if dbl:
            sep = root / "n1_blank_ids_at-prompt.json"
            assert json.loads(sep.read_bytes()) == bl
            ec_cfg, ee = M.load_arm(
                m, M.e_arm(m) + "+C" + str(cl) + ".w.m1", "prompt", bl, sep
            )
            assert (
                all(e_cfg[k] == ec_cfg[k] for k in signature)
                and e_cfg["parts"][0] == ec_cfg["parts"][0]
            )
            ec = graded(ee)
        arms = {"O": graded(o), "E": graded(e)}
        nm = "neutral:C" + str(cl) + ".w.m1"
        targets = sorted(
            Path("Iteration_01/capability").glob(
                "**/gsm8k/" + m + "/b*/" + M.tok(nm) + "-at-prompt.json"
            )
        )
        if (root / (M.tok(nm) + "-at-prompt.json")).exists():
            _, cc = M.load_arm(m, nm, "prompt", ids)
            arms["C1"] = graded(cc)
        else:
            print(
                "GSM missing full matching pure C1",
                m,
                "possible other paths",
                list(map(str, targets)),
            )
        wanted = [(85.4, 86.5), (93.8, 93.5), (95.5, 95.2), (92.6, 92.8), (94.5, 94.5)][
            ORDER.index(m)
        ]
        assert (
            tuple(
                round(100 * summarize(arms[k], ids)[1] / len(ids), 1)
                for k in ("O", "E")
            )
            == wanted
        )
        rep = report("gsm8k_frozen_numeric", m, ids, arms, ec)
        print(
            "  correct and blank counts for historical accuracy cross-check",
            summarize(rep["O"], ids)[1],
            summarize(rep["E"], ids)[1],
            len(bl),
        )


if __name__ == "__main__":
    for task, fun in [("facts", facts), ("qaqa", qaqa), ("gsm", gsm)]:
        if len(sys.argv) == 1 or task in sys.argv[1:]:
            fun()
