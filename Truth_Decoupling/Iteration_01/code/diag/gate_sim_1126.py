#!/usr/bin/env python3
# Iteration_01 exploration (executor shen, Claude Opus 5.5, 2026-11-26), work queue
# discussion/work_queue/2026-11-26_queue_p.md, section 3: readings 1-5 of the per-question switch ("gate"). Read-only:
# the switch probabilities of diag/gate_fit_1126.py and EXISTING answers and verdicts; no card, no judge call, nothing
# chosen here (trap = probability > 0.5). Not reviewed by GPT. Exploration readings; the held-out verdicts of
# TruthfulQA, FalseQA and (QA)^2 had been read before in other comparisons (the switch was not trained on them).
# Gated system: on a question where the switch is ON hand in the cell of the system (N1, or single E), where it is OFF
#   hand in the original model's cell (no write). Natural answer counts (nothing is equalised).
# Loaders (unchanged programs; their known nets are reproduced first and a mismatch stops before any reading):
#   FalseQA held-out 600: diag/falseqa_read.py as diag/falseqa_ruleZ_1126.py uses it (check_bank, setting, load_rows on
#     the 100 development questions, choose, load_rows on the held-out, systems_of): O, E, N1, Conly. Known nets: all 600
#     of results/write_positions/falseqa_ruleZ_1126_heldout_20261126.txt. Intervals: falseqa_read.pair_ci (pairs).
#   TruthfulQA test (main list per model): exactly as diag/e_n1_threeway_1126.py (iti_table: MP.prompt_split, the main
#     list cx, ML.n1_pick on the tune split, S.system for base / E / the N1 pick). Known nets:
#     results/write_positions/e_n1_threeway_1126_test_20261126.txt. Intervals: policy_select2.ci (questions).
#   (QA)^2 held-out 470: diag/qaqa_pilot_table.Raw on the full bank with the work dir test_full570, the frozen TruthfulQA
#     settings qaqa_pilot_table.SET (as that program's default run with --task qaqafull ... --skip-ids): O = "x",
#     E = R.e(s), N1 = R.n1_system(s, caps). Known nets: results/newsets/qaqa/qaqa_frozen_470_five_models_20261122.txt.
#     Intervals: newsets_score.joint_ci (questions).
#   PopQA / TriviaQA test and test2: diag/ns_prompt_table.load and newsets_score.system (alias match), as
#     diag/n1_on_sentence2.fact reads "no write" and "only E". Known nets:
#     results/write_positions/n1_on_sentence_factqa_five_models_20261122.txt. Intervals: newsets_score.joint_ci.
#   MMLU 1,000: capability/ecmmlu_1126/<model>.json (cap_mcq_sweep rows): arms x, x:E<el>.w.p<s>, x:E<el>.w.p<s2>;
#     intervals cap_mcq_sweep.ci (items).
# AUROC intervals (reading 1): 2,000 resamples, seed 20261126, percentiles 2.5 / 97.5; FalseQA resamples pairs, (QA)^2
#   resamples questions.
# usage (container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/gate_sim_1126.py [known]
#   ("known": only the loaders, the known nets and the forced checks, before any fit file exists)
import json
import random
import sys
from pathlib import Path

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/code/scripts/capability"]
OUT = Path("Iteration_01/newsets/gate_1126")
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
EC = {"llama3.1-8b": (16, 19), "qwen3.5-9b": (15, 16), "qwen3.8-27b": (28, 33), "gemma3-12b": (24, 25), "gemma3-27b": (28, 31)}
STRENGTH = {"llama3.1-8b": ("3.5", "5"), "qwen3.5-9b": ("2", "3"), "qwen3.8-27b": ("1.75", "3"), "gemma3-12b": ("2.5", "4"),
            "gemma3-27b": ("1", "2")}
POOLS = ("A", "B")
TAGS = (("chosen", "chosen layer"), ("E", "E layer"), ("C", "C layer"))
QAQA_BANK = "Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl"
QAQA_PILOT = "Iteration_01/newsets/bank/qaqa_pilot_bank_20261122.jsonl"
FACT = (("popqa", "test"), ("popqa", "test2"), ("triviaqa", "test"), ("triviaqa", "test2"))
# known nets (source files named in the header)
K_FQA = {"llama3.1-8b": (100, 137, 240, 179), "qwen3.5-9b": (312, 313, 334, 333), "qwen3.8-27b": (197, 223, 341, 288),
         "gemma3-12b": (152, 169, 228, 202), "gemma3-27b": (187, 275, 328, 190)}            # O, E, N1, Conly (600)
K_FQA_UPPER = {"llama3.1-8b": (310, 202), "qwen3.5-9b": (382, 356), "qwen3.8-27b": (396, 260), "gemma3-12b": (333, 270),
               "gemma3-27b": (348, 290)}     # plan section 0: perfect switch (premise false), gated N1 / gated E
K_TQ = {"llama3.1-8b": (523, -86, 40, 62), "qwen3.5-9b": (523, 73, 168, 186), "qwen3.8-27b": (521, 47, 142, 175),
        "gemma3-12b": (522, 12, 103, 158), "gemma3-27b": (523, 3, 105, 144)}               # n, O, E, N1
K_QQ = {"llama3.1-8b": (-105, 17, 4), "qwen3.5-9b": (-24, 63, 55), "qwen3.8-27b": (-5, 63, 57), "gemma3-12b": (-59, 65, 63),
        "gemma3-27b": (-31, 40, 42)}                                                         # O, E, N1 (470)
K_FACT = {("popqa", "test"): {"llama3.1-8b": (-111, 15), "qwen3.5-9b": (-199, -57), "qwen3.8-27b": (-128, -69), "gemma3-12b": (-223, -23), "gemma3-27b": (-163, -125)},
          ("popqa", "test2"): {"llama3.1-8b": (-97, 19), "qwen3.5-9b": (-216, -34), "qwen3.8-27b": (-116, -53), "gemma3-12b": (-239, -1), "gemma3-27b": (-170, -104)},
          ("triviaqa", "test"): {"llama3.1-8b": (140, 127), "qwen3.5-9b": (74, 128), "qwen3.8-27b": (121, 140), "gemma3-12b": (159, 126), "gemma3-27b": (259, 260)},
          ("triviaqa", "test2"): {"llama3.1-8b": (196, 149), "qwen3.5-9b": (116, 150), "qwen3.8-27b": (149, 145), "gemma3-12b": (166, 109), "gemma3-27b": (277, 280)}}
NB, SEED = 2000, 20261126


def fmt(t):
    return "%+d [%+d, %+d]" % tuple(t)


def tq_ci(v):
    """policy_select2.ci as numbers (same seed, resamples and percentile positions); checked against its string."""
    rng = random.Random(20261116)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) for _ in range(4000))
    return sum(v), bs[100], bs[3899]


def auroc(s, y):
    pos = [a for a, b in zip(s, y) if b]
    neg = [a for a, b in zip(s, y) if not b]
    if not pos or not neg:
        return float("nan")
    allv = sorted((v, i) for i, v in enumerate(s))
    ranks = [0.0] * len(s)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[allv[k][1]] = (i + j) / 2.0 + 1
        i = j + 1
    rp = sum(r for r, b in zip(ranks, y) if b)
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def auroc_ci(units, score, y):
    """units = list of lists of question indices (a pair, or one question); resampled with replacement."""
    rng = random.Random(SEED)
    pt = auroc(score, y)
    bs = []
    for _ in range(NB):
        ix = [i for _ in units for i in units[rng.randrange(len(units))]]
        v = auroc([score[i] for i in ix], [y[i] for i in ix])
        if v == v:
            bs.append(v)
    bs.sort()
    return pt, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]


def gate(on, A, B, qs):
    return {q: (A[q] if on[q] else B[q]) for q in qs}


def forced_checks(label, qs, A, B):
    """Switch forced on for every question -> the system itself; forced off -> the original; cell by cell."""
    allon, alloff = {q: True for q in qs}, {q: False for q in qs}
    g1, g0 = gate(allon, A, B, qs), gate(alloff, A, B, qs)
    if any(g1[q] != A[q] for q in qs) or any(g0[q] != B[q] for q in qs):
        raise SystemExit("FORCED CHECK FAILED: %s" % label)
    return "%s: forced on = system at %d / %d cells, forced off = original at %d / %d cells" % (
        label, sum(g1[q] == A[q] for q in qs), len(qs), sum(g0[q] == B[q] for q in qs), len(qs))


def load_fits():
    F = {}
    for m in ORDER:
        for p in POOLS:
            d = json.load(open(OUT / m / ("fit_%s_gate_1126.json" % p), encoding="utf-8"))
            if d["model"] != m or d["pool"] != p or (d["e_layer"], d["c_layer"]) != EC[m]:
                raise SystemExit("fit file %s %s" % (m, p))
            F[(m, p)] = d
    return F


def main():
    import falseqa_read as FR
    import iti_table as A
    import qaqa_pilot_table as Q
    import newsets_score as NS
    import ns_prompt_table as NP
    import cap_mcq_sweep as CMS
    MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
    G = {"R": "R", "W": "W", "X": "N", "H": "N"}
    VAL = {"R": 1, "W": -1, "N": 0}
    jci = lambda v: NS.joint_ci([list(v)])  # noqa: E731
    known_only = sys.argv[1:] == ["known"]
    print("# gate_1126 simulation (work queue 2026-11-26_queue_p, section 3); switch ON = probability > 0.5; existing answers and verdicts only")
    checks = []
    # ---------------------------------------------------------------- loading and known nets
    ok, rows = FR.check_bank()
    if not ok:
        raise SystemExit("FalseQA bank check failed")
    dev = {r["id"] for r in rows if r["set"] == "dev"}
    held = {r["id"] for r in rows if r["set"] == "heldout"}
    pair_of = {r["id"]: r["pair_id"] for r in rows}
    prem = {r["id"]: r["premise"] for r in rows}
    fids = [r["id"] for r in rows if r["set"] == "heldout"]
    pilot = set(json.loads(l)["id"] for l in open(QAQA_PILOT, encoding="utf-8") if l.strip())
    qtype = {json.loads(l)["id"]: json.loads(l)["qtype"] for l in open(QAQA_BANK, encoding="utf-8") if l.strip()}
    D = {}
    M_fact = NS.models()
    for m in ORDER:
        d = {}
        st = FR.setting(m)
        if (st[0], st[2]) != EC[m] or st[1] != STRENGTH[m][0]:
            raise SystemExit("%s: falseqa_read setting %s differs from E/C layers %s and strength %s" % (m, st, EC[m], STRENGTH[m][0]))
        R, R2 = FR.load_rows(m, dev)
        caps, _ = FR.choose(R, R2, st)
        H, H2 = FR.load_rows(m, held)
        sysd = FR.systems_of(H, H2, st, caps, H.ids)
        if list(H.ids) != fids:
            raise SystemExit("FalseQA held-out order")
        fq = {n: {q: sysd[n][q] for q in fids} for n in ("O", "E", "N1", "Conly")}
        got = tuple(sum(fq[n][q][0] for q in fids) for n in ("O", "E", "N1", "Conly"))
        print("# known nets FalseQA held-out 600 %s: O / E / N1 / Conly %s, reference %s" % (m, got, K_FQA[m]))
        if got != K_FQA[m]:
            raise SystemExit("KNOWN NET MISMATCH (FalseQA) %s" % m)
        trap = {q: prem[q] == "false" for q in fids}
        up = (sum(gate(trap, fq["N1"], fq["O"], fids)[q][0] for q in fids), sum(gate(trap, fq["E"], fq["O"], fids)[q][0] for q in fids))
        print("#   perfect switch (premise false): gated N1 %+d, gated E %+d; plan section 0 says %+d, %+d -> %s" % (
            up[0], up[1], K_FQA_UPPER[m][0], K_FQA_UPPER[m][1], "same" if up == K_FQA_UPPER[m] else "DIFFERENT"))
        d["fqa"] = fq
        # TruthfulQA
        display, layer, s = A.KEY[m]
        if layer != EC[m][0]:
            raise SystemExit("%s: TruthfulQA main-table E layer %d" % (m, layer))
        test, err = MP.prompt_split(display, layer, 1, s)
        src = [x for x in ML.sources(display)[2] if x[0] == layer]
        cx = [q for q in test[3] if q in set(src[0][3].qs)]
        if cx != test[3]:
            test, err = MP.prompt_split(display, layer, 1, s, qs=cx)
        tune, err2 = MP.prompt_split(display, layer, 0, s)
        pick = ML.n1_pick(tune[0], tune[1], s)
        sp, tx, _, tids = test
        tq = {}
        for name, cfg in (("O", ("base",)), ("E", ("E", s)), ("N1", pick)):
            states = S.system(sp, tx, cfg)[0]
            tq[name] = {q: VAL[G[x]] for q, x in zip(tids, states)}
        got = (len(tids), sum(tq["O"].values()), sum(tq["E"].values()), sum(tq["N1"].values()))
        print("# known nets TruthfulQA test %s (N1 setting %s): n / O / E / N1 %s, reference %s" % (m, pick, got, K_TQ[m]))
        if got != K_TQ[m]:
            raise SystemExit("KNOWN NET MISMATCH (TruthfulQA) %s" % m)
        d["tq"], d["tq_ids"] = tq, list(tids)
        # (QA)^2
        el, s_q, cl, cap_e, cap_own, kc = Q.SET[m]
        if (el, cl) != EC[m] or s_q != STRENGTH[m][0]:
            raise SystemExit("%s: qaqa_pilot_table.SET %s" % (m, Q.SET[m]))
        RQ = Q.Raw(m, "qaqafull", QAQA_BANK, ["test_full570"])
        qids = [q for q in RQ.ids if q not in pilot]
        o, e, n1 = RQ.arm_system("x"), RQ.arm_system(RQ.e(s_q)), RQ.n1_system(s_q, cap_e, cap_own)
        qq = {"O": {q: o[0][q] for q in qids}, "E": {q: e[0][q] for q in qids}, "N1": {q: n1[0][q] for q in qids}}
        got = tuple(sum(qq[n].values()) for n in ("O", "E", "N1"))
        print("# known nets (QA)^2 held-out %d %s: O / E / N1 %s, reference %s" % (len(qids), m, got, K_QQ[m]))
        if len(qids) != 470 or got != K_QQ[m]:
            raise SystemExit("KNOWN NET MISMATCH ((QA)^2) %s" % m)
        d["qq"], d["qq_ids"] = qq, qids
        # fact sets
        for task, split in FACT:
            main_s = NP.MAIN[m][1]
            if NP.MAIN[m][0] != EC[m][0] or main_s != STRENGTH[m][0]:
                raise SystemExit("%s: ns_prompt_table.MAIN %s" % (m, NP.MAIN[m]))
            xsp, xtx = NP.load(task, m, M_fact[m], split)[:2]
            fs = {}
            for name, cfg in (("O", ("base",)), ("E", ("E", NP.dash(main_s)))):
                states, ans = NS.system(xsp, xtx, cfg)[:2]
                fs[name] = {q.split(":", 1)[1]: v for q, v in zip(xsp.qs, NS.vals(states))}
            if any(not q.startswith(task + ":") for q in xsp.qs):
                raise SystemExit("fact keys")
            got = (sum(fs["O"].values()), sum(fs["E"].values()))
            print("# known nets %s %s %s: no write / only E %s, reference %s" % (task, split, m, got, K_FACT[(task, split)][m]))
            if got != K_FACT[(task, split)][m]:
                raise SystemExit("KNOWN NET MISMATCH (%s %s) %s" % (task, split, m))
            d[(task, split)] = fs
        mm = json.load(open("Iteration_01/capability/ecmmlu_1126/%s.json" % m, encoding="utf-8"))
        arms = ["x"] + ["x:E%d.w.p%s" % (EC[m][0], v) for v in STRENGTH[m]]
        if any(a not in mm["arms"] for a in arms) or mm["model"] != m:
            raise SystemExit("%s: ecmmlu arms" % m)
        d["mmlu"] = {a: {r["id"]: r["pick"] == r["gold"] for r in mm["arms"][a]["rows"]} for a in arms}
        d["mmlu_arms"] = arms
        D[m] = d
    print("# every known net reproduced")
    # forced checks
    for m in ORDER:
        d = D[m]
        checks.append(forced_checks("%s FalseQA N1" % m, fids, d["fqa"]["N1"], d["fqa"]["O"]))
        checks.append(forced_checks("%s FalseQA E" % m, fids, d["fqa"]["E"], d["fqa"]["O"]))
        checks.append(forced_checks("%s TruthfulQA N1" % m, d["tq_ids"], d["tq"]["N1"], d["tq"]["O"]))
        checks.append(forced_checks("%s (QA)^2 N1" % m, d["qq_ids"], d["qq"]["N1"], d["qq"]["O"]))
        for task, split in FACT:
            qs = list(d[(task, split)]["O"])
            checks.append(forced_checks("%s %s %s E" % (m, task, split), qs, d[(task, split)]["E"], d[(task, split)]["O"]))
        ids = list(d["mmlu"]["x"])
        for a in d["mmlu_arms"][1:]:
            checks.append(forced_checks("%s MMLU %s" % (m, a), ids, d["mmlu"][a], d["mmlu"]["x"]))
    print("# mandatory checks (switch forced on / off), every system:")
    for c in checks:
        print("#   " + c)
    if known_only:
        return
    FIT = load_fits()
    print("# fit files: " + ", ".join("%s %s chosen L%d" % (m, p, FIT[(m, p)]["chosen_layer"]) for m in ORDER for p in POOLS))

    def probs(m, pool, tag, key):
        return FIT[(m, pool)]["probs"][tag][key]

    def switch(m, pool, tag, key, qs):
        p = probs(m, pool, tag, key)
        if set(qs) - set(p):
            raise SystemExit("%s %s %s: questions without a switch probability" % (m, pool, key))
        return {q: p[q] > 0.5 for q in qs}
    fq_units = {}
    for q in fids:
        fq_units.setdefault(pair_of[q], []).append(fids.index(q))
    fq_units = list(fq_units.values())
    res5 = {}
    for pool in POOLS:
        print("\n" + "=" * 110)
        print("TRAINING POOL %s (%s)" % (pool, {"A": "FalseQA train only", "B": "FalseQA train + other unused training questions"}[pool]))
        for tag, tlabel in TAGS:
            print("\n" + "-" * 110)
            print("pool %s, %s" % (pool, tlabel))
            print("READING 1 (accuracy): AUROC [95%, FalseQA pairs / (QA)^2 questions resampled] | accuracy at 0.5 | judged trap per class")
            for m in ORDER:
                d = D[m]
                L = FIT[(m, pool)]["chosen_layer"] if tag == "chosen" else EC[m][0 if tag == "E" else 1]
                p = probs(m, pool, tag, "fqa_held")
                sc = [p[q] for q in fids]
                y = [prem[q] == "false" for q in fids]
                pt, lo, hi = auroc_ci(fq_units, sc, y)
                acc = sum((v > 0.5) == t for v, t in zip(sc, y))
                p2 = probs(m, pool, tag, "qaqa:held")
                qids = d["qq_ids"]
                sc2 = [p2[q] for q in qids]
                y2 = [qtype[q] != "no_failure" for q in qids]
                pt2, lo2, hi2 = auroc_ci([[i] for i in range(len(qids))], sc2, y2)
                acc2 = sum((v > 0.5) == t for v, t in zip(sc2, y2))
                print("  %-12s L%-2d FalseQA held-out 600: AUROC %.3f [%.3f, %.3f], accuracy %d / 600 (%.1f%%), judged trap: premise false %d / 300, valid %d / 300"
                      " || (QA)^2 held-out 470: AUROC %.3f [%.3f, %.3f], accuracy %d / 470 (%.1f%%), judged trap: failure %d / %d, unverifiable %d / %d, no_failure %d / %d" % (
                          m, L, pt, lo, hi, acc, 100.0 * acc / 600, sum(p[q] > 0.5 for q in fids if prem[q] == "false"), sum(p[q] > 0.5 for q in fids if prem[q] == "valid"),
                          pt2, lo2, hi2, acc2, 100.0 * acc2 / 470,
                          *[x for t in ("failure", "unverifiable", "no_failure") for x in (sum(p2[q] > 0.5 for q in qids if qtype[q] == t), sum(qtype[q] == t for q in qids))]))
            print("READING 2 (share judged trap)")
            for m in ORDER:
                d = D[m]
                parts = []
                tqp = probs(m, pool, tag, "tq_test")
                parts.append("TruthfulQA test (main list %d) %.1f%%" % (len(d["tq_ids"]), 100.0 * sum(tqp[q] > 0.5 for q in d["tq_ids"]) / len(d["tq_ids"])))
                for key, lab in (("popqa_test", "PopQA test"), ("popqa_test2", "PopQA test2"), ("triviaqa_test", "TriviaQA test"),
                                 ("triviaqa_test2", "TriviaQA test2"), ("mmlu_test", "MMLU 1,000"), ("gsm8k_test", "GSM8K 304")):
                    pp = probs(m, pool, tag, key)
                    parts.append("%s %.1f%% (%d / %d)" % (lab, 100.0 * sum(v > 0.5 for v in pp.values()) / len(pp), sum(v > 0.5 for v in pp.values()), len(pp)))
                print("  %-12s %s" % (m, " | ".join(parts)))
            print("READING 3 (simulation, natural answer counts): net, answers in brackets where shown; differences with 95% intervals")
            r5 = {}
            for m in ORDER:
                d = D[m]
                fq = d["fqa"]
                on = switch(m, pool, tag, "fqa_held", fids)
                gN1 = gate(on, fq["N1"], fq["O"], fids)
                gE = gate(on, fq["E"], fq["O"], fids)
                line = []
                for lab, qs in (("premise false", [q for q in fids if prem[q] == "false"]), ("premise valid", [q for q in fids if prem[q] == "valid"]), ("all 600", fids)):
                    df = lambda a, b: fmt(FR.pair_ci({q: a[q][0] - b[q][0] for q in qs}, pair_of, qs))  # noqa: E731
                    line.append("%s: gN1 %+d (%d) | gN1-N1 %s | gN1-Conly %s | gN1-gE %s" % (
                        lab, sum(gN1[q][0] for q in qs), sum(gN1[q][1] for q in qs), df(gN1, fq["N1"]), df(gN1, fq["Conly"]), df(gN1, gE)))
                    if lab == "all 600":
                        r5_fqa = FR.pair_ci({q: gN1[q][0] - fq["N1"][q][0] for q in qs}, pair_of, qs)
                print("  %-12s FalseQA held-out: %s" % (m, " || ".join(line)))
                ids = d["tq_ids"]
                ont = switch(m, pool, tag, "tq_test", ids)
                g = gate(ont, d["tq"]["N1"], d["tq"]["O"], ids)
                c = []
                for x in ("N1", "E", "O"):
                    v = [g[q] - d["tq"][x][q] for q in ids]
                    t3 = tq_ci(v)
                    if fmt(t3) != ps.ci(v):
                        raise SystemExit("tq_ci differs from policy_select2.ci")
                    c.append(t3)
                print("  %-12s TruthfulQA test %d: gN1 %+d (switch on %d) | gN1-N1 %s | gN1-E %s | gN1-O %s" % (
                    m, len(ids), sum(g.values()), sum(ont.values()), fmt(c[0]), fmt(c[1]), fmt(c[2])))
                r5_tq = c[0]
                qids = d["qq_ids"]
                onq = switch(m, pool, tag, "qaqa:held", qids)
                gq = gate(onq, d["qq"]["N1"], d["qq"]["O"], qids)
                print("  %-12s (QA)^2 held-out 470: gN1 %+d (switch on %d) | gN1-N1 %s | gN1-E %s" % (
                    m, sum(gq.values()), sum(onq.values()), fmt(jci(gq[q] - d["qq"]["N1"][q] for q in qids)), fmt(jci(gq[q] - d["qq"]["E"][q] for q in qids))))
                for task, split in FACT:
                    fs = d[(task, split)]
                    qs = list(fs["O"])
                    onf = switch(m, pool, tag, "%s_%s" % (task, split), qs)
                    ge = gate(onf, fs["E"], fs["O"], qs)
                    print("  %-12s %s %s %d: gE %+d (switch on %d) | gE-E %s | gE-O %s" % (
                        m, task, split, len(qs), sum(ge.values()), sum(onf.values()), fmt(jci(ge[q] - fs["E"][q] for q in qs)), fmt(jci(ge[q] - fs["O"][q] for q in qs))))
                r5[m] = [r5_fqa, r5_tq]
            print("READING 4 (MMLU 1,000): accuracy change vs no write, points [95%, items resampled]")
            for m in ORDER:
                d = D[m]
                ids = list(d["mmlu"]["x"])
                on = switch(m, pool, tag, "mmlu_test", ids)
                x = d["mmlu"]["x"]
                parts = []
                for a, lab in zip(d["mmlu_arms"][1:], ("main strength %s" % STRENGTH[m][0], "larger strength %s" % STRENGTH[m][1])):
                    gm = gate(on, d["mmlu"][a], x, ids)
                    cg = CMS.ci([int(gm[q]) - int(x[q]) for q in ids])
                    ce = CMS.ci([int(d["mmlu"][a][q]) - int(x[q]) for q in ids])
                    parts.append("%s: E on switch-on items %+.1f [%+.1f, %+.1f] | E on every item %+.1f [%+.1f, %+.1f]" % (
                        lab, 100 * cg[0], 100 * cg[1], 100 * cg[2], 100 * ce[0], 100 * ce[1], 100 * ce[2]))
                    if a == d["mmlu_arms"][2]:
                        r5[m].append(100 * cg[0])
                print("  %-12s switch on %d / %d | %s" % (m, sum(on.values()), len(ids), " || ".join(parts)))
            res5[(pool, tag)] = r5
    print("\n" + "=" * 110)
    print("READING 5 (the plan's standard, section 3.5, as counts; three conditions per model)")
    print("  (a) FalseQA held-out 'gated N1 - N1' interval above 0 (lower end > 0); (b) TruthfulQA test 'gated N1 - N1' interval not entirely below 0 (upper end >= 0);"
          " (c) larger-strength MMLU change with E only on switch-on items >= -1 point")
    for pool in POOLS:
        for tag, tlabel in TAGS:
            r5 = res5[(pool, tag)]
            n_all = 0
            rows_ = []
            for m in ORDER:
                f, t, mm = r5[m]
                a, b, c = f[1] > 0, t[2] >= 0, mm >= -1.0
                n_all += a and b and c
                rows_.append("%s: (a) %s %s, (b) %s %s, (c) %s %+.1f" % (m, int(a), fmt(f), int(b), fmt(t), int(c), mm))
            print("  pool %s, %-12s: models meeting (a) %d, (b) %d, (c) %d, all three %d of 5" % (
                pool, tlabel, sum(r5[m][0][1] > 0 for m in ORDER), sum(r5[m][1][2] >= 0 for m in ORDER), sum(r5[m][2] >= -1.0 for m in ORDER), n_all))
            for r in rows_:
                print("      " + r)


if __name__ == "__main__":
    main()
