#!/usr/bin/env python3
# Iteration_01 exploration (executor R-ding, Claude Opus 5.5, 2026-11-29 Beijing), work queue
# discussion/work_queue/2026-11-28_queue_r.md section 3.3 (item R10): the strongest opponent "sentence only" (P = the plain
# prompt + the honest sentence E2, no write: x_E2 / S0) described with existing numbers, and two stackings on it computed
# offline. Read-only: existing answers, DeepSeek Pro verdicts, alias matches and capability files; no model, no judge call,
# nothing generated, no existing file changed. Not reviewed by GPT.
# Variant A (the four-row table on the sentence row): original -> P, single E -> P + E(s'), re-ask -> P + E(s') + C(c');
#   computed as routes_1128.schemes(ids, arm)["R1_<c'>"] with O = P, E = P + E(s'), EC = P + E(s') + C (unchanged code).
# Variant B (fill only): P answered -> P; P blank and P + E(s') answered -> P + E(s'); both blank -> P + E(s') + C(c') once.
# Scoring: routes_1128.sc ("handed in only") for every side; nets by routes_1128.net.
# Choices (TruthfulQA tune, 192 questions): s' among s_sel and s_half = s_sel / 2 (work queue I sections 2 and 8): first the
#   capability line (MMLU 13,042 unseen items and GSM8K 1,319 items at cap 512, each at most 1 point below the plain no-write
#   prompt; the files read by diag/sentence_capability.py); then the higher tune net of P + E(s) alone; ties -> s_half.
#   c' among 0.25 / 0.5 / 1 / 2 by the variant's own tune net, ties -> the smaller. FalseQA and (QA)^2 reuse s' and c'; an arm
#   that does not exist there is listed as missing (nothing is generated). When neither strength passes the line, both are
#   shown and marked.
# Capability of a variant: MMLU (option probability) has no blank answer, so A = P + E(s) and B = P on every item. GSM8K
#   (blank = the rule detector's abstention or an empty answer, gsm8k_n1_prompt.blank; right = cap_score.judge_row "R"):
#   A hands in P + E(s) where it answers, nothing where it is blank and P answered, and the P + E + C re-ask where both are
#   blank; B hands in P where it answers, else P + E(s), else the re-ask. No P + E + C pass exists on GSM8K, so those items
#   give a lower bound (none right) and an upper bound (all right). Base = the plain no-write pass, scored as
#   sentence_capability scores it.
# Method M = routes_1128 R1 at the C strength of redo_1128.CM, no sentence, main-table settings (redo_1128 readers, unchanged);
#   single E = the no-sentence E at the main-table strength.
# Order (binding): the original readers run in-process first and their output is compared line by line with the archived
#   files (bundle Iteration_01/diag/r10_1128_bundle_20261129.json, made from the review repository); then the cells read
#   here are checked value by value against archived numbers; new numbers are printed only when all of that agrees.
# Intervals: TruthfulQA questions resampled (gate_sim_1126.tq_ci = policy_select2.ci), FalseQA pairs resampled
#   (falseqa_read.pair_ci), (QA)^2 and PopQA newsets_score.joint_ci (as n1_on_sentence2 / 3 and plain_honesty read them).
#   Five-model sums are reference rows only.
# Output: a text report, then one line "JSON <dump>".
# usage (container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/r10_1128.py
import json
import os
import sys
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/gpt_tools_20261126/gate_audit"]
import redo_1128 as RD             # noqa: E402
import n1_on_sentence as N         # noqa: E402
import n1_on_sentence2 as N2       # noqa: E402
import n1_on_sentence3 as N3       # noqa: E402
import plain_honesty as PH         # noqa: E402
import sentence_capability as SC   # noqa: E402
import gsm8k_n1_prompt as GK       # noqa: E402
import cap_score as CS             # noqa: E402

RT, A, F = RD.RT, RD.A, RD.F
ML, ps = RD.ML, RD.ps
Q, NP = N3.Q, N3.NP
ORDER, KS, CM, PH0 = RT.ORDER, RT.KS, RD.CM, RD.PH0
BUNDLE = "Iteration_01/diag/r10_1128_bundle_20261129.json"
DUMP = {}


# ------------------------------------------------------------------------------------------------ helpers
def fmt(t):
    return "%+d [%+d, %+d]" % tuple(t)


def jci(v):
    return [int(x) for x in Q.S.joint_ci([list(v)])]


def hi(cells, ids):
    """handed-in-only {q: (score, handed)} of an arm (routes_1128.sc)."""
    return {q: RT.sc(cells[q]) for q in ids}


def nh(d, ids):
    return RT.net(d, ids), sum(bool(d[q][1]) for q in ids)


def show(d, ids):
    return "%+d (%d)" % nh(d, ids)


def orig(cells, ids):
    """Original scoring (the judge's value whatever the blank flag, answered count): only for the value checks."""
    return sum(cells[q][0] for q in ids), sum(bool(cells[q][1]) for q in ids)


def var_a(ids, arm, c):
    """Variant A: the four-row table on the sentence row = routes_1128 R1 at C strength c."""
    return RT.schemes(ids, arm)["R1_" + c][0]


def var_b(ids, arm, c):
    """Variant B (fill only): P answered -> P; P blank and P + E answered -> P + E; both blank -> P + E + C(c) once."""
    O, E, X = arm["O"], arm["E"], arm["EC" + c]
    return {q: (RT.sc(O[q]) if O[q][1] else (RT.sc(E[q]) if E[q][1] else RT.sc(X[q]))) for q in ids}


VARS = (("A", var_a), ("B", var_b))


def with_placeholders(arm, ids):
    for k in KS:
        arm.setdefault("C" + k, {q: PH0 for q in ids})
    return arm


def check_value(label, got, want):
    ok = got == want
    DUMP.setdefault("value_checks", []).append([label, ok, got, want])
    print("  value check %-86s %s" % (label, ("SAME %s" % (got,)) if ok else "DIFFERENT: got %s, archived %s" % (got, want)))
    if not ok:
        raise SystemExit("value check failed: %s" % label)


def line_check(label, text, prefix, part):
    """A line of the archived text that starts with prefix and contains part."""
    ok = any(l.startswith(prefix) and part in l for l in text.splitlines())
    DUMP.setdefault("value_checks", []).append([label, ok, prefix + " ... " + part, None])
    print("  value check %-86s %s" % (label, "found in the archived output" if ok else "NOT FOUND: %s ... %s" % (prefix, part)))
    if not ok:
        raise SystemExit("value check failed: %s" % label)


def strengths(m):
    return N3.strengths(m)          # (s_sel, s_half), dotted


# ------------------------------------------------------------------------------------------------ 1 regressions
def compare_run(label, fns, want, rec):
    """fns: [(fn, argv)]; stdout of all runs joined, compared with want; when it differs, stdout and stderr merged."""
    outs, errs, codes = [], [], []
    for fn, argv in fns:
        o, e, c = RD.capture_split(fn, argv)
        outs.append(o)
        errs.append(e)
        codes.append(c)
    got = "".join(outs)
    if got.splitlines() != want.splitlines() and any(errs):
        merged = "".join(RD.capture(fn, argv)[0] for fn, argv in fns)
        if merged.splitlines() == want.splitlines():
            got, label = merged, label + " (stdout and stderr merged)"
    ok = RD.compare(label, got, want, rec)
    err_lines = [l for e in errs for l in e.splitlines() if l.strip()]
    rec[label]["stderr_lines"] = err_lines
    rec[label]["exit_codes"] = codes
    if err_lines:
        print("    stderr (%d lines): %s" % (len(err_lines), " | ".join(err_lines[:5])[:600]))
    bad = [c for c in codes if c not in (None, 0)]
    return ok and not bad


def regressions(B):
    rec = DUMP.setdefault("regression", {})
    oks = []
    print("# 1 regressions: the original readers run in-process against their archived files")
    # n1_on_sentence.py: the archive is the tune run, then the test run without its repeated step-0 lines
    o1, e1, c1 = RD.capture_split(N.main, ["n1_on_sentence.py", "tune"])
    o2, e2, c2 = RD.capture_split(N.main, ["n1_on_sentence.py", "test"])
    l1, l2 = o1.splitlines(), o2.splitlines()
    ok = RD.compare("n1_on_sentence.py tune, then test without its step-0 lines vs n1_on_sentence_five_models_20261122.txt",
                    "\n".join(l1 + l2[5:]), B["n1s"], rec)
    same0 = l2[:5] == l1[:5]
    print("    the 5 step-0 lines of the test run equal those of the tune run: %s; stderr lines %d; exit codes %s, %s" % (
        same0, len((e1 + e2).splitlines()), c1, c2))
    oks.append(ok and same0 and c1 in (None, 0) and c2 in (None, 0))
    oks.append(compare_run("n1_on_sentence3.py fact, then final vs n1_on_sentence_strength_five_models_20261123.txt",
                           [(N3.main, ["n1_on_sentence3.py", "fact"]), (N3.main, ["n1_on_sentence3.py", "final"])], B["n1s3"], rec))
    oks.append(compare_run("n1_on_sentence2.py qaqa vs n1_on_sentence_qaqa_five_models_20261122.txt",
                           [(N2.main, ["n1_on_sentence2.py", "qaqa"])], B["n1s2_qaqa"], rec))
    oks.append(compare_run("falseqa_read.py table vs falseqa_heldout_table_20261124.txt",
                           [(lambda: F.run_table(list(ORDER)), None)], B["falseqa"], rec))
    oks.append(compare_run("plain_honesty.py tqa, qaqa, fact vs plain_honesty_prompt_five_models_20261122.txt",
                           [(PH.tqa, None), (PH.qaqa, None), (PH.fact, None)], B["plain_honesty"], rec))
    SC.HALF = True
    oks.append(compare_run("sentence_capability.py table --half vs sentence_capability_half_five_models_20261123.txt",
                           [(SC.table, None)], B["cap_half"], rec))
    return all(oks)


# ------------------------------------------------------------------------------------------------ 2 data
def tq_sentence(m, i, s_dot):
    """The sentence row on TruthfulQA split i at E strength s_dot (n1_on_sentence3.tq_build), or None when absent."""
    qs = N.restricted(m, i)
    try:
        sp, tx = N3.tq_build(m, i, s_dot, qs)
    except SystemExit as e:
        if str(e).startswith("missing conditions"):
            return None
        raise
    s = N.dash(s_dot)
    ids = list(sp.qs)

    def cell(c):
        return {q: (RT.VAL[sp.st[(q, c)]], not sp.blank[(q, c)]) for q in ids}
    arm = {"O": cell("baseline_x"), "E": cell(sp.cE(s))}
    for k, name in zip(ps.LAD, KS):
        arm["EC" + name] = cell(sp.cEC(s, k))
    return {"ids": ids, "arm": with_placeholders(arm, ids), "sp": sp, "tx": tx, "dropped": len(qs) - len(ids)}


def tq_data(B):
    routes = json.loads(B["routes_json"])
    rec = DUMP.setdefault("checks_truthfulqa", {})
    print("\n# 2a TruthfulQA: the method (routes_1128 R1) against routes_1128_20261128.json, and the sentence row")
    D = {}
    for m in ORDER:
        R = RD.tq_routes(m)
        RD.routes_check(rec, "TruthfulQA", m, R["M"], R["ids"], routes)
        ss, sh = strengths(m)
        d = {"R": R, "tune": {}, "test": {}, "ss": ss, "sh": sh}
        for s in (ss, sh):
            d["tune"][s] = tq_sentence(m, 0, s)
            if d["tune"][s] is None:
                raise SystemExit("%s: TruthfulQA tune lacks the sentence arms at E %s" % (m, s))
            d["test"][s] = tq_sentence(m, 1, s)
        if d["test"][ss] is None:
            raise SystemExit("%s: TruthfulQA test lacks the sentence arms at s_sel" % m)
        D[m] = d
    return D


def tq_checks(D, B):
    print("\n# 2b TruthfulQA value checks of the cells read here")
    R1 = json.loads(B["R1_json"])["R1"]
    key = [k for k in R1 if k.startswith("## R1")][0]
    for m in ORDER:
        d = D[m]
        ss, sh = d["ss"], d["sh"]
        ch3 = json.load(open(N3.CHOICE3 % m, encoding="utf-8"))
        s_main = A.KEY[m][2]
        step1 = {N.dot(s_main), N.dot(N.half(s_main))}
        for s in (ss, sh):
            t = d["tune"][s]
            cfg = ML.n1_pick(t["sp"], t["tx"], N.dash(s))
            sn1 = N.net(N.run_sys(t["sp"], t["tx"], cfg))
            check_value("%s TruthfulQA tune SN1 net at E %s vs newsets/h22/n1s3_choice rows" % (m, s), sn1[0],
                        ch3["rows"][s]["truthfulqa_tune_sn1"])
            if s in step1:
                part = "| S0 %+d (%d) | SE %+d (%d) | SN1 %+d (%d) with" % (orig(t["arm"]["O"], t["ids"]) + orig(t["arm"]["E"], t["ids"]) + sn1)
                line_check("%s TruthfulQA tune S0 / SE / SN1 at E %s vs n1_on_sentence archive" % (m, s), B["n1s"],
                           "  %-12s E %-5s %d tune q" % (m, s, len(t["ids"])), part)
        x = d["test"][ss]
        part = "    S0 %+d (%d) | SE %+d (%d) | SN1" % (orig(x["arm"]["O"], x["ids"]) + orig(x["arm"]["E"], x["ids"]))
        line_check("%s TruthfulQA test S0 / SE at s_sel %s vs n1_on_sentence archive" % (m, ss), B["n1s"], part, "")
        if ch3["chosen"] == sh and d["test"][sh] is not None:
            y = d["test"][sh]
            line_check("%s TruthfulQA test SE at s_half %s vs n1_on_sentence3 archive (SEm)" % (m, sh), B["n1s3"],
                       "    %-12s %3d q | no write" % (m, len(y["ids"])), "| SEm %+d (%d) | SN1m" % orig(y["arm"]["E"], y["ids"]))
        R = d["R"]
        ids = [q for q in R["ids"] if q in set(x["ids"])]
        if ids != R["ids"]:
            raise SystemExit("%s: the sentence row lacks %d of the method's TruthfulQA test questions" % (m, len(R["ids"]) - len(ids)))
        P, E = hi(x["arm"]["O"], ids), hi(R["arm"]["E"], ids)
        r = R1[key][m]
        check_value("%s TruthfulQA test P handed in (net, handed) vs out_redo_1128_R1 x_E2" % m, list(nh(P, ids)),
                    [r["x_E2"]["net"], r["x_E2"]["handed"]])
        check_value("%s TruthfulQA test M - P vs out_redo_1128_R1 M-x_E2" % m, RD.tq([R["M"][q][0] - P[q][0] for q in ids]), r["M-x_E2"])
        check_value("%s TruthfulQA test single E handed in vs out_redo_1128_R1" % m, list(nh(E, ids)),
                    [r["single E"]["net"], r["single E"]["handed"]])


def fq_data(B):
    routes = json.loads(B["routes_json"])
    rec = DUMP.setdefault("checks_falseqa", {})
    print("\n# 2c FalseQA held-out 600: the method against routes_1128_20261128.json, and the sentence row")
    ok, rows = RD.capture_ret(F.check_bank)
    if not ok:
        raise SystemExit("FalseQA bank check failed")
    held = {r["id"] for r in rows if r["set"] == "heldout"}
    D = {"pair_of": {r["id"]: r["pair_id"] for r in rows}}
    for m in ORDER:
        ids, arm, n1 = RT.fqa_split(m, rows, "heldout")
        Md = RD.method_of(ids, arm, m)
        RD.routes_check(rec, "FalseQA", m, Md, ids, routes)
        H, H2 = F.load_rows(m, held)
        if list(H2.ids) != ids:
            raise SystemExit("%s: FalseQA question order" % m)
        el, s_main, cl, ss = F.setting(m)

        def cells(a):
            return {q: (H2.cell(a, q), not H2.blank(a, q)) for q in ids}
        sent, miss = {}, {}
        for s in strengths(m):
            arms = ["x_E2:E%d.w.p%s" % (el, s)] + ["x_E2:E%d.w.p%s+C%d.w.m%s" % (el, s, cl, k) for k in KS]
            if s != ss:
                lack = [a for a in arms if not (F.gdir(m) / (F.G.arm_token(a) + ".json")).is_file()]
                if lack:
                    sent[s], miss[s] = None, "no generation file for %s" % ", ".join(lack)
                    continue
                try:
                    F.alias(H2, arms)
                except FileNotFoundError as e:
                    sent[s], miss[s] = None, str(e)[:200]
                    continue
            a = {"O": cells("x"), "E": cells(H2.e(s))}
            for k in KS:
                a["EC" + k] = cells(H2.ec(s, k))
            sent[s] = with_placeholders(a, ids)
        D[m] = {"ids": ids, "arm": arm, "M": Md, "sent": sent, "miss": miss, "ss": ss, "S0": cells("x")}
    return D


def fq_checks(D, B):
    print("\n# 2d FalseQA value checks")
    R5 = json.loads(B["R5_json"])["R5"]["table"]["held-out, all"]
    pair_of = D["pair_of"]
    for m in ORDER:
        d = D[m]
        ids, ss = d["ids"], d["ss"]
        a = d["sent"][ss]
        line_check("%s FalseQA S0 / SE (original scoring, all 600) vs falseqa_heldout_table" % m, B["falseqa"], "  %-12s " % m,
                   "| S0 %+d (%d) | SE %+d (%d) | SN1" % (orig(a["O"], ids) + orig(a["E"], ids)))
        P, E = hi(d["S0"], ids), hi(d["arm"]["E"], ids)
        check_value("%s FalseQA P handed in vs out_redo_1128_R5 S0" % m, list(nh(P, ids)), R5[m]["S0"])
        check_value("%s FalseQA single E handed in vs out_redo_1128_R5 E" % m, list(nh(E, ids)), R5[m]["E"])
        check_value("%s FalseQA M - P natural vs out_redo_1128_R5" % m,
                    [int(x) for x in F.pair_ci({q: d["M"][q][0] - P[q][0] for q in ids}, pair_of, ids)], R5[m]["M-S0 natural"])


def qq_data(B):
    print("\n# 2e (QA)^2 570 (read on the 470 held-out): the method (redo_1128 R4 cells) and the sentence row")
    skip = set(json.load(open(RD.TUNEIDS, encoding="utf-8"))["ids"])
    if skip != set(N3.PILOT):
        raise SystemExit("(QA)^2 pilot id lists differ")
    D = {}
    for m in ORDER:
        el, s_main, cl = Q.SET[m][:3]
        R = Q.Raw(m, "qaqafull", RD.FULLB, ["test_full570"])
        ids = list(R.ids)
        ids470 = [q for q in ids if q not in skip]

        def cells(RR, a):
            RR.need(a)
            return {q: (RR.cell(a, q), not RR.blank(a, q)) for q in ids}
        arm = {"O": cells(R, "x"), "E": cells(R, R.e(s_main))}
        for k in KS:
            for nm, a in (("EC" + k, R.ec(s_main, k)), ("C" + k, R.c(k))):
                arm[nm] = cells(R, a) if R.has(a) else {q: PH0 for q in ids}
        if not R.has(R.ec(s_main, CM[m])):
            raise SystemExit("%s: (QA)^2 E + C at the method's strength has no judged pass" % m)
        Md = RD.method_of(ids, arm, m)
        sent, miss = {}, {}
        for s in strengths(m):
            try:
                R2 = N3.qaqa_raw(m, s)
            except FileNotFoundError as e:
                sent[s], miss[s] = None, str(e)[:200]
                continue
            if R2.ids != R.ids:
                raise SystemExit("%s: (QA)^2 banks differ" % m)
            a = {"O": cells(R2, "x"), "E": cells(R2, R2.e(s))}
            for k in KS:
                a["EC" + k] = cells(R2, R2.ec(s, k))
            sent[s] = with_placeholders(a, ids)
        D[m] = dict(ids=ids, ids470=ids470, arm=arm, M=Md, sent=sent, miss=miss)
    return D


def qq_checks(D, B):
    print("\n# 2f (QA)^2 value checks (470 held-out)")
    R4 = json.loads(B["R4_json"])["R4"]
    key = [k for k in R4 if k.startswith("## R4 (QA)^2 470") and k.endswith("all types: M minus each opponent (handed in only)")][0]
    for m in ORDER:
        d = D[m]
        ids = d["ids470"]
        ss, sh = strengths(m)
        ch3 = json.load(open(N3.CHOICE3 % m, encoding="utf-8"))
        r = R4[key][m]
        check_value("%s (QA)^2 470 M handed in vs out_redo_1128_R4" % m, list(nh(d["M"], ids)), [r["M"]["net"], r["M"]["handed"]])
        check_value("%s (QA)^2 470 single E handed in vs out_redo_1128_R4" % m, list(nh(hi(d["arm"]["E"], ids), ids)),
                    [r["single E"]["net"], r["single E"]["handed"]])
        a = d["sent"][ss]
        prefix = "    %-12s %3d q | no write" % (m, len(ids))
        text = B["n1s3"].split("## 8 (QA)^2 470 held-out, all types, chosen strength")[1].split("\n## ")[0]
        line_check("%s (QA)^2 470 S0 / SE s_sel (original scoring) vs n1_on_sentence3 archive" % m, text, prefix,
                   "| S0 %+d (%d) | SE s_sel %+d (%d) | SN1 s_sel" % (orig(a["O"], ids) + orig(a["E"], ids)))
        c = d["sent"][ch3["chosen"]]
        line_check("%s (QA)^2 470 SE at the section-8 strength %s vs n1_on_sentence3 archive (SEm)" % (m, ch3["chosen"]), text, prefix,
                   "| SEm %+d (%d) | SN1m" % orig(c["E"], ids))


def pop_data(B):
    print("\n# 2g PopQA test / test2 (alias match): no write, single E and the method at the TruthfulQA settings; P = x_E2")
    Mm = NP.S.models()
    D = {}
    for split in ("test", "test2"):
        b, bp = NP.bank("popqa", split)
        text = B["plain_honesty"].split("## popqa %s (alias match" % split)[1].split("\n## ")[0]
        for m in ORDER:
            xsp, xtx, xh, _, _, _ = NP.load("popqa", m, Mm[m], split)
            ms = NP.dash(NP.MAIN[m][1])
            if ms not in xh:
                raise SystemExit("%s PopQA %s: the main strength is incomplete" % (m, split))
            ids = list(xsp.qs)

            def cell(c):
                return {q: ((xsp.st[(q, c)] == "R") - (xsp.st[(q, c)] == "W"), not xsp.blank[(q, c)]) for q in ids}
            arm = {"O": cell("baseline_x"), "E": cell(xsp.cE(ms))}
            for k, name in zip(NP.LAD, KS):
                arm["EC" + name] = cell(xsp.cEC(ms, k))
            arm = with_placeholders(arm, ids)
            Md = RD.method_of(ids, arm, m)
            p = os.path.join(NP.S.NS, "gen_prompt", "h22", "popqa", m, split, "x_E2.json")
            g = NP.G.load_generation(p, SimpleNamespace(task="popqa", split=split, model=m, items=bp), list(b))
            if g["contract"]["arm"] != "x_E2" or g["contract"].get("positions") is not None:
                raise SystemExit("%s is not the no-write x_E2 pass" % p)
            z = {}
            for r in g["rows"]:
                blank = bool(r["rule_abstain"]) or not r["text"].strip()
                v = NP.AM.verdict(r["text"], b[r["id"]]["gold"], not blank)
                z["popqa:%s" % r["id"]] = ((v == "R") - (v == "W"), not blank)
            if list(z) != ids:
                raise SystemExit("%s PopQA %s: question order of x_E2" % (m, split))
            line_check("%s PopQA %s no write / single E / cell 0 vs plain_honesty archive" % (m, split), text,
                       "    %-12s %3d q | no write %+d (%d) | single E %+d (%d) |" % ((m, len(ids)) + orig(arm["O"], ids) + orig(arm["E"], ids)),
                       "| cell 0 %+d (%d) ||" % orig(z, ids))
            D[(split, m)] = dict(ids=ids, arm=arm, M=Md, P=z)
    return D


def cap_data():
    print("\n# 2h capability files (sentence_capability.py readers; GSM8K rows re-read here for the blank flags)")
    SC.HALF = True
    seen = set(SC.read(SC.SEEN)["ids"])
    gold = CS.gold_of("gsm8k")
    D = {}
    for m in ORDER:
        mids, mm = SC.mmlu_model(m, seen)
        gids, gd = SC.gsm_model(m, gold)
        SC.TRUNC.pop(m, None)
        layer = A.KEY[m][1]
        ss, sh = strengths(m)
        tok = lambda s: s.replace(".", "-")  # noqa: E731
        g0 = "%s/gen_main/gsm8k/%s/b8" % (SC.CAP, m)
        g1 = "%s/gen_main/gsm8k_h22/%s/b8" % (SC.CAP, m)
        files = {"base": g0 + "/neutral.json", "P": g1 + "/neutral_E2.json",
                 ss: g1 + "/neutral_E2--E%d-w-p%s-at-prompt.json" % (layer, tok(ss)),
                 sh: g1 + "/neutral_E2--E%d-w-p%s-at-prompt.json" % (layer, tok(sh))}
        rows = {}
        for k, p in files.items():
            g = SC.read(p)
            c = g["contract"]
            if c["max_new_tokens"] != 512 or c["batch_size"] != 8 or not c["full_pass"]:
                raise SystemExit("%s: not a full 512-token batch-8 pass" % p)
            rows[k] = {r["id"]: r for r in g["rows"]}
            if sorted(rows[k]) != sorted(gids):
                raise SystemExit("%s: GSM8K items differ" % p)
        right = {k: {i: CS.judge_row("gsm8k", r, gold(i))[0] == "R" for i, r in v.items()} for k, v in rows.items()}
        for k, kk in (("base", "base"), ("P", "P"), (ss, "P+E"), (sh, "P+E half")):
            check_value("%s GSM8K per-item correctness of %s equals sentence_capability.gsm_model's" % (m, kk), right[k] == gd[kk], True)
        for kk in ("base", "P", "P+E", "P+E half"):
            if mm.get(kk) is None:
                raise SystemExit("%s: MMLU %s missing" % (m, kk))
        D[m] = dict(mids=mids, mm=mm, gids=gids, rows=rows, right=right, ss=ss, sh=sh)
    return D


# ------------------------------------------------------------------------------------------------ 3 capability of a variant
def line_ok(delta_count, n):
    """at most 1 point below: 100 * delta / n >= -1, in integers"""
    return 100 * delta_count >= -n


def cap_cost(C, m, var, s):
    c = C[m]
    mids, n_m = c["mids"], len(c["mids"])
    base_m = c["mm"]["base"]
    pe = c["mm"]["P+E" if s == c["ss"] else "P+E half"]
    arm_m = pe if var == "A" else c["mm"]["P"]
    dm = [int(arm_m[i]) - int(base_m[i]) for i in mids]
    ids, n = c["gids"], len(c["gids"])
    P, PE, rP, rPE, rB = c["rows"]["P"], c["rows"][s], c["right"]["P"], c["right"][s], c["right"]["base"]
    lo, unk, cnt = [], 0, {"P blank": 0, "P+E blank": 0, "both blank": 0}
    for i in ids:
        pb, eb = GK.blank(P[i]), GK.blank(PE[i])
        cnt["P blank"] += pb
        cnt["P+E blank"] += eb
        cnt["both blank"] += pb and eb
        if var == "A":
            v = rPE[i] if not eb else (False if not pb else None)
        else:
            v = rP[i] if not pb else (rPE[i] if not eb else None)
        if v is None:
            unk += 1
            v = False
        lo.append(int(v) - int(rB[i]))
    dlo = sum(lo)
    dhi = dlo + unk
    mm_ok = line_ok(sum(dm), n_m)
    g = "yes" if line_ok(dlo, n) else ("no" if not line_ok(dhi, n) else "undetermined")
    ok = "yes" if (mm_ok and g == "yes") else ("no" if (not mm_ok or g == "no") else "undetermined")
    base_blank_right = sum(1 for i in ids if GK.blank(c["rows"]["base"][i]) and rB[i])
    return {"mmlu": 100.0 * sum(dm) / n_m, "mmlu_ci": SC.boot([dm]), "mmlu_pass": mm_ok,
            "gsm_lo": 100.0 * dlo / n, "gsm_hi": 100.0 * dhi / n, "gsm_lo_ci": SC.boot([lo]), "gsm_unknown": unk,
            "gsm_pass": g, "pass": ok, "counts": cnt, "base_blank_but_right": base_blank_right,
            "how": ("MMLU = P + E(s) on every item (no blank); GSM8K: P + E(s) where it answers, nothing where it is blank and P answered, "
                    "the missing P + E + C re-ask where both are blank" if var == "A" else
                    "MMLU = P on every item (no blank); GSM8K: P where it answers, else P + E(s), else the missing P + E + C re-ask")}


def cap_str(k):
    g = ("%+.2f" % k["gsm_lo"]) if k["gsm_unknown"] == 0 else ("%+.2f to %+.2f" % (k["gsm_lo"], k["gsm_hi"]))
    return "MMLU %+.2f (%s) | GSM8K %s (%d both-blank items without a re-ask) -> line %s" % (
        k["mmlu"], "passes" if k["mmlu_pass"] else "FAILS", g, k["gsm_unknown"],
        {"yes": "PASSES", "no": "FAILS", "undetermined": "UNDETERMINED"}[k["pass"]])


# ------------------------------------------------------------------------------------------------ 4 report
def opponent(TQ, FQ, QQ, PQ, C):
    out = {}
    print("\n" + "=" * 110 + "\n# 3 the opponent P = plain prompt + honest sentence (x_E2), no write; every side handed in only")
    # TruthfulQA test
    print("\n## 3a TruthfulQA test (DeepSeek Pro): net (handed in); intervals: questions resampled")
    per = {}
    for m in ORDER:
        d = TQ[m]
        R, x = d["R"], d["test"][d["ss"]]
        ids = R["ids"]
        P, M, E = hi(x["arm"]["O"], ids), R["M"], hi(R["arm"]["E"], ids)
        per[m] = (ids, P, M, E)
        r = {"n": len(ids), "P": nh(P, ids), "M": nh(M, ids), "single E": nh(E, ids),
             "M-P": RD.tq([M[q][0] - P[q][0] for q in ids]), "single E-P": RD.tq([E[q][0] - P[q][0] for q in ids])}
        out.setdefault("truthfulqa_test", {})[m] = r
        print("  %-12s %d q | P %s | M %s | single E %s | M - P %s | single E - P %s" % (
            m, len(ids), show(P, ids), show(M, ids), show(E, ids), fmt(r["M-P"]), fmt(r["single E-P"])))
    common = sorted(set.intersection(*(set(per[m][0]) for m in ORDER)))
    r = {"n": len(common), "M-P": RD.tq([sum(per[m][2][q][0] - per[m][1][q][0] for m in ORDER) for q in common]),
         "single E-P": RD.tq([sum(per[m][3][q][0] - per[m][1][q][0] for m in ORDER) for q in common])}
    out["truthfulqa_test"]["five models (reference only)"] = r
    print("  %-12s %d common q (reference only) | M - P %s | single E - P %s" % ("five models", len(common), fmt(r["M-P"]), fmt(r["single E-P"])))
    # FalseQA
    print("\n## 3b FalseQA held-out 600 (DeepSeek Pro): net (handed in); intervals: pairs resampled")
    pair_of = FQ["pair_of"]
    tot = {"M-P": {}, "single E-P": {}}
    for m in ORDER:
        d = FQ[m]
        ids = d["ids"]
        P, M, E = hi(d["S0"], ids), d["M"], hi(d["arm"]["E"], ids)
        r = {"n": len(ids), "P": nh(P, ids), "M": nh(M, ids), "single E": nh(E, ids),
             "M-P": [int(x) for x in F.pair_ci({q: M[q][0] - P[q][0] for q in ids}, pair_of, ids)],
             "single E-P": [int(x) for x in F.pair_ci({q: E[q][0] - P[q][0] for q in ids}, pair_of, ids)]}
        for q in ids:
            tot["M-P"][q] = tot["M-P"].get(q, 0) + M[q][0] - P[q][0]
            tot["single E-P"][q] = tot["single E-P"].get(q, 0) + E[q][0] - P[q][0]
        out.setdefault("falseqa_600", {})[m] = r
        print("  %-12s %d q | P %s | M %s | single E %s | M - P %s | single E - P %s" % (
            m, len(ids), show(P, ids), show(M, ids), show(E, ids), fmt(r["M-P"]), fmt(r["single E-P"])))
    ids = FQ[ORDER[0]]["ids"]
    r = {k: [int(x) for x in F.pair_ci(v, pair_of, ids)] for k, v in tot.items()}
    out["falseqa_600"]["five models (reference only)"] = r
    print("  %-12s (reference only) | M - P %s | single E - P %s" % ("five models", fmt(r["M-P"]), fmt(r["single E-P"])))
    # (QA)^2
    print("\n## 3c (QA)^2 470 held-out (DeepSeek Pro): net (handed in); intervals: newsets_score.joint_ci")
    tot = {"M-P": [0] * 470, "single E-P": [0] * 470}
    for m in ORDER:
        d = QQ[m]
        ids = d["ids470"]
        P, M, E = hi(d["sent"][strengths(m)[0]]["O"], ids), d["M"], hi(d["arm"]["E"], ids)
        vm, ve = [M[q][0] - P[q][0] for q in ids], [E[q][0] - P[q][0] for q in ids]
        tot["M-P"] = [a + b for a, b in zip(tot["M-P"], vm)]
        tot["single E-P"] = [a + b for a, b in zip(tot["single E-P"], ve)]
        r = {"n": len(ids), "P": nh(P, ids), "M": nh(M, ids), "single E": nh(E, ids), "M-P": jci(vm), "single E-P": jci(ve)}
        out.setdefault("qaqa_470", {})[m] = r
        print("  %-12s %d q | P %s | M %s | single E %s | M - P %s | single E - P %s" % (
            m, len(ids), show(P, ids), show(M, ids), show(E, ids), fmt(r["M-P"]), fmt(r["single E-P"])))
    r = {k: jci(v) for k, v in tot.items()}
    out["qaqa_470"]["five models (reference only)"] = r
    print("  %-12s (reference only) | M - P %s | single E - P %s" % ("five models", fmt(r["M-P"]), fmt(r["single E-P"])))
    # PopQA
    for split in ("test", "test2"):
        print("\n## 3d PopQA %s, 500 questions, ALIAS MATCH (no judge); the method at the TruthfulQA settings (frozen); intervals: joint_ci" % split)
        tot = None
        for m in ORDER:
            d = PQ[(split, m)]
            ids = d["ids"]
            P, M, E = hi(d["P"], ids), d["M"], hi(d["arm"]["E"], ids)
            vm, ve = [M[q][0] - P[q][0] for q in ids], [E[q][0] - P[q][0] for q in ids]
            tot = (vm, ve) if tot is None else ([a + b for a, b in zip(tot[0], vm)], [a + b for a, b in zip(tot[1], ve)])
            r = {"n": len(ids), "P": nh(P, ids), "M": nh(M, ids), "single E": nh(E, ids), "M-P": jci(vm), "single E-P": jci(ve)}
            out.setdefault("popqa_" + split, {})[m] = r
            print("  %-12s %d q | P %s | M %s | single E %s | M - P %s | single E - P %s" % (
                m, len(ids), show(P, ids), show(M, ids), show(E, ids), fmt(r["M-P"]), fmt(r["single E-P"])))
        r = {"M-P": jci(tot[0]), "single E-P": jci(tot[1])}
        out["popqa_" + split]["five models (reference only)"] = r
        print("  %-12s (reference only) | M - P %s | single E - P %s" % ("five models", fmt(r["M-P"]), fmt(r["single E-P"])))
    # capability of the sentence itself
    print("\n## 3e capability of the sentence itself (P, no write) against the plain no-write prompt; points, paired bootstrap")
    for m in ORDER:
        c = C[m]
        dm = [int(c["mm"]["P"][i]) - int(c["mm"]["base"][i]) for i in c["mids"]]
        dg = [int(c["right"]["P"][i]) - int(c["right"]["base"][i]) for i in c["gids"]]
        r = {"mmlu": SC.boot([dm]), "gsm8k": SC.boot([dg]), "mmlu_items": len(dm), "gsm8k_items": len(dg)}
        out.setdefault("capability", {})[m] = r
        print("  %-12s MMLU (%d unseen items) P - base %s | GSM8K (%d items, cap 512) P - base %s" % (m, len(dm), r["mmlu"], len(dg), r["gsm8k"]))
    print("  measured for all five models on both sets; nothing else of the sentence's own capability was measured")
    DUMP["opponent"] = out


def choose(TQ, C):
    print("\n" + "=" * 110 + "\n# 4 choices on TruthfulQA tune (192 questions), handed in only")
    out = {}
    for m in ORDER:
        d = TQ[m]
        ss, sh = d["ss"], d["sh"]
        for v, fn in VARS:
            per = {}
            for s in (ss, sh):
                t = d["tune"][s]
                ids, arm = t["ids"], t["arm"]
                nets = [(RT.net(fn(ids, arm, k), ids), k) for k in KS]
                c = max(nets, key=lambda x: (x[0], -KS.index(x[1])))[1]
                per[s] = {"tune_q": len(ids), "S0": nh(hi(arm["O"], ids), ids), "P+E": nh(hi(arm["E"], ids), ids),
                          "variant_by_c": {k: n for n, k in nets}, "c": c, "cap": cap_cost(C, m, v, s)}
            passing = [s for s in (ss, sh) if per[s]["cap"]["pass"] == "yes"]
            pick = (lambda ss_: max(ss_, key=lambda s: (per[s]["P+E"][0], s == sh)))
            chosen = pick(passing) if passing else None
            rec = {"per": per, "passing": passing, "chosen": chosen, "effect_rule_only": pick([ss, sh]), "s_sel": ss, "s_half": sh}
            out["%s|%s" % (m, v)] = rec
            for s in (ss, sh):
                p = per[s]
                print("  %-12s %s E %-6s (%s) | tune P %s, P + E %s | variant by C: %s -> c' %s | %s" % (
                    m, v, s, "s_sel" if s == ss else "s_half", "%+d (%d)" % p["S0"], "%+d (%d)" % p["P+E"],
                    " ".join("%s %+d" % (k, p["variant_by_c"][k]) for k in KS), p["c"], cap_str(p["cap"])))
            print("  %-12s %s -> s' %s%s" % (
                m, v, chosen if chosen else "NONE (neither strength passes the capability line)",
                "" if chosen else "; both strengths are shown below, marked; the effect rule alone would take %s" % rec["effect_rule_only"]))
    print("  capability of a variant: A: %s | B: %s" % (cap_cost(C, ORDER[0], "A", TQ[ORDER[0]]["ss"])["how"], cap_cost(C, ORDER[0], "B", TQ[ORDER[0]]["ss"])["how"]))
    print("  base items blank by the rule but right (GSM8K, plain no-write pass): " + " | ".join(
        "%s %d" % (m, out["%s|A" % m]["per"][TQ[m]["ss"]]["cap"]["base_blank_but_right"]) for m in ORDER))
    DUMP["choices"] = out
    return out


def readings(TQ, FQ, QQ, CH):
    print("\n" + "=" * 110 + "\n# 5 the two stackings: each variant minus P, minus the method M (no sentence), minus single E (no sentence); handed in only")
    out = {}
    pair_of = FQ["pair_of"]
    for v, fn in VARS:
        for sett in ("TruthfulQA test", "FalseQA 600", "(QA)^2 470"):
            print("\n## variant %s, %s" % (v, sett))
            agg = {"s_sel": {}, "s_half": {}}
            for m in ORDER:
                ch = CH["%s|%s" % (m, v)]
                shown = [ch["chosen"]] if ch["chosen"] else [ch["s_sel"], ch["s_half"]]
                for s in shown:
                    tag = "s_sel" if s == ch["s_sel"] else "s_half"
                    c = ch["per"][s]["c"]
                    mark = "s' (passes the line)" if ch["chosen"] else "FAILS THE CAPABILITY LINE (%s)" % tag
                    capk = ch["per"][s]["cap"]
                    if sett == "TruthfulQA test":
                        x = TQ[m]["test"][s]
                        if x is None:
                            print("  %-12s E %-6s C %-4s %s: MISSING (no P + E(%s) / P + E + C passes on TruthfulQA test)" % (m, s, c, mark, s))
                            out.setdefault("missing", []).append([v, sett, m, s, "no TruthfulQA test passes at this E strength"])
                            continue
                        R = TQ[m]["R"]
                        ids = R["ids"]
                        lack = set(ids) - set(x["ids"])
                        if lack:
                            raise SystemExit("%s: the TruthfulQA test sentence arms at E %s lack %d of the method's questions" % (m, s, len(lack)))
                        X = fn(x["ids"], x["arm"], c)
                        P, M, E = hi(TQ[m]["test"][TQ[m]["ss"]]["arm"]["O"], ids), R["M"], hi(R["arm"]["E"], ids)
                        ci = lambda a, b: RD.tq([a[q][0] - b[q][0] for q in ids])  # noqa: E731
                    elif sett == "FalseQA 600":
                        d = FQ[m]
                        if d["sent"][s] is None:
                            print("  %-12s E %-6s C %-4s %s: MISSING (%s)" % (m, s, c, mark, d["miss"][s][:160]))
                            out.setdefault("missing", []).append([v, sett, m, s, d["miss"][s]])
                            continue
                        ids = d["ids"]
                        X = fn(ids, d["sent"][s], c)
                        P, M, E = hi(d["S0"], ids), d["M"], hi(d["arm"]["E"], ids)
                        ci = lambda a, b: [int(z) for z in F.pair_ci({q: a[q][0] - b[q][0] for q in ids}, pair_of, ids)]  # noqa: E731
                    else:
                        d = QQ[m]
                        if d["sent"][s] is None:
                            print("  %-12s E %-6s C %-4s %s: MISSING (%s)" % (m, s, c, mark, d["miss"][s][:160]))
                            out.setdefault("missing", []).append([v, sett, m, s, d["miss"][s]])
                            continue
                        ids = d["ids470"]
                        X = fn(d["ids"], d["sent"][s], c)
                        P, M, E = hi(d["sent"][strengths(m)[0]]["O"], ids), d["M"], hi(d["arm"]["E"], ids)
                        ci = lambda a, b: jci([a[q][0] - b[q][0] for q in ids])  # noqa: E731
                    r = {"s": s, "tag": tag, "c": c, "mark": mark, "n": len(ids), "X": nh(X, ids), "P": nh(P, ids), "M": nh(M, ids),
                         "single E": nh(E, ids), "X-P": ci(X, P), "X-M": ci(X, M), "X-single E": ci(X, E),
                         "right": sum(X[q][0] == 1 for q in ids), "wrong": sum(X[q][0] == -1 for q in ids),
                         "capability": {k: capk[k] for k in ("mmlu", "gsm_lo", "gsm_hi", "gsm_unknown", "pass")}}
                    out.setdefault("%s|%s" % (v, sett), {}).setdefault(m, []).append(r)
                    agg[tag][m] = (ids, X, P, M, E)
                    print("  %-12s E %-6s C %-4s %s | %d q | %s %s (right %d wrong %d) | P %s | M %s | single E %s" % (
                        m, s, c, mark, len(ids), v, show(X, ids), r["right"], r["wrong"], show(P, ids), show(M, ids), show(E, ids)))
                    print("  %-12s %s - P %s | %s - M %s | %s - single E %s | capability: %s" % (
                        "", v, fmt(r["X-P"]), v, fmt(r["X-M"]), v, fmt(r["X-single E"]), cap_str(capk)))
            # reference rows: the five models at one strength each (only when all five have a reading there)
            for tag in ("s_sel", "s_half"):
                g = agg[tag]
                if len(g) != len(ORDER):
                    print("  five models at %s (reference only): not every model has a reading at %s here (%d of 5)" % (tag, tag, len(g)))
                    continue
                common = sorted(set.intersection(*(set(g[m][0]) for m in ORDER)))
                if sett == "TruthfulQA test":
                    ci = RD.tq
                elif sett == "FalseQA 600":
                    ci = lambda vv: [int(z) for z in F.pair_ci(dict(zip(common, vv)), pair_of, common)]  # noqa: E731
                else:
                    ci = jci
                r = {}
                for nm, j in (("X-P", 2), ("X-M", 3), ("X-single E", 4)):
                    r[nm] = ci([sum(g[m][1][q][0] - g[m][j][q][0] for m in ORDER) for q in common])
                r["X"] = [sum(RT.net(g[m][1], common) for m in ORDER), sum(sum(bool(g[m][1][q][1]) for q in common) for m in ORDER)]
                r["n"] = len(common)
                out.setdefault("%s|%s|five models (reference only)" % (v, sett), {})[tag] = r
                print("  five models at %s, each with its own c' (reference only; %d common q) | %s %+d (%d) | %s - P %s | %s - M %s | %s - single E %s" % (
                    tag, len(common), v, r["X"][0], r["X"][1], v, fmt(r["X-P"]), v, fmt(r["X-M"]), v, fmt(r["X-single E"])))
    DUMP["readings"] = out


def main():
    with open(BUNDLE, encoding="utf-8") as f:
        B = json.load(f)
    if not regressions(B):
        print("\nSTOPPED: a regression differs; no new number is printed")
        print("JSON " + json.dumps(DUMP, sort_keys=True))
        raise SystemExit(1)
    TQ = tq_data(B)
    tq_checks(TQ, B)
    FQ = fq_data(B)
    fq_checks(FQ, B)
    QQ = qq_data(B)
    qq_checks(QQ, B)
    PQ = pop_data(B)
    C = cap_data()
    print("\nALL REGRESSIONS AND VALUE CHECKS AGREE; new numbers follow")
    opponent(TQ, FQ, QQ, PQ, C)
    CH = choose(TQ, C)
    readings(TQ, FQ, QQ, CH)
    print("JSON " + json.dumps(DUMP, sort_keys=True))


if __name__ == "__main__":
    main()
