#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Opus 5.5, 2026-11-27), work queue discussion/work_queue/2026-11-26_queue_p.md
# section 5: the readings of segments A and B on a subset of the questions (the 100-question passes). Read-only; no model,
# no judge call, nothing chosen here. Not reviewed by GPT.
# The model blocks are diag/table_gate2_1127.model_block (A, FalseQA) and diag/table_gate2b_1127.model_block (B,
# TruthfulQA) with two changes only: (1) every check of the original runs unchanged on the whole question list; (2) the
# new cells, the systems at s2 and every reading are restricted to the subset (A: the 50 pairs of
# q100_gate2_1127.DRAW_A; B: the Dr's 100 in the main list), and the verdicts are read from the work dir WORK:
#   full  = the full-run work dir (its judged cells must include every cell that can be handed in on the subset)
#   q100  = the 100-question work dir (its judged cells must equal the cells that can be handed in on the subset)
# SUB = all reproduces the original table's reading lines exactly (the regression check of this copy); SUB = q100 is the
# 100-question reading. Intervals: A pairs resampled (falseqa_read.pair_ci), B questions resampled (gate_sim_1126.tq_ci),
# both on the subset.
# usage (container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/table_q100_gate2_1127.py a WORK SUB MODEL ...
#   python3 -B Iteration_01/diag/table_q100_gate2_1127.py b WORK SUB MODEL ...
import collections
import json
import os
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/gpt_tools_20261126/gate_audit"]
import q100_gate2_1127 as QH       # noqa: E402


# ------------------------------------------------------------------------------------------------------------------- A
def a_verdicts(w, jin):
    """table_gate2_1127.verdicts with the work dir given"""
    import falseqa_gate2_1127 as AH
    from judge_two_axis_run import load_verified_single
    _, cells, votes = load_verified_single(str(w), "pro", generation_path=os.path.join(str(w), jin), require_complete=False)
    val, pred, declined = {}, {}, {}
    for key, c in cells.items():
        if not c["prediction"].strip():
            continue
        v = votes[key]
        k = (c["pair_id"].split(":", 1)[1], c["condition"])
        if v["status"] == "scored":
            val[k] = AH.Q.VALUE[v["label"]]
        elif v.get("basis") == "pro_declined":
            val[k] = 0
            declined[k] = c["prediction"]
        else:
            raise SystemExit("%s: unresolved vote for %s %s" % (w, k[0], k[1]))
        pred[k] = c["prediction"]
    return val, pred, declined


def a_block(m, rows, mg_all, work, sub, report):
    import falseqa_gate2_1127 as H
    import table_gate2_1127 as T
    F, Q, GS = H.F, H.Q, H.GS
    el, s, cl, s2, st = H.setting2(m)
    B = H.base_state(m, rows)
    ids_all, on = B.ids, set(B.on)
    qt = {r["id"]: r["question"] for r in rows}
    pair = {r["id"]: r["pair_id"] for r in rows}
    prem = {r["id"]: r["premise"] for r in rows}
    sysd = F.systems_of(B.R, B.R2, st, B.caps, ids_all)
    # ---- 0. checks of the original, on all 600 (unchanged)
    if tuple(sum(sysd[k][q][0] for q in ids_all) for k in ("O", "E", "N1", "Conly")) != GS.K_FQA[m]:
        raise SystemExit("%s: falseqa_read nets differ from the known ones" % m)
    O = sysd["O"]
    g = {n: T.gated(ids_all, on, sysd[n], O) for n in ("N1", "E", "Conly")}
    natt = {n: (sum(c[0] for c in g[n].values()), sum(c[1] for c in g[n].values())) for n in g}
    if (len(on), natt) != T.AUD[m]:
        raise SystemExit("%s: gated systems at s differ from the audit: ON %d %s, audit %s" % (m, len(on), natt, T.AUD[m]))
    for n in ("N1", "E", "Conly"):
        if T.gated(ids_all, set(ids_all), sysd[n], O) != sysd[n] or T.gated(ids_all, set(), sysd[n], O) != O:
            raise SystemExit("%s: forced check of %s" % (m, n))
    M = mg_all
    onl_all = [q for q in ids_all if q in on]
    aud = []
    for n in ("E", "Conly"):
        r = T.equal_q(g["N1"], g[n], ids_all, on, M, qt, pair)
        aud.append(tuple(r[0]) if r[0] is not None else None)
    for strat in (False, True):
        tp, _ = T.two_pass_q(onl_all, sysd["E"], sysd["N1"], O, sysd["TE"], M, qt, strat)
        aud.append(None if tp is None else tuple(F.pair_ci({q: (sysd["N1"][q][0] - tp[q][0]) if q in on else 0 for q in ids_all}, pair, ids_all)))
    if tuple(aud) != T.AUD_EQ[m]:
        raise SystemExit("%s: equal-count / two-pass lines at s differ from the audit: %s vs %s" % (m, aud, T.AUD_EQ[m]))
    R = B.R
    e = R.e(s)
    reb = T.n1_rebuild(ids_all, lambda q: (R.cell(e, q), not R.blank(e, q), R.text(e, q)), lambda q: R.blank("x", q),
                       lambda q: R.text("x", q), lambda k, q: (R.cell(R.ec(s, k), q), not R.blank(R.ec(s, k), q), R.text(R.ec(s, k), q)),
                       B.caps["N1"])
    if reb != sysd["N1"]:
        raise SystemExit("%s: the N1 rebuild at s differs from falseqa_read on %d questions" % (m, sum(reb[q] != sysd["N1"][q] for q in ids_all)))
    # ---- the subset
    if sub == "q100":
        P = set(QH.read(QH.DRAW_A)["pairs"])
        ids = [q for q in ids_all if pair[q] in P]
        sub_txt = "subset: %d pairs of %s (sha256 %s), %d questions" % (len(P), QH.DRAW_A, QH.sha16(QH.DRAW_A), len(ids))
    else:
        ids = list(ids_all)
        sub_txt = "subset: all %d questions" % len(ids)
    onl = [q for q in ids if q in on]
    # ---- new cells at s2, verdicts of WORK
    new, cond, _ = H.load_new(m, "full", rows)
    w = QH.work_a(m, work)
    val, pred, declined = a_verdicts(w, H.JIN)
    need = H.needed(m, B, new, onl)
    needk = {(q, cond[n]) for n, l in need.items() for q in l}
    if work == "q100" or sub == "all":
        if set(val) != needk:
            raise SystemExit("%s: judged cells %d differ from the cells that can be handed in %d (extra %d, missing %d)" % (
                m, len(val), len(needk), len(set(val) - needk), len(needk - set(val))))
    elif not needk <= set(val):
        raise SystemExit("%s: %d cells that can be handed in on the subset have no verdict in %s" % (m, len(needk - set(val)), w))
    for (q, c) in needk:
        n = next(k for k, v in cond.items() if v == c)
        if pred[(q, c)].strip() != new[n][q]["text"].strip():
            raise SystemExit("%s: the judged answer of %s under %s is not the text of the new file" % (m, q, c))

    def cell(n, q):
        r = new[n][q]
        if not r["text"].strip():
            return (0, False, r["text"], None)
        if (q, cond[n]) not in val:
            raise SystemExit("%s: %s of %s has no verdict but is used" % (m, n, q))
        return (val[(q, cond[n])], not H.blank_row(r), r["text"], None)
    E2 = {q: cell("E2", q) for q in onl}
    N12 = T.n1_rebuild(onl, lambda q: E2[q][:3], lambda q: R.blank("x", q), lambda q: R.text("x", q),
                       lambda k, q: cell("EC2_" + k, q)[:3], B.caps["N1"])
    TE2 = {q: (cell("TE2", q) if not E2[q][1] else None) for q in onl}
    gs2 = {"N1": T.gated(ids, on, N12, O), "E": T.gated(ids, on, E2, O)}
    for n, S in (("N1", N12), ("E", E2)):
        if any(gs2[n][q] != S[q] for q in onl) or any(gs2[n][q] != O[q] for q in ids if q not in on):
            raise SystemExit("%s: forced check of %s at s2" % (m, n))
    dec = sorted(k for k in declined if k in needk)
    print("\n" + "=" * 110)
    print("## %s: E%d s %s -> s2 %s, C%d; caps N1 (E's blanks %s, own blanks %s), Conly %s; switch ON %d / 600; %s; ON in the subset %d "
          "(premise false %d, valid %d); verdicts: %s work dir" % (
              m, el, s, s2, cl, Q.show_cap(B.caps["N1"][0]), Q.show_cap(B.caps["N1"][1]), B.caps["Conly"], len(on), sub_txt, len(onl),
              sum(prem[q] == "false" for q in onl), sum(prem[q] == "valid" for q in onl), work))
    print("checks: falseqa_read nets %s; audit ON / natural %s; audit equal-count and two-pass lines at s %s; N1 rebuild at s "
          "= falseqa_read on 600; forced ON / OFF; judged cells %s cells that can be handed in on the subset (%d)%s -> PASS" % (
              GS.K_FQA[m], natt, aud, "=" if (work == "q100" or sub == "all") else "include the", len(needk),
              "; declined (value 0): %s" % dec if dec else ""))
    print("new cells on the ON questions of the subset: %s" % " | ".join("%s %d" % (n, len(l)) for n, l in need.items()))
    groups = [("premise false", [q for q in ids if prem[q] == "false"]), ("premise valid", [q for q in ids if prem[q] == "valid"]),
              ("all", ids)]
    tp2 = {}
    for strat in (False, True):
        tp, info = T.two_pass_q(onl, E2, N12, O, {q: (TE2[q] if TE2[q] is not None else (0, False, "", None)) for q in onl}, M, qt, strat)
        tp2[strat] = (None if tp is None else T.gated(ids, on, tp, O), info)
        tpn, infn = T.two_pass_q(onl, E2, N12, O, {q: (TE2[q] if TE2[q] is not None else (0, False, "", None)) for q in onl}, M, qt,
                                 strat, natural=True)
        tp2[("nat", strat)] = (T.gated(ids, on, tpn, O), infn)
    print("-- 1 MAIN: gated N1(s2) - gated N1(s)")
    for lab, qs in groups:
        a, b = gs2["N1"], g["N1"]
        print("  %-14s natural %s | s2 %s | s %s" % (lab, T.fmt(T.nat(a, b, qs, pair)), T.show_hrw(T.hrw(a, qs)), T.show_hrw(T.hrw(b, qs))))
        r = T.equal_q(a, b, qs, on, M, qt, pair)
        if r[0] is None:
            print("  %-14s equal: %s (%s)" % ("", T.NA, r[1]))
        else:
            print("  %-14s equal   %s (ON answers %d vs %d -> %d each) | s2 %s | s %s" % (
                "", T.fmt(r[0]), r[2][0], r[2][1], r[2][2], T.show_hrw(r[3]), T.show_hrw(r[4])))
        if lab == "all":
            report[m] = r[0]
    print("-- 2 gated N1(s2) minus: gated single E(s2); gated Conly; gated two-pass E(s2)")
    for lab, qs in groups:
        for name, other in (("E(s2)", gs2["E"]), ("Conly", g["Conly"])):
            a = gs2["N1"]
            r = T.equal_q(a, other, qs, on, M, qt, pair)
            eqs = ("%s (ON answers %d vs %d -> %d each) | N1(s2) %s | %s %s" % (T.fmt(r[0]), r[2][0], r[2][1], r[2][2], T.show_hrw(r[3]),
                                                                              name, T.show_hrw(r[4])) if r[0] is not None else "%s (%s)" % (T.NA, r[1]))
            print("  %-14s - %-6s natural %s (%s: %s) | equal %s" % (lab, name, T.fmt(T.nat(a, other, qs, pair)), name,
                                                                     T.show_hrw(T.hrw(other, qs)), eqs))
        for strat in (True, False):
            tpn, infn = tp2[("nat", strat)]
            tp, info = tp2[strat]
            lab2 = "two-pass E(s2)%s" % (" with provenance classes + veto" if strat else " one pool, no veto")
            line = "  %-14s - %s: natural (every passing fill, %d) %s (%s)" % (lab, lab2, infn, T.fmt(T.nat(gs2["N1"], tpn, qs, pair)),
                                                                              T.show_hrw(T.hrw(tpn, qs)))
            if tp is None:
                line += " | same fill slots as N1: %s (deficit %s)" % (T.NA, info)
            else:
                line += " | same fill slots as N1 (%d kept) %s (%s)" % (info, T.fmt(T.nat(gs2["N1"], tp, qs, pair)), T.show_hrw(T.hrw(tp, qs)))
            print(line)
    print("-- 3 reference: gated single E(s2) - gated single E(s); ON questions changing between s and s2")
    for lab, qs in groups:
        print("  %-14s natural %s | E(s2) %s | E(s) %s" % (lab, T.fmt(T.nat(gs2["E"], g["E"], qs, pair)), T.show_hrw(T.hrw(gs2["E"], qs)),
                                                          T.show_hrw(T.hrw(g["E"], qs))))
    ab = sum(1 for q in onl if sysd["E"][q][1] and not E2[q][1])
    ba = sum(1 for q in onl if not sysd["E"][q][1] and E2[q][1])
    print("  ON questions: answered at s -> blank at s2 %d; blank at s -> answered at s2 %d; blank at both %d; answered at both %d" % (
        ab, ba, sum(1 for q in onl if not sysd["E"][q][1] and not E2[q][1]), sum(1 for q in onl if sysd["E"][q][1] and E2[q][1])))
    ab = sum(1 for q in onl if sysd["N1"][q][1] and not N12[q][1])
    ba = sum(1 for q in onl if not sysd["N1"][q][1] and N12[q][1])
    print("  N1 on ON questions: answered at s -> blank at s2 %d; blank at s -> answered at s2 %d; N1(s2) fills by rung %s" % (
        ab, ba, dict(sorted(collections.Counter(N12[q][3] for q in onl if N12[q][1] and N12[q][3] != "0").items()))))


def run_a(work, sub, models):
    import falseqa_gate2_1127 as H
    import table_gate2_1127 as T
    rows = H.bank_rows()
    print("# q100_gate2_1127, segment A (FalseQA held-out), WORK %s, SUB %s: table_gate2_1127.model_block with the readings on the subset" % (work, sub))
    print("# exploration, not reviewed by GPT; the held-out verdicts of FalseQA had been read in other comparisons (screening only)")
    report = {}
    for m in models:
        a_block(m, rows, T.margins(m), work, sub, report)
    print("\n" + "=" * 110)
    print("## main reading per model (equal counts, all questions of the subset): gated N1(s2) - gated N1(s)")
    for m in models:
        r = report.get(m)
        print("  %-12s %s" % (m, "not comparable" if r is None else "%s -> lower end above 0: %s" % (T.fmt(r), "yes" if r[1] > 0 else "no")))


# ------------------------------------------------------------------------------------------------------------------- B
def b_block(m, work, sub):
    import falseqa_matched as FM
    import tqa_gate2b_1127 as H
    import table_gate2b_1127 as TB
    GS, S, ps, LAD = H.GS, H.S, H.ps, H.LAD
    G = H.G
    v, side = TB.v, TB.side
    B = H.base(m)
    qs = B.ids
    sp, tx = B.sp, B.tx
    out = []
    p = out.append
    # ---- checks of the original on the whole main list (unchanged)
    o = {q: (sp.st[(q, "baseline_x")], not sp.blank[(q, "baseline_x")]) for q in qs}
    cE = sp.cE(B.sd)
    e1 = {q: (sp.st[(q, cE)], not sp.blank[(q, cE)]) for q in qs}
    st, an, _ = S.system(sp, tx, B.pick)
    n1 = {q: (x, y) for q, x, y in zip(qs, st, an)}
    got = (len(qs), sum(v(o[q][0]) for q in qs), sum(v(e1[q][0]) for q in qs), sum(v(n1[q][0]) for q in qs))
    if got != GS.K_TQ[m]:
        raise SystemExit("KNOWN NET MISMATCH %s: %s, reference %s" % (m, got, GS.K_TQ[m]))
    bb = {q: sp.blank[(q, "baseline_x")] for q in qs}
    bt = {q: tx.ec[(q, "baseline_x")] for q in qs}
    ecs = {q: (sp.st[(q, cE)], sp.blank[(q, cE)], tx.ec[(q, cE)]) for q in qs}
    rcs = {k: {q: (sp.st[(q, sp.cEC(B.sd, k))], sp.blank[(q, sp.cEC(B.sd, k))], tx.ec[(q, sp.cEC(B.sd, k))]) for q in qs} for k in LAD}
    r1 = H.n1_rule(qs, bb, bt, ecs, rcs, B.cap_e, B.cap_own)
    badr = [q for q in qs if (r1[q]["state"], r1[q]["answered"]) != n1[q]]
    if badr:
        raise SystemExit("%s: N1(s) rebuilt with the rule differs from S.system on %d questions" % (m, len(badr)))
    on_l = FM.verify_gate(m, "tq_test", qs)
    on = {q: q in set(on_l) for q in qs}
    g1_all = {q: (n1[q] if on[q] else o[q]) for q in qs}
    s1 = GS.fmt(TB.diff(g1_all, n1, qs))
    if (len(on_l), s1) != TB.STAGE1[m]:
        raise SystemExit("%s: stage 1 (switch on, gated N1 - N1) %s %s, reference %s" % (m, len(on_l), s1, TB.STAGE1[m]))
    # ---- the subset
    if sub == "q100":
        D = set(QH.dr100())
        qs_r = [q for q in qs if q in D]
        sub_txt = "subset: the Dr's 100 (%s, sha256 %s), %d in the main list" % (QH.DR100, QH.DR100_SHA, len(qs_r))
    else:
        qs_r = list(qs)
        sub_txt = "subset: all %d questions" % len(qs_r)
    gate = lambda a: {q: (a[q] if on[q] else o[q]) for q in qs_r}  # noqa: E731
    g1 = gate(n1)
    # ---- the new cells, verdicts of WORK
    new = H.load_new(B)
    cn = H.conds(m)
    w = QH.work_b(B, work)
    vst, _ = ps.load(w, "pro")
    need = H.needed(B, new, qs_r)
    want = {(q, cn[n]) for n, l in need.items() for q in l}
    judged, badtext = set(), 0
    for line in open(os.path.join(w, "judge2", "cells.jsonl"), encoding="utf-8"):
        c = json.loads(line)
        if c["prediction"].strip():
            judged.add((c["pair_id"], c["condition"]))
            n = next((n for n, x in cn.items() if x == c["condition"]), None)
            if n is None or c["prediction"] != new[n][c["pair_id"]]["text"]:
                badtext += 1
    if work == "q100" or sub == "all":
        if judged != want or badtext:
            raise SystemExit("%s: judged cells %d, cells that can be handed in %d (only judged %d, only needed %d); text mismatches %d" % (
                m, len(judged), len(want), len(judged - want), len(want - judged), badtext))
    elif not want <= judged or badtext:
        raise SystemExit("%s: %d cells that can be handed in on the subset are not judged in %s; text mismatches %d" % (
            m, len(want - judged), w, badtext))
    miss = [k for k in want if vst.get(k, "?") == "?"]
    if miss:
        raise SystemExit("%s: %d cells that can be handed in have no Pro verdict (%s)" % (m, len(miss), miss[:3]))

    def cell(n, q):
        r = new[n][q]
        return (vst.get((q, cn[n]), "X") if (q, cn[n]) in want else "X", H.blank_row(r), r["text"])
    e2c = {q: cell("E2", q) for q in qs_r}
    rc2 = {k: {q: cell("EC2_" + k, q) for q in qs_r} for k in LAD}
    r2 = H.n1_rule(qs_r, bb, bt, e2c, rc2, B.cap_e, B.cap_own)
    for q in qs_r:
        x = r2[q]
        src = ("E2" if x["cond"] == "E" else "EC2_" + x["cond"])
        if (q, cn[src]) not in want and x["state"] != "X":
            raise SystemExit("%s %s: N1(s2) uses an unjudged cell" % (m, q))
        if (q, cn[src]) not in want and new[src][q]["text"].strip() and (x["answered"] or src != "E2"):
            raise SystemExit("%s %s: N1(s2) hands in an unjudged non-empty cell" % (m, q))
    n2 = {q: (r2[q]["state"], r2[q]["answered"]) for q in qs_r}
    e2 = {q: (e2c[q][0], not e2c[q][1]) for q in qs_r}
    e2_unj = [q for q in qs_r if new["E2"][q]["text"].strip() and (q, cn["E2"]) not in want]
    if e2_unj:
        raise SystemExit("%s: %d non-empty E2 cells not judged" % (m, len(e2_unj)))
    chk = []
    for lab, a in (("N1(s2)", n2), ("E(s2)", e2)):
        chk.append(GS.forced_checks("%s %s" % (m, lab), qs_r, a, {q: o[q] for q in qs_r}))
    gn2 = gate(n2)
    ge2 = gate(e2)
    ge1 = gate(e1)
    on_r = [q for q in qs_r if on[q]]
    p("## %s (%s): main list %d questions; E layer %d, s %s -> s2 %s; C layer %d; caps E's / own %s / %s; switch ON %d; %s; ON in the subset %d; "
      "verdicts: %s work dir" % (TB.SHOW[m], m, len(qs), B.el, B.s, B.s2, B.cl, H.DOT[B.cap_e], H.DOT[B.cap_own], len(on_l), sub_txt, len(on_r), work))
    p("   checks passed: known nets n / O / E / N1 %s; stage 1 switch on %d, gated N1 - N1 at s %s; N1(s) rebuilt by the rule = S.system on %d / %d questions;"
      " judged cells %s cells that can be handed in on the subset (%d); judged text = generation text" % (
          got, len(on_l), s1, len(qs), len(qs), "=" if (work == "q100" or sub == "all") else "include the", len(want)))
    for c in chk:
        p("   " + c)
    p("   new cells judged: " + ", ".join("%s %d" % (H.arms(m)[n], len(l)) for n, l in need.items()))
    rows = {}

    def rd(key, label, a, b):
        t = TB.diff(a, b, qs_r)
        rows[key] = t
        p("   %-4s %-44s %-16s | left: handed in / right / wrong %s | right side %s" % (key, label, GS.fmt(t), side(a, qs_r), side(b, qs_r)))
    p("   nets: no write %+d | E(s) %+d | E(s2) %+d | N1(s) %+d | N1(s2) %+d | gated: E(s) %+d, E(s2) %+d, N1(s) %+d, N1(s2) %+d" % tuple(
        sum(v(d[q][0]) for q in qs_r) for d in (o, e1, e2, n1, n2, ge1, ge2, g1, gn2)))
    rd("R1", "gated N1(s2) - N1(s) [main]", gn2, n1)
    rd("R2", "gated N1(s2) - gated N1(s)", gn2, g1)
    rd("R3a", "N1(s2) - N1(s), ungated [reference]", n2, n1)
    rd("R3b", "E(s2) - E(s), ungated [reference]", e2, e1)
    rd("R4g", "gated N1(s2) - gated E(s2)", gn2, ge2)
    rd("R4u", "N1(s2) - E(s2), ungated", n2, e2)
    ww = [q for q in qs_r if G[o[q][0]] + G[e1[q][0]] == "WW"]

    def dist(d):
        c = collections.Counter("R" if d[q][0] == "R" else "W" if d[q][0] == "W" else ("H" if d[q][1] else "N") for q in ww)
        return "right %d, wrong %d, hedge %d, not handed in %d" % (c["R"], c["W"], c["H"], c["N"])
    p("   R5   'wrong with no write and wrong under E(s)': %d questions | N1(s) [baseline]: %s | gated N1(s2): %s | N1(s2): %s" % (
        len(ww), dist(n1), dist(gn2), dist(n2)))

    def vd(qq, d):
        c = collections.Counter(d[q][0] for q in qq)
        return "right %d wrong %d hedge %d judged-no-answer %d" % (c["R"], c["W"], c["H"], c["X"])
    b2 = {q: e2c[q][1] for q in qs_r}
    own = [q for q in qs_r if b2[q] and bb[q]]
    eb = [q for q in qs_r if b2[q] and not bb[q]]
    handed2 = [q for q in qs_r if not b2[q]]
    notes = collections.Counter(r2[q]["note"] for q in qs_r if b2[q])
    rel_own = [q for q in own if r2[q]["note"] == "released"]
    rel_eb = [q for q in eb if r2[q]["note"] == "released"]
    fin = [q for q in qs_r if n2[q][1]]
    c = collections.Counter(n2[q][0] for q in qs_r)
    p("   flow of N1(s2) by the rule: pass 1 answered %d, blank %d | pass 2 E(s2) answered %d (handed in; %s), blank %d = own %d + E's %d |"
      " pass 3 notes %s | released own %d (%s), E's %d (%s) | vetoed %d | still blank %d | handed in %d, not handed in %d | net %+d" % (
          sum(not bb[q] for q in qs_r), sum(bb[q] for q in qs_r), len(handed2), vd(handed2, n2), sum(b2.values()), len(own), len(eb),
          dict(notes), len(rel_own), vd(rel_own, n2), len(rel_eb), vd(rel_eb, n2), notes["vetoed"],
          sum(b2.values()) - len(rel_own) - len(rel_eb), len(fin), len(qs_r) - len(fin), c["R"] - c["W"]))
    p("   gated N1(s2): switch ON %d (handed in %d), OFF %d (handed in %d, the no-write answers)" % (
        len(on_r), sum(gn2[q][1] for q in qs_r if on[q]), len(qs_r) - len(on_r), sum(gn2[q][1] for q in qs_r if not on[q])))
    return out, rows


def run_b(work, sub, models):
    import tqa_gate2b_1127 as H
    import table_gate2b_1127 as TB
    GS = H.GS
    print("# q100_gate2_1127, segment B (TruthfulQA test, main list), WORK %s, SUB %s: table_gate2b_1127.model_block with the readings on the subset" % (work, sub))
    print("# exploration, not reviewed by GPT; the test split had been read before in other comparisons")
    res = {}
    for m in models:
        out, rows = b_block(m, work, sub)
        print("\n".join(out))
        res[m] = rows
    print("\n## summary (one row per model; no pooled row): difference [95%]")
    print("   %-13s %-18s %-18s %-18s %-18s %-18s %-18s" % ("model", "R1 gN1(s2)-N1(s)", "R2 gN1(s2)-gN1(s)", "R3 N1(s2)-N1(s)",
                                                            "R3 E(s2)-E(s)", "R4 gN1(s2)-gE(s2)", "R4 N1(s2)-E(s2)"))
    for m in models:
        r = res[m]
        print("   %-13s %-18s %-18s %-18s %-18s %-18s %-18s" % (TB.SHOW[m], *[GS.fmt(r[k]) for k in ("R1", "R2", "R3a", "R3b", "R4g", "R4u")]))


def main():
    a = sys.argv[1:]
    if len(a) < 4 or a[0] not in ("a", "b") or a[1] not in ("full", "q100") or a[2] not in ("all", "q100"):
        raise SystemExit("usage: table_q100_gate2_1127.py a|b full|q100 all|q100 MODEL ...")
    order = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
    if any(m not in order for m in a[3:]):
        raise SystemExit("unknown model in %s" % a[3:])
    if a[1] == "q100" and a[2] == "all":
        raise SystemExit("the 100-question work dir has no verdicts outside the subset")
    (run_a if a[0] == "a" else run_b)(a[1], a[2], list(a[3:]))


if __name__ == "__main__":
    main()
