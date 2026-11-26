#!/usr/bin/env python3
# Iteration_01 exploration (executor xu, Claude Opus 5.5, 2026-11-27), work queue discussion/work_queue/2026-11-26_queue_p.md
# section 5, segment B (progress entry 2026-11-27 05:12). Not reviewed by GPT. Read-only table: existing TruthfulQA test
# answers and verdicts (main list, read as diag/gate_sim_1126.py), the five new generation files of
# diag/tqa_gate2b_1127.py and the DeepSeek Pro verdicts of the work dir <run>/eval_extra/judge_test_gate2b_1127_tqa_test;
# no card, no judge call, nothing is chosen here. The test split had been read before in other comparisons.
# N1(s2) (the method's text rule; tqa_gate2b_1127.n1_rule = final_records.ladder_record with veto): pass 1 no write (not
#   handed in); pass 2 single E(s2): answered by the rule -> hand it in; blank -> rungs E(s2) + C(k) from 0.25 up to the
#   cap of the class (own blank = no-write answer blank by the rule; caps of the main table), the first rung answered by
#   the rule is handed in, except (veto) when the class is "E's blank" and the rung's normalised text equals the no-write
#   answer's (release_origin.norm). A question not handed in keeps the verdict of its E cell, as the existing N1 code.
# Gated system: switch ON (pool B, chosen layer, p > 0.5; falseqa_matched.verify_gate(m, "tq_test", ids)) -> the system's
#   cell, OFF -> the no-write cell. Natural answer counts. Intervals: gate_sim_1126.tq_ci (questions resampled).
# Checks before any reading (a mismatch stops): known nets K_TQ; stage 1 (switch-on count and gated N1 - N1 at s, pool B,
#   chosen layer, as printed in results/write_positions/gate_1126_sim_20261126.txt); N1(s) rebuilt with n1_rule = S.system's
#   N1 on every question (state and answered); forced all-ON = ungated and all-OFF = no write, cell by cell; judged cells
#   = cells that can be handed in; judged text = generation text; every cell the systems use has a verdict.
# usage (container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/table_gate2b_1127.py MODEL ...
#                                                      python3 -B Iteration_01/diag/table_gate2b_1127.py smoke MODEL
import collections
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import tqa_gate2b_1127 as H      # noqa: E402

GS, S, ps, LAD = H.GS, H.S, H.ps, H.LAD
G, VAL = H.G, H.VAL
STAGE1 = {"llama3.1-8b": (394, "-29 [-52, -5]"), "qwen3.5-9b": (385, "-3 [-18, +12]"), "qwen3.8-27b": (412, "-11 [-29, +7]"),
          "gemma3-12b": (390, "-8 [-32, +16]"), "gemma3-27b": (403, "-21 [-37, -7]")}
SHOW = {"llama3.1-8b": "Llama-3.1-8B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.8-27b": "Qwen3.8-27B", "gemma3-12b": "Gemma-3-12B",
        "gemma3-27b": "Gemma-3-27B"}


def v(state):
    return VAL[G[state]]


def side(sysd, qs):
    """handed in / right / wrong of one system ({q: (state, answered)})"""
    return "%d / %d / %d" % (sum(sysd[q][1] for q in qs), sum(sysd[q][0] == "R" for q in qs), sum(sysd[q][0] == "W" for q in qs))


def diff(a, b, qs):
    return GS.tq_ci([v(a[q][0]) - v(b[q][0]) for q in qs])


def model_block(m, smoke):
    import falseqa_matched as FM
    B = H.base(m)
    qs = B.ids
    sp, tx = B.sp, B.tx
    out = []
    p = out.append
    # ---- known nets (n, O, E, N1) and the N1(s) rebuild
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
    # ---- switch and stage 1
    on_l = FM.verify_gate(m, "tq_test", qs)
    on = {q: q in set(on_l) for q in qs}
    gate = lambda a: {q: (a[q] if on[q] else o[q]) for q in qs}  # noqa: E731
    g1 = gate(n1)
    s1 = GS.fmt(diff(g1, n1, qs))
    if (len(on_l), s1) != STAGE1[m]:
        raise SystemExit("%s: stage 1 (switch on, gated N1 - N1) %s %s, reference %s" % (m, len(on_l), s1, STAGE1[m]))
    # ---- the new cells
    new = H.load_new(B)
    cn = H.conds(m)
    w = H.work(B, "smoke" if smoke else "full")
    vst, _ = ps.load(w, "pro")
    if smoke:
        qs_r = [q for q in qs if q in set(H.read(H.SMOKE_IDS)["ids"])]
    else:
        qs_r = qs
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
    if judged != want or badtext:
        raise SystemExit("%s: judged cells %d, cells that can be handed in %d (only judged %d, only needed %d); text mismatches %d" % (
            m, len(judged), len(want), len(judged - want), len(want - judged), badtext))
    miss = [k for k in want if vst.get(k, "?") == "?"]
    if miss:
        raise SystemExit("%s: %d cells that can be handed in have no Pro verdict (%s)" % (m, len(miss), miss[:3]))

    def cell(n, q):
        r = new[n][q]
        return (vst.get((q, cn[n]), "X") if (q, cn[n]) in want else "X", H.blank_row(r), r["text"])
    e2c = {q: cell("E2", q) for q in qs_r}
    rc2 = {k: {q: cell("EC2_" + k, q) for q in qs_r} for k in LAD}
    r2 = H.n1_rule(qs_r, bb, bt, e2c, rc2, B.cap_e, B.cap_own)
    # every state that enters N1(s2) or E(s2) comes from a judged cell or from a blank cell (state X)
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
    # forced checks (cell by cell)
    chk = []
    for lab, a in (("N1(s2)", n2), ("E(s2)", e2)):
        chk.append(GS.forced_checks("%s %s" % (m, lab), qs_r, a, {q: o[q] for q in qs_r}))
    if smoke:
        p("## smoke %s: %d questions (ids from %s); caps E's / own %s / %s; checks passed: known nets %s, stage 1 %s, N1(s) rule = S.system on %d, "
          "judged cells = cells that can be handed in (%d), judged text = generation text" % (
              m, len(qs_r), H.SMOKE_IDS, B.cap_e, B.cap_own, got, STAGE1[m], len(qs), len(want)))
        for c in chk:
            p("   " + c)
        for q in qs_r:
            x = r2[q]
            gv = n2[q] if on[q] else o[q]
            p("   %s switch %s (p > 0.5: %s) | no write: blank %s, verdict %s | E(s2): blank %s, judged %s, verdict %s | rungs (blank, judged, verdict): %s" % (
                q, "ON" if on[q] else "OFF", on[q], bb[q], o[q][0], e2c[q][1], (q, cn["E2"]) in want, e2c[q][0],
                " ".join("%s:(%s,%s,%s)" % (H.DOT[k], int(rc2[k][q][1]), int((q, cn["EC2_" + k]) in want), rc2[k][q][0]) for k in LAD)))
            p("      N1(s2): %s, class %s, rung %s -> handed in %s, verdict %s | gated N1(s2): handed in %s, verdict %s | N1(s): %s / %s (%s)" % (
                x["note"], x["cand"], H.DOT.get(x["rung"], x["rung"]), x["answered"], x["state"], gv[1], gv[0], n1[q][0], n1[q][1], r1[q]["note"]))
        return out, None
    gn2 = gate(n2)
    ge2 = gate(e2)
    ge1 = gate(e1)
    p("## %s (%s): main list %d questions; E layer %d, s %s -> s2 %s; C layer %d; caps E's / own %s / %s; switch ON %d" % (
        SHOW[m], m, len(qs), B.el, B.s, B.s2, B.cl, H.DOT[B.cap_e], H.DOT[B.cap_own], len(on_l)))
    p("   checks passed: known nets n / O / E / N1 %s; stage 1 switch on %d, gated N1 - N1 at s %s; N1(s) rebuilt by the rule = S.system on %d / %d questions;"
      " judged cells = cells that can be handed in (%d); judged text = generation text" % (got, len(on_l), s1, len(qs), len(qs), len(want)))
    for c in chk:
        p("   " + c)
    p("   new cells judged: " + ", ".join("%s %d" % (H.arms(m)[n], len(l)) for n, l in need.items()))
    rows = {}

    def rd(key, label, a, b):
        t = diff(a, b, qs)
        rows[key] = t
        p("   %-4s %-44s %-16s | left: handed in / right / wrong %s | right side %s" % (key, label, GS.fmt(t), side(a, qs), side(b, qs)))
    p("   nets: no write %+d | E(s) %+d | E(s2) %+d | N1(s) %+d | N1(s2) %+d | gated: E(s) %+d, E(s2) %+d, N1(s) %+d, N1(s2) %+d" % tuple(
        sum(v(d[q][0]) for q in qs) for d in (o, e1, e2, n1, n2, ge1, ge2, g1, gn2)))
    rd("R1", "gated N1(s2) - N1(s) [main]", gn2, n1)
    rd("R2", "gated N1(s2) - gated N1(s)", gn2, g1)
    rd("R3a", "N1(s2) - N1(s), ungated [reference]", n2, n1)
    rd("R3b", "E(s2) - E(s), ungated [reference]", e2, e1)
    rd("R4g", "gated N1(s2) - gated E(s2)", gn2, ge2)
    rd("R4u", "N1(s2) - E(s2), ungated", n2, e2)
    # R5: wrong with no write and wrong under single E(s) (ww_look_1126 definition)
    ww = [q for q in qs if G[o[q][0]] + G[e1[q][0]] == "WW"]

    def dist(d):
        c = collections.Counter("R" if d[q][0] == "R" else "W" if d[q][0] == "W" else ("H" if d[q][1] else "N") for q in ww)
        return "right %d, wrong %d, hedge %d, not handed in %d" % (c["R"], c["W"], c["H"], c["N"])
    p("   R5   'wrong with no write and wrong under E(s)': %d questions | N1(s) [baseline]: %s | gated N1(s2): %s | N1(s2): %s" % (
        len(ww), dist(n1), dist(gn2), dist(n2)))
    # flow of N1(s2) by the rule (n1_flow_rule_1127 layout)

    def vd(qq, d):
        c = collections.Counter(d[q][0] for q in qq)
        return "right %d wrong %d hedge %d judged-no-answer %d" % (c["R"], c["W"], c["H"], c["X"])
    b2 = {q: e2c[q][1] for q in qs}
    own = [q for q in qs if b2[q] and bb[q]]
    eb = [q for q in qs if b2[q] and not bb[q]]
    handed2 = [q for q in qs if not b2[q]]
    notes = collections.Counter(r2[q]["note"] for q in qs if b2[q])
    rel_own = [q for q in own if r2[q]["note"] == "released"]
    rel_eb = [q for q in eb if r2[q]["note"] == "released"]
    fin = [q for q in qs if n2[q][1]]
    c = collections.Counter(n2[q][0] for q in qs)
    p("   flow of N1(s2) by the rule: pass 1 answered %d, blank %d | pass 2 E(s2) answered %d (handed in; %s), blank %d = own %d + E's %d |"
      " pass 3 notes %s | released own %d (%s), E's %d (%s) | vetoed %d | still blank %d | handed in %d, not handed in %d | net %+d" % (
          sum(not bb[q] for q in qs), sum(bb[q] for q in qs), len(handed2), vd(handed2, n2), sum(b2.values()), len(own), len(eb),
          dict(notes), len(rel_own), vd(rel_own, n2), len(rel_eb), vd(rel_eb, n2), notes["vetoed"],
          sum(b2.values()) - len(rel_own) - len(rel_eb), len(fin), len(qs) - len(fin), c["R"] - c["W"]))
    gfin = [q for q in qs if gn2[q][1]]
    p("   gated N1(s2): switch ON %d (handed in %d), OFF %d (handed in %d, the no-write answers)" % (
        len(on_l), sum(gn2[q][1] for q in qs if on[q]), len(qs) - len(on_l), sum(gn2[q][1] for q in qs if not on[q])))
    assert len(gfin) == sum(gn2[q][1] for q in qs)
    return out, rows


def main():
    a = sys.argv[1:]
    smoke = a[:1] == ["smoke"]
    models = a[1:] if smoke else a
    if not models or any(m not in H.ORDER for m in models):
        raise SystemExit("usage: table_gate2b_1127.py MODEL ... | smoke MODEL")
    models = [m for m in H.ORDER if m in models]
    print("# gate2b_1127 (work queue 2026-11-26_queue_p section 5, segment B): TruthfulQA test, main list; N1 and single E at s and at s2,"
          " with and without the frozen switch (pool B, chosen layer, p > 0.5); natural answer counts; intervals: questions resampled (gate_sim_1126.tq_ci)")
    print("# exploration, not reviewed by GPT; the test split had been read before in other comparisons")
    res = {}
    for m in models:
        out, rows = model_block(m, smoke)
        print("\n".join(out))
        res[m] = rows
    if smoke:
        return
    print("\n## summary (one row per model; no pooled row): difference [95%]")
    print("   %-13s %-18s %-18s %-18s %-18s %-18s %-18s" % ("model", "R1 gN1(s2)-N1(s)", "R2 gN1(s2)-gN1(s)", "R3 N1(s2)-N1(s)",
                                                            "R3 E(s2)-E(s)", "R4 gN1(s2)-gE(s2)", "R4 N1(s2)-E(s2)"))
    for m in models:
        r = res[m]
        print("   %-13s %-18s %-18s %-18s %-18s %-18s %-18s" % (SHOW[m], *[GS.fmt(r[k]) for k in ("R1", "R2", "R3a", "R3b", "R4g", "R4u")]))
    missing = [SHOW[m] for m in H.ORDER if m not in models]
    if missing:
        print("   not judged yet: " + ", ".join(missing))


if __name__ == "__main__":
    main()
