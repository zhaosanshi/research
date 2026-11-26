#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-27), work queue discussion/work_queue/2026-11-26_queue_p.md
# section 5, segment B. Read-only check, no model, no judge call. Not reviewed by GPT.
# Question: segment B decides "blank by the rule" of a NEW cell from its generation row (tqa_gate2b_1127.blank_row), while
# the existing N1 code uses sp.blank read from the judge work dirs. Are the two the same rule?
# Check: on the EXISTING arms at the original strength (single E and the four E + C rungs, write rule "prompt", TruthfulQA
# test, main list), compare blank_row(generation row) with sp.blank and the generation text with tx.ec, cell by cell.
# usage (container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/blank_rule_check_gate2b_1127.py
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import tqa_gate2b_1127 as H      # noqa: E402

total = bad = 0
for m in H.ORDER:
    B = H.base(m)
    el, s, cl, s2 = H.SET[m]
    sp, tx = B.sp, B.tx
    pairs = [("x:E%d.w.p%s" % (el, s), sp.cE(B.sd))]
    for k in H.LAD:
        pairs.append(("x:E%d.w.p%s+C%d.w.m%s" % (el, s, cl, H.DOT[k]), sp.cEC(B.sd, k)))
    for arm, cond in pairs:
        p = os.path.join(B.run, "eval_extra", "gen", H.gen_name(arm))
        if not os.path.exists(p):
            raise SystemExit("%s %s: no generation file under that name" % (m, arm))
        rows = {r["pair_id"]: r for r in H.read(p)["rows"]}
        nb = sum(H.blank_row(rows[q]) != sp.blank[(q, cond)] for q in B.ids)
        nt = sum(rows[q]["text"] != tx.ec[(q, cond)] for q in B.ids)
        total += len(B.ids)
        bad += nb + nt
        print("%-12s %-28s questions %d | blank flag differs %d | text differs %d | blank by the rule %d" % (
            m, arm, len(B.ids), nb, nt, sum(sp.blank[(q, cond)] for q in B.ids)))
print("cells compared %d | differences %d" % (total, bad))
if bad:
    raise SystemExit(1)
