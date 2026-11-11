# -*- coding: utf-8 -*-
"""Read-only: how the PROMPT factors themselves fare under the criteria of plan 28 (no intervention, hence no random control).
E analogue: plain x -> x_E1.  C analogue: conf -> caut (toward caution) and caut -> conf (toward confident).
Uses the functions of e44_round1_tables.py.   usage: e44_prompt_factor_cells.py STEP0_WORK"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t
work = sys.argv[1]
for judge in ("allenai", "pro"):
    gen, specs, ids, state, rule, words = t.load(work, judge)
    T = t.tau(len(ids))
    print("## judge %s (tau = %d)\n" % (judge, T))
    b, c = "baseline_x", "baseline_x_E1"
    v, n = t.per_question(state, words, ids, b, c)
    sR, loR, _ = t.ci(v["dR"]); sG, loG, _ = t.ci(v["same"])
    print("- E analogue x -> x_E1: dR %s (above tau and interval above 0: %s); G_same %s on %d questions (interval above 0: %s); dAns %+d (within +-%d: %s); out / in %d / %d" % (
        sR, loR > 0 and sum(v["dR"]) > T, sG, n["same"], loG > 0, sum(v["dAns"]), T, abs(sum(v["dAns"])) <= T, n["out"], n["in"]))
    for b, c, sign, name in (("baseline_conf", "baseline_caut", -1, "toward caution: conf -> caut"), ("baseline_caut", "baseline_conf", +1, "toward confident: caut -> conf")):
        v, n = t.per_question(state, words, ids, b, c)
        base_ans = sum(state[(i, b)] in "RW" for i in ids)
        sA, loA, hiA = t.ci(v["dAns"]); sX, loX, hiX = t.ci(v["dX"]); sK, loK, hiK = t.ci(v["keep"])
        eff = (hiA < 0 and loX > 0 and -sum(v["dAns"]) > T) if sign < 0 else (loA > 0 and hiX < 0 and sum(v["dAns"]) > T)
        ret = 2 * n["keep"] >= base_ans and abs(sum(v["keep"])) <= T and loK <= 0 <= hiK and n["toH"] <= T
        print("- C analogue %s: dAns %s, dX %s, effect conditions (without the random control): %s | keep n %d (need >= %d), keep dR %s, W>R / R>W %d / %d, toH %d: retention %s" % (
            name, sA, sX, eff, n["keep"], (base_ans + 1) // 2, sK, n["up"], n["down"], n["toH"], ret))
    print()
