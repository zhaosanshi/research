#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-28). Zhao Lei 13:53: two routes for the simple rule on
# TruthfulQA and FalseQA, five models - (1) the 2x2 of "original answered / E answered" with E+C only where both abstain
# (which C strength?), (2) E+C on every question with a growing C - and their mixtures. Read-only: existing answers and
# DeepSeek Pro verdicts; no model, no judge call, nothing generated. Not reviewed by GPT.
# Arms (one pass each, write rule "prompt", the main-table E layer / strength and C layer of each model):
#   O no write | E x:E<el>.w.p<s> | C(k) x:C<cl>.w.m<k> | EC(k) x:E<el>.w.p<s>+C<cl>.w.m<k>, k = 0.25 0.5 1 2.
# Scoring (GPT's "handed in only", 2026-11-28_simple_n1_claude_review.md): an answer that is blank by the method's own text
#   rule is NOT handed in and scores 0 whatever the judge said about its text; a handed-in answer scores +1 right, -1 wrong,
#   0 otherwise. Natural answer counts.
# Schemes (X = what is handed in):
#   A(k)    EC(k) on every question
#   R1(k)   route 1: E answered -> E; E blank and O answered -> nothing; both blank -> EC(k) once
#   R1L(c)  route 1 with a ladder in the both-blank cell: EC(0.25), EC(0.5), ... up to cap c, the first answered rung
#   D(k)    two steps: E answered -> E; E blank -> EC(k) once (whatever O did)     DL(c) the same with the ladder
#   R2u     route 2 as written by Zhao Lei: a = EC(0.25), b = EC(0.5); b answered -> b; a answered and b blank -> nothing;
#           both blank -> EC(1), then EC(2), the first answered
#   R2L     pure ladder on every question: the first answered of EC(0.25), EC(0.5), EC(1), EC(2)
#   N1      the current method (TruthfulQA: S.system with ML.n1_pick on tune; FalseQA: falseqa_read.n1_cells with the caps
#           of falseqa_read.choose on the development questions)
#   R1orc   upper bound of route 1 over the C strength: in the both-blank cell the best outcome among the four rungs per
#           question (right if any rung is right; else nothing if any rung is blank; else wrong) - not a method
#   "t" variants: the strength / cap chosen per model on the tuning questions (TruthfulQA tune 192, FalseQA development 100)
#   as the highest tuning net of that scheme, ties -> the smaller.
# Opponents: single E; single C at the strength chosen on the tuning questions (highest tuning net, ties -> smaller).
# Intervals: TruthfulQA questions resampled (gate_sim_1126.tq_ci), FalseQA pairs resampled (falseqa_read.pair_ci).
# Cross-check: R1(1) and N1 against the nets and handed-in counts of GPT's table (2026-11-28_simple_n1_claude_review.md).
# Output: a text report, then one line "JSON <dump>" with every number.
# usage (container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/routes_1128.py
import json
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/gpt_tools_20261126/gate_audit"]
import iti_table as A             # noqa: E402
import gate_sim_1126 as GS        # noqa: E402
import falseqa_read as F          # noqa: E402

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
VAL = {"R": 1, "W": -1, "X": 0, "H": 0}
KS = ("0.25", "0.5", "1", "2")
# GPT's table: (R1(1) handed in, net), (N1 handed in, net)
GPT = {("TruthfulQA", "llama3.1-8b"): ((352, 63), (476, 62)), ("TruthfulQA", "qwen3.5-9b"): ((437, 177), (444, 186)),
       ("TruthfulQA", "qwen3.8-27b"): ((463, 171), (490, 175)), ("TruthfulQA", "gemma3-12b"): ((358, 128), (489, 159)),
       ("TruthfulQA", "gemma3-27b"): ((469, 135), (510, 144)), ("FalseQA", "llama3.1-8b"): ((468, 212), (509, 239)),
       ("FalseQA", "qwen3.5-9b"): ((523, 342), (523, 334)), ("FalseQA", "qwen3.8-27b"): ((544, 346), (544, 341)),
       ("FalseQA", "gemma3-12b"): ((398, 228), (398, 228)), ("FalseQA", "gemma3-27b"): ((537, 326), (563, 328))}


def tqa_split(m, idx):
    display, layer, s = A.KEY[m]
    sp, tx, singles, ids = MP.prompt_split(display, layer, idx, s)[0]
    if idx == 1:
        src = [x for x in ML.sources(display)[2] if x[0] == layer]
        cx = [q for q in ids if q in set(src[0][3].qs)]
        if cx != ids:
            sp, tx, singles, ids = MP.prompt_split(display, layer, idx, s, qs=cx)[0]
    lad = list(ps.LAD)
    arm = {}

    def cell(cond):
        return {q: (VAL[sp.st[(q, cond)]], not sp.blank[(q, cond)]) for q in ids}
    arm["O"], arm["E"] = cell("baseline_x"), cell(sp.cE(s))
    for k, name in zip(lad, KS):
        arm["EC" + name] = cell(sp.cEC(s, k))
        st, an = singles[k]
        if len(st) != len(ids):
            raise SystemExit("%s: single C %s has %d cells for %d questions" % (m, name, len(st), len(ids)))
        arm["C" + name] = {q: (VAL[x], bool(a)) for q, x, a in zip(ids, st, an)}
    n1 = None
    if idx == 1:
        tune = MP.prompt_split(display, layer, 0, s)[0]
        st, an, _ = S.system(sp, tx, ML.n1_pick(tune[0], tune[1], s))
        n1 = {q: (VAL[x], bool(a)) for q, x, a in zip(ids, st, an)}
    return list(ids), arm, n1


def fqa_split(m, rows, which):
    keep = {r["id"] for r in rows if r["set"] == which}
    R, _ = F.load_rows(m, keep)
    st = F.setting(m)
    s = st[1]
    ids = list(R.ids)

    def cell(a):
        return {q: (R.cell(a, q), not R.blank(a, q)) for q in ids}
    arm = {"O": cell("x"), "E": cell(R.e(s))}
    for k in KS:
        arm["EC" + k] = cell(R.ec(s, k))
        arm["C" + k] = cell(R.c(k))
    n1 = None
    if which == "heldout":
        dev = {r["id"] for r in rows if r["set"] == "dev"}
        tr, tr2 = F.load_rows(m, dev)
        caps, _ = F.choose(tr, tr2, st)
        c = F.n1_cells(R, s, caps["N1"], ids)
        n1 = {q: (c[q][0], bool(c[q][1])) for q in ids}
    return ids, arm, n1


def sc(c):
    """handed in only: (score, handed in)"""
    return (c[0], True) if c[1] else (0, False)


NO = (0, False)


def schemes(ids, arm):
    """{name: ({q: (score, handed in)}, generations)}"""
    out = {}
    O, E = arm["O"], arm["E"]
    EC = {k: arm["EC" + k] for k in KS}
    n = len(ids)
    eb = [q for q in ids if not E[q][1]]
    bb = [q for q in eb if not O[q][1]]
    out["O"] = ({q: sc(O[q]) for q in ids}, n)
    out["E"] = ({q: sc(E[q]) for q in ids}, n)
    for k in KS:
        out["C" + k] = ({q: sc(arm["C" + k][q]) for q in ids}, n)
        out["A" + k] = ({q: sc(EC[k][q]) for q in ids}, n)
        out["R1_" + k] = ({q: (sc(E[q]) if E[q][1] else (NO if O[q][1] else sc(EC[k][q]))) for q in ids}, n + len(eb) + len(bb))
        out["D_" + k] = ({q: (sc(E[q]) if E[q][1] else sc(EC[k][q])) for q in ids}, n + len(eb))

    def ladder(q, cap, lad=KS):
        tried = 0
        for k in lad[:lad.index(cap) + 1]:
            tried += 1
            if EC[k][q][1]:
                return sc(EC[k][q]), tried
        return NO, tried
    for cap in KS:
        d, g = {}, n + len(eb)
        for q in ids:
            if E[q][1]:
                d[q] = sc(E[q])
            elif O[q][1]:
                d[q] = NO
            else:
                d[q], t = ladder(q, cap)
                g += t
        out["R1L_" + cap] = (d, g)
        d, g = {}, n
        for q in ids:
            if E[q][1]:
                d[q] = sc(E[q])
            else:
                d[q], t = ladder(q, cap)
                g += t
        out["DL_" + cap] = (d, g)
    # route 2 as written: a = EC(0.25), b = EC(0.5)
    d, g = {}, n
    for q in ids:
        a, b = EC["0.25"][q], EC["0.5"][q]
        if b[1]:
            d[q] = sc(b)
        else:
            g += 1                      # the 0.25 pass is only needed when the 0.5 pass is blank
            if a[1]:
                d[q] = NO
            else:
                g += 1
                if EC["1"][q][1]:
                    d[q] = sc(EC["1"][q])
                else:
                    g += 1
                    d[q] = sc(EC["2"][q]) if EC["2"][q][1] else NO
    out["R2u"] = (d, g)
    d, g = {}, 0
    for q in ids:
        d[q], t = ladder(q, "2")
        g += t
    out["R2L"] = (d, g)
    # upper bound of route 1 over the C strength
    d = {}
    for q in ids:
        if E[q][1]:
            d[q] = sc(E[q])
        elif O[q][1]:
            d[q] = NO
        else:
            cs = [sc(EC[k][q]) for k in KS]
            if any(v == 1 and h for v, h in cs):
                d[q] = (1, True)
            elif any(not h for v, h in cs):
                d[q] = NO
            else:
                d[q] = max(cs)
    out["R1orc"] = (d, None)
    return out


def net(d, ids):
    return sum(d[q][0] for q in ids)


def best(names, tune, tids):
    return max(names, key=lambda x: (net(tune[x][0], tids), -names.index(x)))


def main():
    ok, rows = F.check_bank()
    if not ok:
        raise SystemExit("FalseQA bank check FAILED")
    pair = {r["id"]: r["pair_id"] for r in rows}
    dump = {}
    for task in ("TruthfulQA", "FalseQA"):
        for m in ORDER:
            if task == "TruthfulQA":
                tids, tarm, _ = tqa_split(m, 0)
                ids, arm, n1 = tqa_split(m, 1)

                def ci(a, b):
                    return [int(x) for x in GS.tq_ci([a[q][0] - b[q][0] for q in ids])]
            else:
                tids, tarm, _ = fqa_split(m, rows, "dev")
                ids, arm, n1 = fqa_split(m, rows, "heldout")

                def ci(a, b):
                    return [int(x) for x in F.pair_ci({q: a[q][0] - b[q][0] for q in ids}, pair, ids)]
            T, X = schemes(tids, tarm), schemes(ids, arm)
            X["N1"] = ({q: sc(n1[q]) for q in ids}, None)
            ck = best(["C" + k for k in KS], T, tids)
            tuned = {"At": best(["A" + k for k in KS], T, tids), "R1_t": best(["R1_" + k for k in KS], T, tids),
                     "D_t": best(["D_" + k for k in KS], T, tids), "R1L_t": best(["R1L_" + k for k in KS], T, tids),
                     "DL_t": best(["DL_" + k for k in KS], T, tids)}
            for name, src in tuned.items():
                X[name] = X[src]
            e, c, n1d = X["E"][0], X[ck][0], X["N1"][0]
            got = ((sum(X["R1_1"][0][q][1] for q in ids), net(X["R1_1"][0], ids)), (sum(n1d[q][1] for q in ids), net(n1d, ids)))
            rec = {"n": len(ids), "c_single": ck[1:], "tuned": {k: v.split("_")[-1].lstrip("A") for k, v in tuned.items()},
                   "gpt_check": {"mine": got, "gpt": GPT[(task, m)], "same": got == GPT[(task, m)]}, "schemes": {}}
            print("\n== %s %s: n %d | single C strength chosen on tuning %s | tuned %s | cross-check with GPT's table: mine %s, GPT %s -> %s" % (
                task, m, len(ids), ck[1:], rec["tuned"], got, GPT[(task, m)], "SAME" if got == GPT[(task, m)] else "DIFFERENT"))
            names = (["O", "E", ck] + ["A" + k for k in KS] + ["At"] + ["R1_" + k for k in KS] + ["R1_t"] + ["R1L_" + k for k in KS] + ["R1L_t"]
                     + ["D_" + k for k in KS] + ["D_t"] + ["DL_" + k for k in KS] + ["DL_t", "R2u", "R2L", "N1", "R1orc"])
            for name in names:
                d, g = X[name]
                h = sum(d[q][1] for q in ids)
                r = {"net": net(d, ids), "handed": h, "right": sum(d[q][0] == 1 for q in ids), "wrong": sum(d[q][0] == -1 for q in ids),
                     "dE": ci(d, e), "dC": ci(d, c), "dN1": ci(d, n1d), "gens": g}
                rec["schemes"]["Cdev" if name == ck else name] = r
                print("   %-7s net %+4d handed in %3d (right %3d wrong %3d) | - E %+4d [%+d, %+d] | - C %+4d [%+d, %+d] | - N1 %+4d [%+d, %+d] | generations %s" % (
                    name, r["net"], h, r["right"], r["wrong"], *r["dE"], *r["dC"], *r["dN1"], g))
            # the 2x2 of provenance and what each arm scores inside each cell
            O, E = arm["O"], arm["E"]
            cells = {}
            for oa in (True, False):
                for ea in (True, False):
                    qq = [q for q in ids if O[q][1] == oa and E[q][1] == ea]
                    key = ("O answered" if oa else "O blank") + " / " + ("E answered" if ea else "E blank")
                    f = lambda a: (sum(sc(a[q])[0] for q in qq), sum(a[q][1] for q in qq), sum(a[q][1] and a[q][0] == 1 for q in qq),  # noqa: E731
                                   sum(a[q][1] and a[q][0] == -1 for q in qq))
                    cells[key] = {"n": len(qq), "O": f(O), "E": f(E), "C": f(arm[ck]), "EC1": f(arm["EC1"]),
                                  "EC_by_k": {k: f(arm["EC" + k]) for k in KS}}
                    print("   cell %-24s n %3d | (net, handed in, right, wrong): O %s | E %s | C %s | EC(1) %s" % (
                        key, len(qq), cells[key]["O"], cells[key]["E"], cells[key]["C"], cells[key]["EC1"]))
            rec["cells"] = cells
            # intervals of the differences inside each cell (zeros outside the cell; the same resampling as above)
            zero = {q: (0, True) for q in ids}

            def cid(a, b, qq):
                s = set(qq)
                return ci({q: (((sc(a[q])[0] - sc(b[q])[0]) if q in s else 0), True) for q in ids}, zero)
            cci = {}
            for oa in (True, False):
                for ea in (True, False):
                    qq = [q for q in ids if O[q][1] == oa and E[q][1] == ea]
                    key = ("O answered" if oa else "O blank") + " / " + ("E answered" if ea else "E blank")
                    cci[key] = {"E-O": cid(E, O, qq), "EC1-E": cid(arm["EC1"], E, qq), "EC1-C": cid(arm["EC1"], arm[ck], qq),
                                "C-O": cid(arm[ck], O, qq), "EC1-O": cid(arm["EC1"], O, qq)}
                    print("   cell %-24s differences inside the cell: E - O %s | EC(1) - E %s | EC(1) - C %s | C - O %s" % (
                        key, cci[key]["E-O"], cci[key]["EC1-E"], cci[key]["EC1-C"], cci[key]["C-O"]))
            rec["cell_ci"] = cci
            # scheme against scheme
            ex = {}
            for a, b in (("R1L_2", "R1_2"), ("R1L_2", "R1_1"), ("R1L_1", "R1_1"), ("R1_1", "D_1"), ("R1L_2", "DL_2"), ("R1L_2", "R2L"),
                         ("R1L_2", "R2u"), ("R1_1", "A1"), ("R1L_t", "R1_1"), ("R1L_t", "R1L_2")):
                ex[a + "-" + b] = ci(X[a][0], X[b][0])
            rec["extra"] = ex
            if task == "FalseQA":       # the two halves of every pair: premise false / premise valid
                prem = {r["id"]: r["premise"] for r in rows}
                pr = {}
                for g in ("false", "valid"):
                    qq = [q for q in ids if prem[q] == g]
                    pr[g] = {"n": len(qq)}
                    for name in ("O", "E", ck, "R1_1", "R1L_t", "R1L_2", "R2u", "N1"):
                        pr[g]["Cdev" if name == ck else name] = net(X[name][0], qq)
                rec["premise"] = pr
                print("   by premise (nets): " + " | ".join("%s: %s" % (g, {k: v for k, v in pr[g].items()}) for g in ("false", "valid")))
            print("   scheme minus scheme: " + " | ".join("%s %s" % (k, v) for k, v in ex.items()))
            # the same schemes on the tuning questions (the single-C strength is chosen there, which favours single C)
            if task == "TruthfulQA":
                def cit(a, b):
                    return [int(x) for x in GS.tq_ci([a[q][0] - b[q][0] for q in tids])]
            else:
                def cit(a, b):
                    return [int(x) for x in F.pair_ci({q: a[q][0] - b[q][0] for q in tids}, pair, tids)]
            tu = {"n": len(tids)}
            for name in ("R1_1", "R1L_2", "R1L_1", "R2u", "R2L", "D_1", "A1"):
                tu[name] = {"net": net(T[name][0], tids), "dE": cit(T[name][0], T["E"][0]), "dC": cit(T[name][0], T[ck][0])}
            tu["E"] = net(T["E"][0], tids)
            tu["Cdev"] = net(T[ck][0], tids)
            rec["tune"] = tu
            print("   tuning questions (n %d): E %+d, C %+d | " % (len(tids), tu["E"], tu["Cdev"]) + " | ".join(
                "%s %+d: - E %s, - C %s" % (k, tu[k]["net"], tu[k]["dE"], tu[k]["dC"]) for k in ("R1_1", "R1L_2", "R2u", "D_1")))
            # monotonicity of answering in the C strength (route 2's assumption)
            mono = {}
            for i in range(len(KS) - 1):
                a, b = arm["EC" + KS[i]], arm["EC" + KS[i + 1]]
                mono["%s->%s" % (KS[i], KS[i + 1])] = (sum(a[q][1] and not b[q][1] for q in ids), sum((not a[q][1]) and b[q][1] for q in ids))
            rec["mono"] = mono
            print("   answering along the C strength (answered at the smaller and blank at the larger, blank at the smaller and answered at the larger): %s" % mono)
            dump[task + "|" + m] = rec
    print("JSON " + json.dumps(dump, sort_keys=True))


if __name__ == "__main__":
    main()
