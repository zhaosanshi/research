#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-23), work queue discussion/work_queue/2026-11-23_queue_l.md,
# sections 1-4: "fill only". Read-only: no model, no generation, no judge call, nothing chosen here.
# S0 = plain prompt + honest sentence, no write (x_E2). B = the questions S0 leaves blank by the rule-based flag (never by
# the verdict). Every fill system hands in S0's cell outside B; inside B the cell of one other pass (that pass's cell
# also when it is blank, as plain_honesty_fill.py's F1 / T2):
#   fill EC = SN1 (honest sentence + E, its blanks re-asked with C; queue I choice, n1s_choice_<model>.json)
#   fill E  = SE (honest sentence + E)
#   fill C  = SC (queue I 7.2; S0 first, blanks re-asked with C alone up to the tune cap; = n1_on_sentence2.sc_system)
#   fill prompt = conf_E2 (must-answer + sentence, no write) = the existing T2
#   fill N1 = the no-sentence main-table N1 = the existing F1
# Sources (all imported unchanged): TruthfulQA n1_on_sentence.build / restricted, n1_on_sentence2.c_cells / sc_system,
# selfjudge_two_answers.native (S0, conf_E2, N1 as plain_honesty_fill.py); (QA)^2 the construction of
# selfjudge_stack_pick.qaqa (n1_arms, caps chosen on the 100 pilot questions), plus SE = the arm x_E2:E<L>.w.p<s_sel> and
# conf_E2 of the no-sentence bank when judged; fact sets selfjudge_stack_pick.fact (SE only, alias match).
# Self-judge margins: selfjudge_stack_pick.lookup (batch size 1 only: tj1_s0 / tj1_conf / tj1_n1 / st1_items / sp_items,
# and fo_items when it exists), condition baseline (no write, five models), keyed by the statement sha1.
# Equal fill counts: per model, the fill system with more answered fills in B drops its fills from the lowest margin up
# (ties by question order; selfjudge_stack_pick.equal on the B questions only).
# Net = right - wrong with the verdict value of a blank cell kept (main-table convention); answers = answered cells.
# Intervals: TruthfulQA policy_select2.ci; (QA)^2 and fact sets newsets_score.joint_ci on the per-question sums.
# usage (scoring container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/fill_only.py check|missing|tqa|qaqa|fact|sec4
import json
import os
import sys
from pathlib import Path

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import selfjudge_probe as SP          # noqa: E402
import selfjudge_two_answers as TA    # noqa: E402
import selfjudge_stack_pick as SKP    # noqa: E402
import n1_on_sentence as N            # noqa: E402
import n1_on_sentence2 as N2          # noqa: E402

A, MP, ML, S, ps = N.A, N.MP, N.ML, N.S, N.ps
V, ORDER = N.V, N.ORDER
LEVELS = ("0", "0.25", "0.5", "1", "2")
SRCOF = {"fill EC": "SN1", "fill E": "SE", "fill C": "SCf", "fill prompt": "conf", "fill N1": "N1"}
FO_SRC = ("fo_items", "fo_chk", "fo_chka", "fo_chkb")
SP.SOURCES.update(FO_SRC)

# ---- section 3 references -------------------------------------------------------------------------------------- #
REF5 = {"test": {"S0": (706, 2269), "SN1": (947, 2438), "SE": (894, 2126), "fill C": (759, 2552),
                 "fill prompt": (689, 2598), "fill N1": (781, 2518)},
        "tune": {"S0": (244, 804), "fill prompt": (268, 958), "fill N1": (301, 927)}}
# per model, results/write_positions/plain_honesty_fill_tqa_five_models_20261122.txt (F1, T2)
REF_F1 = {"test": {"llama3.1-8b": (55, 515), "qwen3.5-9b": (172, 481), "qwen3.8-27b": (201, 498), "gemma3-12b": (163, 518),
                   "gemma3-27b": (188, 518)},
          "tune": {"llama3.1-8b": (32, 192), "qwen3.5-9b": (66, 170), "qwen3.8-27b": (93, 188), "gemma3-12b": (42, 187),
                   "gemma3-27b": (68, 190)}}
REF_T2 = {"test": {"llama3.1-8b": (40, 523), "qwen3.5-9b": (159, 523), "qwen3.8-27b": (165, 521), "gemma3-12b": (145, 521),
                   "gemma3-27b": (178, 522)},
          "tune": {"llama3.1-8b": (30, 192), "qwen3.5-9b": (58, 190), "qwen3.8-27b": (76, 192), "gemma3-12b": (32, 192),
                   "gemma3-27b": (72, 192)}}
REF_NQ = {"test": {"llama3.1-8b": 523, "qwen3.5-9b": 523, "qwen3.8-27b": 521, "gemma3-12b": 522, "gemma3-27b": 523},
          "tune": {m: 192 for m in ORDER}}
# results/write_positions/n1_on_sentence_single_c_five_models_20261122.txt (7.2): SC per model on test, tune caps
REF_SC = {"llama3.1-8b": (52, 520), "qwen3.5-9b": (167, 492), "qwen3.8-27b": (185, 513), "gemma3-12b": (161, 521),
          "gemma3-27b": (194, 518)}
REF_SCCAP = {"llama3.1-8b": "2", "qwen3.5-9b": "0-25", "qwen3.8-27b": "1", "gemma3-12b": "1", "gemma3-27b": "0-5"}
REF_TRANS = {("wrong", "right"): 178, ("wrong", "none"): 81, ("right", "wrong"): 134, ("right", "none"): 60,
             ("none", "right"): 206, ("none", "wrong"): 74}
REF_QQ = {"S0": (40, 2013), "SN1": (294, 2139), "SE": (287, 1886)}


def fmt(t):
    return "%+d (%d)" % t


def cls(v):
    return {1: "right", -1: "wrong", 0: "none"}[v]


# ---- TruthfulQA ------------------------------------------------------------------------------------------------ #
def lvl_of(sp, s, r, a):
    """C level of a handed-in record of S.system: "0" = E's own answer, else the released rung."""
    if not a:
        return None
    if r["cond"] == sp.cE(s):
        return "0"
    if r["note"] == "released":
        return ps.NAME[r["rung"]]
    return None


def sc_cap(model):
    """n1_on_sentence2.singlec's tune choice of the single-C cap (recomputed, compared with the 7.2 printout)."""
    ch = N2.choice(model)
    ssel = ch["strength"]
    tsp, ttx = N.build(model, 0, "x_E2", ssel, N.restricted(model, 0))
    t0 = N.run_sys(tsp, ttx, ("base",))
    tc = N2.c_cells(model, 0, tsp.qs)
    tq = [q for q in tsp.qs if all(q in tc[k] for k in ps.LAD)]
    nets = [(sum(N2.sc_system(t0, tc, k, tq)[q][0] for q in tq), k) for k in ps.LAD]
    return max(nets, key=lambda x: (x[0], -ps.LAD.index(x[1])))[1]


def c_texts(model, i):
    display, layer, _ = A.KEY[model]
    D = MP.MT.Data(display, i)
    split = ("tune", "test")[i]
    run = D.m["ec"][i].split("/eval_extra/")[0]
    cl = D.m["c"]
    out = {}
    for w, ks in ((MP.work(run, split, "h22c4"), ("0-25",)), (MP.work(run, split, "n1sc"), ("0-5", "1", "2"))):
        t = MP.fr.raw_texts(w)
        for k in ks:
            c = "C_L%d_x_E2-w-m%s" % (cl, k)
            for (q, cc), x in t.items():
                if cc == c:
                    out[(q, k)] = x
    return out


def tq(model, i, cap):
    """(qs, {q: cells}) on TruthfulQA split i; a cell = (value, answered, text, C level)."""
    name = ("tune", "test")[i]
    display, layer, s_main = A.KEY[model]
    ch = N2.choice(model)
    sc, cfg = ch["strength"], tuple(ch["cfg"])
    cq = N.restricted(model, i)
    psp, ptx = N.build(model, i, "x_E2", sc, cq)
    rec = {}
    for nm, c in (("S0", ("base",)), ("SE", ("E", sc)), ("SN1", cfg)):
        st, an, rc = S.system(psp, ptx, c)
        for q, x, a, r in zip(psp.qs, st, an, rc):
            if r["p"] != q:
                raise SystemExit("%s: record order" % model)
            lv = lvl_of(psp, sc, r, a)
            if a and nm != "S0" and lv is None:
                raise SystemExit("%s %s: answered %s cell without a C level (%s)" % (model, q, nm, r["note"]))
            rec.setdefault(q, {})[nm] = (V[x], bool(a), r["text"], lv)
    cs = N2.c_cells(model, i, psp.qs)
    ctx = c_texts(model, i)
    D = MP.MT.Data(display, i)
    st1, bl1 = ps.load(D.m["d"][i], "pro")
    t1 = MP.fr.raw_texts(D.m["d"][i])
    NT = TA.native(model, i)
    nid = {q: k for k, q in enumerate(NT["ids"])}
    tsp, ttx = N.build(model, 0, "x", s_main, N.restricted(model, 0))
    xsp, xtx = (tsp, ttx) if i == 0 else N.build(model, i, "x", s_main, cq)
    n1cfg = ML.n1_pick(tsp, ttx, s_main)
    if list(n1cfg) != list(NT["n1cfg"]):
        raise SystemExit("%s: N1 setting %s differs from native's %s" % (model, n1cfg, NT["n1cfg"]))
    stn, ann, rcn = S.system(xsp, xtx, n1cfg)
    xix = {q: k for k, q in enumerate(xsp.qs)}
    step1 = SP.read_json(os.path.join(SP.OUT, model, "items.json"))
    qtext = {(x["split"], x["id"]): x["question"] for x in step1["items"]}
    qs = [q for q in psp.qs if q in xix and q in nid and st1.get((q, "baseline_conf_E2"), "?") != "?"
          and NT["st"].get((q, "baseline_x_E2"), "?") != "?" and all(q in cs[k] for k in ps.LAD)]
    lad = ps.LAD[:ps.LAD.index(cap) + 1]
    s0v = {q: rec[q]["S0"][:2] for q in qs}
    scref = N2.sc_system(s0v, cs, cap, qs)
    out = {}
    for q in qs:
        d = dict(rec[q])
        k, j = nid[q], xix[q]
        s0n = (V[NT["st"][(q, "baseline_x_E2")]], not NT["bl"][(q, "baseline_x_E2")])
        if s0n != d["S0"][:2]:
            raise SystemExit("%s %s: S0 of native and of build differ" % (model, q))
        nv, na, nr = V[NT["n1"][0][k]], bool(NT["n1"][1][k]), NT["n1"][2][k]
        if (V[stn[j]], bool(ann[j]), rcn[j]["cond"], rcn[j]["text"]) != (nv, na, nr["cond"], nr["text"]):
            raise SystemExit("%s %s: N1 of native and of build differ" % (model, q))
        d["N1"] = (nv, na, nr["text"], lvl_of(xsp, s_main, rcn[j], ann[j]))
        if na and d["N1"][3] is None:
            raise SystemExit("%s %s: N1 answer without a C level" % (model, q))
        d["obl"] = bool(xsp.blank[(q, "baseline_x")])
        d["conf"] = (V[st1[(q, "baseline_conf_E2")]], not bl1[(q, "baseline_conf_E2")], t1[(q, "baseline_conf_E2")], None)
        if d["S0"][1]:
            d["SCf"] = d["S0"]
        else:
            kk = next((x for x in lad if cs[x][q][1]), None)
            d["SCf"] = d["S0"] if kk is None else (cs[kk][q][0], True, ctx[(q, kk)], ps.NAME[kk])
        if d["SCf"][:2] != scref[q]:
            raise SystemExit("%s %s: fill C differs from n1_on_sentence2.sc_system" % (model, q))
        d["qt"] = qtext[(name, q)]
        out[q] = d
    return qs, out


# ---- (QA)^2 ---------------------------------------------------------------------------------------------------- #
def qq(model):
    """(ids, {q: cells}, {q: (qtype, pilot)}, has prompt fill); selfjudge_stack_pick.qaqa's construction."""
    import qaqa_pilot_table as Q
    NSD = Q.S.NS
    bankp = os.path.join(NSD, "bank/qaqa_full_bank_20261122.jsonl")
    with open(bankp, encoding="utf-8") as f:
        bank = {r["id"]: r for r in (json.loads(l) for l in f)}
    pilot = set(json.load(open(os.path.join(NSD, "splits/qaqa_pilot_20261122.json")))["ids"])
    fx = json.load(open(os.path.join(NSD, "splits/qaqa_tune_choice_fixedE_20261122.json")))[model]
    el, _, cl = Q.SET[model][:3]
    ssel = N2.choice(model)["strength_dot"]
    base = Q.Raw(model, "qaqafull", bankp, ("test_full570", "test_full570_tuned", "test_full570_strong"))
    R = Q.Raw(model, "qaqafull", bankp, ["test_full570_h22n", "test_full570_h22n1", "test_full570_h22z"])
    R.gdir = Path(NSD) / "gen_prompt" / "h22" / "qaqafull" / model / "test"
    arms = ["x_E2", "x_E2:E%d.w.p%s" % (el, ssel)] + ["x_E2:E%d.w.p%s+C%d.w.m%s" % (el, ssel, cl, k) for k in N2.LADK]
    for a in arms:
        R.need(a)
        alias = a.replace("x_E2:", "x:") if ":" in a else "x"
        R.rows[alias], R.cond[alias] = R.rows[a], R.cond[a]
    if R.ids != base.ids:
        raise SystemExit("banks differ")
    pq = [q for q in R.ids if q in pilot]
    grid = [((ce, co), sum(R.n1_system(ssel, ce, co)[0][q] for q in pq)) for ce in Q.CAPS for co in Q.CAPS]
    best = max(grid, key=lambda x: (x[1], -grid.index(x)))[0]
    sarm, sv, sa = SKP.n1_arms(R, ssel, best[0], best[1])
    narm, nv, na = SKP.n1_arms(base, fx["n1_strength"], fx["n1_cap_e"], fx["n1_cap_own"])
    se = R.e(ssel)
    has_conf = base.has("conf_E2")

    def lv(Rr, arm, s, a):
        if not a:
            return None
        return "0" if arm == Rr.e(s) else arm.rsplit(".m", 1)[1]
    per = {}
    for q in R.ids:
        d = {"S0": (R.cell("x", q), not R.blank("x", q), R.text("x", q), None),
             "SE": (R.cell(se, q), not R.blank(se, q), R.text(se, q), "0" if not R.blank(se, q) else None),
             "SN1": (sv[q], bool(sa[q]), R.text(sarm[q], q), lv(R, sarm[q], ssel, sa[q])),
             "N1": (nv[q], bool(na[q]), base.text(narm[q], q), lv(base, narm[q], fx["n1_strength"], na[q])),
             "obl": base.blank("x", q), "qt": bank[q]["question"]}
        if has_conf:
            d["conf"] = (base.cell("conf_E2", q), not base.blank("conf_E2", q), base.text("conf_E2", q), None)
        per[q] = d
    info = {q: (base.qtype[q], q in pilot) for q in R.ids}
    print("# (QA)^2 %s: s_sel %s, SN1 caps %s / %s; no-sentence N1 %s caps %s / %s; must-answer + sentence (conf_E2) judged: %s" % (
        model, ssel, best[0], best[1], fx["n1_strength"], fx["n1_cap_e"], fx["n1_cap_own"], "yes" if has_conf else "no"))
    return list(R.ids), per, info, has_conf


# ---- fact sets ------------------------------------------------------------------------------------------------- #
def fact_cells(F):
    """selfjudge_stack_pick.fact output -> {key: (qs, {q: cells})} with SE = its St."""
    out = {}
    for key, (qs, per, _) in F.items():
        d = {}
        for q in qs:
            c = per[q]
            d[q] = {"S0": (c["S0"][0], c["S0"][1], c["S0"][5], None), "SE": (c["St"][0], c["St"][1], c["St"][5], "0"),
                    "qt": c["S0"][4] or c["St"][4]}
            if c["S0"][1] and c["S0"][4] is None or c["St"][1] and c["St"][4] is None:
                raise SystemExit("fact cell without question")
        out[key] = (list(qs), d)
    return out


# ---- systems and readings -------------------------------------------------------------------------------------- #
def cell(d, name):
    if name in ("S0", "SN1", "SE"):
        return d[name]
    src = SRCOF[name]
    if name == "fill C":
        return d["SCf"]
    return d["S0"] if d["S0"][1] else d[src]


def margin(M, d, c):
    if not c[1]:
        return None
    return M.get(SP.sha1(SP.statement(d["qt"], c[2])), {}).get("baseline")


def ci_tq(v):
    return ps.ci(list(v))


def ci_j(v):
    import newsets_score as NS
    return "%+d [%+d, %+d]" % tuple(NS.joint_ci([list(v)]))


def readings(label, data, models, common, fills, ci, M, levels=True, trans=True):
    """data[m] = (qs, {q: cells}); common = the questions of the five-model rows."""
    names = ["S0", "SN1", "SE"] + list(fills) if "SN1" in data[models[0]][1][common[0]] else ["S0", "SE"] + list(fills)
    rows = [(m, (m,), data[m][0]) for m in models] + [("five models", tuple(models), common)]
    print("\n" + "=" * 110 + "\n%s" % label)
    print("-- reading 1: nets (answers); B = S0 blank by the rule; fills = answered cells inside B: right / wrong / no information")
    for lab, ms, qs in rows:
        tot = {n: (sum(cell(data[m][1][q], n)[0] for m in ms for q in qs), sum(cell(data[m][1][q], n)[1] for m in ms for q in qs))
               for n in names}
        nB = sum(not data[m][1][q]["S0"][1] for m in ms for q in qs)
        nB0 = sum(data[m][1][q]["S0"][0] == 0 for m in ms for q in qs)
        print("  %-12s %4d q | %s" % (lab, len(qs), " | ".join("%s %s" % (n, fmt(tot[n])) for n in names)))
        fl = []
        for f in fills:
            got = [cell(data[m][1][q], f) for m in ms for q in qs if not data[m][1][q]["S0"][1]]
            got = [c for c in got if c[1]]
            fl.append("%s %d: %d / %d / %d" % (f, len(got), sum(c[0] == 1 for c in got), sum(c[0] == -1 for c in got),
                                              sum(c[0] == 0 for c in got)))
        print("               B %d (S0 value 0, i.e. blank or no information: %d) | fills %s" % (nB, nB0, " | ".join(fl)))
    print("-- reading 2: each fill system minus S0 (paired interval)")
    for lab, ms, qs in rows:
        print("  %-12s %s" % (lab, " | ".join("%s - S0 %s" % (f, ci([sum(cell(data[m][1][q], f)[0] - data[m][1][q]["S0"][0] for m in ms)
                                                                        for q in qs])) for f in fills)))
    others = [f for f in fills if f != "fill EC"]
    if "fill EC" in fills and others:
        print("-- reading 3: fill EC minus the other fills; natural fill counts, then equal fill counts (per model the side with "
              "more fills drops its fills from the lowest self-judge margin up, baseline condition, batch size 1)")
        for lab, ms, qs in rows:
            nat, eq = [], []
            for f in others:
                dn, de, cnt, miss = {}, {}, [], 0
                for m in ms:
                    P = data[m][1]
                    bq = [q for q in qs if not P[q]["S0"][1]]
                    Aa = {q: cell(P[q], "fill EC")[:2] + (margin(M[m], P[q], cell(P[q], "fill EC")),) for q in bq}
                    Bb = {q: cell(P[q], f)[:2] + (margin(M[m], P[q], cell(P[q], f)),) for q in bq}
                    na, nb = sum(x[1] for x in Aa.values()), sum(x[1] for x in Bb.values())
                    big = Aa if na > nb else Bb
                    mm = sum(1 for x in big.values() if x[1] and x[2] is None) if na != nb else 0
                    miss += mm
                    cnt.append((na, nb))
                    for q in qs:
                        dn[q] = dn.get(q, 0) + cell(P[q], "fill EC")[0] - cell(P[q], f)[0]
                    if mm:
                        continue
                    va, vb, _ = SKP.equal(bq, Aa, Bb)
                    for q in bq:
                        de[q] = de.get(q, 0) + va[q][0] - vb[q][0]
                nat.append("fill EC - %s %s (fills %d vs %d)" % (f, ci([dn[q] for q in qs]), sum(x[0] for x in cnt), sum(x[1] for x in cnt)))
                if miss:
                    eq.append("fill EC - %s n/a (%d margins missing)" % (f, miss))
                else:
                    eq.append("fill EC - %s %s (at %d)" % (f, ci([de.get(q, 0) for q in qs]), sum(min(x) for x in cnt)))
            print("  %-12s natural: %s" % (lab, " | ".join(nat)))
            print("  %-12s equal:   %s" % ("", " | ".join(eq)))
    if levels:
        print("-- reading 4: fill EC's fills by the C level that released them (0 = E alone): count right / wrong; "
              "and the no-sentence N1's answers on the questions the original model leaves blank")
        for lab, ms, qs in rows:
            for tag, sel, src in (("fill EC on B", lambda P, q: not P[q]["S0"][1], "SN1"),
                                  ("N1 on original-blank", lambda P, q: P[q]["obl"], "N1")):
                if src == "N1" and "fill N1" not in fills:
                    continue
                cs = [data[m][1][q][src] for m in ms for q in qs if sel(data[m][1], q) and data[m][1][q][src][1]]
                print("  %-12s %-21s %s" % (lab, tag, " | ".join("C %s: %d (%d / %d)" % (
                    k, sum(c[3] == k for c in cs), sum(c[3] == k and c[0] == 1 for c in cs), sum(c[3] == k and c[0] == -1 for c in cs))
                    for k in LEVELS)))
    if trans:
        print("-- reading 5: SN1 minus fill EC = SN1's changes on the questions S0 answered (value classes; none = blank or "
              "no information); right->wrong with a negated-premise opening (NEG_RE)")
        for lab, ms, qs in rows:
            c = {}
            negrw = 0
            for m in ms:
                for q in qs:
                    d = data[m][1][q]
                    if not d["S0"][1]:
                        continue
                    k = (cls(d["S0"][0]), cls(d["SN1"][0]))
                    c[k] = c.get(k, 0) + 1
                    if k == ("right", "wrong") and d["SN1"][1] and SP.NEG_RE.match(d["SN1"][2]):
                        negrw += 1
            dd = ci([sum(data[m][1][q]["SN1"][0] - cell(data[m][1][q], "fill EC")[0] for m in ms) for q in qs])
            print("  %-12s SN1 - fill EC %s | wrong->right %d | right->wrong %d (negated opening %d) | wrong->none %d | right->none %d"
                  " | none->right %d | none->wrong %d | unchanged class %d" % (
                      lab, dd, c.get(("wrong", "right"), 0), c.get(("right", "wrong"), 0), negrw, c.get(("wrong", "none"), 0),
                      c.get(("right", "none"), 0), c.get(("none", "right"), 0), c.get(("none", "wrong"), 0),
                      sum(v for k, v in c.items() if k[0] == k[1])))
    sys.stdout.flush()


# ---- modes ----------------------------------------------------------------------------------------------------- #
def load_tq():
    caps = {}
    for m in ORDER:
        caps[m] = sc_cap(m)
        print("# %s: single-C cap on tune %s (7.2: %s) -> %s" % (m, ps.NAME[caps[m]], ps.NAME[REF_SCCAP[m]],
                                                               "OK" if caps[m] == REF_SCCAP[m] else "MISMATCH"))
        if caps[m] != REF_SCCAP[m]:
            raise SystemExit("SC CAP MISMATCH")
    T = {}
    for i, name in ((1, "test"), (0, "tune")):
        T[name] = {m: tq(m, i, caps[m]) for m in ORDER}
    return T


def common_of(data):
    return sorted(set.intersection(*(set(data[m][0]) for m in ORDER)))


def check_tq(T):
    ok = True
    for name in ("test", "tune"):
        data = T[name]
        for m in ORDER:
            qs, P = data[m]
            f1 = (sum(cell(P[q], "fill N1")[0] for q in qs), sum(cell(P[q], "fill N1")[1] for q in qs))
            t2 = (sum(cell(P[q], "fill prompt")[0] for q in qs), sum(cell(P[q], "fill prompt")[1] for q in qs))
            hit = len(qs) == REF_NQ[name][m] and f1 == REF_F1[name][m] and t2 == REF_T2[name][m]
            extra = ""
            if name == "test":
                sc = (sum(cell(P[q], "fill C")[0] for q in qs), sum(cell(P[q], "fill C")[1] for q in qs))
                sn = (sum(P[q]["SN1"][0] for q in qs), sum(P[q]["SN1"][1] for q in qs))
                hit &= sc == REF_SC[m] and sn == SKP.SK.REF_SN1[m]
                extra = " | fill C %s [ref %s] | SN1 %s [ref %s]" % (fmt(sc), fmt(REF_SC[m]), fmt(sn), fmt(SKP.SK.REF_SN1[m]))
            print("check %s %-12s %d q [ref %d]: fill N1 %s [F1 ref %s] | fill prompt %s [T2 ref %s]%s -> %s" % (
                name, m, len(qs), REF_NQ[name][m], fmt(f1), fmt(REF_F1[name][m]), fmt(t2), fmt(REF_T2[name][m]), extra,
                "OK" if hit else "MISMATCH"))
            ok &= hit
        common = common_of(data)
        for k, ref in REF5[name].items():
            got = (sum(cell(data[m][1][q], k)[0] for m in ORDER for q in common), sum(cell(data[m][1][q], k)[1] for m in ORDER for q in common))
            print("check %s five models %d common q: %s %s [ref %s] -> %s" % (name, len(common), k, fmt(got), fmt(ref),
                                                                             "OK" if got == ref else "MISMATCH"))
            ok &= got == ref
        if name == "test":
            ok &= len(common) == 520
            c = {}
            for m in ORDER:
                for q in common:
                    d = data[m][1][q]
                    k = (cls(d["S0"][0]), cls(d["SN1"][0]))
                    c[k] = c.get(k, 0) + 1
            for k, ref in REF_TRANS.items():
                print("check test S0 -> SN1 %s->%s %d [ref %d] -> %s" % (k[0], k[1], c.get(k, 0), ref, "OK" if c.get(k, 0) == ref else "MISMATCH"))
                ok &= c.get(k, 0) == ref
            nb = sum(not data[m][1][q]["S0"][1] for m in ORDER for q in common)
            n0 = sum(data[m][1][q]["S0"][0] == 0 for m in ORDER for q in common)
            print("note: test five models, S0 blank by the rule (B) %d; S0 value class none (blank or no information) %d" % (nb, n0))
    return ok


def check_qq(Q):
    ok = True
    held = [q for q in Q[ORDER[0]][0] if not Q[ORDER[0]][2][q][1]]
    for k, ref in REF_QQ.items():
        got = (sum(Q[m][1][q][k][0] for m in ORDER for q in held), sum(Q[m][1][q][k][1] for m in ORDER for q in held))
        print("check (QA)^2 %d held out, five models, %s %s [ref %s] -> %s" % (len(held), k, fmt(got), fmt(ref), "OK" if got == ref else "MISMATCH"))
        ok &= got == ref
    return ok


def load_qq():
    return {m: qq(m) for m in ORDER}


def margins():
    return {m: SKP.lookup(m) for m in ORDER}


def run_check():
    T = load_tq()
    Q = load_qq()
    ok = check_tq(T) & check_qq(Q)
    print("section 3 checks: %s" % ("ALL OK" if ok else "MISMATCH"))
    if not ok:
        raise SystemExit(1)
    return T, Q


def missing(T=None, Q=None, F=None):
    """Answered fills (and section 4 candidates) without a batch-1 score; writes nothing."""
    M = margins()
    tot = 0
    for m in ORDER:
        line = []
        if T is not None:
            for name in ("test", "tune"):
                qs, P = T[name][m]
                for f in ("fill EC", "fill E", "fill C", "fill prompt", "fill N1"):
                    n = sum(1 for q in qs if not P[q]["S0"][1] and cell(P[q], f)[1] and margin(M[m], P[q], cell(P[q], f)) is None)
                    tot += n
                    line.append("tqa %s %s %d" % (name, f, n))
        if Q is not None:
            ids, P, _, hc = Q[m]
            for f in ("fill EC", "fill E", "fill N1") + (("fill prompt",) if hc else ()):
                n = sum(1 for q in ids if not P[q]["S0"][1] and cell(P[q], f)[1] and margin(M[m], P[q], cell(P[q], f)) is None)
                tot += n
                line.append("qaqa %s %d" % (f, n))
        if F is not None:
            for key, (qs, P) in F[m].items():
                n = sum(1 for q in qs if not P[q]["S0"][1] and P[q]["SE"][1] and margin(M[m], P[q], P[q]["SE"]) is None)
                tot += n
                line.append("%s fill E %d" % (key, n))
        print("missing %s: %s" % (m, " | ".join(line)))
    print("missing total %d" % tot)
    return M


def run_tqa():
    T, Q = run_check()
    M = margins()
    fills = ("fill EC", "fill E", "fill C", "fill prompt", "fill N1")
    for name in ("test", "tune"):
        data = T[name]
        readings("TruthfulQA %s (DeepSeek Pro)" % name, data, ORDER, common_of(data), fills, ci_tq, M)


def run_qaqa():
    Q = load_qq()
    if not check_qq(Q):
        raise SystemExit("(QA)^2 CHECK MISMATCH")
    M = margins()
    import qaqa_pilot_table as QP
    hc = all(Q[m][3] for m in ORDER)
    fills = ("fill EC", "fill E", "fill N1") + (("fill prompt",) if hc else ())
    if not hc:
        print("# fill prompt: no judged must-answer + sentence pass on (QA)^2 for %s -> none" % ", ".join(m for m in ORDER if not Q[m][3]))
    info = Q[ORDER[0]][2]
    groups = [("470 held out, all types", lambda q: not info[q][1])]
    groups += [("470 held out, %s" % tl, (lambda t: lambda q: not info[q][1] and info[q][0] == t)(t)) for t, tl in QP.TYPES]
    groups += [("100 pilot (tune), all types", lambda q: info[q][1])]
    for lab, sel in groups:
        qs = [q for q in Q[ORDER[0]][0] if sel(q)]
        data = {m: (qs, Q[m][1]) for m in ORDER}
        readings("(QA)^2 %s (DeepSeek Pro)" % lab, data, ORDER, qs, fills, ci_j, M)


def run_fact():
    M = margins()
    F = {m: fact_cells(SKP.fact(m)) for m in ORDER}
    for key in F[ORDER[0]]:
        data = {m: F[m][key] for m in ORDER}
        common = sorted(set.intersection(*(set(data[m][0]) for m in ORDER)))
        readings("%s (alias match)" % key, data, ORDER, common, ("fill E",), ci_j, M, levels=False, trans=False)
        rows = [(m, (m,), data[m][0]) for m in ORDER] + [("five models", ORDER, common)]
        print("-- SE minus fill E (SE's changes on the questions S0 answered)")
        for lab, ms, qs in rows:
            print("  %-12s SE - fill E %s" % (lab, ci_j([sum(data[m][1][q]["SE"][0] - cell(data[m][1][q], "fill E")[0] for m in ms) for q in qs])))


def run_sec4():
    """Queue L section 4: single E - S0 and N1 - S0 (no sentence vs the sentence), (QA)^2 470 by type and the fact sets;
    natural and equal answer counts, then both sides through rule Z (selfjudge_stack_pick.systems, baseline margins)."""
    import qaqa_pilot_table as QP
    M = margins()
    for which in ("qaqa", "fact"):
        D = {m: (SKP.qaqa(m) if which == "qaqa" else SKP.fact(m)) for m in ORDER}
        for m in ORDER:
            miss = sum(1 for key in D[m] for q in D[m][key][0] for c in D[m][key][1][q].values()
                       if c[1] and SP.sha1(c[2]) not in M[m])
            if miss:
                raise SystemExit("%s: %d answered candidates without a batch-1 score" % (m, miss))
        if which == "qaqa":
            info = D[ORDER[0]]["qaqa"][2]
            held = [q for q in D[ORDER[0]]["qaqa"][0] if not info[q][1]]
            for k, ref in SKP.REF_QQ470.items():
                got = (sum(D[m]["qaqa"][1][q][k][0] for m in ORDER for q in held), sum(D[m]["qaqa"][1][q][k][1] for m in ORDER for q in held))
                print("check sec4 (QA)^2 470 %s %s [ref %s] -> %s" % (k, fmt(got), fmt(ref), "OK" if got == ref else "MISMATCH"))
                if got != ref:
                    raise SystemExit("SEC4 CHECK MISMATCH")
            groups = [("(QA)^2 470 held out, all types", "qaqa", lambda q: not info[q][1])]
            groups += [("(QA)^2 470 held out, %s" % tl, "qaqa", (lambda t: lambda q: not info[q][1] and info[q][0] == t)(t)) for t, tl in QP.TYPES]
            comps = (("E", "S0"), ("N1", "S0"), ("Ez", "S0z"), ("N1z", "S0z"))
        else:
            groups = [("%s (alias match)" % key, key, None) for key in D[ORDER[0]]]
            comps = (("E", "S0"), ("Ez", "S0z"))
        for lab, key, sel in groups:
            print("\n" + "=" * 110 + "\nsection 4, %s: no-sentence single E / N1 vs the sentence S0 (rule Z: margin < 0 -> no answer, "
                  "negated openings exempt)" % lab)
            Sy = {}
            for m in ORDER:
                qs = [q for q in D[m][key][0] if sel is None or sel(q)]
                Sy[m] = (qs, {q: SKP.systems(D[m][key][1][q], "baseline", M[m]) for q in qs})
            common = sorted(set.intersection(*(set(Sy[m][0]) for m in ORDER)))
            rows = [(m, (m,), Sy[m][0]) for m in ORDER] + [("five models", ORDER, common)]
            names = sorted({a for c in comps for a in c}, key=lambda x: ("E", "N1", "S0", "Ez", "N1z", "S0z").index(x))
            for lab2, ms, qs in rows:
                print("  %-12s %4d q | %s" % (lab2, len(qs), " | ".join("%s %+d (%d)" % (
                    n, sum(Sy[m][1][q][n][0] for m in ms for q in qs), sum(Sy[m][1][q][n][1] for m in ms for q in qs)) for n in names)))
                nat, eq = [], []
                for a, b in comps:
                    dn, de, cnt = {}, {}, []
                    for m in ms:
                        Aa = {q: Sy[m][1][q][a] for q in qs}
                        Bb = {q: Sy[m][1][q][b] for q in qs}
                        va, vb, nn = SKP.equal(qs, Aa, Bb)
                        cnt.append(nn)
                        for q in qs:
                            dn[q] = dn.get(q, 0) + Aa[q][0] - Bb[q][0]
                            de[q] = de.get(q, 0) + va[q][0] - vb[q][0]
                    nat.append("%s - %s %s (answers %d vs %d)" % (a, b, ci_j([dn[q] for q in qs]), sum(x[0] for x in cnt), sum(x[1] for x in cnt)))
                    eq.append("%s - %s %s (at %d)" % (a, b, ci_j([de[q] for q in qs]), sum(min(x) for x in cnt)))
                print("               natural: %s" % " | ".join(nat))
                print("               equal:   %s" % " | ".join(eq))
            sys.stdout.flush()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "check":
        run_check()
    elif mode == "missing":
        T, Q = run_check()
        F = {m: fact_cells(SKP.fact(m)) for m in ORDER}
        missing(T, Q, F)
    elif mode == "tqa":
        run_tqa()
    elif mode == "qaqa":
        run_qaqa()
    elif mode == "fact":
        run_fact()
    elif mode == "sec4":
        run_sec4()
    else:
        raise SystemExit("usage: fill_only.py check|missing|tqa|qaqa|fact|sec4")


if __name__ == "__main__":
    main()
