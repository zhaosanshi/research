# Iteration_01 analysis (Claude Opus 5.5, 2026-11-23, queue I section 8), read-only, no model, no judge call. Exploration.
# The E strength of the sentence method (P = x_E2) chosen on several tune splits: s_sel (step one) or s_half = s_sel / 2.
#   choose MODEL  per strength: TruthfulQA tune SN1 net (n1_pick's caps) + (QA)^2 100 pilot questions SN1 net (caps over
#                 every pair, ties to the earlier, as 7.1) + TriviaQA tune SE net + PopQA tune SE net (alias match); the
#                 strength with the larger sum, ties to s_half; writes newsets/h22/n1s3_choice_<model>.json (never
#                 overwritten with another choice).
#   fact          the four fact test batches: SE(s_half) - SE(s_sel), SE(s_half) - S0, SE(s_half) - no write, SE(s_half) -
#                 single E (no sentence, main strength); right -> wrong (S0 -> SE) answers opening with a negation
#                 (diag/selfjudge_probe.NEG_RE), both strengths side by side.
#   final         the chosen strength (SN1m / SEm): TruthfulQA test, (QA)^2 470 held-out (by type), fact batches; minus S0,
#                 minus single E (no sentence), minus no write, minus the s_sel method.
# TruthfulQA arms on x_E2 come from the work dirs h22c4 (S0), n1s (step one), n1s3_<split> (s_half where step one has
# none); (QA)^2 from test_full570_h22z (S0), _h22n / _h22n1 (s_sel), _h22n3 (s_half); fact sets from
# newsets/gen_prompt/h22/<task>/<model>/<split>/. Intervals: policy_select2.ci (TruthfulQA), newsets_score.joint_ci (the
# other sets), on the per-question sums of the common questions.
# usage (scoring container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/n1_on_sentence3.py MODE [MODEL]
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import n1_on_sentence as N    # noqa: E402
import n1_on_sentence2 as N2  # noqa: E402
import qaqa_pilot_table as Q  # noqa: E402
import ns_prompt_table as NP  # noqa: E402
import selfjudge_probe as SPB  # noqa: E402

A, MP, ML, S, ps, V, ORDER = N.A, N.MP, N.ML, N.S, N.ps, N.V, N.ORDER
NSD = Q.S.NS
CHOICE3 = "Iteration_01/newsets/h22/n1s3_choice_%s.json"
BANKQ = os.path.join(NSD, "bank/qaqa_full_bank_20261122.jsonl")
PILOT = set(json.load(open(os.path.join(NSD, "splits/qaqa_pilot_20261122.json")))["ids"])
LADK = ("0.25", "0.5", "1", "2")
FACT = (("popqa", "test"), ("popqa", "test2"), ("triviaqa", "test"), ("triviaqa", "test2"))


def strengths(model):
    ss = json.load(open(N.CHOICE % model))["strength_dot"]
    h = float(ss) / 2
    return ss, ("%.4f" % h).rstrip("0").rstrip(".")


def tq_build(model, i, s_dot, qs):
    """N.build with the s_half work dir of this section added (renamed exactly as N.build renames)."""
    s = N.dash(s_dot)
    old = MP.work
    display, layer, _ = A.KEY[model]
    D = MP.MT.Data(display, i)
    split = ("tune", "test")[i]
    run = D.m["ec"][i].split("/eval_extra/")[0]
    w3 = MP.work(run, split, "n1s3")
    cond = "E_L%d_x_E2-w-p%s" % (layer, s)
    in3 = MP.has(w3) and cond in json.load(open(w3 + "/judge_input.json"))["conditions"]
    if not in3:                                      # step one's n1s dirs hold this strength on this split
        return N.build(model, i, "x_E2", s, qs)
    # this section's n1s3 dir holds the strength (step one's n1s dir lacks it): read it in place of n1s (same renaming)
    MP.work = lambda r, sp, name: old(r, sp, "n1s3" if name == "n1s" else name)
    try:
        return N.build(model, i, "x_E2", s, qs)
    finally:
        MP.work = old


def tq_sn1(model, i, s_dot, qs, cfg=None):
    sp, tx = tq_build(model, i, s_dot, qs)
    if cfg is None:
        tsp, ttx = tq_build(model, 0, s_dot, N.restricted(model, 0))
        cfg = ML.n1_pick(tsp, ttx, N.dash(s_dot))
    return sp, tx, cfg


def qaqa_raw(model, s_dot):
    el, _, cl = Q.SET[model][:3]
    R = Q.Raw(model, "qaqafull", BANKQ, ["test_full570_h22n", "test_full570_h22n1", "test_full570_h22n3", "test_full570_h22z"])
    R.gdir = Path(NSD) / "gen_prompt" / "h22" / "qaqafull" / model / "test"
    arms = ["x_E2", "x_E2:E%d.w.p%s" % (el, s_dot)] + ["x_E2:E%d.w.p%s+C%d.w.m%s" % (el, s_dot, cl, k) for k in LADK]
    for a in arms:
        R.need(a)
        alias = a.replace("x_E2:", "x:") if ":" in a else "x"
        R.rows[alias], R.cond[alias] = R.rows[a], R.cond[a]
    return R


def qaqa_sn1(R, s_dot):
    pq = [q for q in R.ids if q in PILOT]
    grid = [((ce, co), sum(R.n1_system(s_dot, ce, co)[0][q] for q in pq)) for ce in Q.CAPS for co in Q.CAPS]
    best = max(grid, key=lambda x: (x[1], -grid.index(x)))[0]
    return best, R.n1_system(s_dot, best[0], best[1])


def fact_arm(task, model, split, arm):
    b, bp = NP.bank(task, split)
    p = os.path.join(NP.S.NS, "gen_prompt", "h22", task, model, split, NP.G.arm_token(arm) + ".json")
    g = NP.G.load_generation(p, SimpleNamespace(task=task, split=split, model=model, items=bp), list(b))
    if g["contract"]["arm"] != arm or g["contract"].get("positions") != ("prompt" if ":" in arm else None):
        raise SystemExit("%s is not the pass %s" % (p, arm))
    out, txt = {}, {}
    for r in g["rows"]:
        blank = bool(r["rule_abstain"]) or not r["text"].strip()
        v = NP.AM.verdict(r["text"], b[r["id"]]["gold"], not blank)
        q = "%s:%s" % (task, r["id"])
        out[q], txt[q] = ((v == "R") - (v == "W"), not blank), r["text"]
    return out, txt


def fact_base(task, model, split):
    """no write and single E (main strength) from the plain-prompt files, as ns_prompt_table reads them."""
    M = NP.S.models()
    el, main_s, *_ = NP.MAIN[model]
    xsp, xtx, *_ = NP.load(task, model, M[model], split)
    d = {}
    for name, cfg in (("no write", ("base",)), ("only E", ("E", NP.dash(main_s)))):
        st, ans = NP.S.system(xsp, xtx, cfg)[:2]
        d[name] = {q: (v, bool(a)) for q, v, a in zip(xsp.qs, NP.S.vals(st), ans)}
    return list(xsp.qs), d


def choose(model):
    el = Q.SET[model][0]
    ss, sh = strengths(model)
    rows = {}
    ct = N.restricted(model, 0)
    for s in (ss, sh):
        sp, tx, cfg = tq_sn1(model, 0, s, ct)
        tq = N.net(N.run_sys(sp, tx, cfg))[0]
        caps, sn = qaqa_sn1(qaqa_raw(model, s), s)
        qn = sum(sn[0][q] for q in PILOT)
        fe = {task: sum(v[0] for v in fact_arm(task, model, "tune", "x_E2:E%d.w.p%s" % (el, s))[0].values()) for task in ("triviaqa", "popqa")}
        rows[s] = dict(truthfulqa_tune_sn1=tq, tq_cfg=list(cfg), qaqa_pilot_sn1=qn, qaqa_caps=list(caps), triviaqa_tune_se=fe["triviaqa"],
                       popqa_tune_se=fe["popqa"], total=tq + qn + fe["triviaqa"] + fe["popqa"])
        print("  %-12s E %-6s TruthfulQA tune SN1 %+d (%s) | (QA)^2 pilot SN1 %+d (caps %s / %s) | TriviaQA tune SE %+d | PopQA tune SE %+d | sum %+d" % (
            model, s, tq, cfg, qn, caps[0], caps[1], fe["triviaqa"], fe["popqa"], rows[s]["total"]))
    pick = sh if rows[sh]["total"] >= rows[ss]["total"] else ss
    out = CHOICE3 % model
    rec = {"model": model, "s_sel": ss, "s_half": sh, "chosen": pick, "rows": rows}
    if os.path.exists(out):
        if json.load(open(out))["chosen"] != pick:
            raise SystemExit("%s exists with another choice" % out)
    else:
        with open(out, "x") as f:
            json.dump(rec, f, indent=1)
    print("  chosen %s: E %s" % (model, pick))
    print(pick)


def neg(t):
    return bool(SPB.NEG_RE.match(t or ""))


def fact():
    jci = lambda v: "%+d [%+d, %+d]" % tuple(NP.S.joint_ci([v]))  # noqa: E731
    for task, split in FACT:
        per, qsm, negs = {}, {}, {}
        for model in ORDER:
            el = Q.SET[model][0]
            ss, sh = strengths(model)
            qs, d = fact_base(task, model, split)
            d["S0"], t0 = fact_arm(task, model, split, "x_E2")
            d["SE s_sel"], t1 = fact_arm(task, model, split, "x_E2:E%d.w.p%s" % (el, ss))
            d["SE s_half"], t2 = fact_arm(task, model, split, "x_E2:E%d.w.p%s" % (el, sh))
            per[model], qsm[model] = d, qs
            negs[model] = {k: (sum(1 for q in qs if d["S0"][q][0] == 1 and d[k][q][0] == -1),
                               sum(1 for q in qs if d["S0"][q][0] == 1 and d[k][q][0] == -1 and neg(t[q])))
                           for k, t in (("SE s_sel", t1), ("SE s_half", t2))}
        common = sorted(set.intersection(*(set(v) for v in qsm.values())))
        N2.line(per, ("no write", "only E", "S0", "SE s_sel", "SE s_half"),
                (("SE s_half", "SE s_sel"), ("SE s_half", "S0"), ("SE s_half", "no write"), ("SE s_half", "only E")),
                lambda m: common if m == "five models" else qsm[m], jci, ORDER, "8 %s %s (alias match)" % (task, split))
        print("    right under S0 -> wrong under SE (of them opening with a negation): " + " | ".join(
            "%s s_sel %d (%d), s_half %d (%d)" % ((m,) + negs[m]["SE s_sel"] + negs[m]["SE s_half"]) for m in ORDER)
            + " | five models s_sel %d (%d), s_half %d (%d)" % tuple(sum(negs[m][k][j] for m in ORDER) for k in ("SE s_sel", "SE s_half") for j in (0, 1)))


def final():
    jci = lambda v: "%+d [%+d, %+d]" % tuple(NP.S.joint_ci([v]))  # noqa: E731
    ch = {m: json.load(open(CHOICE3 % m)) for m in ORDER}
    print("# chosen strengths: " + ", ".join("%s %s (s_sel %s)" % (m, ch[m]["chosen"], ch[m]["s_sel"]) for m in ORDER))
    # TruthfulQA test
    per, qsm = {}, {}
    for m in ORDER:
        display, layer, s_main = A.KEY[m]
        ss, sc = ch[m]["s_sel"], ch[m]["chosen"]
        c1 = json.load(open(N.CHOICE % m))
        cx, ct = N.restricted(m, 1), N.restricted(m, 0)
        xsp, xtx = N.build(m, 1, "x", s_main, cx)
        tsp, ttx = N.build(m, 0, "x", s_main, ct)
        psp, ptx, cfg = tq_sn1(m, 1, sc, cx, None if sc != ss else tuple(c1["cfg"]))
        ssp, stx = N.build(m, 1, "x_E2", c1["strength"], cx)
        d = {"no write": N.run_sys(xsp, xtx, ("base",)), "only E": N.run_sys(xsp, xtx, ("E", s_main)),
             "N1": N.run_sys(xsp, xtx, ML.n1_pick(tsp, ttx, s_main)), "S0": N.run_sys(psp, ptx, ("base",)),
             "SEm": N.run_sys(psp, ptx, ("E", N.dash(sc))), "SN1m": N.run_sys(psp, ptx, cfg),
             "SN1 s_sel": N.run_sys(ssp, stx, tuple(c1["cfg"])), "SE s_sel": N.run_sys(ssp, stx, ("E", c1["strength"]))}
        qs = [q for q in psp.qs if all(q in d[k] for k in d)]
        per[m], qsm[m] = d, qs
    common = sorted(set.intersection(*(set(v) for v in qsm.values())))
    N2.line(per, ("no write", "only E", "N1", "S0", "SE s_sel", "SN1 s_sel", "SEm", "SN1m"),
            (("SN1m", "S0"), ("SEm", "S0"), ("SN1m", "N1"), ("SEm", "only E"), ("SN1m", "no write"), ("SN1m", "SN1 s_sel"), ("SEm", "SE s_sel")),
            lambda mm: common if mm == "five models" else qsm[mm], N.ps.ci, ORDER, "8 TruthfulQA test at the chosen strength")
    # (QA)^2
    fixed = json.load(open(os.path.join(NSD, "splits/qaqa_tune_choice_fixedE_20261122.json")))
    per, qtype = {}, None
    for m in ORDER:
        ss, sc = ch[m]["s_sel"], ch[m]["chosen"]
        base = Q.Raw(m, "qaqafull", BANKQ, ("test_full570", "test_full570_tuned", "test_full570_strong"))
        fx = fixed[m]
        n1 = base.n1_system(fx["n1_strength"], fx["n1_cap_e"], fx["n1_cap_own"])
        Rs, Rc = qaqa_raw(m, ss), qaqa_raw(m, sc)
        _, sns = qaqa_sn1(Rs, ss)
        _, snc = qaqa_sn1(Rc, sc)
        qtype = base.qtype
        dd = {"no write": base.arm_system("x"), "only E": base.arm_system(base.e(fx["single_E"])), "N1": (n1[0], n1[1]),
              "S0": Rc.arm_system("x"), "SE s_sel": Rs.arm_system(Rs.e(ss)), "SN1 s_sel": (sns[0], sns[1]),
              "SEm": Rc.arm_system(Rc.e(sc)), "SN1m": (snc[0], snc[1])}
        per[m] = {k: {q: (v[0][q], bool(v[1][q])) for q in base.ids} for k, v in dd.items()}
    ids = [q for q in per[ORDER[0]]["S0"] if q not in PILOT]
    for t, tl in (("all", "all types"),) + Q.TYPES:
        qs = [q for q in ids if t == "all" or qtype[q] == t]
        N2.line(per, ("no write", "only E", "N1", "S0", "SE s_sel", "SN1 s_sel", "SEm", "SN1m"),
                (("SN1m", "S0"), ("SEm", "S0"), ("SN1m", "N1"), ("SEm", "only E"), ("SN1m", "no write"), ("SN1m", "SN1 s_sel"), ("SEm", "SE s_sel")),
                lambda mm: qs, jci, ORDER, "8 (QA)^2 470 held-out, %s, chosen strength" % tl)
    # fact sets
    for task, split in FACT:
        per, qsm = {}, {}
        for m in ORDER:
            el = Q.SET[m][0]
            qs, d = fact_base(task, m, split)
            d["S0"] = fact_arm(task, m, split, "x_E2")[0]
            d["SE s_sel"] = fact_arm(task, m, split, "x_E2:E%d.w.p%s" % (el, ch[m]["s_sel"]))[0]
            d["SEm"] = fact_arm(task, m, split, "x_E2:E%d.w.p%s" % (el, ch[m]["chosen"]))[0]
            per[m], qsm[m] = d, qs
        common = sorted(set.intersection(*(set(v) for v in qsm.values())))
        N2.line(per, ("no write", "only E", "S0", "SE s_sel", "SEm"),
                (("SEm", "S0"), ("SEm", "only E"), ("SEm", "no write"), ("SEm", "SE s_sel")),
                lambda mm: common if mm == "five models" else qsm[mm], jci, ORDER, "8 %s %s at the chosen strength (alias match)" % (task, split))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "choose":
        choose(sys.argv[2])
    elif mode == "fact":
        fact()
    elif mode == "final":
        final()
    else:
        raise SystemExit("usage: n1_on_sentence3.py choose MODEL | fact | final")


if __name__ == "__main__":
    main()
