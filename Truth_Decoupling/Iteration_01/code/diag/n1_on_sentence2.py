# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22, queue I section 7), read-only, no model, no judge call. Exploration.
# P = x_E2 (plain prompt + honest sentence), writes at "prompt", s_sel = the E strength chosen in step one
# (newsets/h22/n1s_choice_<model>.json). diag/n1_on_sentence.py is imported unchanged (its builder and choice files).
# Modes:
#   onepass  7.5: one pass of P + E + C. On the TruthfulQA tune split the cell (E strength of step one's two, C of
#            0.25 / 0.5 / 1 / 2) with the highest net (ties: smaller C, then smaller E); on test (only when that E is s_sel)
#            one-pass - S0 and SN1 - one-pass. Existing verdicts only.
#   singlec  7.2: SC = P first; a question P leaves blank is re-asked with P + C at 0.25, 0.5, ... up to one cap (the
#            first answered one is taken); the cap (0.25 / 0.5 / 1 / 2) chosen on tune by net, ties to the smaller cap.
#            Test: SC - S0, SN1 - SC, SE - SC. C from eval_extra/judge_<split>_h22c4_<split> (0.25) and
#            judge_<split>_n1sc_<split> (0.5 / 1 / 2).
#   qaqa     7.1: (QA)^2 570, P + E(s_sel) and P + E(s_sel) + C (gen dir newsets/gen_prompt/h22/qaqafull/<model>/test,
#            judge work dirs test_full570_h22n, or _h22n1 where _h22n stopped at the cache-merge check (Llama-3.1-8B,
#            Gemma-3-27B: rejudged with the S0 cache only, lead session 00:02), and _h22z (P)); SN1 by
#            diag/qaqa_pilot_table.Raw.n1_system with the arms renamed to the plain-prompt names. Caps: main = chosen on
#            the 100 pilot questions over every pair of caps (none allowed; ties to the earlier pair, as
#            qaqa_pilot_table.tune with E fixed); side = the caps of step one's TruthfulQA tune choice. Compared with
#            the no-sentence systems of the fixed-E (QA)^2 choice (qaqa_tune_choice_fixedE_20261122.json).
#   fact     7.4: PopQA / TriviaQA test and test2, alias match: SE = P + E(s_sel), S0 = P, no-sentence single E at the
#            main strength and the plain no-write pass (diag/ns_prompt_table.py's reading).
# Intervals: TruthfulQA policy_select2.ci (per model and on the per-question sums of the common questions); (QA)^2 and
# fact sets newsets_score.joint_ci on the per-question sums.
# usage (scoring container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/n1_on_sentence2.py MODE
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import n1_on_sentence as N    # noqa: E402

A, MP, ML, S, ps = N.A, N.MP, N.ML, N.S, N.ps
V, ORDER = N.V, N.ORDER
LADK = ("0.25", "0.5", "1", "2")


def choice(model):
    return json.load(open(N.CHOICE % model))


def ci_tq(v):
    return ps.ci(v)


def line(per, names, diffs, qs_of, ci, models=ORDER, title=""):
    print("\n## %s" % title)
    for m in list(models) + ["five models"]:
        ms = models if m == "five models" else (m,)
        qs = qs_of(m)
        label = m if m != "five models" or len(models) == 5 else "%d together" % len(models)
        print("    %-12s %3d q | " % (label, len(qs)) + " | ".join("%s %+d (%d)" % (
            n, sum(per[x][n][q][0] for x in ms for q in qs), sum(per[x][n][q][1] for x in ms for q in qs)) for n in names)
            + " || " + " | ".join("%s - %s %s" % (a, b, ci([sum(per[x][a][q][0] - per[x][b][q][0] for x in ms) for q in qs])) for a, b in diffs))


# ----------------------------------------------------------------------------- 7.5
def onepass():
    per, qsm = {}, {}
    for model in ORDER:
        display, layer, s_main = A.KEY[model]
        ch = choice(model)
        ssel, cfg = ch["strength"], tuple(ch["cfg"])
        ct, cx = N.restricted(model, 0), N.restricted(model, 1)
        cands = []
        for s in (s_main, N.half(s_main)):
            sp, tx = N.build(model, 0, "x_E2", s, ct)
            for k in ps.LAD:
                c = S.shell(layer, sp.m["c"]).cEC(s, k)
                cands.append((sum(V[sp.st[(q, c)]] for q in sp.qs), sum(not sp.blank[(q, c)] for q in sp.qs), s, k))
        best = sorted(cands, key=lambda x: (-x[0], float(ps.NAME[x[3]]), float(N.dot(x[2]))))[0]
        print("# %s tune, one pass of P + E + C: " % model + " | ".join("E %s C %s %+d (%d)" % (N.dot(s), ps.NAME[k], n, a) for n, a, s, k in cands)
              + " -> chosen E %s C %s (%+d)" % (N.dot(best[2]), ps.NAME[best[3]], best[0]))
        if best[2] != ssel:
            print("    the chosen E strength is not s_sel (%s): no test reading" % N.dot(ssel))
            continue
        sp, tx = N.build(model, 1, "x_E2", ssel, cx)
        c = S.shell(layer, sp.m["c"]).cEC(ssel, best[3])
        d = {"S0": N.run_sys(sp, tx, ("base",)), "SN1": N.run_sys(sp, tx, cfg),
             "one pass": {q: (V[sp.st[(q, c)]], not sp.blank[(q, c)]) for q in sp.qs}}
        per[model], qsm[model] = d, list(sp.qs)
    if len(per) < len(ORDER):
        print("\n# test: %d of 5 models have a reading; the five-model total is over those models only" % len(per))
    models = [m for m in ORDER if m in per]
    common = sorted(set.intersection(*(set(qsm[m]) for m in models))) if models else []
    line(per, ("S0", "one pass", "SN1"), (("one pass", "S0"), ("SN1", "one pass")),
         lambda m: common if m == "five models" else qsm[m], ci_tq, models, "7.5 TruthfulQA test: one pass of P + E + C (cell chosen on tune)")


# ----------------------------------------------------------------------------- 7.2
def c_cells(model, i, qs):
    display, layer, _ = A.KEY[model]
    D = MP.MT.Data(display, i)
    split = ("tune", "test")[i]
    run = D.m["ec"][i].split("/eval_extra/")[0]
    cl = D.m["c"]
    out = {}
    for w, ks in ((MP.work(run, split, "h22c4"), ("0-25",)), (MP.work(run, split, "n1sc"), ("0-5", "1", "2"))):
        if not MP.has(w):
            raise FileNotFoundError("%s: no Pro verdicts in %s" % (model, w))
        st, bl = ps.load(w, "pro")
        for k in ks:
            c = "C_L%d_x_E2-w-m%s" % (cl, k)
            out[k] = {q: (V[st[(q, c)]], not bl[(q, c)]) for q in qs if st.get((q, c), "?") != "?"}
    return out


def sc_system(s0, cs, cap, qs):
    lad = ps.LAD[:ps.LAD.index(cap) + 1]
    out = {}
    for q in qs:
        if s0[q][1]:
            out[q] = s0[q]
            continue
        k = next((k for k in lad if cs[k][q][1]), None)
        out[q] = s0[q] if k is None else cs[k][q]
    return out


def singlec():
    per, qsm = {}, {}
    for model in ORDER:
        ch = choice(model)
        ssel, cfg = ch["strength"], tuple(ch["cfg"])
        ct, cx = N.restricted(model, 0), N.restricted(model, 1)
        tsp, ttx = N.build(model, 0, "x_E2", ssel, ct)
        t0 = N.run_sys(tsp, ttx, ("base",))
        tc = c_cells(model, 0, tsp.qs)
        tq = [q for q in tsp.qs if all(q in tc[k] for k in ps.LAD)]
        nets = [(sum(sc_system(t0, tc, k, tq)[q][0] for q in tq), k) for k in ps.LAD]
        cap = max(nets, key=lambda x: (x[0], -ps.LAD.index(x[1])))[1]
        print("# %s tune SC by cap: %s -> cap %s (%d tune questions)" % (model, " | ".join("%s %+d" % (ps.NAME[k], n) for n, k in nets), ps.NAME[cap], len(tq)))
        xsp, xtx = N.build(model, 1, "x_E2", ssel, cx)
        x0 = N.run_sys(xsp, xtx, ("base",))
        xc = c_cells(model, 1, xsp.qs)
        qs = [q for q in xsp.qs if all(q in xc[k] for k in ps.LAD)]
        per[model] = {"S0": x0, "SC": sc_system(x0, xc, cap, qs), "SE": N.run_sys(xsp, xtx, ("E", ssel)), "SN1": N.run_sys(xsp, xtx, cfg)}
        qsm[model] = qs
    common = sorted(set.intersection(*(set(qsm[m]) for m in ORDER)))
    line(per, ("S0", "SC", "SE", "SN1"), (("SC", "S0"), ("SN1", "SC"), ("SE", "SC")),
         lambda m: common if m == "five models" else qsm[m], ci_tq, ORDER, "7.2 TruthfulQA test: P + C only (SC), cap chosen on tune")


# ----------------------------------------------------------------------------- 7.1
def qaqa():
    import qaqa_pilot_table as Q
    NSD = Q.S.NS
    bankp = os.path.join(NSD, "bank/qaqa_full_bank_20261122.jsonl")
    pilot = set(json.load(open(os.path.join(NSD, "splits/qaqa_pilot_20261122.json")))["ids"])
    fixed = json.load(open(os.path.join(NSD, "splits/qaqa_tune_choice_fixedE_20261122.json")))
    jci = lambda v: "%+d [%+d, %+d]" % tuple(Q.S.joint_ci([v]))  # noqa: E731
    per, qtype = {}, None
    for model in ORDER:
        el, _, cl = Q.SET[model][:3]
        ch = choice(model)
        ssel = ch["strength_dot"]
        tq_caps = [None if c is None else c.replace("-", ".") for c in ch["cfg"][2:4]]
        base = Q.Raw(model, "qaqafull", bankp, ("test_full570", "test_full570_tuned", "test_full570_strong"))
        fx = fixed[model]
        n1 = base.n1_system(fx["n1_strength"], fx["n1_cap_e"], fx["n1_cap_own"])
        R = Q.Raw(model, "qaqafull", bankp, ["test_full570_h22n", "test_full570_h22n1", "test_full570_h22z"])
        R.gdir = Path(NSD) / "gen_prompt" / "h22" / "qaqafull" / model / "test"
        arms = ["x_E2", "x_E2:E%d.w.p%s" % (el, ssel)] + ["x_E2:E%d.w.p%s+C%d.w.m%s" % (el, ssel, cl, k) for k in LADK]
        for a in arms:
            R.need(a)
            alias = a.replace("x_E2:", "x:") if ":" in a else "x"
            R.rows[alias], R.cond[alias] = R.rows[a], R.cond[a]
        if R.ids != base.ids:
            raise SystemExit("banks differ")
        qtype = base.qtype
        pq = [q for q in R.ids if q in pilot]
        grid = [((ce, co), sum(R.n1_system(ssel, ce, co)[0][q] for q in pq)) for ce in Q.CAPS for co in Q.CAPS]
        best = max(grid, key=lambda x: (x[1], -grid.index(x)))[0]
        sn = R.n1_system(ssel, best[0], best[1])
        st = R.n1_system(ssel, tq_caps[0], tq_caps[1])
        d = {"no write": base.arm_system("x"), "only E": base.arm_system(base.e(fx["single_E"])), "N1": (n1[0], n1[1]),
             "S0": R.arm_system("x"), "SE": R.arm_system(R.e(ssel)), "SN1": (sn[0], sn[1]), "SN1 tq caps": (st[0], st[1])}
        per[model] = {k: {q: (v[0][q], bool(v[1][q])) for q in R.ids} for k, v in d.items()}
        per[model]["_trans"] = None
        print("# (QA)^2 %s: s_sel %s; SN1 caps chosen on the 100 pilot questions: E's blanks %s / own blanks %s; TruthfulQA tune caps %s / %s; no-sentence N1 at E %s caps %s / %s, single E %s" % (
            model, ssel, best[0], best[1], tq_caps[0], tq_caps[1], fx["n1_strength"], fx["n1_cap_e"], fx["n1_cap_own"], fx["single_E"]))
    ids = list(per[ORDER[0]]["S0"])
    names = ("no write", "S0", "only E", "SE", "N1", "SN1", "SN1 tq caps")
    diffs = (("SN1", "S0"), ("SE", "S0"), ("SN1", "SE"), ("SN1", "N1"), ("SE", "only E"), ("SN1 tq caps", "S0"), ("SE", "no write"))
    for label, inside in (("470 held-out questions", False), ("100 pilot (tune) questions", True)):
        for t, tl in (("all", "all types"),) + Q.TYPES:
            qs = [q for q in ids if (q in pilot) == inside and (t == "all" or qtype[q] == t)]
            line(per, names, diffs, lambda m: qs, jci, ORDER, "7.1 (QA)^2 %s, %s (DeepSeek Pro)" % (label, tl))
            if t == "all":
                for m in list(ORDER) + ["five models"]:
                    ms = ORDER if m == "five models" else (m,)
                    print("    S0 -> SN1 %-12s %s" % (m, N.transitions([(per[x]["S0"][q], per[x]["SN1"][q]) for x in ms for q in qs])))


# ----------------------------------------------------------------------------- 7.4
def fact():
    import ns_prompt_table as NP
    G, AM, NSS = NP.G, NP.AM, NP.S
    M = NSS.models()
    jci = lambda v: "%+d [%+d, %+d]" % tuple(NSS.joint_ci([v]))  # noqa: E731
    for task, split in (("popqa", "test"), ("popqa", "test2"), ("triviaqa", "test"), ("triviaqa", "test2")):
        per, qsm = {}, {}
        b, bp = NP.bank(task, split)
        for model in ORDER:
            el, main_s, *_ = NP.MAIN[model]
            ssel = choice(model)["strength_dot"]
            xsp, xtx, xh, _, _, _ = NP.load(task, model, M[model], split)
            d = {}
            for name, cfg in (("no write", ("base",)), ("only E", ("E", NP.dash(main_s)))):
                states, ans = NSS.system(xsp, xtx, cfg)[:2]
                d[name] = {q: (v, bool(a)) for q, v, a in zip(xsp.qs, NSS.vals(states), ans)}
            for name, arm in (("S0", "x_E2"), ("SE", "x_E2:E%d.w.p%s" % (el, ssel))):
                p = os.path.join(NSS.NS, "gen_prompt", "h22", task, model, split, G.arm_token(arm) + ".json")
                g = G.load_generation(p, SimpleNamespace(task=task, split=split, model=model, items=bp), list(b))
                if g["contract"]["arm"] != arm or g["contract"].get("positions") != ("prompt" if ":" in arm else None):
                    raise SystemExit("%s is not the pass %s" % (p, arm))
                z = {}
                for r in g["rows"]:
                    blank = bool(r["rule_abstain"]) or not r["text"].strip()
                    v = AM.verdict(r["text"], b[r["id"]]["gold"], not blank)
                    z["%s:%s" % (task, r["id"])] = ((v == "R") - (v == "W"), not blank)
                d[name] = z
            if list(d["S0"]) != list(xsp.qs) or list(d["SE"]) != list(xsp.qs):
                raise SystemExit("question order")
            per[model], qsm[model] = d, list(xsp.qs)
        common = sorted(set.intersection(*(set(v) for v in qsm.values())))
        line(per, ("no write", "only E", "S0", "SE"), (("SE", "S0"), ("SE", "only E"), ("SE", "no write")),
             lambda m: common if m == "five models" else qsm[m], jci, ORDER, "7.4 %s %s (alias match)" % (task, split))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"onepass": onepass, "singlec": singlec, "qaqa": qaqa, "fact": fact}.get(mode, lambda: sys.exit("usage: n1_on_sentence2.py onepass|singlec|qaqa|fact"))()


if __name__ == "__main__":
    main()
