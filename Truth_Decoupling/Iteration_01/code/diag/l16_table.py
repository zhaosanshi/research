# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# Llama-3.1-8B with the E write at LAYER 16 instead of 11 (launcher logs/newsets_20261119/l16_all.sh): TruthfulQA
# free-form, DeepSeek Pro, score = right - wrong, single E and N1 at strengths 2.5 / 3 / 3.5 / 4, next to the MMLU
# change on the full test split (capability/full_l16; the 13,042 items outside the diagnostic list, and the two halves
# of diag/per_model_table.py). The C direction, the single-C arms, the no-write answers and the prompt-only baseline are
# those of the layer-11 tables (work dirs of main_table_same_n.py); the layer-16 arms come from
# eval_extra/judge_<split>_l16_<split>. N1's two caps are chosen on tune per strength, by the rule of diag/dose_mid.py.
# For reference the layer-11 numbers of diag/dose_mid.py are printed with their MMLU change.
# The strong form at the same strength (must-answer + E at layer 16, its N most confident answers, N = N1's answers)
# is printed when launchers/explore_20261120/l16_strong.sh has produced it (work dir judge_test_dmustl16_test and
# eval_extra/confidence_pml16_test.json); the files are read by strong_mid.must_dir, as for layer 11.
# The mean-difference direction at layer 16 with the same write amount (launcher l16_mdg.sh, work dir
# judge_test_l16mdg_test) is compared with the learned direction when its verdicts exist.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/l16_table.py
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import dose_mid as DM
import per_model_table as PM
import strong_mid as SM

MT, S = DM.MT, DM.S
ps, ro, fr = MT.ps, MT.ro, MT.fr
CAPS, LAD = MT.CAPS, ps.LAD
MODEL, NG, LAYER = "Llama-3.1-8B", "llama3.1-8b", 16
DOSES = ("2-5", "3", "3-5", "4")
CAP = "Iteration_01/capability"


def val(x):
    return (x == "R") - (x == "W")


def load(i):
    D = MT.Data(MODEL, i)
    split = ("tune", "test")[i]
    w = "%s/eval_extra/judge_%s_l16_%s" % (D.m["ec"][i].split("/eval_extra/")[0], split, split)
    if not os.path.exists(w + "/judge2/single_pro.jsonl"):
        return None
    st, bl = dict(D.sp.st), dict(D.sp.blank)
    st1, bl1 = ps.load(w, "pro")
    st.update(st1)
    bl.update(bl1)
    tx = dict(fr.raw_texts(D.m["ec"][i]))
    tx.update(fr.raw_texts(w))
    sh = S.shell(LAYER, D.m["c"])
    need = ["baseline_x"] + [c for s in DOSES for c in [sh.cE(s)] + [sh.cEC(s, k) for k in LAD]]
    return D, S.NewSplit(LAYER, D.m["c"], st, bl, need, qs=D.sp.qs), fr.Texts(tx, tx)


def mmlu16():
    p = "%s/full_l16/%s.json" % (CAP, NG)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        j = json.load(f)
    if j["n"] != 14042:
        raise SystemExit("%s is not a full-set file" % p)
    out = {"0": {r["id"]: r["pick"] == r["gold"] for r in j["arms"]["x"]["rows"]}}
    for arm, v in j["arms"].items():
        pre = "x:E%d.w.p" % LAYER
        if arm.startswith(pre):
            out[arm[len(pre):].replace(".", "-")] = {r["id"]: r["pick"] == r["gold"] for r in v["rows"]}
    return out


def mm_text(acc, s, dev, rep, fresh):
    if acc is None or s not in acc:
        return "MMLU: not run"
    d = lambda ids: [acc[s][i] - acc["0"][i] for i in ids]
    return "MMLU fresh %+.2f [%+.2f, %+.2f] | selection half %+.2f | report half %+.2f" % (
        PM.boot(d(fresh)) + (100.0 * sum(d(dev)) / len(dev), 100.0 * sum(d(rep)) / len(rep)))


def main():
    t, x = load(0), load(1)
    if t is None:
        raise SystemExit("no layer-16 verdicts on tune yet")
    Dt, tsp, ttx = t
    acc = mmlu16()
    dev = rep = fresh = None
    if acc:
        dev, rep = PM.halves(list(acc["0"]))
        with open(PM.SEEN, encoding="utf-8") as f:
            seen = set(json.load(f)["ids"])
        fresh = [i for i in acc["0"] if i not in seen]
    print("## %s, E at layer %d (C at layer %d): tune %d questions%s" % (
        MODEL, LAYER, Dt.m["c"], len(tsp.qs), "" if x else "; no test verdicts yet"))
    picks = {}
    for s in DOSES:
        cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        picks[s] = max(cands, key=lambda c: (ps.score(S.system(tsp, ttx, c)[0], 1), -cands.index(c)))
        e, n1 = S.system(tsp, ttx, ("E", s)), S.system(tsp, ttx, picks[s])
        print("  tune E %-4s single E net %+4d (answers %3d) | N1 %s net %+4d (answers %3d)" % (
            DM.show(s), sum(map(val, e[0])), sum(e[1]), ro.name_of(picks[s]), sum(map(val, n1[0])), sum(n1[1])))
    if x is None:
        for s in DOSES:
            print("  E %-4s %s" % (DM.show(s), mm_text(acc, s, dev, rep, fresh)))
        return
    Dx, xsp, xtx = x
    qs = xsp.qs
    base = S.system(xsp, xtx, ("base",))
    da0 = MT.handed_in(Dx, ("DA",))
    da = dict(zip(Dx.sp.qs, da0[0]))
    da_s = [da[q] for q in qs]
    cks = [k for k in ps.LAD if k in Dt.sp.cks and k in Dx.sp.cks]
    ck = max(cks, key=lambda k: (ps.score(ro.states(Dt.sp, Dt.tx, ("C", k)), 1), -cks.index(k)))
    c0 = MT.plain(Dx, ("C", ck))
    cby = dict(zip(Dx.sp.qs, c0[0]))
    cs = [cby[q] for q in qs]
    print("## test %d questions (old work dir %d): no write net %+d (answers %d) | single C (%s) net %+d | prompt only net %+d" % (
        len(qs), len(Dx.sp.qs), sum(map(val, base[0])), sum(base[1]), ps.NAME[ck], sum(map(val, cs)), sum(map(val, da_s))))
    run = Dx.m["ec"][1].split("/eval_extra/")[0]
    w16 = run + "/eval_extra/judge_test_dmustl16_test"
    must, conf, note = ({}, {}, "must-answer + E at layer 16 not judged yet")
    if os.path.exists(w16 + "/judge2/single_pro.jsonl"):
        must, conf, note = SM.must_dir(run, w16, (("pml16_test", DOSES),), xsp, list(DOSES))
    if note:
        print("  strong form: %s" % note)
    for s in DOSES:
        e, n1 = S.system(xsp, xtx, ("E", s)), S.system(xsp, xtx, picks[s])
        d = lambda other: ps.ci([val(a) - val(b) for a, b in zip(n1[0], other)])
        if s in must and s in conf:
            de, rows, N = must[s], conf[s], sum(n1[1])
            order = sorted([i for i, a in enumerate(de[1]) if a], key=lambda i: (-rows[qs[i]]["mean_logp"], i))
            if len(order) >= N:
                strong = MT.keep(de[0], order, N)
                print("  E %-4s strong form (must-answer + E at layer 16, top %d): net %+4d | N1 minus the strong form %s | must-answer + E, every answer: net %+4d (answers %3d)" % (
                    DM.show(s), N, sum(map(val, strong[0])), d(strong[0]), sum(map(val, de[0])), sum(de[1])))
            else:
                print("  E %-4s strong form: only %d must-answer answers to choose from (fewer than N1's %d)" % (DM.show(s), len(order), N))
        print("  E %-4s single E net %+4d (answers %3d) | N1 net %+4d (answers %3d) | %s" % (
            DM.show(s), sum(map(val, e[0])), sum(e[1]), sum(map(val, n1[0])), sum(n1[1]), mm_text(acc, s, dev, rep, fresh)))
        print("         N1 minus: no write %s | single C %s | single E %s | prompt only %s | single E minus no write %s" % (
            d(base[0]), d(cs), d(e[0]), d(da_s), ps.ci([val(a) - val(b) for a, b in zip(e[0], base[0])])))
    wm = run + "/eval_extra/judge_test_l16mdg_test"
    if os.path.exists(wm + "/judge2/single_pro.jsonl"):
        st1, bl1 = ps.load(wm, "pro")
        print("## the mean-difference direction at layer %d with the same write amount (kind mdg), single write (launcher l16_mdg.sh)" % LAYER)
        for s in DOSES:
            c = "E_L%d_x-mdg-p%s" % (LAYER, s)
            if any((q, c) not in st1 for q in qs):
                continue
            keep = [i for i, q in enumerate(qs) if st1[(q, c)] != "?"]
            md = [st1[(qs[i], c)] for i in keep]
            e, n1 = S.system(xsp, xtx, ("E", s)), S.system(xsp, xtx, picks[s])
            dd = lambda a, b: ps.ci([val(x) - val(y) for x, y in zip(a, b)])
            print("  E %-4s mean difference: net %+4d (answers %3d, %d questions) | learned single E minus it %s | N1 minus it %s | it minus no write %s" % (
                DM.show(s), sum(map(val, md)), sum(not bl1[(qs[i], c)] for i in keep), len(keep),
                dd([e[0][i] for i in keep], md), dd([n1[0][i] for i in keep], md), dd(md, [base[0][i] for i in keep])))
    print("## for reference, layer %d (diag/dose_mid.py, same questions where both exist)" % Dx.m["e"])
    D11, sp11, tx11, doses11, _, _ = DM.merged_all(MODEL, 1)
    _, tsp11, ttx11, _, _, _ = DM.merged_all(MODEL, 0)
    acc11 = PM.mmlu(MODEL, Dx.m["e"])
    for s in doses11:
        cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        pick = max(cands, key=lambda c: (ps.score(S.system(tsp11, ttx11, c)[0], 1), -cands.index(c)))
        e, n1 = S.system(sp11, tx11, ("E", s)), S.system(sp11, tx11, pick)
        m = ""
        if s in acc11 and fresh:
            m = " | MMLU fresh %+.2f" % (100.0 * sum(acc11[s][i] - acc11["0"][i] for i in fresh) / len(fresh))
        print("  E %-4s single E net %+4d (answers %3d) | N1 net %+4d (answers %3d)%s" % (
            DM.show(s), sum(map(val, e[0])), sum(e[1]), sum(map(val, n1[0])), sum(n1[1]), m))


if __name__ == "__main__":
    main()
