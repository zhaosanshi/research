# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# A model with the E write at ANOTHER LAYER than the one chosen for TruthfulQA alone (the general form of the first
# version, diag/l16_table.py, which was Llama only). Registered so far:
#   Llama-3.1-8B   layer 16 instead of 11, strengths 2.5 / 3 / 3.5 / 4   (launchers l16_all.sh, l16b_all.sh, tag l16)
#   Gemma-3-27B    layer 28 instead of 26, strengths 1 / 1.5 / 2         (launcher g27_l28.sh, tag l28)
#   Gemma-3-12B    layer 24 instead of 22, strengths 1.5 / 2 / 2.5 / 3   (launcher alt_chain.sh, tag l24)
# Both layers were found on the MMLU layer map (capability/layermap): the E write costs far less MMLU there.
# Per model: TruthfulQA free-form, DeepSeek Pro, score = right - wrong, single E and N1 at each strength, next to the
# MMLU change on the full test split (capability/full_<tag>; the 13,042 items outside the diagnostic list, and the two
# halves of diag/per_model_table.py). The C direction, the single-C arms, the no-write answers and the prompt-only
# baseline are those of the model's usual tables (work dirs of main_table_same_n.py); the arms of the other layer come
# from eval_extra/judge_<split>_<tag>_<split>. N1's two caps are chosen on tune per strength, by the rule of
# diag/dose_mid.py. For reference the numbers of the usual layer (diag/dose_mid.py) are printed with their MMLU change.
# Where the files exist: the strong form at the same layer and strength (must-answer + E, its N most confident
# answers, N = N1's answers; work dir judge_test_dmust<tag>_test and eval_extra/confidence_pm<tag>_test.json, read by
# strong_mid.must_dir), and the mean-difference direction at the same layer with the same write amount (kind mdg, work
# dir judge_test_<tag>mdg_test).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/alt_layer.py [MODEL ...]
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
CAP = "Iteration_01/capability"


def val(x):
    return (x == "R") - (x == "W")


class Alt:
    def __init__(self, model, ng, layer, doses, tag):
        self.model, self.ng, self.layer, self.doses, self.tag = model, ng, layer, tuple(doses), tag

    def load(self, i):
        """(usual Data, Split at the other layer over old + new verdicts, texts) for tune (0) or test (1), or None."""
        D = MT.Data(self.model, i)
        split = ("tune", "test")[i]
        w = "%s/eval_extra/judge_%s_%s_%s" % (D.m["ec"][i].split("/eval_extra/")[0], split, self.tag, split)
        if not os.path.exists(w + "/judge2/single_pro.jsonl"):
            return None
        st, bl = dict(D.sp.st), dict(D.sp.blank)
        st1, bl1 = ps.load(w, "pro")
        tx = dict(fr.raw_texts(D.m["ec"][i]))
        tx1 = fr.raw_texts(w)
        # the new work dir judges the no-write answers again: they must be the answers and verdicts of the usual
        # tables, otherwise "no write" would mean two things; the usual ones are kept
        bad = [k for k in st1 if k in st and (st1[k] != st[k] or bl1[k] != bl[k] or tx1.get(k) != tx.get(k))]
        if bad:
            raise SystemExit("%s: %d cells of %s differ from the usual tables, e.g. %s" % (self.model, len(bad), w, bad[:3]))
        st.update(st1)
        bl.update(bl1)
        tx.update(tx1)
        sh = S.shell(self.layer, D.m["c"])
        need = ["baseline_x"] + [c for s in self.doses for c in [sh.cE(s)] + [sh.cEC(s, k) for k in LAD]]
        return D, S.NewSplit(self.layer, D.m["c"], st, bl, need, qs=D.sp.qs), fr.Texts(tx, tx)

    def mmlu(self):
        """{strength: {item id: correct}} with "0" for no write, or None while the full-set file is not there."""
        p = "%s/full_%s/%s.json" % (CAP, self.tag, self.ng)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        if j["n"] != 14042:
            raise SystemExit("%s is not a full-set file" % p)
        out = {"0": {r["id"]: r["pick"] == r["gold"] for r in j["arms"]["x"]["rows"]}}
        pre = "x:E%d.w.p" % self.layer
        for arm, v in j["arms"].items():
            if arm.startswith(pre):
                out[arm[len(pre):].replace(".", "-")] = {r["id"]: r["pick"] == r["gold"] for r in v["rows"]}
        return out

    def strong(self, run, xsp):
        """({strength: (states, answered)}, {strength: confidence rows}, note) of must-answer + E at this layer."""
        w = "%s/eval_extra/judge_test_dmust%s_test" % (run, self.tag)
        if not os.path.exists(w + "/judge2/single_pro.jsonl"):
            return {}, {}, "must-answer + E at layer %d not judged yet" % self.layer
        return SM.must_dir(run, w, (("pm%s_test" % self.tag, self.doses),), xsp, list(self.doses))


ALTS = (Alt("Llama-3.1-8B", "llama3.1-8b", 16, ("2-5", "3", "3-5", "4"), "l16"),
        Alt("Gemma-3-27B", "gemma3-27b", 28, ("1", "1-5", "2"), "l28"),
        Alt("Gemma-3-12B", "gemma3-12b", 24, ("1-5", "2", "2-5", "3"), "l24"))


def mm_text(acc, s, dev, rep, fresh):
    if acc is None or s not in acc:
        return "MMLU: not run"
    d = lambda ids: [acc[s][i] - acc["0"][i] for i in ids]
    return "MMLU fresh %+.2f [%+.2f, %+.2f] | selection half %+.2f | report half %+.2f" % (
        PM.boot(d(fresh)) + (100.0 * sum(d(dev)) / len(dev), 100.0 * sum(d(rep)) / len(rep)))


def report(a):
    t, x = a.load(0), a.load(1)
    if t is None:
        print("## %s: no layer-%d verdicts on tune yet" % (a.model, a.layer))
        return
    Dt, tsp, ttx = t
    acc = a.mmlu()
    dev = rep = fresh = None
    if acc:
        dev, rep = PM.halves(list(acc["0"]))
        with open(PM.SEEN, encoding="utf-8") as f:
            seen = set(json.load(f)["ids"])
        fresh = [i for i in acc["0"] if i not in seen]
    print("## %s, E at layer %d (C at layer %d): tune %d questions%s" % (
        a.model, a.layer, Dt.m["c"], len(tsp.qs), "" if x else "; no test verdicts yet"))
    picks = {}
    for s in a.doses:
        cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        picks[s] = max(cands, key=lambda c: (ps.score(S.system(tsp, ttx, c)[0], 1), -cands.index(c)))
        e, n1 = S.system(tsp, ttx, ("E", s)), S.system(tsp, ttx, picks[s])
        print("  tune E %-4s single E net %+4d (answers %3d) | N1 %s net %+4d (answers %3d)" % (
            DM.show(s), sum(map(val, e[0])), sum(e[1]), ro.name_of(picks[s]), sum(map(val, n1[0])), sum(n1[1])))
    if x is None:
        for s in a.doses:
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
    must, conf, note = a.strong(run, xsp)
    if note:
        print("  strong form: %s" % note)
    for s in a.doses:
        e, n1 = S.system(xsp, xtx, ("E", s)), S.system(xsp, xtx, picks[s])
        d = lambda other: ps.ci([val(p) - val(q) for p, q in zip(n1[0], other)])
        if s in must and s in conf:
            de, rows, N = must[s], conf[s], sum(n1[1])
            order = sorted([i for i, ok in enumerate(de[1]) if ok], key=lambda i: (-rows[qs[i]]["mean_logp"], i))
            if len(order) >= N:
                strong = MT.keep(de[0], order, N)
                print("  E %-4s strong form (must-answer + E at layer %d, top %d): net %+4d | N1 minus the strong form %s | must-answer + E, every answer: net %+4d (answers %3d)" % (
                    DM.show(s), a.layer, N, sum(map(val, strong[0])), d(strong[0]), sum(map(val, de[0])), sum(de[1])))
            else:
                print("  E %-4s strong form: only %d must-answer answers to choose from (fewer than N1's %d)" % (DM.show(s), len(order), N))
        print("  E %-4s single E net %+4d (answers %3d) | N1 net %+4d (answers %3d) | %s" % (
            DM.show(s), sum(map(val, e[0])), sum(e[1]), sum(map(val, n1[0])), sum(n1[1]), mm_text(acc, s, dev, rep, fresh)))
        print("         N1 minus: no write %s | single C %s | single E %s | prompt only %s | single E minus no write %s" % (
            d(base[0]), d(cs), d(e[0]), d(da_s), ps.ci([val(p) - val(q) for p, q in zip(e[0], base[0])])))
    wm = "%s/eval_extra/judge_test_%smdg_test" % (run, a.tag)
    if os.path.exists(wm + "/judge2/single_pro.jsonl"):
        st1, bl1 = ps.load(wm, "pro")
        print("## the mean-difference direction at layer %d with the same write amount (kind mdg), single write" % a.layer)
        for s in a.doses:
            c = "E_L%d_x-mdg-p%s" % (a.layer, s)
            if any((q, c) not in st1 for q in qs):
                continue
            keep = [i for i, q in enumerate(qs) if st1[(q, c)] != "?"]
            md = [st1[(qs[i], c)] for i in keep]
            e, n1 = S.system(xsp, xtx, ("E", s)), S.system(xsp, xtx, picks[s])
            dd = lambda u, v: ps.ci([val(p) - val(q) for p, q in zip(u, v)])
            print("  E %-4s mean difference: net %+4d (answers %3d, %d questions) | learned single E minus it %s | N1 minus it %s | it minus no write %s" % (
                DM.show(s), sum(map(val, md)), sum(not bl1[(qs[i], c)] for i in keep), len(keep),
                dd([e[0][i] for i in keep], md), dd([n1[0][i] for i in keep], md), dd(md, [base[0][i] for i in keep])))
    print("## for reference, layer %d (diag/dose_mid.py, on its own question list: not paired with the rows above)" % Dx.m["e"])
    D0, sp0, tx0, doses0, _, _ = DM.merged_all(a.model, 1)
    _, tsp0, ttx0, _, _, _ = DM.merged_all(a.model, 0)
    acc0 = PM.mmlu(a.model, Dx.m["e"])
    for s in doses0:
        cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        pick = max(cands, key=lambda c: (ps.score(S.system(tsp0, ttx0, c)[0], 1), -cands.index(c)))
        e, n1 = S.system(sp0, tx0, ("E", s)), S.system(sp0, tx0, pick)
        m = ""
        if s in acc0 and fresh:
            m = " | MMLU fresh %+.2f" % (100.0 * sum(acc0[s][i] - acc0["0"][i] for i in fresh) / len(fresh))
        print("  E %-4s single E net %+4d (answers %3d) | N1 net %+4d (answers %3d)%s" % (
            DM.show(s), sum(map(val, e[0])), sum(e[1]), sum(map(val, n1[0])), sum(n1[1]), m))


def main():
    names = sys.argv[1:]
    for a in ALTS:
        if not names or a.model in names:
            report(a)
            print()


if __name__ == "__main__":
    main()
