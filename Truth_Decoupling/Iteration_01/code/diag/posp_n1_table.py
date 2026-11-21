# Iteration_01 analysis (Claude Opus 5.5, 2026-11-21), read-only, no model, no judge call.
# TruthfulQA free-form, DeepSeek Pro, score = right - wrong: N1 with BOTH writes at the core rule "prompt" (question
# tokens and assistant header; no decode step written) against N1 at the training rule (every generated token written
# as well), same layer and strength = the main-table setting at a budget of 1 MMLU point (the MMLU read is the same
# under the two rules). N1 = E on every question; blanks re-asked with E + C up the ladder, the two caps chosen on TUNE
# (diag/main_table_layers.n1_pick, unchanged), each rule choosing its own caps.
# Inputs per split: the usual no-write cells; single E at "prompt" (work dir judge_<split>_posprompt<tag>_<split>,
# launchers/explore_20261120/posp.sh); the E + C arms at "prompt" (judge_<split>_posprompt2<tag>_<split>, posp2.sh;
# their conditions carry the suffix "-at-prompt", removed here so that the usual N1 code reads them). The no-write
# cells of both work dirs must equal the usual ones; the training-rule side is diag/alt_layer.py's split, which gives
# the main table's N1 (checked: the printed row must equal the main table's).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/posp_n1_table.py
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import alt_layer as AL
import main_table_layers as ML
import prompt_sweep_table as PW

MT, S, ps, fr, DM = AL.MT, AL.S, AL.ps, AL.fr, AL.DM
SETS = (("Llama-3.1-8B", "l16", "3-5"), ("Gemma-3-12B", "l24", "2-5"))
SUF = "-at-prompt"


def val(x):
    return (x == "R") - (x == "W")


def prompt_split(alt, i, s):
    """(usual Data, Split at the prompt rule, its texts, Split at the training rule, its texts) or (None, why)."""
    got = alt.load(i)
    if not got:
        return None, "the training-rule work dir of this layer is missing"
    D, sp_c, tx_c = got
    split = ("tune", "test")[i]
    run = D.m["ec"][i].split("/eval_extra/")[0]
    works = ["%s/eval_extra/judge_%s_posprompt%s%s_%s" % (run, split, n, alt.tag, split) for n in ("", "2")]
    for w in works:
        if not os.path.exists(w + "/judge2/single_pro.jsonl"):
            return None, "no Pro verdicts yet in %s" % os.path.basename(w)
    usual = fr.raw_texts(D.m["ec"][i])
    st, bl, tx = {}, {}, {}
    for q in D.sp.qs:
        k = (q, "baseline_x")
        st[k], bl[k], tx[k] = sp_c.st[k], sp_c.blank[k], usual[k]
    for w, two in zip(works, (False, True)):
        st1, bl1 = ps.load(w, "pro")
        t1 = fr.raw_texts(w)
        for (q, c), v in st1.items():
            if c == "baseline_x":
                if (q, c) in st and (v != st[(q, c)] or bl1[(q, c)] != bl[(q, c)] or t1.get((q, c)) != tx[(q, c)]):
                    raise SystemExit("%s: the no-write cell of %s differs from the usual one (%s)" % (alt.model, os.path.basename(w), q))
                continue
            if two:
                if not c.endswith(SUF):
                    raise SystemExit("%s: condition %s in %s lacks %s" % (alt.model, c, os.path.basename(w), SUF))
                c = c[:-len(SUF)]
            elif c.endswith(SUF):
                raise SystemExit("%s: a two-write condition %s in the single-write work dir" % (alt.model, c))
            k = (q, c)
            if k in st:
                raise SystemExit("%s: condition %s appears in two work dirs" % (alt.model, c))
            st[k], bl[k], tx[k] = v, bl1[(q, c + SUF) if two else (q, c)], t1[(q, c + SUF) if two else (q, c)]
    sh = S.shell(alt.layer, D.m["c"])
    need = ["baseline_x", sh.cE(s)] + [sh.cEC(s, k) for k in ps.LAD]
    sp = S.NewSplit(alt.layer, D.m["c"], st, bl, need, qs=D.sp.qs)
    if sp.qs != sp_c.qs:
        raise SystemExit("%s %s: the two rules cover different questions (%d / %d)" % (alt.model, split, len(sp.qs), len(sp_c.qs)))
    return (D, sp, fr.Texts(tx, tx), sp_c, tx_c), None


def row(label, st, ans):
    return "    %-46s answers %3d | right %3d wrong %3d | net %+4d" % (label, sum(ans), st.count("R"), st.count("W"), sum(map(val, st)))


def main():
    for model, tag, s in SETS:
        alt = next(a for a in AL.ALTS if a.model == model and a.tag == tag)
        print("\n## %s, layer %d, E strength %s: N1 at the prompt rule against N1 at the training rule" % (model, alt.layer, DM.show(s)))
        T, why_t = prompt_split(alt, 0, s)
        X, why_x = prompt_split(alt, 1, s)
        if T is None or X is None:
            print("    not complete: tune %s; test %s" % (why_t or "ok", why_x or "ok"))
            continue
        Dt, tsp_p, ttx_p, tsp_c, ttx_c = T
        Dx, xsp_p, xtx_p, xsp_c, xtx_c = X
        pick_p, pick_c = ML.n1_pick(tsp_p, ttx_p, s), ML.n1_pick(tsp_c, ttx_c, s)
        for label, sp, tx, pick in (("prompt rule", tsp_p, ttx_p, pick_p), ("training rule", tsp_c, ttx_c, pick_c)):
            e = S.system(sp, tx, ("E", s))[0]
            n1 = S.system(sp, tx, pick)[0]
            print("    tune, %-14s single E net %+4d | N1 chosen on tune: %s, net %+4d" % (label, sum(map(val, e)), ro_name(pick), sum(map(val, n1))))
        qs = xsp_p.qs
        base = S.system(xsp_c, xtx_c, ("base",))
        e_c, n_c = S.system(xsp_c, xtx_c, ("E", s)), S.system(xsp_c, xtx_c, pick_c)
        e_p, n_p = S.system(xsp_p, xtx_p, ("E", s)), S.system(xsp_p, xtx_p, pick_p)
        da0 = MT.handed_in(Dx, ("DA",))
        da = dict(zip(Dx.sp.qs, da0[0]))
        da_s = [da[q] for q in qs]
        run = Dx.m["ec"][1].split("/eval_extra/")[0]
        ptune, ptest = PW.prompts_of(run, "tune", Dt.sp.qs), PW.prompts_of(run, "test", Dx.sp.qs)
        best = None
        if ptune and ptest:
            order = sorted(ptune, key=lambda p: (-sum(map(val, ptune[p][0])), p != "conf_E2", p))
            if order[0] in ptest:
                by = dict(zip(Dx.sp.qs, ptest[order[0]][0]))
                best = (order[0], [by[q] for q in qs])
        print("  test (%d questions)" % len(qs))
        print(row("no write", base[0], base[1]))
        print(row("single E, training rule", e_c[0], e_c[1]))
        print(row("N1, training rule (the main table's row)", n_c[0], n_c[1]))
        print(row("single E, prompt rule", e_p[0], e_p[1]))
        print(row("N1, prompt rule", n_p[0], n_p[1]))
        print("    %-46s net %+4d" % ("prompt only (must-answer + sentence)", sum(map(val, da_s))))
        if best:
            print("    %-46s net %+4d" % ("best prompt on tune: %s" % best[0], sum(map(val, best[1]))))
        d = lambda a, b: ps.ci([val(x) - val(y) for x, y in zip(a, b)])  # noqa: E731
        print("    N1 at the prompt rule minus: N1 at the training rule %s | single E at the prompt rule %s | no write %s | prompt only %s%s" % (
            d(n_p[0], n_c[0]), d(n_p[0], e_p[0]), d(n_p[0], base[0]), d(n_p[0], da_s),
            "" if not best else " | the best prompt on tune %s" % d(n_p[0], best[1])))
        print("    single E at the prompt rule minus at the training rule %s" % d(e_p[0], e_c[0]))


def ro_name(pick):
    return ML.ro.name_of(pick)


if __name__ == "__main__":
    main()
