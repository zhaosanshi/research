# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# TruthfulQA free-form, test split, DeepSeek Pro, score = right - wrong: single E and N1 at every E strength a model has,
# model by model (the lead, 2026-11-20 08:33: list every model and try the strengths between 1 and 2; 15:24: the method
# is shared, the strength may differ by model).
# Strengths 1 and 2 come from the existing E+C work dirs; the others from the work dirs judge_<split>_<tag>_<split> of
# the same runs (same engine, same judge; launchers logs/newsets_20261119/dose_mid_all.sh, q9_high.sh, pm_all.sh):
# dosemid = 1.25 / 1.5 / 1.75 (five models), dosehigh = 2.5 / 3, dosehigh2 = 3.5 / 4 (only the models that have them,
# judged on both splits). For every strength the two caps of N1 are chosen on tune among the candidates of
# main_table_same_n.py restricted to that strength, by the same rule; nothing is chosen on test. The per-question
# records come from final_records.py unchanged (through newsets_score.NewSplit, a Split over a supplied verdict table).
# Reference systems: no write; the prompt-only baseline (must-answer prompt + the honest sentence, nothing written).
# Last section: one strength per model, chosen on tune (the strength whose N1 has the highest tune score); test numbers
# of that choice. Several models together: per question the differences are summed and questions are resampled.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/dose_mid.py [--names] [MODEL ...]
#   --names  only check that every needed condition exists in the cells of the new work dirs (before judging)
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_same_n as MT
import newsets_score as S

ps, ro, fr = MT.ps, MT.ro, MT.fr
CAPS, LAD = MT.CAPS, ps.LAD
BASE = ("1", "2")
TAGS = (("dosemid", ("1-25", "1-5", "1-75")), ("dosehigh", ("2-5", "3")), ("dosehigh2", ("3-5", "4")))
COMMON = ("1", "1-25", "1-5", "1-75", "2")
MAIN = ("Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B")


def show(s):
    return s.replace("-", ".")


def new_work(D, i, tag):
    run = D.m["ec"][i].split("/eval_extra/")[0]
    split = ("tune", "test")[i]
    return "%s/eval_extra/judge_%s_%s_%s" % (run, split, tag, split)


def tags_of(D, judged=True):
    """The extra work dirs of this model that exist for both splits (with Pro verdicts if judged)."""
    f = "/judge2/single_pro.jsonl" if judged else "/judge2/cells.jsonl"
    return [(tag, ds) for tag, ds in TAGS if all(os.path.exists(new_work(D, i, tag) + f) for i in (0, 1))]


def needed(sp0, doses):
    return [c for s in doses for c in [sp0.cE(s)] + [sp0.cEC(s, k) for k in LAD]]


def check_names(name):
    for i in (0, 1):
        D = MT.Data(name, i)
        for tag, ds in tags_of(D, judged=False):
            w = new_work(D, i, tag)
            have = set()
            with open(w + "/judge2/cells.jsonl", encoding="utf-8") as f:
                for line in f:
                    have.add(json.loads(line)["condition"])
            miss = [c for c in needed(D.sp, ds) + ["baseline_x"] if c not in have]
            print("%s %s: %d conditions in %s; missing %s" % (name, ("tune", "test")[i], len(have), w.split("/")[-1], miss or "none"))


def merged(name, i):
    """(old Data, Split over old + new verdicts, texts, strengths, cells present in two dirs, cells whose verdict differs)."""
    D = MT.Data(name, i)
    sp0 = D.sp
    st, bl = dict(sp0.st), dict(sp0.blank)
    tx = dict(fr.raw_texts(D.m["ec"][i]))
    doses, both, diff = list(BASE), 0, 0
    for tag, ds in tags_of(D):
        w = new_work(D, i, tag)
        st1, bl1 = ps.load(w, "pro")
        both += sum(k in st for k in st1)
        diff += sum(k in st and st1[k] != st[k] for k in st1)
        st.update(st1)
        bl.update(bl1)
        tx.update(fr.raw_texts(w))
        doses += ds
    doses.sort(key=lambda s: float(show(s)))
    sp = S.NewSplit(D.m["e"], D.m["c"], st, bl, ["baseline_x"] + needed(sp0, doses), qs=sp0.qs)
    return D, sp, fr.Texts(tx, tx), doses, both, diff


def val(states):
    return [(x == "R") - (x == "W") for x in states]


def one(name, tots, table):
    Dt, tsp, ttx, tdoses, _, _ = merged(name, 0)
    Dx, xsp, xtx, doses, both, diff = merged(name, 1)
    if tdoses != doses:
        raise SystemExit("%s: tune has strengths %s, test %s" % (name, tdoses, doses))
    print("\n## %s: test %d questions (old work dir %d), tune %d; cells judged in two work dirs %d, verdict differs in %d" % (
        name, len(xsp.qs), len(Dx.sp.qs), len(tsp.qs), both, diff))
    base = S.system(xsp, xtx, ("base",))
    print(MT.line("no write", base[0], base[1]))
    da0 = MT.handed_in(Dx, ("DA",))
    by = dict(zip(Dx.sp.qs, zip(da0[0], da0[1])))
    da = ([by[p][0] for p in xsp.qs], [by[p][1] for p in xsp.qs])
    print(MT.line("prompt only: must-answer + the honest sentence, nothing written", *da))

    def add(key, v):
        for t in tots:
            n, d = t.setdefault(key, [0, {}])
            t[key][0] = n + 1
            for p, x in zip(xsp.qs, v):
                d[p] = d.get(p, 0) + x

    for s in doses:
        cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        pick = max(cands, key=lambda x: (ps.score(S.system(tsp, ttx, x)[0], 1), -cands.index(x)))
        tune_net = ps.score(S.system(tsp, ttx, pick)[0], 1)
        e = S.system(xsp, xtx, ("E", s))
        n1 = S.system(xsp, xtx, pick)
        print(MT.line("E %s: single E" % show(s), e[0], e[1]))
        print(MT.line("E %s: N1 (on tune: %s; tune net %d)" % (show(s), ro.name_of(pick), tune_net), n1[0], n1[1]))
        d_base = [x - y for x, y in zip(val(n1[0]), val(base[0]))]
        d_e = [x - y for x, y in zip(val(n1[0]), val(e[0]))]
        d_da = [x - y for x, y in zip(val(n1[0]), val(da[0]))]
        print("    E %s: N1 minus no write %s | N1 minus single E %s | N1 minus prompt only %s" % (
            show(s), ps.ci(d_base), ps.ci(d_e), ps.ci(d_da)))
        if s in COMMON:
            add("E %s: N1 minus no write" % show(s), d_base)
            add("E %s: N1 minus single E" % show(s), d_e)
            add("E %s: N1 minus prompt only" % show(s), d_da)
        table.setdefault(name, {})[s] = dict(net=sum(val(n1[0])), ans=sum(n1[1]), tune=tune_net, d_da=d_da, d_base=d_base, d_e=d_e)
    best = max(doses, key=lambda s: (table[name][s]["tune"], -doses.index(s)))
    table[name]["best"] = best
    for key in ("d_base", "d_e", "d_da"):
        add("strength chosen per model on tune: N1 %s" % {"d_base": "minus no write", "d_e": "minus single E", "d_da": "minus prompt only"}[key],
            table[name][best][key])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--names"]
    names = args or list(ro.FIVE)
    if "--names" in sys.argv[1:]:
        for n in names:
            check_names(n)
        sys.exit(0)
    tot, tot3, table = {}, {}, {}
    for n in names:
        one(n, (tot, tot3) if n in MAIN else (tot,), table)
    print("\n## N1 minus the prompt-only baseline, by model and E strength (net on test; in brackets N1's test net, test answers, tune net)")
    for n in names:
        ds = [s for s in table[n] if s != "best"]
        print("  %-14s %s" % (n, " | ".join("E %s: %+d (%d, %d, tune %d)" % (
            show(s), sum(table[n][s]["d_da"]), table[n][s]["net"], table[n][s]["ans"], table[n][s]["tune"]) for s in ds)))
    print("\n## one strength per model, chosen on tune (highest tune score of N1; ties to the lower strength)")
    for n in names:
        b = table[n]["best"]
        r = table[n][b]
        print("  %-14s E %-4s | N1 test net %4d, answers %3d | minus no write %s | minus single E %s | minus prompt only %s" % (
            n, show(b), r["net"], r["ans"], ps.ci(r["d_base"]), ps.ci(r["d_e"]), ps.ci(r["d_da"])))
    for title, t, k in (("the models above together", tot, len(names)), ("the main-text models among them", tot3, sum(n in MAIN for n in names))):
        print("\n## %s (per question the differences of the models are summed; questions resampled)" % title)
        for key, (n_models, d) in t.items():
            if n_models == k:
                print("  %-58s %s (%d questions)" % (key, ps.ci([d[p] for p in sorted(d)]), len(d)))
