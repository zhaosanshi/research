# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# TruthfulQA free-form, test split, normal prompt, DeepSeek Pro, score = right - wrong: the learned E direction (DAS)
# against the mean-difference direction written at the same layer, strengths 1 and 2, five models.
# Kinds of the engine (e44_eval_extra.py): w = the learned direction; mdg = the mean-difference direction at the same
# nominal amount as w; md = the mean-difference direction with its own gap. The mean-difference answers come from the
# work dirs judge_test_mdg_test / judge_test_md_test (launcher logs/newsets_20261119/pm_all.sh); the learned-direction
# answers and the no-write answers from the model's existing E+C work dir of the test split.
# Second part (when logs/newsets_20261119/md_sweep.sh has run): the mean-difference baseline at ITS OWN best layer and
# coefficient, chosen on tune among the swept layers and coefficients 1 / 2 / 4 (diag/md_choose.py), against the learned
# direction at its own layer with the strength chosen on tune among the strengths the model has (single E), and against
# N1 at that strength.
# Several models together: per question the differences are summed and questions are resampled.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/md_table.py
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import dose_mid as DM
import main_table_same_n as MT

ps, ro = MT.ps, MT.ro
S, CAPS = DM.S, MT.CAPS


def val(x):
    return (x == "R") - (x == "W")


def row(label, st, bl, qs, c):
    s = [st[(q, c)] for q in qs]
    return "    %-34s answers %3d | right %3d wrong %3d | net %+4d" % (
        label, sum(not bl[(q, c)] for q in qs), s.count("R"), s.count("W"), sum(map(val, s)))


def main():
    tot = {}
    for model in ro.FIVE:
        D = MT.Data(model, 1)
        sp = D.sp
        run = D.m["ec"][1].split("/eval_extra/")[0]
        el = D.m["e"]
        st, bl = dict(sp.st), dict(sp.blank)
        missing = []
        for kind in ("mdg", "md"):
            w = "%s/eval_extra/judge_test_%s_test" % (run, kind)
            if not os.path.exists(w + "/judge2/single_pro.jsonl"):
                missing.append(kind)
                continue
            st1, bl1 = ps.load(w, "pro")
            st.update(st1)
            bl.update(bl1)
        conds = {("w", s): sp.cE(s) for s in "12"}
        for kind in ("mdg", "md"):
            for s in "12":
                conds[(kind, s)] = "E_L%d_x-%s-p%s" % (el, kind, s)
        have = {k: c for k, c in conds.items() if all((q, c) in st for q in sp.qs)}
        qs = [q for q in sp.qs if all(st[(q, c)] != "?" for c in have.values())]
        print("\n## %s: %d questions (of %d)%s" % (model, len(qs), len(sp.qs), "" if not missing else "; no verdicts for " + ", ".join(missing)))
        print(row("no write", st, bl, qs, "baseline_x"))
        for s in "12":
            for kind, name in (("w", "learned direction (DAS)"), ("mdg", "mean difference, same amount"), ("md", "mean difference, own gap")):
                if (kind, s) in have:
                    print(row("E %s: %s" % (s, name), st, bl, qs, have[(kind, s)]))
            for kind in ("mdg", "md"):
                if (kind, s) in have and ("w", s) in have:
                    d = [val(st[(q, have[("w", s)])]) - val(st[(q, have[(kind, s)])]) for q in qs]
                    b = [val(st[(q, have[(kind, s)])]) - val(st[(q, "baseline_x")]) for q in qs]
                    print("    E %s: learned minus %s %s | %s minus no write %s" % (s, kind, ps.ci(d), kind, ps.ci(b)))
                    for key, v in (("E %s: learned minus %s" % (s, kind), d), ("E %s: %s minus no write" % (s, kind), b)):
                        t = tot.setdefault(key, [0, {}])
                        t[0] += 1
                        for q, x in zip(qs, v):
                            t[1][q] = t[1].get(q, 0) + x
            if ("w", s) in have:
                b = [val(st[(q, have[("w", s)])]) - val(st[(q, "baseline_x")]) for q in qs]
                print("    E %s: learned minus no write %s" % (s, ps.ci(b)))
                t = tot.setdefault("E %s: learned minus no write" % s, [0, {}])
                t[0] += 1
                for q, x in zip(qs, b):
                    t[1][q] = t[1].get(q, 0) + x
    print("\n## the five models together (per question the differences of the models are summed; questions resampled)")
    for key, (n, d) in tot.items():
        if n == len(ro.FIVE):
            print("  %-40s %s (%d questions)" % (key, ps.ci([d[q] for q in sorted(d)]), len(d)))
    best_layer()


def best_layer():
    tot, done = {}, 0
    print("\n\n# the mean-difference baseline at its own best layer and coefficient (chosen on tune)")
    for model in ro.FIVE:
        Dt, tsp, ttx, _, _, _ = DM.merged_all(model, 0)
        Dx, xsp, xtx, doses, _, _ = DM.merged_all(model, 1)
        run = Dx.m["ec"][1].split("/eval_extra/")[0]
        cpath, w = run + "/eval_extra/mdsweep_choice.json", run + "/eval_extra/judge_test_mdbest_test"
        if not (os.path.exists(cpath) and os.path.exists(w + "/judge2/single_pro.jsonl")):
            print("\n## %s: sweep or its test verdicts not there yet" % model)
            continue
        with open(cpath, encoding="utf-8") as f:
            ch = json.load(f)
        c = ch["choice"]
        st, bl = ps.load(w, "pro")
        cond = c["condition"]
        qs = [q for q in xsp.qs if st.get((q, cond), "?") != "?"]
        idx = [i for i, q in enumerate(xsp.qs) if q in set(qs)]
        md = [st[(q, cond)] for q in qs]
        e_dose = max(doses, key=lambda s: (ps.score(S.system(tsp, ttx, ("E", s))[0], 1), -doses.index(s)))
        cands = [("E", e_dose)] + [("N1", e_dose, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
        pick = max(cands, key=lambda x: (ps.score(S.system(tsp, ttx, x)[0], 1), -cands.index(x)))
        e_all, n_all, b_all = S.system(xsp, xtx, ("E", e_dose))[0], S.system(xsp, xtx, pick)[0], S.system(xsp, xtx, ("base",))[0]
        e, n1, base = [e_all[i] for i in idx], [n_all[i] for i in idx], [b_all[i] for i in idx]
        rows = sorted(ch["rows"], key=lambda r: -r["net"])
        print("\n## %s: learned direction at layer %d; mean difference swept over %d layers x %d coefficients on tune (no write on tune: %s)" % (
            model, ch["das_layer"], len({r["layer"] for r in ch["rows"]}), len({r["strength"] for r in ch["rows"]}), ch["no_write_net"]))
        print("    tune, best five: " + " | ".join("L%d x%g: %+d" % (r["layer"], r["strength"], r["net"]) for r in rows[:5]))
        print("    chosen: layer %d, coefficient %g. test (%d questions): answers %d | right %d wrong %d | net %+d" % (
            c["layer"], c["strength"], len(qs), sum(not bl[(q, cond)] for q in qs), md.count("R"), md.count("W"), sum(map(val, md))))
        d_e = [val(x) - val(y) for x, y in zip(e, md)]
        d_n = [val(x) - val(y) for x, y in zip(n1, md)]
        d_b = [val(x) - val(y) for x, y in zip(md, base)]
        print("    learned single E at strength %s (chosen on tune): net %+d | minus the mean difference %s" % (DM.show(e_dose), sum(map(val, e)), ps.ci(d_e)))
        print("    N1 at that strength: net %+d | minus the mean difference %s | the mean difference minus no write %s" % (
            sum(map(val, n1)), ps.ci(d_n), ps.ci(d_b)))
        done += 1
        for key, v in (("learned single E minus mean difference (each at its own best on tune)", d_e),
                       ("N1 minus mean difference at its own best", d_n), ("mean difference at its own best minus no write", d_b)):
            t = tot.setdefault(key, {})
            for q, x in zip(qs, v):
                t[q] = t.get(q, 0) + x
    if done == len(ro.FIVE):
        print("\n## the five models together")
        for key, d in tot.items():
            print("  %-72s %s (%d questions)" % (key, ps.ci([d[q] for q in sorted(d)]), len(d)))


if __name__ == "__main__":
    main()
