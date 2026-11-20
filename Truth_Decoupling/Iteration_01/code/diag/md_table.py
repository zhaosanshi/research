# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# TruthfulQA free-form, test split, normal prompt, DeepSeek Pro, score = right - wrong: the learned E direction (DAS)
# against the mean-difference direction written at the same layer, strengths 1 and 2, five models.
# Kinds of the engine (e44_eval_extra.py): w = the learned direction; mdg = the mean-difference direction at the same
# nominal amount as w; md = the mean-difference direction with its own gap. The mean-difference answers come from the
# work dirs judge_test_mdg_test / judge_test_md_test (launcher logs/newsets_20261119/pm_all.sh); the learned-direction
# answers and the no-write answers from the model's existing E+C work dir of the test split.
# Several models together: per question the differences are summed and questions are resampled.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/md_table.py
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_same_n as MT

ps, ro = MT.ps, MT.ro


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


if __name__ == "__main__":
    main()
