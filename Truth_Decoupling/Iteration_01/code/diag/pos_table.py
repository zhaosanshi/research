# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# TruthfulQA free-form, test split, DeepSeek Pro, score = right - wrong: the learned E written only from the last prompt
# token on (core rule "gen"; work dir judge_test_posgen_test of logs/newsets_20261119/pos_all.sh, made through
# scripts/review_checks/e44_positions.py) against the same write under the training rule "content" (first user-content
# token .. end), against no write and against the prompt-only baseline (must-answer + the honest sentence).
# The condition names of the two rules are identical inside their own work dirs; here the "gen" cells are read from
# their own dir only and never merged into the tables of the content rule.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/pos_table.py [MODEL ...]
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_same_n as MT

ps, ro = MT.ps, MT.ro


def val(x):
    return (x == "R") - (x == "W")


def line(label, s, b):
    return "    %-46s answers %3d | right %3d wrong %3d | net %+4d" % (label, sum(not x for x in b), s.count("R"), s.count("W"), sum(map(val, s)))


def main():
    names = sys.argv[1:] or list(ro.FIVE)
    tot = {}
    for model in names:
        D = MT.Data(model, 1)
        sp, el = D.sp, D.m["e"]
        run = D.m["ec"][1].split("/eval_extra/")[0]
        w = run + "/eval_extra/judge_test_posgen_test"
        if not os.path.exists(w + "/judge2/single_pro.jsonl"):
            print("\n## %s: no Pro verdicts for the gen rule yet" % model)
            continue
        gst, gbl = ps.load(w, "pro")
        on = {}                                             # content rule on top of the prompt (idea_a.sh / idea_a2.sh)
        for name in ("ideaA4_test", "ideaA4b_test"):
            p = "%s/eval_extra/judge_test_%s" % (run, name)
            if os.path.exists(p + "/judge2/single_pro.jsonl"):
                on.update(ps.load(p, "pro")[0])
        gconds = sorted({c for _, c in gst if c.startswith("E_L")}, key=lambda c: ("conf" in c, float(c.split("-p")[-1].replace("-", "."))))
        qs = [q for q in sp.qs if all(gst.get((q, c), "?") != "?" for c in gconds)]
        da0 = MT.handed_in(D, ("DA",))
        da = dict(zip(sp.qs, da0[0]))
        base = [sp.st[(q, "baseline_x")] for q in qs]
        print("\n## %s: %d questions (of %d)" % (model, len(qs), len(sp.qs)))
        print(line("no write", base, [sp.blank[(q, "baseline_x")] for q in qs]))
        print("    %-46s net %+4d" % ("prompt only (must-answer + sentence)", sum(val(da[q]) for q in qs)))
        for c in gconds:
            s = c.split("-p")[-1]
            g = [gst[(q, c)] for q in qs]
            on_prompt = "_conf_E2-" in c
            print(line("%s E %s, gen rule" % ("on the prompt," if on_prompt else "normal prompt,", s.replace("-", ".")), g, [gbl[(q, c)] for q in qs]))
            ref = [da[q] for q in qs] if on_prompt else base
            msg = "        minus %s %s" % ("prompt only" if on_prompt else "no write", ps.ci([val(x) - val(y) for x, y in zip(g, ref)]))
            key = ("on prompt " if on_prompt else "normal ") + s
            t = tot.setdefault(key + ": gen rule minus " + ("prompt only" if on_prompt else "no write"), [0, {}])
            t[0] += 1
            for q, x, y in zip(qs, g, ref):
                t[1][q] = t[1].get(q, 0) + val(x) - val(y)
            if not on_prompt:
                msg += " | minus prompt only %s" % ps.ci([val(x) - val(da[q]) for x, q in zip(g, qs)])
                cc = "E_L%d_x-w-p%s" % (el, s)
                if all((q, cc) in sp.st for q in qs):
                    k = [sp.st[(q, cc)] for q in qs]
                    msg += " | content rule at the same strength: answers %d net %+d; gen minus content %s" % (
                        sum(not sp.blank[(q, cc)] for q in qs), sum(map(val, k)), ps.ci([val(x) - val(y) for x, y in zip(g, k)]))
            elif all((q, c) in on for q in qs):
                k = [on[(q, c)] for q in qs]
                msg += " | content rule at the same strength: net %+d; gen minus content %s" % (
                    sum(map(val, k)), ps.ci([val(x) - val(y) for x, y in zip(g, k)]))
            print(msg)
    print("\n## the models above together where every model has the cell (per question summed; questions resampled)")
    for key, (n, d) in tot.items():
        if n == len(names):
            print("  %-48s %s (%d questions)" % (key, ps.ci([d[q] for q in sorted(d)]), len(d)))


if __name__ == "__main__":
    main()
