# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# TruthfulQA free-form, DeepSeek Pro, score = right - wrong: E written on top of the prompt-only baseline (prompt
# conf_E2 = must-answer + the honest sentence) at several strengths, against that prompt with nothing written, model by
# model. Work dirs judge_<split>_<NAME> of logs/newsets_20261119/idea_a.sh, idea_a2.sh and pm_all.sh; the baseline is the
# condition baseline_conf_E2 of the must-answer work dir of the same split (final_records.py, system ("DA",)).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/on_prompt_table.py SPLIT NAME [NAME ...]
#   e.g. test ideaA4_test ideaA4b_test | tune ideaA6_tune
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_same_n as MT

ps, ro = MT.ps, MT.ro


def val(x):
    return (x == "R") - (x == "W")


def main():
    split, names = sys.argv[1], sys.argv[2:]
    i = ("tune", "test").index(split)
    tot = {}
    for model in ro.FIVE:
        D = MT.Data(model, i)
        run = D.m["ec"][i].split("/eval_extra/")[0]
        base_s, base_a = MT.handed_in(D, ("DA",))
        base = dict(zip(D.sp.qs, base_s))
        print("\n## %s, %s: prompt only (must-answer + the honest sentence): net %+d, answers %d of %d" % (
            model, split, sum(val(x) for x in base_s), sum(base_a), len(base_s)))
        for name in names:
            w = "%s/eval_extra/judge_%s_%s" % (run, split, name)
            if not os.path.exists(w + "/judge2/single_pro.jsonl"):
                print("    %s: no Pro verdicts" % name)
                continue
            st, bl = ps.load(w, "pro")
            conds = sorted({c for _, c in st}, key=lambda c: float(c.split("-p")[-1].replace("-", ".")) if "-p" in c else -1)
            for c in conds:
                qs = [q for q in D.sp.qs if (q, c) in st]
                d = [val(st[(q, c)]) - val(base[q]) for q in qs]
                und = sum(st[(q, c)] == "?" for q in qs)
                print("    %-28s net %+4d | right %3d wrong %3d | blank %3d | minus prompt only %s | %d questions%s" % (
                    c, sum(val(st[(q, c)]) for q in qs), sum(st[(q, c)] == "R" for q in qs), sum(st[(q, c)] == "W" for q in qs),
                    sum(bl[(q, c)] for q in qs), ps.ci(d), len(qs), "" if not und else " | %d without verdict" % und))
                key = c.split("-p")[-1] if "-p" in c else c
                t = tot.setdefault(key, {})
                for q, x in zip(qs, d):
                    t.setdefault(q, []).append(x)
    print("\n## all five models together where every model has the strength (per question summed; questions resampled)")
    for key, t in tot.items():
        full = [sum(v) for v in t.values() if len(v) == len(ro.FIVE)]
        if full:
            print("  strength %-6s minus prompt only %s (%d questions)" % (key.replace("-", "."), ps.ci(full), len(full)))


if __name__ == "__main__":
    main()
