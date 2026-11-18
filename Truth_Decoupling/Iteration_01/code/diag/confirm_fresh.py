# Iteration_01 analysis (Claude Fable 5.2, 2026-11-18), read-only, no model, no judge call.
# Confirmation of the schemes "two caps" and N1 on test questions that took no part in devising them.
# 赵磊 2026-11-18: adopt the new scheme and have the remaining 425 test questions of the two Gemma-3 models judged by DeepSeek
# Pro (off-peak hours). Until now Pro had judged only the Dr's 100 test questions (98 used) of these two models; the two
# schemes were devised after looking at those 98 (and at the 523 of Llama / Qwen). The other test questions are fresh for Pro.
# Settings are the ones chosen on the tune split (Pro), exactly as before; nothing is re-chosen here.
# Reported for each model on: fresh = test questions outside the Dr's 100; old = the 98; all = both.
# Predictions written before the Pro verdicts existed (Fable 5.2, 2026-11-18 08:5x). The AllenAI verdicts of the same
# questions were known (N1 minus single E on all 523: +93 [+71, +116] and +52 [+37, +68]), so F1 is not a blind guess:
#   F1  on the fresh questions, N1 minus single E has an interval above 0 for both models
#   F2  on the fresh questions, N1 is not more than 5 points below the method for either model
#   F3  fresh, the two models pooled: of E's own blanks, the answers released as the ORIGINAL answer are right less often
#       than those released as a NEW answer
#   F4  fresh, pooled: released own blanks are right more often at dose 0.25 / 0.5 than at dose 1 / 2
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/confirm_fresh.py [--judge allenai]
#   (--judge allenai is a dry run of the code on the AllenAI verdicts; the confirmation itself is the default, pro)
import collections
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import two_caps as tc
import release_origin as ro

JUDGE = "allenai" if "--judge" in sys.argv and sys.argv[sys.argv.index("--judge") + 1] == "allenai" else "pro"
G3 = "Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b"
G27 = "Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b"
FULL = {"Gemma-3-12B": (G3 + "/eval_extra/judge_test_t25_test", G3 + "/judge_test_testD_E_L22"),
        "Gemma-3-27B": (G27 + "/eval_extra/judge_test_t31_test", G27 + "/judge_test_testD_E_L26")}
CAPS = ro.CAPS


def pb_own(sp, s):
    e = sp.cE(s)
    return [sp.dst[(p, sp.dE(s))] if (sp.blank[(p, "baseline_x")] and sp.blank[(p, e)]) else sp.st[(p, e)] for p in sp.qs]


def main():
    print("# verdicts of the full test split: %s" % JUDGE)
    pooled = collections.defaultdict(list)
    for name, (ec_full, d_full) in FULL.items():
        m = cf.MODELS[name]
        tune = ps.Split(m, 0)                                   # tune, Pro: the settings come from here, as before
        old = ps.Split(m, 1)                                    # the Dr's 98, Pro
        full = ps.Split(dict(m, ec=(m["ec"][0], ec_full), d=(m["d"][0], d_full), judge=(m["judge"][0], JUDGE)), 1)
        ttx, ftx = ro.texts(m["ec"][0]), ro.texts(ec_full)
        old_ids = set(old.qs)
        idx = {"fresh": [i for i, p in enumerate(full.qs) if p not in old_ids],
               "old": [i for i, p in enumerate(full.qs) if p in old_ids],
               "all": list(range(len(full.qs)))}
        fam = {
            "single E": [("E", "1"), ("E", "2")],
            "single C": [("C", k) for k in ps.LAD if k in tune.cks and k in full.cks],
            "method": [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in ps.LAD],
            "two caps": tc.FAM["two caps"],
            "N1": [("E", s) for s in "12"] + [("N1", s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)],
        }
        pick = {f: max(c, key=lambda x: (ps.score(ro.states(tune, ttx, x), 1), -c.index(x))) for f, c in fam.items()}
        row = {f: ro.states(full, ftx, pick[f]) for f in fam}
        fB = [("B", "1"), ("B", "2")]
        pB = max(fB, key=lambda x: (ps.score(tune.states(x), 1), -fB.index(x)))
        sO = max("12", key=lambda s: (ps.score(pb_own(tune, s), 1), -int(s)))
        row["prompt B"] = full.states(pB)
        row["prompt B own"] = pb_own(full, sO)
        print("\n## %s: %d test questions with a complete verdict (%d fresh, %d old); settings: %s" % (
            name, len(full.qs), len(idx["fresh"]), len(idx["old"]), "; ".join("%s = %s" % (f, ro.name_of(pick[f])) for f in fam)))
        for part in ("fresh", "old", "all"):
            sel = idx[part]
            sub = {f: [v[i] for i in sel] for f, v in row.items()}
            print("  %-5s (%3d): " % (part, len(sel)) + " | ".join("%s %d" % (f, ps.score(sub[f], 1)) for f in sub))
            d = lambda a, b: ps.ci([x - y for x, y in zip(cf.val(sub[a]), cf.val(sub[b]))])
            print("        N1 minus single E %s | minus single C %s | minus method %s | minus prompt B %s | minus prompt B own %s" % (
                d("N1", "single E"), d("N1", "single C"), d("N1", "method"), d("N1", "prompt B"), d("N1", "prompt B own")))
            print("        two caps minus single E %s | method minus single E %s" % (d("two caps", "single E"), d("method", "single E")))
        s = pick["method"][1]
        fresh_ids = {full.qs[i] for i in idx["fresh"]}
        for p, origin, k, st, whose in ro.released(full, ftx, s):
            if p in fresh_ids:
                pooled[(origin, "dose", k)].append(st)
                pooled[(origin, "whose", whose)].append(st)
    print("\n## fresh questions, the two models pooled: released answers, right / wrong")
    for origin in ("own", "E's"):
        print("  %-4s blanks by dose:  " % origin + " | ".join("%s %s" % (ps.NAME[k], ro.fmt(pooled[(origin, "dose", k)])) for k in ps.LAD))
        print("  %-4s blanks by whose: " % origin + " | ".join("%s %s" % (w, ro.fmt(pooled[(origin, "whose", w)])) for w in ("original", "new")))


if __name__ == "__main__":
    main()
