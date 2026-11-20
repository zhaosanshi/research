# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# TruthfulQA free-form, DeepSeek Pro, score = right - wrong: every prompt of a run as a prompt-only baseline (no write),
# tune and test (work dirs judge_<split>_promptsweep_<split> of logs/newsets_20261119/prompt_sweep.sh; the answers are
# the run's frozen no-write answers). Prompt names: x = neutral, conf = must-answer, caut = cautious base; _E2 = with
# the honest sentence the E direction was trained on, _E1 = an earlier honest sentence, _F = "answer in a complete
# sentence", _M = the two-pole sentence.
# Per model: the best prompt on tune (ties to conf_E2, the baseline used so far), then N1 at every E strength the model
# has (caps chosen on tune, as diag/dose_mid.py) against that prompt on test; the prompt that happens to be best on test
# is printed for reference only. Several models together: per question the differences are summed, questions resampled.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/prompt_sweep_table.py [MODEL ...]
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import dose_mid as DM

MT, S = DM.MT, DM.S
ps, ro = MT.ps, MT.ro
CAPS = MT.CAPS


def val(x):
    return (x == "R") - (x == "W")


def prompts_of(run, split, qs):
    w = "%s/eval_extra/judge_%s_promptsweep_%s" % (run, split, split)
    if not os.path.exists(w + "/judge2/single_pro.jsonl"):
        return None
    st, bl = ps.load(w, "pro")
    out = {}
    for c in sorted({c for _, c in st if c.startswith("baseline_")}):
        if all(st.get((q, c), "?") != "?" for q in qs):
            out[c[len("baseline_"):]] = ([st[(q, c)] for q in qs], sum(not bl[(q, c)] for q in qs))
    return out


def main():
    names = sys.argv[1:] or list(ro.FIVE)
    tot = {}
    for model in names:
        Dt, tsp, ttx, _, _, _ = DM.merged_all(model, 0)
        Dx, xsp, xtx, doses, _, _ = DM.merged_all(model, 1)
        run = Dx.m["ec"][1].split("/eval_extra/")[0]
        tune, test = prompts_of(run, "tune", tsp.qs), prompts_of(run, "test", xsp.qs)
        if not tune or not test:
            print("\n## %s: prompt verdicts missing (tune %s, test %s)" % (model, bool(tune), bool(test)))
            continue
        order = sorted(tune, key=lambda p: (-sum(map(val, tune[p][0])), p != "conf_E2", p))
        best = order[0]
        print("\n## %s: tune %d questions, test %d" % (model, len(tsp.qs), len(xsp.qs)))
        for p in order:
            t = tune[p]
            x = test.get(p)
            print("    %-9s tune net %+4d (answers %3d) | test %s%s" % (
                p, sum(map(val, t[0])), t[1], "net %+4d (answers %3d)" % (sum(map(val, x[0])), x[1]) if x else "no complete verdicts",
                "   <- best on tune" if p == best else ""))
        if best not in test or "conf_E2" not in test:
            print("    the chosen prompt has no complete test verdicts")
            continue
        top_test = max(test, key=lambda p: sum(map(val, test[p][0])))
        b, c2, o = test[best][0], test["conf_E2"][0], test[top_test][0]
        print("    chosen on tune: %s (test net %+d); conf_E2 test net %+d; best on test (reference only): %s %+d" % (
            best, sum(map(val, b)), sum(map(val, c2)), top_test, sum(map(val, o))))
        for s in doses:
            cands = [("E", s)] + [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
            pick = max(cands, key=lambda x: (ps.score(S.system(tsp, ttx, x)[0], 1), -cands.index(x)))
            n1 = S.system(xsp, xtx, pick)[0]
            d_b = [val(x) - val(y) for x, y in zip(n1, b)]
            d_c = [val(x) - val(y) for x, y in zip(n1, c2)]
            d_o = [val(x) - val(y) for x, y in zip(n1, o)]
            print("    E %-4s N1 net %+4d | minus the prompt chosen on tune %s | minus conf_E2 %s | minus the best on test %s" % (
                DM.show(s), sum(map(val, n1)), ps.ci(d_b), ps.ci(d_c), ps.ci(d_o)))
            if s in DM.COMMON:
                t = tot.setdefault("E %s: N1 minus each model's prompt chosen on tune" % DM.show(s), [0, {}])
                t[0] += 1
                for q, v in zip(xsp.qs, d_b):
                    t[1][q] = t[1].get(q, 0) + v
    print("\n## the models above together (per question the differences of the models are summed; questions resampled)")
    for key, (n, d) in tot.items():
        if n == len(names):
            print("  %-58s %s (%d questions)" % (key, ps.ci([d[q] for q in sorted(d)]), len(d)))


if __name__ == "__main__":
    main()
