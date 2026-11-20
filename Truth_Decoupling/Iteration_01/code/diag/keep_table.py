# Iteration_01 analysis (Claude Fable 5.2, 2026-11-20), read-only, no model, no judge call.
# The E direction trained with multiple-choice items among its training questions (logs/newsets_20261119/keep_pilot.sh;
# run Iteration_01/results/ff_keep/<model>/e44/<model>_keep) next to the model's current E direction:
#   1. the training losses of both directions on their held-out training questions (from the training logs);
#   2. TruthfulQA, DeepSeek Pro, score = right - wrong, single E on the normal prompt at each strength, tune and test:
#      new direction against no write, against the current direction at the same strength, against the prompt-only
#      baseline; and the must-answer prompt + E at strengths 1 and 2 against the current direction;
#   3. the MMLU numbers are printed by diag/cap_mcq_fresh.py on capability/keep/<model>.json (not here).
# The two directions have different gaps: "strength s" is s times each direction's own gap, as everywhere.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/keep_table.py [MODEL ...]
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import dose_mid as DM

MT, S = DM.MT, DM.S
ps, ro = MT.ps, MT.ro
NG = {"Llama-3.1-8B": "llama3.1-8b", "Qwen3.5-9B": "qwen3.5-9b", "Qwen3.8-27B": "qwen3.8-27b", "Gemma-3-12B": "gemma3-12b", "Gemma-3-27B": "gemma3-27b"}


def val(x):
    return (x == "R") - (x == "W")


def losses(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    v = d["losses"]["val"]
    return "gap %.3f, cos(learned, mean difference) %.2f | held-out cause loss: no write %.4f, learned %.4f (closes %.0f%%) | isolate: no write %.5f, learned %.5f | %d training questions, %.0f s" % (
        d["gap"], d["cosine_w_md"], v["no_write"]["cause"], v["trained"]["cause"], 100 * (1 - v["trained"]["cause"] / v["no_write"]["cause"]),
        v["no_write"]["isolate"], v["trained"]["isolate"], len(d["training_contract"]["train_ids"]), d["seconds"])


def main():
    for model in sys.argv[1:] or ["Llama-3.1-8B"]:
        ng = NG[model]
        new = "Iteration_01/results/ff_keep/%s/e44/%s_keep" % (ng, ng)
        for i, split in ((0, "tune"), (1, "test")):
            D, sp, tx, doses, _, _ = DM.merged_all(model, i)
            el = D.m["e"]
            if i == 0:
                old_run = D.m["ec"][0].split("/eval_extra/")[0]
                print("\n## %s, E layer %d" % (model, el))
                print("  current direction: " + losses("%s/train_logs/E_L%d.json" % (old_run, el)))
                p = "%s/train_logs/E_L%d.json" % (new, el)
                if os.path.exists(p):
                    print("  new direction:     " + losses(p))
            w = "%s/eval_extra/judge_%s_keep_%s" % (new, split, split)
            if not os.path.exists(w + "/judge2/single_pro.jsonl"):
                print("  %s: no Pro verdicts yet" % split)
                continue
            st, bl = ps.load(w, "pro")
            conds = sorted({c for _, c in st if c.startswith("E_L")}, key=lambda c: ("_conf-" in c, float(c.split("-p")[-1].replace("-", "."))))
            qs = [q for q in sp.qs if all(st.get((q, c), "?") != "?" for c in conds)]
            idx = [k for k, q in enumerate(sp.qs) if q in set(qs)]
            base_all = S.system(sp, tx, ("base",))[0]
            base = [base_all[k] for k in idx]
            da0 = MT.handed_in(D, ("DA",))
            da = dict(zip(D.sp.qs, da0[0]))
            print("  %s, %d questions: no write net %+d (answers %d) | prompt only net %+d" % (
                split, len(qs), sum(map(val, base)), sum(not sp.blank[(q, "baseline_x")] for q in qs), sum(val(da[q]) for q in qs)))
            for c in conds:
                s = c.split("-p")[-1]
                g = [st[(q, c)] for q in qs]
                line = "    %-22s answers %3d | right %3d wrong %3d | net %+4d" % (
                    ("must-answer + E %s" if "_conf-" in c else "normal prompt, E %s") % s.replace("-", "."), sum(not bl[(q, c)] for q in qs), g.count("R"), g.count("W"), sum(map(val, g)))
                if "_conf-" in c:
                    if s in ("1", "2"):
                        o0 = MT.handed_in(D, ("DE", s))
                        o = dict(zip(D.sp.qs, o0[0]))
                        line += " | current direction: net %+d; new minus current %s" % (sum(val(o[q]) for q in qs), ps.ci([val(x) - val(o[q]) for x, q in zip(g, qs)]))
                    line += " | minus prompt only %s" % ps.ci([val(x) - val(da[q]) for x, q in zip(g, qs)])
                else:
                    line += " | minus no write %s | minus prompt only %s" % (
                        ps.ci([val(x) - val(y) for x, y in zip(g, base)]), ps.ci([val(x) - val(da[q]) for x, q in zip(g, qs)]))
                    if s in doses:
                        o_all = S.system(sp, tx, ("E", s))
                        o = [o_all[0][k] for k in idx]
                        line += " | current direction: answers %d net %+d; new minus current %s" % (
                            sum(o_all[1][k] for k in idx), sum(map(val, o)), ps.ci([val(x) - val(y) for x, y in zip(g, o)]))
                print(line)


if __name__ == "__main__":
    main()
