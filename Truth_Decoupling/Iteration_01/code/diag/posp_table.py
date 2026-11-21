# Iteration_01 analysis (Claude Opus 5.5, 2026-11-21), read-only, no model, no judge call.
# TruthfulQA free-form, DeepSeek Pro, score = right - wrong: single E written at the core rule "prompt" (the question
# tokens and the assistant header; no decode step written; launchers/explore_20261120/posp.sh) against single E at the
# training rule "content" (every generated token written as well) at the same layer, for the models whose main-table
# layer is registered in diag/alt_layer.py.
# Rule (stated in posp.sh before any verdict was read): the strength at "prompt" is the one with the best single-E net
# on TUNE (ties to the smaller strength); test is read at that strength. The comparison is single E at the training
# rule at the main-table strength (budget 1: Llama-3.1-8B 3.5, Gemma-3-12B 2.5), which the main table uses.
# Work dirs: <run>/eval_extra/judge_<split>_posprompt<tag>_<split>; their no-write cells must equal the usual ones.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/posp_table.py
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import alt_layer as AL

MT, ps = AL.MT, AL.ps
SETS = (("Llama-3.1-8B", "l16", 16, ("3.5", "5", "7"), "3.5"), ("Gemma-3-12B", "l24", 24, ("2.5", "4", "6"), "2.5"))


def val(x):
    return (x == "R") - (x == "W")


def tok(s):
    return s.replace(".", "-")


def line(label, st, bl, qs, cond):
    sts = [st[(q, cond)] for q in qs]
    return "    %-44s answers %3d | right %3d wrong %3d | net %+4d" % (
        label, sum(not bl[(q, cond)] for q in qs), sts.count("R"), sts.count("W"), sum(map(val, sts)))


def main():
    for model, tag, layer, strengths, main_s in SETS:
        alt = next(a for a in AL.ALTS if a.model == model and a.tag == tag)
        print("\n## %s, E at layer %d: the prompt rule against the training rule" % (model, layer))
        chosen = None
        for i, split in enumerate(("tune", "test")):
            got = alt.load(i)
            if not got:
                print("    %s: the content-rule work dir of this layer is not there" % split)
                return
            D, sp, tx = got
            qs = list(sp.qs)
            run = D.m["ec"][i].split("/eval_extra/")[0]
            w = "%s/eval_extra/judge_%s_posprompt%s_%s" % (run, split, tag, split)
            if not os.path.exists(w + "/judge2/single_pro.jsonl"):
                print("    %s: no Pro verdicts yet in %s" % (split, os.path.basename(w)))
                continue
            st1, bl1 = ps.load(w, "pro")
            bad = [q for q in qs if (q, "baseline_x") in st1 and (st1[(q, "baseline_x")] != sp.st[(q, "baseline_x")] or bl1[(q, "baseline_x")] != sp.blank[(q, "baseline_x")])]
            if bad:
                raise SystemExit("%s %s: the no-write verdicts of the prompt-rule work dir differ from the usual ones on %d questions" % (model, split, len(bad)))
            conds = {s: "E_L%d_x-w-p%s" % (layer, tok(s)) for s in strengths}
            missing = [s for s, c in conds.items() if any(st1.get((q, c), "?") == "?" for q in qs)]
            if missing:
                print("    %s: incomplete verdicts for strengths %s" % (split, missing))
                continue
            print("  %s (%d questions)" % (split, len(qs)))
            print(line("no write", sp.st, sp.blank, qs, "baseline_x"))
            ccond = "E_L%d_x-w-p%s" % (layer, tok(main_s))
            have_content = all(sp.st.get((q, ccond), "?") != "?" for q in qs)
            if have_content:
                print(line("E %s, training rule (every token)" % main_s, sp.st, sp.blank, qs, ccond))
            for s in strengths:
                print(line("E %s, prompt rule" % s, st1, bl1, qs, conds[s]))
            if split == "tune":
                nets = {s: sum(val(st1[(q, conds[s])]) for q in qs) for s in strengths}
                chosen = max(strengths, key=lambda s: (nets[s], -strengths.index(s)))
                print("    chosen on tune for the prompt rule: strength %s (net %+d)" % (chosen, nets[chosen]))
                continue
            if chosen is None:
                print("    no tune choice: test not read")
                continue
            pr = [st1[(q, conds[chosen])] for q in qs]
            base = [sp.st[(q, "baseline_x")] for q in qs]
            print("    at the tune choice %s: prompt rule minus no write %s" % (chosen, ps.ci([val(x) - val(y) for x, y in zip(pr, base)])))
            if have_content:
                ct = [sp.st[(q, ccond)] for q in qs]
                print("    prompt rule (%s) minus training rule (%s) %s | training rule minus no write %s" % (
                    chosen, main_s, ps.ci([val(x) - val(y) for x, y in zip(pr, ct)]), ps.ci([val(x) - val(y) for x, y in zip(ct, base)])))
                same = [s for s in strengths if s == main_s]
                if same and same[0] != chosen:
                    pm = [st1[(q, conds[main_s])] for q in qs]
                    print("    for reference, both at %s: prompt rule minus training rule %s" % (main_s, ps.ci([val(x) - val(y) for x, y in zip(pm, ct)])))


if __name__ == "__main__":
    main()
