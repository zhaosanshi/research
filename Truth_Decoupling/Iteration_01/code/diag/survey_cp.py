# read-only survey: judged dirs holding C toward caution (C_L<c>_x-w-p...) conditions, per run root
import glob, json, os, collections
import sys
ROOTS = sys.argv[1:] or ["results_e44_factorial_das/llama3.1-8b_side_E2", "results_newmodels_ff_20261112/e44/qwen3.5-9b_E2",
         "results_newmodels_ff_20261112/e44/qwen3.8-27b-dm", "Iteration_01/results/ff/gemma4/12b/e44/gemma4-12b",
         "Iteration_01/results/ff_xu1/gemma4/12b/e44/gemma4-12b", "Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b",
         "Iteration_01/results/ff/gemma4/31b/e44/gemma4-31b-dm", "Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b"]
for r in ROOTS:
    print("\n##", r)
    for f in sorted(glob.glob(r + "/**/judge2/single_*.jsonl", recursive=True)):
        conds = collections.Counter(); qs = set(); unk = collections.Counter()
        for l in open(f, encoding="utf-8"):
            x = json.loads(l)
            c = x["condition"]
            if "_x-w-p" in c and c.startswith("C_L"):
                conds[c] += 1
                if x["status"] != "scored" and not (x.get("basis") == "empty_output_no_judge_vote" and x.get("empty_output") is True):
                    unk[c] += 1
            if c == "baseline_x":
                qs.add(x["pair_id"])
        if conds:
            print("  %s | baseline_x %d | %s" % (os.path.relpath(f, r), len(qs), " ".join("%s:%d(?%d)" % (c, n, unk[c]) for c, n in sorted(conds.items()))))
