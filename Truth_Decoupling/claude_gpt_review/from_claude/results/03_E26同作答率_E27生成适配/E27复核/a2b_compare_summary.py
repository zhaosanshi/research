"""Compare collaborator's all_generation_runs_summary.json (54 rows) with own per-item recount."""
import json, sys, collections
S = "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad"
D = "/Users/airsanshi/github/research/Truth_Decoupling/claude_gpt_review/from_gpt/main_experiment_20261106"
rows = json.load(open(D + "/all_generation_runs_summary.json"))
mism = []
cache = {}
for r in rows:
    f = r["file"]
    if f not in cache:
        cache[f] = json.load(open(S + "/gpt18/gen/" + f))["evaluations"]
    ev = cache[f]
    c = r["condition"]
    main = [e for e in ev if not e["gold_is_abstention"]]
    cnt = collections.Counter(e["judge"][c]["label"] for e in main)
    own = {"n_raw": len(ev), "n_main": len(main), "correct": cnt["correct"], "wrong": cnt["wrong"], "abstain": cnt["abstain"],
           "replacement_character_raw": sum("�" in e["outputs"][c] for e in ev),
           "unfinished_raw": sum(not e["finished"][c] for e in ev)}
    for k, v in own.items():
        if r.get(k) != v:
            mism.append((f, c, k, r.get(k), v))
print("rows", len(rows), "mismatches", len(mism))
for m in mism: print(m)
