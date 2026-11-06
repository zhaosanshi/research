"""Compare E27 baseline/original_das (regenerated) with Claude's v1u baseline / DAS L34 a3 results."""
import json, sys
S = "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad"
R = "/Users/airsanshi/github/research/Truth_Decoupling/claude_gpt_review/from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u/"
out = {}
for model in ("qwen3-8b", "llama3.1-8b"):
    for split, run in (("test", "ce_alpha3_ep3"), ("tune", "ce_alpha3_ep3"), ("tune", "ce_alpha1_ep3")):
        p = "%s/gpt18/gen/%s/%s/das/%s_scored_items.json" % (S, model, run, split)
        try:
            e27 = {e["pair_id"]: e for e in json.load(open(p))["evaluations"]}
        except FileNotFoundError:
            continue
        v = json.load(open(R + model + "/%s_scored_items.json" % split))
        vev = {e["pair_id"]: e for e in v["evaluations"]}
        a = run.split("_")[1][5:]
        res = {"v1u_conditions": v["conditions"], "same_ids": sorted(e27) == sorted(vev)}
        for mine, theirs in (("baseline", "baseline"), ("original_das", [c for c in v["conditions"] if c.endswith("L%d_a%s" % ({"qwen3-8b": 34, "llama3.1-8b": 29}[model], a))])):
            if isinstance(theirs, list):
                if not theirs:
                    res[mine] = "no matching v1u condition"
                    continue
                theirs = theirs[0]
            dout = sum(e27[k]["outputs"][mine] != vev[k]["outputs"][theirs] for k in e27)
            dlab = sum(e27[k]["judge"][mine]["label"] != vev[k]["judge"][theirs]["label"] for k in e27)
            dlab_same_text = sum(e27[k]["judge"][mine]["label"] != vev[k]["judge"][theirs]["label"]
                                 for k in e27 if e27[k]["outputs"][mine] == vev[k]["outputs"][theirs])
            res[mine] = {"vs": theirs, "output_text_diffs": dout, "label_diffs": dlab, "label_diffs_given_same_text": dlab_same_text}
        out["%s/%s/%s" % (model, split, run)] = res
        print(model, split, run, json.dumps(res))
json.dump(out, open(S + "/audit_results/a2c_v1u_compare.json", "w"), indent=1)
