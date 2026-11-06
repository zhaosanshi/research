"""Task 3/7: selection discipline and identity fingerprints (own reimplementation of the documented hash scheme)."""
import json, hashlib, glob, os
S = "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad"
G = S + "/gpt18/gen"
R = "/Users/airsanshi/github/research/Truth_Decoupling/claude_gpt_review/from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u/"


def fp(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


q = json.load(open(R + "qwen3-8b/questions.json"))
split_ids = {k: [r["pair_id"] for r in v] for k, v in q["splits"].items()}
sel = json.load(open(G + "/qwen3-8b/heldout_selection.json"))
out = {"selection": {k: sel[k] for k in sel if k != "source_sha256"}}
rows = []
for d in sorted(glob.glob(G + "/*/*/*/")) + [G + "/qwen3-8b/eos_pilot3/"]:
    a = json.load(open(d + "adapter.json"))
    pt = sha(d + "adapter.pt")
    ident = fp([pt, a["source"], a["config"]])
    for fn in ("tune.json", "test.json", "check.json"):
        if not os.path.exists(d + fn):
            continue
        g = json.load(open(d + fn))
        split = g["split"]
        ids = [e["pair_id"] for e in g["evaluations"]]
        sig = fp([ident, g["condition_specs"], split, ids, fn == "check.json"])
        exp_specs = [{"name": "baseline", "kind": "baseline", "eligible_for_selection": False, "uses_reference_answer": False},
                     {"name": "original_das", "kind": "E", "context": "qa_das", "layers": a["layers"], "alpha": a["alpha"],
                      "mode": "continuous", "uses_reference_answer": False, "eligible_for_selection": False},
                     {"name": a["method"], "kind": "E", "context": a["method"], "layers": a["layers"], "alpha": a["alpha"],
                      "mode": "continuous", "uses_reference_answer": False, "eligible_for_selection": False}]
        row = {"run": d.replace(G + "/", ""), "file": fn, "split": split, "n": len(ids),
               "ids_equal_source_split": ids == split_ids.get(split, [])[:len(ids)] if fn != "check.json" else ids == split_ids["tune"][:3],
               "training_sha256_eq_adapter_pt": g["training_sha256"] == pt,
               "bank_protocol_fp_recomputed": g["bank_protocol_fingerprint"] == ident,
               "protocol_fp_recomputed": g["protocol_fingerprint"] == sig,
               "condition_specs_equal_release_code_construction": g["condition_specs"] == exp_specs,
               "questions_sha256_eq_source": g["questions_sha256"] == sha(R + a["model"] + "/questions.json"),
               "training_json_sha_eq_source": a["source"]["training_json_sha256"] == sha(R + a["model"] + "/training.json"),
               "bank_json_sha_eq_source": a["source"]["bank_json_sha256"] == sha(R + a["model"] + "/bank.json"),
               "pilot_flags": [g.get("pilot"), g.get("wiring_pilot"), g.get("diagnostic_only")],
               "eos_keys_in_train_config": "include_eos" in a["train_config"]}
        if split == "test":
            key = row["run"].rstrip("/").replace("qwen3-8b/", "") + "/adapter.pt"
            row["adapter_pt_hash_frozen_in_selection"] = sel["source_sha256"].get(key) == pt
        rows.append(row)
for r in rows:
    print(json.dumps(r))
out["rows"] = rows
# selection rule check: which DAS alpha had the highest Qwen tuning correct count?
cnt = {}
for a in ("1", "3"):
    ev = json.load(open(G + "/qwen3-8b/ce_alpha%s_ep3/das/tune_scored_items.json" % a))["evaluations"]
    cnt[a] = sum(e["judge"]["das_generation_adapter"]["label"] == "correct" for e in ev if not e["gold_is_abstention"])
out["qwen_das_tune_correct_by_alpha"] = cnt
print("Qwen DAS tune correct by alpha:", cnt)
# outputs identical to baseline at alpha 1?
for model in ("qwen3-8b", "llama3.1-8b"):
    for run in ("ce_alpha1_ep3", "ce_alpha3_ep3", "ce_eos_alpha3_ep3"):
        for init in ("das", "md", "random"):
            p = "%s/%s/%s/%s/tune_scored_items.json" % (G, model, run, init)
            if not os.path.exists(p):
                continue
            ev = json.load(open(p))["evaluations"]
            same = sum(e["outputs"][init + "_generation_adapter"] == e["outputs"]["baseline"] for e in ev)
            print(model, run, init, "adapter outputs identical to baseline: %d/%d" % (same, len(ev)))
json.dump(out, open(S + "/audit_results/a3_selection.json", "w"), indent=1)
