"""Task 4: training-data identity and overlap (IDs, normalised question text, near duplicates, answer text)."""
import json, re, sys, difflib, collections, itertools

S = "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad"
G = S + "/gpt18/gen"
R = "/Users/airsanshi/github/research/Truth_Decoupling/claude_gpt_review/from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u/"
STOP = set("a an the of to in on is are was were do does did you your what which who whom whose when where why how "
           "if it its that this these those for with be by as at from or and can will would should could there their "
           "has have had i we they he she his her them our my me us not no yes".split())


def norm(t):
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return " ".join(t.split())


def toks(t):
    return {w for w in norm(t).split() if w not in STOP}


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


out = {}
for model in ("qwen3-8b", "llama3.1-8b"):
    q = json.load(open(R + model + "/questions.json"))
    sp = {k: {r["pair_id"]: r for r in v} for k, v in q["splits"].items()}
    etrain = {r["pair_id"]: r for r in q["e_train"]}
    res = {"split_sizes": {k: len(v) for k, v in sp.items()}, "e_train": len(etrain)}
    res["e_train_subset_of_train_split"] = set(etrain) <= set(sp["train"])
    res["e_tune_equals_tune_split"] = [r["pair_id"] for r in q["e_tune"]] == [r["pair_id"] for r in q["splits"]["tune"]]
    for a, b in itertools.combinations(sp, 2):
        res["id_overlap_%s_%s" % (a, b)] = sorted(set(sp[a]) & set(sp[b]))
    res["id_overlap_etrain_test"] = sorted(set(etrain) & set(sp["test"]))
    res["id_overlap_etrain_tune"] = sorted(set(etrain) & set(sp["tune"]))
    # exact normalised question text
    for other in ("tune", "test"):
        idx = collections.defaultdict(list)
        for pid, r in sp[other].items():
            idx[norm(r["question"])].append(pid)
        res["exact_norm_question_match_train_%s" % other] = [
            (pid, idx[norm(r["question"])]) for pid, r in sp["train"].items() if norm(r["question"]) in idx]
        # near duplicates
        near = []
        for pid, r in sp["train"].items():
            for pid2, r2 in sp[other].items():
                n1, n2 = norm(r["question"]), norm(r2["question"])
                sm = difflib.SequenceMatcher(None, n1, n2).ratio()
                jj = jac(toks(r["question"]), toks(r2["question"]))
                if sm >= 0.70 or jj >= 0.5:
                    near.append({"train": pid, other: pid2, "seqratio": round(sm, 3), "jaccard": round(jj, 3),
                                 "q_train": r["question"], "q_" + other: r2["question"],
                                 "train_in_e_train": pid in etrain})
        near.sort(key=lambda x: -max(x["seqratio"], x["jaccard"]))
        res["near_dup_train_%s" % other] = near
    # answer-level overlap: training targets (positive_answer) vs test gold answers
    pos = {pid: norm(r["positive_answer"]) for pid, r in etrain.items()}
    pos_idx = collections.defaultdict(list)
    for pid, t in pos.items():
        pos_idx[t].append(pid)
    ans_hits = []
    for pid, r in sp["test"].items():
        for g in r["gold"]:
            if norm(g) in pos_idx:
                ans_hits.append({"test": pid, "gold": g, "train_pairs": pos_idx[norm(g)]})
    res["test_gold_equal_to_a_training_target"] = ans_hits
    res["n_test_questions_with_gold_equal_training_target"] = len({h["test"] for h in ans_hits})
    out[model] = res
    print("==", model, res["split_sizes"], "e_train", res["e_train"], "subset", res["e_train_subset_of_train_split"],
          "e_tune==tune", res["e_tune_equals_tune_split"])
    print("  id overlaps:", {k: len(v) for k, v in res.items() if k.startswith("id_overlap")})
    print("  exact norm question matches:", {k: v for k, v in res.items() if k.startswith("exact_norm")})
    for other in ("tune", "test"):
        print("  near-dup train-%s pairs: %d" % (other, len(res["near_dup_train_%s" % other])))
        for x in res["near_dup_train_%s" % other][:40]:
            print("    %.2f/%.2f %s(%s) | %s :: %s | %s" % (x["seqratio"], x["jaccard"], x["train"], "E" if x["train_in_e_train"] else "x",
                                                     x[other], x["q_train"], x["q_" + other]))
    print("  test gold == training target:", res["n_test_questions_with_gold_equal_training_target"])
    for h in ans_hits[:30]:
        print("    ", h)
    if model == "llama3.1-8b":
        break_same = json.load(open(R + "qwen3-8b/questions.json")) == q
        print("  llama questions.json identical to qwen:", break_same)
json.dump(out, open(S + "/audit_results/a4_overlap.json", "w"), indent=1, ensure_ascii=False)
