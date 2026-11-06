"""Supplement to task 5: answer length and hedging markers per condition x label (test, main 523)."""
import json, re, collections, statistics
d = json.load(open("/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad/audit_results/a5_quality_items.json"))
HEDGE = re.compile(r"possible answers?:|it depends|depends on|uncertain|not guaranteed|may vary|varies|vary\b|no single|not universally", re.I)
LIST = re.compile(r"possible answers?:", re.I)
res = {}
for c in ("baseline", "original_das", "das_adapter", "md_adapter", "random_adapter"):
    rows = [v[c] for v in d.values() if v[c]["main"]]
    lens = [len(r["text"].split()) for r in rows]
    by = {}
    for lab in ("correct", "wrong", "abstain"):
        sub = [r for r in rows if r["label"] == lab]
        by[lab] = {"n": len(sub), "hedge": sum(bool(HEDGE.search(r["text"])) for r in sub),
                   "possible_answers_list": sum(bool(LIST.search(r["text"])) for r in sub),
                   "median_words": statistics.median([len(r["text"].split()) for r in sub]) if sub else None}
    res[c] = {"median_words": statistics.median(lens), "mean_words": round(statistics.mean(lens), 1),
              "share_ge_25_words": round(sum(l >= 25 for l in lens) / len(lens), 3), "by_label": by}
    print(c, json.dumps(res[c]))
# gains (DAS correct, baseline not correct): how many are hedged?
g = [v for v in d.values() if v["das_adapter"]["main"] and v["das_adapter"]["label"] == "correct" and v["baseline"]["label"] != "correct"]
print("DAS gains:", len(g), "hedged:", sum(bool(HEDGE.search(v["das_adapter"]["text"])) for v in g),
      "possible-answers list:", sum(bool(LIST.search(v["das_adapter"]["text"])) for v in g),
      ">=25 words:", sum(len(v["das_adapter"]["text"].split()) >= 25 for v in g))
json.dump(res, open("/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad/audit_results/a5b_style.json", "w"), indent=1)
