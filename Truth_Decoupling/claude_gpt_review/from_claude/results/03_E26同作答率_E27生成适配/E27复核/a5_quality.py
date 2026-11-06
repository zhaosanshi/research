"""Task 5: output quality on the Qwen alpha-3 test set, cross-tabbed with judge labels,
plus quality-adjusted net gains; also gain decomposition on training-template siblings."""
import json, re, sys, collections, difflib

S = "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad"
G = S + "/gpt18/gen"
R = "/Users/airsanshi/github/research/Truth_Decoupling/claude_gpt_review/from_claude/results/02_问答DAS_v1/第16轮_DAS正结果搜索/v1u/"
OUT = S + "/audit_results"

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
THINK = re.compile(r"</?think>")
HTAGS = re.compile(r"\.htags|htags")


def words(t):
    return re.findall(r"\S+", t.lower())


def repetition(t):
    """Obvious repetition: (a) an n-gram (n=1..8 words) repeated >=3 times back-to-back;
    (b) a 2-12 char unit repeated >=4 times back-to-back (e.g. 'H.H.H.H.'); (c) a sentence/clause of
    >=4 words occurring twice."""
    w = [re.sub(r"[^\w']", "", x) for x in words(t)]
    w = [x for x in w if x]
    for n in range(1, 9):
        for i in range(0, len(w) - 3 * n + 1):
            if w[i:i + n] == w[i + n:i + 2 * n] == w[i + 2 * n:i + 3 * n]:
                if n == 1 and w[i] in ("no", "very", "ha", "la"):  # allow emphatic doubling
                    continue
                return "ngram%d:%s" % (n, " ".join(w[i:i + n]))
    m = re.search(r"(.{2,12}?)\1{3,}", t, flags=re.S)
    if m and m.group(1).strip():
        return "chars:%r" % m.group(1)
    sents = [" ".join(re.sub(r"[^\w' ]", " ", s.lower()).split()) for s in re.split(r"[.!?;\n]", t)]
    sents = [s for s in sents if len(s.split()) >= 4]
    c = collections.Counter(sents)
    dup = [s for s, k in c.items() if k >= 2]
    if dup:
        return "dup_sentence:%s" % dup[0][:60]
    return None


def feats(text, finished):
    return {"fffd": "�" in text, "unterminated": not finished, "repetition": repetition(text) is not None,
            "cjk": bool(CJK.search(text)), "think_leak": bool(THINK.search(text)), "htags": bool(HTAGS.search(text))}


def garbage(f):
    """'garbage' = visible corruption: U+FFFD, CJK in an English answer, leaked think tag, .htags artefact."""
    return f["fffd"] or f["cjk"] or f["think_leak"] or f["htags"]


runs = {i: {e["pair_id"]: e for e in json.load(open("%s/qwen3-8b/ce_alpha3_ep3/%s/test_scored_items.json" % (G, i)))["evaluations"]}
        for i in ("das", "md", "random")}
ids = list(runs["das"])
COND = {"baseline": ("das", "baseline"), "original_das": ("das", "original_das"),
        "das_adapter": ("das", "das_generation_adapter"), "md_adapter": ("md", "md_generation_adapter"),
        "random_adapter": ("random", "random_generation_adapter")}
rows = {}
for pid in ids:
    rows[pid] = {}
    for c, (run, name) in COND.items():
        e = runs[run][pid]
        f = feats(e["outputs"][name], e["finished"][name])
        f["label"] = e["judge"][name]["label"]
        f["main"] = not e["gold_is_abstention"]
        f["text"] = e["outputs"][name]
        f["rep_detail"] = repetition(e["outputs"][name])
        rows[pid][c] = f

res = {"per_condition": {}, "crosstab": {}, "adjusted": {}}
FEATS = ("fffd", "unterminated", "repetition", "cjk", "think_leak", "htags")
for c in COND:
    allr = [rows[p][c] for p in ids]
    mainr = [r for r in allr if r["main"]]
    res["per_condition"][c] = {
        "raw529": {k: sum(r[k] for r in allr) for k in FEATS},
        "main523": {k: sum(r[k] for r in mainr) for k in FEATS},
        "main523_any_garbage": sum(garbage(r) for r in mainr),
        "main523_garbage_or_unterminated_or_repetition": sum(garbage(r) or r["unterminated"] or r["repetition"] for r in mainr),
    }
    ct = {}
    for lab in ("correct", "wrong", "abstain"):
        sub = [r for r in mainr if r["label"] == lab]
        ct[lab] = {"n": len(sub), **{k: sum(r[k] for r in sub) for k in FEATS},
                   "any_garbage": sum(garbage(r) for r in sub),
                   "garbage_or_unterm_or_rep": sum(garbage(r) or r["unterminated"] or r["repetition"] for r in sub)}
    res["crosstab"][c] = ct

# quality-adjusted correct counts (same rule applied to every condition)
rules = {
    "fffd": lambda r: r["fffd"],
    "garbage(fffd|cjk|think|htags)": garbage,
    "garbage|unterminated": lambda r: garbage(r) or r["unterminated"],
    "garbage|unterminated|repetition": lambda r: garbage(r) or r["unterminated"] or r["repetition"],
}
for name, rule in rules.items():
    adj = {}
    for c in COND:
        mainr = [rows[p][c] for p in ids if rows[p][c]["main"]]
        adj[c] = sum(r["label"] == "correct" and not rule(r) for r in mainr)
    adj["net_das_adapter_vs_baseline"] = adj["das_adapter"] - adj["baseline"]
    adj["net_das_adapter_vs_random"] = adj["das_adapter"] - adj["random_adapter"]
    adj["net_das_adapter_vs_md"] = adj["das_adapter"] - adj["md_adapter"]
    # paired McNemar on adjusted correctness, DAS adapter vs baseline
    up = sum(1 for p in ids if rows[p]["das_adapter"]["main"] and (rows[p]["das_adapter"]["label"] == "correct" and not rule(rows[p]["das_adapter"]))
             and not (rows[p]["baseline"]["label"] == "correct" and not rule(rows[p]["baseline"])))
    down = sum(1 for p in ids if rows[p]["das_adapter"]["main"] and not (rows[p]["das_adapter"]["label"] == "correct" and not rule(rows[p]["das_adapter"]))
               and (rows[p]["baseline"]["label"] == "correct" and not rule(rows[p]["baseline"])))
    import math
    n = up + down; k = min(up, down)
    adj["das_vs_baseline_discordant"] = [up, down]
    adj["das_vs_baseline_mcnemar_p"] = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    res["adjusted"][name] = adj

# DAS adapter 'correct' answers among the 60 baseline->correct gains: how many are garbage/unterminated?
gain_ids = [p for p in ids if rows[p]["das_adapter"]["main"] and rows[p]["das_adapter"]["label"] == "correct" and rows[p]["baseline"]["label"] != "correct"]
res["das_gain_items"] = {"n": len(gain_ids),
                         "fffd": sum(rows[p]["das_adapter"]["fffd"] for p in gain_ids),
                         "unterminated": sum(rows[p]["das_adapter"]["unterminated"] for p in gain_ids),
                         "repetition": sum(rows[p]["das_adapter"]["repetition"] for p in gain_ids),
                         "any_garbage": sum(garbage(rows[p]["das_adapter"]) for p in gain_ids),
                         "garbage_or_unterm_or_rep": sum(garbage(rows[p]["das_adapter"]) or rows[p]["das_adapter"]["unterminated"] or rows[p]["das_adapter"]["repetition"] for p in gain_ids)}

# ---- training-template siblings: gain decomposition ----
q = json.load(open(R + "qwen3-8b/questions.json"))
etrain = q["e_train"]


def norm(t):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", t.lower().replace("’", "'")).split())


STOP = set("a an the of to in on is are was were do does did you your what which who whom whose when where why how if it its that this these those for with be by as at from or and can will would should could there their has have had i we they he she his her them our my me us not no yes".split())


def toks(t):
    return {w for w in norm(t).split() if w not in STOP}


test_q = {r["pair_id"]: r for r in q["splits"]["test"]}
sib = set()
for pid, r in test_q.items():
    for t in etrain:
        sm = difflib.SequenceMatcher(None, norm(t["question"]), norm(r["question"])).ratio()
        a, b = toks(t["question"]), toks(r["question"])
        jj = len(a & b) / len(a | b) if a | b else 0
        if sm >= 0.70 or jj >= 0.5 or norm(t["positive_answer"]) in {norm(g) for g in r["gold"]}:
            sib.add(pid)
dec = {}
for name, group in (("siblings", [p for p in ids if p in sib]), ("non_siblings", [p for p in ids if p not in sib])):
    g = [p for p in group if rows[p]["das_adapter"]["main"]]
    c = lambda cond: sum(rows[p][cond]["label"] == "correct" for p in g)
    dec[name] = {"n": len(g), "baseline": c("baseline"), "das_adapter": c("das_adapter"), "random_adapter": c("random_adapter"),
                 "md_adapter": c("md_adapter")}
res["sibling_decomposition"] = dec
res["sibling_ids"] = sorted(sib)

# ---- does the DAS adapter copy training targets on test? ----
targets = [norm(t["positive_answer"]) for t in etrain]
copies = []
for p in ids:
    o = norm(rows[p]["das_adapter"]["text"])
    best = max(((difflib.SequenceMatcher(None, o, t).ratio(), t) for t in targets), key=lambda x: x[0])
    if best[0] >= 0.85:
        copies.append({"pid": p, "ratio": round(best[0], 3), "output": rows[p]["das_adapter"]["text"][:120], "target": best[1],
                       "label": rows[p]["das_adapter"]["label"]})
res["das_outputs_near_copy_of_training_target"] = copies

json.dump(res, open(OUT + "/a5_quality.json", "w"), indent=1, ensure_ascii=False)
json.dump({p: {c: {k: v for k, v in rows[p][c].items()} for c in COND} for p in ids},
          open(OUT + "/a5_quality_items.json", "w"), indent=0, ensure_ascii=False)

for c, v in res["per_condition"].items():
    print("%-15s raw529 %s | main523 %s anyGarbage %d g|u|r %d" % (c, v["raw529"], v["main523"], v["main523_any_garbage"],
                                                                  v["main523_garbage_or_unterminated_or_repetition"]))
print()
for c, ct in res["crosstab"].items():
    print(c, json.dumps(ct))
print()
for k, v in res["adjusted"].items():
    print(k, v)
print("DAS gain items:", res["das_gain_items"])
print("sibling decomposition:", dec)
print("near copies of training targets by DAS adapter:", len(copies))
for x in copies:
    print("   ", x)
