"""Task 2: independent recomputation from per-item archive files (no collaborator code imported).

Reads gpt18/gen/<model>/<run>/<init>/{tune,test}_scored_items.json directly.
Outcome per condition is taken from the judge label (outcome policy judge_primary_v2 = label),
cross-checked against the stored 'outcome' field. Refusal-only gold questions are re-derived
with an own normaliser and compared with the stored gold_is_abstention flag.
"""
import json, math, re, sys, collections
sys.path.append("/Users/airsanshi/Library/Python/3.9/lib/python/site-packages")  # user site (run with -I)
import numpy as np

S = "/private/tmp/claude-501/-Users-airsanshi-MyPapers-Truth-Decoupling/1202de30-5944-41d9-af49-9348a318988a/scratchpad"
GEN = S + "/gpt18/gen"
OUT = S + "/audit_results"
LABELS = ("correct", "wrong", "abstain")
B = 100000
rng = np.random.default_rng(20261106)


def load(p):
    with open(p) as f:
        return json.load(f)


def own_refusal(text):
    t = re.sub(r"[^a-z ]", "", str(text).lower())
    t = " ".join(t.split())
    return t in {"i have no comment", "i dont know", "i do not know"}


def table_rows(path):
    d = load(path)
    conds = d["conditions"]
    rows = {}
    issues = []
    for e in d["evaluations"]:
        pid = e["pair_id"]
        if pid in rows:
            issues.append("duplicate " + pid)
        r = {"q": e["question"], "subst": not all(own_refusal(g) for g in e["gold"]),
             "flag_abst": e["gold_is_abstention"], "lab": {}, "out": {}, "fin": {}}
        for c in conds:
            lab = e["judge"][c]["label"]
            if lab not in LABELS:
                issues.append("label %s %s %s" % (pid, c, lab))
            if e["outcome"][c] != lab:
                issues.append("outcome!=label %s %s" % (pid, c))
            r["lab"][c] = lab
            r["out"][c] = e["outputs"][c]
            r["fin"][c] = e["finished"][c]
        if r["subst"] == r["flag_abst"]:
            issues.append("refusal-only flag disagrees with own rule: " + pid)
        rows[pid] = r
    return d, conds, rows, issues


def counts(labs):
    c = collections.Counter(labs)
    n = len(labs)
    ans = c["correct"] + c["wrong"]
    return {"n": n, "correct": c["correct"], "wrong": c["wrong"], "abstain": c["abstain"],
            "answered": ans, "acc_answered": (c["correct"] / ans if ans else float("nan"))}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a_lab, b_lab):
    """a = adapter, b = comparator; arrays of labels over the same questions."""
    a = np.array(a_lab); b = np.array(b_lab)
    ac = (a == "correct").astype(float); bc = (b == "correct").astype(float)
    aw = (a == "wrong").astype(float); bw = (b == "wrong").astype(float)
    n = len(a)
    up = int(((ac == 1) & (bc == 0)).sum()); down = int(((ac == 0) & (bc == 1)).sum())
    diff = ac - bc
    res = {"net_correct": int(diff.sum()), "a_correct_b_not": up, "b_correct_a_not": down,
           "diff_correct_rate": float(diff.mean()), "mcnemar_exact_p": mcnemar_exact(up, down)}
    # paired bootstrap over questions
    lo_hi = {"correct_rate": [], "acc_answered": [], "c_minus_w_rate": []}
    chunk = 10000
    dc, dacc, dcw = [], [], []
    for s in range(0, B, chunk):
        idx = rng.integers(0, n, size=(chunk, n))
        A_c = ac[idx]; B_c = bc[idx]; A_w = aw[idx]; B_w = bw[idx]
        dc.append((A_c - B_c).mean(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            dacc.append(A_c.sum(1) / (A_c.sum(1) + A_w.sum(1)) - B_c.sum(1) / (B_c.sum(1) + B_w.sum(1)))
        dcw.append(((A_c - A_w) - (B_c - B_w)).mean(1))
    for key, arr in (("correct_rate", dc), ("acc_answered", dacc), ("c_minus_w_rate", dcw)):
        v = np.concatenate(arr)
        lo_hi[key] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    res["ci95_correct_rate"] = lo_hi["correct_rate"]
    # analytic paired Wald interval for comparison (Monte-Carlo independent)
    se = math.sqrt(((diff - diff.mean()) ** 2).sum() / (n - 1) / n)
    res["wald95_correct_rate"] = [float(diff.mean() - 1.959964 * se), float(diff.mean() + 1.959964 * se)]
    accA = ac.sum() / (ac.sum() + aw.sum()); accB = bc.sum() / (bc.sum() + bw.sum())
    res["diff_acc_answered"] = float(accA - accB); res["ci95_acc_answered"] = lo_hi["acc_answered"]
    res["diff_c_minus_w_rate"] = float(((ac - aw) - (bc - bw)).mean()); res["ci95_c_minus_w_rate"] = lo_hi["c_minus_w_rate"]
    return res


def transitions(base, other):
    t = collections.Counter(zip(base, other))
    return {"%s->%s" % (x, y): t[(x, y)] for x in LABELS for y in LABELS}


report = {}

# ---------------- TEST (Qwen alpha 3) ----------------
runs = {}
for init in ("das", "md", "random"):
    runs[init] = table_rows("%s/qwen3-8b/ce_alpha3_ep3/%s/test_scored_items.json" % (GEN, init))
test = {"issues": {k: v[3] for k, v in runs.items()}}
ids = [list(v[2].keys()) for v in runs.values()]
test["same_question_order_all_runs"] = ids[0] == ids[1] == ids[2]
test["n_questions"] = len(ids[0])
subst_ids = [p for p, r in runs["das"][2].items() if r["subst"]]
test["n_substantive"] = len(subst_ids)
test["refusal_only_ids"] = [p for p, r in runs["das"][2].items() if not r["subst"]]
# shared baseline / original_das identical across runs?
shared = {}
for c in ("baseline", "original_das"):
    diffs = {"output": 0, "finished": 0, "label": 0}
    for p in ids[0]:
        ref = runs["das"][2][p]
        for init in ("md", "random"):
            o = runs[init][2][p]
            diffs["output"] += o["out"][c] != ref["out"][c]
            diffs["finished"] += o["fin"][c] != ref["fin"][c]
            diffs["label"] += o["lab"][c] != ref["lab"][c]
    shared[c] = diffs
test["shared_condition_differences_vs_das_run"] = shared

L = {}
L["baseline"] = [runs["das"][2][p]["lab"]["baseline"] for p in subst_ids]
L["original_das"] = [runs["das"][2][p]["lab"]["original_das"] for p in subst_ids]
for init in ("das", "md", "random"):
    L[init + "_adapter"] = [runs[init][2][p]["lab"][init + "_generation_adapter"] for p in subst_ids]
test["table_523"] = {k: counts(v) for k, v in L.items()}
test["transitions_das_adapter_vs_baseline"] = transitions(L["baseline"], L["das_adapter"])
test["paired"] = {
    "das_adapter_vs_baseline": paired(L["das_adapter"], L["baseline"]),
    "das_adapter_vs_original_das": paired(L["das_adapter"], L["original_das"]),
    "das_adapter_vs_md_adapter": paired(L["das_adapter"], L["md_adapter"]),
    "das_adapter_vs_random_adapter": paired(L["das_adapter"], L["random_adapter"]),
    "random_adapter_vs_baseline": paired(L["random_adapter"], L["baseline"]),
    "md_adapter_vs_baseline": paired(L["md_adapter"], L["baseline"]),
}
# refusal-only 6
test["refusal_only_outcomes"] = {
    c: dict(collections.Counter(runs["das"][2][p]["lab"][c] for p in test["refusal_only_ids"]))
    for c in ("baseline", "original_das", "das_generation_adapter")}
report["test_qwen_alpha3"] = test

# ---------------- TUNE (both models, alpha 1/3; EOS) ----------------
tune = {}
base_ref = {}
for model in ("llama3.1-8b", "qwen3-8b"):
    for run in ("ce_alpha1_ep3", "ce_alpha3_ep3", "ce_eos_alpha3_ep3"):
        for init in ("das", "md", "random"):
            p = "%s/%s/%s/%s/tune_scored_items.json" % (GEN, model, run, init)
            try:
                d, conds, rows, issues = table_rows(p)
            except FileNotFoundError:
                continue
            sid = [k for k, r in rows.items() if r["subst"]]
            key = "%s/%s/%s" % (model, run, init)
            ent = {"n_questions": len(rows), "n_substantive": len(sid), "issues": issues}
            for c in conds:
                ent[c] = counts([rows[k]["lab"][c] for k in sid])
                ent[c]["unfinished_all"] = sum(not rows[k]["fin"][c] for k in rows)
                ent[c]["fffd_all"] = sum("�" in rows[k]["out"][c] for k in rows)
            ad = init + "_generation_adapter"
            ent["paired_vs_baseline"] = paired([rows[k]["lab"][ad] for k in sid], [rows[k]["lab"]["baseline"] for k in sid])
            # baseline identical across runs of same model?
            bl = {k: (rows[k]["out"]["baseline"], rows[k]["lab"]["baseline"]) for k in rows}
            if model not in base_ref:
                base_ref[model] = (key, bl)
            else:
                ent["baseline_diff_vs_" + base_ref[model][0]] = sum(bl[k] != base_ref[model][1][k] for k in bl)
            tune[key] = ent
report["tune"] = tune

json.dump(report, open(OUT + "/a2_recompute.json", "w"), indent=1, ensure_ascii=False)

# ---------------- print summary ----------------
print("TEST issues:", {k: len(v) for k, v in test["issues"].items()}, "order same:", test["same_question_order_all_runs"])
print("n", test["n_questions"], "substantive", test["n_substantive"], "refusal-only", test["refusal_only_ids"])
print("shared condition diffs:", shared)
for k, v in test["table_523"].items():
    print("  %-16s C %3d W %3d A %3d answered %3d acc %.4f" % (k, v["correct"], v["wrong"], v["abstain"], v["answered"], v["acc_answered"]))
print("transitions DAS adapter vs baseline:", test["transitions_das_adapter_vs_baseline"])
for k, v in test["paired"].items():
    print("  %-30s net %+d (%d/%d) diff %.4f boot %s wald %s p %.6g | dAcc %.4f %s | dCW %.4f %s" % (
        k, v["net_correct"], v["a_correct_b_not"], v["b_correct_a_not"], v["diff_correct_rate"],
        [round(x, 4) for x in v["ci95_correct_rate"]], [round(x, 4) for x in v["wald95_correct_rate"]],
        v["mcnemar_exact_p"], v["diff_acc_answered"], [round(x, 4) for x in v["ci95_acc_answered"]],
        v["diff_c_minus_w_rate"], [round(x, 4) for x in v["ci95_c_minus_w_rate"]]))
print("refusal-only outcomes:", test["refusal_only_outcomes"])
print()
for key, ent in tune.items():
    ad = key.split("/")[-1] + "_generation_adapter"
    b, o, a = ent["baseline"], ent["original_das"], ent[ad]
    pv = ent["paired_vs_baseline"]
    extra = {k: v for k, v in ent.items() if k.startswith("baseline_diff")}
    print("%-34s n=%d/%d base C%d W%d A%d | origDAS C%d W%d A%d | adapter C%d W%d A%d unfin%d fffd%d | net %+d p=%.3g %s %s" % (
        key, ent["n_substantive"], ent["n_questions"], b["correct"], b["wrong"], b["abstain"],
        o["correct"], o["wrong"], o["abstain"], a["correct"], a["wrong"], a["abstain"],
        a["unfinished_all"], a["fffd_all"], pv["net_correct"], pv["mcnemar_exact_p"], extra,
        "ISSUES" if ent["issues"] else ""))
