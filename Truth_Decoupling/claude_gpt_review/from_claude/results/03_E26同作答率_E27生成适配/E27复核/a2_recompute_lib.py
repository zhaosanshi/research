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


