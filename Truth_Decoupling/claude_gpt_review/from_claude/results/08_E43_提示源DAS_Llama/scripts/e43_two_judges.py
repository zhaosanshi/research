#!/usr/bin/env python3
"""Read-only: the same cells under AllenAI and DeepSeek Pro (one working directory judged by both).
Per condition counts, flips by how the text changed, agreement by answer length, labels given to rule refusals.
usage: e43_two_judges.py WORK_DIR
"""
import collections, json, sys
from pathlib import Path

work = Path(sys.argv[1])
gen = json.loads((work / "judge_input.json").read_text())
ev = {e["pair_id"]: e for e in gen["evaluations"]}
ids = gen["question_ids"]
specs = {s["name"]: s for s in gen["condition_specs"]}
norm = lambda t: " ".join((t or "").casefold().split()).rstrip(".!? ")
W = lambda t: len((t or "").split())


def load(judge):
    st, raw = {}, collections.Counter()
    for line in (work / "judge2" / ("single_%s.jsonl" % judge)).open():
        r = json.loads(line)
        if r["rule_abstain"]:
            raw[r["label"]] += 1
        if judge == "allenai":
            st[(r["pair_id"], r["condition"])] = "R" if r["rule_abstain"] else r["label"]      # A / B / N / R(ule refusal)
        else:
            st[(r["pair_id"], r["condition"])] = {"A": "A", "B": "B", "C": "R", "H": "H"}[r["label"]]
    return st, raw


al, al_ref = load("allenai")
pr, pr_ref = load("pro")
print("## per condition (192 questions). AllenAI: answered-right / answered-wrong / true-but-uninformative / rule refusal. Pro: right / wrong / hedge / refusal (its own label)")
print("| condition | AllenAI | Pro |")
print("|---|---|---|")
for c in gen["conditions"]:
    a = collections.Counter(al[(i, c)] for i in ids); p = collections.Counter(pr[(i, c)] for i in ids)
    print("| %s | %d / %d / %d / %d | %d / %d / %d / %d |" % (c, a["A"], a["B"], a["N"], a["R"], p["A"], p["B"], p["H"], p["R"]))


def split(st, cond, base):
    out = {}
    for kind in ("same", "short", "long"):
        n = wr = rw = 0
        for i in ids:
            b, x = ev[i]["outputs"][base], ev[i]["outputs"][cond]
            k = "same" if norm(x) == norm(b) else ("long" if W(x) >= W(b) + 2 else "short")
            if k != kind:
                continue
            n += 1
            wr += (st[(i, base)], st[(i, cond)]) == ("B", "A")
            rw += (st[(i, base)], st[(i, cond)]) == ("A", "B")
        out[kind] = (n, wr, rw)
    return out


print("\n## flips by how the text changed. cell = n / wrong→right / right→wrong")
print("| comparison | judge | same text | changed, not longer | changed and longer | all: wrong→right / right→wrong |")
print("|---|---|---|---|---|---|")
pairs = [("baseline_x_E", "baseline_x"), ("baseline_x_caut", "baseline_x_conf")]
pairs += [(n, s["base_condition"]) for n, s in specs.items() if s["kind"] == "intervention"]
for cond, base in pairs:
    for name, st in (("AllenAI", al), ("Pro", pr)):
        sp = split(st, cond, base)
        print("| %s vs %s | %s | %s | %d / %d |" % (cond, base, name, " | ".join("%d / %d / %d" % sp[k] for k in ("same", "short", "long")),
                                                   sum(sp[k][1] for k in sp), sum(sp[k][2] for k in sp)))
buckets = collections.defaultdict(collections.Counter)
for i in ids:
    for c in gen["conditions"]:
        a, p = al[(i, c)], pr[(i, c)]
        if a == "R" or p == "R":
            continue
        w = W(ev[i]["outputs"][c])
        buckets["1-3 words" if w <= 3 else ("4-7 words" if w <= 7 else "8+ words")][(a == "A", p == "A")] += 1
print("\n## agreement on right / not right by answer length (all conditions, refusals excluded)")
print("| answer length | cells | AllenAI right | Pro right | both | AllenAI only | Pro only |")
print("|---|---|---|---|---|---|---|")
for b in ("1-3 words", "4-7 words", "8+ words"):
    c = buckets[b]; n = max(1, sum(c.values()))
    print("| %s | %d | %d (%.0f%%) | %d (%.0f%%) | %d | %d | %d |" % (
        b, n, c[(True, True)] + c[(True, False)], 100 * (c[(True, True)] + c[(True, False)]) / n,
        c[(True, True)] + c[(False, True)], 100 * (c[(True, True)] + c[(False, True)]) / n, c[(True, True)], c[(True, False)], c[(False, True)]))
print("\nrule refusals over all conditions: AllenAI labels %s | Pro labels %s" % (dict(al_ref), dict(pr_ref)))
