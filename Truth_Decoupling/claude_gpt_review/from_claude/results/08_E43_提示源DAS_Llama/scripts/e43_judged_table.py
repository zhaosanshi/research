#!/usr/bin/env python3
"""Read a single-judge (AllenAI) working directory and print per condition, with rule abstentions separated:
answered & A / answered & B / answered & N / abstained (rule) and the paired change against the arm's own base prompt.
AllenAI's info judge labels about half of the plain "I don't know." answers informative, so A alone mixes in refusals;
this table never counts a rule abstention as right or wrong.
usage: e43_judged_table.py WORK_DIR [--arms main recover ...]
"""
import argparse, collections, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path)
ap.add_argument("--arms", nargs="*", default=None)
a = ap.parse_args()

gen = json.loads((a.work / "judge_input.json").read_text())
specs = {s["name"]: s for s in gen["condition_specs"]}
rows = [json.loads(l) for l in (a.work / "judge2" / "single_allenai.jsonl").open()]
state = {}
for r in rows:
    if r["status"] != "scored":
        s = "?"
    elif r["rule_abstain"]:
        s = "R"                      # rule abstention, whatever the judge said
    else:
        s = r["label"]               # A / B / N among answered
    state[(r["pair_id"], r["condition"])] = s
ids = gen["question_ids"]


def counts(cond):
    c = collections.Counter(state[(i, cond)] for i in ids)
    return c


def line(cond):
    c = counts(cond)
    ans = c["A"] + c["B"] + c["N"]
    return "%d / %d / %d / %d (%.0f%%)" % (c["A"], c["B"], c["N"], c["R"], 100.0 * c["A"] / max(1, ans))


print("cell = answered&A / answered&B / answered&N / abstained (precision among answered)")
print("\n| prompt control | cells |")
print("|---|---|")
for name, s in specs.items():
    if s["kind"] == "prompt_control":
        print("| %s | %s |" % (name, line(name)))

fams = sorted({s["family"] for s in specs.values() if "family" in s})
for fam in fams:
    arms = [x for x in dict.fromkeys(s["arm"] for s in specs.values() if s.get("family") == fam) if a.arms is None or x in a.arms]
    layers = sorted({s["layer"] for s in specs.values() if s.get("family") == fam})
    for arm in arms:
        if arm == "none":
            continue
        any_spec = next(s for s in specs.values() if s.get("family") == fam and s["arm"] == arm)
        base = any_spec["base_condition"]
        cb = counts(base)
        print("\n## %s %s (base %s: %s)" % (fam, arm, base, line(base)))
        print("| L | cells | dA | dB | dAbst | wrong→right | right→wrong | wrong→abst | right→abst | abst→right | abst→wrong |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for L in layers:
            cond = "%s_L%d_%s" % (fam, L, arm)
            if cond not in specs:
                continue
            c = counts(cond)
            t = collections.Counter((state[(i, base)], state[(i, cond)]) for i in ids)
            print("| %d | %s | %+d | %+d | %+d | %d | %d | %d | %d | %d | %d |" % (
                L, line(cond), c["A"] - cb["A"], c["B"] - cb["B"], c["R"] - cb["R"],
                t[("B", "A")], t[("A", "B")], t[("B", "R")], t[("A", "R")], t[("R", "A")], t[("R", "B")]))
