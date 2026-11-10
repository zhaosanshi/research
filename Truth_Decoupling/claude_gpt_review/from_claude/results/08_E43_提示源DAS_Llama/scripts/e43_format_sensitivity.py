#!/usr/bin/env python3
"""Read-only: how much do the AllenAI labels move when ONLY the surface form changes?
For each intervention condition, take the questions whose answer equals the base prompt's answer after normalization
(case, whitespace, trailing . ! ?) but differs as a raw string, and count label changes. Also splits every
condition's answers by "ends with a period" and reports the judge's truth-yes rate in each group.
usage: e43_format_sensitivity.py WORK_DIR [--arm main]
"""
import argparse, collections, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path)
ap.add_argument("--arm", default="main")
a = ap.parse_args()
norm = lambda t: " ".join((t or "").casefold().split()).rstrip(".!? ")
gen = json.loads((a.work / "judge_input.json").read_text())
specs = {s["name"]: s for s in gen["condition_specs"]}
out = {(e["pair_id"], c): e["outputs"][c] for e in gen["evaluations"] for c in gen["conditions"]}
lab = {}
for l in (a.work / "judge2" / "single_allenai.jsonl").open():
    r = json.loads(l)
    lab[(r["pair_id"], r["condition"])] = (r["label"], r["truth"], r["info"], r["rule_abstain"])
ids = gen["question_ids"]
print("| condition | same text, only surface differs (n) | label changed | truth axis changed (T>F / F>T) | info axis changed |")
print("|---|---|---|---|---|")
tot = collections.Counter()
for name, s in specs.items():
    if s.get("arm") != a.arm:
        continue
    base = s["base_condition"]
    same = [i for i in ids if out[(i, name)] != out[(i, base)] and norm(out[(i, name)]) == norm(out[(i, base)])]
    lc = sum(1 for i in same if lab[(i, name)][0] != lab[(i, base)][0])
    tf = sum(1 for i in same if lab[(i, base)][1] == "T" and lab[(i, name)][1] == "F")
    ft = sum(1 for i in same if lab[(i, base)][1] == "F" and lab[(i, name)][1] == "T")
    ic = sum(1 for i in same if lab[(i, name)][2] != lab[(i, base)][2])
    tot.update(n=len(same), lc=lc, tf=tf, ft=ft, ic=ic)
    print("| %s | %d | %d | %d / %d | %d |" % (name, len(same), lc, tf, ft, ic))
print("| total | %d | %d | %d / %d | %d |" % (tot["n"], tot["lc"], tot["tf"], tot["ft"], tot["ic"]))
