# -*- coding: utf-8 -*-
"""Read-only: every question whose label changed between a base prompt (no write) and one arm, with both answers,
under one judge, plus the other judge's labels. Standard library only.
usage: e44_flips.py WORK FAMILY LAYER ARM BASE [--judge pro]"""
import sys, importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("t", str(Path(__file__).with_name("e44_round1_tables.py")))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)

work, family, layer, arm, base = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
judge = sys.argv[sys.argv.index("--judge") + 1] if "--judge" in sys.argv else "pro"
other = "allenai" if judge == "pro" else "pro"
gen, specs, ids, state, rule, words = t.load(work, judge)
_, _, _, state2, _, _ = t.load(work, other)
b, c = "baseline_" + base, t.cond(family, layer, arm)
rows = {r["pair_id"]: r for r in gen["evaluations"]}
NAME = {"R": "right", "W": "wrong", "H": "hedge", "X": "abstained"}
groups = {}
for i in ids:
    a, z = state[(i, b)], state[(i, c)]
    if a != z:
        groups.setdefault((a, z), []).append(i)
print("# Answers whose label changed: %s (no write) -> %s, layer %d, judge %s; %d questions" % (b, c, layer, judge, len(ids)))
print("\nCounts: " + "; ".join("%s -> %s: %d" % (NAME[a], NAME[z], len(v)) for (a, z), v in sorted(groups.items(), key=lambda kv: -len(kv[1]))))
print("Labels in brackets are the other judge (%s) for the same two answers.\n" % other)
for (a, z), v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    print("## %s -> %s (%d)\n" % (NAME[a], NAME[z], len(v)))
    print("| question | no write | with the write | %s |" % other)
    print("|---|---|---|---|")
    for i in v:
        r = rows[i]
        q = (r.get("question") or "").replace("|", "/").replace("\n", " ")
        x = (r["outputs"].get(b) or "").replace("|", "/").replace("\n", " ")
        y = (r["outputs"].get(c) or "").replace("|", "/").replace("\n", " ")
        print("| %s | %s | %s | %s -> %s |" % (q, x, y, NAME[state2[(i, b)]], NAME[state2[(i, c)]]))
    print()
