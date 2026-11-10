# -*- coding: utf-8 -*-
"""Read-only: E44 step-0 prompt controls under both judges (same four states as e44_round1_tables.py), paired changes
between prompt pairs, and raw rows for one pair. CPU, no model.   usage: e44_step0_two_judges.py WORK [--rows A B N]"""
import collections, json, sys
from pathlib import Path
work = Path(sys.argv[1])
gen = json.loads((work / "judge_input.json").read_text(encoding="utf-8"))
ids = gen["question_ids"]
ev = {r["pair_id"]: r for r in gen["evaluations"]}
state = {}
for judge in ("allenai", "pro"):
    for l in (work / "judge2" / ("single_%s.jsonl" % judge)).open(encoding="utf-8"):
        r = json.loads(l)
        if judge == "allenai":
            s = "X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]]
        else:
            s = {"A": "R", "B": "W", "H": "H", "C": "X"}[r["label"]]
        state[(judge, r["pair_id"], r["condition"])] = s
conds = gen["conditions"]
words = lambda c: sum(len((ev[i]["outputs"][c] or "").split()) for i in ids) / len(ids)  # noqa: E731
print("| prompt | AllenAI + rule: R / W / H / X | Pro: R / W / H / X | mean words |")
print("|---|---|---|---|")
for c in conds:
    row = []
    for judge in ("allenai", "pro"):
        k = collections.Counter(state[(judge, i, c)] for i in ids)
        row.append("%d / %d / %d / %d" % (k["R"], k["W"], k["H"], k["X"]))
    print("| %s | %s | %s | %.1f |" % (c.replace("baseline_", ""), row[0], row[1], words(c)))
pairs = [("x", "x_E1"), ("x", "x_F"), ("conf", "conf_E1"), ("conf", "conf_E2"), ("conf_M", "conf_E1"), ("caut", "caut_E1"), ("conf", "caut"), ("conf_E1", "caut_E1")]
print("\n| pair a -> b | judge | dR | W>R / R>W | X>R / X>W | R>X / W>X | d answered | answered-both n, dR | same normalized text |")
print("|---|---|---|---|---|---|---|---|---|")
norm = lambda t: "".join(ch for ch in (t or "").lower() if ch.isalnum() or ch == " ").strip()  # noqa: E731
for a, b in pairs:
    A, B = "baseline_" + a, "baseline_" + b
    same = sum(norm(ev[i]["outputs"][A]) == norm(ev[i]["outputs"][B]) for i in ids)
    for judge in ("allenai", "pro"):
        m = collections.Counter((state[(judge, i, A)], state[(judge, i, B)]) for i in ids)
        R = lambda c: sum(v for (x, y), v in m.items() if (x if c == 0 else y) == "R")  # noqa: E731
        ans = lambda c: sum(v for (x, y), v in m.items() if (x if c == 0 else y) in "RW")  # noqa: E731
        both = {k: v for k, v in m.items() if k[0] in "RW" and k[1] in "RW"}
        nb = sum(both.values()); db = sum(v for (x, y), v in both.items() if y == "R") - sum(v for (x, y), v in both.items() if x == "R")
        print("| %s -> %s | %s | %+d | %d / %d | %d / %d | %d / %d | %+d | %d, %+d | %d |" % (
            a, b, judge, R(1) - R(0), m[("W", "R")], m[("R", "W")], m[("X", "R")], m[("X", "W")], m[("R", "X")], m[("W", "X")],
            ans(1) - ans(0), nb, db, same))
if "--rows" in sys.argv:
    k = sys.argv.index("--rows"); a, b, n = sys.argv[k + 1], sys.argv[k + 2], int(sys.argv[k + 3])
    A, B = "baseline_" + a, "baseline_" + b
    for title, want in (("Pro W under %s -> R under %s" % (a, b), ("W", "R")), ("Pro R under %s -> W under %s" % (a, b), ("R", "W"))):
        print("\n### %s (first %d)" % (title, n))
        shown = 0
        for i in ids:
            if (state[("pro", i, A)], state[("pro", i, B)]) == want and shown < n:
                shown += 1
                print("- Q: %s\n  gold: %s\n  %s: %s\n  %s: %s   [AllenAI %s -> %s]" % (
                    ev[i]["question"], (ev[i].get("best_answer") or "")[:110], a, ev[i]["outputs"][A][:130], b, ev[i]["outputs"][B][:130],
                    state[("allenai", i, A)], state[("allenai", i, B)]))
