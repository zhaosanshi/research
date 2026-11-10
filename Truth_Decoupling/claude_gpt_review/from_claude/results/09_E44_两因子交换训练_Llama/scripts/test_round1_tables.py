# -*- coding: utf-8 -*-
"""Synthetic check of e44_round1_tables.py: known effects must come out with the right sign, slot and verdict."""
import json, random, subprocess, sys, tempfile
from pathlib import Path
rng = random.Random(1)
N = 192
ids = ["q%d" % i for i in range(N)]
base = {}
for i, q in enumerate(ids):                      # baseline x: 78 R, 88 W, 3 H, 23 X
    base[q] = "R" if i < 78 else "W" if i < 166 else "H" if i < 169 else "X"

def flip(src, n_up=0, n_down=0, n_refuse=0, n_answer=0, seed=0):
    r = random.Random(seed); out = dict(src)
    W = [q for q in ids if src[q] == "W"]; R = [q for q in ids if src[q] == "R"]; X = [q for q in ids if src[q] == "X"]
    for q in r.sample(W, n_up): out[q] = "R"
    for q in r.sample(R, n_down): out[q] = "W"
    rest = [q for q in ids if out[q] in "RW" and out[q] == src[q]]
    for q in r.sample([q for q in rest if src[q] == "W"], n_refuse): out[q] = "X"
    for q in r.sample(X, n_answer): out[q] = "W"
    return out

def build(family, layers, table):
    conds, specs, st = ["baseline_x"], [{"name": "baseline_x", "base_condition": "baseline_x", "kind": "prompt_control"}], {"baseline_x": base}
    for l in layers:
        for arm, s in table[l].items():
            name = "%s_L%d_%s" % (family, l, arm.replace(".", "-"))
            conds.append(name); specs.append({"name": name, "base_condition": "baseline_x", "family": family, "layer": l, "arm": arm})
            st[name] = s
    return conds, specs, st

def write(tmp, family, conds, specs, st, judge="allenai"):
    work = Path(tmp) / ("work_" + family); (work / "judge2").mkdir(parents=True)
    (work / "judge_input.json").write_text(json.dumps({"question_ids": ids, "conditions": conds, "condition_specs": specs}))
    lab = {"R": "A", "W": "B", "H": "N", "X": "A"} if judge == "allenai" else {"R": "A", "W": "B", "H": "H", "X": "C"}
    with (work / "judge2" / ("single_%s.jsonl" % judge)).open("w") as f:
        for c in conds:
            for q in ids:
                f.write(json.dumps({"pair_id": q, "condition": c, "status": "scored", "label": lab[st[c][q]], "rule_abstain": st[c][q] == "X"}) + "\n")
    return work

tmp = tempfile.mkdtemp()
run = Path(tmp) / "run"; (run / "mc").mkdir(parents=True)
# E: layer 0 nothing; layer 1 +20 right, no refusal change; layer 2 +30 right but 15 refusals answered (dX -15); layer 3 +8 right; layer 4 +1 but best margin
E = {}
plan = {0: (0, 0, 0, 0), 1: (25, 5, 0, 0), 2: (20, 5, 0, 15), 3: (12, 4, 0, 0), 4: (3, 2, 0, 0)}
for l, (up, down, ref, ans) in plan.items():
    main = flip(base, up, down, ref, ans, seed=l)
    if l == 2:                                    # the 15 answered refusals are all right in this synthetic case
        for q in ids:
            if base[q] == "X" and main[q] == "W": main[q] = "R"
    E[l] = {"x.w.p1": main, "x.w.m1": flip(base, 2, 3, seed=50 + l), "x.md.p1": flip(base, 4, 3, seed=60 + l), "x.rand.p1": flip(base, 3, 4, seed=70 + l)}
    (run / "mc" / ("E_L%d_x-w-p1_tune.json" % l)).write_text(json.dumps({"summary": {"margin_change_mean": 0.9 if l == 4 else 0.1 * l}}))
conds, specs, st = build("E", sorted(E), E)
wE = write(tmp, "E", conds, specs, st)
wEp = Path(tmp) / "pro"; wEp.mkdir(); import shutil
w2 = write(str(wEp), "E", conds, specs, st, judge="pro")
# C: layer 0 nothing; layer 1 two-way (+20 / -15) with correctness preserved; layer 2 one-way +30; layer 3 two-way +10/-8
C = {}
planC = {0: (0, 0), 1: (20, 15), 2: (30, 0), 3: (10, 8)}
for l, (ref, ans) in planC.items():
    C[l] = {"x.w.p1": flip(base, 0, 0, ref, 0, seed=l), "x.w.m1": flip(base, 0, 0, 0, ans, seed=l),
            "x.md.p1": flip(base, 0, 0, ref // 2, 0, seed=9 + l), "x.md.m1": flip(base, 0, 0, 0, ans // 2, seed=9 + l),
            "x.rand.p1": flip(base, 0, 0, 1, 0, seed=20 + l), "x.rand.m1": flip(base, 0, 0, 0, 1, seed=20 + l)}
conds, specs, st = build("C", sorted(C), C)
wC = write(tmp, "C", conds, specs, st)
S = str(Path(__file__).with_name("e44_round1_tables.py"))
def go(*args):
    out = subprocess.run([sys.executable, S, *args], capture_output=True, text=True)
    if out.returncode: print(out.stderr); raise SystemExit("FAILED " + " ".join(args))
    return out.stdout
o = go("screen", str(run), "E", str(wE)); print(o)
assert "shortlist layers: 2 1 3 4" in o, "E shortlist"       # dR: L2 +30, L1 +20, L3 +8 (|dX|<=6), slot 4 = L4 by margin
o = go("cells", str(run), "E", str(wE), "--layers", "1", "2", "0"); print(o)
assert o.count("PASS") >= 2
o = go("cells", str(run), "E", str(w2), "--judge", "pro", "--layers", "1"); print(o[:600])
o = go("screen", str(run), "C", str(wC)); print(o)
assert "shortlist layers: 1 3" in o, "C shortlist"
o = go("cells", str(run), "C", str(wC), "--layers", "1", "2", "0"); print(o)
assert "TWO-WAY PASS" in o and "ONE-WAY (x.w.p1)" in o
o = go("cross", str(run), "E", str(wE), "--layers", "1"); print(o)
print("synthetic checks passed")
