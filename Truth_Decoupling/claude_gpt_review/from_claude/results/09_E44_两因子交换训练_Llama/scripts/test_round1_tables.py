# -*- coding: utf-8 -*-
"""Synthetic check of e44_round1_tables.py: known effects must come out with the right sign, slot and verdict.
Includes the length guard (amendment 1): a gain that sits only in lengthened answers must fail check (4)."""
import json, random, subprocess, sys, tempfile
from pathlib import Path
N = 192
ids = ["q%d" % i for i in range(N)]
base = {q: ("R" if i < 78 else "W" if i < 166 else "H" if i < 169 else "X") for i, q in enumerate(ids)}   # 78 R, 88 W, 3 H, 23 X
SHORT, LONG = "a b c", "a b c d e f g h i"                                                              # 3 words, 9 words


def flip(src, n_up=0, n_down=0, n_refuse=0, n_answer=0, seed=0):
    r = random.Random(seed); out = dict(src)
    W = [q for q in ids if src[q] == "W"]; R = [q for q in ids if src[q] == "R"]; X = [q for q in ids if src[q] == "X"]
    for q in r.sample(W, n_up): out[q] = "R"
    for q in r.sample(R, n_down): out[q] = "W"
    rest = [q for q in ids if out[q] == "W" and src[q] == "W"]
    for q in r.sample(rest, n_refuse): out[q] = "X"
    for q in r.sample(X, n_answer): out[q] = "W"
    return out


def texts(src, arm, lengthen_changed):
    """Changed answers become LONG when lengthen_changed, otherwise every answer keeps its length."""
    return {q: (LONG if (lengthen_changed and arm[q] != src[q]) else SHORT) for q in ids}


def write(tmp, family, table, judge="allenai"):
    conds = ["baseline_x"]; specs = [{"name": "baseline_x", "base_condition": "baseline_x", "kind": "prompt_control"}]
    st = {"baseline_x": base}; tx = {"baseline_x": {q: SHORT for q in ids}}
    for l in sorted(table):
        for arm, (s, t) in table[l].items():
            name = "%s_L%d_%s" % (family, l, arm.replace(".", "-"))
            conds.append(name); specs.append({"name": name, "base_condition": "baseline_x", "family": family, "layer": l, "arm": arm})
            st[name] = s; tx[name] = t
    work = Path(tmp) / ("work_%s_%s" % (family, judge)); (work / "judge2").mkdir(parents=True)
    (work / "judge_input.json").write_text(json.dumps({
        "question_ids": ids, "conditions": conds, "condition_specs": specs,
        "evaluations": [{"pair_id": q, "outputs": {c: tx[c][q] for c in conds}} for q in ids]}))
    lab = {"R": "A", "W": "B", "H": "N", "X": "A"} if judge == "allenai" else {"R": "A", "W": "B", "H": "H", "X": "C"}
    with (work / "judge2" / ("single_%s.jsonl" % judge)).open("w") as f:
        for c in conds:
            for q in ids:
                f.write(json.dumps({"pair_id": q, "condition": c, "status": "scored", "label": lab[st[c][q]], "rule_abstain": st[c][q] == "X"}) + "\n")
    return work


tmp = tempfile.mkdtemp()
run = Path(tmp) / "run"; (run / "mc").mkdir(parents=True)
# E layers:  0 nothing | 1 +20 right, same length (a content effect) | 2 +30 right but 15 abstentions answered (coverage change)
#            3 +8 right, same length | 4 +1 right, best reference margin | 5 +25 right, every changed answer becomes long (format-like)
#            6 +6 right, same length (slot 5 candidate)
plan = {0: (0, 0, 0, 0, False), 1: (25, 5, 0, 0, False), 2: (20, 5, 0, 15, False), 3: (12, 4, 0, 0, False), 4: (3, 2, 0, 0, False),
        5: (30, 5, 0, 0, True), 6: (8, 2, 0, 0, False)}
E = {}
for l, (up, down, ref, ans, lengthen) in plan.items():
    main = flip(base, up, down, ref, ans, seed=l)
    if l == 2:
        for q in ids:
            if base[q] == "X" and main[q] == "W": main[q] = "R"
    E[l] = {"x.w.p1": (main, texts(base, main, lengthen))}
    for arm, (u, d, sd) in {"x.w.m1": (2, 3, 50), "x.md.p1": (4, 3, 60), "x.rand.p1": (3, 4, 70)}.items():
        s = flip(base, u, d, seed=sd + l); E[l][arm] = (s, texts(base, s, False))
    (run / "mc" / ("E_L%d_x-w-p1_tune.json" % l)).write_text(json.dumps({"summary": {"margin_change_mean": 0.9 if l == 4 else 0.01 * l}}))
wE = write(tmp, "E", E); wEp = write(tmp, "E", E, judge="pro")
# C layers: 0 nothing | 1 two-way (+20 / -15), correctness kept | 2 one-way +30 | 3 two-way +10 / -8
C = {}
for l, (ref, ans) in {0: (0, 0), 1: (20, 15), 2: (30, 0), 3: (10, 8)}.items():
    C[l] = {}
    for arm, s in {"x.w.p1": flip(base, 0, 0, ref, 0, seed=l), "x.w.m1": flip(base, 0, 0, 0, ans, seed=l),
                   "x.md.p1": flip(base, 0, 0, ref // 2, 0, seed=9 + l), "x.md.m1": flip(base, 0, 0, 0, ans // 2, seed=9 + l),
                   "x.rand.p1": flip(base, 0, 0, 1, 0, seed=20 + l), "x.rand.m1": flip(base, 0, 0, 0, 1, seed=20 + l)}.items():
        C[l][arm] = (s, texts(base, s, False))
wC = write(tmp, "C", C)
S = str(Path(__file__).with_name("e44_round1_tables.py"))


def go(*args):
    out = subprocess.run([sys.executable, S, *args], capture_output=True, text=True)
    if out.returncode:
        print(out.stderr); raise SystemExit("FAILED " + " ".join(args))
    return out.stdout


def section(text, layer):
    return text.split("## layer %d\n" % layer)[1].split("## layer")[0]


o = go("screen", str(run), "E", str(wE)); print(o)
# dR: L2 +30, L5 +25 (slots 1, 2); slot 3 = L1 (+20, |dX| <= 6); slot 4 = L4 (reference margin); slot 5 = L3 (not-longer dR +8 > L6 +6)
assert "shortlist layers: 2 5 1 4 3" in o, "E shortlist"
o = go("cells", str(run), "E", str(wE), "--layers", "1", "2", "5", "0"); print(o)
s1, s2, s5, s0 = (section(o, l) for l in (1, 2, 5, 0))
assert "(4) not-longer dR interval above 0: True  =>  PASS" in s1 and "E retention: d answered = +0, within +-6: True  =>  PASS" in s1
assert "=>  PASS" in s2.split("- E retention")[0] and "=>  not met" in s2.split("- E retention")[1]      # effect yes, retention no
assert "(1) dR interval above 0: True" in s5 and "(4) not-longer dR interval above 0: False  =>  not met" in s5   # format-like gain fails the guard
assert "=>  not met" in s0.split("- E retention")[0]
o = go("cells", str(run), "E", str(wEp), "--judge", "pro", "--layers", "1"); assert "=>  PASS" in o
o = go("screen", str(run), "C", str(wC)); print(o)
assert "shortlist layers: 1 3" in o, "C shortlist"
o = go("cells", str(run), "C", str(wC), "--layers", "1", "2", "0"); print(o)
assert "TWO-WAY PASS" in section(o, 1) and "ONE-WAY (x.w.p1)" in section(o, 2) and "C effect: not met" in section(o, 0)
go("cross", str(run), "E", str(wE), "--layers", "1")
print("synthetic checks passed")
