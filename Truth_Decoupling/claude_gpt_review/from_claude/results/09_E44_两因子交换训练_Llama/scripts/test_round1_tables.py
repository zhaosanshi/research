# -*- coding: utf-8 -*-
"""Synthetic check of e44_round1_tables.py (criteria v2, plan 28): known effects must come out with the right slot and verdict,
and the constructed false positives found by the red team must be stopped. Standard library only."""
import json, random, subprocess, sys, tempfile
from pathlib import Path
N = 192
ids = ["q%d" % i for i in range(N)]
base = {q: ("R" if i < 78 else "W" if i < 166 else "H" if i < 169 else "X") for i, q in enumerate(ids)}   # 78 R, 88 W, 3 H, 23 X
SHORT, LONG = "a b c", "a b c d e f g h i"                                                              # 3 words, 9 words


def mutate(moves, seed=0):
    """moves: (from_state, to_state, count, lengthen). Each question changes at most once. Returns (states, texts)."""
    r = random.Random(seed); out = dict(base); text = {q: SHORT for q in ids}; used = set()
    for a, b, k, lengthen in moves:
        pool = [q for q in ids if base[q] == a and q not in used]
        for q in r.sample(pool, k):
            out[q] = b; used.add(q)
            if lengthen:
                text[q] = LONG
    return out, text


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
# ---------------------------------------------------------------- E layers
E_main = {
    0: [],                                                                   # nothing
    1: [("W", "R", 25, False), ("R", "W", 5, False)],                        # content effect at the same length: +20
    2: [("W", "R", 20, False), ("R", "W", 5, False), ("X", "R", 15, False)],  # +30 but 15 abstentions answered (coverage change)
    3: [("W", "R", 12, False), ("R", "W", 4, False)],                        # +8
    4: [("W", "R", 3, False), ("R", "W", 2, False)],                         # +1, best reference margin
    5: [("W", "R", 30, True), ("R", "W", 5, True)],                          # +25, every changed answer is long (format-like)
    6: [("W", "R", 8, False), ("R", "W", 2, False)],                         # +6
    7: [("W", "R", 16, True), ("R", "W", 4, True), ("X", "R", 8, False), ("W", "X", 8, False)],   # red-team case: long flips + 8 abstentions -> short right + 8 answers -> abstained
}
E = {}
for l, moves in E_main.items():
    E[l] = {"x.w.p1": mutate(moves, seed=l)}
    E[l]["x.w.m1"] = mutate([("W", "R", 2, False), ("R", "W", 3, False)], seed=50 + l)
    E[l]["x.md.p1"] = mutate([("W", "R", 4, False), ("R", "W", 3, False)], seed=60 + l)
    E[l]["x.rand.p1"] = mutate([("W", "R", 3, False), ("R", "W", 4, False)], seed=70 + l)
    (run / "mc" / ("E_L%d_x-w-p1_tune.json" % l)).write_text(json.dumps({"summary": {"margin_change_mean": 0.9 if l == 4 else 0.01 * l}}))
wE = write(tmp, "E", E); wEp = write(tmp, "E", E, judge="pro")
# ---------------------------------------------------------------- C layers
C_arms = {
    0: ([], []),                                                             # nothing
    1: ([("W", "X", 20, False)], [("X", "W", 15, False)]),                   # two-way: 20 fewer / 15 more answers, correctness kept
    2: ([("W", "X", 30, False)], []),                                        # one-way toward caution
    3: ([("W", "X", 10, False)], [("X", "W", 8, False)]),                    # two-way, smaller
    4: ([("R", "H", 30, False), ("W", "X", 10, False)], []),                 # red-team case: 30 right answers turned into hedges
    5: ([], [("X", "H", 9, False)]),                                         # red-team case: abstentions -> hedges, nobody answers more
    6: ([("W", "X", 4, False)], [("X", "W", 4, False)]),                     # red-team case: 4 questions each way (inside tau)
    7: ([("W", "X", 20, False), ("W", "R", 13, False), ("R", "W", 13, False)], [("X", "W", 15, False)]),   # red-team case: 26 offsetting flips
}
C = {}
for l, (p, m) in C_arms.items():
    C[l] = {"x.w.p1": mutate(p, seed=l), "x.w.m1": mutate(m, seed=100 + l),
            "x.md.p1": mutate([("W", "X", 2, False)], seed=9 + l), "x.md.m1": mutate([("X", "W", 2, False)], seed=19 + l),
            "x.rand.p1": mutate([("W", "X", 1, False)], seed=20 + l), "x.rand.m1": mutate([("X", "W", 1, False)], seed=30 + l)}
wC = write(tmp, "C", C)
# a C layer where the random arm itself removes 12 answers on one sign and adds 12 on the other (the red team's sign trap)
C2 = {0: {"x.w.p1": mutate([("W", "X", 9, False)], seed=1), "x.w.m1": mutate([], seed=2),
          "x.md.p1": mutate([], seed=3), "x.md.m1": mutate([], seed=4),
          "x.rand.p1": mutate([("X", "W", 12, False)], seed=5), "x.rand.m1": mutate([("W", "X", 12, False)], seed=6)}}
wC2 = write(str(Path(tmp) / "c2"), "C", C2) if (Path(tmp) / "c2").mkdir() is None else None
S = str(Path(__file__).with_name("e44_round1_tables.py"))


def go(*args):
    out = subprocess.run([sys.executable, S, *args], capture_output=True, text=True)
    if out.returncode:
        print(out.stdout); print(out.stderr); raise SystemExit("FAILED " + " ".join(args))
    return out.stdout


def section(text, layer):
    return text.split("## layer %d\n" % layer)[1].split("## layer")[0]


def line(text, start):
    return [x for x in text.splitlines() if x.startswith(start)][0]


o = go("screen", str(run), "E", str(wE)); print(o)
# G_same among |dAns| <= 6: L1 +20, L3 +8, L6 +6 (slots 1-3); slot 4 = L4 (reference margin); slot 5 = L2 (largest raw dR, +30)
assert "shortlist layers: 1 3 6 4 2" in o, "E shortlist"
o = go("cells", str(run), "E", str(wE), "--layers", "1", "2", "5", "7", "0"); print(o)
s = {l: section(o, l) for l in (1, 2, 5, 7, 0)}
assert line(s[1], "- E effect").endswith("=>  PASS") and "=>  PASS (question by question as well" in line(s[1], "- E retention")
assert line(s[2], "- E effect").endswith("=>  PASS") and line(s[2], "- E retention").endswith("=>  not met")       # coverage change
assert "(e1) dR +25 above tau and interval above 0: True" in s[5] and line(s[5], "- E effect").endswith("=>  not met")   # format-like
assert "interval above 0: False; (e3)" in line(s[5], "- E effect")
assert line(s[7], "- E effect").endswith("=>  not met") and "rate only" in line(s[7], "- E retention")             # red-team case stopped
assert line(s[0], "- E effect").endswith("=>  not met")
assert "effect=True" in line(s[1], "KEY") and "effect=False" in line(s[7], "KEY")
o = go("cells", str(run), "E", str(wEp), "--judge", "pro", "--layers", "1"); assert line(o, "- E effect").endswith("=>  PASS")
o = go("screen", str(run), "C", str(wC)); print(o)
# two-way T > 6: L1 (15), L7 (15, but 26 flips keep dR 0 -> retention met by net; contrast slot), L3 (8). Retention-met slots: 1, 7 (ties by layer); slot 3: 3
assert "shortlist layers: 1 7 3" in o, "C shortlist"
o = go("cells", str(run), "C", str(wC), "--layers", "1", "2", "4", "5", "6", "7", "0"); print(o)
s = {l: section(o, l) for l in (1, 2, 4, 5, 6, 7, 0)}
assert "TWO-WAY PASS" in s[1] and "retention on the arm(s) that count: PASS" in s[1]
assert "ONE-WAY (x.w.p1)" in s[2]
assert "ONE-WAY (x.w.p1)" in s[4] and "retention on the arm(s) that count: not met" in s[4]          # 30 answers turned into hedges
assert "C effect: not met" in s[5]                                                                   # abstained -> hedge is not answering
assert "C effect: not met" in s[6]                                                                   # 4 questions each way stays inside tau
assert "TWO-WAY PASS" in s[7] and "exceed tau: content changed question by question" in s[7]         # offsetting flips are flagged
assert "C effect: not met" in s[0]
o = go("cells", str(run), "C", str(Path(tmp) / "c2" / "work_C_allenai"), "--layers", "0"); print(o)
assert "C effect: not met" in o                                                                      # +9 does not beat the random arm's -12 sign
go("cross", str(run), "E", str(wE), "--layers", "1")
print("synthetic checks passed")
