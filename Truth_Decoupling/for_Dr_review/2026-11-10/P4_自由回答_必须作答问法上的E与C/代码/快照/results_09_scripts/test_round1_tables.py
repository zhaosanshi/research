# -*- coding: utf-8 -*-
"""Synthetic check of e44_round1_tables.py (criteria v4: plans 28 + 29 + 30). Known effects must come out with the
right slot and status; the constructed false positives of the red team and of GPT rounds 30 and 31 must not pass.
Standard library only."""
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


def write(tmp, family, table, judge="allenai", tag=""):
    conds = ["baseline_x", "baseline_x_F"] + sorted({"baseline_" + arm.split(".")[0] for l in table for arm in table[l]} - {"baseline_x", "baseline_x_F"})
    specs = [{"name": c, "base_condition": c, "kind": "prompt_control"} for c in conds]
    st = {c: base for c in conds}; tx = {c: {q: SHORT for q in ids} for c in conds}
    for l in sorted(table):
        for arm, (s, t) in table[l].items():
            name = "%s_L%d_%s" % (family, l, arm.replace(".", "-"))
            conds.append(name)
            specs.append({"name": name, "base_condition": "baseline_" + arm.split(".")[0], "family": family, "layer": l, "arm": arm})
            st[name] = s; tx[name] = t
    work = Path(tmp) / ("work_%s_%s%s" % (family, judge, tag)); (work / "judge2").mkdir(parents=True)
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
    7: [("W", "R", 16, True), ("R", "W", 4, True), ("X", "R", 8, False), ("W", "X", 8, False)],   # red-team case: churn that cancels
}
E_format = {1: [("W", "R", 15, False), ("R", "W", 3, False)],                # the content effect survives on the format background
            5: [("W", "R", 2, False), ("R", "W", 2, False)]}                 # the format-like gain does not
E = {}
for l, moves in E_main.items():
    E[l] = {"x.w.p1": mutate(moves, seed=l)}
    E[l]["x.w.m1"] = mutate([("W", "R", 2, False), ("R", "W", 3, False)], seed=50 + l)
    E[l]["x.md.p1"] = mutate([("W", "R", 4, False), ("R", "W", 3, False)], seed=60 + l)
    E[l]["x.rand.p1"] = mutate([("W", "R", 3, False), ("R", "W", 4, False)], seed=70 + l)
    if l != 2:                                                               # layer 2 keeps a single random sign: no verdict possible
        E[l]["x.rand.m1"] = mutate([("W", "R", 4, False), ("R", "W", 3, False)], seed=80 + l)
    if l in E_format:
        E[l]["x_F.w.p1"] = mutate(E_format[l], seed=90 + l)
    (run / "mc" / ("E_L%d_x-w-p1_tune.json" % l)).write_text(json.dumps({"summary": {"margin_change_mean": 0.9 if l == 4 else 0.01 * l}}))
wE = write(tmp, "E", E); wEp = write(tmp, "E", E, judge="pro")
# ---------------------------------------------------------------- C layers
C_arms = {
    0: ([], []),                                                             # nothing
    1: ([("W", "X", 20, False)], [("X", "W", 15, False)]),                   # two-way: 20 fewer / 15 more answers, content untouched
    2: ([("W", "X", 30, False)], []),                                        # one-way toward caution
    3: ([("W", "X", 10, False)], [("X", "W", 8, False)]),                    # two-way, smaller
    4: ([("R", "H", 30, False), ("W", "X", 10, False)], []),                 # red-team case: 30 right answers turned into hedges
    5: ([], [("X", "H", 9, False)]),                                         # red-team case: abstentions -> hedges, nobody answers more
    6: ([("W", "X", 4, False)], [("X", "W", 4, False)]),                     # red-team case: 4 questions each way (inside tau)
    7: ([("W", "X", 20, False), ("W", "R", 13, False), ("R", "W", 13, False)], [("X", "W", 15, False)]),   # red-team case: 26 offsetting flips
}


def c_table(arms, rand=(("W", "X", 1, False), ("X", "W", 1, False))):
    out = {}
    for l, (p, m) in arms.items():
        out[l] = {"x.w.p1": mutate(p, seed=l), "x.w.m1": mutate(m, seed=100 + l),
                  "x.md.p1": mutate([("W", "X", 2, False)], seed=9 + l), "x.md.m1": mutate([("X", "W", 2, False)], seed=19 + l),
                  "x.rand.p1": mutate([rand[0]], seed=20 + l), "x.rand.m1": mutate([rand[1]], seed=30 + l)}
    return out


wC = write(tmp, "C", c_table(C_arms))
# the random arm itself adds 12 answers on one sign and removes 12 on the other (sign trap)
wC2 = write(tmp, "C", c_table({0: ([("W", "X", 9, False)], [])}, rand=(("X", "W", 12, False), ("W", "X", 12, False))), tag="_trap")
# two-way layer whose caution arm damages content (20 right -> wrong among the kept answers) + a clean one-way layer (GPT round 30)
wC3 = write(tmp, "C", c_table({0: ([("W", "X", 20, False), ("R", "W", 20, False)], [("X", "W", 15, False)]),
                               1: ([("W", "X", 30, False)], [])}), tag="_fallback")
# GPT round 31: (0) the learned arm gains +20 but one random sign gains +25; (1) on the complete-sentence background the "gain" is only abstentions answered
E2 = {}
for l in (0, 1):
    E2[l] = {"x.w.p1": mutate([("W", "R", 25, False), ("R", "W", 5, False)], seed=l),
             "x.rand.p1": mutate([("W", "R", 30, False), ("R", "W", 5, False)] if l == 0 else [("W", "R", 3, False), ("R", "W", 4, False)], seed=70 + l),
             "x.rand.m1": mutate([("W", "R", 4, False), ("R", "W", 3, False)], seed=80 + l),
             "x_F.w.p1": mutate([("W", "R", 15, False), ("R", "W", 3, False)] if l == 0 else [("X", "R", 10, False)], seed=90 + l)}
wE2 = write(tmp, "E", E2, tag="_r31")
# everything refused under the caution arm: nothing is answered both times, so retention is undefined (never 0)
wC4 = write(tmp, "C", c_table({0: ([("W", "X", 88, False), ("R", "X", 78, False)], [])}), tag="_empty")
# GPT round 33: on another base prompt the complete-sentence condition is not tested, so the status must stay incomplete
wE3 = write(tmp, "E", {0: {"conf.w.p1": mutate([("W", "R", 25, False), ("R", "W", 5, False)], seed=1),
                          "conf.rand.p1": mutate([("W", "R", 3, False), ("R", "W", 4, False)], seed=71),
                          "conf.rand.m1": mutate([("W", "R", 4, False), ("R", "W", 3, False)], seed=81)}}, tag="_otherbase")
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
# dR among layers with an answered-rate change inside the tolerance: L5 +25, L1 +20 (tie with L7 broken by layer) -> slots 1-2;
# slot 3 = largest G_same among the rest (L3, +8); slot 4 = reference margin (L4); slot 5 = largest raw dR left (L2, +30): diagnostic only
assert "shortlist layers: 5 1 3 4 2" in o and "layers to be qualified: 5 1 3 4" in o, "E shortlist"
assert "not a gate" in o
o = go("cells", str(run), "E", str(wE), "--layers", "1", "2", "5", "7", "0"); print(o)
s = {l: section(o, l) for l in (1, 2, 5, 7, 0)}
assert "A=met B=met" in line(s[1], "KEY")
assert "MISSING x.rand.m1" in s[2] and "A=not_met" in line(s[2], "KEY")                 # one random sign missing AND coverage changed
assert "A=not_met" in line(s[5], "KEY") and "inside the tolerance: False" in line(s[5], "- F |")   # format-like gain: nothing on the x_F background
assert "A=incomplete B=not_met" in line(s[7], "KEY") and "moved out / in 8 / 8" in line(s[7], "- retention on x")   # churn: no B retention, F not judged
assert "A=not_met" in line(s[0], "KEY")
o = go("cells", str(run), "E", str(wEp), "--judge", "pro", "--layers", "1"); assert "A=met B=met" in line(o, "KEY")
o = go("cells", str(run), "E", str(wE2), "--layers", "0", "1"); print(o)
s = {l: section(o, l) for l in (0, 1)}
assert "A=not_met" in line(s[0], "KEY") and "above each (point): False" in line(s[0], "- random controls")       # a random sign does better
assert "A=not_met" in line(s[1], "KEY") and "inside the tolerance: False" in line(s[1], "- F |")                # x_F gain is only more answering
o = go("screen", str(run), "C", str(wC)); print(o)
assert "shortlist layers: 1 7 3" in o and "layers to be qualified: 1 7" in o, "C shortlist"
o = go("cells", str(run), "C", str(wC), "--layers", "1", "2", "4", "5", "6", "7", "0"); print(o)
s = {l: section(o, l) for l in (1, 2, 4, 5, 6, 7, 0)}
assert "A=TWO-WAY B=TWO-WAY" in line(s[1], "KEY")
assert "A=ONE-WAY(x.w.p1) B=ONE-WAY(x.w.p1)" in line(s[2], "KEY")
assert "A=not_met" in line(s[4], "KEY") and "A_effect_only=ONE-WAY(x.w.p1)" in line(s[4], "KEY")    # 30 answers turned into hedges
assert "A=not_met" in line(s[5], "KEY")                                         # abstained -> hedge is not answering
assert "A=not_met" in line(s[6], "KEY")                                         # 4 questions each way stays inside the tolerance
assert "A=TWO-WAY B=ONE-WAY(x.w.m1)" in line(s[7], "KEY") and "W>R 13 / R>W 13" in s[7]   # offsetting flips: point inside, interval not
assert "A=not_met" in line(s[0], "KEY")
o = go("cells", str(run), "C", str(wC2), "--layers", "0"); print(o)
assert "A=not_met" in line(o, "KEY")                                            # -9 does not beat the random arm's own -12
o = go("screen", str(run), "C", str(wC3)); print(o)
assert "one-way layers below" in o and "layers to be qualified: 1 0" in o, "one-way fallback"
o = go("cells", str(run), "C", str(wC4), "--layers", "0"); print(o)
assert "right-rate change undefined" in o and "A=not_met" in line(o, "KEY") and "A_effect_only=ONE-WAY(x.w.p1)" in line(o, "KEY")
o = go("screen", str(run), "C", str(wC4)); assert "undefined" in o
o = go("cells", str(run), "E", str(wE3), "--layers", "0", "--base", "conf"); print(o)
assert "A=incomplete" in line(o, "KEY") and "tested_A_without_F=met" in line(o, "KEY") and "NOT TESTED" in o
go("cross", str(run), "E", str(wE), "--layers", "1")
print("synthetic checks passed")
