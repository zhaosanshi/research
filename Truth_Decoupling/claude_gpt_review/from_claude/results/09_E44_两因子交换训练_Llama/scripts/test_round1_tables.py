# -*- coding: utf-8 -*-
"""Synthetic check of e44_round1_tables.py (criteria v3: plan 28 as amended by plan 29). Known effects must come out with the
right slot and verdict; the constructed false positives of the red team and of GPT round 30 must not pass.
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
    conds = ["baseline_x", "baseline_x_F"]
    specs = [{"name": c, "base_condition": c, "kind": "prompt_control"} for c in conds]
    st = {c: base for c in conds}; tx = {c: {q: SHORT for q in ids} for c in conds}
    for l in sorted(table):
        for arm, (s, t) in table[l].items():
            name = "%s_L%d_%s" % (family, l, arm.replace(".", "-"))
            conds.append(name)
            specs.append({"name": name, "base_condition": "baseline_x_F" if arm.startswith("x_F.") else "baseline_x", "family": family, "layer": l, "arm": arm})
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
# dR among |dAns| <= 6: L5 +25, L1 +20 (tie with L7 broken by layer) -> slots 1-2; slot 3 = largest G_same among the rest (L3, +8);
# slot 4 = reference margin (L4); slot 5 = largest raw dR left (L2, +30, coverage changed): diagnostic only
assert "shortlist layers: 5 1 3 4 2" in o and "candidate layers: 5 1 3 4" in o, "E shortlist"
assert "not a gate" in o
o = go("cells", str(run), "E", str(wE), "--layers", "1", "2", "5", "7", "0"); print(o)
s = {l: section(o, l) for l in (1, 2, 5, 7, 0)}
assert line(s[1], "- E effect").endswith("=>  PASS") and line(s[1], "- E against the format control").endswith("=>  PASS")
assert "point inside +-3 pp: True" in line(s[1], "- E retention") and "effect=PASS format=PASS" in line(s[1], "KEY")
assert "INCOMPLETE" in line(s[2], "- E effect") and "effect=INCOMPLETE" in line(s[2], "KEY")           # one random sign only: no verdict
assert "point inside +-3 pp: False" in line(s[2], "- E retention")                                       # coverage changed
assert line(s[5], "- E effect").endswith("=>  PASS") and line(s[5], "- E against the format control").endswith("=>  not met")   # format-like gain
assert "format=INCOMPLETE" in line(s[7], "KEY") and "moved out / in: 8 / 8" in line(s[7], "- E retention")  # churn reported, no format verdict
assert "(equivalence, needed for a stage-B claim): False" in line(s[7], "- E retention")
assert line(s[0], "- E effect").endswith("=>  not met")
o = go("cells", str(run), "E", str(wEp), "--judge", "pro", "--layers", "1"); assert line(o, "- E effect").endswith("=>  PASS")
o = go("screen", str(run), "C", str(wC)); print(o)
assert "shortlist layers: 1 7 3" in o and "candidate layers: 1 7" in o, "C shortlist"
o = go("cells", str(run), "C", str(wC), "--layers", "1", "2", "4", "5", "6", "7", "0"); print(o)
s = {l: section(o, l) for l in (1, 2, 4, 5, 6, 7, 0)}
assert "effect=TWO-WAY retention=equivalent" in line(s[1], "KEY")
assert "effect=ONE-WAY(x.w.p1)" in line(s[2], "KEY")
assert "effect=ONE-WAY(x.w.p1) retention=not_met" in line(s[4], "KEY")        # 30 answers turned into hedges
assert "effect=not_met" in line(s[5], "KEY")                                  # abstained -> hedge is not answering
assert "effect=not_met" in line(s[6], "KEY")                                  # 4 questions each way stays inside tau
assert "effect=TWO-WAY retention=point_inside" in line(s[7], "KEY") and "W>R 13 / R>W 13" in s[7]   # offsetting flips: not equivalent, shown
assert "effect=not_met" in line(s[0], "KEY")
o = go("cells", str(run), "C", str(wC2), "--layers", "0"); print(o)
assert "effect=not_met" in line(o, "KEY")                                     # -9 does not beat the random arm's own -12
o = go("screen", str(run), "C", str(wC3)); print(o)
assert "one-way candidates below" in o and "candidate layers: 1 0" in o, "one-way fallback"
go("cross", str(run), "E", str(wE), "--layers", "1")
print("synthetic checks passed")
