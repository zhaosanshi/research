# -*- coding: utf-8 -*-
"""Read-only: the criteria-v2 quantity G_same for any two conditions of a judged export (plan 28, section 2.1):
dR restricted to the questions answered (right or wrong) under both conditions whose answer under b has at most one more
word than under a. Same states and the same bootstrap as e44_round1_tables.py. CPU, standard library only.
usage: e44_same_length_v2.py WORK COND_A:COND_B [COND_A:COND_B ...]"""
import collections, json, random, sys
from pathlib import Path
BOOT, SEED, LEN_TOL = 10000, 20261111, 1
work = Path(sys.argv[1]); pairs = [tuple(x.split(":")) for x in sys.argv[2:]]
gen = json.loads((work / "judge_input.json").read_text(encoding="utf-8"))
ids = gen["question_ids"]; ev = {r["pair_id"]: r for r in gen["evaluations"]}
state = {}
for judge in ("allenai", "pro"):
    for l in (work / "judge2" / ("single_%s.jsonl" % judge)).open(encoding="utf-8"):
        r = json.loads(l)
        s = ("X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]]) if judge == "allenai" else {"A": "R", "B": "W", "H": "H", "C": "X"}[r["label"]]
        state[(judge, r["pair_id"], r["condition"])] = s
nw = lambda i, c: len((ev[i]["outputs"][c] or "").split())  # noqa: E731
rng = random.Random(SEED); n = len(ids)
IDX = [[rng.randrange(n) for _ in range(n)] for _ in range(BOOT)]
def ci(d):
    sums = sorted(sum(d[j] for j in row) for row in IDX)
    return "%+d [%+d, %+d]" % (sum(d), sums[int(0.025 * BOOT)], sums[int(0.975 * BOOT) - 1])
print("answered = right or wrong; `same` = answered under both, and at most %d more word(s) under b; `longer` = answered under both, more words; tau = %d\n" % (LEN_TOL, int(0.03 * n + 0.5)))
print("| a -> b | judge | all: dR [95%] | d answered | out / in | same: n, G_same [95%], W>R / R>W | answered both but longer: n, dR [95%], W>R / R>W |")
print("|---|---|---|---|---|---|---|")
for a, b in pairs:
    for judge in ("allenai", "pro"):
        A = lambda i: state[(judge, i, a)]; B = lambda i: state[(judge, i, b)]  # noqa: E731
        ans = lambda s: s in "RW"  # noqa: E731
        dR = [(B(i) == "R") - (A(i) == "R") for i in ids]
        both = [ans(A(i)) and ans(B(i)) for i in ids]
        short = [both[k] and (nw(i, b) - nw(i, a)) <= LEN_TOL for k, i in enumerate(ids)]
        same = [dR[k] if short[k] else 0 for k in range(n)]
        longer = [dR[k] if (both[k] and not short[k]) else 0 for k in range(n)]
        up = lambda mask: sum(1 for k, i in enumerate(ids) if mask[k] and A(i) == "W" and B(i) == "R")  # noqa: E731
        dn = lambda mask: sum(1 for k, i in enumerate(ids) if mask[k] and A(i) == "R" and B(i) == "W")  # noqa: E731
        lg = [both[k] and not short[k] for k in range(n)]
        print("| %s -> %s | %s | %s | %+d | %d / %d | %d, %s, %d / %d | %d, %s, %d / %d |" % (
            a.replace("baseline_", ""), b.replace("baseline_", ""), judge, ci(dR), sum(ans(B(i)) - ans(A(i)) for i in ids),
            sum(ans(A(i)) and not ans(B(i)) for i in ids), sum((not ans(A(i))) and ans(B(i)) for i in ids),
            sum(short), ci(same), up(short), dn(short), sum(lg), ci(longer), up(lg), dn(lg)))
