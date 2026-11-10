# -*- coding: utf-8 -*-
"""Read-only: for prompt pairs a -> b (step 0, no intervention), split the questions by how much longer the answer under b is,
and report the right-count change in each stratum under both judges. Calibrates a 'not just longer answers' guard.
usage: e44_step0_length_strata.py WORK [--pairs COND_A:COND_B ...]"""
import collections, json, sys
from pathlib import Path
import random
work = Path(sys.argv[1])
gen = json.loads((work / "judge_input.json").read_text(encoding="utf-8"))
ids = gen["question_ids"]; ev = {r["pair_id"]: r for r in gen["evaluations"]}
state = {}
for judge in ("allenai", "pro"):
    for l in (work / "judge2" / ("single_%s.jsonl" % judge)).open(encoding="utf-8"):
        r = json.loads(l)
        s = ("X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]]) if judge == "allenai" else {"A": "R", "B": "W", "H": "H", "C": "X"}[r["label"]]
        state[(judge, r["pair_id"], r["condition"])] = s
nw = lambda i, c: len((ev[i]["outputs"][c] or "").split())  # noqa: E731
def boot(d, n_boot=10000, seed=20261111):
    rng = random.Random(seed); n = len(d)
    if n == 0: return 0, 0
    sums = sorted(sum(d[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return sums[int(0.025 * n_boot)], sums[int(0.975 * n_boot) - 1]
pairs = [("x", "x_F"), ("x", "x_E1"), ("conf", "conf_E1"), ("conf", "conf_E2"), ("conf_M", "conf_E1"), ("caut", "caut_E1")]
RAW = "--pairs" in sys.argv                       # --pairs COND_A:COND_B ... with full condition names (any export)
if RAW:
    pairs = [tuple(x.split(":")) for x in sys.argv[sys.argv.index("--pairs") + 1:]]
print("stratum 'not longer' = the answer under b has at most 1 word more than under a; 'longer' = at least 2 words more.")
print("dR = right-count change; the interval resamples ALL questions (questions outside the stratum contribute 0).\n")
print("| pair a -> b | judge | all: dR | not longer: n, dR [95%], W>R / R>W | longer: n, dR [95%], W>R / R>W | same text: n |")
print("|---|---|---|---|---|---|")
norm = lambda t: "".join(ch for ch in (t or "").lower() if ch.isalnum() or ch == " ").strip()  # noqa: E731
for a, b in pairs:
    A, B = (a, b) if RAW else ("baseline_" + a, "baseline_" + b)
    longer = {i: (nw(i, B) - nw(i, A)) >= 2 for i in ids}
    same = sum(norm(ev[i]["outputs"][A]) == norm(ev[i]["outputs"][B]) for i in ids)
    for judge in ("allenai", "pro"):
        cells = []
        for flag in (False, True):
            sub = [i for i in ids if longer[i] == flag]
            d = [((state[(judge, i, B)] == "R") - (state[(judge, i, A)] == "R")) if longer[i] == flag else 0 for i in ids]
            lo, hi = boot(d)
            up = sum(state[(judge, i, A)] == "W" and state[(judge, i, B)] == "R" for i in sub)
            down = sum(state[(judge, i, A)] == "R" and state[(judge, i, B)] == "W" for i in sub)
            cells.append("%d, %+d [%+d, %+d], %d / %d" % (len(sub), sum(d), lo, hi, up, down))
        tot = sum((state[(judge, i, B)] == "R") - (state[(judge, i, A)] == "R") for i in ids)
        print("| %s -> %s | %s | %+d | %s | %s | %d |" % (a, b, judge, tot, cells[0], cells[1], same))
