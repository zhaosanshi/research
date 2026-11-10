#!/usr/bin/env python3
"""Read-only: does the learned coordinate track behaviour, per question? (CPU, saved records + frozen_train.json only)
C: within the cautious prompt, AUC of the coordinate for "the frozen answer is a refusal"; paired coordinate difference
   for switch pairs (confident answers, cautious refuses) vs the other pairs.
E: AUC of the paired difference for "x_E changed the answer"; per-pair gold CE by whether the gold answer has a
   negation word. Both for the learned direction w and for the mean-difference direction.
usage: e43_pair_audit.py RUN_DIR
"""
import json, re, sys
from pathlib import Path
import torch

run = Path(sys.argv[1])
frozen = json.loads((run / "frozen_train.json").read_text())
fq = {r["pair_id"]: r for r in frozen["questions"]}
NEG = re.compile(r"\b(no|not|never|none|nothing|nobody|neither|nor|cannot|without|isn't|aren't|wasn't|weren't|don't|doesn't|"
                 r"didn't|won't|wouldn't|can't|couldn't|shouldn't|hasn't|haven't|hadn't)\b|n't\b", re.I)
norm = lambda t: " ".join((t or "").casefold().split()).rstrip(".!? ")


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ms(v):
    t = torch.tensor(v, dtype=torch.float64)
    return "%+.2f±%.2f" % (float(t.mean()), float(t.std(unbiased=False))) if len(v) else "—"


print("## C: does the coordinate track the refusal decision? (96 training questions)")
print("| L | switch pairs n | diff switch (w) | diff other (w) | AUC z_caut→refuse (w) | diff switch (md) | diff other (md) | AUC z_caut→refuse (md) | CE+ switch / other | CE- switch / other |")
print("|---|---|---|---|---|---|---|---|---|---|")
for L in range(32):
    p = run / "dirs" / ("C_L%d.pt" % L)
    if not p.exists():
        continue
    r = torch.load(p, map_location="cpu", weights_only=False)
    ids = r["pair_ids"]
    ref_caut = [bool(fq[i]["answers"]["x_caut"]["refusal_rule"]) for i in ids]
    ref_conf = [bool(fq[i]["answers"]["x_conf"]["refusal_rule"]) for i in ids]
    switch = [a and not b for a, b in zip(ref_caut, ref_conf)]
    out = []
    for key in ("endpoints", "endpoints_md"):
        zp = r[key]["plus"]["values"]; zm = r[key]["minus"]["values"]      # plus = x_caut, minus = x_conf
        d = [a - b for a, b in zip(zp, zm)]
        sgn = 1.0 if sum(d) >= 0 else -1.0
        d = [sgn * x for x in d]
        out += [ms([x for x, s in zip(d, switch) if s]), ms([x for x, s in zip(d, switch) if not s]),
                "%.2f" % auc([sgn * z for z in zp], ref_caut)]
    pp = {x["pair_id"]: x for x in r["per_pair_ce_after"]}
    cs = lambda k, flag: sum(pp[i][k] for i, s in zip(ids, switch) if s == flag) / max(1, sum(1 for s in switch if s == flag))
    print("| %d | %d | %s | %s | %s | %s | %s | %s | %.2f / %.2f | %.2f / %.2f |" % (
        L, sum(switch), out[0], out[1], out[2], out[3], out[4], out[5],
        cs("ce_plus", True), cs("ce_plus", False), cs("ce_minus", True), cs("ce_minus", False)))

print("\n## E: does the coordinate shift track what the instruction changes? (89 training questions)")
print("| L | changed n | diff changed (w) | diff same (w) | AUC diff→changed (w) | AUC diff→changed (md) | gold CE+ neg-gold / other | CE- changed / same | pairs CE+<0.5 |")
print("|---|---|---|---|---|---|---|---|---|")
for L in range(32):
    p = run / "dirs" / ("E_L%d.pt" % L)
    if not p.exists():
        continue
    r = torch.load(p, map_location="cpu", weights_only=False)
    ids = r["pair_ids"]
    changed = [norm(fq[i]["answers"]["x_E"]["text"]) != norm(fq[i]["answers"]["x"]["text"]) for i in ids]
    gneg = [bool(NEG.search(fq[i].get("best_answer") or "")) for i in ids]
    res = []
    for key in ("endpoints", "endpoints_md"):
        zp = r[key]["plus"]["values"]; zm = r[key]["minus"]["values"]      # plus = x_E, minus = x
        d = [a - b for a, b in zip(zp, zm)]
        sgn = 1.0 if sum(d) >= 0 else -1.0
        d = [sgn * x for x in d]
        res.append((ms([x for x, c in zip(d, changed) if c]), ms([x for x, c in zip(d, changed) if not c]), auc(d, changed)))
    pp = {x["pair_id"]: x for x in r["per_pair_ce_after"]}
    m = lambda k, flags, want: sum(pp[i][k] for i, f in zip(ids, flags) if f == want) / max(1, sum(1 for f in flags if f == want))
    print("| %d | %d | %s | %s | %.2f | %.2f | %.2f / %.2f | %.2f / %.2f | %d |" % (
        L, sum(changed), res[0][0], res[0][1], res[0][2], res[1][2],
        m("ce_plus", gneg, True), m("ce_plus", gneg, False), m("ce_minus", changed, True), m("ce_minus", changed, False),
        sum(1 for i in ids if pp[i]["ce_plus"] < 0.5)))
print("\ngold with negation word: %d of %d E pairs" % (sum(gneg), len(gneg)))
