# Iteration_01 diagnostic (Claude Opus 5.5, 2026-11-17), read-only: no model, no GPU, no judge call.
# Step 2 of novelty/ours/2026-11-17_e_doubt_c_commit.md section 7: on the tune questions, does a doubt gate beat the current
# method under penalties for wrong answers? E is fixed at strength 1 (the only strength with readouts on tune). States and
# scoring as policy_select2.py (imported); doubt readings as doubt_signal.py (d = log p("I don't know.") with E minus without).
#   method      E 1; on the questions E leaves blank, C toward answering raised 0.25 -> 0.5 -> 1 -> 2 up to a cap
#   gate        questions whose reading exceeds the threshold are not answered (score 0); the rest as the method
#   gate2       two thresholds: blanks are released (C up to the cap) only at or below t_release; answered questions are
#               withheld above t_withhold
#   gate B      as gate, but the blanks at or below the threshold are asked again with the must-answer prompt under E 1
#   controls    gate with the reading d replaced by d_rand, a0, aE, and by -a0, -aE (red team 11-17: wrong answers start with a
#               LOWER a0, so the levels get both signs; the gate always withholds the questions whose reading is high)
# Thresholds: the reading's 90 / 80 / 70 / 60 / 50 % quantile over all tune questions of the model (or none); caps: none,
# 0.25, 0.5, 1, 2. Selection without overfitting the report: the 192 questions are split in half by a seeded shuffle, every
# setting is chosen on one half and scored on the other, then the halves swap and the two held-out scores are added; repeated
# for 20 seeds (20261117 ...). Ties go to the weaker setting (no threshold first, then the highest quantile; smaller cap first).
# Criterion written before the run (item 6): at penalty 1 or 3, gate minus method has a positive mean and is positive in at
# least 16 of the 20 seeds -> "holds on tune".
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/doubt_policy.py MODEL [MODEL ...]
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policy_select2 as P   # noqa: E402  (main() guarded: importing prints nothing)

LAMS = (0, 1, 3)
QS = (None, 0.9, 0.8, 0.7, 0.6, 0.5)          # quantile of the reading above which questions are withheld (None = no gate)
CAPS = (None,) + tuple(P.LAD)                  # C cap on the blanks (None = E alone)
SEEDS = tuple(range(20261117, 20261117 + 20))
READS = ("d", "d_rand", "a0", "aE", "-a0", "-aE")
TUNE_D = {"Gemma-4-31B": P.G31 + "/judge_tune_proD_E_L24"}   # tune must-answer work dir where policy_select2 has d=None


def readings(m, el):
    """Per-question readings from the run's mc files, checked against the tune grid as doubt_signal.py does."""
    run = m["ec"][0].split("/eval_extra/")[0]
    g = json.load(open(m["ec"][0] + "/judge_input.json", encoding="utf-8"))
    rows = {}
    for key, f in (("a0", "base_x_tune.json"), ("aE", "E_L%d_x-w-p1_tune.json" % el), ("aR", "E_L%d_x-rand-p1_tune.json" % el)):
        d = json.load(open(os.path.join(run, "mc", f), encoding="utf-8"))
        if d.get("identity_sha256") != g.get("identity_sha256"):
            raise SystemExit("%s: %s belongs to another run identity" % (run, f))
        r = {x["pair_id"]: x for x in d["rows"]}
        if len(r) != len(d["rows"]) or set(r) != set(g["question_ids"]):
            raise SystemExit("%s: %s and the grid hold different questions" % (run, f))
        if any(not x.get("abstain_scored") for x in d["rows"]):
            raise SystemExit("%s: %s has unscored readouts" % (run, f))
        if key == "aE":
            part = {c["name"]: c for c in g["condition_specs"]}["E_L%d_x-w-p1" % el]["parts"][0]
            if part["direction_file_sha256"] != d["direction_file_sha256"] or abs(float(part["amount"]) - float(d["write"])) > 1e-3:
                raise SystemExit("%s: the readout's E write differs from the grid's" % run)
        rows[key] = {p: float(x["abstain_logp"]) for p, x in r.items()}
    return {"d": {p: rows["aE"][p] - rows["a0"][p] for p in rows["a0"]}, "d_rand": {p: rows["aR"][p] - rows["a0"][p] for p in rows["a0"]},
            "a0": rows["a0"], "aE": rows["aE"], "-a0": {p: -v for p, v in rows["a0"].items()}, "-aE": {p: -v for p, v in rows["aE"].items()}}


class Tune:
    def __init__(self, name):
        self.m = P.MODELS[name]
        self.sp = P.Split(self.m, 0)                         # tune split: states, blank flags, must-answer states
        self.qs = list(self.sp.qs)
        self.r = readings(self.m, self.m["e"])
        missing = [p for p in self.qs if p not in self.r["d"]]
        if missing:
            raise SystemExit("%s: readings missing for %s" % (name, missing[:3]))
        self.thr = {k: {q: (None if q is None else sorted(v[p] for p in self.qs)[int(q * (len(self.qs) - 1))]) for q in QS}
                    for k, v in self.r.items()}
        sp, e = self.sp, self.sp.cE("1")
        self.sE = {p: sp.st[(p, e)] for p in self.qs}
        self.bE = {p: sp.blank[(p, e)] for p in self.qs}
        self.rel = {}                                        # state after releasing a blank with C up to each cap
        for p in self.qs:
            for cap in CAPS:
                s = self.sE[p]
                if self.bE[p] and cap is not None:
                    for k in P.LAD[:P.LAD.index(cap) + 1]:
                        c = sp.cEC("1", k)
                        if not sp.blank[(p, c)]:
                            s = sp.st[(p, c)]
                            break
                self.rel[(p, cap)] = s
        dst = sp.dst
        if dst is None and name in TUNE_D:
            dst = P.load(TUNE_D[name], self.m["judge"][0])[0]
        self.mustE = None if dst is None else {p: dst.get((p, sp.dE("1")), "?") for p in self.qs}
        if self.mustE is not None and any(v == "?" for v in self.mustE.values()):
            raise SystemExit("%s: must-answer E 1 verdicts missing on tune" % name)
        self.base = {p: sp.st[(p, "baseline_x")] for p in self.qs}

    def state(self, fam, cfg, p):
        if fam == "no write":
            return self.base[p]
        if fam == "E alone":
            return self.sE[p]
        if fam == "method":
            return self.rel[(p, cfg[0])]
        if fam in ("gate",) + tuple("gate:" + k for k in READS):
            k = "d" if fam == "gate" else fam.split(":")[1]
            t, cap = self.thr[k][cfg[0]], cfg[1]
            return "X" if (t is not None and self.r[k][p] > t) else self.rel[(p, cap)]
        if fam == "gate2":
            tr, tw, cap = self.thr["d"][cfg[0]], self.thr["d"][cfg[1]], cfg[2]
            if self.bE[p]:
                return "X" if (tr is not None and self.r["d"][p] > tr) else self.rel[(p, cap)]
            return "X" if (tw is not None and self.r["d"][p] > tw) else self.sE[p]
        if fam == "gate B":
            t = self.thr["d"][cfg[0]]
            if t is not None and self.r["d"][p] > t:
                return "X"
            return self.mustE[p] if self.bE[p] else self.sE[p]
        raise ValueError(fam)


FAMS = {  # candidate settings, weakest first (ties go to the earlier one)
    "no write": [()],
    "E alone": [()],
    "method": [(c,) for c in CAPS],
    "gate": [(q, c) for q in QS for c in CAPS],
    "gate2": [(qr, qw, c) for qr in QS for qw in QS for c in CAPS],
    "gate B": [(q,) for q in QS],
}
FAMS.update({"gate:" + k: FAMS["gate"] for k in READS if k != "d"})


def score(states, lam):
    return sum(s == "R" for s in states) - lam * sum(s == "W" for s in states)


def run_model(name):
    T = Tune(name)
    fams = {f: c for f, c in FAMS.items() if not (f == "gate B" and T.mustE is None)}
    out = ["\n## %s (tune %d questions, judge %s; E %d at strength 1; C %d)" % (name, len(T.qs), T.m["judge"][0], T.m["e"], T.m["c"])]
    for lam in LAMS:
        res = {f: [] for f in fams}                          # per seed: (held-out score, right, wrong, answered, chosen settings)
        for seed in SEEDS:
            order = list(T.qs)
            random.Random(seed).shuffle(order)
            halves = (order[:len(order) // 2], order[len(order) // 2:])
            for f, cands in fams.items():
                tot = [0.0, 0, 0, 0, []]
                for a, b in ((0, 1), (1, 0)):
                    sel, rep = halves[a], halves[b]
                    best = max(cands, key=lambda c: (score([T.state(f, c, p) for p in sel], lam), -cands.index(c)))
                    st = [T.state(f, best, p) for p in rep]
                    tot[0] += score(st, lam)
                    tot[1] += sum(s == "R" for s in st)
                    tot[2] += sum(s == "W" for s in st)
                    tot[3] += sum(s in "RW" for s in st)
                    tot[4].append(best)
                res[f].append(tot)
        out.append("- penalty %s per wrong answer (held-out score summed over both halves; mean [min, max] over %d seeds)" % (lam, len(SEEDS)))
        n = len(T.qs)
        for f in fams:
            v = [x[0] for x in res[f]]
            cov = sum(x[3] for x in res[f]) / (len(SEEDS) * n)
            risk = sum(x[2] for x in res[f]) / max(1, sum(x[3] for x in res[f]))
            out.append("    %-11s score %7.1f [%6.1f, %6.1f] | right %5.1f wrong %5.1f | coverage %.2f, wrong among answered %.2f" % (
                f, sum(v) / len(v), min(v), max(v), sum(x[1] for x in res[f]) / len(SEEDS), sum(x[2] for x in res[f]) / len(SEEDS), cov, risk))
        for a, b in (("gate", "method"), ("gate", "E alone"), ("gate2", "method"), ("gate", "gate B"), ("gate", "gate:d_rand"),
                     ("gate", "gate:a0"), ("gate", "gate:aE"), ("gate", "gate:-a0"), ("gate", "gate:-aE"), ("method", "E alone")):
            if a not in res or b not in res:
                continue
            dlt = [x[0] - y[0] for x, y in zip(res[a], res[b])]
            out.append("    %s minus %s: mean %+.1f [%+.1f, %+.1f], positive in %d / %d seeds" % (
                a, b, sum(dlt) / len(dlt), min(dlt), max(dlt), sum(x > 0 for x in dlt), len(dlt)))
        if lam in (1, 3):
            dlt = [x[0] - y[0] for x, y in zip(res["gate"], res["method"])]
            ok = sum(dlt) / len(dlt) > 0 and sum(x > 0 for x in dlt) >= 16
            out.append("    criterion at penalty %s (gate minus method: mean > 0 and positive in >= 16 / 20): %s" % (lam, "HOLDS" if ok else "does not hold"))
        picks = {}
        for x in res["gate"]:
            for c in x[4]:
                picks[c] = picks.get(c, 0) + 1
        top = sorted(picks.items(), key=lambda kv: -kv[1])[:3]
        out.append("    gate settings chosen most often (withheld above quantile / C cap): %s" % ", ".join(
            "%s / %s x%d" % ("none" if c[0] is None else "%d%%" % round(c[0] * 100), P.NAME[c[1]], k) for c, k in top))
    return out


def main():
    names = sys.argv[1:]
    if not names:
        sys.exit("name the models (only those judged usable in step 1)")
    bad = [n for n in names if n not in P.MODELS]
    if bad:
        sys.exit("unknown model(s) %s; known: %s" % (bad, list(P.MODELS)))
    out = ["# Doubt gate on the tune questions (step 2 of novelty/ours/2026-11-17_e_doubt_c_commit.md)"]
    for n in names:
        out += run_model(n)
    print("\n".join(out))


if __name__ == "__main__":
    main()
