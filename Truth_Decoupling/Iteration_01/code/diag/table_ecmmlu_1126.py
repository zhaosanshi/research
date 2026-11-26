"""MMLU by option probability with E and C written together: read-only table (2026-11-26, executor wu, Claude Opus 5.5).

Plan: discussion/work_queue/2026-11-26_queue_o.md, section 5. Input: the cap_mcq_sweep.py outputs of the five models
(Iteration_01/capability/ecmmlu_1126/<model>.json) and that program's own report text. No model, no card.

Per model and E strength (main s, larger s2): accuracy with no write; change vs no write in points for single E and for
E + C (toward answering, sign m) at 0.5 / 1 / 2; the paired difference "E + C minus single E" in points with a 95%
interval (items resampled, seed 20261126, 2,000 resamples, percentile indices as cap_mcq_sweep.ci); the single-C arm
vs no write. Checks before printing: every arm covers the same items as x; the writes recorded in each arm are the
planned layers / strengths with C negative; JSON accuracies equal recomputed ones; the "single E vs no write"
accuracy and change equal the report's printed numbers. At the end: the plan's pre-registered reading as plain counts.

usage: python3 -I table_ecmmlu_1126.py --report REPORT_TXT MODEL_JSON...
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

SPEC = {  # model: (E layer, C layer, main E strength, larger E strength)
    "llama3.1-8b": (16, 19, "3.5", "5"),
    "qwen3.5-9b": (15, 16, "2", "3"),
    "qwen3.8-27b": (28, 33, "1.75", "3"),
    "gemma3-12b": (24, 25, "2.5", "4"),
    "gemma3-27b": (28, 31, "1", "2"),
}
ORDER = list(SPEC)
LEVELS = ("0.5", "1", "2")
SEED, NBOOT = 20261126, 2000


def die(msg):
    sys.exit("table_ecmmlu_1126: " + msg)


def correct(d, arm, ids):
    rows = d["arms"][arm]["rows"]
    if [r["id"] for r in rows] != ids:
        die("%s %s: items differ from the no-write arm" % (d["model"], arm))
    return [int(r["pick"] == r["gold"]) for r in rows]


def check_writes(d, arm, plan):
    got = [(w["family"], w["layer"], float(w["beta"])) for w in d["arms"][arm]["writes"]]
    if got != plan:
        die("%s %s: recorded writes %s, planned %s" % (d["model"], arm, got, plan))
    if d["arms"][arm]["positions"] != "content":
        die("%s %s: write position %s" % (d["model"], arm, d["arms"][arm]["positions"]))


def boot(v, rng):
    n = len(v)
    idx = range(n)
    bs = sorted(sum(v[i] for i in rng.choices(idx, k=n)) / n for _ in range(NBOOT))
    return sum(v) / n, bs[int(0.025 * NBOOT)], bs[int(0.975 * NBOOT) - 1]


def parse_report(path):
    """{model: {arm without 'x:': (accuracy text, change text)}} and {model: no-write accuracy text}."""
    out, base, cur = {}, {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^== (\S+) \((\d+) items\), no write (\d\.\d{3})$", line)
        if m:
            cur = m.group(1)
            out[cur], base[cur] = {}, m.group(3)
            continue
        m = re.match(r"^    (\S+)\s+(\d\.\d{3}) \(([+-]\d+\.\d) \[", line)
        if m and cur:
            out[cur][m.group(1)] = (m.group(2), m.group(3))
    return out, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    data = {}
    for p in a.paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if d["model"] in data:
            die("model %s given twice" % d["model"])
        data[d["model"]] = d
    if sorted(data) != sorted(ORDER):
        die("expected the five models %s, got %s" % (ORDER, sorted(data)))
    if len({d["ids_file_sha256"] for d in data.values()}) != 1 or len({d["items_file_sha256"] for d in data.values()}) != 1:
        die("the models were run on different item / id files")
    rep, rep_base = parse_report(a.report)
    rng = random.Random(SEED)
    res = {}
    for name in ORDER:
        d, (el, cl, s1, s2) = data[name], SPEC[name]
        if d.get("skipped"):
            die("%s: skipped arms %s" % (name, d["skipped"]))
        arms = ["x", "x:C%d.w.m2" % cl]
        for s in (s1, s2):
            arms += ["x:E%d.w.p%s" % (el, s)] + ["x:E%d.w.p%s+C%d.w.m%s" % (el, s, cl, k) for k in LEVELS]
        if list(d["arms"]) != arms:
            die("%s: arms %s, planned %s" % (name, list(d["arms"]), arms))
        ids = [r["id"] for r in d["arms"]["x"]["rows"]]
        if len(ids) != d["n"] or len(set(ids)) != len(ids):
            die("%s: %d rows for n = %d (or repeated ids)" % (name, len(ids), d["n"]))
        cor = {arm: correct(d, arm, ids) for arm in arms}
        for arm in arms:
            if abs(sum(cor[arm]) / len(ids) - d["arms"][arm]["accuracy"]) > 1e-12:
                die("%s %s: stored accuracy differs from the recomputed one" % (name, arm))
        check_writes(d, "x", [])
        check_writes(d, arms[1], [("C", cl, -2.0)])
        for s in (s1, s2):
            check_writes(d, "x:E%d.w.p%s" % (el, s), [("E", el, float(s))])
            for k in LEVELS:
                check_writes(d, "x:E%d.w.p%s+C%d.w.m%s" % (el, s, cl, k), [("E", el, float(s)), ("C", cl, -float(k))])
        n = len(ids)
        acc = {arm: sum(cor[arm]) / n for arm in arms}
        chg = {arm: 100 * (acc[arm] - acc["x"]) for arm in arms}
        # check against the program's own report: no-write accuracy and single E (and single C) accuracy / change
        if name not in rep or rep_base[name] != "%.3f" % acc["x"]:
            die("%s: no-write accuracy differs from the report" % name)
        for arm in [arms[1]] + ["x:E%d.w.p%s" % (el, s) for s in (s1, s2)]:
            key = arm.split(":", 1)[1]
            want = ("%.3f" % acc[arm], "%+.1f" % chg[arm])
            if rep[name].get(key) != want:
                die("%s %s: table %s, report %s" % (name, key, want, rep[name].get(key)))
        r = {"n": n, "x": acc["x"], "c_only": chg[arms[1]], "by_s": {}}
        for s in (s1, s2):
            e = "x:E%d.w.p%s" % (el, s)
            row = {"e": chg[e], "ec": {}, "diff": {}}
            for k in LEVELS:
                ec = "x:E%d.w.p%s+C%d.w.m%s" % (el, s, cl, k)
                row["ec"][k] = chg[ec]
                row["diff"][k] = tuple(100 * v for v in boot([u - w for u, w in zip(cor[ec], cor[e])], rng))
            r["by_s"][s] = row
        res[name] = r

    print("MMLU by option probability, E and C written together (1,000-item list; plan queue_o section 5)")
    print("Points = percentage points. 'E + C - E' = same items, E + C minus single E at the same E strength;")
    print("95%% interval from resampling the items (seed %d, %d resamples). C is toward answering (sign m)." % (SEED, NBOOT))
    print("Check passed: single E and single C vs no write equal the cap_mcq_sweep.py report (accuracy and change).")
    for name in ORDER:
        r, (el, cl, s1, s2) = res[name], SPEC[name]
        print("\n== %s (E layer %d, C layer %d; %d items): no write %.3f | single C%d m2 vs no write %+.1f" % (
            name, el, cl, r["n"], r["x"], cl, r["c_only"]))
        print("   %-8s %-9s %-11s %-11s %-11s | %-22s %-22s %-22s" % (
            "E", "single E", "E+C 0.5", "E+C 1", "E+C 2", "E+C 0.5 - E", "E+C 1 - E", "E+C 2 - E"))
        for s, tag in ((s1, "main"), (s2, "larger")):
            row = r["by_s"][s]
            diffs = ["%+.1f [%+.1f, %+.1f]" % row["diff"][k] for k in LEVELS]
            print("   %-8s %-9s %-11s %-11s %-11s | %-22s %-22s %-22s" % (
                "%s %s" % (s, tag[0]), "%+.1f" % row["e"], *("%+.1f" % row["ec"][k] for k in LEVELS), *diffs))
    print("\n(E column: strength, m = main-table strength, l = larger strength)")
    print("\nPre-registered reading (plan section 5), at the larger E strength, change vs no write in points:")
    both = 0
    for name in ORDER:
        s2 = SPEC[name][3]
        row = res[name]["by_s"][s2]
        e_below = row["e"] < -1
        ok = [k for k in LEVELS if row["ec"][k] >= -1]
        both += e_below and bool(ok)
        print("   %-12s strength %-4s single E %+.2f -> below -1 point: %s; C levels with E + C not below -1 point: %s" % (
            name, s2, row["e"], "yes" if e_below else "no", ", ".join(ok) if ok else "none"))
    n_e = sum(res[n]["by_s"][SPEC[n][3]]["e"] < -1 for n in ORDER)
    print("   models with single E below -1 point: %d of 5" % n_e)
    print("   models with single E below -1 point AND some C level with E + C not below -1 point: %d of 5 (plan: at least 3)" % both)


if __name__ == "__main__":
    main()
