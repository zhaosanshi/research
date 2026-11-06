#!/usr/bin/env python3
"""Independent recount of an E26 run from its raw per-item records (Claude, 2026-11-06).

usage: e26_independent_recount.py RUN_DIR [RUN_DIR ...]
Does not import core.selective_eval or the E26 script. Re-implements the exact-quota rule (k = floor(q*N); only
non-abstaining items are eligible; order by score descending, ties by sha256 of [protocol, seed, id], then id) and
  1. recounts every arm of every coverage of report.json on the TEST split (commitment, simple and random scores);
  2. re-derives every frozen choice of selection.json on the TUNING split (max delivered correct; ties -> smaller total
     |alpha|, fewer layers, layer tuple, name);
  3. checks that the chosen C / E / E+C conditions actually changed the model: mean gate-margin shift and True/False
     flips versus the unsteered baseline (TF), and per-item edit counts plus changed outputs (QA).
Exit code 1 if any recount or re-derived choice differs.
"""
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

PROTOCOL, SEED = "actual_coverage_v1", 42


def h(item):
    return hashlib.sha256(json.dumps([PROTOCOL, SEED, item], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def rnd(item):
    return (int(h(item)[:14], 16) >> 3) / float(2 ** 53)


def quota_correct(rows, q, key):
    rows = sorted(rows, key=lambda r: r["id"])
    f = Fraction(str(q))
    k = f.numerator * len(rows) // f.denominator
    eligible = [r for r in rows if r["outcome"] != "abstain"]
    if len(eligible) < k:
        return None
    order = sorted(eligible, key=lambda r: (-key(r), h(r["id"]), r["id"]))
    return sum(r["outcome"] == "correct" for r in order[:k])


def load(run, task, split):
    if task == "tf":
        d = json.loads((run / ("tf_%s.json" % split)).read_text())
        return d["rows"], d["condition_specs"]
    d = json.loads((run / ("qa_%s_scored.json" % split)).read_text())
    return {c: [r for r in rs if not r.get("gold_is_abstention")] for c, rs in d["rows"].items()}, d["condition_specs"]


SCORE = lambda r: r["score"]
SIMPLE = lambda r: r["simple_score"]
bad = 0
for run in map(Path, sys.argv[1:]):
    rep = json.loads((run / "report.json").read_text())
    sel = json.loads((run / "selection.json").read_text())
    model = rep["model"]
    for task in ("tf", "qa"):
        test, _ = load(run, task, "test")
        tune, specs = load(run, task, "tune")
        specs = {c["name"]: c for c in specs}
        n_cells = n_choice = 0
        for cov, picked in sel["tasks"][task]["choices"].items():
            arms = {"baseline_commitment": (test["baseline"], SCORE), "baseline_simple": (test["baseline"], SIMPLE),
                    "baseline_random": (test["baseline"], lambda r: rnd(r["id"]))}
            for fam, name in picked.items():
                if name is not None:
                    arms[fam] = (test[name], SIMPLE if fam.endswith("_simple") else SCORE)
            reported = rep["tables"][task][cov]["quota"]["coverages"][0]["arms"]
            for arm, (rows, key) in arms.items():
                n_cells += 1
                mine, theirs = quota_correct(rows, cov, key), reported.get(arm, {}).get("correct")
                if mine != theirs:
                    bad += 1
                    print("RECOUNT DIFFERS", model, task, cov, arm, mine, theirs)
            for fam, chosen in picked.items():
                n_choice += 1
                simple = fam.endswith("_simple")
                method, kind = (fam[:-7] if simple else fam).rsplit("_", 1)
                cands = []
                for name, s in specs.items():
                    if s["method"] != method or s["kind"] != kind:
                        continue
                    c = quota_correct(tune[name], cov, SIMPLE if simple else SCORE)
                    if c is None:
                        continue
                    e, cc = s.get("e") or {}, s.get("c") or {}
                    layers = e.get("layers", [e["layer"]] if "layer" in e else []) + ([cc["layer"]] if cc else [])
                    cands.append(((-c, abs(e.get("alpha", 0)) + abs(cc.get("alpha", 0)), len(layers), tuple(layers), name), name))
                mine = min(cands)[1] if cands else None
                if mine != chosen:
                    bad += 1
                    print("CHOICE DIFFERS", model, task, cov, fam, mine, chosen)
        print("%s %s: %d test arm-cells recounted, %d frozen choices re-derived" % (model, task, n_cells, n_choice))
        pick = sel["tasks"][task]["choices"]["0.5"]
        base = {r["id"]: r for r in test["baseline"]}
        for fam in ("das_C", "das_E", "das_EC"):
            name = pick.get(fam)
            if name is None:
                continue
            rows = test[name]
            if task == "tf":
                shift = sum(r["score"] - base[r["id"]]["score"] for r in rows) / len(rows)
                flips = sum(r["predicted_true"] != base[r["id"]]["predicted_true"] for r in rows)
                print("   TF %-6s %-20s gate-margin shift %+.3f | True/False flips %d / %d" % (fam, name, shift, flips, len(rows)))
            else:
                changed = sum(r["output"] != base[r["id"]]["output"] for r in rows)
                print("   QA %-6s %-20s edits %s | outputs changed %d / %d" % (fam, name, rows[0].get("edits"), changed, len(rows)))
print("ALL MATCH" if bad == 0 else "%d DIFFERENCES" % bad)
sys.exit(1 if bad else 0)
