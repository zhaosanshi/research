#!/usr/bin/env python3
"""Quality-aware summary of an E27 release rerun (Claude, 2026-11-06).

usage: e27_quality_summary.py RERUN_ROOT OUT_JSON
RERUN_ROOT holds <init>_s<seed>/{tune,test}_scored_items.json (review_checks/e27_release_rerun.sh). For every run and split,
on the substantive-reference questions only (refusal-only references excluded, as in the main table):
  correct / wrong / abstain as judged; outputs containing U+FFFD; unfinished outputs; mean answer length in words;
  correct counts under three rules  R0 as judged | R1 a "correct" answer containing U+FFFD counts as wrong |
  R2 R1 plus a "correct" answer that did not finish counts as wrong (the same rule is applied to every condition);
  paired net change versus the unsteered baseline of the same file with the exact two-sided sign (McNemar) test.
Also DAS-init and mean-difference-init versus random init (each seed and their mean) under each rule.
"""
import json
import sys
from math import comb
from pathlib import Path

root, out_path = Path(sys.argv[1]), sys.argv[2]


def sign_p(up, down):
    n = up + down
    if n == 0:
        return None
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(up, down) + 1)) / 2 ** n)


def labels(rows, cond, rule):
    out = {}
    for e in rows:
        o, text, fin = e["outcome"][cond], e["outputs"][cond], e["finished"][cond]
        if rule >= 1 and o == "correct" and "�" in text:
            o = "wrong"
        if rule >= 2 and o == "correct" and not fin:
            o = "wrong"
        out[e["pair_id"]] = o
    return out


def paired(a, b):
    up = sum(a[k] == "correct" and b[k] != "correct" for k in a)
    down = sum(a[k] != "correct" and b[k] == "correct" for k in a)
    return {"net": up - down, "up": up, "down": down, "p": sign_p(up, down)}


result = {"root": str(root), "runs": {}}
runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "test_scored_items.json").exists())
cache = {}
for run in runs:
    rr = {}
    for split in ("tune", "test"):
        d = json.loads((run / ("%s_scored_items.json" % split)).read_text())
        rows = [e for e in (d["evaluations"] if isinstance(d, dict) else d) if not e.get("gold_is_abstention")]
        conds = list(rows[0]["outcome"].keys())
        adapter = [c for c in conds if c.endswith("_generation_adapter")]
        if len(adapter) != 1 or "baseline" not in conds:
            raise SystemExit("%s/%s: unexpected conditions %s" % (run.name, split, conds))
        sp = {"n": len(rows), "conditions": {}}
        for c in conds:
            texts = [e["outputs"][c] for e in rows]
            stats = {"judged": {k: sum(e["outcome"][c] == k for e in rows) for k in ("correct", "wrong", "abstain")},
                     "fffd": sum("�" in t for t in texts), "unfinished": sum(not e["finished"][c] for e in rows),
                     "mean_words": sum(len(t.split()) for t in texts) / len(texts)}
            for rule in (0, 1, 2):
                lab = labels(rows, c, rule)
                stats["R%d_correct" % rule] = sum(v == "correct" for v in lab.values())
                if c != "baseline":
                    stats["R%d_vs_baseline" % rule] = paired(lab, labels(rows, "baseline", rule))
                cache[(run.name, split, c, rule)] = lab
            sp["conditions"][c] = stats
        sp["adapter_condition"] = adapter[0]
        rr[split] = sp
    result["runs"][run.name] = rr

# DAS / mean-difference init versus each random seed on the test split
cmp = {}
rand = [r for r in result["runs"] if r.startswith("random_")]
for name in [r for r in result["runs"] if not r.startswith("random_")]:
    a_cond = result["runs"][name]["test"]["adapter_condition"]
    for rule in (0, 1, 2):
        per = {}
        for rn in rand:
            r_cond = result["runs"][rn]["test"]["adapter_condition"]
            per[rn] = paired(cache[(name, "test", a_cond, rule)], cache[(rn, "test", r_cond, rule)])
        mean_random = sum(result["runs"][rn]["test"]["conditions"][result["runs"][rn]["test"]["adapter_condition"]]["R%d_correct" % rule]
                          for rn in rand) / max(1, len(rand))
        cmp["%s_R%d" % (name, rule)] = {"per_random_seed": per, "adapter_correct": result["runs"][name]["test"]["conditions"][a_cond]["R%d_correct" % rule],
                                        "mean_random_correct": mean_random}
result["vs_random"] = cmp
json.dump(result, open(out_path, "w"), indent=1, ensure_ascii=False)

print("%-12s %-5s %-26s %4s %4s %4s | %4s %4s %5s | R0 %4s (net, p)       R1 %4s (net, p)       R2 %4s (net, p)" % (
    "run", "split", "condition", "cor", "wrg", "abs", "fffd", "unfn", "words", "", "", ""))
for name, rr in result["runs"].items():
    for split in ("tune", "test"):
        for c, s in rr[split]["conditions"].items():
            if c == "original_das" and split == "tune":
                continue
            line = "%-12s %-5s %-26s %4d %4d %4d | %4d %4d %5.1f |" % (name, split, c[:26], s["judged"]["correct"], s["judged"]["wrong"],
                                                                     s["judged"]["abstain"], s["fffd"], s["unfinished"], s["mean_words"])
            for rule in (0, 1, 2):
                v = s.get("R%d_vs_baseline" % rule)
                line += " %4d" % s["R%d_correct" % rule] + ((" (%+d, %.3g)" % (v["net"], v["p"] if v["p"] is not None else float("nan"))) if v else " (base)       ").ljust(17)
            print(line)
for k, v in cmp.items():
    print("%-16s adapter %d vs random mean %.1f | per seed: %s" % (k, v["adapter_correct"], v["mean_random_correct"],
          ", ".join("%s %+d (p %.3g)" % (rn, x["net"], x["p"] if x["p"] is not None else float("nan")) for rn, x in v["per_random_seed"].items())))
print("saved", out_path)
