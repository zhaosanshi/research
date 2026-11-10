#!/usr/bin/env python3
"""Interim judge-free tables straight from gen/*.json + train_logs/*.json (no summary.json needed, so it
can be read while the run is still going). One table per family x split; one row per layer.
cell = changed_normalized / refusal before->after / matches side prompt (baseline matches) / prefill rel
usage: e43_interim.py RUN_DIR [--split tune] [--family E|C] [--dump FAMILY LAYER ARM [--max N]]
"""
import argparse, glob, json, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--split", default="tune")
ap.add_argument("--family", default=None)
ap.add_argument("--dump", nargs=3, metavar=("FAMILY", "LAYER", "ARM"))
ap.add_argument("--max", type=int, default=20)
ap.add_argument("--only-content", action="store_true", help="dump only rows whose normalized text changed")
a = ap.parse_args()

ARMS = {"E": ["main", "recover", "md_main", "random_main"],
        "C": ["main", "to_conf", "conf_to_caut", "caut_to_conf", "md_main", "random_main"]}

train = {}
for f in glob.glob(str(a.run_dir / "train_logs" / "*.json")):
    d = json.load(open(f))
    train[(d["family"], d["layer"])] = d

gens = {}
for f in glob.glob(str(a.run_dir / "gen" / "*.json")):
    name = os.path.basename(f)[:-5]
    fam, rest = name.split("_", 1)
    layer_s, rest = rest.split("_", 1)
    arm, split = rest.rsplit("_", 1)
    if split != a.split:
        continue
    d = json.load(open(f))
    if d.get("pilot"):
        continue
    gens[(fam, int(layer_s[1:]), arm)] = d["summary"]

for fam in ("E", "C"):
    if a.family and fam != a.family:
        continue
    layers = sorted({k[1] for k in gens if k[0] == fam})
    if not layers:
        continue
    arms = [x for x in ARMS[fam] if any(k[0] == fam and k[2] == x for k in gens)]
    n = next(iter(v["n"] for k, v in gens.items() if k[0] == fam))
    print("\n## %s on %s (n=%d). cell = changed_norm / refusal before->after / =side (base =side) / prefill_rel" % (fam, a.split, n))
    print("| L | CE+ / CE- | gap (md gap) | cos(w,md) | " + " | ".join(arms) + " |")
    print("|---|---|---|---|" + "---|" * len(arms))
    for L in layers:
        t = train.get((fam, L))
        if t:
            ce = t["ce_after_mean"]
            ep = t["endpoints"]; md = t.get("endpoints_md") or {}
            gap = ep["plus"]["mean"] - ep["minus"]["mean"]
            mdgap = (md["plus"]["mean"] - md["minus"]["mean"]) if md else float("nan")
            head = "%.2f / %.2f | %+.2f (%+.2f) | %+.2f" % (ce["plus"], ce["minus"], gap, mdgap, t.get("cosine_w_md", float("nan")))
        else:
            head = "— | — | —"
        cells = []
        for arm in arms:
            s = gens.get((fam, L, arm))
            if s is None:
                cells.append("—"); continue
            w = "" if s.get("prefill_rel_mean") is None else " %.2f" % s["prefill_rel_mean"]
            cells.append("%d / %d->%d / %d (%d)%s" % (s["changed_normalized"], s["refusal_before"], s["refusal_after"],
                                                     s["matches_side_prompt"], s["baseline_matches_side_prompt"], w))
        print("| %d | %s | %s |" % (L, head, " | ".join(cells)))

if a.dump:
    fam, L, arm = a.dump
    path = a.run_dir / "gen" / ("%s_L%s_%s_%s.json" % (fam, L, arm, a.split))
    d = json.load(open(path))
    s = d["summary"]
    print("\n## %s: changed %d (normalized %d) / %d; refusal %d->%d; =side %d (base %d)" % (
        path.name, s["changed"], s["changed_normalized"], s["n"], s["refusal_before"], s["refusal_after"],
        s["matches_side_prompt"], s["baseline_matches_side_prompt"]))
    shown = 0
    for r in d["rows"]:
        if not r["changed"] or (a.only_content and not r["changed_normalized"]):
            continue
        flag = ""
        if r["baseline_refusal_rule"] != r["refusal_rule"]:
            flag = " [refusal %s->%s]" % (r["baseline_refusal_rule"], r["refusal_rule"])
        print("- %s%s | %s\n    base: %s\n    new : %s\n    side: %s" % (
            r["pair_id"], flag, r["question"][:90], (r["baseline_text"] or "")[:140].replace("\n", " "),
            (r["text"] or "")[:140].replace("\n", " "), (r["side_prompt_text"] or "")[:120].replace("\n", " ")))
        shown += 1
        if shown >= a.max:
            break
