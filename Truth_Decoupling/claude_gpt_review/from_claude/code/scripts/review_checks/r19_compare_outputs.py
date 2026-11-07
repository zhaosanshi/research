#!/usr/bin/env python3
"""Compare Claude's reproduction of GPT round-19 GPU diagnostics with GPT's outputs (Claude, 2026-11-07).

usage: r19_compare_outputs.py MINE_ROOT GPT_ROOT OUT_JSON
MINE_ROOT = results_r19_claude_20261107, GPT_ROOT = results_r19_codex_20261106 (server). For the three trace / dose JSONs it
records whether the files are byte-identical; for the generation runs it counts, per arm, items whose output text or finished
flag differ, and lists any other top-level key that differs. Exit code 1 if anything differs.
"""
import hashlib
import json
import sys
from pathlib import Path

mine, theirs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report, bad = {"byte_identical": {}, "generation_runs": {}}, 0
for name in ("qwen_trace.json", "qwen_current_dose.json", "llama_trace.json"):
    a, b = mine / name, theirs / name
    same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
    report["byte_identical"][name] = {"identical": same, "sha256": sha(a) if a.exists() else None}
    bad += not same
for run in ("random_controls/qwen3-8b/pilot3", "random_controls/qwen3-8b/full192", "random_controls/llama3.1-8b/full192",
            "generation_dose/qwen3-8b/full192", "llama_pure_das_tune192"):
    a = json.loads((mine / run / "tune.json").read_text())
    b = json.loads((theirs / run / "tune.json").read_text())
    ea = {e["pair_id"]: e for e in a["evaluations"]}
    eb = {e["pair_id"]: e for e in b["evaluations"]}
    entry = {"n": len(ea), "same_items": set(ea) == set(eb), "conditions_equal": a["conditions"] == b["conditions"], "arms": {},
             "other_top_level_keys_differing": [k for k in sorted(set(a) | set(b)) if k != "evaluations" and a.get(k) != b.get(k)]}
    bad += (not entry["same_items"]) + (not entry["conditions_equal"]) + len(entry["other_top_level_keys_differing"])
    for c in a["conditions"]:
        d_out = sum(ea[p]["outputs"][c] != eb[p]["outputs"][c] for p in ea if p in eb)
        d_fin = sum(ea[p]["finished"][c] != eb[p]["finished"][c] for p in ea if p in eb)
        entry["arms"][c] = {"outputs_differ": d_out, "finished_differ": d_fin}
        bad += d_out + d_fin
    report["generation_runs"][run] = entry
report["all_identical"] = bad == 0
out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
print("ALL IDENTICAL" if bad == 0 else "%d DIFFERENCES" % bad)
sys.exit(1 if bad else 0)
