#!/usr/bin/env python3
"""Cross-run consistency of the two-axis grading: the same complete input (question, answer, references) must end with
the same verdict in every run. GPT round 22 found 5 of 332 shared inputs with different v5 finals across the two E42
runs; round 23 asked for an explicit check on the new runs rather than an assumption.

usage: judge2_cross_run_check.py RUN_DIR RUN_DIR [RUN_DIR ...] [--out FILE]
Reads each run's judge2/cells.jsonl and judge2/final.jsonl, groups cells by input_sha, and reports shared inputs,
inputs whose final (label, truth, info, info_subtype) differ between runs (with the sources of each verdict), and
inputs unresolved in one run but final in another. Exit code 1 if any conflict exists.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE_ROOT)
from core.io_utils import read_jsonl  # noqa: E402


def load(run_dir):
    j2 = os.path.join(run_dir, "judge2")
    cells = {c["cell_id"]: c for c in read_jsonl(os.path.join(j2, "cells.jsonl"))}
    finals = {r["cell_id"]: r for r in read_jsonl(os.path.join(j2, "final.jsonl"))}
    by_input = defaultdict(list)
    for cid, c in cells.items():
        f = finals.get(cid)
        if f is None or c["gold_is_abstention"]:
            continue
        by_input[c["input_sha"]].append({"run": run_dir, "cell_id": cid, "label": f.get("label"), "truth": f.get("truth"),
                                         "info": f.get("info"), "info_subtype": f.get("info_subtype"), "source": f.get("source"),
                                         "status": f.get("status"), "question": c["question"], "prediction": c["prediction"]})
    return by_input


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    per_run = [load(r) for r in a.run_dirs]
    all_inputs = defaultdict(list)
    for m in per_run:
        for sha, rows in m.items():
            all_inputs[sha].extend(rows)
    shared = {sha: rows for sha, rows in all_inputs.items() if len({r["run"] for r in rows}) > 1}
    conflicts, partial = [], []
    for sha, rows in shared.items():
        finals = [r for r in rows if r["status"] == "final"]
        verdicts = {(r["label"], r["truth"], r["info"], r["info_subtype"]) for r in finals}
        if len(verdicts) > 1:
            conflicts.append({"input_sha": sha, "question": rows[0]["question"], "prediction": rows[0]["prediction"],
                              "verdicts": [{k: r[k] for k in ("run", "cell_id", "label", "truth", "info", "info_subtype", "source")} for r in finals]})
        elif finals and len(finals) < len(rows):
            partial.append({"input_sha": sha, "resolved_in": sorted({r["run"] for r in finals}),
                            "unresolved_in": sorted({r["run"] for r in rows if r["status"] != "final"})})
    report = {"runs": a.run_dirs, "n_inputs_total": len(all_inputs), "n_inputs_shared": len(shared),
              "n_conflicting_finals": len(conflicts), "n_partially_resolved": len(partial),
              "conflicts": conflicts, "partially_resolved": partial}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
    print(json.dumps({k: v for k, v in report.items() if k not in ("conflicts", "partially_resolved")}, ensure_ascii=False))
    for c in conflicts[:20]:
        print("CONFLICT", c["question"][:60], "|", c["prediction"][:60], "|", [(v["run"].rsplit("/", 1)[-1], v["label"], v["source"]) for v in c["verdicts"]])
    sys.exit(1 if conflicts else 0)


if __name__ == "__main__":
    main()
