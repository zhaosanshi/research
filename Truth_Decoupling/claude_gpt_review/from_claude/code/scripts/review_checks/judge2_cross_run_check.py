#!/usr/bin/env python3
"""Cross-run consistency of the two-axis grading: the same complete input (question, answer, references) must end with
the same verdict in every run. GPT round 22 found 5 of 332 shared inputs with different v5 finals across the two E42
runs; round 23 asked for an explicit check on the new runs rather than an assumption.

usage: judge2_cross_run_check.py RUN_DIR RUN_DIR [RUN_DIR ...] [--out FILE]
Verifies each run's complete provenance before comparing its full semantic inputs.
Exit 0 means complete, verified and consistent shared decisions; 1 is a conflict;
2 means invalid/incomplete/no shared inputs. --allow-unresolved reports a partial
snapshot but never makes that snapshot pass.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

CODE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, CODE_ROOT)
from scripts.review_checks import judge_two_axis_run as R  # noqa: E402


def load(run_dir, require_complete=True):
    j2 = os.path.join(run_dir, "judge2")
    meta = R._read_json(os.path.join(j2, 'meta.json'))
    _, cells, finals = R.load_verified_final(run_dir, os.path.join(run_dir, meta['generation']),
                                            require_complete=require_complete)
    by_input = defaultdict(list)
    for cid, c in cells.items():
        f = finals[cid]     # The verified loader rejects missing or duplicate rows.
        if c["gold_is_abstention"]:
            continue
        by_input[c["input_sha"]].append({"run": run_dir, "cell_id": cid, "label": f.get("label"), "truth": f.get("truth"),
                                         "info": f.get("info"), "info_subtype": f.get("info_subtype"), "source": f.get("source"),
                                         "status": f.get("status"), "question": c["question"], "prediction": c["prediction"]})
    return by_input


def check(run_dirs, allow_unresolved=False):
    run_dirs = [os.path.realpath(r) for r in run_dirs]
    if len(run_dirs) < 2 or len(set(run_dirs)) != len(run_dirs):
        raise ValueError('at least two distinct runs are required')
    per_run = [load(r, require_complete=not allow_unresolved) for r in run_dirs]
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
    unresolved = [r for rows in all_inputs.values() for r in rows if r['status'] != 'final']
    complete = not unresolved
    verified_consistent = complete and bool(shared) and not conflicts
    report = {"runs": run_dirs, "n_inputs_total": len(all_inputs), "n_inputs_shared": len(shared),
              "n_conflicting_finals": len(conflicts), "n_partially_resolved": len(partial),
              "n_unresolved_cells": len(unresolved), "semantic_decisions_complete": complete,
              "verified_consistent": verified_consistent,
              "conflicts": conflicts, "partially_resolved": partial}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+"); ap.add_argument("--out", default=None)
    ap.add_argument('--allow-unresolved', action='store_true')
    a = ap.parse_args()
    try:
        report = check(a.run_dirs, allow_unresolved=a.allow_unresolved)
        status = 1 if report['n_conflicting_finals'] else (0 if report['verified_consistent'] else 2)
    except (ValueError, OSError, KeyError) as exc:
        report = {'runs': a.run_dirs, 'verified_consistent': False, 'error': str(exc)}
        status = 2
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
    print(json.dumps({k: v for k, v in report.items() if k not in ("conflicts", "partially_resolved")}, ensure_ascii=False))
    for c in report.get('conflicts', [])[:20]:
        print("CONFLICT", c["question"][:60], "|", c["prediction"][:60], "|", [(v["run"].rsplit("/", 1)[-1], v["label"], v["source"]) for v in c["verdicts"]])
    sys.exit(status)


if __name__ == "__main__":
    main()
