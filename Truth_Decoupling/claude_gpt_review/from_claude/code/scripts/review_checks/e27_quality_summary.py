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
import argparse
import hashlib
import json
import sys
from math import comb
from pathlib import Path


def sign_p(up, down):
    n = up + down
    if n == 0:
        return 1.0
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
    if set(a) != set(b):
        raise ValueError("paired comparison requires identical question IDs")
    up = sum(a[k] == "correct" and b[k] != "correct" for k in a)
    down = sum(a[k] != "correct" and b[k] == "correct" for k in a)
    return {"net": up - down, "up": up, "down": down, "p": sign_p(up, down)}



def validate_scored(d, run, split):
    if (not isinstance(d, dict) or d.get("completed") is not True
            or d.get("grading_completed") is not True or d.get("split") != split
            or not isinstance(d.get("model"), str)):
        raise ValueError("quality summary requires completed identified semantic grades")
    rows, conds = d.get("evaluations", []), d.get("conditions", [])
    if not rows or not conds or len(conds) != len(set(conds)) or "baseline" not in conds:
        raise ValueError("empty or invalid quality cohort/conditions")
    ids = [r.get("pair_id") for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("question IDs must be unique nonempty strings")
    for r in rows:
        if type(r.get("gold_is_abstention")) is not bool:
            raise ValueError("reference partition must be explicitly recorded for every question")
        for key in ("outputs", "finished", "outcome", "judge"):
            if set(r.get(key, {})) != set(conds):
                raise ValueError("missing or extra condition in " + key)
        for c in conds:
            if (not isinstance(r["outputs"][c], str) or type(r["finished"][c]) is not bool
                    or r["outcome"][c] not in ("correct", "wrong", "abstain")
                    or r["judge"][c].get("label") != r["outcome"][c]
                    or r["judge"][c].get("rubric_id") != d.get("judge_rubric_id")):
                raise ValueError("unresolved or inconsistent output/completion/semantic grade")
    raw_path = Path(run) / (split + ".json")
    if not raw_path.exists():
        raise ValueError("raw generation is required to verify quality flags and response text")
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != d.get("grading_source_generation_sha256"):
        raise ValueError("scored items are stale relative to raw generation")
    raw = json.loads(raw_bytes)
    raw_rows = raw.get("evaluations", [])
    if (raw.get("completed") is not True or raw.get("model") != d["model"]
            or raw.get("conditions") != conds or raw.get("split") != split
            or [r.get("pair_id") for r in raw_rows] != ids):
        raise ValueError("raw and scored cohorts/conditions do not match")
    for a, b in zip(raw_rows, rows):
        for key in ("question", "gold", "outputs", "finished"):
            if a.get(key) != b.get(key):
                raise ValueError("graded " + key + " differs from raw generation")
    filtered = [r for r in rows if not r["gold_is_abstention"]]
    if not filtered:
        raise ValueError("no substantive-reference questions")
    return filtered, conds


def summarize(root):
    root = Path(root)
    result = {"root": str(root), "runs": {},
              "method_family": "supervised_generation_adapter",
              "random_label": "random initialization followed by correct-answer training; not an untrained random control",
              "quality_rules_are_sensitivity_analyses_unless_predeclared": True,
              "source_hashes": {}}
    runs = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "test_scored_items.json").exists())
    if not runs:
        raise ValueError("no complete E27 test runs found")
    cache, shared = {}, {}
    for run in runs:
        rr = {}
        for split in ("tune", "test"):
            source = run / ("%s_scored_items.json" % split)
            d = json.loads(source.read_text())
            rows, conds = validate_scored(d, run, split)
            result["source_hashes"][str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
            shared_key = (d["model"], split)
            baseline = {r["pair_id"]: {k: r[k]["baseline"] for k in ("outputs", "finished", "outcome")} for r in rows}
            if shared_key in shared and shared[shared_key] != baseline:
                raise ValueError("baseline outputs/completion/grades differ across runs; comparisons are not aligned")
            shared[shared_key] = baseline
            adapter = [c for c in conds if c.endswith("_generation_adapter")]
            if len(adapter) != 1 or "baseline" not in conds:
                raise SystemExit("%s/%s: unexpected conditions %s" % (run.name, split, conds))
            declared_rule = d.get('primary_quality_rule')
            if declared_rule is not None:
                selection_path = run / 'selection.json'
                if declared_rule not in ('R0', 'R1', 'R2') or not selection_path.exists():
                    raise ValueError('declared primary quality rule has no matching freeze artifact')
                selection = json.loads(selection_path.read_text())
                if (selection.get('primary_quality_rule') != declared_rule
                        or selection.get('fingerprint') != d.get('pretest_freeze_fingerprint')):
                    raise ValueError('primary quality rule differs from the frozen test declaration')
            sp = {"n": len(rows), "raw_n": len(d['evaluations']),
                  "excluded_refusal_only_ids": [r['pair_id'] for r in d['evaluations'] if r['gold_is_abstention']],
                  "primary_quality_rule": declared_rule,
                  "quality_plan_status": 'pretest_artifact_recorded' if declared_rule else 'legacy_unrecorded_sensitivity_only',
                  "conditions": {}}
            for c in conds:
                texts = [e["outputs"][c] for e in rows]
                stats = {"judged": {k: sum(e["outcome"][c] == k for e in rows) for k in ("correct", "wrong", "abstain")},
                         "fffd": sum("�" in t for t in texts), "unfinished": sum(not e["finished"][c] for e in rows),
                         "fffd_and_unfinished": sum("�" in e["outputs"][c] and not e["finished"][c] for e in rows),
                         "correct_fffd": sum(e["outcome"][c] == "correct" and "�" in e["outputs"][c] for e in rows),
                         "correct_unfinished": sum(e["outcome"][c] == "correct" and not e["finished"][c] for e in rows),
                         "correct_bad_union": sum(e["outcome"][c] == "correct" and ("�" in e["outputs"][c] or not e["finished"][c]) for e in rows),
                         "mean_words": sum(len(t.split()) for t in texts) / len(texts)}
                for rule in (0, 1, 2):
                    lab = labels(rows, c, rule)
                    stats["R%d_correct" % rule] = sum(v == "correct" for v in lab.values())
                    if c != "baseline":
                        stats["R%d_vs_baseline" % rule] = paired(lab, labels(rows, "baseline", rule))
                    cache[(run.name, split, c, rule)] = lab
                stats['primary_correct'] = stats[declared_rule + '_correct'] if declared_rule else None
                if c != 'baseline':
                    stats['primary_vs_baseline'] = stats[declared_rule + '_vs_baseline'] if declared_rule else None
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
                              for rn in rand) / len(rand) if rand else None
            cmp["%s_R%d" % (name, rule)] = {"per_random_seed": per, "adapter_correct": result["runs"][name]["test"]["conditions"][a_cond]["R%d_correct" % rule],
                                            "mean_random_correct": mean_random}
    result["vs_random"] = cmp
    if len({k[0] for k in shared}) != 1:
        raise ValueError("each quality summary must contain exactly one model")
    result["random_comparison_status"] = "available" if rand else "unavailable_no_random_initialized_trained_run"
    return result


def print_summary(result):
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
    for k, v in result["vs_random"].items():
        if v["mean_random_correct"] is None:
            continue
        print("%-16s adapter %d vs trained-random mean %.1f | per seed: %s" % (k, v["adapter_correct"], v["mean_random_correct"],
              ", ".join("%s %+d (p %.3g)" % (rn, x["net"], x["p"] if x["p"] is not None else float("nan")) for rn, x in v["per_random_seed"].items())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out_json", type=Path)
    args = p.parse_args()
    result = summarize(args.root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=1, ensure_ascii=False, allow_nan=False))
    print_summary(result)
    print("saved", args.out_json)


if __name__ == "__main__":
    main()
