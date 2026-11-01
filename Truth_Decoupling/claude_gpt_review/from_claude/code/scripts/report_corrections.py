#!/usr/bin/env python3
"""Count corrections and harms from existing E23 per-item outputs; no inference.

Uses the E condition recorded as selected on tuning data. Counts statements on
the held-out B mask, not original questions. Harm rate is conditioned on items
the baseline classified correctly, matching the plan's baseline-known split.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def transitions(labels, before_p, after_p):
    labels = np.asarray(labels, dtype=bool)
    before_p, after_p = np.asarray(before_p, dtype=float), np.asarray(after_p, dtype=float)
    if labels.ndim != 1 or labels.shape != before_p.shape or labels.shape != after_p.shape or not labels.size:
        raise ValueError("need nonempty aligned one-dimensional per-item arrays")
    if not (np.isfinite(before_p).all() and np.isfinite(after_p).all()):
        raise ValueError("non-finite prediction probabilities")
    before, after = (before_p > .5) == labels, (after_p > .5) == labels
    cc, cw = int((before & after).sum()), int((before & ~after).sum())
    wc, ww = int((~before & after).sum()), int((~before & ~after).sum())
    n = len(labels)
    return {"n_statements": n, "baseline_correct": cc + cw, "intervened_correct": cc + wc,
            "correct_to_correct": cc, "correct_to_wrong": cw, "wrong_to_correct": wc, "wrong_to_wrong": ww,
            "net_correct_gain": wc - cw, "net_accuracy_gain_pp": 100 * (wc - cw) / n,
            "harm_percent_of_baseline_correct": 100 * cw / (cc + cw) if cc + cw else None,
            "correction_percent_of_baseline_wrong": 100 * wc / (wc + ww) if wc + ww else None}


def read_result(root, model, tag):
    directory = root / "E23" / model
    metadata = json.loads((directory / ("knob_sweep_" + tag + ".json")).read_text())
    chosen = metadata["selection"]["E"]["chosen"]
    if not chosen or chosen == "baseline":
        raise ValueError("no selected E intervention for %s" % model)
    with np.load(directory / ("knob_sweep_" + tag + ".npz"), allow_pickle=False) as data:
        mask = ~data["half_a"].astype(bool)
        result = transitions(data["labels"][mask], data["baseline__ptrue"][mask], data[chosen + "__ptrue"][mask])
    # Confirm that these per-item predictions reproduce the previously reported table.
    for condition, correct in (("baseline", result["baseline_correct"]), (chosen, result["intervened_correct"])):
        reported_j = metadata["conds"][condition]["B"]["J"]
        implied_accuracy = (1 + reported_j) / 2
        if abs(correct / result["n_statements"] - implied_accuracy) > 1e-8:
            raise ValueError("per-item accuracy differs from the balanced-set J report: %s" % condition)
    return {"model": model, "condition": chosen, "split": "B (test)", **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="existing results root, containing E23")
    parser.add_argument("--models", nargs="+", default=["llama3.1-8b", "qwen3-8b"])
    parser.add_argument("--tag", default="das_tf_spans")
    parser.add_argument("--out", type=Path, required=True, help="new output stem (.json and .md)")
    args = parser.parse_args()
    outputs = [Path(str(args.out) + suffix) for suffix in (".json", ".md")]
    if any(p.exists() for p in outputs):
        parser.error("output exists; choose a new --out stem")
    rows = [read_result(args.root, model, args.tag) for model in args.models]
    lines = ["# 证据干预：纠正与损伤", "", "单位为真假陈述；条件沿用调参集记录的选择，统计测试集。损伤率的分母是基线答对的陈述数。", "",
             "| 模型 | 条件 | 原对→仍对 | 原对→改错 | 原错→改对 | 原错→仍错 | 净多判对 | 原对题损伤率 | 净准确率提升 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        harm = r["harm_percent_of_baseline_correct"]
        lines.append("| %s | %s | %d | %d | %d | %d | %+d | %s | %+.3f 个百分点 |" % (
            r["model"], r["condition"], r["correct_to_correct"], r["correct_to_wrong"], r["wrong_to_correct"],
            r["wrong_to_wrong"], r["net_correct_gain"], ("%.3f%%" % harm) if harm is not None else "无基线正确项", r["net_accuracy_gain_pp"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    outputs[1].write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
