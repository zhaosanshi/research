#!/usr/bin/env python3
"""Report readout layer and measured single-layer write choice separately.

Read layer: maximal E1 probe separability (all ties retained).
Write candidate: maximal tuning-set dJ with |d self-reported answer rate| <= .03.
This retains E23's existing threshold. It does not certify the 2x2 hypothesis.
No test score, DAS training loss or LA-Score participates in either choice.
Uses Python's standard library; no model inference or parameter fitting.
"""
import argparse
import json
import math
from pathlib import Path
import re


SINGLE_E = re.compile(r"^E_L(\d+)_a([0-9.eE+\-]+)$")
METRICS = ("J", "AUROC", "B", "ans")
E_LEAK = 0.03


def summarize(summary, profile, sweep):
    """Inputs are E1 summary/profile and one full-layer E23 sweep JSON."""
    n_layers = int(sweep["num_layers"])
    expected = set(range(n_layers))
    if n_layers < 1 or set(sweep["layers"]) != expected:
        raise ValueError("E23 must cover every decoder layer, including first and last")
    if summary["model"] != sweep["model"] or summary["num_layers"] != n_layers:
        raise ValueError("E1 and E23 refer to different models or layer counts")
    if summary.get("direction_prompt_allow_idk") is not False or sweep.get("direction_prompt_allow_idk") is not False:
        raise ValueError("this report requires matching forced True/False prompts")
    if sweep.get("teacher_split") or sweep.get("tune_pairs", 0) <= 0:
        raise ValueError("use a declared tuning set (--tune-pairs), not mixed selection protocols")
    if sum(summary["n_E"]) != sweep["n_est"]:
        raise ValueError("E1/E23 estimation-set sizes differ")
    if sweep["e_estimator"] == "das":
        training, evaluation = summary["das"], sweep["das"]
        if (training["variant"] != evaluation["variant"]
                or set(training["E_trained_layers"]) != expected
                or training["n_pairs"]["E"] * 2 != sweep["n_est"]):
            raise ValueError("DAS training variant, layer coverage or pair count differs")
        for layer in expected:
            if sorted(evaluation["per_layer"][str(layer)]["seeds"]) != sorted(training["seeds"]):
                raise ValueError("DAS seeds differ across training/evaluation at layer %d" % layer)

    read_rows = {int(k): v for k, v in profile["E"]["rows"].items()}
    if set(read_rows) != expected:
        raise ValueError("E1 readout profile does not cover every decoder layer")
    finite_read = {l: float(r["sep"]) for l, r in read_rows.items() if math.isfinite(float(r["sep"]))}
    if not finite_read:
        raise ValueError("no finite readout score")
    top_read = max(finite_read.values())
    read_layers = sorted(l for l, value in finite_read.items() if value == top_read)

    conds = sweep["conds"]
    candidates = {l: [] for l in expected}
    expected_alphas = set(map(float, sweep["e_alphas"]))
    if not expected_alphas:
        raise ValueError("no E intervention strengths declared")

    def delta(name, split):
        return {k: float(conds[name][split][k]) - float(conds["baseline"][split][k]) for k in METRICS}

    for name in conds:
        match = SINGLE_E.fullmatch(name)
        if match:
            layer, alpha = int(match[1]), float(match[2])
            if layer not in expected:
                raise ValueError("condition references an out-of-range layer")
            candidates[layer].append((alpha, name))
    per_layer = []
    eligible = []
    for layer in sorted(expected):
        if {a for a, _ in candidates[layer]} != expected_alphas:
            raise ValueError("incomplete E strength grid at layer %d" % layer)
        ok = []
        for alpha, name in sorted(candidates[layer]):
            change = delta(name, "A")  # A is the fixed tuning set, B the test set.
            if not all(math.isfinite(v) for v in change.values()):
                raise ValueError("non-finite tuning metric in %s" % name)
            if abs(change["ans"]) <= E_LEAK:
                ok.append((change["J"], -alpha, name))
        chosen = max(ok)[2] if ok else None
        per_layer.append({"layer": layer, "read_sep": finite_read.get(layer), "condition": chosen,
                          "tune": delta(chosen, "A") if chosen else None,
                          "test": delta(chosen, "B") if chosen else None})
        if chosen:
            eligible.append(per_layer[-1])
    # Stable tie break: smaller layer number; alpha ties were handled above.
    best = max(eligible, key=lambda row: (row["tune"]["J"], -row["layer"])) if eligible else None
    return {"model": sweep["model"], "num_layers": n_layers, "read_layers": read_layers,
            "read_sep": top_read, "read_basis": "E1 estimation-set probe separability",
            "write_candidate": best, "write_basis": "tuning dJ, subject to |d self-report| <= 0.03",
            "read_write_distances": {str(l): abs(l - best["layer"]) for l in read_layers} if best else None,
            "positive_tuning_gain": bool(best and best["tune"]["J"] > 0),
            "per_layer": per_layer,
            "qualification": "Candidate selection only. Bias and test deltas are reported; no confidence intervals or 2x2 success claim."}


def markdown(result):
    best = result["write_candidate"]
    lines = ["# 读层与实测写层：%s" % result["model"], "",
             "覆盖 %d 层；每层各自训练方向并在该层干预。" % result["num_layers"], "",
             "- 读层：%s；依据估计集探针的真假区分分数，并列层全部保留。" % result["read_layers"],
             "- 写层候选：%s；只按调参集的实际干预效果选择。" % (best["condition"] if best else "无满足现有约束的候选"),
             "- 读写层距离：%s。" % result["read_write_distances"],
             "- 候选在调参集上是否有正增益：%s。" % result["positive_tuning_gain"], "",
             "选层沿用现有口径：自报作答比例变化不超过 0.03，再取判别指标 J 提升最大的配置。训练损失、读层分数和测试集效果均不参与写层选择。",
             "偏置、排序判别指标及测试集结果另列；此表没有置信区间，选出候选不代表 2×2 已成立。多层联合配置不参与本单层报告。", "",
             "| 层 | 读层分数 | 调参选出的强度配置 | 调参 ΔJ | 调参 Δ自报 | 调参 Δ偏置 | 测试 ΔJ | 测试 Δ排序判别 | 测试 Δ自报 | 测试 Δ偏置 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in result["per_layer"]:
        if row["condition"]:
            a, b = row["tune"], row["test"]
            lines.append("| %d | %s | %s | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f |" % (
                row["layer"], row["read_sep"], row["condition"], a["J"], a["ans"], a["B"], b["J"], b["AUROC"], b["ans"], b["B"]))
        else:
            lines.append("| %d | %s | 无合格候选 | | | | | | | |" % (row["layer"], row["read_sep"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-dir", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True, help="one E23 JSON covering all single layers")
    parser.add_argument("--out", type=Path, required=True, help="new output stem for .json and .md")
    args = parser.parse_args()
    for suffix in (".json", ".md"):
        if Path(str(args.out) + suffix).exists():
            parser.error("output exists; choose a new --out stem")
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    result = summarize(read(args.e1_dir / "summary.json"), read(args.e1_dir / "layer_scores.json"), read(args.sweep))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.out) + ".md").write_text(markdown(result), encoding="utf-8")
    print("读层=%s；写层候选=%s" % (result["read_layers"], result["write_candidate"]["condition"] if result["write_candidate"] else "无"))


if __name__ == "__main__":
    main()
