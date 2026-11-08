#!/usr/bin/env python3
"""N7 step 1: the C mid-strength curve, one table per model from the per-strength E41 analyses.

usage: n7_curve_table.py E41_DIR [--models llama3.1-8b qwen3-8b gemma3-4b] [--out PREFIX]
E41_DIR holds <model>/e41_n7_a<alpha>_analysis.json (written by review_checks/n7_run.sh).
Per strength and per C sign the table reports, on the test half: C alone's self-selected answer count K (how far the
knob moved), E+C's correct answers among its own self-selected K, and the pre-declared contrasts of e41_analysis.py:
  P1 = E+C natural minus E thresholded on its gate margin to the same K
  P2 = E+C natural minus E thresholded on its True/False confidence to the same K   (primary)
with the pair-bootstrap 95% CI and the Holm-corrected bootstrap p (Holm within model and strength, 4 contrasts).
Criterion (pre-declared in the notepad): the Dr.'s sentence "at the same answer rate the combination beats any single
direction" holds only if P2 > 0 with Holm p < 0.05 at >= 1 strength on >= 2 models. Everything else is the curve.
"""
import argparse
import glob
import json
import os
import re

ALPHA = re.compile(r"e41_n7_a([0-9.]+)_analysis\.json$")


def load_model(e41_dir, model):
    rows = []
    for path in sorted(glob.glob(os.path.join(e41_dir, model, "e41_n7_a*_analysis.json"))):
        m = ALPHA.search(path)
        if not m:
            continue
        A = json.load(open(path, encoding="utf-8"))
        c_names = list(A["fixed_C"].keys())
        c_abst = [c for c in c_names if A["fixed_C"][c]["sign"] == -1][0]
        c_ans = [c for c in c_names if A["fixed_C"][c]["sign"] == 1][0]
        E = A["chosen_E"]
        summ = A["secondary"]["summaries"]
        row = {"alpha": float(m.group(1)), "E": E, "n_test": A["n_test"],
               "baseline": {"K": summ["baseline"]["K"], "correct": summ["baseline"]["correct_natural"], "J": summ["baseline"]["J"]},
               "E_only": {"K": summ[E]["K"], "correct": summ[E]["correct_natural"], "J": summ[E]["J"]}}
        for tag, c in (("abst", c_abst), ("ans", c_ans)):
            ec = "%s+%s" % (E, c)
            p1 = A["primary"]["P1 %s vs E@gate" % ec]
            p2 = A["primary"]["P2 %s vs E@conf" % ec]
            row[tag] = {"C": c, "C_K": summ[c]["K"], "C_correct": summ[c]["correct_natural"], "EC_K": summ[ec]["K"],
                        "EC_correct": summ[ec]["correct_natural"], "EC_J": summ[ec]["J"],
                        "P1": p1["diff"], "P1_ci": p1["ci95"], "P1_holm": p1["holm_p"],
                        "P2": p2["diff"], "P2_ci": p2["ci95"], "P2_holm": p2["holm_p"]}
        rows.append(row)
    return sorted(rows, key=lambda r: r["alpha"])


def table(model, rows):
    L = ["## %s（E 固定在 2×2 选中的 %s；测试半 %d 条陈述）" % (model, rows[0]["E"] if rows else "?", rows[0]["n_test"] if rows else 0), "",
         "| C 力度 α | 方向 | C 单独自答数 K | E+C 自答数 K | E+C 自答中答对 | P1 对 E@门控同 K [95% CI] Holm p | **P2 对 E@置信同 K [95% CI] Holm p** |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        for tag, name in (("abst", "推弃答"), ("ans", "推作答")):
            d = r[tag]
            L.append("| %g | %s | %d | %d | %d | %+d [%+.1f, %+.1f] %.3f | **%+d [%+.1f, %+.1f] %.3f** |" % (
                r["alpha"], name, d["C_K"], d["EC_K"], d["EC_correct"], d["P1"], d["P1_ci"][0], d["P1_ci"][1], d["P1_holm"],
                d["P2"], d["P2_ci"][0], d["P2_ci"][1], d["P2_holm"]))
    if rows:
        L += ["", "基线自答 K %d / 答对 %d；E 单独自答 K %d / 答对 %d。" % (rows[0]["baseline"]["K"], rows[0]["baseline"]["correct"],
                                                                  rows[0]["E_only"]["K"], rows[0]["E_only"]["correct"])]
    return "\n".join(L)


def verdict(all_rows):
    hits = []
    for model, rows in all_rows.items():
        for r in rows:
            for tag in ("abst", "ans"):
                d = r[tag]
                if d["P2"] > 0 and d["P2_holm"] < 0.05:
                    hits.append((model, r["alpha"], tag, d["P2"], d["P2_holm"]))
    models_with_hit = {h[0] for h in hits}
    return {"significant_P2_positive": hits, "models_with_a_hit": sorted(models_with_hit),
            "criterion_met": len(models_with_hit) >= 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e41_dir"); ap.add_argument("--models", nargs="+", default=["llama3.1-8b", "qwen3-8b", "gemma3-4b"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    all_rows = {m: load_model(a.e41_dir, m) for m in a.models}
    v = verdict(all_rows)
    md = ["# N7 第一步：C 中间力度曲线（博士原句的最后一次检验）", "",
          "事先标准：至少一个力度、至少两个模型上，E+C 自答的答对数显著多于 E 按置信度卡到同样多题（P2，Holm 校正）。**结论：%s**（有显著正 P2 的模型：%s）。" % (
              "成立" if v["criterion_met"] else "不成立", ", ".join(v["models_with_a_hit"]) or "无"), ""]
    for m, rows in all_rows.items():
        md.append(table(m, rows) if rows else "## %s：无结果" % m)
        md.append("")
    text = "\n".join(md)
    print(text)
    if a.out:
        open(a.out + ".md", "w", encoding="utf-8").write(text + "\n")
        json.dump({"rows": all_rows, "verdict": v}, open(a.out + ".json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
