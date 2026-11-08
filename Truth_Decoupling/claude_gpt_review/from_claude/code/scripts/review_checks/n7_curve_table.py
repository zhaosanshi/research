#!/usr/bin/env python3
"""N7 step 1: the C mid-strength curve, one table per model from the per-strength E41 analyses.

usage: n7_curve_table.py E41_DIR [--models llama3.1-8b qwen3-8b gemma3-4b] [--out PREFIX]
E41_DIR holds <model>/e41_n7_a<alpha>_analysis.json (written by review_checks/n7_run.sh).
Per strength and per C sign the table reports, on the test half: C alone's self-selected answer count K (how far the
knob moved), E+C's correct answers among its own self-selected K, and the pre-declared contrasts of e41_analysis.py:
  P1 = E+C natural minus E thresholded on its gate margin to the same K
  P2 = E+C natural minus E thresholded on its True/False confidence to the same K   (primary)
with pair-bootstrap 95% CIs and Holm correction across ALL predeclared strengths and both P1/P2/signs within model.
The resulting criterion concerns E thresholding only; it cannot establish superiority to C alone or the paper's full
claim. Missing strengths/models are incomplete evidence, not a negative result.
"""
import argparse
import glob
import json
import os
import re

DEFAULT_ALPHAS = (.25, .5, .75, 1., 1.5, 2.)

ALPHA = re.compile(r"e41_n7_a([0-9.]+)_analysis\.json$")


def load_model(e41_dir, model):
    rows = []; seen = set()
    for path in sorted(glob.glob(os.path.join(e41_dir, model, "e41_n7_a*_analysis.json"))):
        m = ALPHA.search(path)
        if not m:
            continue
        alpha = float(m.group(1))
        if alpha in seen: raise ValueError("duplicate alpha files for " + model)
        seen.add(alpha)
        A = json.load(open(path, encoding="utf-8"))
        if A.get("model") != model: raise ValueError("analysis belongs to another model")
        c_names = list(A["fixed_C"].keys())
        c_abst = [c for c in c_names if A["fixed_C"][c]["sign"] == -1][0]
        c_ans = [c for c in c_names if A["fixed_C"][c]["sign"] == 1][0]
        if any(float(A["fixed_C"][c]["alpha"]) != alpha for c in (c_abst, c_ans)):
            raise ValueError("filename strength differs from the actual C metadata")
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
                        "P1": p1["diff"], "P1_ci": p1["ci95"], "P1_raw_p": p1["boot_p"], "P1_holm_per_strength_historical": p1["holm_p"],
                        "P2": p2["diff"], "P2_ci": p2["ci95"], "P2_raw_p": p2["boot_p"], "P2_holm_per_strength_historical": p2["holm_p"]}
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


def verdict(all_rows, expected_alphas=DEFAULT_ALPHAS):
    """Correct the complete prespecified strength family; never select a per-strength winner uncorrected."""
    import math
    expected = tuple(float(a) for a in expected_alphas)
    if not expected or len(expected) != len(set(expected)) or any(not math.isfinite(a) or a <= 0 for a in expected):
        raise ValueError('expected strengths must be distinct, positive, finite values')
    hits, missing = [], {}
    for model, rows in all_rows.items():
        by_alpha = {r['alpha']: r for r in rows}
        if len(by_alpha) != len(rows) or set(by_alpha) - set(expected):
            raise ValueError('duplicate or undeclared strength in ' + model)
        missing[model] = sorted(set(expected) - set(by_alpha))
        if rows and (len({r['E'] for r in rows}) != 1 or len({r['n_test'] for r in rows}) != 1
                     or any(r['baseline'] != rows[0]['baseline'] or r['E_only'] != rows[0]['E_only'] for r in rows)):
            raise ValueError('E, cohort size, baseline or E-only result changed across strengths')
        entries = []
        for alpha in expected:
            for tag in ('abst', 'ans'):
                for contrast in ('P1', 'P2'):
                    row = by_alpha.get(alpha)
                    # Missing tests count in the declared family but cannot give a completed decision.
                    value = row[tag][contrast + '_raw_p'] if row is not None else 1.
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError('invalid raw bootstrap p')
                    entries.append((value, row, tag, contrast))
        ordered = sorted(range(len(entries)), key=lambda i: entries[i][0])
        running = 0.
        for rank, index in enumerate(ordered):
            value, row, tag, contrast = entries[index]
            running = max(running, min(1., (len(entries) - rank) * value))
            if row is not None: row[tag][contrast + '_holm'] = running
        for row in rows:
            for tag in ('abst', 'ans'):
                d = row[tag]
                if d['P2'] > 0 and d['P2_holm'] < .05:
                    hits.append((model, row['alpha'], tag, d['P2'], d['P2_holm']))
    complete = bool(all_rows) and not any(missing.values())
    models_with_hit = {h[0] for h in hits}
    return {'significant_P2_positive': hits, 'models_with_a_hit': sorted(models_with_hit),
            'criterion_met': len(models_with_hit) >= 2 if complete else None,
            'status': 'complete' if complete else 'incomplete', 'missing_strengths': missing,
            'expected_strengths': list(expected), 'holm_tests_per_model': 4 * len(expected),
            'claim_scope': 'E+C versus E confidence-thresholded to the same K; C-only at that K was not tested here',
            'full_any_single_direction_claim_established': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e41_dir"); ap.add_argument("--models", nargs="+", default=["llama3.1-8b", "qwen3-8b", "gemma3-4b"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS), help="complete prespecified strength family")
    a = ap.parse_args()
    all_rows = {m: load_model(a.e41_dir, m) for m in a.models}
    v = verdict(all_rows, a.alphas)
    label = "未完成" if v["criterion_met"] is None else ("支持该有限比较" if v["criterion_met"] else "未见该有限比较的显著支持")
    md = ["# N7 第一步：C 中间力度曲线", "",
          "有限比较：E+C与单E置信筛题在同K下比较；Holm按每模型全部预声明力度×两方向×P1/P2统一校正。结果：**%s**（显著正P2模型：%s）。" % (label, ", ".join(v["models_with_a_hit"]) or "无"),
          "尚未在此检验同K的单C，因此不能据此宣布组合胜过任一单方向；未完成的数据也不能写成负结果。", ""]
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
