#!/usr/bin/env python3
"""The figure for the Dr's true/false comparison (plan of 2026-11-10): one dot-and-interval panel per comparison, read from
dr_tf_table.py's json (Claude, 2026-11-12). Its table twin is the .md that dr_tf_table.py writes next to the json.

Left panel : AUROC change against the original model, mean difference (orange) and DAS (blue), point and pair-bootstrap 95% CI.
Right panel: DAS minus mean difference (neutral ink: a derived difference, not a third method).
One row per model x report set (the Dr's 100 pairs, all 529 test pairs). Shared x scale; zero line; reading rule in the subtitle.
Colors: reference categorical slots 1 (blue) and 2 (orange), validated light and dark (validate_palette.py: all checks pass).

usage: dr_tf_figure.py <dr_tf_table json> --out <stem> [--metric AUROC] [--title ...]
output: <stem>_light.png and <stem>_dark.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "DAS": "#2a78d6", "meandiff": "#eb6834", "diff": "#52514e"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "DAS": "#3987e5", "meandiff": "#d95926", "diff": "#c3c2b7"},
}
NAMES = {"qwen3.5-9b": "Qwen3.5-9B", "qwen3.8-27b": "Qwen3.8-27B", "qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama-3.1-8B",
         "gemma3-4b": "Gemma-3-4B", "mistral-7b": "Mistral-7B"}
SETS = (("100 pairs", "博士的 100 对"), ("529 pairs", "全部 529 对"))


def rows_of(rep, metric):
    rows = []
    for model, m in rep["models"].items():
        for key, label in SETS:
            s = m["sets"].get(key)
            if s is None:
                continue
            d = s["diffs"]
            get = lambda name: (d[name][metric]["point"], d[name][metric]["ci95"]) if name in d else None
            rows.append({"label": "%s\n%s" % (NAMES.get(model, model), label), "meandiff": get("meandiff - original"),
                         "DAS": get("DAS - original"), "diff": get("DAS - meandiff"), "chosen": m.get("chosen", {})})
    return rows


def draw(rows, theme, out, metric, title):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": ["Hiragino Sans GB", "PingFang HK", "Arial Unicode MS", "sans-serif"], "font.size": 10,
                         "axes.unicode_minus": False})
    n = len(rows)
    head, foot, per_row, width = 1.25, 0.62, 0.62, 10.5          # inches: title + subtitle + legend, x-axis band, one row
    height = head + foot + per_row * n
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True, gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.06})
    fig.patch.set_facecolor(t["surface"])
    vals = [v for r in rows for k in ("meandiff", "DAS", "diff") if r[k] for v in (r[k][1][0], r[k][1][1], r[k][0])]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    pad = 0.08 * (hi - lo or 0.01)
    for ax in axes:
        ax.set_facecolor(t["surface"])
        ax.set_xlim(lo - pad, hi + pad)
        ax.axvline(0, color=t["axis"], lw=1.0, zorder=1)
        ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0)
    ys = list(range(n))[::-1]

    def point(ax, y, val, color):
        if val is None:
            return
        p, (a, b) = val
        ax.plot([a, b], [y, y], color=color, lw=2.0, solid_capstyle="round", zorder=3)
        ax.plot([p], [y], marker="o", ms=7, color=color, mec=t["surface"], mew=1.6, zorder=4)

    for r, y in zip(rows, ys):
        point(axes[0], y + 0.13, r["meandiff"], t["meandiff"])
        point(axes[0], y - 0.13, r["DAS"], t["DAS"])
        point(axes[1], y, r["diff"], t["diff"])
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r["label"] for r in rows], color=t["ink"])
    axes[0].set_ylim(-0.7, n - 0.3)
    axes[0].set_title("相对原模型的 %s 变化" % metric, loc="left", color=t["ink"], fontsize=10.5, pad=6)
    axes[1].set_title("DAS − 均值差", loc="left", color=t["ink"], fontsize=10.5, pad=6)
    for ax in axes:
        ax.set_xlabel("%s 差" % metric, color=t["ink2"])
    legend = [Line2D([0], [0], color=t["meandiff"], lw=2, marker="o", ms=7, mec=t["surface"], mew=1.6, label="均值差 − 原模型"),
              Line2D([0], [0], color=t["DAS"], lw=2, marker="o", ms=7, mec=t["surface"], mew=1.6, label="DAS − 原模型")]
    inch = 1.0 / height                                            # one inch in figure coordinates (vertical)
    fig.text(0.012, 1 - 0.12 * inch, title, ha="left", va="top", color=t["ink"], fontsize=12)
    fig.text(0.012, 1 - 0.42 * inch, "点 = 点估计；横线 = 配对自助重抽 95%% 区间（整对重抽 2000 次）。区间整段在 0 右边才算提升；%s 按 p_true 算。"
             % metric, ha="left", va="top", color=t["ink2"], fontsize=9)
    lg = fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.006, 1 - 0.62 * inch), ncol=2, frameon=False, fontsize=9,
                    handlelength=2.2, columnspacing=1.6)
    for txt in lg.get_texts():
        txt.set_color(t["ink2"])
    fig.subplots_adjust(left=0.135, right=0.985, top=1 - head * inch, bottom=foot * inch)
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="AUROC", choices=("AUROC", "AUROC_margin", "J", "yes_rate", "B"))
    ap.add_argument("--title", default="真假判断：DAS 与均值差相对原模型的 AUROC 变化")
    args = ap.parse_args()
    rep = json.load(open(args.table_json, encoding="utf-8"))
    rows = rows_of(rep, args.metric)
    if not rows:
        raise SystemExit("no rows in %s" % args.table_json)
    for theme in THEMES:
        draw(rows, theme, "%s_%s.png" % (args.out, theme), args.metric, args.title)
    print("wrote %s_light.png and %s_dark.png (%d rows)" % (args.out, args.out, len(rows)))


if __name__ == "__main__":
    main()
