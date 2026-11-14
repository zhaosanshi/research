# -*- coding: utf-8 -*-
"""P2 figure (Claude, 2026-11-14): change in right answers on the 529 test questions (523 scored; the 27B's E+C work dir
522), both judges with paired 95% intervals, from the JSON of ff_p2_summary.py (no number typed by hand).
Left: E alone on the must-answer prompt (prompt-only control, the learned direction at strengths 1 / 2, the mean-difference
and the better of the two random directions at strength 2). Right: the plain prompt at E strength 2 (E alone, E+C with the
C strength chosen by rule abstentions, E swapped for the better of the two random directions + the same C); for the 27B
the rule needs no C at strength 2 (E alone is within tolerance), so only E alone and the random direction are drawn.
Colours: categorical slots 1 / 2 of the validated default palette (light and dark steps), plus a marker shape per judge.
usage: ff_p2_figure.py SUMMARY.json OUT_PREFIX   -> OUT_PREFIX_light.png, OUT_PREFIX_dark.png"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#1c1c1a", "muted": "#6b6a64", "grid": "#e4e3de", "zero": "#9a998f",
              "pro": "#2a78d6", "allenai": "#eb6834"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#a8a79c", "grid": "#33332f", "zero": "#6f6e66",
             "pro": "#3987e5", "allenai": "#d95926"},
}
MODELS = [("llama", "Llama-3.1-8B", 11), ("9b", "Qwen3.5-9B", 15), ("27b", "Qwen3.8-27B", 28)]


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "Songti SC", "Arial Unicode MS", "Noto Sans CJK SC"):
        if name in have:
            return name
    raise SystemExit("no CJK font found")


def better_random(block, judge, a, b):
    """the random-direction condition (of two signs) with more right answers under this judge"""
    return max((block[judge][a], block[judge][b]), key=lambda s: s["dR"])


def rows_left(M, el):
    E = M["E_must_answer"]
    out = []
    for label, cond in (("只加 E2 句（改提示）", "baseline_conf_E2"), ("E 方向，力度 1", "E_L%d_conf-w-p1" % el),
                        ("E 方向，力度 2", "E_L%d_conf-w-p2" % el), ("均值差方向，力度 2", "E_L%d_conf-md-p2" % el)):
        out.append((label, {j: E[j][cond] for j in ("pro", "allenai")}))
    out.append(("随机方向，力度 2", {j: better_random(E, j, "E_L%d_conf-rand-p2" % el, "E_L%d_conf-rand-m2" % el) for j in ("pro", "allenai")}))
    return out


def ec_block(M, label_start):
    return next(b for b in M["EC"] if b["label"].startswith(label_start))


def rows_right(key, M, el):
    if key == "llama":
        b, cl, k = ec_block(M, "C19, rules v1.1"), 19, "0-8"
    elif key == "9b":
        b, cl, k = ec_block(M, "C16"), 16, "0-5"
    else:
        b = ec_block(M, "C33, rules v1 ")
        return [("只推 E，力度 2", {j: b[j]["E_L%d_x-w-p2" % el] for j in ("pro", "allenai")}),
                ("E+C，力度 2：不需要 C", None),
                ("随机方向，力度 2", {j: better_random(b, j, "E_L%d_x-rand-p2" % el, "E_L%d_x-rand-m2" % el) for j in ("pro", "allenai")})]
    ec = "X_x--E%d-w-p2-and-C%d-w-m%s" % (el, cl, k)
    return [("只推 E，力度 2", {j: b[j]["E_L%d_x-w-p2" % el] for j in ("pro", "allenai")}),
            ("E+C，力度 2 + C %s" % k.replace("-", "."), {j: b[j][ec] for j in ("pro", "allenai")}),
            ("E 换成随机方向 + C", {j: better_random(b, j, "X_x--E%d-rand-p2-and-C%d-w-m%s" % (el, cl, k),
                                                      "X_x--E%d-rand-m2-and-C%d-w-m%s" % (el, cl, k)) for j in ("pro", "allenai")})]


def panel(ax, groups, th, title, xlim, xlabel="答对题数的变化（相对不写入；523 题）"):
    y, ticks, labels, seps = 0, [], [], []
    for model, rows in groups:
        ax.annotate(model, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-8, 0), textcoords="offset points",
                    color=th["text"], fontsize=10.5, fontweight="bold", va="center", ha="right", annotation_clip=False)
        y += 1.0
        for label, pts in rows:
            ticks.append(y); labels.append(label)
            if pts is None:
                ax.text(3, y, "只推 E 已在容限内，规则不加 C", color=th["muted"], fontsize=8.5, va="center")
            else:
                for j, dy, mk in (("pro", -0.17, "o"), ("allenai", 0.17, "s")):
                    s = pts[j]
                    ax.plot([s["dR_lo"], s["dR_hi"]], [y + dy, y + dy], color=th[j], lw=2, solid_capstyle="round", zorder=3)
                    ax.plot([s["dR"]], [y + dy], marker=mk, ms=6.5, color=th[j], mec=th["surface"], mew=1.2, zorder=4)
            y += 1.0
        seps.append(y - 0.5)
        y += 0.6
    for s in seps[:-1]:
        ax.axhline(s + 0.3, color=th["grid"], lw=0.8, zorder=1)
    ax.axvline(0, color=th["zero"], lw=1.2, zorder=2)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9, color=th["text"])
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color=th["grid"], lw=0.8, zorder=0)
    ax.tick_params(axis="x", colors=th["muted"], labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel(xlabel, color=th["muted"], fontsize=9)
    ax.set_title(title, color=th["text"], fontsize=11.5, loc="left", pad=10)
    ax.set_facecolor(th["surface"])


def main():
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    prefix = sys.argv[2]
    plt.rcParams["font.family"] = pick_font()
    plt.rcParams["axes.unicode_minus"] = False
    for mode, th in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.55})
        fig.patch.set_facecolor(th["surface"])
        left = [(name, rows_left(d["models"][key], el)) for key, name, el in MODELS]
        right = [(name, rows_right(key, d["models"][key], el)) for key, name, el in MODELS]
        panel(axes[0], left, th, "必须作答问法：只推 E", (-40, 178))
        panel(axes[1], right, th, "普通问法（允许说\"不知道\"）：E+C", (-40, 178), "答对题数的变化（相对不写入；523 题，27B 一栏 522 题）")
        handles = [plt.Line2D([0], [0], color=th["pro"], marker="o", lw=2, ms=6.5, mec=th["surface"], label="DeepSeek V4 Pro"),
                   plt.Line2D([0], [0], color=th["allenai"], marker="s", lw=2, ms=6.5, mec=th["surface"], label="AllenAI 判分模型")]
        leg = fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.985, 0.995))
        for txt in leg.get_texts():
            txt.set_color(th["text"])
        fig.text(0.012, 0.012, "点 = 答对题数变化；横线 = 配对自助重抽 95% 区间（10000 次）。27B 的普通问法一栏来自 v1 的目录（522 题：Pro 无法判定的 1 题已排除）。"
                 "随机方向取两个符号里答对更多的那个。", color=th["muted"], fontsize=7.8)
        fig.subplots_adjust(left=0.2, right=0.985, top=0.9, bottom=0.11)
        out = "%s_%s.png" % (prefix, mode)
        fig.savefig(out, dpi=170, facecolor=th["surface"])
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
