# -*- coding: utf-8 -*-
"""P2 tables (Claude, 2026-11-14): the complete free-form tables on the 529 test questions, generated from the JSON of
ff_p2_summary.py (no number typed by hand). Each cell: change against the condition without any write, "Pro (AllenAI)";
brackets are paired 95% bootstrap intervals (10,000 resamples) of the table code.
usage: ff_p2_tables.py SUMMARY.json > ff_table_529.md"""
import json
import sys

MODELS = [("llama", "Llama-3.1-8B", 11), ("9b", "Qwen3.5-9B", 15), ("27b", "Qwen3.8-27B", 28)]
LABELS = {
    "C19, rules v1 (R6 layer)": "C 第 19 层，规则 v1（R6 选的层）",
    "C31, rules v1 (second C layer)": "C 第 31 层，规则 v1（第二个 C 层）",
    "C19, rules v1.1, E strength 2": "C 第 19 层，规则 v1.1（加密档，只重跑力度 2）",
    "C16, rules v1 (R6 layer; within tolerance, so also v1.1)": "C 第 16 层，规则 v1（已在容限内，v1.1 同此）",
    "C33, rules v1 (C 0.25 overshoots the answers)": "C 第 33 层，规则 v1（C 0.25 让作答多出约 90 题，只作参照）",
    "C33, rules v1.1, E strength 1": "C 第 33 层，规则 v1.1（力度 1）",
    "C19, rules v1": "C 第 19 层，规则 v1", "C31, rules v1": "C 第 31 层，规则 v1",
    "C16, rules v1": "C 第 16 层，规则 v1", "C33, rules v1": "C 第 33 层，规则 v1",
}
VERDICTS = [("combination wins", "E+C 胜（两家区间都在 0 以上）"), ("comparator wins", "对照胜（两家区间都在 0 以下）"),
            ("leans to the combination", "偏向 E+C（两家点估计都 > 0，区间含 0）"),
            ("leans to the comparator", "偏向对照（两家点估计都 < 0，区间含 0）"), ("cannot tell", "分不出（两家方向不一致或为 0）")]


def zh_verdict(v):
    return next((z for e, z in VERDICTS if v.startswith(e)), v)


def ci(s):
    return "%+d [%+d, %+d]" % (s["dR"], s["dR_lo"], s["dR_hi"])


def ans(s):
    return "%+d [%+d, %+d]" % (s["dAns"], s["dAns_lo"], s["dAns_hi"])


def both(block, cond, f=ci):
    return "%s（%s）" % (f(block["pro"][cond]), f(block["allenai"][cond]))


def tok(s):
    return str(s).replace(".", "-")


def main():
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    P = print
    P("# 自由回答，529 道测试题：完整对照表（由 ff_p2_summary.py 的 JSON 生成）\n")
    P("每格是相对不写入时的变化：先写 DeepSeek V4 Pro，括号里是 AllenAI。方括号是配对自助重抽的 95% 区间（10000 次）。")
    P("题数：523（6 道金标只有\"无可奉告\"的题已排除）；27B 的 v1 E+C 目录 522（Pro 无法判定的 1 题已排除，`45` 第 3.5 节）。\n")

    P("## 1. E：必须作答问法，答对题数变化\n")
    P("| 模型（E 层） | 只加 E2 句（改提示） | E 方向，力度 1 | E 方向，力度 2 | 均值差方向，力度 1 | 均值差方向，力度 2 | 随机方向，力度 2（两个符号） | 平均词数：不写入 → 力度 2 |")
    P("|---|---|---|---|---|---|---|---|")
    for key, name, el in MODELS:
        E = d["models"][key]["E_must_answer"]
        r2p, r2m = "E_L%d_conf-rand-p2" % el, "E_L%d_conf-rand-m2" % el
        w2 = E["pro"]["E_L%d_conf-w-p2" % el]
        P("| %s（%d） | %s | %s | %s | %s | %s | %+d / %+d（%+d / %+d） | %.1f → %.1f |" % (
            name, el, both(E, "baseline_conf_E2"), both(E, "E_L%d_conf-w-p1" % el), both(E, "E_L%d_conf-w-p2" % el),
            both(E, "E_L%d_conf-md-p1" % el), both(E, "E_L%d_conf-md-p2" % el),
            E["pro"][r2p]["dR"], E["pro"][r2m]["dR"], E["allenai"][r2p]["dR"], E["allenai"][r2m]["dR"],
            w2["words_base"], w2["words"]))

    P("\n## 2. C：普通问法，力度 0.25，作答题数变化（冻结判据 v4 的 B 栏看区间）\n")
    P("| 模型 · C 层 | 往谨慎 | 往敢答 | 随机方向（两个符号） | 判据 v4 B 栏：Pro / AllenAI |")
    P("|---|---|---|---|---|")
    for key, name, el in MODELS:
        for L, C in d["models"][key]["C"].items():
            k = C["criterion_v4"]
            P("| %s · %s | %s | %s | %+d / %+d（%+d / %+d） | %s / %s |" % (
                name, L, both(C, "C_L%s_x-w-p0-25" % L, ans), both(C, "C_L%s_x-w-m0-25" % L, ans),
                C["pro"]["C_L%s_x-rand-p0-25" % L]["dAns"], C["pro"]["C_L%s_x-rand-m0-25" % L]["dAns"],
                C["allenai"]["C_L%s_x-rand-p0-25" % L]["dAns"], C["allenai"]["C_L%s_x-rand-m0-25" % L]["dAns"],
                k.get("pro", {}).get("B"), k.get("allenai", {}).get("B")))

    P("\n## 3. E+C：普通问法（允许说\"不知道\"），C 的力度按规则弃答数事先选、不看对错\n")
    P("| 模型 · 目录 | E 力度 + C（方向、力度） | 只推 E：答对 / 作答 | 只推 C（同档）：答对 / 作答 | **E+C：答对 / 作答** | E 换成随机方向 + C：答对（两个符号） | C 换成随机方向：答对（两个符号） | 平均词数：不写入 → E+C |")
    P("|---|---|---|---|---|---|---|---|")
    for key, name, el in MODELS:
        for b in d["models"][key]["EC"]:
            cl, sg = b["c_layer"], b["c_sign"]
            other = "p" if sg == "m" else "m"
            for s, k in b["choice"].items():
                ec = "X_x--" + "-and-".join(tok(p) for p in ["E%d.w.p%s" % (el, s), "C%d.w.%s%s" % (cl, sg, k)])
                ea, ca = "E_L%d_x-w-p%s" % (el, tok(s)), "C_L%d_x-w-%s%s" % (cl, sg, tok(k))
                er = ["X_x--E%d-rand-%s%s-and-C%d-w-%s%s" % (el, sgn, tok(s), cl, sg, tok(k)) for sgn in ("p", "m")]
                cr = ["X_x--E%d-w-p%s-and-C%d-rand-%s%s" % (el, tok(s), cl, sgn, tok(k)) for sgn in (sg, other)]
                pr, al = b["pro"], b["allenai"]
                cell = lambda c: "%+d / %+d" % (pr[c]["dR"], pr[c]["dAns"])  # noqa: E731
                P("| %s · %s | %s + %s %s | %s | %s | **%s / %+d**（%s） | %+d / %+d | %+d / %+d | %.1f → %.1f |" % (
                    name, LABELS.get(b["label"], b["label"]), s, "往敢答" if sg == "m" else "往谨慎", k, cell(ea), cell(ca),
                    ci(pr[ec]), pr[ec]["dAns"], ci(al[ec]), pr[er[0]]["dR"], pr[er[1]]["dR"], pr[cr[0]]["dR"], pr[cr[1]]["dR"],
                    pr[ec]["words_base"], pr[ec]["words"]))
    P("\n27B 力度 2：规则 v1.1 判定只推 E 已在容限内，不加 C；只推 E 见 v1 目录那一行的\"只推 E\"一列。")

    P("\n## 4. 同作答数对照（规则选的那一档；E+C 减对照，答对题数）\n")
    P("| 模型 · 目录 | E 力度 + C | N（E+C 作答的题） | 对\"只推 C + 按把握挑题\"（弱形式） | 对\"必须作答问法只推 E + 按把握挑题\"（强形式） | 事先写下的读法 |")
    P("|---|---|---|---|---|---|")
    for key, name, el in MODELS:
        for q in d["models"][key]["equal_N"]:
            for opk, rows in q["primary_rows"].items():
                s, k = opk.split("|")
                def f(j, sysname):
                    r = rows.get(j, {}).get(sysname)
                    return "比不了" if r is None else "%+d [%+d, %+d]" % (r["dR"], r["dR_lo"], r["dR_hi"])
                weak = "C -%s + selection" % k
                strong = "E + selection (must-answer prompt)"
                verdict = q["verdicts"].get("E +%s, C -%s (primary) | %s" % (s, k, strong), "—")
                P("| %s · %s | %s + %s | %s | %s（%s） | %s（%s） | %s |" % (
                    name, LABELS.get(q["label"], q["label"]), s, k, rows.get("N"), f("pro", weak), f("allenai", weak), f("pro", strong), f("allenai", strong), zh_verdict(verdict)))


if __name__ == "__main__":
    main()
