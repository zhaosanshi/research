# -*- coding: utf-8 -*-
"""Read-only DESCRIPTION of an E+C export of ff_explore_ec.sh (from_claude/44 section 5.8). The Llama versions are
results/09_E44_两因子交换训练_Llama/scripts/e44_EC_describe.py and e44_why_E_abstains.py; here the layers and strengths are
parameters. Not a combination analysis (no equal-answer comparison, no selection by confidence); no claim rests on it.
  1. one row per judged condition against its own base prompt, keyed by the full condition name
  2. for each E strength s of the choice file: the questions that E s alone turns into a rule abstention although they
     were answered without any write, their labels without the write, and how many of them E s + C (the chosen k)
     answers again, with which labels
usage: ff_explore_ec_describe.py WORK --judge allenai|pro --el 15 --cl 16 --choice CHOICE.json"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t  # noqa: E402


def tok(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def two(e_part, c_part, el, cl):
    return "x:%s+%s" % ((e_part, c_part) if el < cl else (c_part, e_part))


def single(family, layer, kind, sign, beta):
    """Condition name of a single write in an eval_extra export (the driver's naming)."""
    return "%s_L%d_x-%s-%s%s" % (family, layer, kind, sign, beta.replace(".", "-"))


def combo(el, cl, e_kind, e_sign, s, c_kind, c_sign, k):
    return "X_" + tok(two("E%d.%s.%s%s" % (el, e_kind, e_sign, s), "C%d.%s.%s%s" % (cl, c_kind, c_sign, k), el, cl))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("work")
    p.add_argument("--judge", choices=("allenai", "pro"), default="allenai")
    p.add_argument("--el", type=int, required=True)
    p.add_argument("--cl", type=int, required=True)
    p.add_argument("--choice", type=Path, required=True)
    a = p.parse_args()
    choice = json.load(open(a.choice, encoding="utf-8"))
    if (choice["e_layer"], choice["c_layer"]) != (a.el, a.cl):
        raise SystemExit("the choice file is for other layers")
    gen, specs, ids, state, rule, words = t.load(a.work, a.judge)
    n_q = len(ids)
    el, cl = a.el, a.cl

    def rank(c):
        s = specs[c]
        if s.get("kind") == "prompt_control":
            return (0, c)
        if c.startswith("E_"):
            return (1, c)
        if c.startswith("C_"):
            return (2, c)
        return (3, c)

    names = sorted(specs, key=rank)
    chosen = {combo(el, cl, "w", "p", s, "w", "m", k): s for s, k in choice["choice"].items()}
    print("# %s, judge %s, %d questions (DESCRIPTION only; each condition against its own base prompt)\n" % (Path(a.work).name, a.judge, n_q))
    print("Choice of k fixed before judging (rule abstentions; baseline x = %d): %s\n" % (
        choice["baseline_rule_abstentions"], ", ".join("E +%s -> C -%s" % (s, k) for s, k in choice["choice"].items())))
    print("| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | "
          "keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in names:
        s = specs[c]
        r, w, h, x = t.counts(state, ids, c)
        ra = sum(1 for i in ids if rule[(i, c)])
        mw = sum(words[(i, c)] for i in ids) / n_q
        acc = "%.1f%%" % (100.0 * r / (r + w)) if r + w else "nan"
        tag = " (chosen k)" if c in chosen else ""
        if s.get("kind") == "prompt_control":
            print("| `%s` | — | %d | %d / %d / %d / %d | %d | %s | — | — | — | — | %.1f |" % (c, ra, r, w, h, x, r + w, acc, mw))
            continue
        v, n = t.per_question(state, words, ids, s["base_condition"], c)
        kr = (sum(v["keep"]) / n["keep"]) if n["keep"] else None
        print("| `%s`%s | %s | %d | %d / %d / %d / %d | %d | %s | %s | %+d | %d, %s, %d / %d | %d / %d / %d / %d | %.1f |" % (
            c, tag, s["base_condition"].replace("baseline_", ""), ra, r, w, h, x, r + w, acc, t.ci(v["dR"])[0], sum(v["dAns"]),
            n["keep"], "undefined" if kr is None else "%+.1f pp" % (100 * kr), n["up"], n["down"],
            n["lost_right"], n["lost_wrong"], n["gain_right"], n["gain_wrong"], mw))
    print("\nR / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; "
          "dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.\n")

    print("## The questions E alone turns into \"I don't know\" (rule abstention; answered without any write)\n")
    cnt = lambda qs, c: {k: sum(1 for q in qs if state[(q, c)] == k) for k in "RWHX"}  # noqa: E731
    pct = lambda d: "%.0f%%" % (100.0 * d["R"] / (d["R"] + d["W"])) if d["R"] + d["W"] else "undefined"  # noqa: E731
    for s, k in choice["choice"].items():
        e_alone, ec = single("E", el, "w", "p", s), combo(el, cl, "w", "p", s, "w", "m", k)
        if e_alone not in specs or ec not in specs:
            print("- E +%s: condition missing from this export" % s)
            continue
        new = [q for q in ids if rule[(q, e_alone)] and not rule[(q, "baseline_x")]]
        keep = [q for q in ids if not rule[(q, e_alone)] and not rule[(q, "baseline_x")]]
        nb, kb = cnt(new, "baseline_x"), cnt(keep, "baseline_x")
        back = [q for q in new if not rule[(q, ec)]]
        nbk = cnt(back, ec)
        print("- E +%s: %d questions. Without the write they were R / W / H / X = %d / %d / %d / %d (right %s of R + W); "
              "the questions answered both times were %d / %d / %d / %d (right %s)." % (
                  s, len(new), nb["R"], nb["W"], nb["H"], nb["X"], pct(nb), kb["R"], kb["W"], kb["H"], kb["X"], pct(kb)))
        print("  With E +%s and C -%s (toward confident), %d of them are no longer rule abstentions: R / W / H / X = %d / %d / %d / %d." % (
            s, k, len(back), nbk["R"], nbk["W"], nbk["H"], nbk["X"]))


if __name__ == "__main__":
    main()
