# -*- coding: utf-8 -*-
"""Read-only DESCRIPTION of the E+C export (from_claude/36 section 3): one row per judged condition against its own base
prompt, keyed by the full condition name (a two-write condition is never read as its first layer). This is not the
combination analyser of GPT round 34 (equal-answer comparisons, simple selection); no claim about E+C rests on it.
usage: e44_EC_describe.py WORK [--judge allenai|pro]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e44_round1_tables as t  # noqa: E402

ORDER = [
    ("baseline_x", "no write"),
    ("E_L10_x-w-p2", "E alone (L10, +2)"),
    ("E_L10_x-rand-p2", "random instead of E (+2)"),
    ("E_L10_x-rand-m2", "random instead of E (-2)"),
    ("C_L31_x-w-m0-25", "C alone (L31, -0.25 = toward confident)"),
    ("C_L31_x-w-m0-5", "C alone (-0.5)"),
    ("C_L31_x-w-m1", "C alone (-1)"),
    ("C_L31_x-rand-m0-25", "random instead of C (-0.25)"),
    ("C_L31_x-rand-p0-25", "random instead of C (+0.25)"),
    ("X_x--E10-w-p2-and-C31-w-m0-25", "E + C -0.25 (pre-registered choice)"),
    ("X_x--E10-w-p2-and-C31-w-m0-5", "E + C -0.5"),
    ("X_x--E10-w-p2-and-C31-w-m1", "E + C -1"),
    ("X_x--E10-rand-p2-and-C31-w-m0-25", "random(+) instead of E, + C -0.25"),
    ("X_x--E10-rand-m2-and-C31-w-m0-25", "random(-) instead of E, + C -0.25"),
    ("X_x--E10-w-p2-and-C31-rand-m0-25", "E + random(-) instead of C"),
    ("X_x--E10-w-p2-and-C31-rand-p0-25", "E + random(+) instead of C"),
    ("baseline_x_F", "no write, complete-sentence prompt x_F"),
    ("E_L10_x_F-w-p2", "E alone on x_F"),
    ("C_L31_x_F-w-m0-25", "C alone on x_F (-0.25)"),
    ("X_x_F--E10-w-p2-and-C31-w-m0-25", "E + C -0.25 on x_F"),
    ("baseline_conf_F", "no write, must-answer + complete sentence (conf_F)"),
    ("baseline_conf_E1_F", "no write, conf_F + truth sentence (conf_E1_F)"),
    ("E_L10_conf_F-w-p1", "E alone on conf_F (+1)"),
    ("E_L10_conf_F-w-p2", "E alone on conf_F (+2)"),
    ("E_L10_conf_F-rand-p2", "random instead of E on conf_F (+2)"),
    ("E_L10_conf_F-rand-m2", "random instead of E on conf_F (-2)"),
]


def main():
    work = sys.argv[1]
    judge = sys.argv[sys.argv.index("--judge") + 1] if "--judge" in sys.argv else "allenai"
    gen, specs, ids, state, rule, words = t.load(work, judge)
    n_q = len(ids)
    names = [c for c, _ in ORDER if c in specs] + sorted(c for c in specs if c not in dict(ORDER))
    label = dict(ORDER)
    print("# %s, judge %s, %d questions (DESCRIPTION only; each condition against its own base prompt)\n" % (Path(work).name, judge, n_q))
    print("| condition | base | rule abstentions | R / W / H / X | answered | right among answered | dR [95%] | dAns | "
          "keep: n, right-rate change, W>R / R>W | lost R / lost W / gained R / gained W | mean words |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in names:
        s = specs[c]
        r, w, h, x = t.counts(state, ids, c)
        ra = sum(1 for i in ids if rule[(i, c)])
        mw = sum(words[(i, c)] for i in ids) / n_q
        acc = "%.1f%%" % (100.0 * r / (r + w)) if r + w else "nan"
        base = "—" if s.get("kind") == "prompt_control" else s["base_condition"].replace("baseline_", "")
        head = "| %s `%s` | %s | %d | %d / %d / %d / %d | %d | %s |" % (label.get(c, c), c, base, ra, r, w, h, x, r + w, acc)
        if s.get("kind") == "prompt_control":
            print(head + " — | — | — | — | %.1f |" % mw)
            continue
        v, n = t.per_question(state, words, ids, s["base_condition"], c)
        kr = (sum(v["keep"]) / n["keep"]) if n["keep"] else None
        print(head + " %s | %+d | %d, %s, %d / %d | %d / %d / %d / %d | %.1f |" % (
            t.ci(v["dR"])[0], sum(v["dAns"]), n["keep"], "undefined" if kr is None else "%+.1f pp" % (100 * kr),
            n["up"], n["down"], n["lost_right"], n["lost_wrong"], n["gain_right"], n["gain_wrong"], mw))
    print("\nR / W / H / X = right / wrong / hedge-or-uninformative / abstained; answered = R + W; dR = change in right answers; "
          "dAns = change in answered; keep = questions answered both times. Paired bootstrap as in e44_round1_tables.py.")


if __name__ == "__main__":
    main()
