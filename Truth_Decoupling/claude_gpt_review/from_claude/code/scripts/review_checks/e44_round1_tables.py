#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only tables for the E44 first round, criteria v4 (plans 28 + 29 + 30; GPT rounds 30 and 31).
Added after the criteria were frozen (2026-11-11 03:30, labelled EXPLORATORY in the output, default behaviour unchanged):
`cells --base <prompt> [--main <arm>]` applies the same E conditions to arms on another base prompt.
CPU only, standard library only, nothing is written.

  e44_round1_tables.py screen  RUN_DIR FAMILY WORK_DIR            all layers, AllenAI + rule refusals, shortlist by the frozen rule
  e44_round1_tables.py cells   RUN_DIR FAMILY WORK_DIR --judge J --layers L [L ...]
                                                                  the conditions of this family for one judge, in two columns:
                                                                  A = stage-A screen (point estimates only), B = stage-B confirmation (intervals)
  e44_round1_tables.py cross   RUN_DIR FAMILY WORK_DIR --judge J --layers L [L ...]
                                                                  the other arms against their own base prompt

States per judge:   R right, W wrong, H hedge or uninformative, X abstained.
  allenai: rule abstention -> X whatever the judge said; otherwise A -> R, B -> W, N -> H.
  pro:     A -> R, B -> W, H -> H, C -> X.
"Answered" = R or W. Every comparison is paired on the same questions and judged by the same judge. An unscored cell stops
the script (re-judge first; nothing is imputed). One exception, by rule, not by imputation (2026-11-14, first seen on the
27B): an EMPTY answer is never sent to a judge (judge_two_axis_run.py single: basis empty_output_no_judge_vote) and counts
as X, abstained, under both judges -- the two-axis standard of 2026-11-08 ("完整拒答短语与空输出一律 C").

Per-question quantities for an arm c against its base b:
  dR   right(c) - right(b)          dAns  answered(c) - answered(b)          dX  abstained(c) - abstained(b)
  keep answered under both          out / in  answered -> not answered / not answered -> answered          toH  answered -> hedge
  G_same  dR restricted to keep-questions whose answer has at most LEN_TOL more words: a diagnostic only.
Tolerance TOL = 0.03 as a RATE everywhere (of all questions; of the keep-questions for the keep right-rate). Counts are shown
for reading only: on 192 questions 5 is inside (2.6%), 6 is beyond (3.1%).
Intervals: paired bootstrap over questions, 10,000 resamples, random.Random(20261111), percentile 2.5 / 97.5. When not a single
question changed, the interval used is +-u with u = 1 - 0.05 ** (1 / n) (exact bound for zero events), and that is what is printed.
A rate whose denominator is zero is UNDEFINED (never 0).
Status of a column: not_met if any condition is false; otherwise incomplete if a required arm is missing; otherwise met.
"""
import argparse, collections, json, random
from pathlib import Path

TOL = 0.03
LEN_TOL = 1
BOOT, SEED = 10000, 20261111
RANDS = ("x.rand.p1", "x.rand.m1")          # both signs of one random axis; one judged export per axis
_IDX = {}


def load(work, judge):
    work = Path(work)
    gen = json.loads((work / "judge_input.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (work / "judge2" / ("single_%s.jsonl" % judge)).open(encoding="utf-8")]
    state, rule = {}, {}
    for r in rows:
        key = (r["pair_id"], r["condition"])
        rule[key] = bool(r["rule_abstain"])
        if r["status"] != "scored":
            s = "X" if (r.get("basis") == "empty_output_no_judge_vote" and r.get("empty_output") is True) else "?"
        elif judge == "allenai":
            s = "X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]]
        else:
            s = {"A": "R", "B": "W", "H": "H", "C": "X"}[r["label"]]
        state[key] = s
    specs = {s["name"]: s for s in gen["condition_specs"]}
    # Questions whose only gold answer is an abstention ("I have no comment") are excluded by the judging itself (no judge
    # row for any of their cells, either judge); they leave every table the same way (2026-11-14: the test split has 6,
    # the tune split none -- as the v1 test table of from_claude/10 put them aside)
    per_q = collections.defaultdict(set)
    cells_path = work / "judge2" / "cells.jsonl"
    if cells_path.exists():
        for l in cells_path.open(encoding="utf-8"):
            c = json.loads(l)
            per_q[c["pair_id"]].add(bool(c.get("gold_is_abstention")))
    mixed = [q for q, v in per_q.items() if len(v) > 1]
    if mixed:
        raise SystemExit("gold_is_abstention differs between the cells of one question: %s" % mixed[:3])
    excluded = sorted(q for q, v in per_q.items() if v == {True})
    gen["_excluded_abstention_gold"] = excluded
    ids = [i for i in gen["question_ids"] if i not in set(excluded)]
    bad = [(i, c) for c in gen["conditions"] for i in ids if state.get((i, c), "?") == "?"]
    if bad:
        raise SystemExit("unscored cells: %d (first %s); re-judge them before any table" % (len(bad), bad[:3]))
    words = {(r["pair_id"], c): len((t or "").split()) for r in gen["evaluations"] for c, t in r["outputs"].items()}
    return gen, specs, ids, state, rule, words


def cond(family, layer, arm):
    return "%s_L%d_%s" % (family, layer, arm.replace(".", "-"))


def counts(state, ids, c):
    k = collections.Counter(state[(i, c)] for i in ids)
    return k["R"], k["W"], k["H"], k["X"]


def fmt_counts(state, ids, c):
    return "%d / %d / %d / %d" % counts(state, ids, c)


def per_question(state, words, ids, b, c):
    """Per-question difference vectors and transition counts of arm c against base b."""
    ans = lambda s: s in "RW"  # noqa: E731
    v = {k: [] for k in ("dR", "dAns", "dX", "keep", "both", "same")}
    n = collections.Counter()
    for i in ids:
        x, y = state[(i, b)], state[(i, c)]
        dr = (y == "R") - (x == "R")
        both = ans(x) and ans(y)
        short = both and (words[(i, c)] - words[(i, b)]) <= LEN_TOL
        v["dR"].append(dr); v["dAns"].append(ans(y) - ans(x)); v["dX"].append((y == "X") - (x == "X"))
        v["keep"].append(dr if both else 0); v["both"].append(1 if both else 0); v["same"].append(dr if short else 0)
        n["keep"] += both; n["same"] += short
        n["out"] += ans(x) and not ans(y); n["in"] += (not ans(x)) and ans(y); n["toH"] += ans(x) and y == "H"
        n["up"] += both and x == "W" and y == "R"; n["down"] += both and x == "R" and y == "W"
        n["lost_right"] += x == "R" and not ans(y); n["lost_wrong"] += x == "W" and not ans(y)
        n["gain_right"] += (not ans(x)) and y == "R"; n["gain_wrong"] += (not ans(x)) and y == "W"
        n["ans_b"] += ans(x); n["ans_c"] += ans(y); n["right_b"] += x == "R"; n["right_c"] += y == "R"
    return v, n


def _idx(n):
    if n not in _IDX:
        rng = random.Random(SEED)
        _IDX[n] = [[rng.randrange(n) for _ in range(n)] for _ in range(BOOT)]
    return _IDX[n]


def boot(diffs):
    """Percentile 95% interval of the SUM of per-question differences (paired bootstrap; common resamples for every statistic)."""
    sums = sorted(sum(diffs[j] for j in row) for row in _idx(len(diffs)))
    return sums[int(0.025 * BOOT)], sums[int(0.975 * BOOT) - 1]


def rate_interval(diffs, n):
    """(rate, lo, hi) of sum(diffs) / n. If no question changed at all, the exact zero-event bound replaces the zero-width interval."""
    if not any(diffs):
        u = 1.0 - 0.05 ** (1.0 / n)
        return 0.0, -u, u
    lo, hi = boot(diffs)
    return sum(diffs) / n, lo / n, hi / n


def keep_interval(v, n):
    """(rate, lo, hi) of the right-rate change on the questions answered under both conditions; None when undefined.
    Numerator and denominator are resampled together; a resample without any such question makes the interval undefined."""
    if n["keep"] == 0:
        return None
    if not any(v["keep"]):
        u = 1.0 - 0.05 ** (1.0 / n["keep"])
        return 0.0, -u, u
    vals = []
    for row in _idx(len(v["keep"])):
        d = sum(v["both"][j] for j in row)
        if d == 0:
            return None
        vals.append(sum(v["keep"][j] for j in row) / d)
    vals.sort()
    return sum(v["keep"]) / n["keep"], vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]


def ci(d):
    lo, hi = boot(d)
    return "%+d [%+d, %+d]" % (sum(d), lo, hi), lo, hi


def pp(t):
    return "undefined" if t is None else "%+.1f pp [%+.1f, %+.1f]" % (100 * t[0], 100 * t[1], 100 * t[2])


def status(conditions):
    """conditions: values True / False / None (None = a required arm is missing)."""
    if any(c is False for c in conditions):
        return "not_met"
    if any(c is None for c in conditions):
        return "incomplete"
    return "met"


def minus(a, b):
    return [x - y for x, y in zip(a, b)]


def mc_margin(run, family, layer, arm):
    p = Path(run) / "mc" / ("%s_L%d_%s_tune.json" % (family, layer, arm.replace(".", "-")))
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["summary"].get("margin_change_mean")


def layers_of(specs, family):
    return sorted({s["layer"] for s in specs.values() if s.get("family") == family})


def have(state, ids, c):
    return (ids[0], c) in state


# ----------------------------------------------------------------------------- screen (AllenAI + rule refusals, all layers)
def screen(a):
    gen, specs, ids, state, rule, words = load(a.work, "allenai")
    fam, base, N = a.family, "baseline_x", len(ids)
    L = layers_of(specs, fam)
    rb, wb, hb, xb = counts(state, ids, base)
    print("# %s first-round screen: AllenAI + rule refusals, tune %d questions, against %s (no write); tolerance %.0f%% of the questions\n" % (fam, N, base, 100 * TOL))
    print("baseline x: right / wrong / uninformative / abstained = %d / %d / %d / %d; answered %d; mean words %.1f\n" % (
        rb, wb, hb, xb, rb + wb, sum(words[(i, base)] for i in ids) / N))
    arms = ["x.w.p1", "x.w.m1", "x.md.p1", "x.rand.p1"] if fam == "E" else ["x.w.p1", "x.w.m1", "x.md.p1", "x.md.m1", "x.rand.p1", "x.rand.m1"]
    S = {}
    for l in L:
        for arm in arms:
            c = cond(fam, l, arm)
            v, n = per_question(state, words, ids, base, c)
            S[(l, arm)] = {"cnt": counts(state, ids, c), "dR": sum(v["dR"]), "dAns": sum(v["dAns"]), "dX": sum(v["dX"]),
                           "keep_dR": sum(v["keep"]), "keep_rate": (sum(v["keep"]) / n["keep"]) if n["keep"] else None,
                           "G": sum(v["same"]), "n": n, "words": sum(words[(i, c)] for i in ids) / N}
    inside = lambda x: abs(x) / N <= TOL  # noqa: E731
    beyond = lambda x: x / N > TOL  # noqa: E731
    if fam == "E":
        print("| layer | x.w.p1 R/W/H/X | dR | dAns | dX | out / in | keep: n, dR (W>R / R>W) | G_same (n) [diagnostic] | mean words | rand.p1: dR, dAns | md.p1: dR, dAns | w.m1: dR | max-true margin change |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for l in L:
            s, r, m, neg = S[(l, "x.w.p1")], S[(l, "x.rand.p1")], S[(l, "x.md.p1")], S[(l, "x.w.m1")]
            mg = mc_margin(a.run, fam, l, "x.w.p1")
            print("| %d | %d / %d / %d / %d | %+d | %+d | %+d | %d / %d | %d, %+d (%d / %d) | %+d (%d) | %.1f | %+d, %+d | %+d, %+d | %+d | %s |" % (
                (l,) + s["cnt"] + (s["dR"], s["dAns"], s["dX"], s["n"]["out"], s["n"]["in"], s["n"]["keep"], s["keep_dR"], s["n"]["up"], s["n"]["down"],
                                   s["G"], s["n"]["same"], s["words"], r["dR"], r["dAns"], m["dR"], m["dAns"], neg["dR"],
                                   "—" if mg is None else "%+.4f" % mg)))
        rr = [S[(l, "x.rand.p1")]["dR"] for l in L]
        print("\nreference only (not a gate): the random direction's dR over the %d layers ranges from %+d to %+d" % (len(L), min(rr), max(rr)))
        print("\n## shortlist (plan 29 section 5, as amended by plan 30)\n")
        ok = lambda l: inside(S[(l, "x.w.p1")]["dAns"])  # noqa: E731
        by_r = sorted(L, key=lambda l: (-S[(l, "x.w.p1")]["dR"], l))
        short = [(l, "to be qualified", "slot %d: dR %+d among layers whose answered-rate change is inside the tolerance" % (k + 1, S[(l, "x.w.p1")]["dR"]))
                 for k, l in enumerate([l for l in by_r if ok(l) and S[(l, "x.w.p1")]["dR"] > 0][:2])]
        if not short:
            print("no layer with an answered-rate change inside the tolerance has a positive dR under AllenAI: nothing is sent to Pro (a resource screen, not a finding of no effect).")
        chosen = {l for l, _, _ in short}
        cand = sorted([(-S[(l, "x.w.p1")]["G"], l) for l in L if l not in chosen and ok(l) and S[(l, "x.w.p1")]["G"] > 0])
        if cand:
            short.append((cand[0][1], "to be qualified", "slot 3 (discovery): largest G_same %+d among the other such layers" % -cand[0][0]))
            chosen.add(cand[0][1])
        cand = [(mc_margin(a.run, fam, l, "x.w.p1"), l) for l in L if l not in chosen and ok(l)]
        cand = sorted([(-m, l) for m, l in cand if m is not None])
        if cand:
            short.append((cand[0][1], "to be qualified", "slot 4 (discovery, judge-free): largest max-true margin change %+.4f among the other such layers" % -cand[0][0]))
            chosen.add(cand[0][1])
        rest = sorted([l for l in L if l not in chosen], key=lambda l: (-S[(l, "x.w.p1")]["dR"], l))
        if rest:
            short.append((rest[0], "diagnostic", "slot 5 (diagnostic only, cannot be promoted): largest raw dR %+d among the layers not chosen" % S[(rest[0], "x.w.p1")]["dR"]))
        for l, role, why in short:
            s = S[(l, "x.w.p1")]
            print("- layer %d [%s] — %s; dR %+d, dAns %+d (%.1f%%), out / in %d / %d, G_same %+d (n %d)" % (
                l, role, why, s["dR"], s["dAns"], 100 * s["dAns"] / N, s["n"]["out"], s["n"]["in"], s["G"], s["n"]["same"]))
        print("\nshortlist layers:", " ".join(str(l) for l, _, _ in short))
        print("layers to be qualified:", " ".join(str(l) for l, role, _ in short if role != "diagnostic"))
    else:
        base_ans = rb + wb
        print("| layer | x.w.p1 R/W/H/X | A+ = -dAns | dX | keep: n, right-rate change, W>R / R>W | toH | x.w.m1 R/W/H/X | A- = +dAns | dX | keep: n, right-rate change, W>R / R>W | toH | T = min | rand dAns: p1, m1 | md dAns: p1, m1 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        Tm, ret = {}, {}
        rate_txt = lambda s: "undefined" if s["keep_rate"] is None else "%+.1f pp" % (100 * s["keep_rate"])  # noqa: E731
        for l in L:
            p, m = S[(l, "x.w.p1")], S[(l, "x.w.m1")]
            ap, am = -p["dAns"], m["dAns"]
            Tm[l] = (min(ap, am), ap, am)
            for arm, s in (("p1", p), ("m1", m)):
                ret[(l, arm)] = (s["keep_rate"] is not None and abs(s["keep_rate"]) <= TOL and 2 * s["n"]["keep"] >= base_ans
                                 and s["n"]["toH"] / N <= TOL)
            print("| %d | %d / %d / %d / %d | %+d | %+d | %d, %s, %d / %d | %d | %d / %d / %d / %d | %+d | %+d | %d, %s, %d / %d | %d | %+d | %+d, %+d | %+d, %+d |" % (
                (l,) + p["cnt"] + (ap, p["dX"], p["n"]["keep"], rate_txt(p), p["n"]["up"], p["n"]["down"], p["n"]["toH"]) + m["cnt"] + (
                    am, m["dX"], m["n"]["keep"], rate_txt(m), m["n"]["up"], m["n"]["down"], m["n"]["toH"], Tm[l][0],
                    S[(l, "x.rand.p1")]["dAns"], S[(l, "x.rand.m1")]["dAns"], S[(l, "x.md.p1")]["dAns"], S[(l, "x.md.m1")]["dAns"])))
        rr = [S[(l, arm)]["dAns"] for l in L for arm in ("x.rand.p1", "x.rand.m1")]
        print("\nreference only (not a gate): the random direction's dAns over the %d layers and both signs ranges from %+d to %+d" % (len(L), min(rr), max(rr)))
        print("\n## shortlist (plan 29 section 5, as amended by plan 30)\n")
        order = sorted(L, key=lambda l: (-Tm[l][0], l))
        two_ok = [l for l in order if beyond(Tm[l][0]) and ret[(l, "p1")] and ret[(l, "m1")]]
        short = [(l, "to be qualified", "slot %d: two-way, T = %+d, retention screen met on both arms" % (k + 1, Tm[l][0])) for k, l in enumerate(two_ok[:2])]
        if not short:                                    # fall back to qualified one-way layers (GPT round 30, point 3)
            one = sorted([(-Tm[l][k], l, name) for l in L for k, name in ((1, "p1"), (2, "m1")) if beyond(Tm[l][k]) and ret[(l, name)]])
            for neg, l, name in one:
                if l not in [x[0] for x in short]:
                    short.append((l, "to be qualified", "ONE-WAY slot %d: %s only, change %+d, retention screen met on that arm" % (
                        len(short) + 1, "toward caution" if name == "p1" else "toward confident", -neg)))
                if len(short) == 2:
                    break
            print("no layer is two-way with the retention screen met on both arms: %s" % ("one-way layers below" if short else "no layer passes the screen"))
        chosen = {l for l, _, _ in short}
        rest = sorted([l for l in L if l not in chosen], key=lambda l: (-max(Tm[l][0], 0), -max(Tm[l][1], Tm[l][2]), l))
        if rest and beyond(max(Tm[rest[0]][1], Tm[rest[0]][2])):
            short.append((rest[0], "diagnostic", "slot 3 (diagnostic only, cannot be promoted): largest change among the layers not chosen (T %+d)" % Tm[rest[0]][0]))
        for l, role, why in short:
            print("- layer %d [%s] — %s; A+ %+d, A- %+d" % (l, role, why, Tm[l][1], Tm[l][2]))
        print("\nshortlist layers:", " ".join(str(l) for l, _, _ in short))
        print("layers to be qualified:", " ".join(str(l) for l, role, _ in short if role != "diagnostic"))


# ----------------------------------------------------------------------------- the conditions of one family, one judge
def cells(a):
    gen, specs, ids, state, rule, words = load(a.work, a.judge)
    fam, N = a.family, len(ids)
    bn = a.base                                        # "x" = the frozen criteria; any other base prompt is exploratory
    base = "baseline_" + bn
    st = a.strength                                   # "1" = the frozen criteria; any other strength is exploratory
    main_arm = a.main or (bn + ".w.p" + st)
    rands = (bn + ".rand.p" + st, bn + ".rand.m" + st)
    c_plus, c_minus = "x.w.p" + st, "x.w.m" + st
    rb, wb, hb, xb = counts(state, ids, base)
    print("# %s conditions under judge %s (%d questions; tolerance %.0f%%; paired bootstrap %d resamples, seed %d; against %s)" % (
        fam, a.judge, N, 100 * TOL, BOOT, SEED, base))
    if gen.get("_excluded_abstention_gold"):
        print("Excluded, as in the judging: %d question(s) whose only gold answer is an abstention (%s).\n" % (
            len(gen["_excluded_abstention_gold"]), " ".join(gen["_excluded_abstention_gold"])))
    if bn != "x" or main_arm != "x.w.p1" or st != "1":
        print("EXPLORATORY: base prompt %s, main arm %s, strength %s — outside the frozen criteria (which use base x, arm x.w.p1, strength 1)." % (bn, main_arm, st))
    print("A = stage-A screen: point estimates only. B = stage-B confirmation: intervals; on the tuning questions column B is descriptive.\n")
    print("baseline %s: right / wrong / hedge-or-uninformative / abstained = %d / %d / %d / %d; rule abstentions %d; mean words %.1f\n" % (
        bn, rb, wb, hb, xb, sum(rule[(i, base)] for i in ids), sum(words[(i, base)] for i in ids) / N))
    all_arms = [bn + t + st for t in (".w.p", ".w.m", ".md.p", ".md.m", ".rand.p", ".rand.m")]
    if main_arm not in all_arms:
        all_arms.insert(0, main_arm)
    for l in a.layers:
        print("## layer %d\n" % l)
        print("| arm | R / W / H / X | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | out / in / toH | lost R / lost W / gained R / gained W | G_same (n) [diagnostic] |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        K = {}
        for arm in all_arms:
            c = cond(fam, l, arm)
            if not have(state, ids, c):
                continue
            v, n = per_question(state, words, ids, base, c)
            sR, loR, hiR = ci(v["dR"]); sA, loA, hiA = ci(v["dAns"]); sX, loX, hiX = ci(v["dX"]); sK, _, _ = ci(v["keep"])
            K[arm] = {"v": v, "n": n, "dR": sum(v["dR"]), "dAns": sum(v["dAns"]), "dX": sum(v["dX"]),
                      "loR": loR, "loA": loA, "hiA": hiA, "loX": loX, "hiX": hiX}
            print("| %s | %s | %.1f | %s | %s | %s | %d, %s, %d / %d | %d / %d / %d | %d / %d / %d / %d | %+d (%d) |" % (
                arm, fmt_counts(state, ids, c), sum(words[(i, c)] for i in ids) / N, sR, sA, sX, n["keep"], sK, n["up"], n["down"],
                n["out"], n["in"], n["toH"], n["lost_right"], n["lost_wrong"], n["gain_right"], n["gain_wrong"], sum(v["same"]), n["same"]))
        print()
        missing = [r for r in rands if r not in K]
        if fam == "E" and main_arm in K:
            k = K[main_arm]
            # main effect on the plain prompt
            a1 = k["dR"] / N > TOL
            b1 = a1 and k["loR"] > 0
            # every random control (both signs of the axis are required)
            vs = [(r,) + ci(minus(k["v"]["dR"], K[r]["v"]["dR"])) for r in rands if r in K]
            a3 = None if missing else all(k["dR"] - K[r]["dR"] > 0 for r in rands)
            b3 = None if missing else all(lo > 0 for _, _, lo, _ in vs)
            # the same write on the prompt that already asks for one complete sentence, with its own answered-rate retention
            fb, fc = "baseline_x_F", cond(fam, l, "x_F.w.p" + st)
            aF = bF = None; dRF = dAF = None
            if bn != "x":
                # no complete-sentence variant of this prompt was frozen: condition F is NOT TESTED (never counted as met)
                print("- F, complete-sentence background: NOT TESTED for base %s (no such prompt was frozen); the column status stays incomplete" % bn)
            elif have(state, ids, fb) and have(state, ids, fc):
                vF, nF = per_question(state, words, ids, fb, fc)
                sF, loF, _ = ci(vF["dR"]); dRF, dAF = sum(vF["dR"]), sum(vF["dAns"])
                rF = rate_interval(vF["dAns"], N)
                aF = dRF > 0 and abs(dAF) / N <= TOL
                bF = aF and loF > 0 and rF[1] >= -TOL and rF[2] <= TOL
                print("- F, complete-sentence background (base x_F): x_F %s -> with the E write %s; dR %s; answered-rate change %s; out / in %d / %d; mean words %.1f -> %.1f" % (
                    fmt_counts(state, ids, fb), fmt_counts(state, ids, fc), sF, pp(rF), nF["out"], nF["in"],
                    sum(words[(i, fb)] for i in ids) / N, sum(words[(i, fc)] for i in ids) / N))
            else:
                print("- F, complete-sentence background: arm x_F.w.p1 not judged")
            # answered-rate retention on the plain prompt
            rK = rate_interval(k["v"]["dAns"], N)
            aK = abs(k["dAns"]) / N <= TOL
            bK = rK[1] >= -TOL and rK[2] <= TOL
            A, B = status([a1, a3, aF, aK]), status([b1, b3, bF, bK])
            tested = status([c for c in (a1, a3, aK) if c is not None]) if bn != "x" else None   # the conditions that were tested, F excluded
            print("- main effect: dR %+d (%.1f%% of the questions) | A: beyond the tolerance: %s | B: and interval above 0: %s" % (k["dR"], 100 * k["dR"] / N, a1, b1))
            print("- random controls: %s%s | A: above each (point): %s | B: each interval above 0: %s" % (
                "; ".join("vs %s %s" % (r, s) for r, s, _, _ in vs) or "none judged", ("; MISSING " + ", ".join(missing)) if missing else "", a3, b3))
            print("- F | A: dR on x_F positive and its answered-rate change inside the tolerance: %s | B: dR interval above 0 and answered-rate interval inside: %s" % (
                "not tested" if aF is None else aF, "not tested" if bF is None else bF))
            print("- retention on %s: answered-rate change %s; moved out / in %d / %d | A: point inside: %s | B: whole interval inside: %s" % (bn, pp(rK), k["n"]["out"], k["n"]["in"], aK, bK))
            print("- diagnostic: G_same %+d on %d questions; mean words %.1f -> %.1f" % (
                sum(k["v"]["same"]), k["n"]["same"], sum(words[(i, base)] for i in ids) / N, sum(words[(i, cond(fam, l, main_arm))] for i in ids) / N))
            margin = "NA" if missing else "%+d" % min(k["dR"] - K[r]["dR"] for r in rands)
            print("KEY family=E judge=%s layer=%d A=%s B=%s dR=%+d dR_F=%s dAns=%+d dAns_F=%s rand_margin=%s%s\n" % (
                a.judge, l, A, B, k["dR"], "NA" if dRF is None else "%+d" % dRF, k["dAns"], "NA" if dAF is None else "%+d" % dAF, margin,
                "" if (bn == "x" and main_arm == "x.w.p1" and st == "1") else " EXPLORATORY base=%s main=%s strength=%s%s" % (
                    bn, main_arm, st, "" if tested is None else " tested_A_without_F=%s" % tested)))
        if fam == "C" and bn == "x" and (c_plus in K or c_minus in K):
            base_ans = rb + wb
            qa, qb, ea = {}, {}, {}
            for arm, sign, name in ((c_plus, -1, "toward caution"), (c_minus, +1, "toward confident")):
                if arm not in K:
                    continue
                k = K[arm]; v, n = k["v"], k["n"]
                moved = sign * k["dAns"]                 # answers removed (caution) or added (confident)
                vs = [(r,) + ci(minus(v["dAns"], K[r]["v"]["dAns"])) for r in rands if r in K]
                own_a = moved / N > TOL and sign * k["dX"] < 0
                own_b = own_a and ((k["hiA"] < 0 and k["loX"] > 0) if sign < 0 else (k["loA"] > 0 and k["hiX"] < 0))
                beat_a = None if missing else all(sign * (k["dAns"] - K[r]["dAns"]) > 0 for r in rands)
                beat_b = None if missing else all((hi < 0) if sign < 0 else (lo > 0) for _, _, lo, hi in vs)
                kr = keep_interval(v, n)
                enough = n["keep"] > 0 and 2 * n["keep"] >= base_ans
                hedge = n["toH"] / N <= TOL
                keep_a = enough and hedge and kr is not None and abs(kr[0]) <= TOL
                keep_b = keep_a and kr[1] >= -TOL and kr[2] <= TOL
                ea[arm] = status([own_a, beat_a])
                qa[arm] = status([own_a, beat_a, keep_a]); qb[arm] = status([own_b, beat_b, keep_b])
                acc = lambda r, d: ("%.1f%%" % (100.0 * r / d)) if d else "undefined"  # noqa: E731
                print("- C %s (%s): dAns %+d (%.1f%%), dX %+d | A: beyond the tolerance, abstentions move the same way: %s | B: and both intervals on that side: %s" % (
                    name, arm, k["dAns"], 100 * k["dAns"] / N, k["dX"], own_a, own_b))
                print("    random controls: %s%s | A: larger than each (point): %s | B: each interval on that side: %s" % (
                    "; ".join("vs %s %s" % (r, s) for r, s, _, _ in vs) or "none judged", ("; MISSING " + ", ".join(missing)) if missing else "", beat_a, beat_b))
                print("    retention on the %d questions answered both times (need at least %d): right-rate change %s, W>R %d / R>W %d, answered -> hedge %d | A: point inside: %s | B: whole interval inside: %s" % (
                    n["keep"], (base_ans + 1) // 2, pp(kr), n["up"], n["down"], n["toH"], keep_a, keep_b))
                print("    reported, not a condition: right-rate among ALL answered %s -> %s (answers dropped: %d right, %d wrong; added: %d right, %d wrong)" % (
                    acc(n["right_b"], n["ans_b"]), acc(n["right_c"], n["ans_c"]), n["lost_right"], n["lost_wrong"], n["gain_right"], n["gain_wrong"]))

            def summary(q):
                if any(s == "incomplete" for s in q.values()) and not any(s == "met" for s in q.values()):
                    return "incomplete"
                ok = [arm for arm in q if q[arm] == "met"]
                return "TWO-WAY" if len(ok) == 2 else ("ONE-WAY(%s)" % ok[0] if ok else "not_met")
            ap = -K[c_plus]["dAns"] if c_plus in K else 0
            am = K[c_minus]["dAns"] if c_minus in K else 0
            print("KEY family=C judge=%s layer=%d A=%s B=%s A_effect_only=%s T=%+d Aplus=%+d Aminus=%+d%s\n" % (
                a.judge, l, summary(qa), summary(qb), summary(ea), min(ap, am), ap, am, "" if st == "1" else " EXPLORATORY strength=%s" % st))


# ----------------------------------------------------------------------------- the other arms
def cross(a):
    gen, specs, ids, state, rule, words = load(a.work, a.judge)
    fam = a.family
    print("# %s arms on other prompts, judge %s (each arm against its own base prompt, no write)\n" % (fam, a.judge))
    for l in a.layers:
        print("## layer %d\n" % l)
        print("| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | out / in / toH | G_same (n) |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name, s in sorted(specs.items()):
            if s.get("family") != fam or s.get("layer") != l or s["arm"].startswith("x."):
                continue
            b = s["base_condition"]
            v, n = per_question(state, words, ids, b, name)
            print("| %s | %s | %s | %s | %s | %s | %d, %+d, %d / %d | %d / %d / %d | %+d (%d) |" % (
                s["arm"], fmt_counts(state, ids, b), fmt_counts(state, ids, name), ci(v["dR"])[0], ci(v["dAns"])[0], ci(v["dX"])[0],
                n["keep"], sum(v["keep"]), n["up"], n["down"], n["out"], n["in"], n["toH"], sum(v["same"]), n["same"]))
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("screen", "cells", "cross"))
    ap.add_argument("run"); ap.add_argument("family", choices=("E", "C")); ap.add_argument("work")
    ap.add_argument("--judge", choices=("allenai", "pro"), default="allenai")
    ap.add_argument("--layers", nargs="+", type=int, default=[])
    ap.add_argument("--base", default="x", help="cells, E only: base prompt of the arms (default x = the frozen criteria; anything else is exploratory)")
    ap.add_argument("--main", default=None, help="cells, E only: main arm (default <base>.w.p<strength>; e.g. conf.wq.p1 for the per-question interchange)")
    ap.add_argument("--strength", default="1", help="cells: strength suffix of the arms, as written in the arm names (default 1 = the frozen criteria; e.g. 0.5 is exploratory)")
    a = ap.parse_args()
    if a.mode != "screen" and not a.layers:
        raise SystemExit("cells / cross need --layers")
    {"screen": screen, "cells": cells, "cross": cross}[a.mode](a)


if __name__ == "__main__":
    main()
