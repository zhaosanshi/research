#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only tables for the E44 first round, criteria v3 (plan 28 as amended by plan 29 after GPT round 30).
CPU only, standard library only, nothing is written.

  e44_round1_tables.py screen  RUN_DIR FAMILY WORK_DIR            all layers, AllenAI + rule refusals, shortlist by the frozen rule
  e44_round1_tables.py cells   RUN_DIR FAMILY WORK_DIR --judge J --layers L [L ...]
                                                                  the two cells of this family for one judge (bootstrap intervals)
  e44_round1_tables.py cross   RUN_DIR FAMILY WORK_DIR --judge J --layers L [L ...]
                                                                  the other arms against their own base prompt

States per judge:   R right, W wrong, H hedge or uninformative, X abstained.
  allenai: rule abstention -> X whatever the judge said; otherwise A -> R, B -> W, N -> H.
  pro:     A -> R, B -> W, H -> H, C -> X.
"Answered" = R or W. Every comparison is paired on the same questions and judged by the same judge. An unscored cell stops
the script (re-judge first; nothing is imputed).

Per-question quantities for an arm c against its base b:
  dR   right(c) - right(b)          dAns  answered(c) - answered(b)          dX  abstained(c) - abstained(b)
  keep answered under both          out / in  answered -> not answered / not answered -> answered          toH  answered -> hedge
  G_same  dR restricted to keep-questions whose answer has at most LEN_TOL more words: a DIAGNOSTIC only (it conditions on
          what happened after the write, so it neither proves nor refutes a length effect; GPT round 30, point 2)
Tolerance: 3 percentage points. tau(n) = round(0.03 n) questions (6 of 192, 16 of 529).
Intervals: paired bootstrap over questions, 10,000 resamples, random.Random(20261111), percentile 2.5 / 97.5. An interval of zero
width (no change observed at all) is replaced by +-3/n for the equivalence check (rule of three).
"Equivalent" = the whole interval lies inside +-3 points; "point inside" = only the point estimate does (screening wording).
"""
import argparse, collections, json, random
from pathlib import Path

TAU_PP = 0.03
LEN_TOL = 1
BOOT, SEED = 10000, 20261111
_IDX = {}


def tau(n):
    return int(TAU_PP * n + 0.5)


def load(work, judge):
    work = Path(work)
    gen = json.loads((work / "judge_input.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (work / "judge2" / ("single_%s.jsonl" % judge)).open(encoding="utf-8")]
    state, rule = {}, {}
    for r in rows:
        key = (r["pair_id"], r["condition"])
        rule[key] = bool(r["rule_abstain"])
        if r["status"] != "scored":
            s = "?"
        elif judge == "allenai":
            s = "X" if r["rule_abstain"] else {"A": "R", "B": "W", "N": "H"}[r["label"]]
        else:
            s = {"A": "R", "B": "W", "H": "H", "C": "X"}[r["label"]]
        state[key] = s
    specs = {s["name"]: s for s in gen["condition_specs"]}
    ids = list(gen["question_ids"])
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


def boot_ratio(num, den):
    """Percentile 95% interval of sum(num) / sum(den), numerator and denominator resampled together."""
    vals = []
    for row in _idx(len(num)):
        d = sum(den[j] for j in row)
        vals.append((sum(num[j] for j in row) / d) if d else 0.0)
    vals.sort()
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]


def ci(d):
    lo, hi = boot(d)
    return "%+d [%+d, %+d]" % (sum(d), lo, hi), lo, hi


def equivalent(lo, hi, n, changed):
    """The whole interval of a rate difference inside +-TAU_PP. lo / hi are in rate units; a zero-width interval with no
    observed change is widened to +-3/n (rule of three)."""
    if not changed:
        lo, hi = -3.0 / n, 3.0 / n
    return lo >= -TAU_PP and hi <= TAU_PP


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


def _token(verdict):
    return "INCOMPLETE" if verdict.startswith("INCOMPLETE") else verdict.replace(" ", "_")


def keep_rate(v, n):
    """Change of the right-rate on the questions answered under both conditions (points of that intersection)."""
    return (sum(v["keep"]) / n["keep"]) if n["keep"] else 0.0


# ----------------------------------------------------------------------------- screen (AllenAI + rule refusals, all layers)
def screen(a):
    gen, specs, ids, state, rule, words = load(a.work, "allenai")
    fam, base, T, N = a.family, "baseline_x", tau(len(ids)), len(ids)
    L = layers_of(specs, fam)
    rb, wb, hb, xb = counts(state, ids, base)
    print("# %s first-round screen: AllenAI + rule refusals, tune %d questions, against %s (no write); tau = %d questions\n" % (fam, N, base, T))
    print("baseline x: right / wrong / uninformative / abstained = %d / %d / %d / %d; answered %d; mean words %.1f\n" % (
        rb, wb, hb, xb, rb + wb, sum(words[(i, base)] for i in ids) / N))
    arms = ["x.w.p1", "x.w.m1", "x.md.p1", "x.rand.p1"] if fam == "E" else ["x.w.p1", "x.w.m1", "x.md.p1", "x.md.m1", "x.rand.p1", "x.rand.m1"]
    S = {}
    for l in L:
        for arm in arms:
            c = cond(fam, l, arm)
            v, n = per_question(state, words, ids, base, c)
            S[(l, arm)] = {"cnt": counts(state, ids, c), "dR": sum(v["dR"]), "dAns": sum(v["dAns"]), "dX": sum(v["dX"]),
                           "keep_dR": sum(v["keep"]), "keep_rate": keep_rate(v, n), "G": sum(v["same"]), "n": n,
                           "words": sum(words[(i, c)] for i in ids) / N}
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
        print("\n## shortlist (plan 29, section 3)\n")
        ok = lambda l: abs(S[(l, "x.w.p1")]["dAns"]) <= T  # noqa: E731
        by_r = sorted(L, key=lambda l: (-S[(l, "x.w.p1")]["dR"], l))
        short = [(l, "candidate", "slot %d: dR %+d among layers with |dAns| <= %d" % (k + 1, S[(l, "x.w.p1")]["dR"], T))
                 for k, l in enumerate([l for l in by_r if ok(l) and S[(l, "x.w.p1")]["dR"] > 0][:2])]
        if not short:
            print("no layer with |dAns| <= %d has a positive dR under AllenAI: nothing qualifies for Pro (resource screen, not a finding of no effect)." % T)
        chosen = {l for l, _, _ in short}
        cand = sorted([(-S[(l, "x.w.p1")]["G"], l) for l in L if l not in chosen and ok(l) and S[(l, "x.w.p1")]["G"] > 0])
        if cand:
            short.append((cand[0][1], "candidate", "slot 3 (discovery): largest G_same %+d among the other layers with |dAns| <= %d" % (-cand[0][0], T)))
            chosen.add(cand[0][1])
        cand = [(mc_margin(a.run, fam, l, "x.w.p1"), l) for l in L if l not in chosen and ok(l)]
        cand = sorted([(-m, l) for m, l in cand if m is not None])
        if cand:
            short.append((cand[0][1], "candidate", "slot 4 (discovery, judge-free): largest max-true margin change %+.4f among the other layers with |dAns| <= %d" % (-cand[0][0], T)))
            chosen.add(cand[0][1])
        rest = sorted([l for l in L if l not in chosen], key=lambda l: (-S[(l, "x.w.p1")]["dR"], l))
        if rest:
            short.append((rest[0], "diagnostic", "slot 5 (diagnostic only, cannot be promoted): largest raw dR %+d among the layers not chosen" % S[(rest[0], "x.w.p1")]["dR"]))
        for l, role, why in short:
            s = S[(l, "x.w.p1")]
            print("- layer %d [%s] — %s; dR %+d, dAns %+d, out / in %d / %d, G_same %+d (n %d)" % (l, role, why, s["dR"], s["dAns"], s["n"]["out"], s["n"]["in"], s["G"], s["n"]["same"]))
        print("\nshortlist layers:", " ".join(str(l) for l, _, _ in short))
        print("candidate layers:", " ".join(str(l) for l, role, _ in short if role == "candidate"))
    else:
        base_ans = rb + wb
        print("| layer | x.w.p1 R/W/H/X | A+ = -dAns | dX | keep: n, right-rate change, W>R / R>W | toH | x.w.m1 R/W/H/X | A- = +dAns | dX | keep: n, right-rate change, W>R / R>W | toH | T = min | rand dAns: p1, m1 | md dAns: p1, m1 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        Tm, ret = {}, {}
        for l in L:
            p, m = S[(l, "x.w.p1")], S[(l, "x.w.m1")]
            ap, am = -p["dAns"], m["dAns"]
            Tm[l] = (min(ap, am), ap, am)
            for arm, s in (("p1", p), ("m1", m)):
                ret[(l, arm)] = abs(s["keep_rate"]) <= TAU_PP and 2 * s["n"]["keep"] >= base_ans and s["n"]["toH"] <= T
            print("| %d | %d / %d / %d / %d | %+d | %+d | %d, %+.1f pp, %d / %d | %d | %d / %d / %d / %d | %+d | %+d | %d, %+.1f pp, %d / %d | %d | %+d | %+d, %+d | %+d, %+d |" % (
                (l,) + p["cnt"] + (ap, p["dX"], p["n"]["keep"], 100 * p["keep_rate"], p["n"]["up"], p["n"]["down"], p["n"]["toH"]) + m["cnt"] + (
                    am, m["dX"], m["n"]["keep"], 100 * m["keep_rate"], m["n"]["up"], m["n"]["down"], m["n"]["toH"], Tm[l][0],
                    S[(l, "x.rand.p1")]["dAns"], S[(l, "x.rand.m1")]["dAns"], S[(l, "x.md.p1")]["dAns"], S[(l, "x.md.m1")]["dAns"])))
        rr = [S[(l, arm)]["dAns"] for l in L for arm in ("x.rand.p1", "x.rand.m1")]
        print("\nreference only (not a gate): the random direction's dAns over the %d layers and both signs ranges from %+d to %+d" % (len(L), min(rr), max(rr)))
        print("\n## shortlist (plan 29, section 3)\n")
        order = sorted(L, key=lambda l: (-Tm[l][0], l))
        two_ok = [l for l in order if Tm[l][0] > T and ret[(l, "p1")] and ret[(l, "m1")]]
        short = [(l, "candidate", "slot %d: two-way, T = %+d, retention screen met on both arms" % (k + 1, Tm[l][0])) for k, l in enumerate(two_ok[:2])]
        if not short:                                    # fall back to qualified one-way layers (GPT round 30, point 3)
            one = sorted([(-Tm[l][k], l, name) for l in L for k, name in ((1, "p1"), (2, "m1")) if Tm[l][k] > T and ret[(l, name)]])
            for neg, l, name in one:
                if l not in [x[0] for x in short]:
                    short.append((l, "candidate", "ONE-WAY slot %d: %s only, change %+d, retention screen met on that arm" % (
                        len(short) + 1, "toward caution" if name == "p1" else "toward confident", -neg)))
                if len(short) == 2:
                    break
            print("no layer is two-way with the retention screen met on both arms: %s" % ("one-way candidates below" if short else "no qualified layer"))
        chosen = {l for l, _, _ in short}
        rest = sorted([l for l in L if l not in chosen], key=lambda l: (-max(Tm[l][0], 0), -max(Tm[l][1], Tm[l][2]), l))
        if rest and max(Tm[rest[0]][1], Tm[rest[0]][2]) > T:
            short.append((rest[0], "diagnostic", "slot 3 (diagnostic only, cannot be promoted): largest change among the layers not chosen (T %+d)" % Tm[rest[0]][0]))
        for l, role, why in short:
            print("- layer %d [%s] — %s; A+ %+d, A- %+d" % (l, role, why, Tm[l][1], Tm[l][2]))
        print("\nshortlist layers:", " ".join(str(l) for l, _, _ in short))
        print("candidate layers:", " ".join(str(l) for l, role, _ in short if role == "candidate"))


# ----------------------------------------------------------------------------- the two cells of one family, one judge
def cells(a):
    gen, specs, ids, state, rule, words = load(a.work, a.judge)
    fam, base, T, N = a.family, "baseline_x", tau(len(ids)), len(ids)
    rb, wb, hb, xb = counts(state, ids, base)
    print("# %s cells under judge %s (%d questions; tau = %d; paired bootstrap %d resamples, seed %d; against %s)\n" % (
        fam, a.judge, N, T, BOOT, SEED, base))
    print("baseline x: right / wrong / hedge-or-uninformative / abstained = %d / %d / %d / %d; rule abstentions %d; mean words %.1f\n" % (
        rb, wb, hb, xb, sum(rule[(i, base)] for i in ids), sum(words[(i, base)] for i in ids) / N))
    all_arms = ["x.w.p1", "x.w.m1", "x.md.p1", "x.md.m1", "x.rand.p1", "x.rand.m1"]
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
            sR, loR, hiR = ci(v["dR"]); sA, loA, hiA = ci(v["dAns"]); sX, loX, hiX = ci(v["dX"]); sK, loK, hiK = ci(v["keep"])
            K[arm] = {"v": v, "n": n, "dR": sum(v["dR"]), "dAns": sum(v["dAns"]), "dX": sum(v["dX"]),
                      "loR": loR, "loA": loA, "hiA": hiA, "loX": loX, "hiX": hiX}
            print("| %s | %s | %.1f | %s | %s | %s | %d, %s, %d / %d | %d / %d / %d | %d / %d / %d / %d | %+d (%d) |" % (
                arm, fmt_counts(state, ids, c), sum(words[(i, c)] for i in ids) / N, sR, sA, sX, n["keep"], sK, n["up"], n["down"],
                n["out"], n["in"], n["toH"], n["lost_right"], n["lost_wrong"], n["gain_right"], n["gain_wrong"], sum(v["same"]), n["same"]))
        print()
        rands = [r for r in ("x.rand.p1", "x.rand.m1") if r in K]
        if fam == "E" and "x.w.p1" in K:
            k = K["x.w.p1"]
            e1 = k["loR"] > 0 and k["dR"] > T
            vs = [(r,) + ci(minus(k["v"]["dR"], K[r]["v"]["dR"])) for r in rands]
            if len(rands) < 2:
                e3 = None
            else:
                e3 = all(lo > 0 for _, _, lo, _ in vs)
            vs_txt = "; ".join("vs %s %s" % (r, s) for r, s, _, _ in vs) or "no random arm judged"
            eff = "INCOMPLETE (both signs of the random control are required)" if e3 is None else ("PASS" if (e1 and e3) else "not met")
            print("- E effect: (e1) dR %+d beyond tau and interval above 0: %s; (e3) above every random control — %s: %s  =>  %s" % (
                k["dR"], e1, vs_txt, "missing sign" if e3 is None else e3, eff))
            # format control: the same direction on the prompt that already asks for one complete sentence
            fb, fc = "baseline_x_F", cond(fam, l, "x_F.w.p1")
            fmt, dRF = "INCOMPLETE (arm x_F.w.p1 not judged)", None
            if have(state, ids, fb) and have(state, ids, fc):
                vF, nF = per_question(state, words, ids, fb, fc)
                sF, loF, hiF = ci(vF["dR"]); dRF = sum(vF["dR"])
                fmt = "PASS" if loF > 0 else "not met"
                print("- E against the format control (base x_F, which already asks for one complete sentence): x_F %s -> with the E write %s; dR %s, dAns %+d, mean words %.1f -> %.1f  =>  %s" % (
                    fmt_counts(state, ids, fb), fmt_counts(state, ids, fc), sF, sum(vF["dAns"]),
                    sum(words[(i, fb)] for i in ids) / N, sum(words[(i, fc)] for i in ids) / N, fmt))
            else:
                print("- E against the format control: %s" % fmt)
            lo, hi = boot(k["v"]["dAns"])
            point = abs(k["dAns"]) <= T
            eq = equivalent(lo / N, hi / N, N, any(k["v"]["dAns"]))
            print("- E retention: answered-rate change %+.1f pp [%+.1f, %+.1f]; point inside +-3 pp: %s; whole interval inside (equivalence, needed for a stage-B claim): %s; moved out / in: %d / %d" % (
                100 * k["dAns"] / N, 100 * lo / N, 100 * hi / N, point, eq, k["n"]["out"], k["n"]["in"]))
            print("- diagnostic: G_same %+d on %d questions; mean words %.1f -> %.1f" % (
                sum(k["v"]["same"]), k["n"]["same"], sum(words[(i, base)] for i in ids) / N, sum(words[(i, cond(fam, l, "x.w.p1"))] for i in ids) / N))
            print("KEY family=E judge=%s layer=%d dR=%+d dR_F=%s dAns=%+d point_retention=%s equivalence=%s effect=%s format=%s\n" % (
                a.judge, l, k["dR"], "NA" if dRF is None else "%+d" % dRF, k["dAns"], point, eq, _token(eff), _token(fmt)))
        if fam == "C" and ("x.w.p1" in K or "x.w.m1" in K):
            res = {}
            base_ans = rb + wb
            for arm, sign, name in (("x.w.p1", -1, "toward caution"), ("x.w.m1", +1, "toward confident")):
                if arm not in K:
                    continue
                k = K[arm]; v, n = k["v"], k["n"]
                vs = [(r,) + ci(minus(v["dAns"], K[r]["v"]["dAns"])) for r in rands]
                if sign < 0:
                    own = k["hiA"] < 0 and k["loX"] > 0 and -k["dAns"] > T
                    beat = all(hi < 0 for _, _, _, hi in vs)
                else:
                    own = k["loA"] > 0 and k["hiX"] < 0 and k["dAns"] > T
                    beat = all(lo > 0 for _, _, lo, _ in vs)
                eff = None if len(rands) < 2 else (own and beat)
                rate = keep_rate(v, n)
                lo, hi = boot_ratio(v["keep"], v["both"])
                enough = 2 * n["keep"] >= base_ans
                point = enough and abs(rate) <= TAU_PP and n["toH"] <= T
                eq = enough and n["toH"] <= T and equivalent(lo, hi, max(1, n["keep"]), any(v["keep"]))
                res[arm] = (eff, point, eq)
                acc_b = n["right_b"] / n["ans_b"] if n["ans_b"] else 0.0
                acc_c = n["right_c"] / n["ans_c"] if n["ans_c"] else 0.0
                print("- C %s (%s): dAns %+d, dX %+d; own conditions (both intervals on the intended side, change beyond tau): %s; against every random control — %s: %s  =>  %s" % (
                    name, arm, k["dAns"], k["dX"], own, "; ".join("vs %s %s" % (r, s) for r, s, _, _ in vs) or "none judged",
                    "missing sign" if len(rands) < 2 else beat, "INCOMPLETE (both signs of the random control are required)" if eff is None else ("effect" if eff else "no effect")))
                print("    retention on the %d questions answered both times (need >= %d): right-rate change %+.1f pp [%+.1f, %+.1f], W>R %d / R>W %d, answered -> hedge %d; point inside: %s; equivalence: %s" % (
                    n["keep"], (base_ans + 1) // 2, 100 * rate, 100 * lo, 100 * hi, n["up"], n["down"], n["toH"], point, eq))
                print("    reported, not a condition: right-rate among ALL answered %.1f%% -> %.1f%% (answers dropped: %d right, %d wrong; added: %d right, %d wrong)" % (
                    100 * acc_b, 100 * acc_c, n["lost_right"], n["lost_wrong"], n["gain_right"], n["gain_wrong"]))
            if any(r[0] is None for r in res.values()):
                eff_txt, ret_txt = "INCOMPLETE", "n/a"
            else:
                ok = [arm for arm in res if res[arm][0]]
                eff_txt = "TWO-WAY" if len(ok) == 2 else ("ONE-WAY(%s)" % ok[0] if ok else "not_met")
                ret_txt = ("point_inside" if all(res[arm][1] for arm in ok) else "not_met") if ok else "n/a"
                if ok and all(res[arm][2] for arm in ok):
                    ret_txt = "equivalent"
            ap = -K["x.w.p1"]["dAns"] if "x.w.p1" in K else 0
            am = K["x.w.m1"]["dAns"] if "x.w.m1" in K else 0
            print("- C effect: %s; C retention on the arm(s) that count: %s" % (eff_txt, ret_txt))
            print("KEY family=C judge=%s layer=%d T=%+d Aplus=%+d Aminus=%+d effect=%s retention=%s\n" % (a.judge, l, min(ap, am), ap, am, eff_txt, ret_txt))


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
    a = ap.parse_args()
    if a.mode != "screen" and not a.layers:
        raise SystemExit("cells / cross need --layers")
    {"screen": screen, "cells": cells, "cross": cross}[a.mode](a)


if __name__ == "__main__":
    main()
