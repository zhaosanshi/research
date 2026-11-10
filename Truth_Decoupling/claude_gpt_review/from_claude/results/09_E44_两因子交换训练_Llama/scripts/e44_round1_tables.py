#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only tables for the E44 first round, criteria v2 (plan 28). CPU only, standard library only, nothing is written.

  e44_round1_tables.py screen  RUN_DIR FAMILY WORK_DIR            all layers, AllenAI + rule refusals, shortlist by the frozen rule
  e44_round1_tables.py cells   RUN_DIR FAMILY WORK_DIR --judge J --layers L [L ...]
                                                                  the two cells of this family for one judge (bootstrap intervals)
  e44_round1_tables.py cross   RUN_DIR FAMILY WORK_DIR --judge J --layers L [L ...]
                                                                  per-question interchange / isolate arms against their own base prompt

States per judge:   R right, W wrong, H hedge or uninformative, X abstained.
  allenai: rule abstention -> X whatever the judge said; otherwise A -> R, B -> W, N -> H.
  pro:     A -> R, B -> W, H -> H, C -> X (the rule abstention count is printed next to it).
"Answered" = R or W. Every comparison is paired on the same questions and judged by the same judge; the reference is the plain
prompt x without a write. An unscored cell stops the script (it must be re-judged first; nothing is imputed).

Per-question quantities for an arm c against its base b:
  dR      right(c) - right(b)                       dAns   answered(c) - answered(b)             dX   abstained(c) - abstained(b)
  keep    answered under both b and c               same   keep AND the answer under c has at most LEN_TOL more words than under b
  G_same  dR restricted to `same`  (a gain that is neither a coverage change nor a longer answer)
  out / in   answered -> not answered / not answered -> answered          toH   answered -> hedge
tau(n) = round(0.03 n): the tolerance of 3 percentage points (6 questions of 192, 16 of 529).
Intervals: paired bootstrap over questions, 10,000 resamples, random.Random(20261111), percentile 2.5 / 97.5 (one implementation only).
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
    v = {k: [] for k in ("dR", "dAns", "dX", "keep", "same")}
    n = collections.Counter()
    for i in ids:
        x, y = state[(i, b)], state[(i, c)]
        dr = (y == "R") - (x == "R")
        both = ans(x) and ans(y)
        short = both and (words[(i, c)] - words[(i, b)]) <= LEN_TOL
        v["dR"].append(dr); v["dAns"].append(ans(y) - ans(x)); v["dX"].append((y == "X") - (x == "X"))
        v["keep"].append(dr if both else 0); v["same"].append(dr if short else 0)
        n["keep"] += both; n["same"] += short
        n["out"] += ans(x) and not ans(y); n["in"] += (not ans(x)) and ans(y); n["toH"] += ans(x) and y == "H"
        n["up"] += both and x == "W" and y == "R"; n["down"] += both and x == "R" and y == "W"
        n["x_ans_lost_right"] += x == "R" and y == "X"; n["x_ans_lost_wrong"] += x == "W" and y == "X"
        n["x_gain_right"] += x == "X" and y == "R"; n["x_gain_wrong"] += x == "X" and y == "W"
    return v, n


def boot(diffs):
    """Percentile 95% interval of the SUM of per-question differences (paired bootstrap; common resamples for every statistic)."""
    n = len(diffs)
    if n not in _IDX:
        rng = random.Random(SEED)
        _IDX[n] = [[rng.randrange(n) for _ in range(n)] for _ in range(BOOT)]
    sums = sorted(sum(diffs[j] for j in row) for row in _IDX[n])
    return sums[int(0.025 * BOOT)], sums[int(0.975 * BOOT) - 1]


def ci(d):
    lo, hi = boot(d)
    return "%+d [%+d, %+d]" % (sum(d), lo, hi), lo, hi


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
    fam, base, T = a.family, "baseline_x", tau(len(ids))
    L = layers_of(specs, fam)
    rb, wb, hb, xb = counts(state, ids, base)
    print("# %s first-round screen: AllenAI + rule refusals, tune %d questions, against %s (no write); tau = %d questions\n" % (fam, len(ids), base, T))
    print("baseline x: right / wrong / uninformative / abstained = %d / %d / %d / %d; answered %d; mean words %.1f\n" % (
        rb, wb, hb, xb, rb + wb, sum(words[(i, base)] for i in ids) / len(ids)))
    arms = ["x.w.p1", "x.w.m1", "x.md.p1", "x.rand.p1"] if fam == "E" else ["x.w.p1", "x.w.m1", "x.md.p1", "x.md.m1", "x.rand.p1", "x.rand.m1"]
    S = {}
    for l in L:
        for arm in arms:
            c = cond(fam, l, arm)
            v, n = per_question(state, words, ids, base, c)
            S[(l, arm)] = {"cnt": counts(state, ids, c), "dR": sum(v["dR"]), "dAns": sum(v["dAns"]), "dX": sum(v["dX"]),
                           "keep_dR": sum(v["keep"]), "G": sum(v["same"]), "n": n,
                           "words": sum(words[(i, c)] for i in ids) / len(ids)}
    if fam == "E":
        print("| layer | x.w.p1 R/W/H/X | dR | dAns | dX | G_same (n same) | keep: n, dR (W>R / R>W) | out / in | mean words | rand.p1: dR, G_same, dAns | md.p1: dR, G_same | w.m1: dR, G_same | max-true margin change |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for l in L:
            s, r, m, neg = S[(l, "x.w.p1")], S[(l, "x.rand.p1")], S[(l, "x.md.p1")], S[(l, "x.w.m1")]
            mg = mc_margin(a.run, fam, l, "x.w.p1")
            print("| %d | %d / %d / %d / %d | %+d | %+d | %+d | %+d (%d) | %d, %+d (%d / %d) | %d / %d | %.1f | %+d, %+d, %+d | %+d, %+d | %+d, %+d | %s |" % (
                (l,) + s["cnt"] + (s["dR"], s["dAns"], s["dX"], s["G"], s["n"]["same"], s["n"]["keep"], s["keep_dR"], s["n"]["up"], s["n"]["down"],
                                   s["n"]["out"], s["n"]["in"], s["words"], r["dR"], r["G"], r["dAns"], m["dR"], m["G"], neg["dR"], neg["G"],
                                   "—" if mg is None else "%+.4f" % mg)))
        g_main = max(S[(l, "x.w.p1")]["G"] for l in L); g_rand = max(S[(l, "x.rand.p1")]["G"] for l in L)
        print("\n## shortlist (plan 28, section 3)\n")
        print("gate: largest G_same over the %d layers — learned direction %+d, random direction %+d" % (len(L), g_main, g_rand))
        if g_main <= g_rand:
            print("=> NO GAIN beyond the random direction's best layer: nothing is sent to Pro.")
            return
        ok = lambda l: abs(S[(l, "x.w.p1")]["dAns"]) <= T  # noqa: E731
        by_g = sorted(L, key=lambda l: (-S[(l, "x.w.p1")]["G"], l))
        short = [(l, "slot %d: G_same %+d among layers with |dAns| <= %d" % (k + 1, S[(l, "x.w.p1")]["G"], T)) for k, l in enumerate([l for l in by_g if ok(l)][:3])]
        if len(short) < 3:
            for l in [l for l in by_g if not ok(l)][:3 - len(short)]:
                short.append((l, "slot %d: G_same %+d (retention condition NOT met: dAns %+d)" % (len(short) + 1, S[(l, "x.w.p1")]["G"], S[(l, "x.w.p1")]["dAns"])))
        chosen = {l for l, _ in short}
        cand = [(mc_margin(a.run, fam, l, "x.w.p1"), l) for l in L if l not in chosen and ok(l)]
        cand = sorted([(-m, l) for m, l in cand if m is not None])
        if cand:
            short.append((cand[0][1], "slot 4: largest max-true margin change (%+.4f) among the other layers with |dAns| <= %d (judge-free)" % (-cand[0][0], T)))
            chosen.add(cand[0][1])
        else:
            print("slot 4: no eligible layer (reference readout missing or no other layer with |dAns| <= %d)" % T)
        rest = sorted([l for l in L if l not in chosen], key=lambda l: (-S[(l, "x.w.p1")]["dR"], l))
        if rest:
            short.append((rest[0], "slot 5 (contrast): largest raw dR (%+d) among the layers not chosen" % S[(rest[0], "x.w.p1")]["dR"]))
        for l, why in short:
            s = S[(l, "x.w.p1")]
            print("- layer %d — %s; dR %+d, dAns %+d, G_same %+d (n %d), out / in %d / %d" % (l, why, s["dR"], s["dAns"], s["G"], s["n"]["same"], s["n"]["out"], s["n"]["in"]))
        print("\nshortlist layers:", " ".join(str(l) for l, _ in short))
    else:
        base_ans = rb + wb
        print("| layer | x.w.p1 R/W/H/X | A+ = -dAns | dX | keep: n, dR (W>R / R>W) | toH | x.w.m1 R/W/H/X | A- = +dAns | dX | keep: n, dR (W>R / R>W) | toH | T = min | rand dAns: p1, m1 | md dAns: p1, m1 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        Tm, ret = {}, {}
        for l in L:
            p, m = S[(l, "x.w.p1")], S[(l, "x.w.m1")]
            ap, am = -p["dAns"], m["dAns"]
            Tm[l] = (min(ap, am), ap, am)
            for arm, s in (("p1", p), ("m1", m)):
                ret[(l, arm)] = abs(s["keep_dR"]) <= T and 2 * s["n"]["keep"] >= base_ans and s["n"]["toH"] <= T
            print("| %d | %d / %d / %d / %d | %+d | %+d | %d, %+d (%d / %d) | %d | %d / %d / %d / %d | %+d | %+d | %d, %+d (%d / %d) | %d | %+d | %+d, %+d | %+d, %+d |" % (
                (l,) + p["cnt"] + (ap, p["dX"], p["n"]["keep"], p["keep_dR"], p["n"]["up"], p["n"]["down"], p["n"]["toH"]) + m["cnt"] + (
                    am, m["dX"], m["n"]["keep"], m["keep_dR"], m["n"]["up"], m["n"]["down"], m["n"]["toH"], Tm[l][0],
                    S[(l, "x.rand.p1")]["dAns"], S[(l, "x.rand.m1")]["dAns"], S[(l, "x.md.p1")]["dAns"], S[(l, "x.md.m1")]["dAns"])))
        r_max = max(abs(S[(l, arm)]["dAns"]) for l in L for arm in ("x.rand.p1", "x.rand.m1"))
        m_max = max(max(Tm[l][1], Tm[l][2]) for l in L)
        print("\n## shortlist (plan 28, section 3)\n")
        print("gate: largest answered-count change over the %d layers — learned direction %+d (intended side), random direction %d (either sign)" % (len(L), m_max, r_max))
        if m_max <= r_max:
            print("=> NO CHANGE beyond the random direction's largest: nothing is sent to Pro.")
            return
        order = sorted(L, key=lambda l: (-Tm[l][0], l))
        two = [l for l in order if Tm[l][0] > T]
        short = []
        if two:
            good = [l for l in two if ret[(l, "p1")] and ret[(l, "m1")]]
            short = [(l, "slot %d: two-way, T = %+d, retention conditions met on both arms" % (k + 1, Tm[l][0])) for k, l in enumerate(good[:2])]
            chosen = {l for l, _ in short}
            rest = [l for l in two if l not in chosen]
            if rest:
                short.append((rest[0], "slot 3 (contrast): next largest two-way T = %+d%s" % (
                    Tm[rest[0]][0], "" if (ret[(rest[0], "p1")] and ret[(rest[0], "m1")]) else " (retention conditions NOT met)")))
        else:
            one = sorted([(-Tm[l][k], l, name) for l in L for k, name in ((1, "p1"), (2, "m1")) if Tm[l][k] > T and ret[(l, name)]])
            seen = []
            for neg, l, name in one:
                if l not in [x[0] for x in seen]:
                    seen.append((l, "ONE-WAY slot %d: %s only, change %+d, retention conditions met" % (len(seen) + 1, "toward caution" if name == "p1" else "toward confident", -neg)))
                if len(seen) == 3:
                    break
            short = seen
            print("no layer with T > %d: one-way shortlist" % T)
        if not short:
            print("=> no layer passes the screen.")
            return
        for l, why in short:
            print("- layer %d — %s; A+ %+d, A- %+d" % (l, why, Tm[l][1], Tm[l][2]))
        print("\nshortlist layers:", " ".join(str(l) for l, _ in short))


# ----------------------------------------------------------------------------- the two cells of one family, one judge
def cells(a):
    gen, specs, ids, state, rule, words = load(a.work, a.judge)
    fam, base, T = a.family, "baseline_x", tau(len(ids))
    rb, wb, hb, xb = counts(state, ids, base)
    print("# %s cells under judge %s (%d questions; tau = %d; paired bootstrap %d resamples, seed %d; against %s)\n" % (
        fam, a.judge, len(ids), T, BOOT, SEED, base))
    print("baseline x: right / wrong / hedge-or-uninformative / abstained = %d / %d / %d / %d; rule abstentions %d; mean words %.1f\n" % (
        rb, wb, hb, xb, sum(rule[(i, base)] for i in ids), sum(words[(i, base)] for i in ids) / len(ids)))
    all_arms = ["x.w.p1", "x.w.m1", "x.md.p1", "x.md.m1", "x.rand.p1", "x.rand.m1"]
    for l in a.layers:
        print("## layer %d\n" % l)
        print("| arm | R / W / H / X | rule abst. | words | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR [95%], W>R / R>W | same: n, G_same [95%] | out / in / toH | X>R / X>W / R>X / W>X |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        K = {}
        for arm in all_arms:
            c = cond(fam, l, arm)
            if not have(state, ids, c):
                continue
            v, n = per_question(state, words, ids, base, c)
            sR, loR, hiR = ci(v["dR"]); sA, loA, hiA = ci(v["dAns"]); sX, loX, hiX = ci(v["dX"])
            sK, loK, hiK = ci(v["keep"]); sG, loG, hiG = ci(v["same"])
            K[arm] = {"v": v, "n": n, "dR": sum(v["dR"]), "dAns": sum(v["dAns"]), "dX": sum(v["dX"]), "keep_dR": sum(v["keep"]), "G": sum(v["same"]),
                      "loR": loR, "loA": loA, "hiA": hiA, "loX": loX, "hiX": hiX, "loK": loK, "hiK": hiK, "loG": loG}
            print("| %s | %s | %d | %.1f | %s | %s | %s | %d, %s, %d / %d | %d, %s | %d / %d / %d | %d / %d / %d / %d |" % (
                arm, fmt_counts(state, ids, c), sum(rule[(i, c)] for i in ids), sum(words[(i, c)] for i in ids) / len(ids), sR, sA, sX,
                n["keep"], sK, n["up"], n["down"], n["same"], sG, n["out"], n["in"], n["toH"],
                n["x_gain_right"], n["x_gain_wrong"], n["x_ans_lost_right"], n["x_ans_lost_wrong"]))
        print()
        rands = [r for r in ("x.rand.p1", "x.rand.m1") if r in K]
        if fam == "E" and "x.w.p1" in K and rands:
            k = K["x.w.p1"]
            worst = max(rands, key=lambda r: K[r]["dR"])
            vs, lo, hi = ci(minus(k["v"]["dR"], K[worst]["v"]["dR"]))
            e1, e2, e3 = (k["loR"] > 0 and k["dR"] > T), k["loG"] > 0, lo > 0
            ret = abs(k["dAns"]) <= T
            strong = k["n"]["out"] <= T and k["n"]["in"] <= T
            print("- E effect: (e1) dR %+d above tau and interval above 0: %s; (e2) G_same %+d interval above 0: %s; (e3) right count vs the less favourable random arm (%s%s) %s above 0: %s  =>  %s" % (
                k["dR"], e1, k["G"], e2, worst, "" if len(rands) == 2 else "; only one sign generated", vs, e3, "PASS" if (e1 and e2 and e3) else "not met"))
            print("- E retention: dAns %+d within +-%d: %s  =>  %s%s" % (k["dAns"], T, ret, "PASS" if ret else "not met",
                  (" (question by question as well: out %d, in %d)" % (k["n"]["out"], k["n"]["in"]) if strong else
                   " — rate only: %d answered questions stopped being answered and %d started; do not say the commitment is untouched" % (k["n"]["out"], k["n"]["in"])) if ret else ""))
            print("KEY family=E judge=%s layer=%d G_same=%+d dR=%+d dAns=%+d eligible=%s effect=%s\n" % (a.judge, l, k["G"], k["dR"], k["dAns"], ret, e1 and e2 and e3))
        if fam == "C" and rands and ("x.w.p1" in K or "x.w.m1" in K):
            res = {}
            base_ans = rb + wb
            for arm, sign, name in (("x.w.p1", -1, "toward caution"), ("x.w.m1", +1, "toward confident")):
                if arm not in K:
                    continue
                k = K[arm]
                worst = (min if sign < 0 else max)(rands, key=lambda r: K[r]["dAns"])
                vs, lo, hi = ci(minus(k["v"]["dAns"], K[worst]["v"]["dAns"]))
                if sign < 0:
                    eff = k["hiA"] < 0 and k["loX"] > 0 and -k["dAns"] > T and hi < 0
                else:
                    eff = k["loA"] > 0 and k["hiX"] < 0 and k["dAns"] > T and lo > 0
                ret = (2 * k["n"]["keep"] >= base_ans) and abs(k["keep_dR"]) <= T and k["loK"] <= 0 <= k["hiK"] and k["n"]["toH"] <= T
                strong = (k["n"]["up"] + k["n"]["down"]) <= T
                res[arm] = (eff, ret, strong)
                print("- C %s (%s): dAns %+d, dX %+d, beyond tau and both intervals on the intended side, and vs the less favourable random arm (%s) %s on the intended side: %s | retention: keep n %d (need >= %d), keep dR %+d within +-%d and interval contains 0, toH %d <= %d: %s%s" % (
                    name, arm, k["dAns"], k["dX"], worst, vs, eff, k["n"]["keep"], (base_ans + 1) // 2, k["keep_dR"], T, k["n"]["toH"], T, ret,
                    "" if strong else " (flips W>R %d + R>W %d exceed tau: content changed question by question)" % (k["n"]["up"], k["n"]["down"])))
            ok = [arm for arm in res if res[arm][0]]
            eff_txt = "TWO-WAY PASS" if len(ok) == 2 else ("ONE-WAY (%s)" % ok[0] if ok else "not met")
            ret_txt = ("PASS" if all(res[arm][1] for arm in ok) else "not met") if ok else "n/a"
            print("- C effect: %s; C retention on the arm(s) that count: %s" % (eff_txt, ret_txt))
            ap = -K["x.w.p1"]["dAns"] if "x.w.p1" in K else 0
            am = K["x.w.m1"]["dAns"] if "x.w.m1" in K else 0
            print("KEY family=C judge=%s layer=%d T=%+d Aplus=%+d Aminus=%+d effect=%s retention=%s\n" % (a.judge, l, min(ap, am), ap, am, eff_txt.replace(" ", "_"), ret_txt))


# ----------------------------------------------------------------------------- cross-level arms
def cross(a):
    gen, specs, ids, state, rule, words = load(a.work, a.judge)
    fam = a.family
    print("# %s per-question interchange / isolate / fixed-shift arms on other prompts, judge %s (each arm against its own base prompt, no write)\n" % (fam, a.judge))
    for l in a.layers:
        print("## layer %d\n" % l)
        print("| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dAns [95%] | dX [95%] | keep: n, dR, W>R / R>W | same: n, G_same | out / in / toH |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name, s in sorted(specs.items()):
            if s.get("family") != fam or s.get("layer") != l or s["arm"].startswith("x."):
                continue
            b = s["base_condition"]
            v, n = per_question(state, words, ids, b, name)
            print("| %s | %s | %s | %s | %s | %s | %d, %+d, %d / %d | %d, %+d | %d / %d / %d |" % (
                s["arm"], fmt_counts(state, ids, b), fmt_counts(state, ids, name), ci(v["dR"])[0], ci(v["dAns"])[0], ci(v["dX"])[0],
                n["keep"], sum(v["keep"]), n["up"], n["down"], n["same"], sum(v["same"]), n["out"], n["in"], n["toH"]))
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
