#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only tables for the E44 first round (plan 27, sections 3 and 4.3). CPU only, no model, nothing is written.

  e44_round1_tables.py screen  RUN_DIR FAMILY WORK_DIR            all layers, AllenAI + rule refusals, shortlist by the frozen rule
  e44_round1_tables.py cells   RUN_DIR FAMILY WORK_DIR --judge J --layers L [L ...]
                                                                  the two cells of this family for one judge (bootstrap intervals)
  e44_round1_tables.py cross   RUN_DIR FAMILY WORK_DIR --judge J --layers L [L ...]
                                                                  per-question interchange / isolate arms against their own base prompt

States per judge (plan 27, 3.1):   R right, W wrong, H hedge or uninformative, X abstained.
  allenai: rule abstention -> X whatever the judge said; otherwise A -> R, B -> W, N -> H.
  pro:     A -> R, B -> W, H -> H, C -> X (the rule abstention count is printed next to it).
"Answered" means R or W. All comparisons are paired on the same questions and judged by the same judge.
"""
import argparse, collections, json, random, sys
from pathlib import Path

TOL = 6            # +-6 questions of 192 = the +-3 percentage point tolerance declared in plan 21
BAND = 3           # "no conclusion within +-3 questions"
BOOT, SEED = 10000, 20261111


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
        raise SystemExit("unscored cells: %d (first %s)" % (len(bad), bad[:3]))
    return gen, specs, ids, state, rule


def cond(family, layer, arm):
    return "%s_L%d_%s" % (family, layer, arm.replace(".", "-"))


def counts(state, ids, c):
    k = collections.Counter(state[(i, c)] for i in ids)
    return k["R"], k["W"], k["H"], k["X"]


def fmt_counts(state, ids, c):
    return "%d / %d / %d / %d" % counts(state, ids, c)


def vec(state, ids, c, what):
    """Per-question 0/1 indicator: 'R' right, 'X' abstained, 'ans' answered."""
    if what == "ans":
        return [1 if state[(i, c)] in "RW" else 0 for i in ids]
    return [1 if state[(i, c)] == what else 0 for i in ids]


def common_right(state, ids, b, c):
    """On the questions answered under both b and c: (n, right under b, right under c, W->R, R->W)."""
    both = [i for i in ids if state[(i, b)] in "RW" and state[(i, c)] in "RW"]
    rb = sum(state[(i, b)] == "R" for i in both); rc = sum(state[(i, c)] == "R" for i in both)
    up = sum(state[(i, b)] == "W" and state[(i, c)] == "R" for i in both)
    down = sum(state[(i, b)] == "R" and state[(i, c)] == "W" for i in both)
    return len(both), rb, rc, up, down


def moves(state, ids, b, c):
    ans = lambda s: s in "RW"  # noqa: E731
    out = collections.Counter()
    for i in ids:
        x, y = state[(i, b)], state[(i, c)]
        if ans(x) and y == "X": out["ans>X"] += 1
        if x == "X" and ans(y): out["X>ans"] += 1
        if ans(x) and y == "H": out["ans>H"] += 1
        if x == "H" and ans(y): out["H>ans"] += 1
        if x == "X" and y == "H": out["X>H"] += 1
        if x == "H" and y == "X": out["H>X"] += 1
        if x == "W" and y == "R": out["W>R"] += 1
        if x == "R" and y == "W": out["R>W"] += 1
        if x == "X" and y == "R": out["X>R"] += 1
        if x == "X" and y == "W": out["X>W"] += 1
        if x == "R" and y == "X": out["R>X"] += 1
        if x == "W" and y == "X": out["W>X"] += 1
    return out


def boot(diffs, n_boot=BOOT, seed=SEED):
    """Percentile 95% interval of the SUM of per-question differences, paired bootstrap over questions."""
    n = len(diffs)
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        d = np.asarray(diffs, dtype=float)
        sums = d[rng.integers(0, n, size=(n_boot, n))].sum(axis=1)
        lo, hi = np.percentile(sums, [2.5, 97.5])
        return float(lo), float(hi)
    except ImportError:
        rng = random.Random(seed)
        sums = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
        return sums[int(0.025 * n_boot)], sums[int(0.975 * n_boot) - 1]


def diff(a, b):
    return [x - y for x, y in zip(a, b)]


def ci(d):
    lo, hi = boot(d)
    return "%+d [%+.0f, %+.0f]" % (sum(d), lo, hi), lo, hi


def common_diff(state, ids, b, c):
    """Per-question contribution to (right under c - right under b) restricted to questions answered under both."""
    out = []
    for i in ids:
        if state[(i, b)] in "RW" and state[(i, c)] in "RW":
            out.append((state[(i, c)] == "R") - (state[(i, b)] == "R"))
        else:
            out.append(0)
    return out


def mc_margin(run, family, layer, arm):
    p = Path(run) / "mc" / ("%s_L%d_%s_tune.json" % (family, layer, arm.replace(".", "-")))
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["summary"].get("margin_change_mean")


def layers_of(specs, family):
    return sorted({s["layer"] for s in specs.values() if s.get("family") == family})


# ----------------------------------------------------------------------------- screen
def screen(a):
    gen, specs, ids, state, rule = load(a.work, "allenai")
    fam, base = a.family, "baseline_x"
    L = layers_of(specs, fam)
    rb, wb, hb, xb = counts(state, ids, base)
    print("# %s first-round screen: AllenAI + rule refusals, tune %d questions, all comparisons against %s (no write)\n" % (fam, len(ids), base))
    print("baseline x: right / wrong / uninformative / abstained = %d / %d / %d / %d\n" % (rb, wb, hb, xb))
    arms = ["x.w.p1", "x.w.m1", "x.md.p1", "x.rand.p1"] if fam == "E" else ["x.w.p1", "x.w.m1", "x.md.p1", "x.md.m1", "x.rand.p1", "x.rand.m1"]
    stat = {}
    for l in L:
        for arm in arms:
            c = cond(fam, l, arm)
            r, w, h, x = counts(state, ids, c)
            n, r0, r1, up, down = common_right(state, ids, base, c)
            stat[(l, arm)] = {"R": r, "W": w, "H": h, "X": x, "dR": r - rb, "dX": x - xb, "dAns": (r + w) - (rb + wb),
                              "common_n": n, "common_dR": r1 - r0, "up": up, "down": down}
    if fam == "E":
        print("| layer | x.w.p1 R/W/H/X | dR | dX | d answered | answered-both n, dR (W>R / R>W) | x.w.m1 dR, dX | x.md.p1 dR, dX | x.rand.p1 dR, dX | max-true margin change (w.p1) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for l in L:
            s = stat[(l, "x.w.p1")]
            m = mc_margin(a.run, fam, l, "x.w.p1")
            print("| %d | %d / %d / %d / %d | %+d | %+d | %+d | %d, %+d (%d / %d) | %+d, %+d | %+d, %+d | %+d, %+d | %s |" % (
                l, s["R"], s["W"], s["H"], s["X"], s["dR"], s["dX"], s["dAns"], s["common_n"], s["common_dR"], s["up"], s["down"],
                stat[(l, "x.w.m1")]["dR"], stat[(l, "x.w.m1")]["dX"], stat[(l, "x.md.p1")]["dR"], stat[(l, "x.md.p1")]["dX"],
                stat[(l, "x.rand.p1")]["dR"], stat[(l, "x.rand.p1")]["dX"], "—" if m is None else "%+.4f" % m))
        order = sorted(L, key=lambda l: (-stat[(l, "x.w.p1")]["dR"], l))
        best = stat[(order[0], "x.w.p1")]["dR"]
        print("\n## shortlist (plan 27, 4.3)\n")
        if best <= BAND:
            print("largest dR over all layers is %+d (<= +%d): nothing is sent to Pro; the main hypothesis shows no gain under AllenAI." % (best, BAND))
            return
        short = [(order[0], "slot 1: largest dR"), (order[1], "slot 2: second largest dR")]
        chosen = {order[0], order[1]}
        keep = [l for l in order if l not in chosen and abs(stat[(l, "x.w.p1")]["dX"]) <= TOL]
        if keep:
            short.append((keep[0], "slot 3: largest dR among layers with |dX| <= %d" % TOL))
        else:
            rest = [l for l in order if l not in chosen]
            short.append((rest[0], "slot 3: third largest dR (no layer with |dX| <= %d; retention condition NOT met)" % TOL))
        chosen.add(short[-1][0])
        cand = [(mc_margin(a.run, fam, l, "x.w.p1"), l) for l in L if l not in chosen and abs(stat[(l, "x.w.p1")]["dX"]) <= TOL]
        cand = [(m, l) for m, l in cand if m is not None]
        if cand:
            m, l = sorted(cand, key=lambda t: (-t[0], t[1]))[0]
            short.append((l, "slot 4: largest max-true margin change (%+.4f) among layers with |dX| <= %d" % (m, TOL)))
        else:
            print("slot 4: no eligible layer (reference readout missing or no layer with |dX| <= %d)" % TOL)
        for l, why in short:
            s = stat[(l, "x.w.p1")]
            print("- layer %d — %s; dR %+d, dX %+d, d answered %+d" % (l, why, s["dR"], s["dX"], s["dAns"]))
        print("\nshortlist layers:", " ".join(str(l) for l, _ in short))
    else:
        print("| layer | x.w.p1 R/W/H/X | A+ (dX toward caution) | x.w.m1 R/W/H/X | A- (-dX toward confident) | T = min | answered-both dR: p1, m1 | md: A+, A- | rand: A+, A- |")
        print("|---|---|---|---|---|---|---|---|---|")
        T = {}
        for l in L:
            p, m = stat[(l, "x.w.p1")], stat[(l, "x.w.m1")]
            ap, am = p["dX"], -m["dX"]
            T[l] = (min(ap, am), ap, am)
            print("| %d | %d / %d / %d / %d | %+d | %d / %d / %d / %d | %+d | %+d | %+d (n %d), %+d (n %d) | %+d, %+d | %+d, %+d |" % (
                l, p["R"], p["W"], p["H"], p["X"], ap, m["R"], m["W"], m["H"], m["X"], am, T[l][0],
                p["common_dR"], p["common_n"], m["common_dR"], m["common_n"],
                stat[(l, "x.md.p1")]["dX"], -stat[(l, "x.md.m1")]["dX"], stat[(l, "x.rand.p1")]["dX"], -stat[(l, "x.rand.m1")]["dX"]))
        print("\n## shortlist (plan 27, 4.3)\n")
        order = sorted(L, key=lambda l: (-T[l][0], l))
        two_way = [l for l in order if T[l][0] > BAND]
        short = []
        if two_way:
            short = [(l, "slot %d: T = %+d" % (k + 1, T[l][0])) for k, l in enumerate(two_way[:2])]
            chosen = {l for l, _ in short}
            keep = [l for l in order if l not in chosen and T[l][0] > BAND
                    and abs(stat[(l, "x.w.p1")]["common_dR"]) <= TOL and abs(stat[(l, "x.w.m1")]["common_dR"]) <= TOL]
            if keep:
                short.append((keep[0], "slot 3: largest T (%+d) among the remaining layers whose answered-both dR is within +-%d for both arms" % (T[keep[0]][0], TOL)))
            else:
                print("slot 3: no remaining layer with T > %d and answered-both dR within +-%d for both arms" % (BAND, TOL))
        else:
            order1 = sorted(L, key=lambda l: (-max(T[l][1], T[l][2]), l))
            short = [(l, "ONE-WAY slot %d: max(A+, A-) = %+d (A+ %+d, A- %+d)" % (k + 1, max(T[l][1], T[l][2]), T[l][1], T[l][2])) for k, l in enumerate(order1[:3])]
            print("no layer with T > %d: one-way shortlist" % BAND)
        for l, why in short:
            print("- layer %d — %s; A+ %+d, A- %+d" % (l, why, T[l][1], T[l][2]))
        print("\nshortlist layers:", " ".join(str(l) for l, _ in short))


# ----------------------------------------------------------------------------- the two cells of one family, one judge
def cells(a):
    gen, specs, ids, state, rule = load(a.work, a.judge)
    fam, base = a.family, "baseline_x"
    print("# %s cells under judge %s (tune %d questions; paired bootstrap %d resamples, seed %d; all against %s)\n" % (
        fam, a.judge, len(ids), BOOT, SEED, base))
    print("baseline x: right / wrong / hedge-or-uninformative / abstained = %s; rule abstentions %d\n" % (
        fmt_counts(state, ids, base), sum(rule[(i, base)] for i in ids)))
    for l in a.layers:
        print("## layer %d\n" % l)
        arms = ["x.w.p1", "x.w.m1", "x.md.p1", "x.rand.p1"] if fam == "E" else ["x.w.p1", "x.w.m1", "x.md.p1", "x.md.m1", "x.rand.p1", "x.rand.m1"]
        print("| arm | R / W / H / X | rule abstentions | dR [95%] | d answered [95%] | dX [95%] | answered-both: n, dR [95%], W>R / R>W | ans>X / X>ans / ans>H / H>ans |")
        print("|---|---|---|---|---|---|---|---|")
        keep = {}
        for arm in arms:
            c = cond(fam, l, arm)
            if (ids[0], c) not in state:
                print("| %s | not judged |" % arm); continue
            dR, loR, hiR = ci(diff(vec(state, ids, c, "R"), vec(state, ids, base, "R")))
            dA, loA, hiA = ci(diff(vec(state, ids, c, "ans"), vec(state, ids, base, "ans")))
            dX, loX, hiX = ci(diff(vec(state, ids, c, "X"), vec(state, ids, base, "X")))
            n, r0, r1, up, down = common_right(state, ids, base, c)
            dC, loC, hiC = ci(common_diff(state, ids, base, c))
            mv = moves(state, ids, base, c)
            keep[arm] = {"loR": loR, "hiR": hiR, "loC": loC, "hiC": hiC, "loX": loX, "hiX": hiX,
                         "dAns": sum(diff(vec(state, ids, c, "ans"), vec(state, ids, base, "ans"))), "dC": r1 - r0, "n": n}
            print("| %s | %s | %d | %s | %s | %s | %d, %s, %d / %d | %d / %d / %d / %d |" % (
                arm, fmt_counts(state, ids, c), sum(rule[(i, c)] for i in ids), dR, dA, dX, n, dC, up, down,
                mv["ans>X"], mv["X>ans"], mv["ans>H"], mv["H>ans"]))
        print()
        if fam == "E" and "x.w.p1" in keep and "x.rand.p1" in keep:
            c, r = cond(fam, l, "x.w.p1"), cond(fam, l, "x.rand.p1")
            vs, lo, hi = ci(diff(vec(state, ids, c, "R"), vec(state, ids, r, "R")))
            k = keep["x.w.p1"]
            e1, e2, e3 = k["loR"] > 0, k["loC"] > 0, lo > 0
            pres = abs(k["dAns"]) <= TOL
            print("- E effect: (1) dR interval above 0: %s; (2) answered-both dR interval above 0: %s; (3) right count vs random arm %s, above 0: %s  =>  %s" % (
                e1, e2, vs, e3, "PASS" if (e1 and e2 and e3) else "not met"))
            print("- E retention: d answered = %+d, within +-%d: %s  =>  %s\n" % (k["dAns"], TOL, pres, "PASS" if pres else "not met"))
        if fam == "C" and all(x in keep for x in ("x.w.p1", "x.w.m1", "x.rand.p1", "x.rand.m1")):
            res = {}
            for arm, rand, sign in (("x.w.p1", "x.rand.p1", +1), ("x.w.m1", "x.rand.m1", -1)):
                c, r = cond(fam, l, arm), cond(fam, l, rand)
                vs, lo, hi = ci(diff(vec(state, ids, c, "X"), vec(state, ids, r, "X")))
                k = keep[arm]
                eff = (k["loX"] > 0 and lo > 0) if sign > 0 else (k["hiX"] < 0 and hi < 0)
                pres = abs(k["dC"]) <= TOL and k["loC"] <= 0 <= k["hiC"]
                res[arm] = (eff, pres)
                print("- C %s: abstentions vs no write interval on the intended side: %s; vs random arm %s on the intended side: %s; answered-both dR %+d (n %d), within +-%d and interval contains 0: %s" % (
                    "toward caution (p1)" if sign > 0 else "toward confident (m1)", (k["loX"] > 0) if sign > 0 else (k["hiX"] < 0), vs,
                    (lo > 0) if sign > 0 else (hi < 0), k["dC"], k["n"], TOL, pres))
            both = res["x.w.p1"][0] and res["x.w.m1"][0]
            one = res["x.w.p1"][0] or res["x.w.m1"][0]
            ok_arms = [arm for arm in res if res[arm][0]]
            print("- C effect: %s; C retention on the arm(s) that count: %s\n" % (
                "TWO-WAY PASS" if both else ("ONE-WAY (%s)" % ok_arms[0] if one else "not met"),
                ("PASS" if all(res[arm][1] for arm in ok_arms) else "not met") if ok_arms else "n/a"))


# ----------------------------------------------------------------------------- cross-level arms
def cross(a):
    gen, specs, ids, state, rule = load(a.work, a.judge)
    fam = a.family
    print("# %s per-question interchange / isolate arms under judge %s (each arm against its own base prompt, no write)\n" % (fam, a.judge))
    for l in a.layers:
        print("## layer %d\n" % l)
        print("| arm | base R / W / H / X | arm R / W / H / X | dR [95%] | dX [95%] | answered-both: n, dR, W>R / R>W | ans>X / X>ans |")
        print("|---|---|---|---|---|---|---|")
        for name, s in sorted(specs.items()):
            if s.get("family") != fam or s.get("layer") != l or s["arm"].startswith("x."):
                continue
            b = s["base_condition"]
            dR, _, _ = ci(diff(vec(state, ids, name, "R"), vec(state, ids, b, "R")))
            dX, _, _ = ci(diff(vec(state, ids, name, "X"), vec(state, ids, b, "X")))
            n, r0, r1, up, down = common_right(state, ids, b, name)
            mv = moves(state, ids, b, name)
            print("| %s | %s | %s | %s | %s | %d, %+d, %d / %d | %d / %d |" % (
                s["arm"], fmt_counts(state, ids, b), fmt_counts(state, ids, name), dR, dX, n, r1 - r0, up, down, mv["ans>X"], mv["X>ans"]))
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
