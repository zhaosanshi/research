# Iteration_01 analysis (Claude Fable 5.2, 2026-11-18), read-only, no model, no judge call.
# Ablations asked for in the novelty review of 2026-11-18 (discussion/novelty_review/2026-11-18_gpt_review.md, question 3):
# on the SAME candidates and the SAME E strength (the one the method chose on tune), what is each piece worth?
#   Part 1  provenance. unified cap (one cap for every blank) | two caps with the TRUE provenance (own blank / E's own blank) |
#           two caps with a SHUFFLED provenance (the labels permuted among the candidates, group sizes kept; 1000 shuffles,
#           seed 20261118; tune and test shuffled independently; the caps re-chosen on tune for every shuffle). If shuffled
#           labels do as well as the true ones, the gain of two caps is tuning freedom, not provenance.
#   Part 2  the original-answer veto alone: caps fixed at the two-caps choice, the veto switched on and off.
#   Part 3  is "C lets the model answer, E makes the answer right" more than the two arms answering different questions?
#           On the questions blank with no write, fixed E 2 and C k: the interaction v(E+C) - v(E) - v(C) + v(no write) per
#           question, and v(E+C) - v(C) split by which of the two arms answers.
#   Part 4  sensitivity of N1 to the text normalisation (the review notes that dropping punctuation also drops a minus sign):
#           a second normalisation keeps signed and decimal numbers.
# v = +1 right, -1 wrong, 0 otherwise (DeepSeek Pro). Caps are chosen on tune, results reported on test.
# Predictions written before the run (Fable 5.2, 10:1x):
#   A1  shuffled provenance: the mean test gain is within 3 points of the unified cap in every model; on Llama-3.1-8B and
#       Qwen3.5-9B fewer than 10% of the shuffles reach the gain of the true provenance
#   A2  the veto alone: the test difference is >= 0 in the five models and its interval is above 0 only on Qwen3.8-27B
#   A3  interaction (E 2, C 1): positive on Llama-3.1-8B, Qwen3.8-27B and the two Gemma-3, about 0 on Qwen3.5-9B; its interval
#       is above 0 on Llama-3.1-8B and Qwen3.8-27B
#   A4  the second normalisation changes the test score of N1 by at most 2 points in every model
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/provenance_ablation.py
import collections
import random
import re
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import two_caps as tc
import release_origin as ro

LAD = ps.LAD
CAPS = [None] + LAD
B = 1000
V = {"R": 1, "W": -1}


def v(state):
    return V.get(state, 0)


def within(k, cap):
    return cap is not None and LAD.index(k) <= LAD.index(cap)


def candidates(sp, tx, s):
    e = sp.cE(s)
    out = []
    for i, p in enumerate(sp.qs):
        if not sp.blank[(p, e)]:
            continue
        own = sp.blank[(p, "baseline_x")]
        kstar = next((k for k in LAD if not sp.blank[(p, sp.cEC(s, k))]), None)
        rel, orig = None, False
        if kstar is not None:
            rel = sp.st[(p, sp.cEC(s, kstar))]
            o = (None if sp.blank[(p, sp.cC(kstar))] else tx[(p, sp.cC(kstar))]) if own else tx[(p, "baseline_x")]
            orig = o is not None and o == tx[(p, sp.cEC(s, kstar))]
        out.append({"i": i, "own": own, "k": kstar, "e": v(sp.st[(p, e)]), "rel": None if rel is None else v(rel), "relstate": rel, "orig": orig})
    return out


def gains(cands, labels, k1, k2, veto=False):
    """Per candidate, score with the release minus score with E alone. labels[j] True = treated as an own blank (cap k2)."""
    out = []
    for c, own in zip(cands, labels):
        cap = k2 if own else k1
        take = c["k"] is not None and within(c["k"], cap) and not (veto and not own and c["orig"])
        out.append((c["rel"] - c["e"]) if take else 0)
    return out


TWO = [(None, None)] + [(k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
ONE = [(None, None)] + [(k, k) for k in LAD]


def choose(cands, labels, configs, veto=False):
    return max(configs, key=lambda c: (sum(gains(cands, labels, c[0], c[1], veto)), -configs.index(c)))


def full(sp, cands, g):
    out = [0] * len(sp.qs)
    for c, x in zip(cands, g):
        out[c["i"]] = x
    return out


def norm2(t):
    return " ".join(re.findall(r"-?\d+(?:\.\d+)?|[a-z]+", (t or "").lower()))


def main():
    rng = random.Random(20261118)
    rows4 = []
    print("# Part 1 and 2: provenance and the original-answer veto (gain over E alone at the same E strength; test split)")
    for name, m in cf.MODELS.items():
        tune, test = ps.Split(m, 0), ps.Split(m, 1)
        ttx, etx = ro.texts(m["ec"][0]), ro.texts(m["ec"][1])
        fam_m = [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in LAD]
        s = max(fam_m, key=lambda x: (ps.score(tune.states(x), 1), -fam_m.index(x)))[1]
        ct, ce = candidates(tune, ttx, s), candidates(test, etx, s)
        lt, le = [c["own"] for c in ct], [c["own"] for c in ce]
        one = choose(ct, lt, ONE)
        two = choose(ct, lt, TWO)
        g1 = gains(ce, le, one[0], one[1])
        g2 = gains(ce, le, two[0], two[1])
        sh = []
        for _ in range(B):
            st_, se_ = lt[:], le[:]
            rng.shuffle(st_)
            rng.shuffle(se_)
            c = choose(ct, st_, TWO)
            sh.append(sum(gains(ce, se_, c[0], c[1])))
        sh.sort()
        print("\n## %s (E %s; candidates: tune %d = %d own + %d E's, test %d = %d own + %d E's)" % (
            name, s, len(ct), sum(lt), len(lt) - sum(lt), len(ce), sum(le), len(le) - sum(le)))
        print("  unified cap %s: test gain %+d | two caps, true provenance (E's %s, own %s): test gain %+d | difference %s" % (
            ps.NAME[one[0]], sum(g1), ps.NAME[two[0]], ps.NAME[two[1]], sum(g2),
            ps.ci([a - b for a, b in zip(full(test, ce, g2), full(test, ce, g1))])))
        print("  two caps, shuffled provenance (%d shuffles): test gain mean %+.1f, middle 95%% [%+d, %+d]; shuffles reaching the true-provenance gain: %.1f%%" % (
            B, sum(sh) / B, sh[int(0.025 * B)], sh[int(0.975 * B) - 1], 100.0 * sum(x >= sum(g2) for x in sh) / B))
        gv = gains(ce, le, two[0], two[1], veto=True)
        vet = [c for c, own in zip(ce, le) if (not own) and c["k"] is not None and within(c["k"], two[0]) and c["orig"]]
        cnt = collections.Counter(c["relstate"] for c in vet)
        print("  veto alone, caps fixed (E's %s, own %s): test gain %+d against %+d without it; difference %s | vetoed answers: right %d, wrong %d, other %d" % (
            ps.NAME[two[0]], ps.NAME[two[1]], sum(gv), sum(g2), ps.ci([a - b for a, b in zip(full(test, ce, gv), full(test, ce, g2))]),
            cnt["R"], cnt["W"], len(vet) - cnt["R"] - cnt["W"]))
        tv = gains(ct, lt, two[0], two[1], veto=True)
        print("  veto alone, tune: gain %+d against %+d without it" % (sum(tv), sum(gains(ct, lt, two[0], two[1]))))
        rows4.append((name, m, tune, test))

    print("\n\n# Part 3: questions blank with no write, fixed E 2 and C k (test split): interaction and who answers")
    for name, m, tune, test in rows4:
        b0 = [p for p in test.qs if test.blank[(p, "baseline_x")]]
        for k in ("0-5", "1"):
            vb = [v(test.st[(p, "baseline_x")]) for p in b0]
            vE = [v(test.st[(p, test.cE("2"))]) for p in b0]
            vC = [v(test.st[(p, test.cC(k))]) for p in b0]
            vEC = [v(test.st[(p, test.cEC("2", k))]) for p in b0]
            inter = [a - b - c + d for a, b, c, d in zip(vEC, vE, vC, vb)]
            pat = collections.defaultdict(int)
            n = collections.Counter()
            for p, a, c in zip(b0, vEC, vC):
                key = ("E+C answers" if not test.blank[(p, test.cEC("2", k))] else "E+C blank") + ", " + ("C answers" if not test.blank[(p, test.cC(k))] else "C blank")
                pat[key] += a - c
                n[key] += 1
            print("  %-30s C %-4s n %3d | no write %+d, E %+d, C %+d, E+C %+d | interaction %s | E+C minus C by who answers: %s" % (
                name, ps.NAME[k], len(b0), sum(vb), sum(vE), sum(vC), sum(vEC), ps.ci(inter) if b0 else "-",
                "; ".join("%s (%d) %+d" % (key, n[key], pat[key]) for key in sorted(n))))

    print("\n\n# Part 4: N1 with the second normalisation (signed and decimal numbers kept)")
    fam = lambda: [("E", s_) for s_ in "12"] + [("N1", s_, k1, k2) for s_ in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
    for name, m, tune, test in rows4:
        res = []
        for label, fn in (("first", ro.norm), ("second", norm2)):
            keep = ro.norm
            ro.norm = fn
            ttx, etx = ro.texts(m["ec"][0]), ro.texts(m["ec"][1])
            ro.norm = keep
            f = fam()
            pick = max(f, key=lambda x: (ps.score(ro.states(tune, ttx, x), 1), -f.index(x)))
            st = ro.states(test, etx, pick)
            s_ = pick[1]
            orig = sum(1 for c in candidates(test, etx, s_) if c["orig"])
            res.append((label, ro.name_of(pick), ps.score(st, 1), orig))
        print("  %-30s " % name + " | ".join("%s normalisation: %s, test %d, released answers equal to the original %d" % r for r in res))


if __name__ == "__main__":
    main()
