# Iteration_01 analysis (Claude Fable 5.2, 2026-11-17), read-only, no model, no judge call.
# 赵磊 2026-11-17: "能不能用C先让弃答的答出来，然后再统一用E把错的压成对的？"
# Offline check on the judged grids (verdicts: DeepSeek Pro on both splits). Score = right - wrong only.
# The E and the C write sit in ONE forward pass, so "C first, then E" can only differ from the method in WHICH questions get C.
# Policies on the plain prompt (routing by the rule detector of blanks, as policy_select2; settings chosen on tune, reported on test):
#   single E   E alone at s
#   method     E at s; the questions E alone leaves blank get C raised 0.25 -> 0.5 -> 1 -> 2 until answered, up to a cap K
#   all        E at s and C toward answering at k on EVERY question (the joint arm as it is)
#   baseonly   as method, but C goes only to the questions that are blank with NO write; the blanks E itself creates stay blank
#   cfirst     a question blank with no write gets the smallest C (<= K) at which C ALONE answers it, and E at s is written on top
#              (the joint arm at that k; if it is blank there it stays blank); every other question gets E alone
# Predictions written before the run (Fable 5.2, 2026-11-17 13:2x):
#   P1  "all" does not beat the method on any model (difference <= 0, or its interval holds 0)
#   P2  "baseonly" / "cfirst" do not beat the method with an interval above 0 on any model; on Llama-3.1-8B they are lower
#   P3  the answers E silenced and C then releases are more often right than wrong at C 0.25 and not at the larger caps
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/c_first.py [MODEL ...]
import collections
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps

LAD = ps.LAD
NAME = ps.NAME
LAM = 1
G27 = "Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b"
MODELS = {n: m for n, m in ps.MODELS.items() if "(tune AllenAI)" not in n}
if "Gemma-3-27B" not in MODELS:
    MODELS["Gemma-3-27B"] = dict(ec=(G27 + "/eval_extra/judge_tune_ec31_tune", G27 + "/eval_extra/judge_test_t31_testdr100"),
                                 d=(G27 + "/judge_tune_proD_E_L26", G27 + "/judge_test_testD_E_L26dr100"),
                                 judge=("pro", "pro"), e=26, c=31, order="EC")


def ladder(cap):
    return LAD[:LAD.index(cap) + 1]


def states(sp, cfg):
    kind = cfg[0]
    if kind in ("base", "E", "C", "EC"):
        return sp.states(cfg)
    s = cfg[1]
    e = sp.cE(s)
    if kind == "ALL":
        return [sp.st[(p, sp.cEC(s, cfg[2]))] for p in sp.qs]
    out = []
    for p in sp.qs:
        state = sp.st[(p, e)]
        if sp.blank[(p, "baseline_x")]:
            if kind == "BO":
                if sp.blank[(p, e)]:
                    for k in ladder(cfg[2]):
                        if not sp.blank[(p, sp.cEC(s, k))]:
                            state = sp.st[(p, sp.cEC(s, k))]
                            break
            elif kind == "CF":
                for k in ladder(cfg[2]):
                    if not sp.blank[(p, sp.cC(k))]:
                        state = sp.st[(p, sp.cEC(s, k))]
                        break
            else:
                raise ValueError(cfg)
        out.append(state)
    return out


def describe(cfg):
    k = cfg[0]
    if k in ("base", "E", "C", "EC"):
        return ps.describe(cfg)
    return {"ALL": "E %s + C %s on every question", "BO": "E %s, C up to %s on no-write blanks only",
            "CF": "E %s on top of the C (<= %s) that answers a no-write blank"}[k] % (cfg[1], NAME[cfg[2]])


def val(states_):
    return [(1 if x == "R" else 0) - LAM * (1 if x == "W" else 0) for x in states_]


def cnt(c, keys="RWHX"):
    return " ".join("%s %d" % (k, c.get(k, 0)) for k in keys)


def release_split(sp, s, cap):
    """Questions blank under E alone at s, split by whether they were answered with no write; where the ladder leaves them."""
    e = sp.cE(s)
    out = {"i": collections.Counter(), "ii": collections.Counter()}
    cross = collections.Counter()
    for p in sp.qs:
        if not sp.blank[(p, e)]:
            continue
        g = "ii" if sp.blank[(p, "baseline_x")] else "i"
        final = "X"
        for k in ladder(cap):
            if not sp.blank[(p, sp.cEC(s, k))]:
                final = sp.st[(p, sp.cEC(s, k))]
                break
        out[g][final] += 1
        if g == "i":
            cross[(sp.st[(p, "baseline_x")], final)] += 1
    return out, cross


def report(name, m):
    tune, test = ps.Split(m, 0), ps.Split(m, 1)
    fam = {
        "single E": [("E", "1"), ("E", "2")],
        "method": [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in LAD],
        "all": [("E", s) for s in "12"] + [("ALL", s, k) for s in "12" for k in LAD],
        "baseonly": [("E", s) for s in "12"] + [("BO", s, k) for s in "12" for k in LAD],
        "cfirst": [("E", s) for s in "12"] + [("CF", s, k) for s in "12" for k in LAD],
    }
    print("\n## %s (tune: %s, %d questions; test: %s, %d questions; score = right - wrong)" % (
        name, m["judge"][0], len(tune.qs), m["judge"][1], len(test.qs)))
    for label, sp in (("tune", tune), ("test", test)):
        b = collections.Counter(sp.states(("base",)))
        nb = sum(sp.blank[(p, "baseline_x")] for p in sp.qs)
        print("  no write, %s: %s | rule blanks %d | score %d" % (label, cnt(b), nb, ps.score(sp.states(("base",)), LAM)))
    pick = {f: max(c, key=lambda x: (ps.score(states(tune, x), LAM), -c.index(x))) for f, c in fam.items()}
    row = {}
    for f in fam:
        st = states(test, pick[f])
        row[f] = st
        c = collections.Counter(st)
        print("  %-9s chosen on tune: %-52s | tune score %5.1f | test: %s | score %4d" % (
            f, describe(pick[f]), ps.score(states(tune, pick[f]), LAM), cnt(c), ps.score(st, LAM)))
    for a in ("all", "baseonly", "cfirst"):
        for b in ("method", "single E"):
            d = [x - y for x, y in zip(val(row[a]), val(row[b]))]
            print("  %s minus %s on test: %s" % (a, b, ps.ci(d)))
    # the same policies with the setting of the method forced (no re-selection), so that only the routing differs
    cfgm = pick["method"]
    if cfgm[0] == "EC":
        s, cap = cfgm[1], cfgm[2]
        for kind, a in (("BO", "baseonly"), ("CF", "cfirst"), ("ALL", "all")):
            st = states(test, (kind, s, cap))
            d = [x - y for x, y in zip(val(st), val(row["method"]))]
            print("  at the method's own setting (E %s, C %s): %s minus method on test: %s" % (s, NAME[cap], a, ps.ci(d)))
    else:
        s, cap = cfgm[1], None
        print("  the method chose E alone (no C) on tune")
    # where the method's released answers come from, test and tune, at each cap, for the E strength the method chose
    for label, sp in (("test", test), ("tune", tune)):
        for k in LAD:
            out, cross = release_split(sp, s, k)
            i, ii = out["i"], out["ii"]
            print("  %s, E %s, cap %-4s: E's own new blanks (answered with no write) %3d -> %s (net %+d) | blank with no write too %3d -> %s (net %+d)" % (
                label, s, NAME[k], sum(i.values()), cnt(i), i.get("R", 0) - i.get("W", 0), sum(ii.values()), cnt(ii), ii.get("R", 0) - ii.get("W", 0)))
        out, cross = release_split(sp, s, "2")
        print("  %s, E %s, cap 2, E's own new blanks by the no-write verdict: %s" % (label, s, " | ".join(
            "no-write %s -> %s" % (b, cnt({f: cross.get((b, f), 0) for f in "RWHX"})) for b in "RWH")))
    # questions blank with no write: what C alone and E + C make of them (test)
    b0 = [p for p in test.qs if test.blank[(p, "baseline_x")]]
    print("  test, the %d questions blank with no write:" % len(b0))
    for s_ in "12":
        print("    E %s alone: %s" % (s_, cnt(collections.Counter(test.st[(p, test.cE(s_))] for p in b0))))
    for k in LAD:
        ca = collections.Counter(test.st[(p, test.cC(k))] for p in b0)
        line = "    C %-4s alone: %s (net %+d)" % (NAME[k], cnt(ca), ca.get("R", 0) - ca.get("W", 0))
        for s_ in "12":
            j = collections.Counter(test.st[(p, test.cEC(s_, k))] for p in b0)
            line += " | with E %s: %s (net %+d)" % (s_, cnt(j), j.get("R", 0) - j.get("W", 0))
        print(line)
    # what C does to the questions E alone already answers (test): score change on those questions only
    for s_ in "12":
        ans = [p for p in test.qs if not test.blank[(p, test.cE(s_))]]
        line = "  test, the %d questions E %s alone answers; adding C changes their score by:" % (len(ans), s_)
        for k in LAD:
            a = val([test.st[(p, test.cEC(s_, k))] for p in ans])
            b = val([test.st[(p, test.cE(s_))] for p in ans])
            t = collections.Counter((test.st[(p, test.cE(s_))], test.st[(p, test.cEC(s_, k))]) for p in ans)
            line += " C %s %+d (R->W %d, W->R %d, R->X %d, W->X %d);" % (NAME[k], sum(a) - sum(b), t.get(("R", "W"), 0), t.get(("W", "R"), 0), t.get(("R", "X"), 0), t.get(("W", "X"), 0))
        print(line)
    # the joint arm on every question, test
    for s_ in "12":
        print("  test, every question, E %s: alone %d | " % (s_, ps.score(test.states(("E", s_)), LAM)) + " | ".join(
            "+ C %s %d" % (NAME[k], ps.score(states(test, ("ALL", s_, k)), LAM)) for k in LAD))


def main():
    for name in (sys.argv[1:] or MODELS):
        report(name, MODELS[name])


if __name__ == "__main__":
    main()
