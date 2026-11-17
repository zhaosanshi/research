# Iteration_01 analysis (Claude Fable 5.2, 2026-11-17), read-only, no model, no judge call.
# 赵磊 2026-11-17 19 时: the stability gate was the wrong method, not a proof that it cannot be done; look again, starting from
# "split by where the blank comes from" (E's own blanks get a tighter cap). Two judge-free readings of a released answer:
#   dose    k*, the smallest C strength (0.25 / 0.5 / 1 / 2) at which the question is answered with E written: how hard the
#           model had to be pushed.
#   whose   is the released answer the model's ORIGINAL answer or a NEW one? Original = the no-write answer when the model
#           answered with no write (E's own blanks); the C-alone answer at the same k* when it did not (own blanks; none if C
#           alone is still blank there). "Same" = equal after lower-casing and dropping punctuation.
# Part 1  right / wrong of the released answers by origin x dose (tune and test, per model and pooled over the five models
#         with blanks).
# Part 2  the same by origin x whose answer.
# Part 3  schemes, settings chosen on tune by right - wrong (ties to the earlier, weaker setting), reported on test:
#         two caps (two_caps.py); N1 = two caps, but one of E's own blanks is released only with a NEW answer;
#         N2 = N1, and an own blank is released only with a NEW answer too.
# Part 4  the dose curve of the method (one cap for both kinds) next to the must-answer prompt: how many questions are
#         answered and the score, at each cap; prompt B = every blank asked again with the must-answer prompt.
# Predictions written before the run (Fable 5.2, 19:2x; the test table of the C-alone comparison at cap 2 was already seen):
#   R1  pooled over the five models, the share right among released own blanks falls with the dose on tune as on test
#       (0.25 >= 0.5 >= 1 >= 2), and for E's own blanks no dose is above 60% right
#   R2  on tune, in each of the three models with 523 test questions, E's own blanks released with the ORIGINAL answer are
#       right less often than those released with a NEW answer
#   R3  N1 is not below two caps on the test split in any of the five models, and the sum over the five is at least 10 higher
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/release_origin.py
import collections
import json
import re
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import two_caps as tc

LAD = ps.LAD
CAPS = [None] + LAD
FIVE = ("Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B", "Gemma-3-12B", "Gemma-3-27B")


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower())).strip()


def texts(work):
    out = {}
    for l in open(work + "/judge2/cells.jsonl", encoding="utf-8"):
        r = json.loads(l)
        out[(r["pair_id"], r["condition"])] = norm(r["prediction"])
    return out


def released(sp, tx, s, cap=None):
    """For every question blank with E at s: (pair, origin, k*, state, whose) of the answer the ladder releases (k* <= cap)."""
    e = sp.cE(s)
    lad = LAD if cap is None else tc.lad(cap)
    out = []
    for p in sp.qs:
        if not sp.blank[(p, e)]:
            continue
        own = sp.blank[(p, "baseline_x")]
        for k in lad:
            c = sp.cEC(s, k)
            if not sp.blank[(p, c)]:
                if own:
                    orig = None if sp.blank[(p, sp.cC(k))] else tx[(p, sp.cC(k))]
                else:
                    orig = tx[(p, "baseline_x")]
                whose = "original" if (orig is not None and orig == tx[(p, c)]) else "new"
                out.append((p, "own" if own else "E's", k, sp.st[(p, c)], whose))
                break
    return out


def states(sp, tx, cfg):
    if cfg[0] not in ("N1", "N2"):
        return tc.states(sp, cfg)
    kind, s, k1, k2 = cfg
    e = sp.cE(s)
    rel = {}
    for cap, origin in ((k1, "E's"), (k2, "own")):
        if cap is None:
            continue
        for p, o, k, st, whose in released(sp, tx, s, cap):
            if o != origin:
                continue
            if whose == "new" or (kind == "N1" and origin == "own"):
                rel[p] = st
    return [rel.get(p, sp.st[(p, e)]) for p in sp.qs]


def name_of(cfg):
    if cfg[0] in ("N1", "N2"):
        return "%s: E %s, own blanks up to %s, E's blanks up to %s" % (cfg[0], cfg[1], ps.NAME[cfg[3]], ps.NAME[cfg[2]])
    return tc.name_of(cfg)


def rw(rows):
    r = sum(x == "R" for x in rows)
    w = sum(x == "W" for x in rows)
    return r, w, len(rows)


def fmt(rows):
    r, w, n = rw(rows)
    return "%3d / %3d (n %3d%s)" % (r, w, n, (", %2.0f%% right" % (100.0 * r / n)) if n >= 10 else "")


def main():
    pooled = collections.defaultdict(list)
    data = {}
    for name, m in cf.MODELS.items():
        tune, test = ps.Split(m, 0), ps.Split(m, 1)
        ttx, etx = texts(m["ec"][0]), texts(m["ec"][1])
        fam_m = [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in LAD]
        pm = max(fam_m, key=lambda x: (ps.score(tune.states(x), 1), -fam_m.index(x)))
        s = pm[1]
        data[name] = (m, tune, test, ttx, etx, pm, s)
        print("\n## %s (E strength of the method: %s)" % (name, s))
        for label, sp, tx in (("tune", tune, ttx), ("test", test, etx)):
            rel = released(sp, tx, s)
            for origin in ("own", "E's"):
                line1 = "  %s, %-4s blanks, by dose:  " % (label, origin)
                for k in LAD:
                    rows = [st for p, o, kk, st, wh in rel if o == origin and kk == k]
                    line1 += "%s %s | " % (ps.NAME[k], fmt(rows))
                    if name in FIVE:
                        pooled[(label, origin, "dose", k)] += rows
                print(line1)
                line2 = "  %s, %-4s blanks, by whose: " % (label, origin)
                for wh in ("original", "new"):
                    rows = [st for p, o, kk, st, w_ in rel if o == origin and w_ == wh]
                    line2 += "%s %s | " % (wh, fmt(rows))
                    if name in FIVE:
                        pooled[(label, origin, "whose", wh)] += rows
                    for k in LAD:
                        if name in FIVE:
                            pooled[(label, origin, "whose x dose", (wh, k))] += [st for p, o, kk, st, w_ in rel if o == origin and w_ == wh and kk == k]
                print(line2)
    print("\n## pooled over the five models with blanks (right / wrong)")
    for label in ("tune", "test"):
        for origin in ("own", "E's"):
            print("  %s, %-4s blanks, by dose:  " % (label, origin) + " | ".join("%s %s" % (ps.NAME[k], fmt(pooled[(label, origin, "dose", k)])) for k in LAD))
            print("  %s, %-4s blanks, by whose: " % (label, origin) + " | ".join("%s %s" % (wh, fmt(pooled[(label, origin, "whose", wh)])) for wh in ("original", "new")))
            for wh in ("original", "new"):
                print("  %s, %-4s blanks, %-8s answers by dose: " % (label, origin, wh) + " | ".join("%s %s" % (ps.NAME[k], fmt(pooled[(label, origin, "whose x dose", (wh, k))])) for k in LAD))

    print("\n\n# Part 3: schemes (chosen on tune, reported on test; right - wrong)")
    tot = collections.Counter()
    for name, (m, tune, test, ttx, etx, pm, s) in data.items():
        fam = {
            "single E": [("E", "1"), ("E", "2")],
            "two caps": tc.FAM["two caps"],
            "N1": [("E", s_) for s_ in "12"] + [("N1", s_, k1, k2) for s_ in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)],
            "N2": [("E", s_) for s_ in "12"] + [("N2", s_, k1, k2) for s_ in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)],
        }
        row = {}
        print("\n## %s" % name)
        for f, c in fam.items():
            pick = max(c, key=lambda x: (ps.score(states(tune, ttx, x), 1), -c.index(x)))
            st = states(test, etx, pick)
            row[f] = st
            if name in FIVE:
                tot[f] += ps.score(st, 1)
            print("  %-9s chosen on tune: %-58s | tune %5.1f | test %s | score %4d" % (
                f, name_of(pick), ps.score(states(tune, ttx, pick), 1), cf.cnt(collections.Counter(st)), ps.score(st, 1)))
        for a in ("N1", "N2"):
            print("  %s minus single E: %-18s | minus two caps: %s" % (
                a, ps.ci([x - y for x, y in zip(cf.val(row[a]), cf.val(row["single E"]))]),
                ps.ci([x - y for x, y in zip(cf.val(row[a]), cf.val(row["two caps"]))])))
    print("\n  sum over the five models, test: " + " | ".join("%s %d" % (f, tot[f]) for f in ("single E", "two caps", "N1", "N2")))

    print("\n\n# Part 4: the dose curve of the method (one cap for both kinds) and the must-answer prompt, test split")
    for name, (m, tune, test, ttx, etx, pm, s) in data.items():
        line = "%-30s E %s |" % (name, s)
        for k in CAPS:
            st = test.states(("E", s)) if k is None else test.states(("EC", s, k))
            ans = sum(x in "RWH" for x in st)
            line += " cap %s: answered %d, score %d |" % (ps.NAME[k], ans, ps.score(st, 1))
        if test.dst is not None:
            b = test.states(("B", s))
            line += " prompt B: answered %d, score %d" % (sum(x in "RWH" for x in b), ps.score(b, 1))
        print(line)


if __name__ == "__main__":
    main()
