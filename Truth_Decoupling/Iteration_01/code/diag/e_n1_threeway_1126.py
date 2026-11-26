#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md).
# Not reviewed by GPT. Read-only: existing answers and DeepSeek Pro verdicts of the TruthfulQA TEST split (main-table
# settings, write rule "prompt"); no card, no judge call, nothing is chosen here. Readings are after-the-fact.
# usage (container, from the project root): python3 -B Iteration_01/diag/e_n1_threeway_1126.py
# one-off, read-only: no write -> single E -> N1 on the TruthfulQA test split (main-table settings, write rule "prompt"):
# where E's gain and N1's gain come from, per model, and what the questions answered both times with different texts look like
import collections
import re
import sys

sys.path.insert(0, "Iteration_01/diag")
import iti_table as A

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
G = {"R": "R", "W": "W", "X": "N", "H": "N"}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


TOT = collections.Counter()
for model in ORDER:
    display, layer, s = A.KEY[model]
    test, err = MP.prompt_split(display, layer, 1, s)
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    cx = [q for q in test[3] if q in set(src[0][3].qs)]
    if cx != test[3]:
        test, err = MP.prompt_split(display, layer, 1, s, qs=cx)
    tune, err2 = MP.prompt_split(display, layer, 0, s)
    pick = ML.n1_pick(tune[0], tune[1], s)
    sp, tx, _, ids = test
    b = S.system(sp, tx, ("base",)); e = S.system(sp, tx, ("E", s)); n = S.system(sp, tx, pick)
    bs, es, ns = [G[x] for x in b[0]], [G[x] for x in e[0]], [G[x] for x in n[0]]
    val = {"R": 1, "W": -1, "N": 0}
    print("%s (%d questions) N1 setting %s: no write net %+d | E net %+d | N1 net %+d" % (
        model, len(ids), pick, sum(val[x] for x in bs), sum(val[x] for x in es), sum(val[x] for x in ns)))
    c3 = collections.Counter(zip(bs, es, ns))
    # E's blanks and own blanks: what N1 does
    for name, sel in (("E's blanks that were RIGHT with no write", ("R", "N")), ("E's blanks that were WRONG with no write", ("W", "N")),
                      ("own blanks (blank with no write and under E)", ("N", "N"))):
        k = {z: c3[(sel[0], sel[1], z)] for z in "RWN"}
        notes = collections.Counter(r["note"] for r, x, y in zip(n[2], bs, es) if (x, y) == sel)
        print("    %-46s n %3d -> N1: right %3d, wrong %3d, still blank %3d (net %+d) | %s" % (
            name, sum(k.values()), k["R"], k["W"], k["N"], k["R"] - k["W"], dict(notes)))
        TOT.update({(name, z): k[z] for z in "RWN"})
    # wrong answers left after N1, by origin
    left = collections.Counter((x, y) for x, y, z in zip(bs, es, ns) if z == "W")
    print("    wrong answers after N1: %d = %s" % (sum(left.values()), ", ".join("%s->%s->W %d" % (x, y, v) for (x, y), v in sorted(left.items()))))
    TOT.update({("left", k): v for k, v in left.items()})
    # answered with no write and under E: same or different text
    same = collections.Counter(); diff = collections.Counter()
    for rb, re_, x, y in zip(b[2], e[2], bs, es):
        if x in "RW" and y in "RW":
            (same if norm(rb["text"]) == norm(re_["text"]) else diff)[(x, y)] += 1
    f = lambda c: ", ".join("%s->%s %d" % (x, y, c[(x, y)]) for x in "RW" for y in "RW")
    print("    answered both times, SAME text: %d (%s) | DIFFERENT text: %d (%s)" % (sum(same.values()), f(same), sum(diff.values()), f(diff)))
    TOT.update({("same", k): v for k, v in same.items()}); TOT.update({("diff", k): v for k, v in diff.items()})
print("FIVE MODELS")
for name in ("E's blanks that were RIGHT with no write", "E's blanks that were WRONG with no write", "own blanks (blank with no write and under E)"):
    print("    %-46s -> N1: right %d, wrong %d, still blank %d" % (name, TOT[(name, "R")], TOT[(name, "W")], TOT[(name, "N")]))
print("    wrong answers after N1 by origin: " + ", ".join("%s->%s->W %d" % (k[1][0], k[1][1], v) for k, v in sorted(TOT.items()) if k[0] == "left"))
for kind in ("same", "diff"):
    print("    answered both times, %s text: " % kind + ", ".join("%s->%s %d" % (x, y, TOT[(kind, (x, y))]) for x in "RW" for y in "RW"))
