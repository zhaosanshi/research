#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md).
# Not reviewed by GPT. Read-only: existing answers and DeepSeek Pro verdicts of the TruthfulQA TEST split (main-table
# settings, write rule "prompt"); no card, no judge call, nothing is chosen here. Readings are after-the-fact.
# usage (container, from the project root): python3 -B Iteration_01/diag/ww_look_1126.py
# read-only exploration (no card, no judging): the answers that are wrong with no write and still wrong under single E
# (TruthfulQA test split, main-table settings, write rule "prompt"): what the other, already judged conditions give on them
import collections, re, sys, os
sys.path.insert(0, "Iteration_01/diag")
import iti_table as A
MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
fr = MP.fr
G = {"R": "R", "W": "W", "X": "N", "H": "N"}
VAL = {"R": 1, "W": -1, "N": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
ONLY = [a for a in sys.argv[1:] if a in ORDER] or ORDER


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def dist(vs):
    c = collections.Counter(vs)
    return "R %3d W %3d N %3d" % (c["R"], c["W"], c["N"])


TOT = collections.defaultdict(collections.Counter)
for model in ONLY:
    display, layer, s = A.KEY[model]
    test, err = MP.prompt_split(display, layer, 1, s)
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    cx = [q for q in test[3] if q in set(src[0][3].qs)]
    if cx != test[3]:
        test, err = MP.prompt_split(display, layer, 1, s, qs=cx)
    tune, _ = MP.prompt_split(display, layer, 0, s)
    pick = ML.n1_pick(tune[0], tune[1], s)
    sp, tx, singles, ids = test
    cE = sp.cE(s)
    cX = {k: sp.cEC(s, k) for k in ps.LAD}
    b = {q: G[sp.st[(q, "baseline_x")]] for q in ids}
    e = {q: G[sp.st[(q, cE)]] for q in ids}
    x = {k: {q: G[sp.st[(q, cX[k])]] for q in ids} for k in ps.LAD}
    n1 = dict(zip(ids, [G[v] for v in S.system(sp, tx, pick)[0]]))
    t0 = {q: norm(tx.d[(q, "baseline_x")]) for q in ids}
    t1 = {q: norm(tx.d[(q, cE)]) for q in ids}
    tX = {k: {q: norm(tx.d[(q, cX[k])]) for q in ids} for k in ps.LAD}
    net = lambda d: sum(VAL[d[q]] for q in ids)
    print("## %s (%d questions; E layer %d strength %s; N1 %s)" % (model, len(ids), layer, s, pick))
    print("  A. net on all questions: no write %+d | E %+d | E + C(answering) on ALL questions at 0.25 / 0.5 / 1 / 2: %s | N1 %+d" % (
        net(b), net(e), " / ".join("%+d (answers %d)" % (net(x[k]), sum(1 for q in ids if x[k][q] != "N")) for k in ps.LAD), net(n1)))
    cls = collections.defaultdict(list)
    for q in ids:
        key = b[q] + e[q]
        if key in ("RR", "WW"):
            key += "=" if t0[q] == t1[q] else "~"
        cls[key].append(q)
    print("  B. classes (no write, E; = same text, ~ different text): " + ", ".join("%s %d" % (k, len(cls[k])) for k in sorted(cls)))
    print("  C. E + C(answering) at each dose on the answered classes: verdicts | text equals the no-write answer / the E answer / neither (among answered)")
    for key in ("WW=", "WW~", "RW", "WR", "RR=", "RR~"):
        qs = cls[key]
        for k in ps.LAD:
            v = [x[k][q] for q in qs]
            ans = [q for q in qs if x[k][q] != "N"]
            a0 = sum(1 for q in ans if tX[k][q] == t0[q]); a1 = sum(1 for q in ans if tX[k][q] == t1[q] and tX[k][q] != t0[q])
            print("     %-3s n %3d dose %-4s: %s | = no-write %3d, = E %3d, neither %3d" % (key, len(qs), k, dist(v), a0, a1, len(ans) - a0 - a1))
            TOT[("C", key, k)].update(v)
    # other judged conditions
    _, run, _ = MP.usual(display, 1)
    other = {}
    def grab(name):
        w = MP.work(run, "test", name)
        if not MP.has(w):
            return
        st, bl = ps.load(w, "pro"); t = fr.raw_texts(w)
        for (q, c), v in st.items():
            if v == "?":
                continue
            other.setdefault((name, c), ({}, {}))
            other[(name, c)][0][q] = G[v]; other[(name, c)][1][q] = norm(t.get((q, c)))
    for name in ("pospromptdl%d" % layer, "promptsweep", "pfam", "n1s", "n1s3", "fillctl", "h22c4"):
        grab(name)
    want = []
    seen = set()
    for (name, c) in sorted(other):
        if c in seen or c == "baseline_x" or c.startswith("X_") or c.startswith("C_"):
            continue
        seen.add(c)
        d, _ = other[(name, c)]
        if all(q in d for q in ids):
            want.append((name, c))
    print("  D. other judged conditions on the classes (verdicts): " )
    for (name, c) in want:
        d, _ = other[(name, c)]
        line = "     %-12s %-28s all: net %+4d |" % (name, c, sum(VAL[d[q]] for q in ids))
        for key in ("WW", "RR", "RW", "WR"):
            qs = [q for q in ids if b[q] + e[q] == key]
            line += " %s(%d): %s |" % (key, len(qs), dist([d[q] for q in qs]))
            TOT[("D", c.replace("L%d" % layer, "L").replace("-p%s" % s, "-pS"), key)].update([d[q] for q in qs])
        print(line)
    # oracle among WW
    ww = [q for q in ids if b[q] + e[q] == "WW"]
    pools = {"E + C doses": [x[k] for k in ps.LAD], "other conditions": [other[w_][0] for w_ in want]}
    for label, ds in pools.items():
        anyr = sum(1 for q in ww if any(d[q] == "R" for d in ds)); anyn = sum(1 for q in ww if any(d[q] == "N" for d in ds))
        print("  E. WW (%d): right under at least one of the %s: %d; blank under at least one: %d" % (len(ww), label, anyr, anyn))
        TOT[("E", label)].update({"anyR": anyr, "anyN": anyn, "n": len(ww)})
    # agreement signals among questions answered under E
    ans = [q for q in ids if e[q] in "RW"]
    print("  F. questions answered under E: %d (right %d, wrong %d). Share right when the E answer text equals the answer of another condition:" % (
        len(ans), sum(1 for q in ans if e[q] == "R"), sum(1 for q in ans if e[q] == "W")))
    sigs = [("no write", b, t0)] + [("E + C %s" % k, x[k], tX[k]) for k in ps.LAD] + [("%s" % c, other[(nm, c)][0], other[(nm, c)][1]) for nm, c in want]
    for label, d, t in sigs:
        agree = [q for q in ans if d[q] != "N" and t[q] == t1[q]]
        differ = [q for q in ans if d[q] != "N" and t[q] != t1[q]]
        blank = [q for q in ans if d[q] == "N"]
        f = lambda qs: "%3d (right %5.1f%%)" % (len(qs), 100.0 * sum(1 for q in qs if e[q] == "R") / len(qs)) if qs else "  0"
        print("     %-30s same text %s | different text %s | other is blank %s" % (label, f(agree), f(differ), f(blank)))
        for nm, qs in (("agree", agree), ("differ", differ), ("blank", blank)):
            TOT[("F", label.replace("L%d" % layer, "L").replace("-p%s" % s, "-pS"), nm)].update([e[q] for q in qs])
print("## FIVE MODELS (sums, reference only)")
for k in sorted(TOT):
    print("  ", k, dict(TOT[k]))
