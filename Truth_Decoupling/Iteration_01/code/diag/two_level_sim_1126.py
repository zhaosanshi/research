#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md).
# Not reviewed by GPT. Read-only: existing answers and DeepSeek Pro verdicts of the TruthfulQA TEST split (main-table
# settings, write rule "prompt"); no card, no judge call, nothing is chosen here. Readings are after-the-fact.
# usage (container, from the project root): python3 -B Iteration_01/diag/two_level_sim_1126.py
# read-only exploration (no card, no judging): simulate "two-level E" on the TruthfulQA test split with already judged
# answers. Rule: N1 as it is; but on a question answered both with no write and under E at the main strength with a
# DIFFERENT text (E "reacted"), take the answer of single E at a higher strength instead (blank there = blank).
import collections, re, sys, os, json, random
sys.path.insert(0, "Iteration_01/diag")
import iti_table as A
MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
fr = MP.fr
G = {"R": "R", "W": "W", "X": "N", "H": "N"}
VAL = {"R": 1, "W": -1, "N": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
SPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def num(tok):
    return float(tok.replace("-", "."))


def ci(d, seed=20261126, n=2000):
    rng = random.Random(seed); m = len(d); out = []
    for _ in range(n):
        out.append(sum(d[rng.randrange(m)] for _ in range(m)))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


for model in ORDER:
    display, layer, s = A.KEY[model]
    data, err = MP.prompt_split(display, layer, SPLIT, s)
    if SPLIT == 1:
        src = [x for x in ML.sources(display)[2] if x[0] == layer]
        cx = [q for q in data[3] if q in set(src[0][3].qs)]
        if cx != data[3]:
            data, err = MP.prompt_split(display, layer, 1, s, qs=cx)
    tune, _ = MP.prompt_split(display, layer, 0, s)
    pick = ML.n1_pick(tune[0], tune[1], s)
    sp, tx, singles, ids = data
    cE = sp.cE(s)
    b = {q: G[sp.st[(q, "baseline_x")]] for q in ids}
    e = {q: G[sp.st[(q, cE)]] for q in ids}
    n1 = dict(zip(ids, [G[v] for v in S.system(sp, tx, pick)[0]]))
    t0 = {q: norm(tx.d[(q, "baseline_x")]) for q in ids}
    t1 = {q: norm(tx.d[(q, cE)]) for q in ids}
    react = [q for q in ids if b[q] != "N" and e[q] != "N" and t0[q] != t1[q]]
    net = lambda d: sum(VAL[d[q]] for q in ids)
    _, run, _ = MP.usual(display, SPLIT)
    base = os.path.join(run, "eval_extra")
    split = ("tune", "test")[SPLIT]
    pat = re.compile(r"^E_L%d_x-w-p([0-9-]+)$" % layer)
    found = []
    for dname in sorted(os.listdir(base)):
        if not (dname.startswith("judge_%s_" % split) and dname.endswith("_" + split)):
            continue
        w = os.path.join(base, dname)
        if not MP.has(w):
            continue
        name = dname[len("judge_%s_" % split):-len("_" + split)]
        try:
            conds = json.load(open(os.path.join(w, "judge_input.json")))["conditions"]
        except Exception:
            continue
        for c in conds:
            m = pat.match(c)
            if m and num(m.group(1)) > num(s):
                found.append((num(m.group(1)), name, c, w))
    print("## %s %s (%d questions): E layer %d main strength %s | no write %+d, E %+d, N1 %+d | E reacted (answered both, text changed): %d = %s" % (
        model, split, len(ids), layer, s, net(b), net(e), net(n1), len(react),
        ", ".join("%s%s %d" % (x, y, sum(1 for q in react if b[q] == x and e[q] == y)) for x in "RW" for y in "RW")))
    cache = {}
    for dose, name, c, w in sorted(found):
        if w not in cache:
            cache[w] = ps.load(w, "pro")[0]
        st = cache[w]
        if any(st.get((q, c), "?") == "?" for q in ids):
            print("   dose %-5s %-16s incomplete verdicts" % (dose, name)); continue
        e2 = {q: G[st[(q, c)]] for q in ids}
        rule = "prompt" if name.startswith("posprompt") else "old"
        final = dict(n1)
        for q in react:
            final[q] = e2[q]
        d = [VAL[final[q]] - VAL[n1[q]] for q in ids]
        lo, hi = ci(d)
        parts = []
        for x in "RW":
            for y in "RW":
                qs = [q for q in react if b[q] == x and e[q] == y]
                c3 = collections.Counter(e2[q] for q in qs)
                parts.append("%s%s->R %d W %d N %d" % (x, y, c3["R"], c3["W"], c3["N"]))
        print("   second level %-5s (%-14s rule %-6s): single E at that strength on ALL %+4d | two-level %+4d (answers %d) | two-level minus N1 %+4d [%+d, %+d] | %s" % (
            dose, name, rule, net(e2), net(final), sum(1 for q in ids if final[q] != "N"), sum(d), lo, hi, "; ".join(parts)))
