#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md).
# Not reviewed by GPT. Read-only: existing answers and DeepSeek Pro verdicts of the TruthfulQA TEST split (main-table
# settings, write rule "prompt"); no card, no judge call, nothing is chosen here. Readings are after-the-fact.
# usage (container, from the project root): python3 -B Iteration_01/diag/ww_dose_1126.py
# read-only exploration (no card, no judging): how the still-wrong answers (wrong with no write and under single E at the
# main strength, write rule "prompt") respond to single E at other strengths that were already judged on the test split
import collections, re, sys, os, json
sys.path.insert(0, "Iteration_01/diag")
import iti_table as A
MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
fr = MP.fr
G = {"R": "R", "W": "W", "X": "N", "H": "N"}
VAL = {"R": 1, "W": -1, "N": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def num(tok):
    return float(tok.replace("-", "."))


for model in ORDER:
    display, layer, s = A.KEY[model]
    test, err = MP.prompt_split(display, layer, 1, s)
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    cx = [q for q in test[3] if q in set(src[0][3].qs)]
    if cx != test[3]:
        test, err = MP.prompt_split(display, layer, 1, s, qs=cx)
    sp, tx, singles, ids = test
    cE = sp.cE(s)
    b = {q: G[sp.st[(q, "baseline_x")]] for q in ids}
    e = {q: G[sp.st[(q, cE)]] for q in ids}
    t0 = {q: norm(tx.d[(q, "baseline_x")]) for q in ids}
    t1 = {q: norm(tx.d[(q, cE)]) for q in ids}
    cls = collections.defaultdict(list)
    for q in ids:
        key = b[q] + e[q]
        if key in ("RR", "WW"):
            key += "=" if t0[q] == t1[q] else "~"
        cls[key].append(q)
    _, run, _ = MP.usual(display, 1)
    base = os.path.join(run, "eval_extra")
    found = []
    pat = re.compile(r"^E_L%d_x-w-p([0-9-]+)$" % layer)
    for d in sorted(os.listdir(base)):
        if not (d.startswith("judge_test_") and d.endswith("_test")):
            continue
        w = os.path.join(base, d)
        if not MP.has(w):
            continue
        name = d[len("judge_test_"):-len("_test")]
        try:
            conds = json.load(open(os.path.join(w, "judge_input.json")))["conditions"]
        except Exception:
            continue
        for c in conds:
            m = pat.match(c)
            if m:
                found.append((num(m.group(1)), name, c, w))
    print("## %s: E layer %d, main strength %s (write rule prompt); classes: %s" % (model, layer, s, ", ".join("%s %d" % (k, len(cls[k])) for k in ("WW=", "WW~", "RR=", "RR~", "RW", "WR"))))
    cache = {}
    for dose, name, c, w in sorted(found):
        if w not in cache:
            cache[w] = (ps.load(w, "pro")[0], fr.raw_texts(w))
        st, t = cache[w]
        if any(st.get((q, c), "?") == "?" for q in ids):
            print("   dose %-5s %-22s incomplete verdicts" % (dose, name)); continue
        d = {q: G[st[(q, c)]] for q in ids}
        rule = "prompt" if name.startswith("posprompt") else "old"
        line = "   dose %-5s %-18s rule %-6s all: net %+4d, answers %3d |" % (dose, name, rule, sum(VAL[d[q]] for q in ids), sum(1 for q in ids if d[q] != "N"))
        for key in ("WW=", "WW~", "RR=", "RR~"):
            c3 = collections.Counter(d[q] for q in cls[key])
            line += " %s: R %3d W %3d N %3d |" % (key, c3["R"], c3["W"], c3["N"])
        print(line)
