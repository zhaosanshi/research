#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md).
# Not reviewed by GPT. Read-only: existing answers and DeepSeek Pro verdicts of the TruthfulQA TEST split (main-table
# settings, write rule "prompt"); no card, no judge call, nothing is chosen here. Readings are after-the-fact.
# usage (container, from the project root): python3 -B Iteration_01/diag/e_transitions_1126.py
# one-off, read-only: what single E changes on the TruthfulQA test split (main-table settings, write rule "prompt")
import collections
import json
import sys

sys.path.insert(0, "Iteration_01/diag")
import iti_table as A

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
V = {"R": 1, "W": -1, "X": 0, "H": 0}
NAME = {1: "right", -1: "wrong", 0: "none"}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
tot = collections.Counter()
ftot = collections.Counter()
for model in ORDER:
    display, layer, s = A.KEY[model]
    test, err = MP.prompt_split(display, layer, 1, s)
    src = [x for x in ML.sources(display)[2] if x[0] == layer]
    cx = [q for q in test[3] if q in set(src[0][3].qs)]
    if cx != test[3]:
        test, err = MP.prompt_split(display, layer, 1, s, qs=cx)
    sp, tx, _, ids = test
    bs = S.system(sp, tx, ("base",))[0]
    es = S.system(sp, tx, ("E", s))[0]
    b = [V[x] for x in bs]
    e = [V[x] for x in es]
    c = collections.Counter(zip(b, e))
    tot.update(c)
    print("%s (%d questions): no write right %d wrong %d none %d (net %+d) | E right %d wrong %d none %d (net %+d)" % (
        model, len(ids), b.count(1), b.count(-1), b.count(0), sum(b), e.count(1), e.count(-1), e.count(0), sum(e)))
    print("    no write -> E: " + ", ".join("%s->%s %d" % (NAME[x], NAME[y], c[(x, y)]) for x in (1, -1, 0) for y in (1, -1, 0)))
    # must-answer arms in the forced work dir
    _, run, _ = MP.usual(display, 1)
    w = MP.work(run, "test", "pospromptdl%d" % layer)
    conds = json.load(open(w + "/judge_input.json"))["conditions"]
    st, bl = ps.load(w, "pro")
    fe = "E_L%d_conf-w-p%s" % (layer, s)
    others = [x for x in conds if x != fe]
    print("    forced work dir conditions: %s" % conds)
    for fb in others:
        if "conf" not in fb or "E_L" in fb:
            continue
        if any(st.get((q, fb), "?") == "?" or st.get((q, fe), "?") == "?" for q in ids):
            print("    %s: incomplete verdicts" % fb); continue
        x = [V[st[(q, fb)]] for q in ids]
        y = [V[st[(q, fe)]] for q in ids]
        fc = collections.Counter(zip(x, y))
        ftot.update(fc)
        print("    must-answer %s: right %d wrong %d none %d (net %+d) | must-answer + E: right %d wrong %d none %d (net %+d)" % (
            fb, x.count(1), x.count(-1), x.count(0), sum(x), y.count(1), y.count(-1), y.count(0), sum(y)))
        print("    must-answer -> must-answer + E: " + ", ".join("%s->%s %d" % (NAME[a], NAME[b2], fc[(a, b2)]) for a in (1, -1, 0) for b2 in (1, -1, 0)))
print("FIVE MODELS, normal prompt, no write -> E: " + ", ".join("%s->%s %d" % (NAME[x], NAME[y], tot[(x, y)]) for x in (1, -1, 0) for y in (1, -1, 0)))
if ftot:
    print("FIVE MODELS, must-answer -> must-answer + E: " + ", ".join("%s->%s %d" % (NAME[x], NAME[y], ftot[(x, y)]) for x in (1, -1, 0) for y in (1, -1, 0)))
