#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md).
# Not reviewed by GPT. Read-only: existing answers and DeepSeek Pro verdicts of the TruthfulQA TEST split (main-table
# settings, write rule "prompt"); no card, no judge call, nothing is chosen here. Readings are after-the-fact.
# usage (container, from the project root): python3 -B Iteration_01/diag/own_blank_c_1126.py
# read-only: on the questions blank with no write AND blank under single E ("own blanks"), TruthfulQA test split, main
# settings, write rule "prompt": what single C (toward answering, same ladder, capped at N1's cap for own blanks) gives,
# next to what N1's third pass (E + C) gives. Existing verdicts only.
import collections, sys
sys.path.insert(0, "Iteration_01/diag")
import iti_table as A
MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
G = {"R": "R", "W": "W", "X": "N", "H": "N"}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
for model in ORDER:
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
    b = {q: G[sp.st[(q, "baseline_x")]] for q in ids}
    e = {q: G[sp.st[(q, cE)]] for q in ids}
    n1 = dict(zip(ids, [G[v] for v in S.system(sp, tx, pick)[0]]))
    k2 = pick[3]
    lad = list(ps.LAD)
    upto = lad[:lad.index(k2) + 1]
    if any(k not in singles for k in upto):
        print("%s: single-C states missing for some dose of %s" % (model, upto)); continue
    idx = {q: i for i, q in enumerate(sp.qs)}
    def single_c(q):
        for k in upto:
            st, ans = singles[k]
            if G[st[idx[q]]] != "N":
                return G[st[idx[q]]]
        return "N"
    own = [q for q in ids if b[q] == "N" and e[q] == "N"]
    c1 = collections.Counter(single_c(q) for q in own)
    c2 = collections.Counter(n1[q] for q in own)
    print("%s: own blanks %d (cap for own blanks %s) | single C up to the cap: right %d wrong %d blank %d (net %+d) | single E: all blank (net 0) | N1 third pass (E + C): right %d wrong %d blank %d (net %+d)" % (
        model, len(own), k2, c1["R"], c1["W"], c1["N"], c1["R"] - c1["W"], c2["R"], c2["W"], c2["N"], c2["R"] - c2["W"]))
