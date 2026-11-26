#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md).
# Not reviewed by GPT. Read-only: existing answers and DeepSeek Pro verdicts of the TruthfulQA TEST split (main-table
# settings, write rule "prompt") and the switch probabilities of plan P (fit_B, chosen layer); no card, no judge call.
# usage (container, from the project root): python3 -B Iteration_01/diag/first_vs_second_1126.py
# read-only: TruthfulQA test split, main settings, write rule "prompt". Questions answered in the first pass (no write)
# and in the second pass (single E) with different texts: what handing in the FIRST answer instead would give, the
# ceiling of a perfect per-question choice, and how the switch of plan P (pool B, chosen layer; ON = probability > 0.5)
# splits these questions. Existing verdicts and existing switch probabilities only.
import collections, json, re, sys
sys.path.insert(0, "Iteration_01/diag")
import iti_table as A
MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
G = {"R": "R", "W": "W", "X": "N", "H": "N"}
VAL = {"R": 1, "W": -1, "N": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
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
    diff = [q for q in ids if b[q] != "N" and e[q] != "N" and norm(tx.d[(q, "baseline_x")]) != norm(tx.d[(q, cE)])]
    c = collections.Counter(b[q] + e[q] for q in diff)
    all_first = sum(VAL[b[q]] - VAL[e[q]] for q in diff)
    ceiling = sum(max(VAL[b[q]], VAL[e[q]]) - VAL[e[q]] for q in diff)
    fit = json.load(open("Iteration_01/newsets/gate_1126/%s/fit_B_gate_1126.json" % model))
    pr = fit["probs"]["chosen"]["tq_test"] if "chosen" in fit["probs"] else None
    if pr is None:
        print(model, "probs keys:", list(fit["probs"].keys())); continue
    off = {q: pr[q] <= 0.5 for q in diff}
    parts = []
    for k in ("WR", "RW", "RR", "WW"):
        qs = [q for q in diff if b[q] + e[q] == k]
        parts.append("%s %d (switch off %d)" % (k, len(qs), sum(off[q] for q in qs)))
    by_switch = sum(VAL[b[q]] - VAL[e[q]] for q in diff if off[q])
    print("%s: answered both with different texts %d: second right first wrong %d, first right second wrong %d, both right %d, both wrong %d | hand in the first answer on all of them: %+d | perfect choice per question: %+d | %s | first answer only where the switch is off: %+d" % (
        model, len(diff), c["WR"], c["RW"], c["RR"], c["WW"], all_first, ceiling, "; ".join(parts), by_switch))
