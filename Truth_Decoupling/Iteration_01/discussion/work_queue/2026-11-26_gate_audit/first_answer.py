"""Compare scored answers with actual submissions when reading the answer-switch table."""

import re
import sys

sys.path.insert(0, "Iteration_01/diag")
import iti_table as A

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
G = {"R": "R", "W": "W", "X": "N", "H": "N"}
VAL = {"R": 1, "W": -1, "N": 0}
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


print(
    "model R/W-both issued-both only-R/W only-issued always-first-delta perfect-choice-delta"
)
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
    diff = [
        q
        for q in ids
        if b[q] != "N"
        and e[q] != "N"
        and norm(tx.d[(q, "baseline_x")]) != norm(tx.d[(q, cE)])
    ]
    delivered = [
        q
        for q in ids
        if not sp.blank[q, "baseline_x"]
        and not sp.blank[q, cE]
        and norm(tx.d[q, "baseline_x"]) != norm(tx.d[q, cE])
    ]
    print(
        model,
        len(diff),
        len(delivered),
        len(set(diff) - set(delivered)),
        len(set(delivered) - set(diff)),
        sum(VAL[b[q]] - VAL[e[q]] for q in delivered),
        sum(max(VAL[b[q]], VAL[e[q]]) - VAL[e[q]] for q in delivered),
    )
