#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-26; plan discussion/work_queue/2026-11-26_queue_o.md).
# Not reviewed by GPT. Read-only: existing answers and DeepSeek Pro verdicts of the TruthfulQA TEST split (main-table
# settings, write rule "prompt"); no card, no judge call, nothing is chosen here. Readings are after-the-fact.
# usage (container, from the project root): python3 -B Iteration_01/diag/reanswer_sim_1126.py
# read-only exploration (no card, no judging): TruthfulQA test split, existing verdicts and self-judge margins.
# "Re-answer instead of drop": where rule Z (self-judge, no write, margin < 0, negated openings exempt) judges single E's
# answer false, take the answer of single E at a higher strength instead of dropping it (blank there = blank).
import collections, json, os, random, re, sys
sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import selfjudge_stage1 as Z1
import iti_table as A
MP, ps = A.MP, A.ps
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
VAL = {"R": 1, "W": -1, "X": 0, "H": 0}


def num(tok):
    return float(tok.replace("-", "."))


def ci(d, seed=20261126, n=2000):
    rng = random.Random(seed); m = len(d); out = []
    for _ in range(n):
        out.append(sum(d[rng.randrange(m)] for _ in range(m)))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n) - 1]


for model in ORDER:
    data = Z1.load_all((model,))[model]
    ids, tq, (layer, s, n1cfg) = data["tq"]
    d = tq["baseline"]
    dropE = {q: bool(d[q]["eans"] and Z1.judged_false(d[q]["mE"], d[q]["negE"], 0.0, True)) for q in ids}
    dropR = {q: bool(d[q]["rel"] is not None and Z1.judged_false(d[q]["rel"][1], d[q]["rel"][2], 0.0, True)) for q in ids}
    n1 = {q: d[q]["n1"] for q in ids}
    e = {q: d[q]["e"] for q in ids}
    s2 = {q: 0 if dropE[q] else n1[q] for q in ids}
    s3 = {q: 0 if dropE[q] or dropR[q] else n1[q] for q in ids}
    flagged = [q for q in ids if dropE[q]]
    display = A.KEY[model][0]
    _, run, _ = MP.usual(display, 1)
    base = os.path.join(run, "eval_extra")
    pat = re.compile(r"^E_L%d_x-w-p([0-9-]+)$" % layer)
    found = []
    for dname in sorted(os.listdir(base)):
        if not (dname.startswith("judge_test_") and dname.endswith("_test")):
            continue
        w = os.path.join(base, dname)
        if not MP.has(w):
            continue
        name = dname[len("judge_test_"):-len("_test")]
        try:
            conds = json.load(open(os.path.join(w, "judge_input.json")))["conditions"]
        except Exception:
            continue
        for c in conds:
            m = pat.match(c)
            if m and num(m.group(1)) > num(str(s)):
                found.append((num(m.group(1)), name, c, w))
    net = lambda v: sum(v[q] for q in ids)
    print("## %s (%d questions; E layer %d strength %s): nets E %+d | N1 %+d | N1 + self-judge on E's answers (drop) %+d | + self-judge on released too %+d" % (
        model, len(ids), layer, s, net(e), net(n1), net(s2), net(s3)))
    print("   E answers judged false: %d (right %d, wrong %d); released answers judged false: %d" % (
        len(flagged), sum(1 for q in flagged if e[q] == 1), sum(1 for q in flagged if e[q] == -1), sum(dropR.values())))
    cache = {}
    for dose, name, c, w in sorted(found):
        if w not in cache:
            cache[w] = ps.load(w, "pro")[0]
        st = cache[w]
        if any(st.get((q, c), "?") == "?" for q in ids):
            print("   second level %-5s %-16s incomplete verdicts or other question ids" % (dose, name)); continue
        e2 = {q: VAL[st[(q, c)]] for q in ids}
        p2 = {q: (e2[q] if dropE[q] else n1[q]) for q in ids}
        p3 = {q: (e2[q] if dropE[q] else (0 if dropR[q] else n1[q])) for q in ids}
        c3 = collections.Counter((e[q], e2[q]) for q in flagged)
        d1 = [p2[q] - s2[q] for q in ids]; d2 = [p3[q] - n1[q] for q in ids]; d3 = [p3[q] - s3[q] for q in ids]
        rule = "prompt" if name.startswith("posprompt") else "old"
        print("   second level %-5s (%-14s rule %-6s): flagged right -> R %d W %d N %d; flagged wrong -> R %d W %d N %d | re-answer minus drop %+d [%+d, %+d] | re-answer + self-judge on released: net %+d, minus N1 %+d [%+d, %+d], minus drop version %+d [%+d, %+d]" % (
            dose, name, rule, c3[(1, 1)], c3[(1, -1)], c3[(1, 0)], c3[(-1, 1)], c3[(-1, -1)], c3[(-1, 0)],
            sum(d1), ci(d1)[0], ci(d1)[1], net(p3), sum(d2), ci(d2)[0], ci(d2)[1], sum(d3), ci(d3)[0], ci(d3)[1]))
