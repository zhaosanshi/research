#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Fable 5.2, 2026-11-28), work queue discussion/work_queue/2026-11-28_queue_q.md.
# Independent recount of the main numbers of diag/table_confirm_1128.py on the FalseQA confirmation questions. Read-only
# (container python3, from /workspace/Truth_Decoupling); no model, no judge call. Not reviewed by GPT.
# It shares no code with table_confirm_1128.py, falseqa_confirm_1128.py, routes_1128.py or falseqa_read.py: the generation
# files and the judge files (judge2/cells.jsonl, judge2/single_pro.jsonl) are read directly, and the rule is written out
# again here:
#   answered = the generation row's rule flag false and non-empty text
#   method: E answered -> E's answer; E blank and O answered -> nothing; both blank -> E+C at the model's c once
#   score "handed in only": a handed-in answer scores +1 (label A), -1 (label B), 0 (labels C, H; judge declined); a blank 0
# Per model: handed in / right / wrong / net of method, single E, single C (the opponent's strength), original, and the two
# differences method - single E, method - single C (point values; the intervals stay with falseqa_read.pair_ci). The
# numbers are compared with the "JSON" line of the table's output and any difference stops the program.
# usage: python3 -B Iteration_01/diag/recount_confirm_1128.py q100|all
import json
import sys

NS = "Iteration_01/newsets"
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
# model: (E layer, E strength, C layer, c of the fill, single-C strength)   plan section 1 table
SETQ = {"llama3.1-8b": (16, "3.5", 19, "2", "2"), "qwen3.5-9b": (15, "2", 16, "0.5", "0.5"),
        "qwen3.8-27b": (28, "1.75", 33, "1", "0.25"), "gemma3-12b": (24, "2.5", 25, "2", "2"),
        "gemma3-27b": (28, "1", 31, "2", "0.25")}
VAL = {"A": 1, "B": -1, "C": 0, "H": 0}


def tok(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def recount(m, stage):
    el, s, cl, c, cc = SETQ[m]
    e = "x:E%d.w.p%s" % (el, s)
    arms = {"O": "x", "E": e, "C": "x:C%d.w.m%s" % (cl, cc), "EC": "%s+C%d.w.m%s" % (e, cl, c)}
    gen, cond = {}, {}
    for n, a in arms.items():
        with open("%s/gen_prompt/falseqa_confirm_1128/%s/test/%s.json" % (NS, m, tok(a)), encoding="utf-8") as f:
            g = json.load(f)
        if g["contract"]["arm"] != a:
            raise SystemExit("%s: the file of %s holds the arm %s" % (m, a, g["contract"]["arm"]))
        gen[n] = {r["id"]: r for r in g["rows"]}
        cond[n] = g["contract"]["condition"]
    w = "%s/judge/falseqa/%s/test_confirm_1128_%s/judge2/" % (NS, m, stage)
    lab = {r["cell_id"]: r for r in jsonl(w + "single_pro.jsonl")}
    pred = {r["cell_id"]: r["prediction"] for r in jsonl(w + "cells.jsonl")}
    seen = []
    for r in jsonl(w + "cells.jsonl"):
        q = r["pair_id"].split(":", 1)[1]
        if q not in seen:
            seen.append(q)
    qs = seen

    def ans(n, q):
        r = gen[n][q]
        return (not r["rule_abstain"]) and r["text"].strip() != ""

    def val(n, q):
        cid = "falseqa:%s|%s" % (q, cond[n])
        r = lab[cid]
        if pred[cid].strip() != gen[n][q]["text"].strip():
            raise SystemExit("%s: the judged text of %s %s is not the generation text" % (m, q, n))
        if r["status"] == "scored":
            return VAL[r["label"]]
        if r.get("basis") == "pro_declined":
            return 0
        raise SystemExit("%s: no verdict for %s %s (%s)" % (m, q, n, r.get("basis")))
    out = {k: {"handed": 0, "right": 0, "wrong": 0, "net": 0} for k in ("method", "E", "C", "O")}
    fills = 0

    def add(k, v):
        out[k]["handed"] += 1
        out[k]["right"] += v == 1
        out[k]["wrong"] += v == -1
        out[k]["net"] += v
    for q in qs:
        for k in ("E", "C", "O"):
            if ans(k, q):
                add(k, val(k, q))
        if ans("E", q):
            add("method", val("E", q))
        elif ans("O", q):
            pass
        elif ans("EC", q):
            fills += 1
            add("method", val("EC", q))
    out["n"] = len(qs)
    out["dE"] = out["method"]["net"] - out["E"]["net"]
    out["dC"] = out["method"]["net"] - out["C"]["net"]
    out["fills_handed"] = fills
    return out


def main():
    a = sys.argv[1:]
    if a not in (["q100"], ["all"]):
        raise SystemExit("usage: recount_confirm_1128.py q100|all")
    stage = a[0]
    tab = "Iteration_01/diag/out_confirm_1128_%s_20261128.txt" % stage
    with open(tab, encoding="utf-8") as f:
        lines = [l for l in f if l.startswith("JSON ")]
    if len(lines) != 1:
        raise SystemExit("%s: %d JSON lines" % (tab, len(lines)))
    js = json.loads(lines[0][5:])
    bad = 0
    for m in ORDER:
        r = recount(m, stage)
        t = js[m]
        want = {"n": t["n"], "dE": t["dE"][0], "dC": t["dC"][0]}
        got = {"n": r["n"], "dE": r["dE"], "dC": r["dC"]}
        for k in ("method", "E", "C", "O"):
            want[k] = {x: t["stats"][k][x] for x in ("handed", "right", "wrong", "net")}
            got[k] = {x: int(r[k][x]) for x in ("handed", "right", "wrong", "net")}
        same = got == want
        bad += not same
        print("recount %-12s n %d | method %d/%d/%d/%+d | E %d/%d/%d/%+d | C %d/%d/%d/%+d | O %d/%d/%d/%+d | method - E %+d | method - C %+d | "
              "fills handed in %d -> %s%s" % (
                  m, r["n"], *[got[k][x] for k in ("method", "E", "C", "O") for x in ("handed", "right", "wrong", "net")],
                  r["dE"], r["dC"], r["fills_handed"], "EQUAL" if same else "DIFFERENT", "" if same else " (table %s)" % want))
    if bad:
        raise SystemExit("RECOUNT FAILED (%d models differ from %s)" % (bad, tab))
    print("RECOUNT OK (5 models equal %s)" % tab)


if __name__ == "__main__":
    main()
