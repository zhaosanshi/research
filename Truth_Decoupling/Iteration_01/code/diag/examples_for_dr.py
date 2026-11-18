# Iteration_01 analysis (Claude Opus 5.5, 2026-11-19), read-only, no model, no judge call.
# Worked examples for the Dr, who asked on 2026-11-15 for a few questions that C turned from no answer into an answer, right
# and wrong ones both (work queue A2, discussion/work_queue/2026-11-19_queue.md).
# The rule, fixed before looking at any example: per model, the test split with Pro verdicts (the two Gemma-3: the full test
# split), N1 with the settings chosen on tune (Pro), the answers N1 actually hands in (final_records.py). Among the questions
# blank under E that N1 releases with C, in question-number order: the first 3 judged right and the first 3 judged wrong;
# then the first 2 vetoed questions (an E's blank whose first answer was its original no-write answer, kept blank).
# Printed per example: id, provenance (own blank / E's blank), dose, question, reference answer, the no-write answer, the E
# answer, the answer N1 hands in, the verdict.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/examples_for_dr.py
import json
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import release_origin as ro
import confirm_fresh as cfr
import final_records as fr

CAPS = ro.CAPS
VERDICT = {"R": "right", "W": "wrong", "H": "hedge", "X": "no answer", "?": "no verdict"}
# --md: one Markdown table row per example for the note to the Dr (labels in Chinese, written as escapes to keep this file ASCII)
MD_TAG = {"right": "\u7b54\u5bf9", "wrong": "\u7b54\u9519", "vetoed": "\u5426\u51b3"}
MD_CAND = {"own": "\u539f\u672c\u4e0d\u7b54", "E's": "\u63a8 E \u540e\u624d\u4e0d\u7b54"}
MD_VERDICT = {"R": "\u5bf9", "W": "\u9519", "H": "\u542b\u7cca", "X": "\u4e0d\u7b54", "?": "\u65e0\u5224\u5206"}
MD = "--md" in sys.argv


def cell(t):
    return (t or "").replace("|", "\\|").replace("\n", " ").strip()


def cells(work):
    out = {}
    for line in open(work + "/judge2/cells.jsonl", encoding="utf-8"):
        r = json.loads(line)
        out[(r["pair_id"], r["condition"])] = r
    return out


def num(p):
    return int(p.split(":")[1])


def show(tag, r, sp, tx, cl, s):
    p = r["p"]
    c = cl[(p, "baseline_x")]
    if MD:
        print("| %s | %s | %s | %s | %s | %s | %s (%s) | %s (%s) |" % (
            MD_TAG[tag], p, MD_CAND[r["cand"]], ps.NAME[r["rung"]] if r["rung"] else "-", cell(c["question"]), cell(c.get("best_answer")),
            cell(tx.ec[(p, "baseline_x")]), MD_VERDICT[sp.st[(p, "baseline_x")]], cell(r["text"]), MD_VERDICT[r["state"]]))
        return
    print("  - [%s] %s | %s | dose %s" % (tag, p, "own blank (no answer with no write either)" if r["cand"] == "own" else
                                         "E's blank (answered with no write)", ps.NAME[r["rung"]] if r["rung"] else "-"))
    print("      question:  %s" % c["question"])
    print("      reference: %s" % c.get("best_answer"))
    print("      no write:  %r (%s)" % (tx.ec[(p, "baseline_x")], VERDICT[sp.st[(p, "baseline_x")]]))
    print("      E alone:   %r" % tx.ec[(p, sp.cE(s))])
    print("      N1 hands in: %r (%s)" % (r["text"], VERDICT[r["state"]]))


def report(name, m_test, tune_m):
    tune = ps.Split(tune_m, 0)
    ttx = ro.texts(tune_m["ec"][0])
    fam = [("E", s) for s in "12"] + [("N1", s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
    pick = max(fam, key=lambda x: (ps.score(ro.states(tune, ttx, x), 1), -fam.index(x)))
    sp = ps.Split(m_test, 1)
    tx = fr.Texts.of(m_test, 1)
    cl = cells(m_test["ec"][1])
    recs = sorted(fr.records(sp, pick, tx), key=lambda r: num(r["p"]))
    rel = [r for r in recs if r["note"] == "released"]
    vet = [r for r in recs if r["note"].startswith("vetoed")]
    right = [r for r in rel if r["state"] == "R"]
    wrong = [r for r in rel if r["state"] == "W"]
    print("\n## %s: %d test questions; N1 = %s" % (name, len(sp.qs), ro.name_of(pick)))
    print("  released by C: %d (right %d, wrong %d, other %d; own blanks %d, E's blanks %d); vetoed %d (their no-write answers: right %d, wrong %d, other %d)" % (
        len(rel), len(right), len(wrong), len(rel) - len(right) - len(wrong), sum(r["cand"] == "own" for r in rel),
        sum(r["cand"] == "E's" for r in rel), len(vet), sum(sp.st[(r["p"], "baseline_x")] == "R" for r in vet),
        sum(sp.st[(r["p"], "baseline_x")] == "W" for r in vet), sum(sp.st[(r["p"], "baseline_x")] not in "RW" for r in vet)))
    if pick[0] != "N1":
        print("  N1 chose E alone on tune: nothing is released")
        return
    for tag, group in (("right", right[:3]), ("wrong", wrong[:3]), ("vetoed", vet[:2])):
        for r in group:
            show(tag, r, sp, tx, cl, pick[1])


def main():
    for name in ro.FIVE:
        m = cf.MODELS[name]
        if name in cfr.FULL:
            ec_full, d_full = cfr.FULL[name]
            mt = dict(m, ec=(m["ec"][0], ec_full), d=(m["d"][0], d_full), judge=(m["judge"][0], "pro"))
        else:
            mt = m
        report(name, mt, m)


if __name__ == "__main__":
    main()
