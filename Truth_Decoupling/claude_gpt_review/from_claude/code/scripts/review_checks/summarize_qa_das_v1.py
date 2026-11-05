#!/usr/bin/env python3
"""Compact read-out of a finished (or partly finished) QA-DAS v1 full run. JSON only, no model, no torch.

usage: summarize_qa_das_v1.py RUN_DIR [RUN_DIR ...]
Prints, per run: proxy shortlist, tuning table (192 questions, real outcomes, eligibility under the frozen rule),
frozen selection, test table (rates, deltas with paired-bootstrap 95% CI, 3x3 transitions, refusal-only-gold partition),
and the diagnostics table (prefill / reference-interchange arms) when present. Missing phases are reported as such.
"""
import json
import sys
from pathlib import Path


def load(run, name):
    p = run / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x):
    return "   n/a" if x is None else "%6.3f" % x


def delta(d, k):
    if not d or k not in d:
        return "            n/a"
    v = d[k]
    return "%+.3f [%+.3f, %+.3f]" % (v["value"], v["ci95"][0], v["ci95"][1])


def judged_table(title, s):
    print("  %s: completed=%s judge=%s rubric=%s n_questions=%s substantive=%s resolved_all=%s" % (
        title, s.get("completed"), s.get("judge"), s.get("judge_rubric_id"), s.get("n_questions"),
        s.get("n_substantive_gold"), s.get("n_resolved_all_conditions")))
    rows, trans = s["rates"], s.get("transitions_vs_baseline", {})
    print("  %-40s %7s %7s %7s %7s | %-22s %-22s %-22s | %s  %s" % (
        "condition", "correct", "wrong", "abstain", "ans", "d correct [CI]", "d abstain [CI]", "d answer_rate [CI]",
        "w->c a->c c->w c->a", "unres trunc"))
    for name, r in rows.items():
        rt, dv, t = r["rates"], r.get("delta_vs_baseline"), trans.get(name, {})
        tr = "" if not t else "%4d %4d %4d %4d" % (t.get("wrong->correct", 0), t.get("abstain->correct", 0),
                                                 t.get("correct->wrong", 0), t.get("correct->abstain", 0))
        print("  %-40s %s %s %s %s | %-22s %-22s %-22s | %-19s %s %.3f" % (
            name, pct(rt["correct"]), pct(rt["wrong"]), pct(rt["abstain"]), pct(rt["answer_rate"]),
            delta(dv, "correct"), delta(dv, "abstain"), delta(dv, "answer_rate"), tr, r.get("n_unresolved"),
            r.get("truncated_rate", float("nan"))))
    ao = {n: r.get("abstention_only_gold_questions") for n, r in rows.items()}
    first = next(iter(ao.values()), None)
    if first and sum(first.values()):
        print("  refusal-only-gold questions (reported separately): " + "; ".join(
            "%s: c%d w%d a%d u%d" % (n, v["correct"], v["wrong"], v["abstain"], v["unresolved"]) for n, v in ao.items()))


for arg in sys.argv[1:]:
    run = Path(arg)
    print("=" * 110)
    print("RUN", run)
    bank = load(run, "bank.json")
    if bank:
        print("  model=%s pairs=%d layers=%d das_candidates=%d md_layers=%d structural_null=%s" % (
            bank["model"], bank["n_E_pairs"], len(bank["layers"]), len(bank["das_candidate_layers"]),
            len(bank["md_layers"]), bank["structural_null_layers"]))
    ps, short = load(run, "proxy_scores.json"), load(run, "shortlist.json")
    if ps and short:
        b = ps["conditions"]["baseline"]
        print("  PROXY (paired answer log-prob accuracy on %d tuning pairs): baseline %.3f (%d)" % (
            len(b["correct"]), b["accuracy"], sum(b["correct"])))
        for est, cands in short["candidates"].items():
            for c in cands:
                r = ps["conditions"][c["name"]]
                print("    %-6s %-36s acc %.3f corrected %2d harmed %2d net %+d" % (
                    est, c["name"], r["accuracy"], r["corrected"], r["harmed"], r["net"]))
    else:
        print("  PROXY: not finished")
    ts = load(run, "tune_scored.json")
    if ts:
        judged_table("TUNE (real generation, %d questions)" % ts.get("n_questions", 0), ts)
    else:
        print("  TUNE: not judged yet")
    sel = load(run, "selection.json")
    if sel:
        print("  FROZEN SELECTION: " + "; ".join("%s -> %s" % (k, v["name"] if v else None) for k, v in sel["selected"].items()))
        for name, rep in sel["reports"].items():
            if name == "baseline":
                continue
            print("    %-36s net_correct %+3d (w->c %2d, a->c %2d, c->w %2d, c->a %2d) net_answered %+3d | eligible=%s %s" % (
                name, rep["net_correct"], rep["corrected"], rep["abstain_to_correct"], rep["damaged"],
                rep["correct_to_abstain"], rep["net_answered"], rep["eligible"], ",".join(rep["ineligibility_reasons"])))
        print("    rule: max |answer-count change| = %s of %s" % (sel["rule"]["max_abs_answer_count_change"], sel["n_questions"]))
    else:
        print("  SELECTION: not frozen yet")
    tst = load(run, "test_scored.json")
    if tst:
        judged_table("TEST (%d questions)" % tst.get("n_questions", 0), tst)
    else:
        print("  TEST: not judged yet")
    dg = load(run, "diagnostics_scored.json")
    if dg:
        judged_table("DIAGNOSTICS on tuning questions (reference arms are not deployable)", dg)
    else:
        print("  DIAGNOSTICS: not judged yet")
