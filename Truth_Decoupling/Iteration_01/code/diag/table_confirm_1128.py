#!/usr/bin/env python3
# Iteration_01 exploration (executor hai, Claude Opus 5.5, 2026-11-28), work queue discussion/work_queue/2026-11-28_queue_q.md
# sections 2.4, 3 and 4. Read-only (container python3, from /workspace/Truth_Decoupling): existing answers and DeepSeek Pro
# verdicts; no model, no judge call, nothing chosen here. Not reviewed by GPT.
# The rule and the scoring are those of diag/routes_1128.py, imported and called unchanged (routes_1128.schemes, .sc, .net):
#   method = R1_<c> (E answered -> E; E blank and O answered -> nothing; both blank -> EC(c) once), ladder variant R1L_<c>,
#   single E = E, single C = C<cC>, original = O; "handed in only" scoring.
# Regression first (stop on any mismatch): on the held-out 600 questions of the 700 bank (routes_1128.fqa_split, unchanged),
#   per model with the plan's c and cC: net / handed in / dE / dC of R1_<c>, and the nets of E, C<cC> (= "Cdev") and O equal
#   the JSON line of diag/out_routes_1128_20261128.txt (= results/write_positions/routes_1128_20261128.json); also the JSON's
#   single-C choice equals cC and its TruthfulQA R1_t equals c.
# New questions (confirm bank, diag/falseqa_confirm_1128.py): per model the eight generation files and the verdicts of the
#   stage's work dir. Checks: the judge input's list of judged cells = the cells the rule needs (recomputed from the
#   generation rows); the non-empty cells of the work dir = exactly those cells, each with a verdict; judged text =
#   generation text. Cells (value, answered): answered = rule flag false and non-empty text; an answered cell that was not
#   judged (not needed by any reading) carries NaN, single C at strengths other than cC is all NaN; every reading is
#   checked finite, so no unjudged cell can enter a number.
# Readings (plan section 3), per model (order llama3.1-8b, qwen3.5-9b, qwen3.8-27b, gemma3-12b, gemma3-27b), no pooled row:
#   1 method - single E, method - single C (falseqa_read.pair_ci, pairs resampled, on the questions read), with handed in /
#     right / wrong / net of method, E, C, O; 2 the same two differences by premise; 3 the four cells (O answered x E
#     answered): counts, and in the both-blank cell the nets of EC(c) and of single C; 4 the ladder R1L_<c>: net and the two
#     differences. Then a summary with one row per model. Stage q100: the line "EXPAND_DECISION ..." (plan section 4: 4 or 5
#     models with method - single E > 0 -> yes).
# usage: python3 -B Iteration_01/diag/table_confirm_1128.py regress | smoke MODEL | q100 | all
import json
import math
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/gpt_tools_20261126/gate_audit"]
import falseqa_confirm_1128 as H  # noqa: E402
import routes_1128 as RT          # noqa: E402

F = H.F
Q = F.Q
KS = H.KS
ROUTES_OUT = "Iteration_01/diag/out_routes_1128_20261128.txt"
NAN = float("nan")


def routes_json():
    with open(ROUTES_OUT, encoding="utf-8") as f:
        lines = [l for l in f if l.startswith("JSON ")]
    if len(lines) != 1:
        raise SystemExit("%s: %d JSON lines" % (ROUTES_OUT, len(lines)))
    return json.loads(lines[0][5:])


def ci(a, b, qs, pair):
    return [int(x) for x in F.pair_ci({q: a[q][0] - b[q][0] for q in qs}, pair, qs)]


def stats(d, qs):
    return {"net": RT.net(d, qs), "handed": sum(d[q][1] for q in qs), "right": sum(d[q][0] == 1 for q in qs),
            "wrong": sum(d[q][0] == -1 for q in qs)}


def regression():
    js = routes_json()
    ok, rows = F.check_bank()
    if not ok:
        raise SystemExit("700 bank check FAILED")
    pair = {r["id"]: r["pair_id"] for r in rows}
    bad = 0
    for m in H.ORDER:
        el, s, cl, c, cc = H.setting(m)
        rec, tq = js["FalseQA|" + m], js["TruthfulQA|" + m]
        ids, arm, _ = RT.fqa_split(m, rows, "heldout")
        X = RT.schemes(ids, arm)
        r1, e, cd, o = X["R1_" + c][0], X["E"][0], X["C" + cc][0], X["O"][0]
        got = {"n": len(ids), "c_single": cc, "tqa_R1_t": c, "R1": {"net": RT.net(r1, ids), "handed": sum(r1[q][1] for q in ids),
                                                                  "dE": ci(r1, e, ids, pair), "dC": ci(r1, cd, ids, pair)},
               "E": RT.net(e, ids), "Cdev": RT.net(cd, ids), "O": RT.net(o, ids)}
        sch = rec["schemes"]
        want = {"n": rec["n"], "c_single": rec["c_single"], "tqa_R1_t": tq["tuned"]["R1_t"],
                "R1": {k: sch["R1_" + c][k] for k in ("net", "handed", "dE", "dC")},
                "E": sch["E"]["net"], "Cdev": sch["Cdev"]["net"], "O": sch["O"]["net"]}
        same = got == want
        bad += not same
        print("regression %-12s c %-4s cC %-4s | R1 net %+d handed %d dE %s dC %s | E %+d C %+d O %+d -> %s%s" % (
            m, c, cc, got["R1"]["net"], got["R1"]["handed"], got["R1"]["dE"], got["R1"]["dC"], got["E"], got["Cdev"], got["O"],
            "EQUAL" if same else "MISMATCH", "" if same else " (want %s)" % want))
    if bad:
        raise SystemExit("REGRESSION FAILED (%d models)" % bad)
    print("REGRESSION OK (5 models equal the routes_1128 JSON)")
    sys.stdout.flush()


def verdicts(m, stage):
    from judge_two_axis_run import load_verified_single
    w = H.work(m, stage)
    _, cells, votes = load_verified_single(str(w), "pro", generation_path=str(w / H.JIN), require_complete=False)
    val, pred, declined = {}, {}, {}
    for key, c in cells.items():
        if not c["prediction"].strip():
            continue
        v = votes[key]
        k = (c["pair_id"].split(":", 1)[1], c["condition"])
        if v["status"] == "scored":
            val[k] = Q.VALUE[v["label"]]
        elif v.get("basis") == "pro_declined":
            val[k] = 0
            declined[k] = True
        else:
            raise SystemExit("%s: no verdict for %s %s" % (m, k[0], k[1]))
        pred[k] = c["prediction"]
    jin = json.load(open(w / H.JIN, encoding="utf-8"))
    return val, pred, declined, jin


def load_model(m, stage, rows, idf):
    mode = "smoke" if stage == "smoke" else "full"
    el, s, cl, c, cc = H.setting(m)
    gen, cond, con, _ = H.load_gen(m, mode, rows)
    qs = H.stage_ids(stage, rows, idf)
    need = H.needed(m, gen, qs)
    val, pred, declined, jin = verdicts(m, stage)
    if jin[H.TAG]["needed"] != {cond[n]: l for n, l in need.items()} or jin[H.TAG]["stage"] != stage:
        raise SystemExit("%s: the judge input's list of judged cells differs from the rule's" % m)
    if jin["question_ids"] != ["falseqa:%s" % q for q in qs]:
        raise SystemExit("%s: the judge input covers other questions" % m)
    needset = {(q, cond[n]) for n, l in need.items() for q in l}
    if set(val) != needset:
        raise SystemExit("%s: judged cells %d, cells the rule needs %d, differing %d" % (m, len(val), len(needset), len(set(val) ^ needset)))
    byc = {cond[n]: n for n in cond}
    for (q, cn), t in pred.items():
        if t.strip() != gen[byc[cn]][q]["text"].strip():
            raise SystemExit("%s: the judged text of %s under %s is not the generation text" % (m, q, byc[cn]))
    arm = {}
    for n in cond:
        d = {}
        for q in qs:
            if not H.answered(gen[n][q]):
                d[q] = (0, False)
            else:
                d[q] = (val[(q, cond[n])], True) if (q, cond[n]) in val else (NAN, True)
        arm[n] = d
    out = {"O": arm["O"], "E": arm["E"], "C" + cc: arm["C"]}
    for k in KS:
        out["EC" + k] = arm["EC" + k]
        if k != cc:
            out["C" + k] = {q: (NAN, True) for q in qs}
    return qs, out, len(val), declined, gen, cond


def finite(d, qs, name, m):
    for q in qs:
        v = d[q][0]
        if isinstance(v, float) and math.isnan(v) or v not in (-1, 0, 1):
            raise SystemExit("%s: %s uses a cell without a verdict (%s)" % (m, name, q))


def reading(m, stage, rows, idf, pair, prem):
    el, s, cl, c, cc = H.setting(m)
    qs, arm, nj, declined, gen, cond = load_model(m, stage, rows, idf)
    X = RT.schemes(qs, arm)
    names = {"method": "R1_" + c, "ladder": "R1L_" + c, "E": "E", "C": "C" + cc, "O": "O"}
    d = {k: X[v][0] for k, v in names.items()}
    for k, v in d.items():
        finite(v, qs, k, m)
    O, E = arm["O"], arm["E"]
    r = {"n": len(qs), "pairs": len({pair[q] for q in qs}), "judged_cells": nj, "declined": len(declined),
         "stats": {k: stats(v, qs) for k, v in d.items()}, "gens": {k: X[v][1] for k, v in names.items()},
         "dE": ci(d["method"], d["E"], qs, pair), "dC": ci(d["method"], d["C"], qs, pair),
         "ladder_dE": ci(d["ladder"], d["E"], qs, pair), "ladder_dC": ci(d["ladder"], d["C"], qs, pair), "premise": {}, "cells": {}}
    for g in ("false", "valid"):
        qq = [q for q in qs if prem[q] == g]
        r["premise"][g] = {"n": len(qq), "dE": ci(d["method"], d["E"], qq, pair), "dC": ci(d["method"], d["C"], qq, pair),
                           "net": {k: RT.net(d[k], qq) for k in ("method", "E", "C", "O")}}
    ecc = arm["EC" + c]
    for oa in (True, False):
        for ea in (True, False):
            qq = [q for q in qs if O[q][1] == oa and E[q][1] == ea]
            key = ("O answered" if oa else "O blank") + " / " + ("E answered" if ea else "E blank")
            cell = {"n": len(qq)}
            if not oa and not ea:
                sc_ec = {q: RT.sc(ecc[q]) for q in qq}
                sc_c = {q: RT.sc(arm["C" + cc][q]) for q in qq}
                finite(sc_ec, qq, "EC(c) in the both-blank cell", m)
                finite(sc_c, qq, "C in the both-blank cell", m)
                cell.update({"EC_c": stats(sc_ec, qq), "C": stats(sc_c, qq)})
            r["cells"][key] = cell
    return r


def show(m, r, c, cc):
    st = r["stats"]
    print("\n== %s (c %s, single C %s): %d questions, %d pairs, judged cells %d, declined %d" % (
        m, c, cc, r["n"], r["pairs"], r["judged_cells"], r["declined"]))
    for k in ("method", "E", "C", "O", "ladder"):
        x = st[k]
        print("   %-7s handed in %3d  right %3d  wrong %3d  net %+4d  generations %s" % (k, x["handed"], x["right"], x["wrong"], x["net"], r["gens"][k]))
    print("   1 method - single E %s | method - single C %s" % (F.fmt_ci(r["dE"]), F.fmt_ci(r["dC"])))
    for g in ("false", "valid"):
        p = r["premise"][g]
        print("   2 premise %-5s (%d): method - E %s | method - C %s | nets method %+d E %+d C %+d O %+d" % (
            g, p["n"], F.fmt_ci(p["dE"]), F.fmt_ci(p["dC"]), p["net"]["method"], p["net"]["E"], p["net"]["C"], p["net"]["O"]))
    for key, cell in r["cells"].items():
        extra = ""
        if "EC_c" in cell:
            a, b = cell["EC_c"], cell["C"]
            extra = " | EC(c) net %+d (handed %d, right %d, wrong %d) | single C net %+d (handed %d, right %d, wrong %d)" % (
                a["net"], a["handed"], a["right"], a["wrong"], b["net"], b["handed"], b["right"], b["wrong"])
        print("   3 cell %-24s n %3d%s" % (key, cell["n"], extra))
    print("   4 ladder R1L_%s net %+d | - single E %s | - single C %s" % (c, st["ladder"]["net"], F.fmt_ci(r["ladder_dE"]), F.fmt_ci(r["ladder_dC"])))


def run_stage(stage, models):
    rows, idf = H.load_bank()
    pair = {r["id"]: r["pair_id"] for r in rows}
    prem = {r["id"]: r["premise"] for r in rows}
    out = {}
    print("\n# confirm bank %s: %d pairs, %d questions; stage %s" % (H.BANK, idf["n_pairs"], idf["n_questions"], stage))
    for m in models:
        el, s, cl, c, cc = H.setting(m)
        out[m] = reading(m, stage, rows, idf, pair, prem)
        show(m, out[m], c, cc)
    print("\n# summary (one row per model; no pooled row)")
    print("   %-12s %4s | %-22s | %-22s | method h/r/w/net | E h/r/w/net | C h/r/w/net | O h/r/w/net | ladder - E" % (
        "model", "n", "method - single E", "method - single C"))
    for m in models:
        r = out[m]
        f = lambda k: "%d/%d/%d/%+d" % (r["stats"][k]["handed"], r["stats"][k]["right"], r["stats"][k]["wrong"], r["stats"][k]["net"])  # noqa: E731
        print("   %-12s %4d | %-22s | %-22s | %s | %s | %s | %s | %s" % (m, r["n"], F.fmt_ci(r["dE"]), F.fmt_ci(r["dC"]),
                                                                  f("method"), f("E"), f("C"), f("O"), F.fmt_ci(r["ladder_dE"])))
    if stage == "q100":
        pos = sum(out[m]["dE"][0] > 0 for m in models)
        if len(models) != 5:
            raise SystemExit("the expansion rule needs the five models")
        print("EXPAND_DECISION positive_models %d of 5 (method - single E > 0) -> EXPAND %s" % (pos, "yes" if pos >= 4 else "no"))
    print("JSON " + json.dumps(out, sort_keys=True))


def run_smoke(m):
    rows, idf = H.load_bank()
    el, s, cl, c, cc = H.setting(m)
    qs, arm, nj, declined, gen, cond = load_model(m, "smoke", rows, idf)
    X = RT.schemes(qs, arm)
    print("smoke %s: %d questions, judged cells %d, declined %d" % (m, len(qs), nj, len(declined)))
    for q in qs:
        cells = " ".join("%s=%s/%s" % (n, "ans" if arm[n][q][1] else "blank",
                                       "na" if isinstance(arm[n][q][0], float) else arm[n][q][0])
                         for n in ["O", "E", "C" + cc] + ["EC" + k for k in KS])
        sch = " ".join("%s=%s" % (k, X[k][0][q]) for k in ("R1_" + c, "R1L_" + c, "E", "C" + cc, "O"))
        print("   %s | %s || %s" % (q, cells, sch))
    for k in ("R1_" + c, "R1L_" + c, "E", "C" + cc, "O"):
        finite(X[k][0], qs, k, m)
        print("   %s net %+d handed %d" % (k, RT.net(X[k][0], qs), sum(X[k][0][q][1] for q in qs)))


def main():
    a = sys.argv[1:]
    if a == ["regress"]:
        regression()
    elif len(a) == 2 and a[0] == "smoke" and a[1] in H.SETQ:
        regression()
        run_smoke(a[1])
    elif a in (["q100"], ["all"]):
        regression()
        run_stage(a[0], H.ORDER)
    else:
        raise SystemExit("usage: table_confirm_1128.py regress | smoke MODEL | q100 | all")


if __name__ == "__main__":
    main()
