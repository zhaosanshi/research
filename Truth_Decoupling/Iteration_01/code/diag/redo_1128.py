#!/usr/bin/env python3
# Iteration_01 exploration (executor R-jia, Claude Opus 5.5, 2026-11-28), work queue
# discussion/work_queue/2026-11-28_queue_r.md sections 2 and 3.1: the offline recomputation R1-R5 with the method ("M")
# in place of N1. Read-only: existing answers and DeepSeek Pro verdicts; no model, no judge call, nothing generated.
# Not reviewed by GPT.
# Method M = routes_1128.schemes(ids, arm)["R1_<c>"], c = 2 / 0.5 / 1 / 2 / 2 (Llama-3.1-8B, Qwen3.5-9B, Qwen3.8-27B,
#   Gemma-3-12B, Gemma-3-27B); routes_1128 is imported and used unchanged. Its scoring routes_1128.sc ("handed in only":
#   an answer blank by the method's text rule scores 0 whatever the judge said) is applied to M and to every opponent.
# Order (binding): every table first runs the ORIGINAL program in-process on its own functions (stdout and stderr in one
#   stream) and compares the output line by line with the archived file, which arrives on stdin in a JSON bundle
#   {key: text} (the archived files are in the review repository, not on the server). Only when equal are the method's
#   numbers of that table printed; otherwise the table stops and the differing lines are printed. In addition the
#   opponent vectors read here are checked against the original program's own "N1 minus X" strings (original scoring,
#   the original N1), so the glue code of this file reads the same cells as the original.
# Arms that do not exist (GSM8K: E + C only on E's blanks and only up to the strengths N1 needed, no single C;
#   (QA)^2: some single C / E + C passes) enter routes_1128.schemes as placeholder cells (0, not handed in); schemes then
#   computes every scheme, but only R1_<c> is read, and it is checked that every cell R1_<c> reads is a real one.
# Intervals as the originals: TruthfulQA questions resampled (gate_sim_1126.tq_ci = policy_select2.ci as numbers);
#   (QA)^2 per model policy_select2.ci, models together newsets_score.joint_ci; FalseQA pairs (falseqa_read.pair_ci);
#   GSM8K items (cap_score.ci). The sum over the five models is a reference row only.
# Output: a text report, then one line "JSON <dump>" with every number.
# usage (container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/redo_1128.py R1|R2|R3|R4|R5 < bundle.json
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/gpt_tools_20261126/gate_audit"]
import routes_1128 as RT          # noqa: E402

A, GS, F = RT.A, RT.GS, RT.F
MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
MT, DM, PW = MP.MT, MP.DM, ML.PW
ORDER = RT.ORDER
CM = {"llama3.1-8b": "2", "qwen3.5-9b": "0.5", "qwen3.8-27b": "1", "gemma3-12b": "2", "gemma3-27b": "2"}
PH0 = (0, False)          # placeholder cell of an arm that does not exist
DUMP = {}
SRC = {}                  # model -> main_table_layers.sources(display), read once


# ------------------------------------------------------------------------------------------------ common helpers
def val(x):
    return (x == "R") - (x == "W")


def hs(v, a):
    """handed in only"""
    return v if a else 0


def fmt(t):
    return "%+d [%+d, %+d]" % tuple(t)


def tq(v):
    t = GS.tq_ci(list(v))
    if fmt(t) != ps.ci(list(v)):
        raise SystemExit("tq_ci differs from policy_select2.ci")
    return [int(x) for x in t]


def capture(fn, argv=None):
    """(text printed on stdout and stderr in one stream, exit code or None) of fn()."""
    buf = io.StringIO()
    old = sys.argv
    if argv is not None:
        sys.argv = list(argv)
    code = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn()
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old
    return buf.getvalue(), code


def capture_split(fn, argv=None):
    """(stdout text, stderr text, exit code or None) of fn(), the two streams kept apart."""
    out, err = io.StringIO(), io.StringIO()
    old = sys.argv
    if argv is not None:
        sys.argv = list(argv)
    code = None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fn()
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = old
    return out.getvalue(), err.getvalue(), code


def compare(label, got, want, rec):
    """Line-by-line comparison of a program's output with its archived file."""
    g, w = got.splitlines(), want.splitlines()
    bad = [i for i in range(max(len(g), len(w))) if (g[i] if i < len(g) else None) != (w[i] if i < len(w) else None)]
    rec[label] = {"equal": not bad, "archived_lines": len(w), "printed_lines": len(g), "differing_lines": len(bad)}
    if not bad:
        print("REGRESSION %s: EQUAL, %d of %d lines identical" % (label, len(w), len(w)))
        return True
    print("REGRESSION %s: DIFFERS on %d lines (archived %d lines, printed %d)" % (label, len(bad), len(w), len(g)))
    for i in bad[:8]:
        print("    line %d archived: %s" % (i + 1, (w[i] if i < len(w) else "<none>")[:400]))
        print("    line %d now     : %s" % (i + 1, (g[i] if i < len(g) else "<none>")[:400]))
    return False


def must_contain(label, text, s, rec):
    ok = s in text
    rec.setdefault("glue_checks", []).append([label, ok])
    print("  glue check %-64s %s" % (label, "found in the original output" if ok else "NOT FOUND: " + s[:300]))
    if not ok:
        raise SystemExit("glue check failed: %s" % label)


def method_of(ids, arm, m):
    X = RT.schemes(ids, arm)
    return X["R1_" + CM[m]][0]


def counts(d, qs):
    return {"net": sum(d[q][0] for q in qs), "handed": sum(bool(d[q][1]) for q in qs),
            "right": sum(d[q][0] == 1 for q in qs), "wrong": sum(d[q][0] == -1 for q in qs)}


def from_states(states, answered, qs):
    """{q: (handed-in-only score, handed)} from aligned state / answered lists; an unscored handed cell stops."""
    out = {}
    for q, x, a in zip(qs, states, answered):
        if a and x == "?":
            raise SystemExit("an answered cell without a verdict (%s)" % q)
        out[q] = (hs(val(x), a), bool(a))
    return out


def tq_routes(m):
    """TruthfulQA test / tune arms of routes_1128, the method, the single-C strength (handed-in-only tune choice)."""
    tids, tarm, _ = RT.tqa_split(m, 0)
    ids, arm, n1 = RT.tqa_split(m, 1)
    T = RT.schemes(tids, tarm)
    ct = RT.best(["R1_" + k for k in RT.KS], T, tids).split("_")[-1]
    if ct != CM[m]:
        raise SystemExit("%s: the method strength chosen on tune is %s, the plan says %s" % (m, ct, CM[m]))
    ck = RT.best(["C" + k for k in RT.KS], T, tids)[1:]
    Md = method_of(ids, arm, m)
    return dict(tids=tids, ids=ids, arm=arm, n1=n1, M=Md, ck=ck)


def routes_check(rec, task, m, Md, ids, routes):
    got = counts(Md, ids)
    ref = routes["%s|%s" % (task, m)]["schemes"]["M"]
    same = all(got[k] == ref[k] for k in ("net", "handed", "right", "wrong"))
    rec.setdefault("routes_check", {})[m] = [got, {k: ref[k] for k in ("net", "handed", "right", "wrong")}, same]
    print("  %-12s M net %+d handed %d (right %d wrong %d) | routes_1128_20261128.json M: net %+d handed %d -> %s" % (
        m, got["net"], got["handed"], got["right"], got["wrong"], ref["net"], ref["handed"], "SAME" if same else "DIFFERENT"))
    if not same:
        raise SystemExit("the method differs from the archived routes_1128 output")


def report_rows(title, per, names, ci_one, ci_all, common, rec):
    """per[m] = {"M": {q: (s, h)}, name: {q: (s, h)}, "ids": [...]}; method minus each opponent per model, then the sum."""
    print("\n" + title)
    out = {}
    for m in ORDER:
        if m not in per:
            continue
        d = per[m]
        ids = d["ids"]
        r = {"n": len(ids), "M": counts(d["M"], ids)}
        cells = ["M %+d (%d)" % (r["M"]["net"], r["M"]["handed"])]
        for n in names:
            if n not in d:
                continue
            r[n] = counts(d[n], ids)
            cells.append("%s %+d (%d)" % (n, r[n]["net"], r[n]["handed"]))
        diffs = []
        for n in names:
            if n not in d:
                diffs.append("%s: missing (%s)" % (n, d.get("missing", {}).get(n, "no data")))
                continue
            t = ci_one([d["M"][q][0] - d[n][q][0] for q in ids])
            r["M-" + n] = t
            diffs.append("M - %s %s" % (n, fmt(t)))
        out[m] = r
        print("  %-12s %d q | net (handed in): %s" % (m, len(ids), " | ".join(cells)))
        print("  %-12s %s" % ("", " | ".join(diffs)))
    if common is not None and all(m in per for m in ORDER):
        tot = {}
        line = []
        for n in names:
            if not all(n in per[m] for m in ORDER):
                line.append("%s: not every model has it" % n)
                continue
            t = ci_all(n, common)
            tot["M-" + n] = t
            line.append("M - %s %s" % (n, fmt(t)))
        out["five models (reference only)"] = dict(tot, n=len(common))
        print("  %-12s %d q (reference only): %s" % ("five models", len(common), " | ".join(line)))
    rec[title] = out
    return out


# ------------------------------------------------------------------------------------------------ R1 TruthfulQA main table
def r1(B):
    import prompt_conf_table as PC
    import plain_honesty as PH
    rec = DUMP.setdefault("R1", {"regression": {}})
    print("# R1 TruthfulQA test at the prompt rule (main settings): the method against the main table's opponents; handed in only")
    out_main, code = capture(MP.main, ["main_table_prompt.py", "1"])
    if not compare("main_table_prompt.py 1 vs main_table_prompt_budget1_20261121.txt", out_main, B["main_table_prompt"], rec["regression"]):
        print("R1 STOPPED: the main table does not reproduce")
        return
    out_pc, code_pc = capture(PC.main, ["prompt_conf_table.py", "1"])
    ok_pc = compare("prompt_conf_table.py 1 vs prompt_conf_table_budget1_20261121.txt", out_pc, B["prompt_conf_table"], rec["regression"])
    ph_want = "\n".join(B["plain_honesty_prompt"].splitlines()[:17])
    out_ph, code_ph = capture(PH.tqa)
    ok_ph = compare("plain_honesty.py tqa vs plain_honesty_prompt_five_models_20261122.txt lines 1-17", out_ph, ph_want, rec["regression"])
    routes = json.loads(B["routes_json"])
    per, singlec = {}, {}
    for m in ORDER:
        display, layer, s = A.KEY[m]
        R = tq_routes(m)
        ids, arm, n1, Md, ck = R["ids"], R["arm"], R["n1"], R["M"], R["ck"]
        routes_check(rec, "TruthfulQA", m, Md, ids, routes)
        nM = sum(Md[q][1] for q in ids)
        d = {"ids": ids, "M": Md, "missing": {}}
        d["no write"] = {q: RT.sc(arm["O"][q]) for q in ids}
        d["single C"] = {q: RT.sc(arm["C" + ck][q]) for q in ids}
        d["single E"] = {q: RT.sc(arm["E"][q]) for q in ids}
        orig = {"no write": [arm["O"][q][0] for q in ids], "single C": [arm["C" + ck][q][0] for q in ids],
                "single E": [arm["E"][q][0] for q in ids]}
        n1v = [n1[q][0] for q in ids]
        N1n = sum(n1[q][1] for q in ids)
        Dt, Dx, srcs = SRC.setdefault(m, ML.sources(display))
        run_x = Dx.m["ec"][1].split("/eval_extra/")[0]
        # the strong form at the method's handed-in count, and at N1's count for the glue check
        sf, why = MP.strong_prompt(run_x, layer, s, ids, nM)
        if sf is None:
            d["missing"]["strong form"] = why
        else:
            d["strong form"] = from_states(sf[0], sf[1], ids)
        sf1, why1 = MP.strong_prompt(run_x, layer, s, ids, N1n)
        orig["the strong form"] = [val(x) for x in sf1[0]]
        # prompt only (must-answer + sentence)
        da0 = MT.handed_in(Dx, ("DA",))
        da = dict(zip(Dx.sp.qs, zip(da0[0], da0[1])))
        d["prompt only"] = from_states([da[q][0] for q in ids], [da[q][1] for q in ids], ids)
        orig["prompt only"] = [val(da[q][0]) for q in ids]
        # the best prompt on tune (the main table's choice rule), with the blank flags of its judged answers
        run = run_x
        ptune, ptest = PW.prompts_of(run, "tune", srcs[0][1].qs), PW.prompts_of(run, "test", srcs[0][3].qs)
        order = sorted(ptune, key=lambda p: (-sum(map(val, ptune[p][0])), p != "conf_E2", p))
        bp = order[0]
        by = dict(zip(srcs[0][3].qs, ptest[bp][0]))
        tj, xj = PC.judged(run, "tune", srcs[0][1].qs), PC.judged(run, "test", ids)
        if [by[q] for q in ids] != xj[bp][0]:
            raise SystemExit("%s: the best prompt's states differ between the two readers" % m)
        d["best prompt on tune"] = from_states(xj[bp][0], xj[bp][1], ids)
        orig["the best prompt on tune"] = [val(by[q]) for q in ids]
        hnet = {p: sum(hs(val(x), a) for x, a in zip(tj[p][0], tj[p][1])) for p in tj}
        bp_h = sorted(hnet, key=lambda p: (-hnet[p], p != "conf_E2", p))[0]
        singlec[m] = {"single_C_strength_handed_in_only": ck, "best_prompt": bp, "best_prompt_if_chosen_handed_in_only": bp_h}
        # glue check against the main table's own line (original scoring, N1)
        names = ("no write", "single C", "single E", "the strong form", "prompt only", "the best prompt on tune")
        line = "    N1 (prompt rule) minus: " + " | ".join("%s %s" % (n, ps.ci([a - b for a, b in zip(n1v, orig[n])])) for n in names)
        must_contain("%s main table opponents" % m, out_main, line, rec)
        # the honest sentence on the plain prompt (x_E2, cell 0)
        if ok_ph:
            st, bl = ps.load(MP.work(MP.usual(display, 1)[1], "test", "h22c4"), "pro")
            if any(st.get((q, "baseline_x_E2"), "?") == "?" for q in ids):
                d["missing"]["x_E2"] = "no verdicts"
            else:
                d["x_E2"] = {q: (hs(val(st[(q, "baseline_x_E2")]), not bl[(q, "baseline_x_E2")]), not bl[(q, "baseline_x_E2")]) for q in ids}
                c0 = [val(st[(q, "baseline_x_E2")]) for q in ids]
                must_contain("%s x_E2 (cell 0)" % m, out_ph, "N1 minus cell 0 %s" % ps.ci([a - b for a, b in zip(n1v, c0)]), rec)
        else:
            d["missing"]["x_E2"] = "plain_honesty regression differs"
        # P1 / P2: no-write prompts selected by the model's own confidence (prompt_conf_table's functions)
        if ok_pc:
            tqs = srcs[0][1].qs
            conf = {}
            for p in PC.PROMPTS:
                if p not in tj or p not in xj:
                    continue
                ct_, _ = PC.confidence(run, "tune", p, tqs, tj[p][2])
                cx_, _ = PC.confidence(run, "test", p, ids, xj[p][2])
                if ct_ is not None and cx_ is not None:
                    conf[p] = (ct_, cx_)
            if "conf_E2" in conf:
                st2, an2, _ = xj["conf_E2"]
                o2 = PC.ranked(an2, conf["conf_E2"][1], ids)
                if len(o2) >= nM:
                    p1 = PC.keep(st2, o2[:nM])
                    d["P1"] = from_states(p1[0], p1[1], ids)
                else:
                    d["missing"]["P1"] = "conf_E2 has %d answers with a confidence, M hands in %d" % (len(o2), nM)
            else:
                d["missing"]["P1"] = "conf_E2 has no usable confidence"
            cands = []
            for p in conf:
                st_, an_, _ = tj[p]
                o_ = PC.ranked(an_, conf[p][0], tqs)
                n_, sc_ = PC.best_cut(st_, o_, [conf[p][0][q] for q in tqs])
                cands.append((p, n_, sc_, o_))
            if cands:
                p, n_, sc_, o_ = sorted(cands, key=lambda c: (-c[2], c[0] != "conf_E2", c[0]))[0]
                st_, an_, _ = xj[p]
                xo = PC.ranked(an_, conf[p][1], ids)
                if n_ == 0:
                    kept = []
                elif n_ == len(o_):
                    kept = xo
                else:
                    tau = conf[p][0][tqs[o_[n_ - 1]]]
                    kept = [i for i in xo if conf[p][1][ids[i]] >= tau]
                p2 = PC.keep(st_, kept)
                d["P2"] = from_states(p2[0], p2[1], ids)
                singlec[m]["P2"] = [p, n_, sc_]
            else:
                d["missing"]["P2"] = "no prompt with confidence files"
        else:
            d["missing"]["P1"] = d["missing"]["P2"] = "prompt_conf_table regression differs"
        per[m] = d
    rec["choices"] = singlec
    # glue check of P1 / P2 against prompt_conf_table's own line (training-rule N1 of main_table_layers)
    if ok_pc:
        for m in ORDER:
            display = A.KEY[m][0]
            mt = {}
            capture(lambda: ML.model_block(display, [1.0], mt))
            qs_, n1_, n1a, _, _, _ = mt[(1.0, "n1 states")][display]
            if list(qs_) != per[m]["ids"]:
                print("  %s: training-rule list differs from the method's list; P glue check skipped" % m)
                continue
            # recompute P1 at N1's count and P2 with the same functions, then compare with the archived string
            Dt, Dx, srcs = SRC[m]
            run = Dx.m["ec"][1].split("/eval_extra/")[0]
            tqs = srcs[0][1].qs
            tj, xj = PC.judged(run, "tune", tqs), PC.judged(run, "test", qs_)
            cx_ = PC.confidence(run, "test", "conf_E2", qs_, xj["conf_E2"][2])[0]
            o2 = PC.ranked(xj["conf_E2"][1], cx_, qs_)
            p1 = PC.keep(xj["conf_E2"][0], o2[:sum(n1a)])
            p2v = [per[m]["P2"][q][0] for q in qs_]
            s_ = "N1 minus: prompt only %s | P1 (same answer count) %s | P2 (prompt and cut chosen on tune) %s" % (
                ps.ci([val(a) - val(b) for a, b in zip(n1_, xj["conf_E2"][0])]), ps.ci([val(a) - val(b) for a, b in zip(n1_, p1[0])]),
                ps.ci([val(a) - b for a, b in zip(n1_, p2v)]))
            must_contain("%s P1 / P2" % m, out_pc, s_, rec)
    names = ["no write", "single C", "single E", "strong form", "prompt only", "best prompt on tune", "x_E2", "P1", "P2"]
    common = sorted(set.intersection(*(set(per[m]["ids"]) for m in ORDER)))

    def ci_all(n, qs):
        return tq([sum(per[m]["M"][q][0] - per[m][n][q][0] for m in ORDER) for q in qs])
    report_rows("## R1 TruthfulQA test, M minus each opponent (all handed in only), questions resampled", per, names, tq, ci_all, common, rec)
    print("\n  single-C strength (handed-in-only tune choice) / best prompt on tune (main-table rule; handed-in-only rule) / P2 choice (prompt, kept on tune, tune net):")
    for m in ORDER:
        print("  %-12s %s" % (m, singlec[m]))


# ------------------------------------------------------------------------------------------------ R2 ITI, CAA, mean difference
ITI_FILE = {"llama3.1-8b": "iti_llama", "gemma3-12b": "iti_gemma3-12b", "gemma3-27b": "iti_gemma3-27b"}


def r2(B):
    import caa_table as CT
    import posm_table as PT
    rec = DUMP.setdefault("R2", {"regression": {}})
    print("# R2 TruthfulQA test at the prompt rule: the method against ITI, CAA and the mean difference; handed in only")
    routes = json.loads(B["routes_json"])
    base = {}
    for m in ORDER:
        R = tq_routes(m)
        routes_check(rec, "TruthfulQA", m, R["M"], R["ids"], routes)
        base[m] = R

    def own_n1(m):
        display, layer, s = A.KEY[m]
        X, _ = MP.prompt_split(display, layer, 1, s)
        T, _ = MP.prompt_split(display, layer, 0, s)
        return X[3], S.system(X[0], X[1], ML.n1_pick(T[0], T[1], s))[0]

    def block(kind, models, cells_of, n1_line):
        per = {}
        for m in models:
            qs, cells, text = cells_of(m)
            ids = base[m]["ids"]
            if set(ids) - set(qs):
                raise SystemExit("%s %s: the method's questions are not all in the opponent's list" % (kind, m))
            q1, n1s = own_n1(m)
            if q1 != qs:
                raise SystemExit("%s %s: question lists differ" % (kind, m))
            must_contain("%s %s" % (m, kind), text, n1_line % ps.ci([val(a) - val(cells[q][0]) for q, a in zip(qs, n1s)]), rec)
            d = {"ids": ids, "M": base[m]["M"], kind: {q: (hs(val(cells[q][0]), cells[q][1]), cells[q][1]) for q in ids}}
            if any(cells[q][1] and cells[q][0] == "?" for q in ids):
                raise SystemExit("%s %s: an answered cell without a verdict" % (kind, m))
            # equal-count bounds as caa_table.equal_count (both keep min(A, B) of their answers, the rest 0)
            st_m = ["R" if base[m]["M"][q][0] == 1 else "W" if base[m]["M"][q][0] == -1 else "X" for q in ids]
            fl_m = [base[m]["M"][q][1] for q in ids]
            st_o = [cells[q][0] if cells[q][1] else "X" for q in ids]
            fl_o = [cells[q][1] for q in ids]
            k = min(sum(fl_m), sum(fl_o))
            sl, sh = CT.retained_bounds(st_m, fl_m, k)
            cl, chi = CT.retained_bounds(st_o, fl_o, k)
            d["equal_count"] = {"kept": k, "low": sl - chi, "high": sh - cl, "M_answers": sum(fl_m), "opp_answers": sum(fl_o)}
            per[m] = d
        common = sorted(set.intersection(*(set(per[m]["ids"]) for m in per))) if len(per) == len(ORDER) else None

        def ci_all(n, qs):
            return tq([sum(per[m]["M"][q][0] - per[m][n][q][0] for m in ORDER) for q in qs])
        out = report_rows("## R2 M minus %s (handed in only), questions resampled" % kind, per, [kind], tq, ci_all, common, rec)
        for m in per:
            e = per[m]["equal_count"]
            out[m]["equal_count"] = e
            print("  %-12s M minus %s with both keeping %d answers: between %+d and %+d (M %d answers, %s %d)" % (
                m, kind, e["kept"], e["low"], e["high"], e["M_answers"], kind, e["opp_answers"]))
    # ITI (faithful_seed42, sigma endq, choice within the MMLU budget)
    texts_iti, texts_caa = {}, {}
    ok_iti = []
    for m in ("llama3.1-8b", "gemma3-12b", "gemma3-27b"):
        t, code = capture(lambda: A.test_table(m, "faithful_seed42", "endq", False))
        texts_iti[m] = t
        if compare("iti_table.py test %s faithful_seed42 vs %s_faithful_seed42_test_20261122.txt" % (m, ITI_FILE[m]), t, B["iti_" + m], rec["regression"]):
            ok_iti.append(m)

    def iti_cells(m):
        display, layer, s = A.KEY[m]
        fit = "faithful_seed42"
        ch = A.read("%s/%s/%s/choice.json" % (A.BASE, m, fit))
        if ch["model"] != m or ch["fit"] != fit or ch["sigma"] != "endq" or ch["budget"] is None:
            raise SystemExit("ITI choice file of %s" % m)
        X, _ = MP.prompt_split(display, layer, 1, s)
        w = MP.work(MP.usual(display, 1)[1], "test", "iti_%s" % fit)
        cond, g = A.condition_of(m, fit, "test", ch["stem"])
        st, bl = ps.load(w, "pro")
        return X[3], {q: (st[(q, cond)], not bl[(q, cond)]) for q in X[3]}, texts_iti[m]
    if ok_iti:
        block("ITI", ok_iti, iti_cells, "N1 minus ITI %s")
    if len(ok_iti) != 3:
        print("ITI: models whose regression differs are not read: %s" % sorted(set(ITI_FILE) - set(ok_iti)))
    # CAA (choice within the MMLU budget)
    ok_caa = []
    for m in ORDER:
        t, code = capture(lambda: CT.test_table(m, False))
        texts_caa[m] = t
        if compare("caa_table.py test %s vs caa_%s_test_*.txt" % (m, m), t, B["caa_" + m], rec["regression"]):
            ok_caa.append(m)

    def caa_cells(m):
        display, layer, s = A.KEY[m]
        mi = CT.Mirror(m)
        ch = CT.read_choice(mi, False)
        X, _ = MP.prompt_split(display, layer, 1, s)
        w = MP.work(mi.run, "test", "caa_main")
        cond, g = mi.generation("test", ch["layer"], ch["mult"])
        CT.verify_work(mi, w)
        st, bl = ps.load(w, "pro")
        return X[3], {q: (st[(q, cond)], not bl[(q, cond)]) for q in X[3]}, texts_caa[m]
    if ok_caa:
        block("CAA", ok_caa, caa_cells, "N1 minus CAA %s")
    if len(ok_caa) != len(ORDER):
        print("CAA: models whose regression differs are not read: %s" % sorted(set(ORDER) - set(ok_caa)))
    # mean difference at the same layer and write amount (posm_table)
    t_md, code = capture(PT.main)
    if compare("posm_table.py vs posm_five_models_20261122.txt", t_md, B["posm"], rec["regression"]):
        def md_cells(m):
            display, layer, s = A.KEY[m]
            X, _ = MP.prompt_split(display, layer, 1, s)
            D, run, utx = MP.usual(display, 1)
            st, bl = ps.load(MP.work(run, "test", "pospromptml%d" % layer), "pro")
            cond = "E_L%d_x-mdg-p%s" % (layer, s)
            return X[3], {q: (st[(q, cond)], not bl[(q, cond)]) for q in X[3]}, t_md
        block("mean difference", list(ORDER), md_cells, "N1 minus mean difference %s")
    else:
        print("mean difference: STOPPED, the posm table does not reproduce")


# ------------------------------------------------------------------------------------------------ R3 GSM8K
def r3(B):
    import gsm8k_n1_prompt as GK
    import cap_score as CS
    rec = DUMP.setdefault("R3", {"regression": {}})
    print("# R3 GSM8K 1,319 items at the prompt rule: the method's accuracy and its change against no write (handed in only)")
    t, code = capture(GK.table)
    if not compare("gsm8k_n1_prompt.py table vs gsm8k_n1_five_models_20261122.txt", t, B["gsm8k"], rec["regression"]):
        print("R3 STOPPED: the GSM8K table does not reproduce")
        return
    gold = CS.gold_of("gsm8k")
    order = GK.frozen_ids()
    VALG = {"R": 1, "W": -1, "X": 0}
    tot, tot_e, out = {}, {}, {}
    print()
    for key, (el, s, cl, cap_e, cap_own, cap) in GK.SET.items():
        c = CM[key]
        bc, base = GK.load_arm(key, "neutral", None, order)
        ec, e = GK.load_arm(key, GK.e_arm(key), "prompt", order)
        same = ("identity_sha256", "frozen_file_sha256", "model_config", "system_text", "user_template")
        if any(bc[k] != ec[k] for k in same):
            raise SystemExit("%s: base and E belong to different runs" % key)
        b = [i for i in order if GK.blank(e[i])]
        re = {}
        if b:
            subset = Path(GK.root(key)) / "n1_blank_ids_at-prompt.json"
            if json.loads(subset.read_text(encoding="utf-8")) != b:
                raise SystemExit("%s: saved re-ask IDs differ from E blanks" % key)
            for k in GK.LAD:
                arm_name = "neutral:E%d.w.p%s+C%d.w.m%s" % (el, s, cl, k)
                try:
                    cc, re[k] = GK.load_arm(key, arm_name, "prompt", b, subset)
                except FileNotFoundError:
                    continue
                parts = cc["parts"]
                if (any(cc[x] != ec[x] for x in (*same, "engine_sha256")) or len(parts) != 2 or parts[0] != ec["parts"][0]
                        or parts[1]["family"] != "C" or parts[1]["kind"] != "w" or parts[1]["layer"] != cl or parts[1]["beta"] != -float(k)):
                    raise SystemExit("%s %s: foreign re-ask direction or run" % (key, arm_name))

        def cell(i, row):
            return (VALG[CS.judge_row("gsm8k", row, gold(i))[0]], not GK.blank(row))
        arm = {"O": {i: cell(i, base[i]) for i in order}, "E": {i: cell(i, e[i]) for i in order}}
        for k in RT.KS:
            arm["EC" + k] = {i: (cell(i, re[k][i]) if (k in re and i in re[k]) else PH0) for i in order}
            arm["C" + k] = {i: PH0 for i in order}
        bb = [i for i in b if GK.blank(base[i])]
        eb_only = [i for i in b if not GK.blank(base[i])]
        lack = [i for i in bb if not (c in re and i in re[c])]
        Md = method_of(order, arm, key)
        for i in order:            # the cells R1_c reads are real ones
            if not arm["E"][i][1] and not arm["O"][i][1] and i not in lack and Md[i] != RT.sc(arm["EC" + c][i]):
                raise SystemExit("%s %s: the method's cell is not the E + C cell" % (key, i))
        n = len(order)
        right = {name: [int(d[i][0] == 1 and d[i][1]) for i in order] for name, d in (("no write", arm["O"]), ("single E", arm["E"]), ("M", Md))}
        acc = {name: 100.0 * sum(v) / n for name, v in right.items()}
        dm = CS.ci([a - b for a, b in zip(right["M"], right["no write"])])
        de = CS.ci([a - b for a, b in zip(right["single E"], right["no write"])])
        dme = CS.ci([a - b for a, b in zip(right["M"], right["single E"])])
        notes = []
        for i in bb:
            if i in lack:
                notes.append("%s (both blank): NO E + C answer at C %s in the files (listed, not generated)" % (i, c))
            else:
                notes.append("%s (both blank): E + C at %s %s" % (i, c, "answered, " + CS.judge_row("gsm8k", re[c][i], gold(i))[0] if arm["EC" + c][i][1] else "blank again"))
        for i in eb_only:
            notes.append("%s (E blank, no write answered): nothing handed in" % i)
        out[key] = {"acc": acc, "M-no write": [100 * x for x in dm], "single E-no write": [100 * x for x in de], "M-single E": [100 * x for x in dme],
                    "blank_under_E": len(b), "both_blank": bb, "E_blank_no_write_answered": eb_only, "lack_EC_at_c": lack, "c": c}
        print("## %s (E%d at %s, C%d, method C strength %s): no write %.1f%% | single E %.1f%% | M %.1f%% | M - no write %+.1f [%+.1f, %+.1f] | single E - no write %+.1f [%+.1f, %+.1f] | M - single E %+.1f [%+.1f, %+.1f] points | blank under E %d" % (
            key, el, s, cl, c, acc["no write"], acc["single E"], acc["M"], 100 * dm[0], 100 * dm[1], 100 * dm[2], 100 * de[0], 100 * de[1], 100 * de[2],
            100 * dme[0], 100 * dme[1], 100 * dme[2], len(b)))
        for x in notes:
            print("    " + x)
        tot[key] = [a - b for a, b in zip(right["M"], right["no write"])]
        tot_e[key] = [a - b for a, b in zip(right["single E"], right["no write"])]
    keys = list(GK.SET)
    mm = CS.ci([sum(tot[k][j] for k in keys) / len(keys) for j in range(len(order))])
    me = CS.ci([sum(tot_e[k][j] for k in keys) / len(keys) for j in range(len(order))])
    out["five models (reference only)"] = {"M-no write": [100 * x for x in mm], "single E-no write": [100 * x for x in me]}
    print("\n## five models (reference only), average change against no write in points (items resampled, the models together): M %+.2f [%+.2f, %+.2f] | single E %+.2f [%+.2f, %+.2f]" % (
        100 * mm[0], 100 * mm[1], 100 * mm[2], 100 * me[0], 100 * me[1], 100 * me[2]))
    rec["table"] = out


# ------------------------------------------------------------------------------------------------ R4 (QA)^2
FULLB = "Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl"
TUNEIDS = "Iteration_01/newsets/splits/qaqa_pilot_20261122.json"


def r4(B):
    import qaqa_pilot_table as Q
    rec = DUMP.setdefault("R4", {"regression": {}})
    print("# R4 (QA)^2 at the frozen TruthfulQA settings (write rule prompt): the method against no write, single E, single C, prompt only and the strong form; handed in only")
    base = ["qaqa_pilot_table.py", "--task", "qaqafull", "--bank", FULLB]
    runs = (("qaqa_570", base + ["--work", "test_full570"], "qaqa_frozen_570_five_models_20261122.txt"),
            ("qaqa_470", ["qaqa_pilot_table.py", "--skip-ids", TUNEIDS] + base[1:] + ["--work", "test_full570"], "qaqa_frozen_470_five_models_20261122.txt"),
            ("qaqa_570_strong", base + ["--work", "test_full570,test_full570_strong"], "qaqa_frozen_570_strong_five_models_20261122.txt"),
            ("qaqa_470_strong", ["qaqa_pilot_table.py", "--skip-ids", TUNEIDS] + base[1:] + ["--work", "test_full570,test_full570_strong"], "qaqa_frozen_470_strong_five_models_20261122.txt"))
    text, ok = {}, {}
    for key, argv, fname in runs:
        label = "qaqa_pilot_table.py %s vs %s" % (" ".join(argv[1:]), fname)
        if key.endswith("_strong"):
            text[key], code = capture(Q.main, argv + list(ORDER))
            ok[key] = compare(label, text[key], B[key], rec["regression"])
            continue
        # The frozen tables were archived at 02:4x on 2026-11-22, before the strong-form passes and their confidence
        # files existed (05 h). Since then qaqa_pilot_table.Raw.confidence prints one "note: ... comparator is left out"
        # line per such file on stderr. Their stdout is compared line by line with the archive; stderr is listed and
        # may hold nothing but those notes.
        out_, err_, code = capture_split(Q.main, argv + list(ORDER))
        text[key] = out_
        eq_ = compare(label + " (stdout)", out_, B[key], rec["regression"])
        notes = [l for l in err_.splitlines() if l.strip()]
        only = all(l.startswith("note: ") and l.endswith("that comparator is left out") for l in notes)
        rec["regression"][label + " (stdout)"]["stderr_lines"] = notes
        print("  stderr of that run (%d lines, %s):" % (len(notes), "only the notes about the later confidence files" if only else "OTHER LINES PRESENT"))
        for l in notes:
            print("    " + l)
        ok[key] = eq_ and only
    if not ok["qaqa_570"] or not ok["qaqa_470"]:
        print("R4 STOPPED: the frozen (QA)^2 tables do not reproduce")
        return
    skip = set(json.load(open(TUNEIDS, encoding="utf-8"))["ids"])
    per = {"570": {}, "470": {}}
    placeholders = {}
    for m in ORDER:
        el, s, cl, cap_e, cap_own, kc = Q.SET[m]
        c = CM[m]
        R = Q.Raw(m, "qaqafull", FULLB, ["test_full570"])
        ids = list(R.ids)
        ids470 = [q for q in ids if q not in skip]

        def cells(RR, a):
            RR.need(a)
            return {q: (RR.cell(a, q), not RR.blank(a, q)) for q in ids}
        arm = {"O": cells(R, "x"), "E": cells(R, R.e(s))}
        ph = []
        for k in RT.KS:
            for nm, a in (("EC" + k, R.ec(s, k)), ("C" + k, R.c(k))):
                if R.has(a):
                    arm[nm] = cells(R, a)
                else:
                    arm[nm] = {q: PH0 for q in ids}
                    ph.append(nm)
        if "EC" + c in ph:
            raise SystemExit("%s: E + C at the method's strength %s has no judged pass" % (m, c))
        placeholders[m] = ph
        Md = method_of(ids, arm, m)
        # glue check against the archived frozen table (original scoring, N1)
        n1v, n1a, _ = R.n1_system(s, cap_e, cap_own)
        orig = {"single E": R.arm_system(R.e(s))[0], "no write": R.arm_system("x")[0], "single C": R.arm_system(R.c(kc))[0],
                "prompt only": R.arm_system("conf_E2")[0]}
        line = " | ".join("N1 minus %s %s" % (n, S.ps.ci([n1v[q] - orig[n][q] for q in ids])) for n in ("single E", "no write", "single C", "prompt only"))
        must_contain("%s (QA)^2 570 opponents" % m, text["qaqa_570"], line, rec)
        opp = {"no write": arm["O"], "single E": arm["E"], "single C": cells(R, R.c(kc)), "prompt only": cells(R, "conf_E2")}
        opp = {n: {q: (hs(v[q][0], v[q][1]), v[q][1]) for q in ids} for n, v in opp.items()}
        strong = {}
        if ok["qaqa_570_strong"] and ok["qaqa_470_strong"]:
            Rs = Q.Raw(m, "qaqafull", FULLB, ["test_full570", "test_full570_strong"])
            sa = "conf:E%d.w.p%s" % (el, s)
            cf = Rs.confidence("qaqafull", sa)
            if cf is None:
                strong = {"570": None, "470": None}
            else:
                n1n = sum(bool(n1a[q]) for q in ids)
                ref = Rs.top_conf(ids, cf, n1n, sa)
                must_contain("%s (QA)^2 570 strong form" % m, text["qaqa_570_strong"],
                             "N1 minus must answer + E + confidence, N1's N %s" % S.ps.ci([n1v[q] - ref[0][q] for q in ids]), rec)
                for lab, qs in (("570", ids), ("470", ids470)):
                    buf = io.StringIO()
                    with contextlib.redirect_stderr(buf):
                        sel = Rs.top_conf(qs, cf, sum(Md[q][1] for q in qs), sa)
                    strong[lab] = None if sel is None else {q: (sel[0][q], sel[1][q]) for q in ids}
                    if sel is None:
                        strong[lab + "_why"] = buf.getvalue().strip()
        for lab, qs in (("570", ids), ("470", ids470)):
            d = {"ids": qs, "M": Md, "missing": {}, "qtype": R.qtype}
            d.update(opp)
            if not strong:
                d["missing"]["strong form"] = "the strong tables do not reproduce"
            elif strong.get(lab) is None:
                d["missing"]["strong form"] = strong.get(lab + "_why") or "no confidence file"
            else:
                d["strong form"] = strong[lab]
            per[lab][m] = d
    rec["placeholder_arms"] = placeholders
    names = ["no write", "single E", "single C", "prompt only", "strong form"]
    for lab in ("570", "470"):
        for t, tl in (("all", "all types"),) + Q.TYPES:
            sub = {}
            for m in ORDER:
                d = dict(per[lab][m])
                d["ids"] = [q for q in d["ids"] if t == "all" or d["qtype"][q] == t]
                sub[m] = d
            common = sub[ORDER[0]]["ids"]
            if any(sub[m]["ids"] != common for m in ORDER):
                raise SystemExit("(QA)^2: the models do not have the same questions")

            def ci_all(n, qs, sub=sub):
                per_m = [[sub[m]["M"][q][0] - sub[m][n][q][0] for q in qs] for m in ORDER]
                return [int(x) for x in S.joint_ci(per_m)]
            report_rows("## R4 (QA)^2 %s questions%s, %s: M minus each opponent (handed in only)" % (
                lab, "" if lab == "570" else " (without the 100 tune questions)", tl), sub, names, tq, ci_all, common, rec)


# ------------------------------------------------------------------------------------------------ R5 FalseQA held-out 600
def r5(B):
    rec = DUMP.setdefault("R5", {"regression": {}})
    print("# R5 FalseQA held-out 600 at the reused settings: N1 replaced by the method in the readings of falseqa_read.py table; handed in only")
    t, code = capture(lambda: F.run_table(list(ORDER)))
    if not compare("falseqa_read.py table vs falseqa_heldout_table_20261124.txt", t, B["falseqa"], rec["regression"]):
        print("R5 STOPPED: the FalseQA table does not reproduce")
        return
    routes = json.loads(B["routes_json"])
    ok, rows = capture_ret(F.check_bank)
    if not ok:
        raise SystemExit("FalseQA bank check failed")
    dev = {r["id"] for r in rows if r["set"] == "dev"}
    held = {r["id"] for r in rows if r["set"] == "heldout"}
    qt = {r["id"]: r["question"] for r in rows}
    pair_of = {r["id"]: r["pair_id"] for r in rows}
    prem = {r["id"]: r["premise"] for r in rows}
    data, Mg = {}, {}
    for m in ORDER:
        st = F.setting(m)
        s = st[1]
        c = CM[m]
        ids, arm, n1 = RT.fqa_split(m, rows, "heldout")
        Md = method_of(ids, arm, m)
        routes_check(rec, "FalseQA", m, Md, ids, routes)
        R, R2 = F.load_rows(m, dev)
        caps, _ = F.choose(R, R2, st)
        H, H2 = F.load_rows(m, held)
        sysd = F.systems_of(H, H2, st, caps, H.ids)
        if list(H.ids) != ids:
            raise SystemExit("FalseQA held-out order")
        # the method's cells with the text handed in (for the self-judge margins of the equal-count readings)
        e, ecc = H.e(s), H.ec(s, c)
        mc = {}
        for q in ids:
            if not H.blank(e, q):
                mc[q] = (H.cell(e, q), True, H.text(e, q), "0")
            elif not H.blank("x", q):
                mc[q] = (0, False, H.text(e, q), None)
            else:
                a = not H.blank(ecc, q)
                mc[q] = (hs(H.cell(ecc, q), a), a, H.text(ecc, q), c if a else None)
            if (mc[q][0], mc[q][1]) != Md[q]:
                raise SystemExit("%s %s: the method's cell differs from routes_1128" % (m, q))
        d = {"M": mc}
        for n in ("O", "E", "N1", "Conly", "S0"):
            d[n] = {q: (hs(x[0], x[1]),) + tuple(x[1:]) for q, x in sysd[n].items()}
            d[n + "_orig"] = sysd[n]
        data[m] = d
        Mg[m] = F.margins_of(m)
    ids = [r["id"] for r in rows if r["set"] == "heldout"]
    groups = [("held-out, premise false", [q for q in ids if prem[q] == "false"]),
              ("held-out, premise valid", [q for q in ids if prem[q] == "valid"]), ("held-out, all", ids)]
    rows_ = [(m, (m,)) for m in ORDER] + [("5 models (reference only)", tuple(ORDER))]

    def nat(ms, qs, a, b):
        return [int(x) for x in F.pair_ci({q: sum(data[m][a][q][0] - data[m][b][q][0] for m in ms) for q in qs}, pair_of, qs)]

    def eq(ms, qs, a, b):
        de, cnt, miss = {q: 0 for q in qs}, [], 0
        for m in ms:
            dd, cc, mm = F.equal_diff(qs, list(qs), data[m][a], data[m][b], qt, Mg[m])
            cnt.append(cc)
            miss += mm
            if dd is not None:
                for q in qs:
                    de[q] += dd[q]
        if miss:
            return {"missing_margins": miss}
        return {"ci": [int(x) for x in F.pair_ci(de, pair_of, qs)], "pool": [sum(x[0] for x in cnt), sum(x[1] for x in cnt), sum(min(x) for x in cnt)]}

    def show_eq(r):
        if "ci" not in r:
            return "n/a (%d margins missing)" % r["missing_margins"]
        return "%s (pool answers %d vs %d -> %d each)" % (fmt(r["ci"]), *r["pool"])
    # glue check: the original N1 lines (original scoring) rebuilt from the cells read here
    for m in ORDER:
        qs = ids
        o = {n: data[m][n + "_orig"] for n in ("N1", "S0", "E", "Conly")}
        save = {n: data[m][n] for n in o}
        for n in o:
            data[m][n] = o[n]
        line = "  %-12s natural %s | equal answers %s" % (m, fmt(nat((m,), qs, "N1", "S0")), show_eq(eq((m,), qs, "N1", "S0")))
        line4 = "  %-12s N1 - E %s | N1 - Conly %s || natural N1 - E %s, N1 - Conly %s" % (
            m, show_eq(eq((m,), qs, "N1", "E")), show_eq(eq((m,), qs, "N1", "Conly")), fmt(nat((m,), qs, "N1", "E")), fmt(nat((m,), qs, "N1", "Conly")))
        for n in save:
            data[m][n] = save[n]
        must_contain("%s FalseQA N1 - S0 (all 600)" % m, t, line, rec)
        must_contain("%s FalseQA N1 - E / Conly (all 600)" % m, t, line4, rec)
    out = {}
    for label, qs in groups:
        print("\n" + "=" * 100 + "\n## %s (%d questions per model, %d pairs); M = the method, opponents handed in only" % (label, len(qs), len({pair_of[q] for q in qs})))
        g = {}
        for lab, ms in rows_:
            r = {"M": [sum(data[m]["M"][q][0] for m in ms for q in qs), sum(data[m]["M"][q][1] for m in ms for q in qs)]}
            for n in ("O", "E", "N1", "Conly", "S0"):
                r[n] = [sum(data[m][n][q][0] for m in ms for q in qs), sum(bool(data[m][n][q][1]) for m in ms for q in qs)]
            r["M-S0 natural"], r["M-S0 equal"] = nat(ms, qs, "M", "S0"), eq(ms, qs, "M", "S0")
            r["M-E natural"], r["M-E equal"] = nat(ms, qs, "M", "E"), eq(ms, qs, "M", "E")
            r["M-Conly natural"], r["M-Conly equal"] = nat(ms, qs, "M", "Conly"), eq(ms, qs, "M", "Conly")
            r["M-N1 natural"] = nat(ms, qs, "M", "N1")
            if label.startswith("held-out, premise valid"):
                r["M-O natural"] = nat(ms, qs, "M", "O")
            g[lab] = r
            print("  %-26s net (answers): M %+d (%d) | %s" % (lab, r["M"][0], r["M"][1], " | ".join("%s %+d (%d)" % (n, r[n][0], r[n][1]) for n in ("O", "E", "N1", "Conly", "S0"))))
            print("  %-26s M - S0 natural %s, equal answers %s" % ("", fmt(r["M-S0 natural"]), show_eq(r["M-S0 equal"])))
            print("  %-26s M - E natural %s, equal %s | M - Conly natural %s, equal %s | M - N1 natural %s%s" % (
                "", fmt(r["M-E natural"]), show_eq(r["M-E equal"]), fmt(r["M-Conly natural"]), show_eq(r["M-Conly equal"]), fmt(r["M-N1 natural"]),
                (" | valid premise: M - O natural %s" % fmt(r["M-O natural"])) if "M-O natural" in r else ""))
        out[label] = g
    rec["table"] = out


def capture_ret(fn):
    """fn()'s return value, with what it prints discarded."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else ""
    if part not in ("R1", "R2", "R3", "R4", "R5"):
        raise SystemExit("usage: redo_1128.py R1|R2|R3|R4|R5 < bundle.json")
    bundle = json.loads(sys.stdin.read())
    {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5}[part](bundle)
    print("JSON " + json.dumps(DUMP, sort_keys=True))


if __name__ == "__main__":
    main()
