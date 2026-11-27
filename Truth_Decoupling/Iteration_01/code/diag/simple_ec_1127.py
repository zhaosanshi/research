#!/usr/bin/env python3
# Iteration_01 exploration (main session, Claude Opus 5.5, 2026-11-27). Zhao Lei 23:50: N1 is too complicated; on TruthfulQA
# and FalseQA, for the five models, without N1's steps, does E+C answer better than single E and single C?
# Read-only: existing answers and DeepSeek Pro verdicts only; no model, no judge call. Not reviewed by GPT.
# Every system is ONE pass, every question written the same way (write rule "prompt", the main-table E layer, strength
# and C layer of each model):
#   O      no write
#   E      x:E<el>.w.p<s>                     (s = the main strength of the model)
#   C(c)   x:C<cl>.w.m<c>                     (C toward answering, c = 0.25 0.5 1 2)
#   EC(c)  x:E<el>.w.p<s>+C<cl>.w.m<c>        (both written in the same pass)
# Score per question (Pro): right +1, no answer or hedge 0, wrong -1. Natural answer counts (no matching).
# One C strength per model and system, chosen on the tuning questions (TruthfulQA tune 192; FalseQA development 100) as the
# c with the highest net (ties -> the smaller c): c_EC for EC, c_C for C. Then read once on the test questions (TruthfulQA
# test main list; FalseQA held-out 600). Also printed: every c on the test questions (EC(c) - C(c) at the same c, the 2x2).
# Intervals: TruthfulQA questions resampled (gate_sim_1126.tq_ci), FalseQA pairs resampled (falseqa_read.pair_ci).
# Added: where EC - E comes from (questions single E answers / leaves blank, by the method's text rule), and the two-step
# system "E; where E is blank by the rule, the one-pass E+C(c) answer" (c chosen on the tuning questions the same way),
# against E, C and the current N1 (TruthfulQA: S.system with ML.n1_pick on tune; FalseQA: falseqa_read.n1_cells with the
# caps of falseqa_read.choose on the development questions; known N1 nets checked).
# usage (container, from /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/simple_ec_1127.py
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/gpt_tools_20261126/gate_audit"]
import iti_table as A             # noqa: E402
import gate_sim_1126 as GS        # noqa: E402
import falseqa_read as F          # noqa: E402

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
SHOW = {"llama3.1-8b": "Llama-3.1-8B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.8-27b": "Qwen3.8-27B", "gemma3-12b": "Gemma-3-12B",
        "gemma3-27b": "Gemma-3-27B"}
VAL = {"R": 1, "W": -1, "X": 0, "H": 0}
DOT = {"0-25": "0.25", "0-5": "0.5", "1": "1", "2": "2", "0.25": "0.25", "0.5": "0.5"}


def pick(nets, lad):
    """the c with the highest net; ties -> the smaller c"""
    return max(lad, key=lambda k: (nets[k], -lad.index(k)))


def hrw(vals):
    return "%d / %d" % (sum(v == 1 for v in vals), sum(v == -1 for v in vals))


# ------------------------------------------------------------------------------------------------- TruthfulQA
def tqa_split(m, idx):
    display, layer, s = A.KEY[m]
    sp, tx, singles, ids = MP.prompt_split(display, layer, idx, s)[0]
    if idx == 1:
        src = [x for x in ML.sources(display)[2] if x[0] == layer]
        cx = [q for q in ids if q in set(src[0][3].qs)]
        if cx != ids:
            sp, tx, singles, ids = MP.prompt_split(display, layer, idx, s, qs=cx)[0]
    lad = list(ps.LAD)
    out = {"O": [VAL[sp.st[(q, "baseline_x")]] for q in ids], "E": [VAL[sp.st[(q, sp.cE(s))]] for q in ids]}
    for k in lad:
        out["EC" + k] = [VAL[sp.st[(q, sp.cEC(s, k))]] for q in ids]
        out["C" + k] = [VAL[x] for x in singles[k][0]]
        if len(out["C" + k]) != len(ids):
            raise SystemExit("%s: single C %s has %d cells for %d questions" % (m, k, len(out["C" + k]), len(ids)))
    out["_eblank"] = {q: bool(sp.blank[(q, sp.cE(s))]) for q in ids}
    out["_oblank"] = {q: bool(sp.blank[(q, "baseline_x")]) for q in ids}
    out["_n1"] = None
    if idx == 1:
        tune = MP.prompt_split(display, layer, 0, s)[0]
        pk = ML.n1_pick(tune[0], tune[1], s)
        out["_n1"] = [VAL[x] for x in S.system(sp, tx, pk)[0]]
    return ids, out, lad, s


def tqa_known(m, test):
    ids, out, lad, s = test
    got = (len(ids), sum(out["O"]), sum(out["E"]))
    if got != GS.K_TQ[m][:3]:
        raise SystemExit("%s: TruthfulQA known nets %s, reference %s" % (m, got, GS.K_TQ[m][:3]))
    return got


# ------------------------------------------------------------------------------------------------- FalseQA
def fqa_rows():
    ok, rows = F.check_bank()
    if not ok:
        raise SystemExit("FalseQA bank check FAILED")
    return rows


def fqa_split(m, rows, which):
    keep = {r["id"] for r in rows if r["set"] == which}
    R, _ = F.load_rows(m, keep)
    st = F.setting(m)
    s = st[1]
    lad = list(F.LAD)
    ids = list(R.ids)
    out = {"O": [R.cell("x", q) for q in ids], "E": [R.cell(R.e(s), q) for q in ids]}
    for k in lad:
        out["EC" + k] = [R.cell(R.ec(s, k), q) for q in ids]
        out["C" + k] = [R.cell(R.c(k), q) for q in ids]
    out["_eblank"] = {q: bool(R.blank(R.e(s), q)) for q in ids}
    out["_oblank"] = {q: bool(R.blank("x", q)) for q in ids}
    out["_n1"] = None
    if which == "heldout":
        dev = {r["id"] for r in rows if r["set"] == "dev"}
        tr, tr2 = F.load_rows(m, dev)
        caps, _ = F.choose(tr, tr2, st)
        n1 = F.n1_cells(R, s, caps["N1"], ids)
        out["_n1"] = [n1[q][0] for q in ids]
    return ids, out, lad, s


def where(ids, ec, e, eblank):
    """EC - E split by single E's answer: answered by the rule / blank by the rule"""
    a = [i for i, q in enumerate(ids) if not eblank[q]]
    b = [i for i, q in enumerate(ids) if eblank[q]]
    f = lambda ix: (len(ix), sum(ec[i] - e[i] for i in ix), sum(e[i] for i in ix), sum(ec[i] for i in ix))  # noqa: E731
    return f(a), f(b)


def two_step(ids, e, ec, eblank):
    """E; where E is blank by the rule, the one-pass E+C answer instead"""
    return [ec[i] if eblank[q] else e[i] for i, q in enumerate(ids)]


def main():
    rows = fqa_rows()
    pair = {r["id"]: r["pair_id"] for r in rows}
    summary = []
    global summary_b
    summary_b = []
    for task in ("TruthfulQA", "FalseQA"):
        print("\n" + "=" * 120)
        print("## %s: tuning set %s, test set %s; one pass each; nets = right - wrong" % (
            task, "tune 192" if task == "TruthfulQA" else "development 100", "test main list" if task == "TruthfulQA" else "held-out 600"))
        for m in ORDER:
            if task == "TruthfulQA":
                tune, test = tqa_split(m, 0), tqa_split(m, 1)
                known = tqa_known(m, test)

                def ci(a, b, ids):
                    return GS.tq_ci([x - y for x, y in zip(a, b)])
            else:
                tune, test = fqa_split(m, rows, "dev"), fqa_split(m, rows, "heldout")
                known = (len(test[0]), sum(test[1]["O"]), sum(test[1]["E"]))
                if known[1:] != GS.K_FQA[m][:2]:
                    raise SystemExit("%s: FalseQA known nets %s, reference %s" % (m, known, GS.K_FQA[m][:2]))

                def ci(a, b, ids):
                    return F.pair_ci({q: x - y for q, x, y in zip(ids, a, b)}, pair, ids)
            tids, tout, lad, s = tune
            ids, out, _, _ = test
            tn = {k: sum(v) for k, v in tout.items() if not k.startswith("_")}
            c_ec = pick({k: tn["EC" + k] for k in lad}, lad)
            c_c = pick({k: tn["C" + k] for k in lad}, lad)
            ec, cc, e, o = out["EC" + c_ec], out["C" + c_c], out["E"], out["O"]
            print("\n### %s (%s): %d test questions; E strength %s; known nets (n, O, E) %s reproduced" % (SHOW[m], m, len(ids), s, known))
            print("   tuning nets: O %+d | E %+d | C %s | EC %s -> c for EC %s, c for C %s" % (
                tn["O"], tn["E"], " ".join("%s:%+d" % (DOT[k], tn["C" + k]) for k in lad),
                " ".join("%s:%+d" % (DOT[k], tn["EC" + k]) for k in lad), DOT[c_ec], DOT[c_c]))
            print("   test nets (right / wrong): O %+d (%s) | E %+d (%s) | C(%s) %+d (%s) | EC(%s) %+d (%s)" % (
                sum(o), hrw(o), sum(e), hrw(e), DOT[c_c], sum(cc), hrw(cc), DOT[c_ec], sum(ec), hrw(ec)))
            r_e, r_c, r_o = ci(ec, e, ids), ci(ec, cc, ids), ci(ec, o, ids)
            print("   EC(%s) - E %s | EC(%s) - C(%s) %s | EC(%s) - O %s" % (
                DOT[c_ec], F.fmt_ci(r_e), DOT[c_ec], DOT[c_c], F.fmt_ci(r_c), DOT[c_ec], F.fmt_ci(r_o)))
            print("   every c on the test questions: " + " | ".join(
                "c %s: C %+d, EC %+d, EC - C %s, EC - E %s" % (DOT[k], sum(out["C" + k]), sum(out["EC" + k]),
                                                               F.fmt_ci(ci(out["EC" + k], out["C" + k], ids)), F.fmt_ci(ci(out["EC" + k], e, ids)))
                for k in lad))
            (na, da, ea, eca), (nb, db, eb, ecb) = where(ids, ec, e, out["_eblank"])
            print("   where EC(%s) - E comes from (single E's answer by the rule): E answered %d questions: %+d (E %+d -> EC %+d) | "
                  "E blank %d questions: %+d (E %+d -> EC %+d)" % (DOT[c_ec], na, da, ea, eca, nb, db, eb, ecb))
            # two steps: E; where E is blank, E+C(c) instead; c chosen on the tuning questions the same way
            t2 = {k: sum(two_step(tids, tout["E"], tout["EC" + k], tout["_eblank"])) for k in lad}
            c_2 = pick(t2, lad)
            s2 = two_step(ids, e, out["EC" + c_2], out["_eblank"])
            n1 = out["_n1"]
            if (task == "TruthfulQA" and sum(n1) != GS.K_TQ[m][3]) or (task == "FalseQA" and sum(n1) != GS.K_FQA[m][2]):
                raise SystemExit("%s %s: N1 net %+d differs from the known one" % (task, m, sum(n1)))
            r2e, r2c, r2n = ci(s2, e, ids), ci(s2, cc, ids), ci(s2, n1, ids)
            print("   two steps (E; E blank -> E+C(%s), c chosen on tuning: %s): net %+d (%s) | - E %s | - C(%s) %s | - N1 (%+d) %s" % (
                DOT[c_2], " ".join("%s:%+d" % (DOT[k], t2[k]) for k in lad), sum(s2), hrw(s2), F.fmt_ci(r2e), DOT[c_c], F.fmt_ci(r2c),
                sum(n1), F.fmt_ci(r2n)))
            rne, rnc = ci(n1, e, ids), ci(n1, cc, ids)
            print("   reference N1 %+d: - E %s | - C(%s) %s" % (sum(n1), F.fmt_ci(rne), DOT[c_c], F.fmt_ci(rnc)))
            # 11-28 addition: GPT's variant (2026-11-28_n1_single_retry.md): E; only where BOTH the no-write answer and E are
            # blank by the rule, the one-pass E+C(c) answer; c fixed at 1 for every model (as GPT), and c chosen on tuning
            ob, tob = out["_oblank"], tout["_oblank"]
            both = lambda qq, eb, obk, ev, ecv: [ecv[i] if (eb[q] and obk[q]) else ev[i] for i, q in enumerate(qq)]  # noqa: E731
            k1 = [k for k in lad if DOT[k] == "1"][0]
            tb = {k: sum(both(tids, tout["_eblank"], tob, tout["E"], tout["EC" + k])) for k in lad}
            c_b = pick(tb, lad)
            for lab_b, kb in (("c fixed 1", k1), ("c chosen on tuning %s" % DOT[c_b], c_b)):
                sb = both(ids, out["_eblank"], ob, e, out["EC" + kb])
                print("   both blank -> E+C (%s): net %+d (%s) | - E %s | - C(%s) %s | - N1 %s | questions re-asked %d" % (
                    lab_b, sum(sb), hrw(sb), F.fmt_ci(ci(sb, e, ids)), DOT[c_c], F.fmt_ci(ci(sb, cc, ids)), F.fmt_ci(ci(sb, n1, ids)),
                    sum(1 for q in ids if out["_eblank"][q] and ob[q])))
                summary_b.append((task, m, DOT[kb], lab_b.split(" ")[1], sum(sb), ci(sb, e, ids), ci(sb, cc, ids), ci(sb, n1, ids)))
            summary.append((task, m, s, DOT[c_ec], DOT[c_c], sum(o), sum(e), sum(cc), sum(ec), r_e, r_c, DOT[c_2], sum(s2), r2e, r2c, sum(n1), r2n,
                            rne, rnc))
    print("\n" + "=" * 120)
    print("## summary, one row per model (no pooled row): one pass each, c chosen on the tuning questions")
    print("   %-10s %-13s %-5s %-6s %-5s %6s %6s %6s %6s   %-18s %-18s" % ("task", "model", "E s", "EC c", "C c", "O", "E", "C", "EC", "EC - E", "EC - C"))
    for t in summary:
        print("   %-10s %-13s %-5s %-6s %-5s %+6d %+6d %+6d %+6d   %-18s %-18s" % (t[0], SHOW[t[1]], t[2], t[3], t[4], t[5], t[6], t[7], t[8],
                                                                             F.fmt_ci(t[9]), F.fmt_ci(t[10])))
    print("\n## summary of the two steps (E; E blank -> E+C(c)), one row per model")
    print("   %-10s %-13s %-5s %6s %6s %6s   %-18s %-18s %-18s" % ("task", "model", "c", "2step", "E", "N1", "2step - E", "2step - C", "2step - N1"))
    for t in summary:
        print("   %-10s %-13s %-5s %+6d %+6d %+6d   %-18s %-18s %-18s" % (t[0], SHOW[t[1]], t[11], t[12], t[6], t[15], F.fmt_ci(t[13]),
                                                                       F.fmt_ci(t[14]), F.fmt_ci(t[16])))
    print("\n## 11-28: E; only where both the no-write answer and E are blank, E+C(c) (GPT's variant), natural answer counts")
    print("   %-10s %-13s %-5s %-8s %6s   %-18s %-18s %-18s" % ("task", "model", "c", "c rule", "net", "- E", "- C", "- N1"))
    for t in summary_b:
        print("   %-10s %-13s %-5s %-8s %+6d   %-18s %-18s %-18s" % (t[0], SHOW[t[1]], t[2], t[3], t[4], F.fmt_ci(t[5]), F.fmt_ci(t[6]), F.fmt_ci(t[7])))
    print("\n## reference: the current N1 against the same single E and single C, one row per model")
    print("   %-10s %-13s %6s   %-18s %-18s" % ("task", "model", "N1", "N1 - E", "N1 - C"))
    for t in summary:
        print("   %-10s %-13s %+6d   %-18s %-18s" % (t[0], SHOW[t[1]], t[15], F.fmt_ci(t[17]), F.fmt_ci(t[18])))


if __name__ == "__main__":
    main()
