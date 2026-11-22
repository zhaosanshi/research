#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-22), work queue discussion/work_queue/2026-11-22_queue_d.md,
# section 5 (step 3 of "self-judge after answering"): the equal-answer-count comparisons. Offline: no model, no judge.
# Nothing is chosen here. Rule = step 2's R5b at true/false E strength 3: an answer of single E that does NOT open with
# a negated premise (selfjudge_probe.NEG_RE) becomes no answer when its self-judgement margin (written, E_a3) is below
# the "other" threshold chosen on the TruthfulQA tune split in step 2 (selfjudge_probe.step2_thresholds, recomputed
# here from the same files and checked against the step-2 output head). The no-write margin is used only to rank
# comparator A.
# Systems on a question list (value: right +1, wrong -1, else 0; answer counts as each source defines "answered"):
#   S1  single E + self-judge (n1 answers).
#   A   single E, the same number of "other"-class answers dropped, lowest NO-WRITE margin first (ties: list order).
#   B   strong single E: must-answer prompt + E at the same layer and strength, the n1 most confident answers kept
#       (confidence = mean answer log-prob scored with E at "prompt"; ties: list order); "not comparable" when fewer
#       than n1 answers carry a confidence.
#   S2  N1 + self-judge: the E answers N1 keeps are dropped by the same flags, N1's released answers untouched (n2).
#   B2  strong single E, n2 kept.   C2  single E's first-pass answers all kept, its blanks filled from the strong
#       ranking up to n2 answers; "not comparable" when n2 < single E's answers or the blanks cannot be filled.
#   reference  N1 minus strong single E at N1's own answer count (must reproduce the known numbers).
# Sources: TruthfulQA test split, main-table list and settings: diag/tq_same_n_decomp.load_model (the version of
#   2026-11-22 16:28) for single E, N1, issued and the strong arm; self-judge margins from step 1
#   (newsets/selfjudge/<model>/items.json, probe.npz). Paired intervals: policy_select2.ci (4000, seed 20261116);
#   two-model total on the sorted common questions, every quantity recomputed on that list (as tq_same_n_decomp).
#   (QA)^2: qaqa_pilot_table.Raw on the 570-question bank with the judge dirs test_full570, test_full570_tuned,
#   test_full570_strong; N1 = Raw.n1_system at the fixed-E tune choice (newsets/splits/qaqa_tune_choice_fixedE_20261122.json);
#   strong arm and confidence as discussion/validation_dataset/2026-11-22_qaqa_replay_b.py (Raw.confidence of
#   "conf:E..."); self-judge margins from step 2 (qaqa_items.json, qaqa_probe.npz). Intervals: newsets_score.joint_ci
#   (4000, seed 20261119), the held-out 470 questions (pilot 100 left out) and all 570.
# usage: (container) docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B \
#            Iteration_01/diag/selfjudge_equal_n.py qaqa-dump MODEL       -> newsets/selfjudge/<MODEL>/qaqa_equal_n.json
#        (host, from /data/sanshizhao/Truth_Decoupling) python3 -B Iteration_01/diag/selfjudge_equal_n.py table
import hashlib
import json
import os
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import selfjudge_probe as SP     # noqa: E402

ORDER = ("llama3.1-8b", "qwen3.5-9b")
COND = "E_a3"
QAQA_WORKS = ["test_full570", "test_full570_tuned", "test_full570_strong"]
CHOICE = "Iteration_01/newsets/splits/qaqa_tune_choice_fixedE_20261122.json"
# known numbers the reference row must reproduce (work queue section 5 / main table / summary section 12.8)
EXPECT_TQ = {"llama3.1-8b": "+51 [+17, +87]", "qwen3.5-9b": "+13 [-11, +37]"}
EXPECT_QQ470 = {"llama3.1-8b": 23, "qwen3.5-9b": 22}
KEYS = ("S1-A", "S1-B", "S2-B2", "S2-C2", "ref")


def sha1(t):
    return hashlib.sha1(t.strip().encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ #
def qaqa_dump(model):
    import math
    import qaqa_pilot_table as qt
    out = os.path.join(SP.OUT, model, "qaqa_equal_n.json")
    if os.path.exists(out):
        raise SystemExit("%s exists" % out)
    ch = SP.read_json(CHOICE)[model]
    s = ch["n1_strength"]
    if not ch.get("e_fixed") or s != qt.SET[model][1]:
        raise SystemExit("%s: the fixed-E choice does not keep the main-table strength" % model)
    R = qt.Raw(model, "qaqafull", SP.QAQA_BANK, QAQA_WORKS)
    ordinary = R.e(s)
    forced = "conf:" + ordinary.split(":", 1)[1]
    for arm in (ordinary, forced):
        R.need(arm)
    n1, issued, _ = R.n1_system(s, ch["n1_cap_e"], ch["n1_cap_own"])
    conf = R.confidence("qaqafull", forced)
    if conf is None:
        raise SystemExit("%s: no confidence file for %s" % (model, forced))
    rows = []
    for q in R.ids:
        eans = not R.blank(ordinary, q)
        fans = not R.blank(forced, q)
        if fans and (conf[q] is None or not math.isfinite(conf[q])):
            raise SystemExit("%s: a strong answer without a finite confidence (%s)" % (model, q))
        if eans and (not issued[q] or n1[q] != R.cell(ordinary, q)):
            raise SystemExit("%s: N1 does not keep E's answer of %s" % (model, q))
        rows.append(dict(id=q, e=R.cell(ordinary, q), eans=eans, e_sha1=sha1(R.text(ordinary, q)), n1=n1[q], issued=bool(issued[q]),
                         f=R.cell(forced, q), fans=fans, conf=conf[q] if fans else None, qtype=R.qtype[q]))
    doc = dict(model=model, ordinary=ordinary, forced=forced, choice=ch, works=QAQA_WORKS, bank_sha256=SP.sha256_file(SP.QAQA_BANK), rows=rows)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=0)
    print("%s: %d questions, E answers %d, N1 answers %d, strong answers %d -> %s" % (
        model, len(rows), sum(r["eans"] for r in rows), sum(r["issued"] for r in rows), sum(r["fans"] for r in rows), out))


# ------------------------------------------------------------------------------------------------------------------ #
def compute(ids, d):
    """d: per question e, eans, n1, issued, f, fans, conf, drop (S1 flag), other (E answer of the 'other' class),
    base (no-write margin, E answers only). Returns (cells per question, summary)."""
    pos = {q: i for i, q in enumerate(ids)}
    drop = {q: bool(d[q]["drop"] and d[q]["eans"]) for q in ids}
    k = sum(drop.values())
    ranked = sorted([q for q in ids if d[q]["fans"] and d[q]["conf"] is not None], key=lambda q: (-d[q]["conf"], pos[q]))

    def strong(n):
        if n > len(ranked):
            return None
        top = set(ranked[:n])
        return {q: (d[q]["f"] if q in top else 0) for q in ids}, n

    s1 = {q: 0 if drop[q] else d[q]["e"] for q in ids}
    n1c = sum(d[q]["eans"] and not drop[q] for q in ids)
    cand = sorted([q for q in ids if d[q]["eans"] and d[q]["other"]], key=lambda q: (d[q]["base"], pos[q]))
    adrop = set(cand[:k])
    a = {q: 0 if q in adrop else d[q]["e"] for q in ids}
    if any(drop[q] and not d[q]["issued"] for q in ids):
        raise ValueError("a dropped E answer is not kept by N1")
    s2 = {q: 0 if drop[q] else d[q]["n1"] for q in ids}
    n2 = sum(d[q]["issued"] for q in ids) - k
    ne = sum(d[q]["eans"] for q in ids)
    b, b2, ref_strong = strong(n1c), strong(n2), strong(sum(d[q]["issued"] for q in ids))
    c2 = None
    if n2 >= ne:
        eb = [q for q in ranked if not d[q]["eans"]]
        if len(eb) >= n2 - ne:
            fill = set(eb[:n2 - ne])
            c2 = {q: (d[q]["f"] if q in fill else d[q]["e"]) for q in ids}, n2
    if ref_strong is None:
        raise SystemExit("the strong arm cannot reach N1's answer count")
    cells = {q: {"S1-A": s1[q] - a[q], "S1-B": s1[q] - b[0][q] if b else None, "S2-B2": s2[q] - b2[0][q] if b2 else None,
                 "S2-C2": s2[q] - c2[0][q] if c2 else None, "ref": d[q]["n1"] - ref_strong[0][q]} for q in ids}
    net = lambda v: sum(v.values())
    summ = dict(k=k, n1=n1c, n2=n2, nE=ne, nN1=sum(d[q]["issued"] for q in ids), nets=dict(
        E=net({q: d[q]["e"] for q in ids}), N1=net({q: d[q]["n1"] for q in ids}), S1=net(s1), A=net(a),
        B=net(b[0]) if b else None, S2=net(s2), B2=net(b2[0]) if b2 else None, C2=net(c2[0]) if c2 else None,
        strong_at_N1=net(ref_strong[0])), n_strong=len(ranked),
        drops=dict(R=sum(drop[q] and d[q]["e"] == 1 for q in ids), W=sum(drop[q] and d[q]["e"] == -1 for q in ids),
                   O=sum(drop[q] and d[q]["e"] == 0 for q in ids)),
        adrops=dict(R=sum(d[q]["e"] == 1 for q in adrop), W=sum(d[q]["e"] == -1 for q in adrop), O=sum(d[q]["e"] == 0 for q in adrop)))
    return cells, summ


def show(label, ids, cells, summ, ci):
    n = summ["nets"]
    print("  %s: %d questions | drops %d (right %d, wrong %d, other %d); A drops right %d, wrong %d, other %d" % (
        label, len(ids), summ["k"], summ["drops"]["R"], summ["drops"]["W"], summ["drops"]["O"],
        summ["adrops"]["R"], summ["adrops"]["W"], summ["adrops"]["O"]))
    print("    answers: E %d, N1 %d, S1 = A = B %d (n1), S2 = B2 = C2 %d (n2); strong answers with confidence %d" % (
        summ["nE"], summ["nN1"], summ["n1"], summ["n2"], summ["n_strong"]))
    print("    nets: E %+d, N1 %+d, strong at N1's count %+d | S1 %+d, A %+d, B %s | S2 %+d, B2 %s, C2 %s" % (
        n["E"], n["N1"], n["strong_at_N1"], n["S1"], n["A"], "not comparable" if n["B"] is None else "%+d" % n["B"],
        n["S2"], "not comparable" if n["B2"] is None else "%+d" % n["B2"], "not comparable" if n["C2"] is None else "%+d" % n["C2"]))
    print("    " + " | ".join("%s %s" % (key, "not comparable" if cells[ids[0]][key] is None else ci([cells[q][key] for q in ids]))
                             for key in KEYS))


def sj_inputs(model):
    """Step-2 thresholds (recomputed from step-1 files) and the per-answer flags of both sets."""
    import numpy as np
    d = os.path.join(SP.OUT, model)
    doc = SP.read_json(d + "/items.json")
    z = np.load(d + "/probe.npz")
    ans = [x for x in doc["items"] if x["answered"]]
    if [str(i) for i in z["ids"]] != [x["id"] for x in ans] or [str(a) for a in z["answer_sha1"]] != [x["answer_sha1"] for x in ans]:
        raise SystemExit("%s: step-1 probe rows" % model)
    mg = {(x["split"], x["id"]): {c: float(z[c + "__margin"][j]) for c in SP.CONDS2} for j, x in enumerate(ans)}
    thr = SP.step2_thresholds([(mg[("tune", x["id"])], SP.V[x["state"]], x["neg"]) for x in ans if x["split"] == "tune"])
    t = thr[COND]["other"][0]
    tq = {x["id"]: dict(state=x["state"], eans=x["answered"], neg=x["neg"], sha=x["answer_sha1"],
                        m=mg.get(("test", x["id"]))) for x in doc["items"] if x["split"] == "test"}
    qd = SP.read_json(d + "/qaqa_items.json")
    zq = np.load(d + "/qaqa_probe.npz")
    qa = [x for x in qd["items"] if x["answered"]]
    if [str(i) for i in zq["ids"]] != [x["id"] for x in qa]:
        raise SystemExit("%s: step-2 probe rows" % model)
    qm = {x["id"]: {c: float(zq[c + "__margin"][j]) for c in SP.CONDS2} for j, x in enumerate(qa)}
    qq = {x["id"]: dict(eans=x["answered"], neg=x["neg"], sha=x["answer_sha1"], v=x["value"], pilot=x["pilot"], m=qm.get(x["id"]))
          for x in qd["items"]}
    return t, thr, tq, qq


def table():
    sys.path.insert(0, "Iteration_01/diag")
    import tq_same_n_decomp as TQ
    import newsets_score as NS
    ci_tq = TQ.ps.ci

    def ci_qq(v):
        tot, lo, hi = NS.joint_ci([list(v)])
        return "%+d [%+d, %+d]" % (tot, lo, hi)

    print("# selfjudge step 3 (work queue 2026-11-22_queue_d, section 5): equal answer counts; rule R5b, true/false E strength 3")
    tqd, qqd = {}, {}
    for model in ORDER:
        t, thr, tq, qq = sj_inputs(model)
        print("# %s: threshold (other class, E_a3, chosen on TruthfulQA tune in step 2) t=%+.4f (tune drops %d, gain %+d)" % (
            model, t, thr[COND]["other"][1], thr[COND]["other"][2]))
        ids, data = TQ.load_model(model)
        _, layer, s, n1cfg, run, e, eans, n1, issued, F = data
        if set(ids) != set(tq) or any(SP.V[tq[q]["state"]] != e[q] or tq[q]["eans"] != eans[q] for q in ids):
            raise SystemExit("%s: tq_same_n_decomp's single E differs from step 1's items" % model)
        dd = {}
        for q in ids:
            x = tq[q]
            other = x["eans"] and not x["neg"]
            dd[q] = dict(e=e[q], eans=eans[q], n1=n1[q], issued=issued[q], f=F[q][0], fans=F[q][1], conf=F[q][2],
                         other=other, drop=other and x["m"][COND] < t, base=x["m"]["baseline"] if x["eans"] else None)
        tqd[model] = (ids, dd, (layer, s, n1cfg))
        qdoc = SP.read_json(os.path.join(SP.OUT, model, "qaqa_equal_n.json"))
        qd = {}
        for r in qdoc["rows"]:
            x = qq[r["id"]]
            if x["eans"] != r["eans"] or x["v"] != r["e"] or x["sha"] != r["e_sha1"]:
                raise SystemExit("%s: (QA)^2 single E differs between step 2 and the dump (%s)" % (model, r["id"]))
            other = x["eans"] and not x["neg"]
            qd[r["id"]] = dict(e=r["e"], eans=r["eans"], n1=r["n1"], issued=r["issued"], f=r["f"], fans=r["fans"], conf=r["conf"],
                               other=other, drop=other and x["m"][COND] < t, base=x["m"]["baseline"] if x["eans"] else None,
                               pilot=x["pilot"])
        qqd[model] = ([r["id"] for r in qdoc["rows"]], qd, qdoc)
    # references first
    ok = True
    for model in ORDER:
        ids, dd, _ = tqd[model]
        cells, _ = compute(ids, dd)
        got = ci_tq([cells[q]["ref"] for q in ids])
        qids, qd, _ = qqd[model]
        held = [q for q in qids if not qd[q]["pilot"]]
        qc, _ = compute(held, qd)
        qgot = sum(qc[q]["ref"] for q in held)
        print("# reference N1 minus strong single E: %s TruthfulQA %s (expected %s) | (QA)^2 470 %+d (expected %+d)" % (
            model, got, EXPECT_TQ[model], qgot, EXPECT_QQ470[model]))
        ok = ok and got == EXPECT_TQ[model] and len(held) == 470 and qgot == EXPECT_QQ470[model]
    if not ok:
        raise SystemExit("REFERENCE MISMATCH: stopping before any step-3 reading")
    print("# references reproduced")
    for model in ORDER:
        ids, dd, (layer, s, n1cfg) = tqd[model]
        print("=" * 100)
        print("%s | TruthfulQA test, E L%d strength %s, N1 %s" % (model, layer, s, n1cfg))
        cells, summ = compute(ids, dd)
        show("TruthfulQA test", ids, cells, summ, ci_tq)
        qids, qd, qdoc = qqd[model]
        print("%s | (QA)^2, %s, strong %s, N1 fixed-E choice caps %s / %s" % (
            model, qdoc["ordinary"], qdoc["forced"], qdoc["choice"]["n1_cap_e"], qdoc["choice"]["n1_cap_own"]))
        for lab, sub in (("(QA)^2 470 held out", [q for q in qids if not qd[q]["pilot"]]), ("(QA)^2 570", qids)):
            c, sm = compute(sub, qd)
            show(lab, sub, c, sm, ci_qq)
    print("=" * 100)
    print("two models together (questions both models have; every quantity recomputed on that list)")
    common = sorted(set(tqd[ORDER[0]][0]) & set(tqd[ORDER[1]][0]))
    cs = {m: compute(common, tqd[m][1])[0] for m in ORDER}
    print("  TruthfulQA test, %d common questions: " % len(common) + " | ".join(
        "%s %s" % (key, "not comparable" if any(cs[m][common[0]][key] is None for m in ORDER)
                   else ci_tq([sum(cs[m][q][key] for m in ORDER) for q in common])) for key in KEYS))
    qids = qqd[ORDER[0]][0]
    if qids != qqd[ORDER[1]][0]:
        raise SystemExit("(QA)^2 question lists differ")
    for lab, sub in (("470 held out", [q for q in qids if not qqd[ORDER[0]][1][q]["pilot"]]), ("570", qids)):
        cs = {m: compute(sub, qqd[m][1])[0] for m in ORDER}
        line = []
        for key in KEYS:
            if any(cs[m][sub[0]][key] is None for m in ORDER):
                line.append("%s not comparable" % key)
            else:
                tot, lo, hi = NS.joint_ci([[cs[m][q][key] for q in sub] for m in ORDER])
                line.append("%s %+d [%+d, %+d]" % (key, tot, lo, hi))
        print("  (QA)^2 %s, %d questions: %s" % (lab, len(sub), " | ".join(line)))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "qaqa-dump" and len(sys.argv) == 3 and sys.argv[2] in ORDER:
        qaqa_dump(sys.argv[2])
    elif sys.argv[1:] == ["table"]:
        table()
    else:
        raise SystemExit("usage: selfjudge_equal_n.py qaqa-dump MODEL | table")


if __name__ == "__main__":
    main()
