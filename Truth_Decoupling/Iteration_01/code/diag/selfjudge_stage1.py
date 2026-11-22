#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-22), work queue discussion/work_queue/2026-11-22_queue_e.md,
# stage one, items 1a / 1b / 1c, Llama-3.1-8B and Qwen3.5-9B. No generation, no judge call. Definitions are the work
# queue's section 1; nothing is chosen except the thresholds of rule P / P' on the TruthfulQA tune split.
# Answer sources whose self-judgement is read (forward passes: selfjudge_probe.py score --sources ...):
#   tq_forced   TruthfulQA must-answer + E at the main-table layer / strength, write rule "prompt" (the strong arm):
#               texts and verdicts as diag/tq_same_n_decomp.forced (work dir judge_test_pospromptd<tag>_test); the
#               test split only (no tune pass of this arm exists).
#   tq_release  TruthfulQA N1's released answers (records "released" of final_records.records at the main table's N1
#               choice), tune and test; each text checked against the generation file of its E + C arm.
#   qq_forced   (QA)^2 570, the strong arm "conf:E..." read with qaqa_pilot_table.Raw (judge dirs test_full570,
#               test_full570_tuned, test_full570_strong).
#   qq_release  (QA)^2 570, N1's released answers at the fixed-E tune choice (Raw.n1_system rebuilt here with the
#               released arm kept; values and answered flags must equal Raw.n1_system's).
# Rule R5b: an answer opening with a negated premise (selfjudge_probe.NEG_RE) is always kept; any other answer is
# "judged false" when its margin is below the threshold. Thresholds per model and answer source, TruthfulQA tune only:
#   P   the threshold of highest tune net over that source's "other"-class answers, ties to the fewest dropped;
#   P'  among thresholds whose tune net is within 2 of the highest, the fewest dropped.
# Systems: S1 = single E + self-judge; S2 = N1 + self-judge on E's first-pass answers; S3 = S2 + self-judge on the
# released answers (with the E-source threshold). B(n) = strong single E keeping its n most confident answers.
# Readings: S1-B, S2-B2, S3-B3 (B at each system's own answer count), S3-S2 and the released answers dropped; reference
# N1 - B(N1's count) and the identity "S2 - (S1's first pass + the strong ranking filling as many of E's blanks as N1
# releases)" = tq_same_n_decomp's N1_minus_Efill. Not computed (work queue rule not executable with existing files,
# see the progress note): B+, C3 (need a must-answer + E threshold chosen on the TruthfulQA tune split, and that arm
# has no tune answers), D (no answer-confidence file of single E's pass).
# Intervals: TruthfulQA policy_select2.ci (4000, seed 20261116); (QA)^2 newsets_score.joint_ci (4000, seed 20261119);
# two models together on the sorted common questions (TruthfulQA) / the same list (QA)^2, quantities recomputed there.
# usage: (host)      python3 -B Iteration_01/diag/selfjudge_stage1.py export-tq MODEL
#        (container) docker exec -u 1000:1001 -w /workspace/Truth_Decoupling truth_dec python3 -B \
#                        Iteration_01/diag/selfjudge_stage1.py export-qq MODEL
#        (GPU)       launchers/explore_20261120/selfjudge_1122.sh MODEL --sources tq_forced,tq_release,qq_forced,qq_release
#        (host)      python3 -B Iteration_01/diag/selfjudge_stage1.py table
import json
import os
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import selfjudge_equal_n as EQ
import selfjudge_probe as SP     # noqa: E402

ORDER = ("llama3.1-8b", "qwen3.5-9b")
CHOICE = "Iteration_01/newsets/splits/qaqa_tune_choice_fixedE_20261122.json"
WORKS = ["test_full570", "test_full570_tuned", "test_full570_strong"]
EXPECT_TQ = {"llama3.1-8b": "+51 [+17, +87]", "qwen3.5-9b": "+13 [-11, +37]"}
EXPECT_FILL = {"llama3.1-8b": "+21 [-4, +46]", "qwen3.5-9b": "+14 [+2, +26]"}
EXPECT_QQ470 = {"llama3.1-8b": 23, "qwen3.5-9b": 22}
sha1 = SP.sha1


def write_items(model, src, items, extra):
    path = os.path.join(SP.OUT, model, src + "_items.json")
    if len({x["id"] for x in items}) != len(items):
        raise SystemExit("%s: duplicate ids" % src)
    SP.save_items(path, dict(model=model, source=src, items=items, **extra))
    a = [x for x in items if x["answered"]]
    print("%s %s: %d items, answered %d (right %d, wrong %d, other %d), negated-premise opening %d -> %s" % (
        model, src, len(items), len(a), sum(x["value"] == 1 for x in a), sum(x["value"] == -1 for x in a),
        sum(x["value"] == 0 for x in a), sum(x["neg"] for x in a), path))


# ------------------------------------------------------------------------------------------------------------------ #
def export_tq(model):
    import tq_same_n_decomp as TQ
    A, MP, ML, S = TQ.A, TQ.MP, TQ.ML, TQ.S
    step1 = SP.read_json(os.path.join(SP.OUT, model, "items.json"))
    qtext = {(x["split"], x["id"]): x["question"] for x in step1["items"]}
    display, layer, s, tune, test = SP.main_lists(A, model)
    ids, data = TQ.load_model(model)
    _, _, _, n1cfg, run, e, eans, n1, issued, F = data
    if ids != test[3]:
        raise SystemExit("%s: tq_same_n_decomp's list differs from the main list" % model)
    # must-answer + E (strong arm), test split
    tag = "l%d" % layer
    w = MP.work(run, "test", "pospromptd" + tag)
    st, bl = TQ.ps.load(w, "pro")
    texts = MP.fr.raw_texts(w)
    cond = "E_L%d_conf-w-p%s" % (layer, s)
    items = []
    for q in ids:
        if (TQ.V[st[(q, cond)]], not bl[(q, cond)]) != F[q][:2]:
            raise SystemExit("%s: forced cell of %s differs from tq_same_n_decomp.forced" % (model, q))
        t = texts[(q, cond)]
        items.append(dict(id="test|" + q, question=qtext[("test", q)], answer=t, answer_sha1=sha1(t), value=F[q][0],
                          answered=F[q][1], conf=F[q][2], neg=bool(SP.NEG_RE.match(t))))
    write_items(model, "tq_forced", items, dict(work_dir=w, condition=cond))
    # N1's released answers, tune and test
    items = []
    for i, (split, (sp, tx, _, qs)) in enumerate((("tune", tune), ("test", test))):
        st2, na, recs = S.system(sp, tx, n1cfg)
        es, ea, _ = S.system(sp, tx, ("E", s))
        _, runs, _ = MP.usual(display, i)
        by_path = {}
        for k, r in enumerate(recs):
            if r["p"] != qs[k]:
                raise ValueError("N1 record order differs from the frozen roster")
            if r["note"] != "released":
                if ea[k] != na[k]:
                    raise SystemExit("%s: N1 answers %s without a release record (%s)" % (model, r["p"], r["note"]))
                continue
            if ea[k] or not na[k] or not r["cond"].startswith("X_x--"):
                raise SystemExit("%s: a released record that is not an answer to an E blank (%s)" % (model, r["p"]))
            gpath = "%s/eval_extra/gen/%s-at-prompt_%s.json" % (runs, r["cond"][2:], split)
            if gpath not in by_path:
                g = SP.read_json(gpath)
                gt = {x["pair_id"]: x["text"] for x in g["rows"]}
                if (g.get("positions") != "prompt" or g.get("split") != split or not g.get("completed")
                        or g.get("partial") or len(gt) != len(g["rows"])):
                    raise ValueError(f"{gpath}: wrong, incomplete or duplicated source")
                by_path[gpath] = gt
            if sha1(by_path[gpath][r["p"]]) != sha1(r["text"]):
                raise SystemExit("%s: released text of %s is not the generation file's (%s)" % (model, r["p"], gpath))
            items.append(dict(id=split + "|" + r["p"], split=split, question=qtext[(split, r["p"])], answer=r["text"],
                              answer_sha1=sha1(r["text"]), value=TQ.V[st2[k]], answered=True, cond=r["cond"],
                              neg=bool(SP.NEG_RE.match(r["text"]))))
        if split == "test":
            rel = {x["id"][5:] for x in items if x["split"] == "test"}
            if rel != {q for q in ids if issued[q] and not eans[q]}:
                raise SystemExit("%s: released list differs from tq_same_n_decomp's issued blanks" % model)
    write_items(model, "tq_release", items, dict(n1_choice=list(n1cfg)))


def export_qq(model):
    import qaqa_pilot_table as qt
    ch = SP.read_json(CHOICE)[model]
    if not ch["e_fixed"] or ch["n1_strength"] != qt.SET[model][1]:
        raise ValueError(f"{model}: not the fixed-E choice")
    s = ch["n1_strength"]
    R = qt.Raw(model, "qaqafull", SP.QAQA_BANK, WORKS)
    with open(SP.QAQA_BANK, encoding="utf-8") as f:
        bank = {r["id"]: r for r in (json.loads(l) for l in f)}
    pilot = set(SP.read_json(SP.QAQA_PILOT)["ids"])
    ordinary = R.e(s)
    forced = "conf:" + ordinary.split(":", 1)[1]
    R.need(ordinary)
    R.need(forced)
    conf = R.confidence("qaqafull", forced)
    items = []
    for q in R.ids:
        t = R.text(forced, q)
        items.append(dict(id=q, question=bank[q]["question"], answer=t, answer_sha1=sha1(t), value=R.cell(forced, q),
                          answered=not R.blank(forced, q), conf=conf[q], neg=bool(SP.NEG_RE.match(t)), pilot=q in pilot,
                          qtype=bank[q]["qtype"]))
    write_items(model, "qq_forced", items, dict(arm=forced, works=WORKS))
    # N1, rebuilt from Raw.n1_system's code with the released arm kept
    v_ref, a_ref, _ = R.n1_system(s, ch["n1_cap_e"], ch["n1_cap_own"])
    items = []
    for q in R.ids:
        if not R.blank(ordinary, q):
            if not a_ref[q] or v_ref[q] != R.cell(ordinary, q):
                raise SystemExit("%s: N1 does not keep E's answer of %s" % (model, q))
            continue
        own = R.blank("x", q)
        cap = ch["n1_cap_own"] if own else ch["n1_cap_e"]
        lad = () if cap is None else qt.LAD[:qt.LAD.index(cap) + 1]
        k = next((k for k in lad if not R.blank(R.ec(s, k), q)), None)
        if k is not None and not own and qt.ro.norm(R.text(R.ec(s, k), q)) == qt.ro.norm(R.text("x", q)):
            k = None
        if k is None:
            if a_ref[q] or v_ref[q] != R.cell(ordinary, q):
                raise SystemExit("%s: rebuilt N1 differs on %s" % (model, q))
            continue
        arm = R.ec(s, k)
        t = R.text(arm, q)
        if not a_ref[q] or v_ref[q] != R.cell(arm, q) or (q, R.cond[arm]) not in R.pred or R.pred[(q, R.cond[arm])].strip() != t.strip():
            raise SystemExit("%s: rebuilt N1 release differs on %s" % (model, q))
        items.append(dict(id=q, question=bank[q]["question"], answer=t, answer_sha1=sha1(t), value=R.cell(arm, q), answered=True,
                          arm=arm, neg=bool(SP.NEG_RE.match(t)), pilot=q in pilot, qtype=bank[q]["qtype"]))
    write_items(model, "qq_release", items, dict(choice=ch, works=WORKS))


# ------------------------------------------------------------------------------------------------------------------ #
def margins(model, src, cond):
    if src == "qaqa":
        doc, _, z = SP.read_probe(model, qaqa=True)
    else:
        doc, _, z = SP.read_probe(model, source=src)
    a = [x for x in doc["items"] if x["answered"]]
    return doc, {x["id"]: float(z[cond + "__margin"][j]) for j, x in enumerate(a)}


def choose(rows):
    """rows: [(margin, value)] of the 'other'-class tune answers. Returns (P, P') as (t, drops, gain)."""
    u = sorted(set(m for m, _ in rows))
    cands = [float("-inf")] + [(a + b) / 2.0 for a, b in zip(u[:-1], u[1:])] + [float("inf")]
    res = [(t, sum(m < t for m, _ in rows), -sum(v for m, v in rows if m < t)) for t in cands]
    g = max(r[2] for r in res)
    p = min((r for r in res if r[2] == g), key=lambda r: r[1])
    pp = min((r for r in res if r[2] >= g - 2), key=lambda r: r[1])
    return p, pp


def ft(t):
    return "%+.4f" % t if abs(t) != float("inf") else ("-inf" if t < 0 else "+inf")


def compute(ids, d, t):
    """d[q]: e, eans, n1, issued, f, fans, conf, mE (E margin or None), negE, rel (released: (value, margin, neg) or None)."""
    pos = {q: i for i, q in enumerate(ids)}
    if not ids or len(ids) != len(pos):
        raise ValueError("empty or duplicate question IDs")
    for q in ids:
        r = d[q]
        if ((r["rel"] is not None) != (r["issued"] and not r["eans"])
                or r["eans"] and (not r["issued"] or r["n1"] != r["e"])
                or r["rel"] is not None and r["rel"][0] != r["n1"]):
            raise ValueError(f"{q}: inconsistent ordinary and released answers")
    dropE = {q: d[q]["eans"] and not d[q]["negE"] and d[q]["mE"] < t for q in ids}
    dropR = {q: d[q]["rel"] is not None and not d[q]["rel"][2] and d[q]["rel"][1] < t for q in ids}
    ranked = sorted([q for q in ids if d[q]["fans"] and d[q]["conf"] is not None], key=lambda q: (-d[q]["conf"], pos[q]))

    def strong(n):
        if n > len(ranked):
            return None
        top = set(ranked[:n])
        return {q: (d[q]["f"] if q in top else 0) for q in ids}

    s1 = {q: 0 if dropE[q] else d[q]["e"] for q in ids}
    s2 = {q: 0 if dropE[q] else d[q]["n1"] for q in ids}
    s3 = {q: 0 if dropE[q] or dropR[q] else d[q]["n1"] for q in ids}
    nE = sum(d[q]["eans"] for q in ids)
    nN1 = sum(d[q]["issued"] for q in ids)
    kE, kR = sum(dropE.values()), sum(dropR.values())
    nrel = sum(d[q]["rel"] is not None for q in ids)
    n = dict(E=nE, N1=nN1, S1=nE - kE, S2=nN1 - kE, S3=nN1 - kE - kR)
    B = {k: strong(n[k]) for k in ("S1", "S2", "S3", "N1")}
    eb = [q for q in ranked if not d[q]["eans"]]
    fill = set(eb[:nrel]) if len(eb) >= nrel else None
    x = {q: (d[q]["f"] if q in fill else s1[q]) for q in ids} if fill is not None else None
    sysv = dict(E={q: d[q]["e"] for q in ids}, N1={q: d[q]["n1"] for q in ids}, S1=s1, S2=s2, S3=s3)
    cells = {}
    for q in ids:
        c = {"S1-B": s1[q] - B["S1"][q] if B["S1"] else None, "S2-B2": s2[q] - B["S2"][q] if B["S2"] else None,
             "S3-B3": s3[q] - B["S3"][q] if B["S3"] else None, "S3-S2": s3[q] - s2[q], "ref": d[q]["n1"] - B["N1"][q] if B["N1"] is not None else None,
             "S2-(S1+fill)": s2[q] - x[q] if x else None}
        cells[q] = c
    nets = {k: sum(v.values()) for k, v in sysv.items()}
    nets.update({"B(%s)" % k: (sum(v.values()) if v else None) for k, v in B.items()})
    dE = [d[q]["e"] for q in ids if dropE[q]]
    dR = [d[q]["rel"][0] for q in ids if dropR[q]]
    rv = [d[q]["rel"][0] for q in ids if d[q]["rel"] is not None]
    summ = dict(n=n, nets=nets, dropE=(len(dE), dE.count(1), dE.count(-1), dE.count(0)),
                dropR=(len(dR), dR.count(1), dR.count(-1), dR.count(0)), released=(len(rv), rv.count(1), rv.count(-1), rv.count(0)),
                n_strong=len(ranked))
    return cells, summ


KEYS = ("S1-B", "S2-B2", "S3-B3", "S3-S2", "ref", "S2-(S1+fill)")


def show(label, ids, cells, summ, ci, short=False):
    n, nets = summ["n"], summ["nets"]
    print("  %s: %d questions | E answers dropped %d (right %d, wrong %d, other %d) | released %d (right %d, wrong %d, other %d), "
          "dropped %d (right %d, wrong %d, other %d)" % ((label, len(ids)) + summ["dropE"] + summ["released"] + summ["dropR"]))
    if not short:
        print("    answers: E %d, N1 %d, S1 %d, S2 %d, S3 %d; strong answers with confidence %d" % (
            n["E"], n["N1"], n["S1"], n["S2"], n["S3"], summ["n_strong"]))
        print("    nets: E %+d, N1 %+d, S1 %+d, S2 %+d, S3 %+d | strong at S1 / S2 / S3 / N1 counts: %s" % (
            nets["E"], nets["N1"], nets["S1"], nets["S2"], nets["S3"],
            " / ".join("not comparable" if nets["B(%s)" % k] is None else "%+d" % nets["B(%s)" % k] for k in ("S1", "S2", "S3", "N1"))))
    print("    " + " | ".join("%s %s" % (k, "not comparable" if cells[ids[0]][k] is None else ci([cells[q][k] for q in ids]))
                             for k in KEYS))


def table():
    import tq_same_n_decomp as TQ
    import newsets_score as NS
    ci_tq = TQ.ps.ci

    def ci_qq(v):
        tot, lo, hi = NS.joint_ci([list(v)])
        return "%+d [%+d, %+d]" % (tot, lo, hi)

    print("# selfjudge stage one (work queue 2026-11-22_queue_e, items 1a / 1b / 1c), Llama-3.1-8B and Qwen3.5-9B")
    print("# rule R5b; main reading = no-write self-judgement ('baseline'), side reading = true/false E strength 3 ('E_a3')")
    print("# NOT COMPUTED: B+ and C3 (no must-answer + E answers on the TruthfulQA tune split -> its threshold cannot be chosen"
          " by rule P); D (no answer-confidence file of single E's pass; the existing tools do not score it, see the notes)")
    print("# THRESHOLDS, TruthfulQA tune, source 'E plain-prompt answers', other class: rule P and P' (t, tune drops, tune gain)")
    T, data = {}, {}
    for model in ORDER:
        st1, _, z = SP.read_probe(model)
        ans = [x for x in st1["items"] if x["answered"]]
        mE = {c: {(x["split"], x["id"]): float(z[c + "__margin"][j]) for j, x in enumerate(ans)} for c in SP.CONDS2}
        for cond in SP.CONDS2:
            rows = [(mE[cond][("tune", x["id"])], SP.V[x["state"]]) for x in ans if x["split"] == "tune" and not x["neg"]]
            p, pp = choose(rows)
            T[(model, cond)] = (p, pp)
            print("  %-12s %-9s P t=%s drops %d gain %+d | P' t=%s drops %d gain %+d  (of %d other-class tune answers)" % (
                model, cond, ft(p[0]), p[1], p[2], ft(pp[0]), pp[1], pp[2], len(rows)))
        data[model] = (st1, mE)
    print("# the must-answer + E source: no tune answers, no threshold")
    # per-model inputs
    TQD, QQD = {}, {}
    for model in ORDER:
        st1, mE = data[model]
        ids, dat = TQ.load_model(model)
        _, layer, s, n1cfg, run, e, eans, n1, issued, F = dat
        neg1 = {x["id"]: x["neg"] for x in st1["items"] if x["split"] == "test"}
        if set(neg1) != set(ids):
            raise ValueError(f"{model}: native and probed test rosters differ")
        reld, rm = {}, {}
        rdoc = SP.read_json(os.path.join(SP.OUT, model, "tq_release_items.json"))
        for cond in SP.CONDS2:
            rm[cond] = margins(model, "tq_release", cond)[1]
        for x in rdoc["items"]:
            if x["split"] == "test":
                reld[x["id"][5:]] = x
        if set(reld) != {q for q in ids if issued[q] and not eans[q]} or list(n1cfg) != rdoc["n1_choice"]:
            raise ValueError(f"{model}: wrong N1 choice or incomplete release roster")
        per = {}
        for cond in SP.CONDS2:
            dd = {}
            for q in ids:
                r = reld.get(q)
                dd[q] = dict(e=e[q], eans=eans[q], n1=n1[q], issued=issued[q], f=F[q][0], fans=F[q][1], conf=F[q][2],
                             mE=mE[cond].get(("test", q)), negE=neg1[q],
                             rel=(r["value"], rm[cond]["test|" + q], r["neg"]) if r else None)
                if r and r["value"] != n1[q]:
                    raise SystemExit("%s: released value of %s" % (model, q))
            per[cond] = dd
        TQD[model] = (ids, per, (layer, s, n1cfg))
        ordinary_doc, _, _ = SP.read_probe(model, qaqa=True)
        qitems = {x["id"]: x for x in ordinary_doc["items"]}
        qe = EQ.read_dump(model, qitems)
        qrel = {x["id"]: x for x in SP.read_json(os.path.join(SP.OUT, model, "qq_release_items.json"))["items"]}
        qperE, qperR = {}, {}
        for cond in SP.CONDS2:
            qperE[cond] = margins(model, "qaqa", cond)[1]
            qperR[cond] = margins(model, "qq_release", cond)[1]
        if set(qrel) != {r["id"] for r in qe["rows"] if r["issued"] and not r["eans"]}:
            raise SystemExit("%s: (QA)^2 released list differs from the step-3 dump" % model)
        qper = {}
        for cond in SP.CONDS2:
            dd = {}
            for r in qe["rows"]:
                q = r["id"]
                x = qitems[q]
                if x["answer_sha1"] != r["e_sha1"] or x["answered"] != r["eans"] or x["value"] != r["e"]:
                    raise SystemExit("%s: (QA)^2 E rows differ" % model)
                rr = qrel.get(q)
                if rr and rr["value"] != r["n1"]:
                    raise SystemExit("%s: (QA)^2 released value of %s" % (model, q))
                dd[q] = dict(e=r["e"], eans=r["eans"], n1=r["n1"], issued=r["issued"], f=r["f"], fans=r["fans"], conf=r["conf"],
                             mE=qperE[cond].get(q), negE=x["neg"], rel=(rr["value"], qperR[cond][q], rr["neg"]) if rr else None,
                             pilot=x["pilot"])
            qper[cond] = dd
        QQD[model] = ([r["id"] for r in qe["rows"]], qper, qe)
    # references
    ok = True
    for model in ORDER:
        ids, per, _ = TQD[model]
        c, _ = compute(ids, per["baseline"], T[(model, "baseline")][0][0])
        if any(c[q]["ref"] is None or c[q]["S2-(S1+fill)"] is None for q in ids):
            raise ValueError(f"{model}: registered TruthfulQA reference is unreachable")
        got = ci_tq([c[q]["ref"] for q in ids])
        fill = ci_tq([c[q]["S2-(S1+fill)"] for q in ids])
        qids, qper, _ = QQD[model]
        held = [q for q in qids if not qper["baseline"][q]["pilot"]]
        qc, _ = compute(held, qper["baseline"], T[(model, "baseline")][0][0])
        if any(qc[q]["ref"] is None for q in held):
            raise ValueError(f"{model}: registered QA2 reference is unreachable")
        qgot = sum(qc[q]["ref"] for q in held)
        print("# reference %s: N1 - strong TruthfulQA %s (expected %s) | S2 - (S1 first pass + fill) TruthfulQA %s (expected N1 - "
              "two-pass single E %s) | (QA)^2 470 N1 - strong %+d (expected %+d)" % (model, got, EXPECT_TQ[model], fill, EXPECT_FILL[model],
                                                                                    qgot, EXPECT_QQ470[model]))
        ok = ok and got == EXPECT_TQ[model] and fill == EXPECT_FILL[model] and qgot == EXPECT_QQ470[model] and len(held) == 470
    if not ok:
        raise SystemExit("REFERENCE MISMATCH: stopping before any stage-one reading")
    print("# references reproduced")
    for cond, short in (("baseline", False), ("E_a3", True)):
        for rule in (0, 1):
            tag = "%s, rule %s" % (cond, ("P", "P'")[rule])
            print("=" * 100)
            print("SELF-JUDGEMENT %s%s" % (tag, "  (side reading, short table)" if short else ""))
            for model in ORDER:
                t = T[(model, cond)][rule][0]
                ids, per, (layer, s, n1cfg) = TQD[model]
                c, sm = compute(ids, per[cond], t)
                print(" %s (t=%s) TruthfulQA test, E L%d strength %s, N1 %s" % (model, ft(t), layer, s, n1cfg))
                show("TruthfulQA test", ids, c, sm, ci_tq, short)
                qids, qper, qe = QQD[model]
                for lab, sub in (("(QA)^2 470 held out", [q for q in qids if not qper[cond][q]["pilot"]]), ("(QA)^2 570", qids)):
                    c, sm = compute(sub, qper[cond], t)
                    show(lab, sub, c, sm, ci_qq, short)
            print(" two models together:")
            common = sorted(set(TQD[ORDER[0]][0]) & set(TQD[ORDER[1]][0]))
            cs = {m: compute(common, TQD[m][1][cond], T[(m, cond)][rule][0])[0] for m in ORDER}
            print("  TruthfulQA test, %d common questions: " % len(common) + " | ".join(
                "%s %s" % (k, "not comparable" if any(cs[m][common[0]][k] is None for m in ORDER)
                           else ci_tq([sum(cs[m][q][k] for m in ORDER) for q in common])) for k in KEYS))
            qids = QQD[ORDER[0]][0]
            for lab, sub in (("470 held out", [q for q in qids if not QQD[ORDER[0]][1][cond][q]["pilot"]]), ("570", qids)):
                cs = {m: compute(sub, QQD[m][1][cond], T[(m, cond)][rule][0])[0] for m in ORDER}
                line = []
                for k in KEYS:
                    if any(cs[m][sub[0]][k] is None for m in ORDER):
                        line.append("%s not comparable" % k)
                    else:
                        tot, lo, hi = NS.joint_ci([[cs[m][q][k] for q in sub] for m in ORDER])
                        line.append("%s %+d [%+d, %+d]" % (k, tot, lo, hi))
                print("  (QA)^2 %s, %d questions: %s" % (lab, len(sub), " | ".join(line)))

# ------------------------------------------------------------------------------------------------------------------ #
# "1 supplement" (work queue e, 17:16): rule Z = an answer of the "other" class is judged false when margin < 0
# (negated-premise openings exempt); the same 0 for every source and set. Main reading: rule Z, no write. Also: rule Z
# with the true/false E write (strength 3); rule Z without the exemption (every answer judged), no write; rule P', no
# write (cells needing a must-answer + E threshold "not comparable").
#   B+(n)  strong single E self-judged with its own threshold; the passed answers (with a confidence) kept by confidence,
#          and when fewer than n passed, the judged-false ones added back by margin, highest first, up to n (count
#          reported); "not comparable" when fewer than n strong answers carry a confidence.
#   C3     first pass = S1's kept E answers; second pass = on E's blanks, the strong answers that pass the self-judge,
#          by confidence, as many as S3 keeps released answers; "not comparable" when too few candidates.
#   D      (QA)^2 only: single E's answers ranked by their own mean answer log-probability (ns_conf_pos.py, writes at
#          "prompt"; newsets/selfjudge/<MODEL>/x--E<L>-w-p<S>-at-prompt.json), the k lowest dropped, k = S1's drops;
#          D_all from every E answer, D_other from the "other" class only.
# ------------------------------------------------------------------------------------------------------------------ #
def load_all():
    import tq_same_n_decomp as TQ
    out = {}
    for model in ORDER:
        st1, _, z = SP.read_probe(model)
        ans = [x for x in st1["items"] if x["answered"]]
        mE = {c: {(x["split"], x["id"]): float(z[c + "__margin"][j]) for j, x in enumerate(ans)} for c in SP.CONDS2}
        tune_rows = {c: [(mE[c][("tune", x["id"])], SP.V[x["state"]]) for x in ans if x["split"] == "tune" and not x["neg"]]
                     for c in SP.CONDS2}
        ids, dat = TQ.load_model(model)
        _, layer, s, n1cfg, run, e, eans, n1, issued, F = dat
        native = {r["pair_id"]: r["text"] for r in SP.read_json(os.path.join(
            run, "eval_extra/gen", f"x--E{layer}-w-p{s}-at-prompt_test.json"))["rows"]}
        plain = {r["id"]: r for r in st1["items"] if r["split"] == "test"}
        if (list(plain) != list(ids) or any(plain[q]["answer_sha1"] != SP.sha1(native[q]) or plain[q]["answered"] != eans[q]
                or SP.V[plain[q]["state"]] != e[q] for q in ids)):
            raise ValueError(f"{model}: native ordinary-E test source differs")
        neg1 = {q: r["neg"] for q, r in plain.items()}
        rel = {x["id"][5:]: x for x in SP.read_json(os.path.join(SP.OUT, model, "tq_release_items.json"))["items"] if x["split"] == "test"}
        fdoc = {x["id"][5:]: x for x in SP.read_json(os.path.join(SP.OUT, model, "tq_forced_items.json"))["items"]}
        if set(fdoc) != set(ids) or set(rel) != {q for q in ids if issued[q] and not eans[q]}:
            raise SystemExit("%s: released list" % model)
        tq = {}
        for c in SP.CONDS2:
            rm = margins(model, "tq_release", c)[1]
            fm = margins(model, "tq_forced", c)[1]
            dd = {}
            for q in ids:
                r, fx = rel.get(q), fdoc[q]
                if fx["answered"] != F[q][1] or fx["value"] != F[q][0] or fx["conf"] != F[q][2]:
                    raise SystemExit("%s: forced items differ from tq_same_n_decomp (%s)" % (model, q))
                if r and r["value"] != n1[q]:
                    raise SystemExit("%s: released value" % model)
                dd[q] = dict(e=e[q], eans=eans[q], n1=n1[q], issued=issued[q], f=F[q][0], fans=F[q][1], conf=F[q][2],
                             mE=mE[c].get(("test", q)), negE=neg1[q], rel=(r["value"], rm["test|" + q], r["neg"]) if r else None,
                             mF=fm.get("test|" + q), negF=fx["neg"], lp=None)
            tq[c] = dd
        ordinary_doc, _, _ = SP.read_probe(model, qaqa=True)
        qitems = {x["id"]: x for x in ordinary_doc["items"]}
        qe = EQ.read_dump(model, qitems)
        qrel = {x["id"]: x for x in SP.read_json(os.path.join(SP.OUT, model, "qq_release_items.json"))["items"]}
        qf = {x["id"]: x for x in SP.read_json(os.path.join(SP.OUT, model, "qq_forced_items.json"))["items"]}
        if set(qf) != set(qitems) or set(qrel) != {r["id"] for r in qe["rows"] if r["issued"] and not r["eans"]}:
            raise SystemExit("%s: (QA)^2 released list" % model)
        lp, lpinfo = d_confidence(model, qe, qitems)
        qq = {}
        for c in SP.CONDS2:
            em, rm, fm = margins(model, "qaqa", c)[1], margins(model, "qq_release", c)[1], margins(model, "qq_forced", c)[1]
            dd = {}
            for r in qe["rows"]:
                q = r["id"]
                x, rr, fx = qitems[q], qrel.get(q), qf[q]
                if x["answer_sha1"] != r["e_sha1"] or x["answered"] != r["eans"] or x["value"] != r["e"] or fx["answered"] != r["fans"] or fx["value"] != r["f"] \
                        or (r["fans"] and fx["conf"] != r["conf"]):
                    raise SystemExit("%s: (QA)^2 rows differ (%s)" % (model, q))
                if rr and rr["value"] != r["n1"]:
                    raise SystemExit("%s: (QA)^2 released value" % model)
                dd[q] = dict(e=r["e"], eans=r["eans"], n1=r["n1"], issued=r["issued"], f=r["f"], fans=r["fans"], conf=r["conf"],
                             mE=em.get(q), negE=x["neg"], rel=(rr["value"], rm[q], rr["neg"]) if rr else None,
                             mF=fm.get(q), negF=fx["neg"], pilot=x["pilot"], lp=lp.get(q) if lp else None)
            qq[c] = dd
        out[model] = dict(tune_rows=tune_rows, tq=(ids, tq, (layer, s, n1cfg)), qq=([r["id"] for r in qe["rows"]], qq), d_info=lpinfo)
    return out


def d_confidence(model, qe, qitems):
    """{question: mean log-prob of single E's answer} from our ns_conf_pos.py output, checked as Raw.confidence checks."""
    import math
    tok = qe["ordinary"].replace(":", "--").replace(".", "-")     # newsets_gen.arm_token of a one-write arm
    path = os.path.join(SP.OUT, model, tok + "-at-prompt.json")
    if not os.path.exists(path):
        return None, "no file %s" % os.path.basename(path)
    d = SP.read_json(path)
    gdir = "Iteration_01/newsets/gen_prompt/qaqafull/%s/test" % model
    rows = {r["id"]: r for r in d["rows"]}
    if (d.get("arm") != qe["ordinary"] or d.get("model") != model or d.get("task") != "qaqafull" or d.get("split") != "test"
            or len(rows) != len(d["rows"]) or list(rows) != list(qitems) or list(rows) != [r["id"] for r in qe["rows"]]
            or d.get("generation_sha256") != SP.sha256_file(os.path.join(gdir, tok + ".json"))
            or not d.get("first_difference_check") or d.get("reference_sha256") != SP.sha256_file(os.path.join(gdir, "x.json"))):
        raise SystemExit("%s: %s does not belong to single E's pass" % (model, path))
    out = {}
    for q, x in qitems.items():
        if rows[q]["text_sha1"] != x["answer_sha1"]:
            raise SystemExit("%s: confidence of %s was scored on another answer" % (model, q))
        r = rows[q]
        score = r.get("mean_logp")
        if score is None:
            if x["answered"] or r.get("n_tokens") != 0:
                raise ValueError(f"{model}: missing answer confidence ({q})")
        elif (not math.isfinite(score) or score > 0 or not isinstance(r.get("n_tokens"), int) or r["n_tokens"] < 1
                or not math.isclose(score, r["sum_logp"] / r["n_tokens"], rel_tol=1e-6, abs_tol=1e-8)):
            raise ValueError(f"{model}: invalid answer confidence ({q})")
        if x["answered"]:
            out[q] = score
    return out, "%s (%s)" % (os.path.basename(path), {k: d.get(k) for k in ("greedy_reproduction", "scoring_batch_size") if k in d})


def judged_false(m, neg, t, exempt):
    return t is not None and m is not None and (not (exempt and neg)) and m < t


def computeZ(ids, d, tE, tF, exempt):
    """tE: threshold of single E's source (also used for the released answers); tF: of the strong arm (None = none)."""
    pos = {q: i for i, q in enumerate(ids)}
    import math
    if not ids or len(pos) != len(ids):
        raise ValueError("empty or duplicated evaluation roster")
    for q in ids:
        r = d[q]
        if ((r["rel"] is not None) != (r["issued"] and not r["eans"])
                or r["eans"] and (not r["issued"] or r["n1"] != r["e"])
                or r["rel"] is not None and r["rel"][0] != r["n1"]):
            raise ValueError(f"{q}: inconsistent ordinary/released answer")
        if r["eans"] and (r["mE"] is None or not math.isfinite(r["mE"])):
            raise ValueError(f"{q}: missing or non-finite ordinary self-judgement")
        if r["rel"] is not None and (r["rel"][1] is None or not math.isfinite(r["rel"][1])):
            raise ValueError(f"{q}: missing or non-finite release self-judgement")
        if r["fans"] and (r["conf"] is None or not math.isfinite(r["conf"]) or r["conf"] > 0
                          or tF is not None and (r["mF"] is None or not math.isfinite(r["mF"]))):
            raise ValueError(f"{q}: invalid strong answer confidence or self-judgement")
    dropE = {q: d[q]["eans"] and judged_false(d[q]["mE"], d[q]["negE"], tE, exempt) for q in ids}
    dropR = {q: d[q]["rel"] is not None and judged_false(d[q]["rel"][1], d[q]["rel"][2], tE, exempt) for q in ids}
    elig = [q for q in ids if d[q]["fans"] and d[q]["conf"] is not None]
    ranked = sorted(elig, key=lambda q: (-d[q]["conf"], pos[q]))
    fF = {q: judged_false(d[q]["mF"], d[q]["negF"], tF, exempt) for q in elig} if tF is not None else None

    def strong(n):
        if n > len(ranked):
            return None
        top = set(ranked[:n])
        return {q: (d[q]["f"] if q in top else 0) for q in ids}, 0

    def strong_plus(n):
        if fF is None or n > len(ranked):
            return None
        passed = [q for q in ranked if not fF[q]]
        keep = passed[:n]
        back = sorted([q for q in ranked if fF[q]], key=lambda q: (-d[q]["mF"], pos[q]))[:max(0, n - len(keep))]
        top = set(keep) | set(back)
        return {q: (d[q]["f"] if q in top else 0) for q in ids}, len(back)

    s1 = {q: 0 if dropE[q] else d[q]["e"] for q in ids}
    s2 = {q: 0 if dropE[q] else d[q]["n1"] for q in ids}
    s3 = {q: 0 if dropE[q] or dropR[q] else d[q]["n1"] for q in ids}
    nE, nN1 = sum(d[q]["eans"] for q in ids), sum(d[q]["issued"] for q in ids)
    kE, kR = sum(dropE.values()), sum(dropR.values())
    nrel = sum(d[q]["rel"] is not None for q in ids)
    n = dict(E=nE, N1=nN1, S1=nE - kE, S2=nN1 - kE, S3=nN1 - kE - kR)
    sysv = dict(E={q: d[q]["e"] for q in ids}, N1={q: d[q]["n1"] for q in ids}, S1=s1, S2=s2, S3=s3)
    opp = {}
    for k in ("S1", "S2", "S3"):
        opp["B(%s)" % k] = strong(n[k])
        opp["B+(%s)" % k] = strong_plus(n[k])
    opp["B(N1)"] = strong(nN1)
    # C3
    c3 = None
    if fF is not None:
        m = nrel - kR
        cand = [q for q in ranked if not d[q]["eans"] and not fF[q]]
        if len(cand) >= m:
            fill = set(cand[:m])
            c3 = {q: (d[q]["f"] if q in fill else s1[q]) for q in ids}, 0
    opp["C3"] = c3
    eb = [q for q in ranked if not d[q]["eans"]]
    fill_n1 = set(eb[:nrel]) if len(eb) >= nrel else None
    x = {q: (d[q]["f"] if q in fill_n1 else s1[q]) for q in ids} if fill_n1 is not None else None
    # D (where confidences exist)
    dd = {}
    if all(d[q]["lp"] is not None for q in ids if d[q]["eans"]) and any(d[q]["lp"] is not None for q in ids):
        for lab, sel in (("D_all", lambda q: True), ("D_other", lambda q: not d[q]["negE"])):
            cand = sorted([q for q in ids if d[q]["eans"] and sel(q)], key=lambda q: (d[q]["lp"], pos[q]))
            dr = set(cand[:kE]) if len(cand) >= kE else None
            dd[lab] = ({q: 0 if q in dr else d[q]["e"] for q in ids}, 0) if dr is not None else None
    opp.update(dd)
    diffs = [("S1-B", "S1", "B(S1)"), ("S2-B2", "S2", "B(S2)"), ("S3-B3", "S3", "B(S3)"), ("S1-B+", "S1", "B+(S1)"),
             ("S2-B+", "S2", "B+(S2)"), ("S3-B+", "S3", "B+(S3)"), ("S3-C3", "S3", "C3"), ("S3-S2", "S3", "S2"), ("ref", "N1", "B(N1)")]
    if dd:
        diffs += [("S1-D_all", "S1", "D_all"), ("S1-D_other", "S1", "D_other")]
    cells = {}
    for q in ids:
        c = {}
        for key, a, b in diffs:
            av = sysv[a]
            bv = sysv[b] if b in sysv else (opp[b][0] if opp[b] is not None else None)
            c[key] = None if bv is None else av[q] - bv[q]
        c["S2-(S1+fill)"] = s2[q] - x[q] if x else None
        cells[q] = c
    vals = lambda fl, src: [src(q) for q in ids if fl(q)]
    dE = vals(lambda q: dropE[q], lambda q: d[q]["e"])
    dR = vals(lambda q: dropR[q], lambda q: d[q]["rel"][0])
    rv = vals(lambda q: d[q]["rel"] is not None, lambda q: d[q]["rel"][0])
    fFv = [d[q]["f"] for q in elig if fF[q]] if fF is not None else None
    cnt = lambda v: (len(v), v.count(1), v.count(-1), v.count(0))
    summ = dict(n=n, nets={k: sum(v.values()) for k, v in sysv.items()},
                opp={k: (None if v is None else (sum(v[0].values()), v[1])) for k, v in opp.items()},
                dropE=cnt(dE), dropR=cnt(dR), released=cnt(rv), strong_false=cnt(fFv) if fFv is not None else None,
                n_strong=len(ranked), keys=[k for k, _, _ in diffs] + ["S2-(S1+fill)"])
    return cells, summ


def showZ(label, ids, cells, summ, ci, short):
    n, nets, opp = summ["n"], summ["nets"], summ["opp"]
    sf = summ["strong_false"]
    print("  %s: %d questions | E answers judged false %d (right %d, wrong %d, other %d) | released %d (right %d, wrong %d, other %d), "
          "judged false %d (right %d, wrong %d, other %d) | strong answers judged false %s" % (
              (label, len(ids)) + summ["dropE"] + summ["released"] + summ["dropR"] +
              ("not judged" if sf is None else "%d (right %d, wrong %d, other %d)" % sf,)))
    if not short:
        print("    answers: E %d, N1 %d, S1 %d, S2 %d, S3 %d (for available opponents, counts match their comparator); strong answers "
              "with confidence %d" % (n["E"], n["N1"], n["S1"], n["S2"], n["S3"], summ["n_strong"]))
        print("    nets: E %+d, N1 %+d, S1 %+d, S2 %+d, S3 %+d" % (nets["E"], nets["N1"], nets["S1"], nets["S2"], nets["S3"]))
        print("    opponents: " + " | ".join("%s %s" % (k, "not comparable" if v is None else ("%+d" % v[0]) + (
            " (added back %d)" % v[1] if k.startswith("B+") else "")) for k, v in opp.items()))
    print("    " + " | ".join("%s %s" % (k, "not comparable" if cells[ids[0]][k] is None else ci([cells[q][k] for q in ids]))
                             for k in summ["keys"]))


def tableZ():
    import tq_same_n_decomp as TQ
    import newsets_score as NS
    ci_tq = TQ.ps.ci

    def ci_qq(v):
        tot, lo, hi = NS.joint_ci([list(v)])
        return "%+d [%+d, %+d]" % (tot, lo, hi)

    A = load_all()
    print("# selfjudge stage one, rule Z (work queue 2026-11-22_queue_e, '1 supplement', 17:16), Llama-3.1-8B and Qwen3.5-9B")
    print("# rule Z: 'other'-class answer judged false when margin < 0; negated-premise openings exempt (NEG_RE of selfjudge_probe.py)")
    print("# P' thresholds (no write, E source, TruthfulQA tune; for the P' version): " + " | ".join(
        "%s %s" % (m, ft(choose(A[m]["tune_rows"]["baseline"])[1][0])) for m in ORDER))
    for m in ORDER:
        print("# D confidence %s: %s" % (m, A[m]["d_info"]))
    # references (rule Z, no write)
    ok = True
    for m in ORDER:
        ids, tq, _ = A[m]["tq"]
        c, _ = computeZ(ids, tq["baseline"], 0.0, 0.0, True)
        if any(c[q]["ref"] is None or c[q]["S2-(S1+fill)"] is None for q in ids):
            raise ValueError(f"{m}: registered reference count unreachable")
        got, fill = ci_tq([c[q]["ref"] for q in ids]), ci_tq([c[q]["S2-(S1+fill)"] for q in ids])
        qids, qq = A[m]["qq"]
        held = [q for q in qids if not qq["baseline"][q]["pilot"]]
        qc, _ = computeZ(held, qq["baseline"], 0.0, 0.0, True)
        if any(qc[q]["ref"] is None for q in held):
            raise ValueError(f"{m}: registered QA2 reference count unreachable")
        qgot = sum(qc[q]["ref"] for q in held)
        print("# reference %s: N1 - strong TruthfulQA %s (expected %s) | rule Z: S2 - (S1 first pass + fill) TruthfulQA %s (expected %s) | "
              "(QA)^2 470 N1 - strong %+d (expected %+d)" % (m, got, EXPECT_TQ[m], fill, EXPECT_FILL[m], qgot, EXPECT_QQ470[m]))
        ok = ok and got == EXPECT_TQ[m] and fill == EXPECT_FILL[m] and qgot == EXPECT_QQ470[m] and len(held) == 470
    if not ok:
        raise SystemExit("REFERENCE MISMATCH: stopping before any rule-Z reading")
    print("# references reproduced")
    versions = [("rule Z, no write (MAIN)", "baseline", lambda m: (0.0, 0.0), True, False),
                ("rule Z, true/false E strength 3", "E_a3", lambda m: (0.0, 0.0), True, True),
                ("rule Z without the negated-premise exemption, no write", "baseline", lambda m: (0.0, 0.0), False, True),
                ("rule P', no write (no must-answer + E threshold)", "baseline",
                 lambda m: (choose(A[m]["tune_rows"]["baseline"])[1][0], None), True, True)]
    for title, cond, th, exempt, short in versions:
        print("=" * 100)
        print("SELF-JUDGEMENT %s" % title)
        for m in ORDER:
            tE, tF = th(m)
            ids, tq, (layer, s, n1cfg) = A[m]["tq"]
            print(" %s (E-source t=%s, strong-arm t=%s) TruthfulQA test, E L%d strength %s, N1 %s" % (
                m, ft(tE), "none" if tF is None else ft(tF), layer, s, n1cfg))
            c, sm = computeZ(ids, tq[cond], tE, tF, exempt)
            showZ("TruthfulQA test", ids, c, sm, ci_tq, short)
            qids, qq = A[m]["qq"]
            for lab, sub in (("(QA)^2 470 held out", [q for q in qids if not qq[cond][q]["pilot"]]), ("(QA)^2 570", qids)):
                c, sm = computeZ(sub, qq[cond], tE, tF, exempt)
                showZ(lab, sub, c, sm, ci_qq, short)
        print(" two models together:")
        common = sorted(set(A[ORDER[0]]["tq"][0]) & set(A[ORDER[1]]["tq"][0]))
        cs = {m: computeZ(common, A[m]["tq"][1][cond], th(m)[0], th(m)[1], exempt) for m in ORDER}
        keys = [k for k in cs[ORDER[0]][1]["keys"] if k in cs[ORDER[1]][1]["keys"]]
        print("  TruthfulQA test, %d common questions: " % len(common) + " | ".join(
            "%s %s" % (k, "not comparable" if any(cs[m][0][common[0]][k] is None for m in ORDER)
                       else ci_tq([sum(cs[m][0][q][k] for m in ORDER) for q in common])) for k in keys))
        qids = A[ORDER[0]]["qq"][0]
        if qids != A[ORDER[1]]["qq"][0]:
            raise SystemExit("(QA)^2 lists differ")
        for lab, sub in (("470 held out", [q for q in qids if not A[ORDER[0]]["qq"][1][cond][q]["pilot"]]), ("570", qids)):
            cs = {m: computeZ(sub, A[m]["qq"][1][cond], th(m)[0], th(m)[1], exempt) for m in ORDER}
            keys = [k for k in cs[ORDER[0]][1]["keys"] if k in cs[ORDER[1]][1]["keys"]]
            line = []
            for k in keys:
                if any(cs[m][0][sub[0]][k] is None for m in ORDER):
                    line.append("%s not comparable" % k)
                else:
                    tot, lo, hi = NS.joint_ci([[cs[m][0][q][k] for q in sub] for m in ORDER])
                    line.append("%s %+d [%+d, %+d]" % (k, tot, lo, hi))
            print("  (QA)^2 %s, %d questions: %s" % (lab, len(sub), " | ".join(line)))


def main():
    if len(sys.argv) == 3 and sys.argv[1] in ("export-tq", "export-qq") and sys.argv[2] in ORDER:
        (export_tq if sys.argv[1] == "export-tq" else export_qq)(sys.argv[2])
    elif sys.argv[1:] == ["table"]:
        table()
    elif sys.argv[1:] == ["tableZ"]:
        tableZ()
    else:
        raise SystemExit("usage: selfjudge_stage1.py export-tq MODEL | export-qq MODEL | table | tableZ")


if __name__ == "__main__":
    main()
