#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-22), work queue discussion/work_queue/2026-11-22_queue_j.md,
# section 6 (23:28): the method on top of the honest sentence (queue I step one, SN1) with the same self-judge on both
# sides. No judge call, no key, no generation. Nothing is chosen.
# Candidates (TruthfulQA tune and test): every answer SN1 hands in (single E on the prompt x_E2, and the answers released
# on its blanks; strength and caps = queue I's tune choice, newsets/h22/n1s_choice_<model>.json), built read-only with
# diag/n1_on_sentence.py (build / restricted, S.system), work dirs eval_extra/judge_<split>_h22c4_<split> and
# eval_extra/judge_<split>_n1s_<split>; SE = single E on x_E2 (its answers are SN1's first pass).
# Self-judge scores: diag/selfjudge_two_answers.score1 unchanged (batch size 1 set in memory, path identity against the
# stored batch-8 tj_s0 scores first); conditions baseline (five models) and E_a3 (the three with a true/false E).
# Sources: st1_items = every answered SE / SN1 answer (id split|q|E for single E's answer, split|q|R for a released one);
# st1_chk / st1_chka = the same 16 test answers (seed 20261122) in the same order, st1_chkb = those 16 shuffled; the three
# must agree exactly (queue section 6 check).
# Systems: SN1z / SEz = every handed-in answer through rule Z (margin < 0 -> no answer; NEG_RE openings exempt); S0z and
# A1z exactly as diag/selfjudge_two_answers.py (batch-1 scores). Net = right - wrong with the cell's verdict value kept
# even when the cell is blank (the main table's convention, as plain_honesty_fill.py / n1_on_sentence.py); answer counts
# count only the cells that are answered; rule Z and the equal-count cut turn a handed-in answer into 0.
# Readings (section 6): SN1z - S0z (main), SN1 - S0z, SN1z - SN1, SEz - S0z; written minus not written of SN1z (three
# models); SN1z and A1z side by side. Natural and equal answer counts (the side with more answers drops its handed-in
# answers from the lowest margin up, ties by roster order, per model; margins of the comparison's condition, own
# condition for written minus not written). Questions: those of selfjudge_two_answers.load (S0 and conf_E2 verdicts)
# that SN1's list also has; models together on the sorted common questions; intervals policy_select2.ci.
# usage (host, from /data/sanshizhao/Truth_Decoupling):
#   python3 -B Iteration_01/diag/selfjudge_stack.py export MODEL     (items + the SN1 reference check)
#   launchers/explore_20261120/selfjudge_stack_1122.sh MODEL GPU TAG  (= selfjudge_stack.py score MODEL, container)
#   python3 -B Iteration_01/diag/selfjudge_stack.py read
import json
import os
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets"]
import selfjudge_probe as SP          # noqa: E402
import selfjudge_two_answers as TA    # noqa: E402

SRC = ("st1_items", "st1_chk", "st1_chka", "st1_chkb")
SP.SOURCES.update(SRC)
ORDER, WRITE, V = TA.ORDER, TA.WRITE, SP.V
REF_SN1 = {"llama3.1-8b": (130, 510), "qwen3.5-9b": (200, 424), "qwen3.8-27b": (217, 487), "gemma3-12b": (194, 513),
           "gemma3-27b": (209, 515)}
REF5 = (947, 2438)
CHOICE = "Iteration_01/newsets/h22/n1s_choice_%s.json"


def sn1(model, i):
    """{q: dict(SE=(v, a, text, cond), SN1=(v, a, text, cond))} on split i, and the reference list of n1_on_sentence.test."""
    import n1_on_sentence as NS
    ch = SP.read_json(CHOICE % model)
    sc, cfg = ch["strength"], tuple(ch["cfg"])
    cq = NS.restricted(model, i)
    sp, tx = NS.build(model, i, "x_E2", sc, cq)
    out = {}
    res = {}
    for name, c in (("SE", ("E", sc)), ("SN1", cfg)):
        states, ans, recs = NS.S.system(sp, tx, c)
        for q, x, a, r in zip(sp.qs, states, ans, recs):
            if r["p"] != q:
                raise SystemExit("%s: record order" % model)
            res.setdefault(q, {})[name] = (V[x], bool(a), r["text"], r["cond"])
    out = res
    ref_qs = None
    if i == 1:
        main_e = NS.run_sys(*NS.build(model, 1, "x", A_KEY(model)[2], NS.restricted(model, 1)), ("E", A_KEY(model)[2]))
        D = NS.MP.MT.Data(A_KEY(model)[0], 1)
        st1, _ = NS.ps.load(D.m["d"][1], "pro")
        ref_qs = [q for q in sp.qs if q in main_e and st1.get((q, "baseline_conf_E2"), "?") != "?"]
    return out, list(sp.qs), ref_qs, (sc, list(cfg))


def A_KEY(model):
    import iti_table as A
    return A.KEY[model]


def export(model):
    step1 = SP.read_json(os.path.join(SP.OUT, model, "items.json"))
    qtext = {(x["split"], x["id"]): x["question"] for x in step1["items"]}
    items, info = [], {}
    for i, name in TA.SPLITS:
        res, qs, ref_qs, (sc, cfg) = sn1(model, i)
        if ref_qs is not None:
            got = (sum(res[q]["SN1"][0] for q in ref_qs), sum(res[q]["SN1"][1] for q in ref_qs))
            print("check %s test, %d q: SN1 %+d (%d) [ref %+d (%d)] -> %s" % (
                model, len(ref_qs), *got, *REF_SN1[model], "OK" if got == REF_SN1[model] else "MISMATCH"))
            if got != REF_SN1[model]:
                raise SystemExit("SN1 REFERENCE MISMATCH")
        nE = nR = 0
        for q in qs:
            e, n = res[q]["SE"], res[q]["SN1"]
            if e[1]:
                items.append(dict(id="%s|%s|E" % (name, q), split=name, question=qtext[(name, q)], answer=e[2],
                                  answer_sha1=SP.sha1(e[2]), value=e[0], answered=True, neg=bool(SP.NEG_RE.match(e[2]))))
                nE += 1
            if n[1]:
                if n[3] == e[3]:
                    if not e[1] or SP.sha1(n[2]) != SP.sha1(e[2]) or n[0] != e[0]:
                        raise SystemExit("%s %s: SN1 keeps a single-E cell that differs (%s)" % (model, name, q))
                    continue
                if e[1]:
                    raise SystemExit("%s %s: SN1 replaces an answered single-E cell (%s)" % (model, name, q))
                items.append(dict(id="%s|%s|R" % (name, q), split=name, question=qtext[(name, q)], answer=n[2],
                                  answer_sha1=SP.sha1(n[2]), value=n[0], answered=True, neg=bool(SP.NEG_RE.match(n[2]))))
                nR += 1
        info[name] = dict(n=len(qs), se_answers=nE, released=nR, strength=sc, cfg=cfg)
        print("%s %s: %d questions, single-E answers %d, released answers %d (E %s, %s)" % (model, name, len(qs), nE, nR, sc, cfg))
    SP.save_items(os.path.join(SP.OUT, model, "st1_items_items.json"),
                  dict(model=model, source="st1_items", splits=info, items=items))
    import numpy as np
    pool = [x for x in items if x["split"] == "test"]
    pick = sorted(np.random.default_rng(20261122).choice(len(pool), 16, replace=False).tolist())
    chk = [pool[k] for k in pick]
    perm = np.random.default_rng(20261123).permutation(16).tolist()
    for src, xs in (("st1_chk", chk), ("st1_chka", chk), ("st1_chkb", [chk[k] for k in perm])):
        SP.save_items(os.path.join(SP.OUT, model, src + "_items.json"), dict(model=model, source=src, items=xs))
    print("  st1_items %d answers; st1_chk / st1_chka / st1_chkb 16 each" % len(items))


def score(model):
    TA.score1(model, list(SRC))


def check_b(model):
    import numpy as np
    maps = {}
    for s in ("st1_chk", "st1_chka", "st1_chkb"):
        doc, _, z = SP.read_probe(model, source=s)
        maps[s] = {x["id"]: {c: float(z[c + "__margin"][j]) for c in SP.conds_of(model, "x")}
                   for j, x in enumerate([x for x in doc["items"] if x["answered"]])}
    base = maps["st1_chk"]
    if len(base) != 16 or any(set(maps[s]) != set(base) for s in maps):
        raise SystemExit("%s: check items differ" % model)
    d = {s: max(abs(maps[s][k][c] - base[k][c]) for k in base for c in base[k]) for s in ("st1_chka", "st1_chkb")}
    _, _, z = SP.read_probe(model, source="st1_items")
    return d["st1_chka"] == 0 and d["st1_chkb"] == 0, "check (b) %s: 16 statements, second pass max |diff| %.6f, shuffled max |diff| %.6f" % (
        model, d["st1_chka"], d["st1_chkb"])


# ------------------------------------------------------------------------------------------------------------------ #
def load(model, base):
    """{split: (qs, {q: {sys: {cond: (v, a, margin)}}})} for S0z / A1z (from base) and SN1 / SN1z / SE / SEz."""
    doc, _, z = SP.read_probe(model, source="st1_items")
    conds = [c for c in ("baseline", "E_a3") if c in SP.conds_of(model, "x")]
    ans = [x for x in doc["items"] if x["answered"]]
    mg = {x["id"]: ({c: float(z[c + "__margin"][j]) for c in conds}, x["neg"], x["answer_sha1"]) for j, x in enumerate(ans)}
    out = {}
    for i, name in TA.SPLITS:
        res, qs, _, _ = sn1(model, i)
        ids, rows, _ = base[name]
        use = [q for q in ids if q in rows and q in res]
        per = {}
        for q in use:
            e, n = res[q]["SE"], res[q]["SN1"]
            d = {}
            for c in conds:
                ts = TA.systems(rows[q], c)

                def cell(t):
                    v, a, tag = t
                    return (v, a, rows[q][tag]["m"][c] if a else None)

                def mine(x, key):
                    v, a = x[0], x[1]
                    if not a:
                        return (v, False, None), (v, False, None)
                    m, neg, sha = mg[key]
                    if sha != SP.sha1(x[2]):
                        raise SystemExit("%s: score of %s is not this answer" % (model, key))
                    raw = (v, True, m[c])
                    return raw, ((0, False, None) if (not neg and m[c] < 0) else raw)
                se, sez = mine(e, "%s|%s|E" % (name, q))
                kn = "%s|%s|E" % (name, q) if (n[1] and n[3] == e[3]) else "%s|%s|R" % (name, q)
                snv, snz = mine(n, kn)
                d[c] = {"S0z": cell(ts["S0z"]), "A1z": cell(ts["A1z"]), "SE": se, "SEz": sez, "SN1": snv, "SN1z": snz}
            per[q] = d
        out[name] = (use, per)
    return out


def equal(qs, per, a, ca, b, cb):
    va = {q: per[q][ca][a] for q in qs}
    vb = {q: per[q][cb][b] for q in qs}
    na, nb = sum(x[1] for x in va.values()), sum(x[1] for x in vb.values())

    def cut(vals, k):
        pos = {q: i for i, q in enumerate(qs)}
        cand = sorted([q for q in qs if vals[q][1]], key=lambda q: (vals[q][2], pos[q]))
        drop = set(cand[:k])
        return {q: ((0, False, None) if q in drop else vals[q]) for q in qs}
    if na > nb:
        va = cut(va, na - nb)
    elif nb > na:
        vb = cut(vb, nb - na)
    return va, vb, (na, nb)


def block(label, models, data, name, qs, conds, ci):
    print("  %s" % label)
    for c in conds:
        print("    [%s] nets (answers): %s" % (c, " | ".join(
            "%s %+d (%d)" % (k, sum(data[m][name][1][q][c][k][0] for m in models for q in qs),
                             sum(data[m][name][1][q][c][k][1] for m in models for q in qs))
            for k in ("S0z", "A1z", "SE", "SEz", "SN1", "SN1z"))))
    comps = [("%s[%s]-%s[%s]" % (a, c, b, c), a, c, b, c) for c in conds
             for a, b in (("SN1z", "S0z"), ("SN1", "S0z"), ("SN1z", "SN1"), ("SEz", "S0z"))]
    if "E_a3" in conds:
        comps.append(("SN1z[E_a3]-SN1z[baseline]", "SN1z", "E_a3", "SN1z", "baseline"))
    nat, eq = [], []
    for key, a, ca, b, cb in comps:
        dn, de, cnt = {}, {}, []
        for m in models:
            P = data[m][name][1]
            va, vb, nn = equal(qs, P, a, ca, b, cb)
            cnt.append(nn)
            for q in qs:
                dn[q] = dn.get(q, 0) + P[q][ca][a][0] - P[q][cb][b][0]
                de[q] = de.get(q, 0) + va[q][0] - vb[q][0]
        nat.append("%s %s (answers %d vs %d)" % (key, ci([dn[q] for q in qs]), sum(x[0] for x in cnt), sum(x[1] for x in cnt)))
        eq.append("%s %s (at %d)" % (key, ci([de[q] for q in qs]), sum(min(x) for x in cnt)))
    print("    natural answer counts:")
    for x in nat:
        print("      " + x)
    print("    equal answer counts (per model, the side with more answers drops its lowest-margin handed-in answers):")
    for x in eq:
        print("      " + x)


def read():
    import iti_table as A
    ci = A.ps.ci
    print("# selfjudge stack (work queue 2026-11-22_queue_j section 6, 23:28); TruthfulQA, DeepSeek Pro verdicts; self-judge "
          "scores batch size 1")
    print("# net = right - wrong with the verdict value of a blank cell kept (main-table convention); answers = answered cells only")
    ok = True
    for m in ORDER:
        o, line = check_b(m)
        print("# " + line + (" -> OK" if o else " -> FAILED"))
        ok &= o
    if not ok:
        raise SystemExit("CHECK (b) FAILED")
    data = {}
    for m in ORDER:
        data[m] = load(m, TA.load(m))
    for _, name in TA.SPLITS:
        print("=" * 110)
        print("TruthfulQA %s" % name)
        for m in ORDER:
            conds = [c for c in ("baseline", "E_a3") if c in SP.conds_of(m, "x")]
            block("%s: %d questions" % (m, len(data[m][name][0])), (m,), data, name, data[m][name][0], conds, ci)
        c5 = sorted(set.intersection(*(set(data[m][name][0]) for m in ORDER)))
        block("five models, %d common questions (no write)" % len(c5), ORDER, data, name, c5, ["baseline"], ci)
        c3 = sorted(set.intersection(*(set(data[m][name][0]) for m in WRITE)))
        block("three models (%s), %d common questions" % (", ".join(WRITE), len(c3)), WRITE, data, name, c3,
              ["baseline", "E_a3"], ci)


def check5():
    """The five-model SN1 reference on the common list of n1_on_sentence.test."""
    per = {}
    for m in ORDER:
        res, _, ref_qs, _ = sn1(m, 1)
        per[m] = {q: res[q]["SN1"] for q in ref_qs}
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    got = (sum(per[m][q][0] for m in ORDER for q in common), sum(per[m][q][1] for m in ORDER for q in common))
    print("check five models, %d common q: SN1 %+d (%d) [ref %+d (%d)] -> %s" % (
        len(common), *got, *REF5, "OK" if got == REF5 else "MISMATCH"))
    if got != REF5:
        raise SystemExit(1)


def main():
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "export" and a[1] in ORDER:
        export(a[1])
    elif len(a) == 2 and a[0] == "score" and a[1] in ORDER:
        score(a[1])
    elif a == ["check5"]:
        check5()
    elif a == ["read"]:
        read()
    else:
        raise SystemExit("usage: selfjudge_stack.py export MODEL | score MODEL | check5 | read")


if __name__ == "__main__":
    main()
