#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-23), work queue discussion/work_queue/2026-11-22_queue_j.md,
# section 7 (00:53, commit 64de088): pick one of the stacked answer (method on top of the honest sentence) and the
# no-sentence single-E answer by the model's own self-judgement. No judge call, no key, no generation, nothing chosen.
# Candidates per model and question (all read-only from existing files):
#   St  TruthfulQA / (QA)^2: SN1 (diag/n1_on_sentence.py build + S.system at queue I's choice, as selfjudge_stack.sn1;
#       (QA)^2: diag/n1_on_sentence2.qaqa's construction, caps chosen on the 100 pilot questions); fact sets: SE.
#   E   no-sentence single E: TruthfulQA the main table's (selfjudge_probe items.json); (QA)^2 qaqa_pilot_table.Raw arm
#       base.e(single_E) of qaqa_tune_choice_fixedE_20261122.json; fact sets "only E" of n1_on_sentence2.fact.
#   N1  no-sentence N1 (TruthfulQA main table, (QA)^2 Raw.n1_system at the fixed-E choice; none on the fact sets).
#   S0  honest sentence, no write;  O  original model, no write (TruthfulQA baseline_x; (QA)^2 / fact "x").
# Self-judge: margin of selfjudge_probe.statement(question, answer), batch size 1 (selfjudge_two_answers.score1), no
# write for all five models and E_a3 for the three with a true/false E. A score is keyed by the statement: existing
# batch-1 scores of the same model, statement and condition (tj1_s0 / tj1_conf / tj1_n1 / st1_items) are reused; the rest
# are scored here (source sp_items, one item per distinct statement; sp_chk / sp_chka / sp_chkb = 16 statements twice in
# order and once shuffled, which must agree exactly).
# Systems ("different" = different statements; "answered" = answered under the existing scorer; net keeps the verdict
# value of a blank cell, the main-table convention; answer counts count answered cells only):
#   B1  one side answered -> that one; both answered and the same statement -> E's; both answered and different -> the
#       higher margin, a tie keeps E's; neither answered -> E's cell (not written in the rules; reported).
#   B1z B1 + rule Z (margin < 0 -> no answer, NEG_RE openings exempt).  TB1 / TB1z: the same between S0 and O (ties O).
#   References St, E, N1, S0, O and their rule-Z versions (*z).
#   Section 7a (01:19, a5ef866): B1d = St's answer opens with a negated premise (NEG_RE), E answered and the statements
#   differ -> the higher margin (tie E); otherwise St's cell. B1o = B1 with E replaced by O (tie O). Readings B1d - St /
#   E / B1 and B1o - St / O, and how often B1d triggers / hands in E's answer.
# Readings (queue 7): 1 one-right-one-wrong pairs picked right; 2 B1 - St / E / N1 / TB1; 3 B1z - Stz / Ez / TB1z;
# 4 (QA)^2 by official type: B1 - St, B1 - E; 5 written minus not written (B1, B1z). Natural and equal answer counts
# (the side with more answers drops its handed-in answers from the lowest margin up, ties by roster order, per model;
# own condition for 5). Totals: five models (no write), three models (both conditions), on the common questions.
# Intervals: TruthfulQA policy_select2.ci; (QA)^2 and fact sets newsets_score.joint_ci on per-question sums.
# usage (host, from /data/sanshizhao/Truth_Decoupling):
#   python3 -B Iteration_01/diag/selfjudge_stack_pick.py check1           (reference numbers, no score needed)
#   python3 -B Iteration_01/diag/selfjudge_stack_pick.py export MODEL     (sp_items: statements without a batch-1 score)
#   launchers/explore_20261120/selfjudge_stack_pick_1122.sh MODEL GPU TAG (= selfjudge_stack_pick.py score MODEL)
#   python3 -B Iteration_01/diag/selfjudge_stack_pick.py read qaqa|tqa|fact [MODEL,MODEL]   (container, no GPU)
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import selfjudge_probe as SP          # noqa: E402
import selfjudge_two_answers as TA    # noqa: E402
import selfjudge_stack as SK          # noqa: E402

SRC = ("sp_items", "sp_chk", "sp_chka", "sp_chkb")
SP.SOURCES.update(SRC)
ORDER, WRITE, V = TA.ORDER, TA.WRITE, SP.V
REUSE = ("tj1_s0", "tj1_conf", "tj1_n1", "st1_items")
FACT = (("popqa", "test"), ("popqa", "test2"), ("triviaqa", "test"), ("triviaqa", "test2"))
REF_QQ470 = {"O": (-224, 1954), "S0": (40, 2013), "E": (248, 1632), "St": (294, 2139), "N1": (254, 2027)}
REF_TQ520 = {"S0": (706, 2269), "E": (560, 1818), "St": (947, 2438)}
FACT_FILE = "Iteration_01/results/write_positions/n1_on_sentence_factqa_five_models_20261122.txt"


def cand(v, a, q, t):
    """(value, answered, statement or None, negated-premise opening, question, answer)."""
    if not a:
        return (v, False, None, False, None, None)
    return (v, True, SP.statement(q, t), bool(SP.NEG_RE.match(t)), q, t)


# ------------------------------------------------------------------------------------------------------------------ #
def tqa(model):
    """{split: (qs, {q: {St, E, N1, S0, O}})}; questions of selfjudge_two_answers.load that SN1's list also has."""
    import tq_same_n_decomp as TQ
    base = TA.load(model)
    step1 = SP.read_json(os.path.join(SP.OUT, model, "items.json"))
    qtext = {(x["split"], x["id"]): x["question"] for x in step1["items"]}
    e_it = {(x["split"], x["id"]): x for x in step1["items"]}
    n1 = TA.n1_answers(model)
    s0 = {x["id"]: x for x in SP.read_json(os.path.join(SP.OUT, model, "tj_s0_items.json"))["items"]}
    display, layer, s, tune, test = SP.main_lists(TQ.A, model)
    out = {}
    for i, name in TA.SPLITS:
        res, _, _, _ = SK.sn1(model, i)
        ids, rows, _ = base[name]
        sp, tx, _, qs_main = (tune, test)[i]
        st_o, an_o, rec_o = TQ.S.system(sp, tx, ("base",))
        orig = {}
        for k, q in enumerate(qs_main):
            if rec_o[k]["p"] != q:
                raise SystemExit("%s: original-model record order" % model)
            orig[q] = (V[st_o[k]], bool(an_o[k]), rec_o[k]["text"])
        use = [q for q in ids if q in rows and q in res]
        per = {}
        for q in use:
            qt = qtext[(name, q)]
            r = rows[q]
            ei = e_it[(name, q)]
            n = res[q]["SN1"]
            nx = n1.get((name, q))
            sx = s0[name + "|" + q]
            d = {"St": cand(n[0], n[1], qt, n[2]),
                 "E": cand(V[ei["state"]], ei["answered"], qt, ei["answer"]),
                 "N1": cand(r["N"]["v"], r["N"]["a"], qt, nx["answer"] if nx else ""),
                 "S0": cand(r["S"]["v"], r["S"]["a"], qt, sx["answer"]),
                 "O": cand(*orig[q][:2], qt, orig[q][2])}
            # statements must equal the ones the existing scores are bound to
            if d["N1"][1] and SP.sha1(d["N1"][2]) != r["N"]["st"] or d["S0"][1] and SP.sha1(d["S0"][2]) != r["S"]["st"]:
                raise SystemExit("%s: statement of %s differs from the stored score's" % (model, q))
            per[q] = d
        out[name] = (use, per, None)
    return out


# ------------------------------------------------------------------------------------------------------------------ #
def n1_arms(R, s, cap_e, cap_own):
    """The arm whose answer N1 hands in per question (Raw.n1_system's logic; values checked against it)."""
    import qaqa_pilot_table as Q
    v_ref, a_ref, _ = R.n1_system(s, cap_e, cap_own)
    e = R.e(s)
    arm = {}
    for q in R.ids:
        if not R.blank(e, q):
            arm[q] = e
            continue
        own = R.blank("x", q)
        cap = cap_own if own else cap_e
        lad = () if cap is None else Q.LAD[:Q.LAD.index(cap) + 1]
        k = next((k for k in lad if not R.blank(R.ec(s, k), q)), None)
        if k is not None and not own and Q.ro.norm(R.text(R.ec(s, k), q)) == Q.ro.norm(R.text("x", q)):
            k = None
        arm[q] = e if k is None else R.ec(s, k)
        a = k is not None or not R.blank(e, q)
        if R.cell(arm[q], q) != v_ref[q] or a != bool(a_ref[q]):
            raise SystemExit("%s: rebuilt N1 differs from Raw.n1_system on %s" % (R.model, q))
    return arm, v_ref, a_ref


def qaqa(model):
    """{"qaqa": (ids, {q: {St, E, N1, S0, O}}, {q: (qtype, pilot)})}, n1_on_sentence2.qaqa's construction."""
    import qaqa_pilot_table as Q
    import n1_on_sentence2 as N2
    NSD = Q.S.NS
    bankp = os.path.join(NSD, "bank/qaqa_full_bank_20261122.jsonl")
    with open(bankp, encoding="utf-8") as f:
        bank = {r["id"]: r for r in (json.loads(l) for l in f)}
    pilot = set(json.load(open(os.path.join(NSD, "splits/qaqa_pilot_20261122.json")))["ids"])
    fx = json.load(open(os.path.join(NSD, "splits/qaqa_tune_choice_fixedE_20261122.json")))[model]
    el, _, cl = Q.SET[model][:3]
    ssel = N2.choice(model)["strength_dot"]
    base = Q.Raw(model, "qaqafull", bankp, ("test_full570", "test_full570_tuned", "test_full570_strong"))
    R = Q.Raw(model, "qaqafull", bankp, ["test_full570_h22n", "test_full570_h22n1", "test_full570_h22z"])
    R.gdir = Path(NSD) / "gen_prompt" / "h22" / "qaqafull" / model / "test"
    arms = ["x_E2", "x_E2:E%d.w.p%s" % (el, ssel)] + ["x_E2:E%d.w.p%s+C%d.w.m%s" % (el, ssel, cl, k) for k in N2.LADK]
    for a in arms:
        R.need(a)
        alias = a.replace("x_E2:", "x:") if ":" in a else "x"
        R.rows[alias], R.cond[alias] = R.rows[a], R.cond[a]
    if R.ids != base.ids:
        raise SystemExit("banks differ")
    pq = [q for q in R.ids if q in pilot]
    grid = [((ce, co), sum(R.n1_system(ssel, ce, co)[0][q] for q in pq)) for ce in Q.CAPS for co in Q.CAPS]
    best = max(grid, key=lambda x: (x[1], -grid.index(x)))[0]
    sarm, sv, sa = n1_arms(R, ssel, best[0], best[1])
    narm, nv, na = n1_arms(base, fx["n1_strength"], fx["n1_cap_e"], fx["n1_cap_own"])
    earm = base.e(fx["single_E"])
    base.need("x")
    base.need(earm)
    per = {}
    for q in R.ids:
        qt = bank[q]["question"]
        per[q] = {"St": cand(sv[q], bool(sa[q]), qt, R.text(sarm[q], q)),
                  "E": cand(base.cell(earm, q), not base.blank(earm, q), qt, base.text(earm, q)),
                  "N1": cand(nv[q], bool(na[q]), qt, base.text(narm[q], q)),
                  "S0": cand(R.cell("x", q), not R.blank("x", q), qt, R.text("x", q)),
                  "O": cand(base.cell("x", q), not base.blank("x", q), qt, base.text("x", q))}
    info = {q: (base.qtype[q], q in pilot) for q in R.ids}
    print("# (QA)^2 %s: s_sel %s, SN1 caps %s / %s; no-sentence single E %s, N1 %s caps %s / %s" % (
        model, ssel, best[0], best[1], fx["single_E"], fx["n1_strength"], fx["n1_cap_e"], fx["n1_cap_own"]))
    return {"qaqa": (list(R.ids), per, info)}


# ------------------------------------------------------------------------------------------------------------------ #
def fact(model):
    """{"task split": (qs, {q: {St (= SE), E, S0, O}}, None)}, n1_on_sentence2.fact's construction."""
    import ns_prompt_table as NP
    import n1_on_sentence2 as N2
    G, AM, NSS = NP.G, NP.AM, NP.S
    M = NSS.models()
    out = {}
    el, main_s, *_ = NP.MAIN[model]
    ssel = N2.choice(model)["strength_dot"]
    for task, split in FACT:
        b, bp = NP.bank(task, split)
        xsp, xtx, _, _, _, _ = NP.load(task, model, M[model], split)
        d = {}
        for name, cfg in (("O", ("base",)), ("E", ("E", NP.dash(main_s)))):
            states, ans, recs = NSS.system(xsp, xtx, cfg)[:3]
            d[name] = {}
            for q, v, a, r in zip(xsp.qs, NSS.vals(states), ans, recs):
                if r["p"] != q:
                    raise SystemExit("record order")
                d[name][q] = (v, bool(a), r["text"])
        for name, arm in (("S0", "x_E2"), ("St", "x_E2:E%d.w.p%s" % (el, ssel))):
            p = os.path.join(NSS.NS, "gen_prompt", "h22", task, model, split, G.arm_token(arm) + ".json")
            g = G.load_generation(p, SimpleNamespace(task=task, split=split, model=model, items=bp), list(b))
            if g["contract"]["arm"] != arm or g["contract"].get("positions") != ("prompt" if ":" in arm else None):
                raise SystemExit("%s is not the pass %s" % (p, arm))
            d[name] = {}
            for r in g["rows"]:
                blank = bool(r["rule_abstain"]) or not r["text"].strip()
                v = AM.verdict(r["text"], b[r["id"]]["gold"], not blank)
                d[name]["%s:%s" % (task, r["id"])] = ((v == "R") - (v == "W"), not blank, r["text"])
        if list(d["S0"]) != list(xsp.qs) or list(d["St"]) != list(xsp.qs):
            raise SystemExit("question order")
        per = {}
        for q in xsp.qs:
            qt = b[q.split(":", 1)[1]]["question"]
            per[q] = {k: cand(d[k][q][0], d[k][q][1], qt, d[k][q][2]) for k in ("St", "E", "S0", "O")}
        out["%s %s" % (task, split)] = (list(xsp.qs), per, None)
    return out


def all_sets(model, which=("tqa", "qaqa", "fact")):
    out = {}
    if "tqa" in which:
        t = tqa(model)
        out["TruthfulQA test"], out["TruthfulQA tune"] = t["test"], t["tune"]
    if "qaqa" in which:
        out.update(qaqa(model))
    if "fact" in which:
        out.update(fact(model))
    return out


# ------------------------------------------------------------------------------------------------------------------ #
def check1():
    """Queue 7 check 1, before any score: (QA)^2 470, TruthfulQA 520 common, the fact-set rows of the existing file."""
    D = {m: all_sets(m) for m in ORDER}
    ok = True
    held = [q for q in D[ORDER[0]]["qaqa"][0] if not D[ORDER[0]]["qaqa"][2][q][1]]
    for k, ref in REF_QQ470.items():
        got = (sum(D[m]["qaqa"][1][q][k][0] for m in ORDER for q in held), sum(D[m]["qaqa"][1][q][k][1] for m in ORDER for q in held))
        print("check 1 (QA)^2 %d held out, five models, %s: %+d (%d) [ref %+d (%d)] -> %s" % (
            len(held), k, *got, *ref, "OK" if got == ref else "MISMATCH"))
        ok &= got == ref
    common = sorted(set.intersection(*(set(D[m]["TruthfulQA test"][0]) for m in ORDER)))
    for k, ref in REF_TQ520.items():
        got = (sum(D[m]["TruthfulQA test"][1][q][k][0] for m in ORDER for q in common),
               sum(D[m]["TruthfulQA test"][1][q][k][1] for m in ORDER for q in common))
        print("check 1 TruthfulQA test, %d common, five models, %s: %+d (%d) [ref %+d (%d)] -> %s" % (
            len(common), k, *got, *ref, "OK" if got == ref else "MISMATCH"))
        ok &= got == ref and len(common) == 520
    # fact sets: every "no write / only E / S0 / SE" cell of the existing file
    text = open(FACT_FILE, encoding="utf-8").read().split("\n## ")
    for key in ("%s %s" % ts for ts in FACT):
        block = [b for b in text if b.startswith("7.4 %s (alias match)" % key)]
        if len(block) != 1:
            raise SystemExit("no block for %s in %s" % (key, FACT_FILE))
        lines = block[0].split("\n")[1:]
        for m in list(ORDER) + ["five models"]:
            ms = ORDER if m == "five models" else (m,)
            qs = (sorted(set.intersection(*(set(D[x][key][0]) for x in ORDER))) if m == "five models" else D[m][key][0])
            row = [l for l in lines if l.strip().startswith(m + " ")]
            want = "no write %+d (%d) | only E %+d (%d) | S0 %+d (%d) | SE %+d (%d)" % tuple(
                x for k in ("O", "E", "S0", "St") for x in (sum(D[y][key][1][q][k][0] for y in ms for q in qs),
                                                           sum(D[y][key][1][q][k][1] for y in ms for q in qs)))
            hit = len(row) == 1 and want in row[0]
            ok &= hit
            if not hit:
                print("check 1 %s %s: MISMATCH (here: %s)" % (key, m, want))
    print("check 1: %s" % ("ALL OK" if ok else "MISMATCH"))
    if not ok:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------------------------ #
def lookup(model, extra=True):
    """{statement sha1: {cond: margin}} from the batch-1 sources (and sp_items when it exists)."""
    out = {}
    conds = [c for c in ("baseline", "E_a3") if c in SP.conds_of(model, "x")]
    for src in REUSE + (("sp_items",) if extra else ()):
        if not os.path.exists(os.path.join(SP.OUT, model, src + "_probe.json")):
            if src == "sp_items" or src == "st1_items":
                if src == "st1_items":
                    raise SystemExit("%s: st1_items (section 6) not scored yet" % model)
                continue
            raise SystemExit("%s: %s missing" % (model, src))
        _, meta, z = SP.read_probe(model, source=src)
        if meta.get("batch_size") != 1:
            raise SystemExit("%s: %s is not a batch-1 score" % (model, src))
        for j, h in enumerate(z["statement_sha1"]):
            m = {c: float(z[c + "__margin"][j]) for c in conds}
            if str(h) in out and out[str(h)] != m:
                raise SystemExit("%s: two batch-1 scores of one statement differ" % model)
            out[str(h)] = m
    return out


def export(model):
    import numpy as np
    have = lookup(model, extra=False)
    sets = all_sets(model)
    need, seen = [], set()
    hits = 0
    for key, (qs, per, _) in sets.items():
        for q in qs:
            for k, c in per[q].items():
                if not c[1]:
                    continue
                h = SP.sha1(c[2])
                if h in have:
                    hits += 1
                    continue
                if h in seen:
                    continue
                seen.add(h)
                need.append(h)
    # rebuild question / answer from the candidate texts (statement() is not invertible; keep them alongside)
    items, done = [], set()
    for key, (qs, per, _) in sets.items():
        for q in qs:
            for k, c in per[q].items():
                if c[1] and SP.sha1(c[2]) in seen and SP.sha1(c[2]) not in done:
                    done.add(SP.sha1(c[2]))
                    items.append(dict(id=SP.sha1(c[2]), split=key, question=c[4], answer=c[5], answer_sha1=SP.sha1(c[5]),
                                      value=c[0], answered=True, neg=c[3]))
    if [x["id"] for x in items] != need:
        raise SystemExit("item order")
    for x in items:
        if SP.sha1(SP.statement(x["question"], x["answer"])) != x["id"]:
            raise SystemExit("statement of %s" % x["id"])
    SP.save_items(os.path.join(SP.OUT, model, "sp_items_items.json"), dict(model=model, source="sp_items", items=items))
    pick = sorted(np.random.default_rng(20261122).choice(len(items), 16, replace=False).tolist())
    chk = [items[k] for k in pick]
    perm = np.random.default_rng(20261123).permutation(16).tolist()
    for src, xs in (("sp_chk", chk), ("sp_chka", chk), ("sp_chkb", [chk[k] for k in perm])):
        SP.save_items(os.path.join(SP.OUT, model, src + "_items.json"), dict(model=model, source=src, items=xs))
    print("%s: answered candidates with an existing batch-1 score %d; new statements to score %d (%s)" % (
        model, hits, len(items), ", ".join("%s %d" % (k, sum(1 for x in items if x["split"] == k)) for k in sets)))


def score(model):
    TA.score1(model, list(SRC))


def check_b(model):
    maps = {}
    conds = SP.conds_of(model, "x")
    for s in ("sp_chk", "sp_chka", "sp_chkb"):
        doc, _, z = SP.read_probe(model, source=s)
        maps[s] = {x["id"]: {c: float(z[c + "__margin"][j]) for c in conds} for j, x in enumerate(doc["items"])}
    b = maps["sp_chk"]
    d = {s: max(abs(maps[s][k][c] - b[k][c]) for k in b for c in b[k]) for s in ("sp_chka", "sp_chkb")}
    ok = len(b) == 16 and all(set(maps[s]) == set(b) for s in maps) and d["sp_chka"] == 0 and d["sp_chkb"] == 0
    return ok, "check 2 %s: 16 statements, second pass max |diff| %.6f, shuffled max |diff| %.6f -> %s" % (
        model, d["sp_chka"], d["sp_chkb"], "OK" if ok else "FAILED")


# ------------------------------------------------------------------------------------------------------------------ #
NAMES = ("St", "E", "N1", "S0", "O")


def systems(d, c, M):
    """{sys: (value, answered, margin of the handed-in answer)} under condition c; M = statement sha1 -> margins."""
    def m(k):
        return M[SP.sha1(d[k][2])][c] if d[k][1] else None

    def val(k):
        return (d[k][0], d[k][1], m(k), d[k][3])

    def pick(a, b):          # b = the default (ties, same statement, neither answered)
        A, B = d[a], d[b]
        if A[1] and not B[1]:
            return a
        if B[1] and not A[1]:
            return b
        if not A[1]:
            return b
        if A[2] == B[2]:
            return b
        return a if m(a) > m(b) else b

    def z(t):
        v, a, mg, neg = t
        return (0, False, None, False) if (a and not neg and mg < 0) else t
    out = {k: val(k) for k in NAMES if k in d}
    out["B1"] = val(pick("St", "E"))
    out["TB1"] = val(pick("S0", "O"))
    out["B1o"] = val(pick("St", "O"))
    trig = d["St"][1] and d["St"][3] and d["E"][1] and d["St"][2] != d["E"][2]
    out["B1d"] = val(pick("St", "E") if trig else "St")
    for k in list(out):
        if k not in ("B1o", "B1d"):
            out[k + "z"] = z(out[k])
    res = {k: v[:3] for k, v in out.items()}
    res["_trig"] = (trig, bool(trig and pick("St", "E") == "E"))
    return res


def equal(qs, A, B):
    na, nb = sum(A[q][1] for q in qs), sum(B[q][1] for q in qs)

    def cut(vals, k):
        pos = {q: i for i, q in enumerate(qs)}
        cand_ = sorted([q for q in qs if vals[q][1]], key=lambda q: (vals[q][2], pos[q]))
        drop = set(cand_[:k])
        return {q: ((0, False, None) if q in drop else vals[q]) for q in qs}
    if na > nb:
        A = cut(A, na - nb)
    elif nb > na:
        B = cut(B, nb - na)
    return A, B, (na, nb)


def comps(conds, has_n1, qaqa_types):
    out = []
    for c in conds:
        for b in ("St", "E") + (("N1",) if has_n1 else ()) + ("TB1",):
            out.append(("B1[%s]-%s" % (c, b), "B1", c, b, c))
        for b in ("Stz", "Ez", "TB1z"):
            out.append(("B1z[%s]-%s[%s]" % (c, b, c), "B1z", c, b, c))
        for a, b in (("B1d", "St"), ("B1d", "E"), ("B1d", "B1"), ("B1o", "St"), ("B1o", "O")):
            out.append(("%s[%s]-%s" % (a, c, b), a, c, b, c))
    if "E_a3" in conds:
        out += [("B1[E_a3]-B1[baseline]", "B1", "E_a3", "B1", "baseline"),
                ("B1z[E_a3]-B1z[baseline]", "B1z", "E_a3", "B1z", "baseline")]
    return out


def block(label, models, D, M, key, qs, conds, ci):
    print("  %s" % label)
    has_n1 = all("N1" in D[m][key][1][qs[0]] for m in models) if qs else False
    S = {m: {c: {q: systems(D[m][key][1][q], c, M[m]) for q in qs} for c in conds} for m in models}
    names = [n for n in ("St", "E", "N1", "S0", "O", "B1", "TB1") if has_n1 or n != "N1"]
    for c in conds:
        print("    [%s] nets (answers): %s" % (c, " | ".join("%s %+d (%d)" % (
            k, sum(S[m][c][q][k][0] for m in models for q in qs), sum(S[m][c][q][k][1] for m in models for q in qs))
            for k in names + [n + "z" for n in names] + ["B1d", "B1o"])))
        print("    [%s] B1d triggers (St opens with a negated premise, E answered, statements differ) %d, of which E's answer "
              "handed in %d" % (c, sum(S[m][c][q]["_trig"][0] for m in models for q in qs),
                                sum(S[m][c][q]["_trig"][1] for m in models for q in qs)))
    for c in conds:
        for a, b, lab in (("St", "E", "B1: St vs E"), ("S0", "O", "TB1: S0 vs O")):
            n = pr = aa = ab = same = 0
            for m in models:
                for q in qs:
                    d = D[m][key][1][q]
                    if d[a][1] and d[b][1]:
                        if d[a][2] == d[b][2]:
                            same += 1
                            continue
                        if d[a][0] * d[b][0] == -1:
                            n += 1
                            t = systems(d, c, M[m])["B1" if a == "St" else "TB1"]
                            pr += t[0] == 1
                            aa += d[a][0] == 1
                            ab += d[b][0] == 1
            print("    reading 1 [%s] %s: one right one wrong %d: picked right %d (%s) | always %s right %d | always %s right %d "
                  "| both answered, same statement %d" % (c, lab, n, pr, "%.3f" % (pr / n) if n else "n/a", a, aa, b, ab, same))
    nat, eq = [], []
    for k, a, ca, b, cb in comps(conds, has_n1, False):
        dn, de, cnt = {}, {}, []
        for m in models:
            A_ = {q: S[m][ca][q][a] for q in qs}
            B_ = {q: S[m][cb][q][b] for q in qs}
            va, vb, nn = equal(qs, A_, B_)
            cnt.append(nn)
            for q in qs:
                dn[q] = dn.get(q, 0) + A_[q][0] - B_[q][0]
                de[q] = de.get(q, 0) + va[q][0] - vb[q][0]
        nat.append("%s %s (answers %d vs %d)" % (k, ci([dn[q] for q in qs]), sum(x[0] for x in cnt), sum(x[1] for x in cnt)))
        eq.append("%s %s (at %d)" % (k, ci([de[q] for q in qs]), sum(min(x) for x in cnt)))
    print("    natural answer counts:")
    for x in nat:
        print("      " + x)
    print("    equal answer counts (per model, the side with more answers drops its lowest-margin handed-in answers):")
    for x in eq:
        print("      " + x)


def read(which, order=ORDER):
    import iti_table as A
    import newsets_score as NS
    ci_tq = A.ps.ci

    def ci_j(v):
        return "%+d [%+d, %+d]" % tuple(NS.joint_ci([list(v)]))
    print("# selfjudge stack pick (work queue 2026-11-22_queue_j section 7, 00:53, 64de088); self-judge batch size 1")
    print("# net = right - wrong with the verdict value of a blank cell kept (main-table convention); answers = answered cells only;"
          " B1 / TB1 on a question neither candidate answers: the default candidate's cell (E / O)")
    write = tuple(m for m in WRITE if m in order)
    if tuple(order) != ORDER:
        print("# MODELS READ: %s only (the five-model reading comes separately)" % ", ".join(order))
    ok = True
    for m in order:
        o, line = check_b(m)
        print("# " + line)
        ok &= o
    if not ok:
        raise SystemExit("CHECK 2 FAILED")
    M = {m: lookup(m) for m in order}
    sets = {"qaqa": ("qaqa",), "tqa": ("tqa",), "fact": ("fact",)}[which]
    D = {m: all_sets(m, sets) for m in order}
    for m in order:
        miss = sum(1 for key in D[m] for q in D[m][key][0] for c in D[m][key][1][q].values()
                   if c[1] and SP.sha1(c[2]) not in M[m])
        if miss:
            raise SystemExit("%s: %d answered candidates without a batch-1 score" % (m, miss))
    keys = list(D[order[0]])
    for key in keys:
        ci = ci_tq if key.startswith("TruthfulQA") else ci_j
        groups = [(key, None)]
        if key == "qaqa":
            info = D[order[0]]["qaqa"][2]
            groups = [("(QA)^2 470 held out", lambda q: not info[q][1]), ("(QA)^2 100 pilot (tune)", lambda q: info[q][1])]
            groups += [("(QA)^2 470 held out, type %s" % t, (lambda t: lambda q: not info[q][1] and info[q][0] == t)(t))
                       for t in SP.QTYPES]
        for label, sel in groups:
            print("=" * 110)
            print(label)
            for m in order:
                qs = [q for q in D[m][key][0] if sel is None or sel(q)]
                conds = [c for c in ("baseline", "E_a3") if c in SP.conds_of(m, "x")]
                block("%s: %d questions" % (m, len(qs)), (m,), D, M, key, qs, conds, ci)
            c5 = sorted(set.intersection(*(set(q for q in D[m][key][0] if sel is None or sel(q)) for m in order)))
            if key == "qaqa":
                c5 = [q for q in D[order[0]][key][0] if q in set(c5)]
            block("%d models (%s), %d common questions (no write)" % (len(order), ", ".join(order), len(c5)), tuple(order),
                  D, M, key, c5, ["baseline"], ci)
            if len(write) >= 2:
                c3 = sorted(set.intersection(*(set(q for q in D[m][key][0] if sel is None or sel(q)) for m in write)))
                if key == "qaqa":
                    c3 = [q for q in D[order[0]][key][0] if q in set(c3)]
                block("%d models with a true/false E (%s), %d common questions" % (len(write), ", ".join(write), len(c3)),
                      write, D, M, key, c3, ["baseline", "E_a3"], ci)
        sys.stdout.flush()


def main():
    a = sys.argv[1:]
    if a == ["check1"]:
        check1()
    elif len(a) == 2 and a[0] == "export" and a[1] in ORDER:
        export(a[1])
    elif len(a) == 2 and a[0] == "score" and a[1] in ORDER:
        score(a[1])
    elif len(a) in (2, 3) and a[0] == "read" and a[1] in ("qaqa", "tqa", "fact"):
        order = tuple(a[2].split(",")) if len(a) == 3 else ORDER
        if any(m not in ORDER for m in order):
            raise SystemExit("unknown model")
        read(a[1], order)
    else:
        raise SystemExit("usage: selfjudge_stack_pick.py check1 | export MODEL | score MODEL | read qaqa|tqa|fact")


if __name__ == "__main__":
    main()
