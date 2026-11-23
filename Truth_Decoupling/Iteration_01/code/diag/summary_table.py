#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, 2026-11-23), work queue discussion/work_queue/2026-11-23_queue_l.md,
# section 5 (one summary table) with the two corrections of section 10.1. Read-only: no model, no generation, no judge
# call, no GPU, nothing chosen here. Every cell is computed from existing answers and verdicts by the existing reading
# programs (imported unchanged); a cell without existing data prints "none" with the reason.
# Rows (net = right - wrong with the verdict value of a blank cell kept, the main-table convention; answers = answered
# cells):
#   O        original model, no write
#   S0       plain prompt + honest sentence (x_E2), no write
#   single E the no-sentence single E (main-table setting)
#   N1       the no-sentence main-table N1
#   fill EC  "fill only": S0's cell where S0 answers, SN1's cell on the questions S0 leaves blank by the rule (fill_only)
#   SN1m     the stack at the strength chosen on the four tune sets (n1_on_sentence3 choose: Llama-3.1-8B and
#            Gemma-3-27B halved, the other three at s_sel)
#   SN1      the stack at s_sel (queue I step one)
#   ITI      three models (Llama-3.1-8B, Gemma-3-12B, Gemma-3-27B), faithful fit, choice within the MMLU budget
# Columns and sources:
#   TruthfulQA test, DeepSeek Pro: O / S0 / single E / N1 / SN1 = selfjudge_stack_pick.tqa; fill EC = fill_only.tq
#     (cross-checked cell by cell on S0, SN1, N1); SN1m = n1_on_sentence3.tq_sn1 at the chosen strength; ITI =
#     iti_table's choice.json and its test work dir (as iti_table.test_table).
#   (QA)^2 470 held-out (all, valid / false / unverifiable assumption), DeepSeek Pro: O / S0 / single E / N1 / SN1 =
#     selfjudge_stack_pick.qaqa; fill EC = fill_only.qq; SN1m = n1_on_sentence3.qaqa_raw + qaqa_sn1 (caps chosen on the
#     100 pilot questions, as 7.1).
#   TriviaQA test / test2, PopQA test / test2 (alias match): O / S0 / single E = selfjudge_stack_pick.fact; N1 =
#     ns_prompt_table's frozen transfer F (main-table strength, TruthfulQA caps at "prompt"); fill EC / SN1m / SN1:
#     none (no C pass on the fact sets); SE (s_sel), SEm (chosen strength) and fill E are printed as references.
#   MMLU (13,042 items not seen before, option probability) and GSM8K (1,319 items, cap 512, batch 8, rule scoring),
#     change against the original model in points (sentence_capability.mmlu_model / gsm_model with --half, boot):
#     O 0; S0 = P; single E = E (no sentence); N1: MMLU = E (no sentence) (four-choice protocol: no blank, so no C
#     re-ask), GSM8K = E (no sentence) at cap 512 with the items blank under E re-asked as gsm8k_n1_prompt (re-asks at
#     cap 512 exist for Llama-3.1-8B, Gemma-3-12B, Gemma-3-27B; the two Qwen models have them only at cap 1024, so their
#     blanks give a range); fill EC: MMLU = S0's (no blank), GSM8K = S0 with its rule blanks filled by P + E(s_sel) and,
#     where that is blank too and the own-blank cap is set, a range (the C re-ask was not generated); SN1m = P + E at the
#     chosen strength; SN1 = P + E(s_sel) (C on the blanks of P + E not generated; blanks counted in the notes); ITI = its
#     chosen point's full-set MMLU file and GSM8K file.
#   Small table (section 5): P + E(s_sel) - P and P + E(s_half) - P, MMLU and GSM8K.
# Intervals: TruthfulQA policy_select2.ci (fill_only.ci_tq); the other sets newsets_score.joint_ci (fill_only.ci_j); on
# per-question sums over the models of a row. Capability: sentence_capability.boot (mean of the per-model changes).
# usage (scoring container, from /workspace/Truth_Decoupling, CUDA_VISIBLE_DEVICES empty):
#   python3 -B Iteration_01/diag/summary_table.py
import json
import os
import sys

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import fill_only as FO                # noqa: E402
import selfjudge_stack_pick as SKP    # noqa: E402
import n1_on_sentence as N            # noqa: E402
import n1_on_sentence3 as N3          # noqa: E402
import ns_prompt_table as NP          # noqa: E402
import iti_table as IT                # noqa: E402
import sentence_capability as SC      # noqa: E402
import gsm8k_n1_prompt as GN          # noqa: E402
import cap_score as CS                # noqa: E402
import release_origin as RO           # noqa: E402

ORDER = N.ORDER
ROWS = ("O", "S0", "single E", "N1", "fill EC", "SN1m", "SN1")
ITI_MODELS = ("llama3.1-8b", "gemma3-12b", "gemma3-27b")
FIT = "faithful_seed42"
QTYPE = dict(N3.Q.TYPES)
COLS = (("tqa", "TruthfulQA test"), ("qq", "(QA)^2 470 all"), ("qq:no_failure", "(QA)^2 valid"), ("qq:failure", "(QA)^2 false"),
        ("qq:unverifiable", "(QA)^2 unverifiable"), ("triviaqa test", "TriviaQA test"), ("triviaqa test2", "TriviaQA test2"),
        ("popqa test", "PopQA test"), ("popqa test2", "PopQA test2"))
FACT_REF = ("SE s_sel", "SEm", "fill E")
CHECKS = []


def check(label, got, ref):
    ok = got == ref
    CHECKS.append((label, got, ref, ok))
    return ok


def vt(c):
    return (int(c[0]), bool(c[1]))


# ---- TruthfulQA test ------------------------------------------------------------------------------------------- #
def tq_cells(m):
    sk_qs, sk, _ = SKP.tqa(m)["test"]
    fqs, fper = FO.tq(m, 1, FO.REF_SCCAP[m])
    ch3 = json.load(open(N3.CHOICE3 % m))
    c1 = json.load(open(N.CHOICE % m))
    ss, sc = ch3["s_sel"], ch3["chosen"]
    psp, ptx, cfg = N3.tq_sn1(m, 1, sc, N.restricted(m, 1), None if sc != ss else tuple(c1["cfg"]))
    snm = N.run_sys(psp, ptx, cfg)
    fset = set(fqs)
    qs = [q for q in sk_qs if q in fset and q in snm]
    out = {}
    for q in qs:
        d, f = sk[q], fper[q]
        row = {"O": vt(d["O"]), "S0": vt(d["S0"]), "single E": vt(d["E"]), "N1": vt(d["N1"]), "SN1": vt(d["St"]),
               "fill EC": vt(FO.cell(f, "fill EC")), "SN1m": vt(snm[q])}
        if (vt(f["S0"]), vt(f["SN1"]), vt(f["N1"])) != (row["S0"], row["SN1"], row["N1"]):
            raise SystemExit("%s %s: fill_only and selfjudge_stack_pick disagree on S0 / SN1 / N1" % (m, q))
        if sc == ss and row["SN1m"] != row["SN1"]:
            raise SystemExit("%s %s: SN1m differs from SN1 at the same strength" % (m, q))
        out[q] = row
    print("# TruthfulQA test %s: %d questions (selfjudge_stack_pick %d, fill_only %d, SN1m %d); SN1m at E %s (s_sel %s), %s" % (
        m, len(qs), len(sk_qs), len(fqs), len(snm), sc, ss, cfg))
    return qs, out


def iti_cells(m):
    display, layer, s = IT.KEY[m]
    ch = IT.read("%s/%s/%s/choice.json" % (IT.BASE, m, FIT))
    if ch["model"] != m or ch["fit"] != FIT or ch["sigma"] != "endq" or ch["budget"] is None:
        raise SystemExit("%s: ITI choice file of another setting" % m)
    X, why = IT.MP.prompt_split(display, layer, 1, s)
    if X is None:
        raise SystemExit(why)
    xsp, xtx, _, qs = X
    _, run, _ = IT.MP.usual(display, 1)
    w = IT.MP.work(run, "test", "iti_%s" % FIT)
    cond, g = IT.condition_of(m, FIT, "test", ch["stem"])
    if g["setting"]["k"] != ch["k"] or g["setting"]["alpha"] != ch["alpha"]:
        raise SystemExit("%s: ITI test generation is not the chosen setting" % m)
    st, bl = IT.ps.load(w, "pro")
    base = IT.S.system(xsp, xtx, ("base",))[0]
    iti = {q: (IT.val(st[(q, cond)]), not bl[(q, cond)]) for q in qs}
    o = {q: IT.val(x) for q, x in zip(qs, base)}
    return ch, qs, iti, o


# ---- (QA)^2 ---------------------------------------------------------------------------------------------------- #
def qq_cells(m):
    ids, per, info = SKP.qaqa(m)["qaqa"]
    fids, fper, finfo, _ = FO.qq(m)
    if fids != ids or finfo != info:
        raise SystemExit("%s: (QA)^2 roster of fill_only differs" % m)
    sc = json.load(open(N3.CHOICE3 % m))["chosen"]
    best, (v, a, _) = N3.qaqa_sn1(N3.qaqa_raw(m, sc), sc)
    out = {}
    for q in ids:
        d, f = per[q], fper[q]
        row = {"O": vt(d["O"]), "S0": vt(d["S0"]), "single E": vt(d["E"]), "N1": vt(d["N1"]), "SN1": vt(d["St"]),
               "fill EC": vt(FO.cell(f, "fill EC")), "SN1m": (int(v[q]), bool(a[q]))}
        if (vt(f["S0"]), vt(f["SN1"]), vt(f["N1"])) != (row["S0"], row["SN1"], row["N1"]):
            raise SystemExit("%s %s: fill_only and selfjudge_stack_pick disagree on (QA)^2" % (m, q))
        out[q] = row
    print("# (QA)^2 %s: SN1m at E %s, caps %s / %s" % (m, sc, best[0], best[1]))
    return ids, out, info


# ---- fact sets ------------------------------------------------------------------------------------------------- #
def fact_cells(m):
    F = SKP.fact(m)
    M = NP.S.models()
    el_np, main_s, _, tq_caps, _ = NP.MAIN[m]
    el = N3.Q.SET[m][0]
    sc = json.load(open(N3.CHOICE3 % m))["chosen"]
    out = {}
    for task, split in SKP.FACT:
        key = "%s %s" % (task, split)
        qs, per, _ = F[key]
        xsp, xtx, have, _, _, _ = NP.load(task, m, M[m], split)
        ms = NP.dash(main_s)
        n1 = None
        if ms in have and list(xsp.qs) == list(qs):
            st, an = NP.S.system(xsp, xtx, ("N1", ms, tq_caps[0], tq_caps[1]))[:2]
            n1 = {q: (int(x), bool(y)) for q, x, y in zip(xsp.qs, NP.S.vals(st), an)}
        sem = N3.fact_arm(task, m, split, "x_E2:E%d.w.p%s" % (el, sc))[0]
        cells = {}
        for q in qs:
            d = per[q]
            row = {"O": vt(d["O"]), "S0": vt(d["S0"]), "single E": vt(d["E"]), "SE s_sel": vt(d["St"]),
                   "SEm": (int(sem[q][0]), bool(sem[q][1]))}
            row["fill E"] = row["S0"] if row["S0"][1] else row["SE s_sel"]
            if n1 is not None:
                row["N1"] = n1[q]
            cells[q] = row
        out[key] = (list(qs), cells, n1 is not None)
    return out


# ---- capability ------------------------------------------------------------------------------------------------ #
def gsm_rows(path):
    g = SC.read(path)
    c = g["contract"]
    if not (g.get("completed") and c["max_new_tokens"] == 512 and c["batch_size"] == 8 and c["full_pass"]):
        raise SystemExit("%s: not a complete 512-token batch-8 pass" % path)
    return {r["id"]: r for r in g["rows"]}


def blank(r):
    return bool(r["rule_abstain"]) or not r["text"].strip()


def capability():
    SC.HALF = True
    seen = set(SC.read(SC.SEEN)["ids"])
    gold = CS.gold_of("gsm8k")
    tok = lambda s: s.replace(".", "-")  # noqa: E731
    C = {"MMLU": {}, "GSM8K": {}}
    notes = []
    gsm_ids = None
    for m in ORDER:
        display, layer, s_main = N.A.KEY[m]
        ch3 = json.load(open(N3.CHOICE3 % m))
        c1 = json.load(open(N.CHOICE % m))
        ss, sc = ch3["s_sel"], ch3["chosen"]
        if ss != SC.sel(m) or ch3["s_half"] != SC.half(m) or sc not in (ss, ch3["s_half"]):
            raise SystemExit("%s: strengths of n1s3_choice and sentence_capability differ" % m)
        pick = "P+E" if sc == ss else "P+E half"
        ids, d = SC.mmlu_model(m, seen)
        mm = {"O": d["base"], "S0": d["P"], "single E": d["E (no sentence)"], "N1": d["E (no sentence)"], "fill EC": d["P"],
              "SN1m": d[pick], "SN1": d["P+E"], "P+E": d["P+E"], "P+E half": d["P+E half"], "P": d["P"]}
        if m in ITI_MODELS:
            ch = IT.read("%s/%s/%s/choice.json" % (IT.BASE, m, FIT))
            g = [x for x in IT.grid(m, FIT, "endq") if x[2] == ch["stem"]]
            if len(g) != 1:
                raise SystemExit("%s: no MMLU file of the chosen ITI point" % m)
            mm["ITI"] = g[0][3]
        C["MMLU"][m] = (ids, {k: [int(v[i]) for i in ids] for k, v in mm.items()})
        gids, gd = SC.gsm_model(m, gold)
        if gsm_ids is None:
            gsm_ids = gids
        if gids != gsm_ids:
            raise SystemExit("GSM8K item order differs between models")
        gg = {"O": gd["base"], "S0": gd["P"], "single E": gd["E (no sentence)"], "SN1m": gd[pick], "SN1": gd["P+E"],
              "P+E": gd["P+E"], "P+E half": gd["P+E half"], "P": gd["P"]}
        G = {k: [int(v[i]) for i in gids] for k, v in gg.items()}
        rng = {}
        # fill EC on GSM8K: S0 (P) answered -> P; rule blank of P -> P + E(s_sel); blank there too and own cap set -> unknown
        cap_dir = "%s/gen_main/gsm8k_h22/%s/b8/" % (SC.CAP, m)
        P = gsm_rows(cap_dir + "neutral_E2.json")
        PE = gsm_rows(cap_dir + "neutral_E2--E%d-w-p%s-at-prompt.json" % (layer, tok(ss)))
        own_cap = c1["cfg"][3] if c1["cfg"][0] == "N1" else None
        base_right = sum(G["O"])
        kr = unk = nb = 0
        fill = []
        for i in gids:
            r = int(CS.judge_row("gsm8k", P[i], gold(i))[0] == "R")
            if blank(P[i]):
                nb += 1
                if blank(PE[i]) and own_cap is not None:
                    unk += 1
                    r = 0
                else:
                    r = int(CS.judge_row("gsm8k", PE[i], gold(i))[0] == "R")
                    kr += r
            fill.append(r)
        if sum(fill) != sum(G["S0"]) + kr:
            raise SystemExit("%s: GSM8K fill count" % m)
        G["fill EC"] = fill               # lower end (each unknown counted not right); the range is in rng
        rng["fill EC"] = (sum(fill) - base_right, sum(fill) + unk - base_right)
        notes.append("GSM8K %s fill EC: S0 rule blanks %d, filled by P + E(s_sel) right %d, blank under P + E with the own cap %s "
                     "set (C re-ask not generated) %d" % (m, nb, kr, own_cap, unk))
        # N1 on GSM8K at cap 512: single E (no sentence); its blanks re-asked as gsm8k_n1_prompt where cap-512 re-asks exist
        epath = "%s/gen_main/gsm8k/%s/b8/neutral--E%d-w-p%s-at-prompt.json" % (SC.CAP, m, layer, s_main)
        if not os.path.exists(epath):
            epath = epath.replace("/gsm8k/", "/gsm8k_h22/")
        E = gsm_rows(epath)
        B0 = gsm_rows("%s/gen_main/gsm8k/%s/b8/neutral.json" % (SC.CAP, m))
        eb = [i for i in gids if blank(E[i])]
        n1 = [int(CS.judge_row("gsm8k", E[i], gold(i))[0] == "R") for i in gids]
        if n1 != G["single E"]:
            raise SystemExit("%s: GSM8K single E file differs from sentence_capability's" % m)
        el, s_gn, cl, cap_e, cap_o, tcap = GN.SET[m]
        unk_n1 = 0
        if eb and tcap == 512:
            if (el, s_gn) != (layer, N.dot(s_main)):
                raise SystemExit("%s: gsm8k_n1_prompt setting differs from the main table" % m)
            sub = os.path.join(GN.root(m), "n1_blank_ids_at-prompt.json")
            if json.load(open(sub)) != eb:
                raise SystemExit("%s: re-ask ids differ from the blanks under E" % m)
            re = {}
            for k in GN.LAD:
                if any(float(k) <= float(cap_o if blank(B0[i]) else cap_e) for i in eb):
                    re[k] = GN.load_arm(m, "neutral:E%d.w.p%s+C%d.w.m%s" % (el, s_gn, cl, k), "prompt", eb, sub)[1]
            pos = {i: j for j, i in enumerate(gids)}
            for i in eb:
                own = blank(B0[i])
                lad = GN.LAD[:GN.LAD.index(cap_o if own else cap_e) + 1]
                k = next((k for k in lad if not blank(re[k][i])), None)
                if k is None or (not own and RO.norm(re[k][i]["text"]) == RO.norm(B0[i]["text"])):
                    continue
                n1[pos[i]] = int(CS.judge_row("gsm8k", re[k][i], gold(i))[0] == "R")
        elif eb:
            unk_n1 = len(eb)
        G["N1"] = n1
        rng["N1"] = (sum(n1) - base_right, sum(n1) + unk_n1 - base_right)
        notes.append("GSM8K %s N1: blanks under single E at cap 512 %d; %s" % (
            m, len(eb), "re-asked from the cap-512 files of gsm8k_n1_prompt" if tcap == 512 else
            "C re-asks exist only at cap %d, so each blank is a range (no right ... right)" % tcap))
        for lab, key in (("SN1", "P+E"), ("SN1m", pick)):
            path = cap_dir + ("neutral_E2--E%d-w-p%s-at-prompt.json" % (layer, tok(ss if key == "P+E" else ch3["s_half"])))
            notes.append("GSM8K %s %s: rule blanks under %s %d (C re-asks not generated; scored as that pass)" % (
                m, lab, key, sum(blank(r) for r in gsm_rows(path).values())))
        if m in ITI_MODELS:
            ch = IT.read("%s/%s/%s/choice.json" % (IT.BASE, m, FIT))
            gi = SC.read("%s/%s/%s/gsm8k/%s.json" % (IT.BASE, m, FIT, ch["stem"]))
            if not (gi.get("completed") is True and gi["model"] == m and gi["max_new_tokens"] == 512 and gi["batch_size"] == 8
                    and [r["id"] for r in gi["rows"]] == gids):
                raise SystemExit("%s: ITI GSM8K file is not a complete pass of the same items" % m)
            G["ITI"] = [int(CS.judge_row("gsm8k", r, gold(r["id"]))[0] == "R") for r in gi["rows"]]
        C["GSM8K"][m] = (gids, G, rng)
    return C, notes


def cap_cell(C, task, row, models):
    """(text, point or range) of the mean change against O in points."""
    per = C[task]
    if any(row not in per[m][1] for m in models):
        return "none", None
    if task == "GSM8K" and row in ("fill EC", "N1"):
        lo = sum(per[m][2][row][0] for m in models) * 100.0 / (len(models) * len(per[models[0]][0]))
        hi = sum(per[m][2][row][1] for m in models) * 100.0 / (len(models) * len(per[models[0]][0]))
        if lo == hi:
            v = [[a - b for a, b in zip(per[m][1][row], per[m][1]["O"])] for m in models]
            return SC.boot(v), lo
        return "%+.2f to %+.2f (range)" % (lo, hi), (lo, hi)
    v = [[a - b for a, b in zip(per[m][1][row], per[m][1]["O"])] for m in models]
    return SC.boot(v), None


# ---- table ----------------------------------------------------------------------------------------------------- #
def col_qs(data, col, m):
    if col.startswith("qq"):
        ids, _, info = data["qq"][m]
        t = col.split(":", 1)[1] if ":" in col else None
        return [q for q in ids if not info[q][1] and (t is None or info[q][0] == t)]
    return data[col][m][0]


def cells_of(data, col, m):
    return data["qq" if col.startswith("qq") else col][m][1]


def common(data, col, models):
    if col.startswith("qq"):
        return col_qs(data, col, models[0])
    s = set.intersection(*(set(col_qs(data, col, m)) for m in models))
    return [q for q in col_qs(data, col, models[0]) if q in s]


def total(data, col, row, models, qs):
    P = {m: cells_of(data, col, m) for m in models}
    if any(row not in P[m][q] for m in models for q in qs):
        return None
    return sum(P[m][q][row][0] for m in models for q in qs), sum(P[m][q][row][1] for m in models for q in qs)


def why_none(col, row):
    if not col.startswith("tqa") and not col.startswith("qq") and row in ("fill EC", "SN1m", "SN1"):
        return "none (no C pass on the fact sets; SE references below)"
    if row == "ITI":
        return "none (ITI only on TruthfulQA)"
    return "none (no existing pass)"


def diff(data, col, a, b, models, qs):
    P = {m: cells_of(data, col, m) for m in models}
    if any(a not in P[m][q] or b not in P[m][q] for m in models for q in qs):
        return None
    ci = FO.ci_tq if col == "tqa" else FO.ci_j
    return ci([sum(P[m][q][a][0] - P[m][q][b][0] for m in models) for q in qs])


def print_block(label, data, C, models, rows, qs_of):
    print("\n" + "=" * 110 + "\n%s" % label)
    print("-- nets (answers); question counts: %s" % " | ".join("%s %d" % (cl, len(qs_of(c))) for c, cl in COLS))
    for r in rows:
        parts = []
        for c, cl in COLS:
            t = total(data, c, r, models, qs_of(c))
            parts.append("%s %s" % (cl, FO.fmt(t) if t is not None else why_none(c, r)))
        mt, _ = cap_cell(C, "MMLU", r, models)
        gt, _ = cap_cell(C, "GSM8K", r, models)
        print("  %-9s %s || MMLU %s | GSM8K %s" % (r, " | ".join(parts), mt, gt))
    print("-- row minus S0 (paired interval)")
    for r in rows:
        if r == "S0":
            continue
        parts = []
        for c, cl in COLS:
            dd = diff(data, c, r, "S0", models, qs_of(c))
            parts.append("%s %s" % (cl, dd if dd is not None else "none"))
        print("  %-9s %s" % (r, " | ".join(parts)))
    sys.stdout.flush()


def main():
    data = {"tqa": {}, "qq": {}}
    for m in ORDER:
        data["tqa"][m] = tq_cells(m)
        data["qq"][m] = qq_cells(m)
        for key, (qs, cells, has_n1) in fact_cells(m).items():
            data.setdefault(key, {})[m] = (qs, cells)
            if not has_n1:
                print("# %s %s: no complete N1 pass at the main-table strength -> N1 none" % (key, m))
    C, notes = capability()
    iti = {m: iti_cells(m) for m in ITI_MODELS}

    # ---- checks ---------------------------------------------------------------------------------------------- #
    tq5 = common(data, "tqa", ORDER)
    check("TruthfulQA test common questions", len(tq5), 520)
    for row, ref in (("S0", (706, 2269)), ("SN1", (947, 2438)), ("fill EC", (819, 2553)), ("single E", (560, 1818))):
        check("TruthfulQA test 520 %s" % row, total(data, "tqa", row, ORDER, tq5), ref)
    for a, ref in (("N1", 23), ("single E", -146)):
        check("TruthfulQA test 520 %s - S0" % a, total(data, "tqa", a, ORDER, tq5)[0] - total(data, "tqa", "S0", ORDER, tq5)[0], ref)
    check("TruthfulQA test 520 SN1m net (GPT strength.txt)", total(data, "tqa", "SN1m", ORDER, tq5)[0], 956)
    q5 = common(data, "qq", ORDER)
    check("(QA)^2 held-out questions", len(q5), 470)
    for row, ref in (("S0", (40, 2013)), ("SN1", (294, 2139)), ("fill EC", (80, 2312)), ("O", SKP.REF_QQ470["O"]),
                     ("single E", SKP.REF_QQ470["E"]), ("N1", SKP.REF_QQ470["N1"])):
        check("(QA)^2 470 %s" % row, total(data, "qq", row, ORDER, q5), ref)
    for t, ref_old, ref_new in (("no_failure", -221, -180), ("failure", 432, 403), ("unverifiable", 83, 67), (None, 294, 290)):
        col = "qq" if t is None else "qq:" + t
        qs = common(data, col, ORDER)
        check("(QA)^2 %s SN1 net (GPT strength.txt)" % (t or "all"), total(data, col, "SN1", ORDER, qs)[0], ref_old)
        check("(QA)^2 %s SN1m net (GPT strength.txt)" % (t or "all"), total(data, col, "SN1m", ORDER, qs)[0], ref_new)
    fref = {"popqa test": ((-824, 1996), (-259, 1039), (-727, 1775), (-444, 1228), -443),
            "popqa test2": ((-838, 2026), (-173, 1067), (-677, 1787), (-371, 1257), -370),
            "triviaqa test": ((753, 2453), (781, 1919), (768, 2354), (579, 1941), 722),
            "triviaqa test2": ((904, 2456), (833, 1961), (881, 2373), (582, 1918), 743)}
    n1ref = {"triviaqa test": ((55, 433), (105, 401), (107, 481), (32, 460), (255, 495)),
             "popqa test": ((-163, 333), (-75, 207), (-269, 455), (-311, 443), (-194, 462)),
             "triviaqa test2": ((72, 424), (129, 409), (118, 484), (27, 469), (278, 498)),
             "popqa test2": ((-121, 309), (-51, 225), (-258, 450), (-291, 437), (-161, 463))}
    for key, ref in fref.items():
        qs = common(data, key, ORDER)
        check("%s common questions" % key, len(qs), 500)
        for row, r in zip(("O", "single E", "S0", "SE s_sel"), ref[:4]):
            check("%s five models %s (n1_on_sentence_factqa)" % (key, row), total(data, key, row, ORDER, qs), r)
        check("%s five models SEm net (GPT strength.txt)" % key, total(data, key, "SEm", ORDER, qs)[0], ref[4])
        for m, r in zip(ORDER, n1ref[key]):
            check("%s %s N1 frozen transfer (ns_prompt_table F)" % (key, m), total(data, key, "N1", (m,), col_qs(data, key, m)), r)
    for m, ref in (("llama3.1-8b", (523, -86, -108, 468)), ("gemma3-12b", (522, 12, 36, 469)), ("gemma3-27b", (523, 3, -6, 459))):
        ch, qs, it, o = iti[m]
        check("ITI %s TruthfulQA test (iti_table test printout)" % m,
              (len(qs), sum(o.values()), sum(v for v, _ in it.values()), sum(a for _, a in it.values())), ref)
        P = cells_of(data, "tqa", m)
        check("ITI %s no write equals O on shared questions" % m, all(P[q]["O"][0] == o[q] for q in qs if q in P), True)
    capref = {("MMLU", "S0"): -0.58, ("MMLU", "single E"): -0.73, ("MMLU", "SN1"): -2.47, ("MMLU", "SN1m"): -1.68,
              ("GSM8K", "S0"): -3.09, ("GSM8K", "single E"): -0.24, ("GSM8K", "SN1"): -7.45, ("GSM8K", "SN1m"): -7.58}
    for (task, row), ref in capref.items():
        txt, _ = cap_cell(C, task, row, ORDER)
        check("%s five-model mean %s" % (task, row), round(float(txt.split()[0]), 2), ref)
    lo, hi = cap_cell(C, "GSM8K", "fill EC", ORDER)[1]
    n = len(C["GSM8K"][ORDER[0]][0])
    s0r = sum(sum(C["GSM8K"][m][1]["S0"]) for m in ORDER)
    check("GSM8K fill EC right-count gain over S0, five models (GPT capability.txt 10 to 14)",
          (round(lo * 5 * n / 100.0) + sum(sum(C["GSM8K"][m][1]["O"]) for m in ORDER) - s0r,
           round(hi * 5 * n / 100.0) + sum(sum(C["GSM8K"][m][1]["O"]) for m in ORDER) - s0r), (10, 14))
    check("GSM8K fill EC five-model mean range (GPT capability.txt -2.94 to -2.88)", (round(lo, 2), round(hi, 2)), (-2.94, -2.88))
    for m, ref in (("llama3.1-8b", 86.5), ("gemma3-12b", 92.8), ("gemma3-27b", 94.5)):
        check("GSM8K %s N1 accuracy %% at cap 512 (gsm8k_n1_prompt printout)" % m,
              round(100.0 * sum(C["GSM8K"][m][1]["N1"]) / n, 1), ref)
    for m, ref in (("llama3.1-8b", -1.2), ("gemma3-12b", -11.5), ("gemma3-27b", -0.2)):
        v = [a - b for a, b in zip(C["GSM8K"][m][1]["ITI"], C["GSM8K"][m][1]["O"])]
        check("GSM8K %s ITI change (iti_gsm8k printout)" % m, round(100.0 * sum(v) / n, 1), ref)
    print("\n# checks")
    for label, got, ref, ok in CHECKS:
        print("check %s: %s [ref %s] -> %s" % (label, got, ref, "OK" if ok else "MISMATCH"))
    if not all(x[3] for x in CHECKS):
        raise SystemExit("CHECK MISMATCH: no table printed")
    print("checks: ALL OK")

    # ---- tables ---------------------------------------------------------------------------------------------- #
    print("\n# work queue 2026-11-23_queue_l section 5 summary table; DeepSeek Pro (TruthfulQA, (QA)^2), alias match (fact sets);"
          " MMLU / GSM8K = change against the original model in points (five models: mean of the per-model changes)")
    print("# SN1m strengths: " + ", ".join("%s %s (s_sel %s)" % (m, json.load(open(N3.CHOICE3 % m))["chosen"],
                                                                json.load(open(N3.CHOICE3 % m))["s_sel"]) for m in ORDER))
    print_block("five models together (common questions)", data, C, ORDER, ROWS, lambda c: common(data, c, ORDER))
    for m in ORDER:
        print_block(m, data, C, (m,), ROWS, lambda c, m=m: col_qs(data, c, m))
    # fact-set references
    print("\n" + "=" * 110 + "\nfact sets, references for the stack rows (no C pass there): SE s_sel, SEm (chosen strength), fill E "
          "(S0, its blanks by SE s_sel); nets (answers) and minus S0")
    for lab, models in [("five models", ORDER)] + [(m, (m,)) for m in ORDER]:
        parts = []
        for c, cl in COLS[5:]:
            qs = common(data, c, models)
            parts.append("%s: %s" % (cl, " | ".join("%s %s, -S0 %s" % (r, FO.fmt(total(data, c, r, models, qs)), diff(data, c, r, "S0", models, qs))
                                                    for r in FACT_REF)))
        print("  %-12s %s" % (lab, " || ".join(parts)))
    # ITI
    print("\n" + "=" * 110 + "\nITI (faithful fit, choice within the MMLU budget), TruthfulQA test")
    for m in ITI_MODELS:
        ch, qs, it, o = iti[m]
        P = cells_of(data, "tqa", m)
        sq = [q for q in col_qs(data, "tqa", m) if q in it]
        for q in sq:
            P[q]["ITI"] = it[q]
        print("  %-12s K %d alpha %g; own list %d q: no write %+d | ITI %s" % (
            m, ch["k"], ch["alpha"], len(qs), sum(o.values()), FO.fmt((sum(v for v, _ in it.values()), sum(a for _, a in it.values())))))
    for lab, models in [("three models", ITI_MODELS)] + [(m, (m,)) for m in ITI_MODELS]:
        qs = [q for q in common(data, "tqa", models) if all("ITI" in cells_of(data, "tqa", m)[q] for m in models)]
        print("  %-12s %d q | %s" % (lab, len(qs), " | ".join("%s %s" % (r, FO.fmt(total(data, "tqa", r, models, qs))) for r in ROWS + ("ITI",))))
        print("  %-12s ITI minus: %s" % ("", " | ".join("%s %s" % (r, diff(data, "tqa", "ITI", r, models, qs)) for r in ROWS)))
        print("  %-12s MMLU %s | GSM8K %s (ITI); for the same models: %s" % (
            "", cap_cell(C, "MMLU", "ITI", models)[0], cap_cell(C, "GSM8K", "ITI", models)[0],
            " | ".join("%s MMLU %s GSM8K %s" % (r, cap_cell(C, "MMLU", r, models)[0], cap_cell(C, "GSM8K", r, models)[0])
                       for r in ("single E", "N1", "S0", "fill EC", "SN1m", "SN1"))))
    print("  ITI on (QA)^2 and the fact sets: none (ITI only on TruthfulQA)")
    # small table
    print("\n" + "=" * 110 + "\nsmall table: P + E at s_sel and s_half, change against P (points, paired bootstrap)")
    for lab, models in [(m, (m,)) for m in ORDER] + [("five models", ORDER)]:
        parts = []
        for task in ("MMLU", "GSM8K"):
            for k in ("P+E", "P+E half"):
                v = [[a - b for a, b in zip(C[task][m][1][k], C[task][m][1]["P"])] for m in models]
                parts.append("%s %s - P %s" % (task, "(P+E s_sel)" if k == "P+E" else "(P+E s_half)", SC.boot(v)))
        print("  %-12s %s" % (lab, " | ".join(parts)))
    print("\n# notes")
    for x in notes:
        print("  " + x)
    print("  MMLU of N1 = single E's and of fill EC = S0's: the four-choice protocol has no blank, so no C re-ask and no fill")


if __name__ == "__main__":
    main()
