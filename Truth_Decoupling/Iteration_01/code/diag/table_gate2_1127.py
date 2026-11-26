#!/usr/bin/env python3
# Iteration_01 exploration (executor you, Claude Opus 5.5, 2026-11-27), work queue discussion/work_queue/2026-11-26_queue_p.md
# section 5, segment A, readings 1-2 and the go/no-go counts. Read-only (container python3, from /workspace/Truth_Decoupling);
# no model, no judge call, nothing chosen here. Not reviewed by GPT. FalseQA held-out 600 had been read in other comparisons.
# Gate: stage 1, pool B, chosen layer, ON = p > 0.5 (audit falseqa_matched.verify_gate, unchanged). Gated system: ON -> the
# system's cell, OFF -> the original model's cell (system O) as it is.
# Original strength s: falseqa_read (choose on the 100 development questions, systems_of on the held-out) = O, E, N1, Conly,
#   TE (conf_E2:E<el>.w.p<s_sel>). Larger strength s2 (new files, diag/falseqa_gate2_1127.py): E2 = x:E<el>.w.p<s2>;
#   N1(s2) = Raw.n1_system's rule with the SAME caps (rebuilt here; the rebuild is checked cell by cell against falseqa_read's
#   N1 at s first); TE2 = conf_E2:E<el>.w.p<s2>.
# Equal answer counts (audit falseqa_matched.rankcut, unchanged): OFF answers locked; within the ON questions of the reading
#   group the side with more answered cells drops its answered cells from the lowest batch-1 self-judge margin up (ties by
#   question order); a missing margin on the side that must drop -> "not comparable". Matching is done per reading group
#   (premise false / valid / all), as falseqa_read.readings does. Intervals: falseqa_read.pair_ci (pairs resampled).
# Two-pass E at s2 (audit construction, unchanged logic): first pass E2; on ON questions where E2 is blank, per provenance
#   class (original answer blank / not), as many fills as N1(s2) makes in that class, taken from the TE2 answers that pass
#   (answered; with the veto, not the returned original answer) by the highest margin; "natural" = every passing TE2 fill.
# Checks first (stop on any mismatch): falseqa_read's known nets; the audit's ON count and gated natural (net, issued) of
#   N1 / E / Conly; the audit's equal-count N1 - E, N1 - Conly and two-pass lines at s (reproduced with these functions);
#   the N1 rebuild at s; forced ON = ungated and forced OFF = O cell by cell (s systems; s2 systems on the ON / OFF sets).
# usage: python3 -B Iteration_01/diag/table_gate2_1127.py [MODEL ...]      (default: the five models; unready ones listed)
#        python3 -B Iteration_01/diag/table_gate2_1127.py smoke MODEL       (the smoke questions, cell by cell)
import sys
from pathlib import Path

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks",
                "Iteration_01/gpt_tools_20261126/gate_audit"]
import falseqa_gate2_1127 as H    # noqa: E402
import falseqa_matched as FM      # noqa: E402

F, Q, GS = H.F, H.Q, H.GS
LAD = H.LAD
# audit output discussion/work_queue/2026-11-26_gate_audit/falseqa_matched.txt
AUD = {"llama3.1-8b": (297, {"N1": (279, 519), "E": (181, 392), "Conly": (174, 568)}),
       "qwen3.5-9b": (315, {"N1": (369, 540), "E": (347, 504), "Conly": (331, 582)}),
       "qwen3.8-27b": (312, {"N1": (379, 527), "E": (251, 338), "Conly": (285, 547)}),
       "gemma3-12b": (305, {"N1": (273, 484), "E": (215, 411), "Conly": (204, 583)}),
       "gemma3-27b": (320, {"N1": (325, 569), "E": (269, 473), "Conly": (191, 491)})}
AUD_EQ = {"llama3.1-8b": ((40, 20, 61), (85, 58, 113), (3, -15, 22), (4, -9, 18)),
          "qwen3.5-9b": ((9, -5, 22), (44, 18, 70), (-5, -17, 7), (-3, -15, 9)),
          "qwen3.8-27b": ((17, -5, 37), (77, 53, 103), (-5, -23, 13), (-1, -17, 15)),
          "gemma3-12b": ((44, 26, 62), (31, 2, 59), (-4, -21, 14), (-6, -21, 10)),
          "gemma3-27b": ((43, 24, 63), (127, 101, 152), (1, -18, 20), (13, -5, 31))}   # N1-E, N1-Conly, two-pass F / T
NA = "not comparable"


def margins(m):
    import selfjudge_probe as SP
    import selfjudge_stack_pick as SKP
    SP.SOURCES.update(F.SRC["smoke"] + F.SRC["full"] + H.sources_all())
    return SKP.lookup(m, extra_sources=(F.SRC["smoke"][0], F.SRC["full"][0], H.SRC["smoke"][0], H.SRC["full"][0]))


def verdicts(m, mode, cond):
    """(q, condition) -> value of the judged non-empty cells of the new work dir; declined cells -> 0 (listed)."""
    from judge_two_axis_run import load_verified_single
    w = H.work(m, mode)
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
            declined[k] = c["prediction"]
        else:
            raise SystemExit("%s: unresolved vote for %s %s" % (m, k[0], k[1]))
        pred[k] = c["prediction"]
    return val, pred, declined


def n1_rebuild(qs, e_cell, x_blank, x_text, rung, caps):
    """Raw.n1_system / falseqa_read.n1_cells: e_cell(q) = (value, answered, text); rung(k, q) = (value, answered, text)."""
    cap_e, cap_own = caps
    out = {}
    for q in qs:
        ev, ea, et = e_cell(q)
        if ea:
            out[q] = (ev, True, et, "0")
            continue
        own = x_blank(q)
        cap = cap_own if own else cap_e
        lad = () if cap is None else LAD[:LAD.index(cap) + 1]
        k = next((k for k in lad if rung(k, q)[1]), None)
        if k is not None and not own and Q.ro.norm(rung(k, q)[2]) == Q.ro.norm(x_text(q)):
            k = None
        out[q] = (ev, False, et, None) if k is None else (rung(k, q)[0], True, rung(k, q)[2], k)
    return out


def gated(ids, on, A, O):
    return {q: (A[q] if q in on else O[q]) for q in ids}


def hrw(S, qs):
    return (sum(S[q][1] for q in qs), sum(S[q][0] == 1 for q in qs), sum(S[q][0] == -1 for q in qs))


def nat(A, B, qs, pair):
    return F.pair_ci({q: A[q][0] - B[q][0] for q in qs}, pair, qs)


def fmt(t):
    return F.fmt_ci(t)


def show_hrw(t):
    return "handed in %d, right %d, wrong %d" % t


def model_block(m, rows, mg_all, report):
    el, s, cl, s2, st = H.setting2(m)
    B = H.base_state(m, rows)
    ids, on = B.ids, set(B.on)
    qt = {r["id"]: r["question"] for r in rows}
    pair = {r["id"]: r["pair_id"] for r in rows}
    prem = {r["id"]: r["premise"] for r in rows}
    sysd = F.systems_of(B.R, B.R2, st, B.caps, ids)
    print("\n" + "=" * 110)
    print("## %s: E%d s %s -> s2 %s, C%d; caps N1 (E's blanks %s, own blanks %s), Conly %s (chosen on the 100 development "
          "questions at s); switch ON %d / 600 (premise false %d, valid %d)" % (
              m, el, s, s2, cl, Q.show_cap(B.caps["N1"][0]), Q.show_cap(B.caps["N1"][1]), B.caps["Conly"], len(on),
              sum(prem[q] == "false" for q in on), sum(prem[q] == "valid" for q in on)))
    # ---- 0. checks
    if tuple(sum(sysd[k][q][0] for q in ids) for k in ("O", "E", "N1", "Conly")) != GS.K_FQA[m]:
        raise SystemExit("%s: falseqa_read nets differ from the known ones" % m)
    O = sysd["O"]
    g = {n: gated(ids, on, sysd[n], O) for n in ("N1", "E", "Conly")}
    natt = {n: (sum(c[0] for c in g[n].values()), sum(c[1] for c in g[n].values())) for n in g}
    if (len(on), natt) != AUD[m]:
        raise SystemExit("%s: gated systems at s differ from the audit: ON %d %s, audit %s" % (m, len(on), natt, AUD[m]))
    for n in ("N1", "E", "Conly"):
        if gated(ids, set(ids), sysd[n], O) != sysd[n] or gated(ids, set(), sysd[n], O) != O:
            raise SystemExit("%s: forced check of %s" % (m, n))
    M = mg_all
    # equal counts at s reproduce the audit (all 600)
    onl = [q for q in ids if q in on]
    aud = []
    for n in ("E", "Conly"):
        r = equal_q(g["N1"], g[n], ids, on, M, qt, pair)
        aud.append(tuple(r[0]) if r[0] is not None else None)
    for strat in (False, True):
        tp, _ = two_pass_q(onl, sysd["E"], sysd["N1"], O, sysd["TE"], M, qt, strat)
        aud.append(None if tp is None else tuple(F.pair_ci({q: (sysd["N1"][q][0] - tp[q][0]) if q in on else 0 for q in ids}, pair, ids)))
    if tuple(aud) != AUD_EQ[m]:
        raise SystemExit("%s: equal-count / two-pass lines at s differ from the audit: %s vs %s" % (m, aud, AUD_EQ[m]))
    # the N1 rebuild at s = falseqa_read's N1, every held-out question
    R = B.R
    e = R.e(s)
    reb = n1_rebuild(ids, lambda q: (R.cell(e, q), not R.blank(e, q), R.text(e, q)), lambda q: R.blank("x", q),
                     lambda q: R.text("x", q), lambda k, q: (R.cell(R.ec(s, k), q), not R.blank(R.ec(s, k), q), R.text(R.ec(s, k), q)),
                     B.caps["N1"])
    if reb != sysd["N1"]:
        raise SystemExit("%s: the N1 rebuild at s differs from falseqa_read on %d questions" % (m, sum(reb[q] != sysd["N1"][q] for q in ids)))
    # ---- new cells at s2
    new, cond, _ = H.load_new(m, "full", rows)
    val, pred, declined = verdicts(m, "full", cond)
    need = H.needed(m, B, new, onl)
    needk = {(q, cond[n]) for n, l in need.items() for q in l}
    if set(val) != needk:
        raise SystemExit("%s: judged cells %d differ from the cells that can be handed in %d (extra %d, missing %d)" % (
            m, len(val), len(needk), len(set(val) - needk), len(needk - set(val))))
    for (q, c), t in pred.items():
        n = next(k for k, v in cond.items() if v == c)
        if t.strip() != new[n][q]["text"].strip():
            raise SystemExit("%s: the judged answer of %s under %s is not the text of the new file" % (m, q, c))

    def cell(n, q):
        r = new[n][q]
        if not r["text"].strip():
            return (0, False, r["text"], None)
        if (q, cond[n]) not in val:
            raise SystemExit("%s: %s of %s has no verdict but is used" % (m, n, q))
        return (val[(q, cond[n])], not H.blank_row(r), r["text"], None)
    E2 = {q: cell("E2", q) for q in onl}
    N12 = n1_rebuild(onl, lambda q: E2[q][:3], lambda q: R.blank("x", q), lambda q: R.text("x", q),
                     lambda k, q: cell("EC2_" + k, q)[:3], B.caps["N1"])
    TE2 = {q: (cell("TE2", q) if not E2[q][1] else None) for q in onl}
    gs2 = {"N1": gated(ids, on, N12, O), "E": gated(ids, on, E2, O)}
    for n, S in (("N1", N12), ("E", E2)):
        if any(gs2[n][q] != S[q] for q in onl) or any(gs2[n][q] != O[q] for q in ids if q not in on):
            raise SystemExit("%s: forced check of %s at s2" % (m, n))
    dec = sorted(k for k in declined)
    print("checks: falseqa_read nets %s; audit ON / natural %s; audit equal-count and two-pass lines at s %s; N1 rebuild at s "
          "= falseqa_read on 600; forced ON / OFF; judged cells = cells that can be handed in (%d)%s -> PASS" % (
              GS.K_FQA[m], natt, aud, len(needk), "; declined (value 0): %s" % dec if dec else ""))
    print("new cells on the ON questions: %s" % " | ".join("%s %d" % (n, len(l)) for n, l in need.items()))
    # ---- groups
    groups = [("premise false", [q for q in ids if prem[q] == "false"]), ("premise valid", [q for q in ids if prem[q] == "valid"]),
              ("all", ids)]
    tp2 = {}
    for strat in (False, True):
        tp, info = two_pass_q(onl, E2, N12, O, {q: (TE2[q] if TE2[q] is not None else (0, False, "", None)) for q in onl}, M, qt, strat)
        tp2[strat] = (None if tp is None else gated(ids, on, tp, O), info)
        tpn, infn = two_pass_q(onl, E2, N12, O, {q: (TE2[q] if TE2[q] is not None else (0, False, "", None)) for q in onl}, M, qt,
                               strat, natural=True)
        tp2[("nat", strat)] = (gated(ids, on, tpn, O), infn)
    print("-- 1 MAIN: gated N1(s2) - gated N1(s)")
    for lab, qs in groups:
        a, b = gs2["N1"], g["N1"]
        print("  %-14s natural %s | s2 %s | s %s" % (lab, fmt(nat(a, b, qs, pair)), show_hrw(hrw(a, qs)), show_hrw(hrw(b, qs))))
        r = equal_q(a, b, qs, on, M, qt, pair)
        if r[0] is None:
            print("  %-14s equal: %s (%s)" % ("", NA, r[1]))
        else:
            print("  %-14s equal   %s (ON answers %d vs %d -> %d each) | s2 %s | s %s" % (
                "", fmt(r[0]), r[2][0], r[2][1], r[2][2], show_hrw(r[3]), show_hrw(r[4])))
        if lab == "all":
            report[m] = r[0]
    print("-- 2 gated N1(s2) minus: gated single E(s2); gated Conly; gated two-pass E(s2)")
    for lab, qs in groups:
        for name, other in (("E(s2)", gs2["E"]), ("Conly", g["Conly"])):
            a = gs2["N1"]
            r = equal_q(a, other, qs, on, M, qt, pair)
            eqs = ("%s (ON answers %d vs %d -> %d each) | N1(s2) %s | %s %s" % (fmt(r[0]), r[2][0], r[2][1], r[2][2], show_hrw(r[3]),
                                                                              name, show_hrw(r[4])) if r[0] is not None else "%s (%s)" % (NA, r[1]))
            print("  %-14s - %-6s natural %s (%s: %s) | equal %s" % (lab, name, fmt(nat(a, other, qs, pair)), name,
                                                                     show_hrw(hrw(other, qs)), eqs))
        for strat in (True, False):
            tpn, infn = tp2[("nat", strat)]
            tp, info = tp2[strat]
            lab2 = "two-pass E(s2)%s" % (" with provenance classes + veto" if strat else " one pool, no veto")
            line = "  %-14s - %s: natural (every passing fill, %d) %s (%s)" % (lab, lab2, infn, fmt(nat(gs2["N1"], tpn, qs, pair)),
                                                                              show_hrw(hrw(tpn, qs)))
            if tp is None:
                line += " | same fill slots as N1: %s (deficit %s)" % (NA, info)
            else:
                line += " | same fill slots as N1 (%d kept) %s (%s)" % (info, fmt(nat(gs2["N1"], tp, qs, pair)), show_hrw(hrw(tp, qs)))
            print(line)
    print("-- 3 reference: gated single E(s2) - gated single E(s); ON questions changing between s and s2")
    for lab, qs in groups:
        print("  %-14s natural %s | E(s2) %s | E(s) %s" % (lab, fmt(nat(gs2["E"], g["E"], qs, pair)), show_hrw(hrw(gs2["E"], qs)),
                                                          show_hrw(hrw(g["E"], qs))))
    ab = sum(1 for q in onl if sysd["E"][q][1] and not E2[q][1])
    ba = sum(1 for q in onl if not sysd["E"][q][1] and E2[q][1])
    print("  ON questions: answered at s -> blank at s2 %d; blank at s -> answered at s2 %d; blank at both %d; answered at both %d" % (
        ab, ba, sum(1 for q in onl if not sysd["E"][q][1] and not E2[q][1]), sum(1 for q in onl if sysd["E"][q][1] and E2[q][1])))
    ab = sum(1 for q in onl if sysd["N1"][q][1] and not N12[q][1])
    ba = sum(1 for q in onl if not sysd["N1"][q][1] and N12[q][1])
    print("  N1 on ON questions: answered at s -> blank at s2 %d; blank at s -> answered at s2 %d; N1(s2) fills by rung %s" % (
        ab, ba, dict(sorted(__import__("collections").Counter(N12[q][3] for q in onl if N12[q][1] and N12[q][3] != "0").items()))))


def equal_q(A, B, qs, on, M, qt, pair):
    """falseqa_matched.rankcut on the ON questions of qs; (ci, reason, (na, nb, k), hrw A', hrw B')."""
    pool = [q for q in qs if q in on]
    na, nb = sum(A[q][1] for q in pool), sum(B[q][1] for q in pool)
    k = min(na, nb)
    cut = []
    for S in (A, B):
        sub = {q: S[q] for q in pool}
        sc = {q: (F.margin(M, qt[q], S[q]) if S[q][1] else None) for q in pool}
        r = FM.rankcut(pool, sub, k, sc)
        if r is None:
            return None, "a margin missing on the side that drops (%d vs %d answered)" % (na, nb), None, None, None
        kept = {q: S[q][1] for q in pool}
        if sum(S[q][1] for q in pool) > k:
            valid = [q for q in pool if S[q][1]]
            drop = set(sorted(valid, key=lambda q: sc[q])[: len(valid) - k])
            kept = {q: S[q][1] and q not in drop for q in pool}
        cut.append((r, kept))
    (va, ka), (vb, kb) = cut
    d = {q: (va[q] - vb[q]) if q in on else (A[q][0] - B[q][0]) for q in qs}

    def h(S, v, kk):
        return (sum((kk[q] if q in on else S[q][1]) for q in qs), sum(((v[q] if q in on else S[q][0]) == 1) for q in qs),
                sum(((v[q] if q in on else S[q][0]) == -1) for q in qs))
    return F.pair_ci(d, pair, qs), None, (na, nb, k), h(A, va, ka), h(B, vb, kb)


def two_pass_q(on, E, N1, O, TE, M, qt, strat, natural=False):
    """The audit's two-pass E (falseqa_matched.evaluate; strat = provenance classes + veto): ({q: cell} on ON, fills) or
    (None, deficit) when the passing TE answers or their margins cannot fill N1's slots."""
    if any(E[q][:3] != N1[q][:3] for q in on if E[q][1]):
        raise SystemExit("N1 does not keep E's answer")
    eb = [q for q in on if not E[q][1]]
    take, deficit = set(), []
    for own in ([0, 1] if strat else [None]):
        pool = [q for q in eb if own is None or int(not O[q][1]) == own]
        quota = sum(N1[q][1] for q in pool)

        def passes(q):
            return TE[q][1] and (not strat or not O[q][1] or Q.ro.norm(O[q][2]) != Q.ro.norm(TE[q][2]))
        eligible = [q for q in pool if passes(q)]
        if natural:
            take.update(eligible)
            continue
        sc = {q: F.margin(M, qt[q], TE[q]) for q in eligible}
        if len(eligible) < quota or any(v is None for v in sc.values()):
            deficit.append((own, quota, len(eligible), "missing", sum(v is None for v in sc.values())))
        else:
            take.update(sorted(eligible, key=lambda q: -sc[q])[:quota])
    if deficit:
        return None, deficit
    return {q: (TE[q] if q in take else E[q]) for q in on}, len(take)


def ready(m):
    import selfjudge_probe as SP
    w = H.work(m, "full")
    if not (w / "judge2" / "single_pro.jsonl").is_file():
        return "not judged"
    root = Path(SP.OUT) / m
    if (root / (H.SRC["full"][0] + "_items.json")).exists() and not (root / (H.SRC["full"][0] + "_probe.json")).exists():
        return "self-judge margins not scored"
    return None


def smoke(m):
    rows = H.bank_rows()
    el, s, cl, s2, st = H.setting2(m)
    B = H.base_state(m, rows)
    ids, on = B.ids, set(B.on)
    qt = {r["id"]: r["question"] for r in rows}
    sysd = F.systems_of(B.R, B.R2, st, B.caps, ids)
    new, cond, want = H.load_new(m, "smoke", rows)
    val, pred, declined = verdicts(m, "smoke", cond)
    onl = [q for q in want if q in on]
    need = H.needed(m, B, new, onl)
    needk = {(q, cond[n]) for n, l in need.items() for q in l}
    print("# smoke %s: questions %s; caps N1 %s; judged cells %d = cells that can be handed in %d: %s" % (
        m, want, B.caps["N1"], len(val), len(needk), set(val) == needk))
    if set(val) != needk:
        raise SystemExit("smoke: judged cells differ")
    M = margins(m)
    R = B.R

    def cell(n, q):
        r = new[n][q]
        if not r["text"].strip():
            return (0, False, r["text"], None)
        return (val[(q, cond[n])], not H.blank_row(r), r["text"], None)
    E2 = {q: cell("E2", q) for q in onl}
    N12 = n1_rebuild(onl, lambda q: E2[q][:3], lambda q: R.blank("x", q), lambda q: R.text("x", q),
                     lambda k, q: cell("EC2_" + k, q)[:3], B.caps["N1"])
    for q in want:
        o = sysd["O"][q]
        print("\n%s premise %s: switch %s (p %.3f) | O value %d answered %s" % (
            q, B.R.qtype[q], "ON" if q in on else "OFF", __import__("json").load(open(H.GS.OUT / m / "fit_B_gate_1126.json"))["probs"]["chosen"]["fqa_held"][q],
            o[0], o[1]))
        print("   N1(s): value %d answered %s level %s | E(s) value %d answered %s" % (sysd["N1"][q][0], sysd["N1"][q][1], sysd["N1"][q][3],
                                                                                    sysd["E"][q][0], sysd["E"][q][1]))
        for n in H.arms(m):
            r = new[n][q]
            k = (q, cond[n])
            mg = F.margin(M, qt[q], (0, True, r["text"])) if r["text"].strip() else None
            print("   %-7s blank %-5s judged %-5s value %-4s margin %-8s | %s" % (n, H.blank_row(r), k in val, val.get(k),
                                                                              "%.3f" % mg if mg is not None else None,
                                                                              r["text"].replace("\n", " ")[:90]))
        if q in on:
            c = N12[q]
            print("   gated N1(s2) hands in: %s (value %d answered %s)" % ("E2" if c[3] == "0" else ("rung " + c[3] if c[3] else "E2 blank, no fill"), c[0], c[1]))
        else:
            print("   gated N1(s2) hands in: O (value %d answered %s)" % (o[0], o[1]))


def main():
    a = sys.argv[1:]
    if a[:1] == ["smoke"]:
        smoke(a[1])
        return
    models = [m for m in H.ORDER if m in a] if a else list(H.ORDER)
    if a and len(models) != len(a):
        raise SystemExit("unknown model in %s" % a)
    rows = H.bank_rows()
    print("# FalseQA held-out 600, the same frozen switch (pool B, chosen layer, ON = p > 0.5); gated N1 at s2 vs gated N1 at s.")
    print("# Exploration; the held-out verdicts of FalseQA had been read in other comparisons (screening only). Pro judge.")
    print("# Equal counts: OFF answers locked; inside ON the side with more answered cells drops its lowest self-judge margins "
          "(batch 1, no write); matched per reading group. Intervals: pairs resampled (falseqa_read.pair_ci).")
    report, missing = {}, []
    for m in models:
        why = ready(m)
        if why:
            missing.append("%s (%s)" % (m, why))
            continue
        model_block(m, rows, margins(m), report)
    print("\n" + "=" * 110)
    print("## go/no-go counts (plan section 5): lower end of the MAIN reading (equal counts, all 600) above 0")
    n = 0
    for m in H.ORDER:
        if m not in report:
            continue
        r = report[m]
        if r is None:
            print("  %-12s %s" % (m, NA))
            continue
        up = r[1] > 0
        n += up
        print("  %-12s %s -> lower end above 0: %s" % (m, fmt(r), "yes" if up else "no"))
    print("  models with the lower end above 0: %d of %d finished" % (n, len(report)))
    if missing:
        print("  missing: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
