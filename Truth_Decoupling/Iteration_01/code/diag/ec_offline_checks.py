# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22), read-only: no model, no GPU, no judge call. Exploration.
# The three offline checks G1, G2, G3 of discussion/work_queue/2026-11-22_queue_g.md (section 1), on existing answers
# and DeepSeek Pro verdicts only. Five models; TruthfulQA (main-table settings at the write rule "prompt", the loaders of
# diag/tq_same_n_decomp.py) and (QA)^2 (the Raw class of diag/qaqa_pilot_table.py; the 470 held-out questions of the
# 570-question run; the caps chosen on the 100 tune questions with E fixed, splits/qaqa_tune_choice_fixedE_20261122.json).
# Value of an answer: right +1, wrong -1, anything else 0 (net = right - wrong).
# Definitions (as in the work queue):
#   E          single E at the main-table layer and strength.
#   N1(k1,k2)  E on every question; a question blank under E is re-asked with E + C at 0.25 / 0.5 / 1 / 2 up to the
#              cap of its kind (k1 = cap for E's blanks, i.e. answered without a write; k2 = cap for the model's own
#              blanks; "none" = that kind is not re-asked); the first non-blank answer is taken, for E's blanks only if
#              it differs from the no-write answer (release_origin.norm). Assembled by the existing code
#              (final_records.records / Raw.n1_system), only the caps change.
#   strong(n)  must-answer prompt + E (same layer, strength, write rule), keeping the n answered answers with the
#              largest confidence (mean token log-probability), ties in question order; one pass. "not comparable" when
#              fewer than n such answers exist.
#   twopass    E's own answers all kept; on E's blanks the must-answer + E answers fill as many blanks as N1 releases
#              there, the most confident first (tq_same_n_decomp "N1_minus_Efill", recovery_pool_b "E-only filling").
#   EC(k)      E + C at strength k written on every question, answered once (the existing "E and C on every question"
#              arms); its answer count is its own.
# G1  per model and C strength k: EC(k) answers and net; EC(k) - strong(n = EC(k)'s answers), EC(k) - E, EC(k) - N1 at
#     the chosen caps. The strength is chosen on the tune split (highest net of EC(k), ties to the smaller k): TruthfulQA
#     tune split; (QA)^2 the 100 tune questions of the pilot run (the data N1's caps were chosen on; the choice on the
#     570-run's copy of the same 100 questions is printed next to it).
# G2  the 25 cap pairs (k1, k2) in {none, 0.25, 0.5, 1, 2}^2; (none, none) = E. Per pair: N1 answers, net, strong(n)
#     net, twopass net. Statistic = mean over the 24 pairs other than (none, none) of the five-model sum of
#     (N1 - opponent); interval: questions resampled (one resample for all pairs and models), percentile 95%.
# G3  on questions blank under E (and on questions E answered wrongly): is there a right, non-blank answer among
#     C = any of the four EC(k) arms, F = must-answer + E, O = the no-write answer, M = must-answer, no write.
#     Counts only. TruthfulQA has no must-answer + E pass at "prompt" on its tune split: F is "none" there.
# Intervals: the bootstrap stream of each table's references (TruthfulQA policy_select2.ci, seed 20261116; (QA)^2
# newsets_score.joint_ci, seed 20261119; 4000 resamples, order statistics 100 and 3899). Per model: the model's own
# questions; models together: the questions every model has (TruthfulQA sorted as main_table_prompt.py; (QA)^2 the bank
# order), per-question differences summed over models, resampled jointly.
# Anchors checked before anything else is printed: N1 - strong at the chosen caps per model and together
# (TruthfulQA +51 +13 +18 -6 +52, together +128 [+58, +197]; (QA)^2 +23 +22 -23 +23 -3, together +42 [-23, +107]).
# usage:
#   TruthfulQA (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/ec_offline_checks.py tqa
#   (QA)^2 (scoring container, -w /workspace/Truth_Decoupling): python3 -B Iteration_01/diag/ec_offline_checks.py qaqa
import contextlib
import io
import json
import sys

import numpy as np

sys.path.insert(0, "Iteration_01/diag")

N_BOOT = 4000
LABELS = ("none", "0.25", "0.5", "1", "2")
LEVELS = LABELS[1:]
PAIRS = [(a, b) for a in LABELS for b in LABELS]
NONTRIVIAL = [p for p in PAIRS if p != ("none", "none")]
MODELS = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
_W = {}


def weights(n, seed):
    """Resample counts, the same index stream as policy_select2.ci / newsets_score.joint_ci."""
    import random
    if (n, seed) not in _W:
        rng = random.Random(seed)
        _W[(n, seed)] = np.asarray([np.bincount([rng.randrange(n) for _ in range(n)], minlength=n)
                                    for _ in range(N_BOOT)], dtype=np.int64)
    return _W[(n, seed)]


def ci(vec, seed, div=1):
    v = np.asarray(vec, dtype=np.int64)
    d = np.sort(weights(len(v), seed) @ v)
    if div == 1:
        return "%+d [%+d, %+d]" % (int(v.sum()), int(d[100]), int(d[3899]))
    return "%+.1f [%+.1f, %+.1f]" % (v.sum() / div, d[100] / div, d[3899] / div)


# ------------------------------------------------------------------ generic per-model data
class M:
    """One model on one split. Every table is {question: value} / {question: answered}.
    e, ea: single E; n1[(k1, k2)] = (values, answered); ec[k] = (values, answered); f = (values, answered, confidence)
    or None; right[src][q] = a right, non-blank answer exists in that source (src C, F, O, M; None = no such pass)."""

    def __init__(self, name, ids, e, ea, n1, ec, f, right, chosen):
        self.name, self.ids, self.e, self.ea, self.n1, self.ec, self.f, self.right, self.chosen = (
            name, list(ids), e, ea, n1, ec, f, right, chosen)


def strong(m, L, n):
    """{q: value} of must-answer + E keeping the n most confident answered answers among L; None when not comparable."""
    pos = {q: i for i, q in enumerate(L)}
    fv, fa, fc = m.f
    elig = [q for q in L if fa[q] and fc[q] is not None]
    if len(elig) < n:
        return None
    top = set(sorted(elig, key=lambda q: (-fc[q], pos[q]))[:n])
    return {q: (fv[q] if q in top else 0) for q in L}


def twopass(m, L, pair):
    v, a = m.n1[pair]
    pos = {q: i for i, q in enumerate(L)}
    fv, fa, fc = m.f
    blanks = [q for q in L if not m.ea[q]]
    k = sum(a[q] for q in blanks)
    if any(m.ea[q] and not a[q] for q in L):
        raise SystemExit("%s %s: N1 dropped an answer of E" % (m.name, pair))
    eb = sorted([q for q in blanks if fa[q] and fc[q] is not None], key=lambda q: (-fc[q], pos[q]))
    if len(eb) < k:
        return None
    sel = set(eb[:k])
    return {q: (fv[q] if q in sel else m.e[q]) for q in L}


def nsum(tab, L):
    return sum(tab[q] for q in L)


# ------------------------------------------------------------------ TruthfulQA loader
def shared_right(ids, read, has_forced):
    """G3 only. One verdict per literal answer on a model/question; never merge across questions.
    Freeze conflict precedence E, original, must-answer original, forced E, then ascending C dose.
    The source's own issued flag still determines whether it can submit the candidate.
    """
    result = {src: {} for src in ("C", "F", "O", "M")}
    for q in ids:
        cells = read(q)
        seen = {}
        correct = {}
        for src, text, issued, right in cells:
            key = text.strip()
            verdict = seen.setdefault(key, right)
            correct[src] = issued and verdict
        for src in ("O", "M"):
            result[src][q] = correct[src]
        if has_forced:
            result["F"][q] = correct["F"]
        result["C"][q] = any(correct[k] for k in LEVELS)
    if not has_forced:
        result["F"] = None
    return result


def tqa_models():
    import iti_table as A
    import tq_same_n_decomp as TD
    MP, ML, S = A.MP, A.ML, A.S
    MT = ML.MT
    V = TD.V
    lab = {None: "none", "0-25": "0.25", "0-5": "0.5", "1": "1", "2": "2"}
    inv = {v: k for k, v in lab.items()}
    out = {}
    for model in MODELS:
        display, layer, s = A.KEY[model]
        tune, err = MP.prompt_split(display, layer, 0, s)
        test, err2 = MP.prompt_split(display, layer, 1, s)
        if tune is None or test is None:
            raise SystemExit(err or err2)
        src = [x for x in ML.sources(display)[2] if x[0] == layer]
        if len(src) != 1:
            raise SystemExit("%s: layer %s is not one of the main table's sources" % (model, layer))
        ct = [q for q in tune[3] if q in set(src[0][1].qs)]
        cx = [q for q in test[3] if q in set(src[0][3].qs)]
        if ct != tune[3]:
            tune, _ = MP.prompt_split(display, layer, 0, s, qs=ct)
        if cx != test[3]:
            test, _ = MP.prompt_split(display, layer, 1, s, qs=cx)
        if tune is None or test is None or tune[3] != ct or test[3] != cx:
            raise SystemExit("%s: the restricted splits are not the main table's" % model)
        n1cfg = ML.n1_pick(tune[0], tune[1], s)
        chosen = ("none", "none") if n1cfg[0] == "E" else (lab[n1cfg[2]], lab[n1cfg[3]])
        _, run, _ = MP.usual(display, 1)
        per = {}
        for i, (sp, tx, _, ids) in ((0, tune), (1, test)):
            def sysd(cfg):
                st, an = S.system(sp, tx, cfg)[:2]
                return {q: V[x] for q, x in zip(ids, st)}, dict(zip(ids, an))
            e, ea = sysd(("E", s))
            n1 = {}
            for p in PAIRS:
                n1[p] = (e, ea) if p == ("none", "none") else sysd(("N1", s, inv[p[0]], inv[p[1]]))
            ec = {k: sysd(("ALL", s, inv[k])) for k in LEVELS}
            f = None
            if i == 1:
                F = TD.forced(run, layer, s, ids)
                f = ({q: F[q][0] for q in ids}, {q: F[q][1] for q in ids}, {q: F[q][2] for q in ids})
            D = MT.Data(display, i)
            mst, man = MT.must(D, "baseline_conf")
            mby = dict(zip(D.sp.qs, zip(mst, man)))
            miss = [q for q in ids if q not in mby]
            if miss:
                raise SystemExit("%s split %d: %d questions have no must-answer (no write) verdict" % (model, i, len(miss)))
            fc = f"E_L{layer}_conf-w-p{s}"
            ftexts = MP.fr.raw_texts(MP.work(run, "test", f"pospromptdl{layer}")) if f is not None else {}
            def sources(q):
                def grid(src, c):
                    return (src, tx.ec[(q, c)], not sp.blank[(q, c)], sp.st[(q, c)] == "R")
                result = [grid("E", sp.cE(s)), grid("O", "baseline_x"),
                          ("M", D.texts.d[(q, "baseline_conf")], mby[q][1], mby[q][0] == "R")]
                if f is not None:
                    result.append(("F", ftexts[(q, fc)], f[1][q], f[0][q] == 1))
                return result + [grid(k, sp.cEC(s, inv[k])) for k in LEVELS]
            right = shared_right(ids, sources, f is not None)
            per[i] = M(model, ids, e, ea, n1, ec, f, right, chosen)
            per[i].types = None
        per["info"] = "layer %d strength %s, N1 caps chosen on tune %s / %s" % (layer, s, chosen[0], chosen[1])
        out[model] = per
    return out


# ------------------------------------------------------------------ (QA)^2 loader
def qaqa_models():
    import qaqa_pilot_table as Q
    bank = "Iteration_01/newsets/bank/qaqa_full_bank_20261122.jsonl"
    splits = "Iteration_01/newsets/splits"
    works = ["test_full570", "test_full570_tuned", "test_full570_strong"]
    with open(splits + "/qaqa_pilot_20261122.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    if Q.G.sha_file(Q.TUNE_BANK) != meta["bank_sha256"]:
        raise SystemExit("pilot id file does not belong to the tune bank")
    tune_ids = set(meta["ids"])
    with open(splits + "/qaqa_tune_choice_fixedE_20261122.json", encoding="utf-8") as fh:
        archived = json.load(fh)
    lab = {None: "none", "0.25": "0.25", "0.5": "0.5", "1": "1", "2": "2"}
    inv = {v: k for k, v in lab.items()}
    out = {}
    for model in MODELS:
        el, s, cl = Q.SET[model][:3]
        fit = Q.Raw(model, "qaqa", Q.TUNE_BANK, ["test_tune100"])
        with contextlib.redirect_stdout(io.StringIO()):
            actual = Q.tune(fit, fit.ids, True)
        ch = archived[model]
        if any(actual[k] != v for k, v in ch.items()) or ch["n1_strength"] != s:
            raise SystemExit("%s: the tune choice does not reproduce" % model)
        chosen = (lab[ch["n1_cap_e"]], lab[ch["n1_cap_own"]])
        R = Q.Raw(model, "qaqafull", bank, works)
        if len(R.ids) != 570 or not tune_ids <= set(R.ids) or len(tune_ids) != 100:
            raise SystemExit("%s: unexpected roster" % model)
        forced = "conf:" + R.e(s).split(":", 1)[1]
        for a in [R.e(s), forced, "conf", "x"] + [R.ec(s, k) for k in Q.LAD]:
            R.need(a)
        conf = R.confidence("qaqafull", forced)
        if conf is None:
            raise SystemExit("%s: no confidence file of the must-answer + E pass" % model)
        if any(conf[q] is None for q in R.ids if not R.blank(forced, q)):
            raise SystemExit("%s: an answered must-answer + E answer has no confidence" % model)

        def build(Rx, ids, with_f):
            e, ea = Rx.arm_system(Rx.e(s))
            n1 = {}
            for p in PAIRS:
                v, a, _ = Rx.n1_system(s, inv[p[0]], inv[p[1]])
                n1[p] = (v, a)
            ec = {k: Rx.arm_system(Rx.ec(s, inv[k])) for k in LEVELS}
            f = None
            right = None
            if with_f:
                fv, fa = Rx.arm_system(forced)
                f = (fv, fa, conf)
                arms = [("E", Rx.e(s)), ("O", "x"), ("M", "conf"), ("F", forced)]
                arms.extend((k, Rx.ec(s, k)) for k in Q.LAD)
                def sources(q):
                    return [(src, Rx.text(arm, q), not Rx.blank(arm, q), Rx.cell(arm, q) > 0)
                            for src, arm in arms]
                right = shared_right(ids, sources, True)
            mm = M(model, ids, e, ea, n1, ec, f, right, chosen)
            mm.types = Rx.qtype
            return mm

        full = build(R, R.ids, True)
        held = [q for q in R.ids if q not in tune_ids]
        tune_full = [q for q in R.ids if q in tune_ids]
        pilot = build(fit, fit.ids, False)
        out[model] = {"full": full, "held": held, "tune_full": tune_full, "pilot": pilot,
                      "info": "E%d strength %s, C layer %d, N1 caps chosen on the 100 tune questions %s / %s" % (el, s, cl, chosen[0], chosen[1])}
    return out


# ------------------------------------------------------------------ the checks
def anchor(per, lists, common, seed, want):
    print("## anchor: N1 - strong at the caps chosen on tune (must reproduce the references)")
    tot = {q: 0 for q in common}
    got = []
    for model in MODELS:
        m, L, info = per[model]
        v, a = m.n1[m.chosen]
        sf = strong(m, L, sum(a[q] for q in L))
        d = sum(v[q] - sf[q] for q in L)
        got.append(d)
        sfc = strong(m, common, sum(a[q] for q in common))
        for q in common:
            tot[q] += v[q] - sfc[q]
        print("    %-12s %s | caps %s / %s | N1 answers %d net %+d | strong net %+d | difference %s" % (
            model, info, m.chosen[0], m.chosen[1], sum(a[q] for q in L), nsum(v, L), nsum(sf, L), ci([v[q] - sf[q] for q in L], seed)))
    together = ci([tot[q] for q in common], seed)
    print("    five models together, %d questions: %s" % (len(common), together))
    ok = got == want[0] and together == want[1]
    print("    anchor %s (want per model %s, together %s)" % ("REPRODUCED" if ok else "DIFFERS", want[0], want[1]))
    if not ok:
        raise SystemExit("anchor differs: stop")


def g1(per, tunes, common, seed, alt_tunes=None):
    print("\n## G1: E + C written on every question, one pass (EC(k)); the strength chosen on tune")
    choice = {}
    for model in MODELS:
        m, L, info = per[model]
        t, TL = tunes[model]
        tn = {k: nsum(t.ec[k][0], TL) for k in LEVELS}
        best = max(LEVELS, key=lambda k: (tn[k], -LEVELS.index(k)))
        choice[model] = best
        line = "    tune (%d questions): " % len(TL) + " | ".join("C %s net %+d (answers %d)" % (k, tn[k], sum(t.ec[k][1][q] for q in TL)) for k in LEVELS)
        line += " | E net %+d | chosen C %s" % (nsum(t.e, TL), best)
        if alt_tunes:
            a, AL = alt_tunes[model]
            an = {k: nsum(a.ec[k][0], AL) for k in LEVELS}
            ab = max(LEVELS, key=lambda k: (an[k], -LEVELS.index(k)))
            line += "\n      same 100 questions in the 570-run's copy: " + " | ".join("C %s %+d" % (k, an[k]) for k in LEVELS) + " -> would choose C %s" % ab
        print("\n  %s (%s)\n%s" % (model, info, line))
        v1, a1 = m.n1[m.chosen]
        print("    test (%d questions): E answers %d net %+d | N1 answers %d net %+d" % (
            len(L), sum(m.ea[q] for q in L), nsum(m.e, L), sum(a1[q] for q in L), nsum(v1, L)))
        for k in LEVELS:
            v, a = m.ec[k]
            n = sum(a[q] for q in L)
            sf = strong(m, L, n)
            s1 = "not comparable" if sf is None else "strong(%d) net %+d | EC - strong %s" % (n, nsum(sf, L), ci([v[q] - sf[q] for q in L], seed))
            print("    C %-4s%s answers %3d right %3d wrong %3d net %+4d | %s | EC - E %s | EC - N1 %s" % (
                k, "*" if k == best else " ", n, sum(v[q] == 1 for q in L), sum(v[q] == -1 for q in L), nsum(v, L), s1,
                ci([v[q] - m.e[q] for q in L], seed), ci([v[q] - v1[q] for q in L], seed)))
    print("\n  five models together, %d questions (* = the strength chosen on tune per model; then each fixed strength for all models)" % len(common))
    for label, pick in [("chosen", choice)] + [("C %s" % k, {mm: k for mm in MODELS}) for k in LEVELS]:
        d_s, d_e, d_n, bad, ans, netv = {q: 0 for q in common}, {q: 0 for q in common}, {q: 0 for q in common}, [], 0, 0
        for model in MODELS:
            m = per[model][0]
            v, a = m.ec[pick[model]]
            v1 = m.n1[m.chosen][0]
            n = sum(a[q] for q in common)
            ans += n
            netv += nsum(v, common)
            sf = strong(m, common, n)
            if sf is None:
                bad.append(model)
            for q in common:
                d_e[q] += v[q] - m.e[q]
                d_n[q] += v[q] - v1[q]
                if sf is not None:
                    d_s[q] += v[q] - sf[q]
        print("    %-7s answers %d net %+d | EC - strong %s | EC - E %s | EC - N1 %s" % (
            label, ans, netv, ("not comparable (%s)" % ", ".join(bad)) if bad else ci([d_s[q] for q in common], seed),
            ci([d_e[q] for q in common], seed), ci([d_n[q] for q in common], seed)))


def g2(per, common, seed):
    print("\n## G2: N1 over the 25 cap pairs (rows: cap for E's blanks; columns: cap for own blanks); (none, none) = E")
    for model in MODELS:
        m, L, info = per[model]
        print("\n  %s (%s), %d questions; cells: N1 answers / N1 net / N1 - strong / N1 - twopass ([chosen])" % (model, info, len(L)))
        for k1 in LABELS:
            cells = []
            for k2 in LABELS:
                v, a = m.n1[(k1, k2)]
                n = sum(a[q] for q in L)
                sf, tp = strong(m, L, n), twopass(m, L, (k1, k2))
                c = "%3d/%+4d/%s/%s" % (n, nsum(v, L), "n.c." if sf is None else "%+4d" % (nsum(v, L) - nsum(sf, L)),
                                       "n.c." if tp is None else "%+4d" % (nsum(v, L) - nsum(tp, L)))
                cells.append(("[%s]" if (k1, k2) == m.chosen else " %s ") % c)
            print("    %-4s " % k1 + " ".join(cells))
    print("\n  five models together, %d questions; per pair: N1 answers, N1 net, N1 - strong, N1 - twopass (sums over models)" % len(common))
    acc = {"strong": {q: 0 for q in common}, "twopass": {q: 0 for q in common}}
    pos = {"strong": 0, "twopass": 0}
    bad = {"strong": [], "twopass": []}
    for p in PAIRS:
        row = {"strong": {q: 0 for q in common}, "twopass": {q: 0 for q in common}}
        ans = netv = 0
        nc = {"strong": [], "twopass": []}
        for model in MODELS:
            m = per[model][0]
            v, a = m.n1[p]
            ans += sum(a[q] for q in common)
            netv += nsum(v, common)
            for name, opp in (("strong", strong(m, common, sum(a[q] for q in common))), ("twopass", twopass(m, common, p))):
                if opp is None:
                    nc[name].append(model)
                    continue
                for q in common:
                    row[name][q] += v[q] - opp[q]
        tag = " chosen by: %s" % ", ".join(mm for mm in MODELS if per[mm][0].chosen == p)
        print("    %-4s / %-4s answers %4d net %+5d | N1 - strong %s | N1 - twopass %s%s" % (
            p[0], p[1], ans, netv, ("n.c. " + ",".join(nc["strong"])) if nc["strong"] else "%+d" % sum(row["strong"].values()),
            ("n.c. " + ",".join(nc["twopass"])) if nc["twopass"] else "%+d" % sum(row["twopass"].values()),
            tag if tag != " chosen by: " else ""))
        if p == ("none", "none"):
            continue
        for name in acc:
            if nc[name]:
                bad[name].append(p)
                continue
            pos[name] += sum(row[name].values()) > 0
            for q in common:
                acc[name][q] += row[name][q]
    for name in acc:
        if bad[name]:
            print("    statistic against %s: not computable, %d pair(s) not comparable: %s" % (name, len(bad[name]), bad[name]))
        else:
            print("    statistic (mean over the 24 pairs of the five-model sum of N1 - %s): %s | pairs with a positive five-model difference: %d of 24" % (
                name, ci([acc[name][q] for q in common], seed, div=24), pos[name]))


def g3_counts(m, L, what):
    """what = 'blank' (E blank) or 'wrong' (E answered, wrong)."""
    if what == "blank":
        qs = [q for q in L if not m.ea[q]]
    else:
        qs = [q for q in L if m.ea[q] and m.e[q] == -1]
    r = m.right
    hasf = r["F"] is not None
    c = lambda q: r["C"][q]  # noqa: E731
    f = lambda q: hasf and r["F"][q]  # noqa: E731
    o = lambda q: r["O"][q]  # noqa: E731
    mm = lambda q: r["M"][q]  # noqa: E731
    out = dict(n=len(qs), C=sum(map(c, qs)), O=sum(map(o, qs)), M=sum(map(mm, qs)),
               onlyO_CO=sum(o(q) and not c(q) for q in qs), onlyC_CO=sum(c(q) and not o(q) for q in qs))
    if hasf:
        out.update(F=sum(map(f, qs)), onlyC=sum(c(q) and not f(q) for q in qs), onlyF=sum(f(q) and not c(q) for q in qs),
                   both=sum(c(q) and f(q) for q in qs), neither=sum(not c(q) and not f(q) for q in qs),
                   onlyC_CFO=sum(c(q) and not f(q) and not o(q) for q in qs),
                   FO_notC=sum((f(q) or o(q)) and not c(q) for q in qs),
                   none_CFO=sum(not (c(q) or f(q) or o(q)) for q in qs),
                   onlyC_all=sum(c(q) and not (f(q) or o(q) or mm(q)) for q in qs),
                   none_all=sum(not (c(q) or f(q) or o(q) or mm(q)) for q in qs))
    return out


KEYS3 = ("n", "C", "F", "O", "M", "onlyC", "onlyF", "both", "neither", "onlyC_CFO", "FO_notC", "none_CFO", "onlyC_all", "none_all",
         "onlyC_CO", "onlyO_CO")


def g3(blocks):
    """blocks: [(title, [(model, M, L)])]"""
    print("\n## G3: right, non-blank candidates by source (C = any EC(k); F = must-answer + E; O = no-write answer; M = must-answer, no write)")
    print("   columns: n questions | C F O M = questions with a right candidate in that source | onlyC onlyF both neither (C vs F)"
          " | onlyC_CFO = C right, F and O not | FO_notC = F or O right, C not | none_CFO | onlyC_all = C right, F O M not | none_all"
          " | onlyC_CO, onlyO_CO = C vs O alone;  '-' = that pass does not exist on this split")
    for title, rows in blocks:
        for what in ("blank", "wrong"):
            print("\n  %s, questions %s" % (title, "blank under E" if what == "blank" else "E answered wrongly"))
            print("    %-14s " % "" + " ".join("%9s" % k for k in KEYS3))
            tot = {}
            for model, m, L in rows:
                c = g3_counts(m, L, what)
                print("    %-14s " % model + " ".join("%9s" % c.get(k, "-") for k in KEYS3))
                for k in KEYS3:
                    if k in c and tot.get(k, 0) is not None:
                        tot[k] = tot.get(k, 0) + c[k]
                    elif k not in c:
                        tot[k] = None
            print("    %-14s " % "five models" + " ".join("%9s" % ("-" if tot.get(k) is None else tot[k]) for k in KEYS3))


def run_tqa():
    data = tqa_models()
    seed = 20261116
    per = {m: (data[m][1], data[m][1].ids, data[m]["info"]) for m in MODELS}
    common = sorted(set.intersection(*(set(data[m][1].ids) for m in MODELS)))
    print("# TruthfulQA test split, write rule prompt, DeepSeek Pro; intervals: policy_select2.ci stream (seed %d)" % seed)
    anchor(per, None, common, seed, ([51, 13, 18, -6, 52], "+128 [+58, +197]"))
    print("    reference twopass at the chosen caps (tq_same_n_decomp N1_minus_Efill), together: %s" % ci(
        [sum(data[m][1].n1[data[m][1].chosen][0][q] - twopass(data[m][1], common, data[m][1].chosen)[q] for m in MODELS) for q in common], seed))
    g1(per, {m: (data[m][0], data[m][0].ids) for m in MODELS}, common, seed)
    g2(per, common, seed)
    g3([("TruthfulQA tune split", [(m, data[m][0], data[m][0].ids) for m in MODELS]),
        ("TruthfulQA test split, common roster", [(m, data[m][1], common) for m in MODELS])])


def run_qaqa():
    import qaqa_pilot_table as Q
    data = qaqa_models()
    seed = 20261119
    held = data[MODELS[0]]["held"]
    if any(data[m]["held"] != held for m in MODELS) or len(held) != 470:
        raise SystemExit("models have different held-out questions")
    per = {m: (data[m]["full"], held, data[m]["info"]) for m in MODELS}
    print("# (QA)^2, 470 held-out questions of the 570-run, write rule prompt, DeepSeek Pro; intervals: newsets_score.joint_ci stream (seed %d)" % seed)
    anchor(per, None, held, seed, ([23, 22, -23, 23, -3], "+42 [-23, +107]"))
    print("    reference twopass at the chosen caps (recovery_pool_b 'original N1 vs E-only filling'), together: %s" % ci(
        [sum(data[m]["full"].n1[data[m]["full"].chosen][0][q] - twopass(data[m]["full"], held, data[m]["full"].chosen)[q] for m in MODELS) for q in held], seed))
    g1(per, {m: (data[m]["pilot"], data[m]["pilot"].ids) for m in MODELS}, held, seed,
       alt_tunes={m: (data[m]["full"], data[m]["tune_full"]) for m in MODELS})
    g2(per, held, seed)
    types = data[MODELS[0]]["full"].types
    blocks = []
    for title, L in (("(QA)^2 100 tune questions (570-run copy)", data[MODELS[0]]["tune_full"]), ("(QA)^2 470 held-out questions", held)):
        for t, lab in (("all", "all"),) + Q.TYPES:
            sub = [q for q in L if t == "all" or types[q] == t]
            blocks.append(("%s, %s (%d)" % (title, lab, len(sub)), [(m, data[m]["full"], sub) for m in MODELS]))
    g3(blocks)


if __name__ == "__main__":
    if sys.argv[1:] == ["tqa"]:
        run_tqa()
    elif sys.argv[1:] == ["qaqa"]:
        run_qaqa()
    else:
        raise SystemExit("usage: ec_offline_checks.py tqa | qaqa")
