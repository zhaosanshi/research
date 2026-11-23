#!/usr/bin/env python3
# Iteration_01 exploration (Claude Opus 5.5, executor E, 2026-11-23), work queue discussion/work_queue/2026-11-23_queue_l.md
# section 11: the fresh confirmation on FalseQA. Read-only except export-margins (writes self-judge item lists, no score).
# No model, no judge call. Nothing is chosen here except what section 11 says is chosen on the 100 development questions.
# Bank Iteration_01/newsets/falseqa/falseqa_full_bank_20261123.jsonl (100 dev rows = 50 pairs first, then 600 held-out
# rows = 300 pairs; field "set"; "premise" false / valid; "qtype" failure / no_failure). 22 arms per model
# (code/launchers/explore_20261120/falseqa_model.sh), gen dir newsets/gen_prompt/falseqa/<model>/test, DeepSeek Pro
# verdicts in newsets/judge/falseqa/<model>/test_full700_dev and _heldout (falseqa_judge.sh).
# Answers and verdicts are read with diag/qaqa_pilot_table.Raw (unchanged; subclass FRaw only restricts the questions
# whose verdicts are required, the generation file is still checked against the whole bank). The sentence row is read
# through a second Raw whose plain names are aliases of the x_E2 arms (as n1_on_sentence2.qaqa / fill_only.qq).
# Systems (per model; a cell = (value, answered, text, C level)):
#   O = x (original); E = x:E<EL>.w.p<S> (single E); N1 = E, blanks re-asked with E + C up to the two caps
#   (Raw.n1_system via selfjudge_stack_pick.n1_arms); Conly = O, blanks re-asked with C alone up to one cap
#   (n1_on_sentence2.sc_system); S0 = x_E2; SE = x_E2:E<EL>.w.p<s_sel>; SN1 = N1 on the sentence row; SC = S0, blanks
#   re-asked with C alone; T2 = conf_E2; TE = conf_E2:E<EL>.w.p<s_sel>.
#   B = the questions S0 leaves blank by the rule flag. Outside B every fill system hands in S0's cell; inside B:
#   fill EC = SN1, fill E = SE, fill C = SC, fill prompt = T2, fill pE2 = SE when SE answers, else TE.
# Chosen on the 100 development questions only (queue L section 11; rule of (QA)^2): the two caps of N1 and of SN1 over
# every pair of (none, 0.25, 0.5, 1, 2), highest dev net, ties to the earlier pair; the one cap of Conly and of SC over
# (0.25, 0.5, 1, 2), highest dev net, ties to the smaller cap.
# Held-out readings (premise false / valid / all; per model and the five models; intervals: newsets_score.joint_ci on the
# per-pair sums over the models, i.e. pairs are resampled):
#   1 N1 - S0, natural and equal answer counts;  2 fill EC - S0;
#   3 fill EC - fill C (pool B), fill EC - fill E (pool B), fill EC - fill pE2 (pool B and SE blank: the shared SE first
#     answers are kept): natural fill counts and equal fill counts inside the pool;
#   4 N1 - E, N1 - Conly at equal answer counts;  5 SN1 - S0, E - S0 (natural, equal), every system's net and answers,
#     and on valid-premise questions every system minus O.
#   Equal counts: per model the side with more answered cells in the pool drops them from the lowest batch-1 self-judge
#   margin up (baseline condition, selfjudge_stack_pick.equal; ties by question order); a missing margin makes that line n/a.
# Self-judge items (export-margins MODEL smoke|full): every answered answer a system can hand in (O, E, S0, SE, T2, TE
# everywhere; E + C rungs where E is blank; C-only rungs where O is blank; sentence E + C rungs where SE is blank;
# sentence C-only rungs where S0 is blank), one item per statement (sha1 of selfjudge_probe.statement), statements
# already scored at batch size 1 left out; sources fqa0_* (smoke, the 16-question check files) / fqa1_* (full), plus
# 16 random items scored twice in order and once shuffled (_chk / _chka / _chkb, as fill_only2.export).
# usage (scoring container, from /workspace/Truth_Decoupling):
#   python3 -B Iteration_01/diag/falseqa_read.py check [MODEL ...]     bank and id file; with models: plan, roster, full files
#   python3 -B Iteration_01/diag/falseqa_read.py status [MODEL ...]    which files and verdicts exist (never fails)
#   python3 -B Iteration_01/diag/falseqa_read.py export-margins MODEL smoke|full
#   python3 -B Iteration_01/diag/falseqa_read.py check-margins MODEL smoke|full
#   python3 -B Iteration_01/diag/falseqa_read.py smoke MODEL [N]        print the first N (3) answers of every arm (server only)
#   python3 -B Iteration_01/diag/falseqa_read.py table [MODEL ...]      dev choices; held-out readings when judged
#   python3 -B Iteration_01/diag/falseqa_read.py selftest               synthetic test of the counting logic
import collections
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["Iteration_01/diag", "Iteration_01/code/scripts/newsets", "Iteration_01/code/scripts/review_checks"]
import qaqa_pilot_table as Q     # noqa: E402

G, S = Q.G, Q.S
TASK = "falseqa"
NSD = "Iteration_01/newsets"
BANK = NSD + "/falseqa/falseqa_full_bank_20261123.jsonl"
IDS_CFG = "Iteration_01/code/configs/newsets/falseqa_ids_20261123.json"
IDS_COPY = NSD + "/falseqa/falseqa_ids_20261123.json"
WORKS = ("test_full700_dev", "test_full700_heldout")
CHOICE = NSD + "/h22/n1s_choice_%s.json"
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
LAD, CAPS = Q.LAD, Q.CAPS
N_DEV, N_HELD = 100, 600
PREM = (("false", "failure"), ("valid", "no_failure"))
SRC = {"smoke": ("fqa0_items", "fqa0_chk", "fqa0_chka", "fqa0_chkb"), "full": ("fqa1_items", "fqa1_chk", "fqa1_chka", "fqa1_chkb")}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gdir(model):
    return Path(NSD) / "gen_prompt" / TASK / model / "test"


def chkdir(model):
    return Path(NSD) / "gen_prompt" / "h22chk" / TASK / model / "test"


def load_bank(path=BANK):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---- bank and id file -------------------------------------------------------------------------------------------- #
def check_bank():
    rows = load_bank()
    bad = []
    ids = [r["id"] for r in rows]
    if len(rows) != N_DEV + N_HELD or len(set(ids)) != len(ids):
        bad.append("%d rows, %d distinct ids" % (len(rows), len(set(ids))))
    if [r.get("set") for r in rows] != ["dev"] * N_DEV + ["heldout"] * N_HELD:
        bad.append("not 100 dev rows followed by 600 held-out rows")
    if any(r.get("split") != "test" or r.get("pro100") is not True for r in rows):
        bad.append("a row without split test / pro100 true (the export takes all rows of split test)")
    for r in rows:
        want = {"false": ("failure", "f"), "valid": ("no_failure", "v")}.get(r.get("premise"))
        if want is None or r.get("qtype") != want[0] or r["id"] != r["pair_id"] + want[1]:
            bad.append("row %s: premise / qtype / id disagree" % r["id"])
        if not isinstance(r.get("gold"), list) or not r["gold"] or any(not isinstance(g, str) or not g.strip() for g in r["gold"]):
            bad.append("row %s: no reference answer" % r["id"])
        if not isinstance(r.get("question"), str) or not r["question"].strip():
            bad.append("row %s: no question" % r["id"])
    pairs = {}
    for r in rows:
        pairs.setdefault((r["set"], r["pair_id"]), []).append(r["premise"])
    if any(sorted(v) != ["false", "valid"] for v in pairs.values()):
        bad.append("a pair without exactly one false and one valid member")
    sets = {s: {p for (t, p) in pairs if t == s} for s in ("dev", "heldout")}
    if sets["dev"] & sets["heldout"] or len(sets["dev"]) != N_DEV // 2 or len(sets["heldout"]) != N_HELD // 2:
        bad.append("pair sets: dev %d, held-out %d, shared %d" % (len(sets["dev"]), len(sets["heldout"]), len(sets["dev"] & sets["heldout"])))
    if Path(IDS_CFG).read_bytes() != Path(IDS_COPY).read_bytes():
        bad.append("the id file in code/configs differs from the copy next to the bank")
    idf = json.loads(Path(IDS_CFG).read_text(encoding="utf-8"))
    if idf.get("bank_sha256", {}).get(os.path.basename(BANK)) != sha_file(BANK):
        bad.append("the id file records another bank")
    for s in ("dev", "heldout"):
        got = {(e["pair_id"], e["ids"]["false"], e["ids"]["valid"]) for e in idf.get(s, [])}
        want = {(p, p + "f", p + "v") for p in sets[s]}
        if got != want:
            bad.append("id file %s pairs differ from the bank" % s)
    print("bank %s (sha256 %s): %d rows | dev %s | held-out %s | gold lengths %s | id file sha256 %s" % (
        BANK, sha_file(BANK)[:16], len(rows),
        dict(collections.Counter(r["premise"] for r in rows if r["set"] == "dev")),
        dict(collections.Counter(r["premise"] for r in rows if r["set"] == "heldout")),
        dict(collections.Counter((r["set"], r["premise"], len(r["gold"])) for r in rows)), sha_file(IDS_CFG)[:16]))
    for b in bad[:20]:
        print("  BANK CHECK: " + b)
    return not bad, rows


# ---- settings and roster ---------------------------------------------------------------------------------------- #
def setting(model):
    el, s, cl = Q.SET[model][:3]
    ss = json.load(open(CHOICE % model, encoding="utf-8"))["strength_dot"]
    return el, s, cl, ss


def roster(model):
    el, s, cl, ss = setting(model)
    e, se, te = "x:E%d.w.p%s" % (el, s), "x_E2:E%d.w.p%s" % (el, ss), "conf_E2:E%d.w.p%s" % (el, ss)
    r = [("O", "x"), ("E", e)] + [("EC" + k, "%s+C%d.w.m%s" % (e, cl, k)) for k in LAD] + [("C" + k, "x:C%d.w.m%s" % (cl, k)) for k in LAD]
    r += [("S0", "x_E2"), ("SE", se)] + [("SEC" + k, "%s+C%d.w.m%s" % (se, cl, k)) for k in LAD]
    r += [("SC" + k, "x_E2:C%d.w.m%s" % (cl, k)) for k in LAD] + [("T2", "conf_E2"), ("TE", te)]
    out = collections.OrderedDict(r)
    if len(out) != 22 or len(set(out.values())) != 22:
        raise SystemExit("%s: the roster is not 22 distinct arms" % model)
    return out


def check_plan(model):
    import n1_on_sentence as N
    el, s, cl, ss = setting(model)
    layer, sdot, _, c = N.plan(model)
    ok = (layer, sdot, c) == (el, s, cl)
    print("plan %s: main table E%s at %s, C%s | qaqa_pilot_table.SET E%s at %s, C%s | s_sel %s -> %s" % (
        model, layer, sdot, c, el, s, cl, ss, "OK" if ok else "MISMATCH"))
    return ok


def gen_args(model):
    return SimpleNamespace(task=TASK, split="test", model=model, items=BANK)


def file_state(model, arm, ids):
    """("full", rows) | ("16", rows) | ("missing", None), every file checked against the bank and its plan."""
    tok = G.arm_token(arm)
    p = gdir(model) / (tok + ".json")
    want_pos = "prompt" if ":" in arm else None
    if p.is_file():
        g = G.load_generation(p, gen_args(model), ids)
        if g["contract"]["arm"] != arm or g["contract"].get("positions") != want_pos:
            raise SystemExit("%s: %s is not the pass %s" % (model, p, arm))
        return "full", g["rows"]
    subs = sorted(glob.glob(str(chkdir(model) / ("s????????????.%s.json" % tok))))
    if len(subs) > 1:
        raise SystemExit("%s: two check files for %s" % (model, arm))
    if subs:
        g = json.loads(Path(subs[0]).read_text(encoding="utf-8"))
        n = len(g["rows"])
        g = G.load_generation(subs[0], gen_args(model), ids[:n], allow_subset=True)
        if g["contract"]["arm"] != arm or g["contract"].get("positions") != want_pos:
            raise SystemExit("%s: %s is not the pass %s" % (model, subs[0], arm))
        return str(n), g["rows"]
    return "missing", None


def run_check(models, need_full):
    ok, rows = check_bank()
    ids = [r["id"] for r in rows]
    for m in models:
        ok &= check_plan(m)
        st = collections.Counter()
        for name, arm in roster(m).items():
            state, _ = file_state(m, arm, ids)
            st[state] += 1
            if need_full and state != "full":
                print("  %s: %s (%s) has no full 700-question file" % (m, name, arm))
                ok = False
        print("files %s: %s" % (m, dict(st)))
    print("check: %s" % ("PASS" if ok else "FAILED"))
    return ok


def run_status(models):
    rows = load_bank()
    ids = [r["id"] for r in rows]
    for m in models:
        st = {name: file_state(m, arm, ids)[0] for name, arm in roster(m).items()}
        line = []
        for w in WORKS:
            p = Path(NSD) / "judge" / TASK / m / w / "judge2"
            n_c = sum(1 for _ in open(p / "cells.jsonl")) if (p / "cells.jsonl").is_file() else 0
            n_s = sum(1 for _ in open(p / "single_pro.jsonl")) if (p / "single_pro.jsonl").is_file() else 0
            line.append("%s cells %d, single_pro rows %d" % (w, n_c, n_s))
        print("%s: files %s | %s" % (m, dict(collections.Counter(st.values())), " | ".join(line)))


# ---- answers and verdicts ---------------------------------------------------------------------------------------- #
class FRaw(Q.Raw):
    """Raw on the FalseQA bank; verdicts are required only on the questions `keep` (dev or held-out)."""

    def __init__(self, model, keep, works=WORKS):
        Q.Raw.__init__(self, model, TASK, BANK, works)
        self.all_ids = list(self.ids)
        self.ids = [q for q in self.all_ids if q in keep]

    def has(self, arm):   # Raw.has with the generation checked on all_ids and the verdicts required on ids only
        if arm in self.rows:
            return True
        p = self.gdir / (G.arm_token(arm) + ".json")
        if not p.is_file():
            return False
        g = G.load_generation(p, self.args, self.all_ids)
        k = g["contract"]
        if k["arm"] != arm or k.get("positions") != ("prompt" if ":" in arm else None):
            raise SystemExit("%s: %s is not the pass %s at the expected write rule" % (self.model, p.name, arm))
        rows = {r["id"]: r for r in g["rows"]}
        if any((q, k["condition"]) not in self.val and rows[q]["text"].strip() for q in self.ids):
            return False
        for q in self.ids:
            key = (q, k["condition"])
            if key in self.pred and self.pred[key].strip() != rows[q]["text"].strip():
                raise SystemExit("%s: the judged answer of %s under %s is not the text now in %s" % (self.model, q, arm, p.name))
        self.rows[arm], self.cond[arm] = rows, k["condition"]
        return True


def load_rows(model, keep, works=WORKS):
    """(R, R2): R the plain row and the must-answer arms, R2 the sentence row under the plain names."""
    ro_ = roster(model)
    R = FRaw(model, keep, works)
    for name in ["O", "E"] + ["EC" + k for k in LAD] + ["C" + k for k in LAD] + ["T2", "TE"]:
        R.need(ro_[name])
    R2 = FRaw(model, keep, works)
    alias(R2, [ro_[n] for n in ["S0", "SE"] + ["SEC" + k for k in LAD] + ["SC" + k for k in LAD]])
    return R, R2


def alias(R2, arms):
    for a in arms:
        R2.need(a)
        b = a.replace("x_E2:", "x:") if ":" in a else "x"
        R2.rows[b], R2.cond[b] = R2.rows[a], R2.cond[a]


def cells(R, arm, qs):
    return {q: (R.cell(arm, q), not R.blank(arm, q), R.text(arm, q), None) for q in qs}


def n1_cells(R, s, caps, qs):
    import selfjudge_stack_pick as SKP
    arm, v, a = SKP.n1_arms(R, s, caps[0], caps[1])
    out = {}
    for q in qs:
        lv = None
        if a[q]:
            lv = "0" if arm[q] == R.e(s) else arm[q].rsplit(".m", 1)[1]
        out[q] = (v[q], bool(a[q]), R.text(arm[q], q), lv)
    return out


def ladder_cells(R, cap, qs):
    """The pass "x" of R first; a question it leaves blank is re-asked with "x:C<CL>.w.m<c>" from 0.25 up to cap, the
    first answered rung is taken (n1_on_sentence2.sc_system, unchanged; its keys are policy_select2.LAD)."""
    import n1_on_sentence2 as N2
    ps = N2.ps
    key = {ps.NAME[k]: k for k in ps.LAD}
    base = cells(R, "x", qs)
    cs = {key[k]: cells(R, R.c(k), qs) for k in LAD}
    out = N2.sc_system(base, cs, key[cap], qs)
    lv = {}
    for q in qs:
        if not base[q][1] and out[q][1]:
            lv[q] = next(k for k in LAD if cs[key[k]][q] is out[q])
    return {q: out[q][:3] + (lv.get(q),) for q in qs}


def choose(R, R2, st):
    """The caps chosen on the questions of R / R2 (the development questions)."""
    el, s, cl, ss = st
    out, lines = {}, []
    for name, RR, sv in (("N1", R, s), ("SN1", R2, ss)):
        grid = [((ce, co), sum(RR.n1_system(sv, ce, co)[0][q] for q in RR.ids)) for ce in CAPS for co in CAPS]
        best = max(grid, key=lambda x: (x[1], -grid.index(x)))
        out[name] = best[0]
        lines.append("%s caps E's blanks %s / own blanks %s (dev net %+d); grid rows cap for E's blanks none..2, columns own: %s" % (
            name, Q.show_cap(best[0][0]), Q.show_cap(best[0][1]), best[1],
            " / ".join(" ".join("%+d" % grid[i * len(CAPS) + j][1] for j in range(len(CAPS))) for i in range(len(CAPS)))))
    for name, RR in (("Conly", R), ("SC", R2)):
        nets = [(sum(ladder_cells(RR, k, RR.ids)[q][0] for q in RR.ids), k) for k in LAD]
        cap = max(nets, key=lambda x: (x[0], -LAD.index(x[1])))[1]
        out[name] = cap
        lines.append("%s cap %s (dev nets %s)" % (name, cap, " | ".join("%s %+d" % (k, n) for n, k in nets)))
    return out, lines


def systems_of(R, R2, st, caps, qs):
    el, s, cl, ss = st
    d = {"O": cells(R, "x", qs), "E": cells(R, R.e(s), qs), "N1": n1_cells(R, s, caps["N1"], qs),
         "Conly": ladder_cells(R, caps["Conly"], qs),
         "S0": cells(R2, "x", qs), "SE": cells(R2, R2.e(ss), qs), "SN1": n1_cells(R2, ss, caps["SN1"], qs),
         "SC": ladder_cells(R2, caps["SC"], qs),
         "T2": cells(R, "conf_E2", qs), "TE": cells(R, "conf_E2:E%d.w.p%s" % (el, ss), qs)}
    f = {k: {} for k in ("fill EC", "fill E", "fill C", "fill prompt", "fill pE2")}
    for q in qs:
        s0 = d["S0"][q]
        inb = not s0[1]
        f["fill EC"][q] = d["SN1"][q] if inb else s0
        f["fill E"][q] = d["SE"][q] if inb else s0
        f["fill C"][q] = d["SC"][q]
        if not inb and d["SC"][q] != s0:
            raise SystemExit("SC differs from S0 on a question S0 answers (%s)" % q)
        f["fill prompt"][q] = d["T2"][q] if inb else s0
        f["fill pE2"][q] = ((d["SE"][q] if d["SE"][q][1] else d["TE"][q]) if inb else s0)
        if d["SE"][q][1] and d["SN1"][q][:3] != d["SE"][q][:3]:
            raise SystemExit("SN1 does not keep SE's answer on %s" % q)
        if d["E"][q][1] and d["N1"][q][:3] != d["E"][q][:3]:
            raise SystemExit("N1 does not keep E's answer on %s" % q)
    d.update(f)
    return d


# ---- counting ------------------------------------------------------------------------------------------------------ #
def statement_sha(qt, text):
    import selfjudge_probe as SP
    return SP.sha1(SP.statement(qt, text))


def margin(M, qt, c):
    if not c[1]:
        return None
    return M.get(statement_sha(qt, c[2]), {}).get("baseline")


def equal_diff(qs, pool, A, B, qt, M):
    """Per-question a - b over qs; inside pool the side with more answered cells drops them from the lowest margin up
    (selfjudge_stack_pick.equal). (diff or None when a needed margin is missing, (na, nb) in the pool, missing count)."""
    import selfjudge_stack_pick as SKP
    Aa = {q: A[q][:2] + (margin(M, qt[q], A[q]),) for q in pool}
    Bb = {q: B[q][:2] + (margin(M, qt[q], B[q]),) for q in pool}
    na, nb = sum(x[1] for x in Aa.values()), sum(x[1] for x in Bb.values())
    big = Aa if na > nb else Bb
    miss = sum(1 for x in big.values() if x[1] and x[2] is None) if na != nb else 0
    if miss:
        return None, (na, nb), miss
    va, vb, _ = SKP.equal(pool, Aa, Bb)
    d = {q: A[q][0] - B[q][0] for q in qs}
    for q in pool:
        d[q] = va[q][0] - vb[q][0]
    return d, (na, nb), 0


def pair_ci(diff_by_q, pair_of, qs):
    """Total and 95% interval with the pairs resampled: the per-question values (already summed over models) are summed
    per pair, then newsets_score.joint_ci (fixed seed, 2,000 resamples of the pairs)."""
    order, sums = [], {}
    for q in qs:
        p = pair_of[q]
        if p not in sums:
            order.append(p)
            sums[p] = 0
        sums[p] += diff_by_q[q]
    return S.joint_ci([[sums[p] for p in order]])


def fmt_ci(t):
    return "%+d [%+d, %+d]" % tuple(t)


# ---- table ----------------------------------------------------------------------------------------------------- #
SYS_ORDER = ("O", "E", "N1", "Conly", "S0", "SE", "SN1", "SC", "T2", "TE", "fill EC", "fill E", "fill C", "fill prompt", "fill pE2")


def margins_of(model):
    import selfjudge_probe as SP
    import selfjudge_stack_pick as SKP
    SP.SOURCES.update(SRC["smoke"] + SRC["full"])
    return SKP.lookup(model, extra_sources=(SRC["smoke"][0], SRC["full"][0]))


def readings(data, models, qt, pair_of, groups, M):
    """data[m] = {system: {q: cell}}; groups = [(label, qs)]."""
    rows = [(m, (m,)) for m in models] + [("%d models" % len(models), tuple(models))]

    def tot(ms, qs, name):
        return sum(data[m][name][q][0] for m in ms for q in qs), sum(data[m][name][q][1] for m in ms for q in qs)

    def nat(ms, qs, a, b):
        return pair_ci({q: sum(data[m][a][q][0] - data[m][b][q][0] for m in ms) for q in qs}, pair_of, qs)

    def eq(ms, qs, a, b, pool_of):
        de, cnt, miss = {q: 0 for q in qs}, [], 0
        for m in ms:
            pool = [q for q in qs if pool_of(m, q)]
            d, c, mm = equal_diff(qs, pool, data[m][a], data[m][b], qt, M.get(m, {}))
            cnt.append(c)
            miss += mm
            if d is not None:
                for q in qs:
                    de[q] += d[q]
        if miss:
            return "n/a (%d margins missing)" % miss
        return "%s (pool answers %d vs %d -> %d each)" % (fmt_ci(pair_ci(de, pair_of, qs)), sum(c[0] for c in cnt),
                                                          sum(c[1] for c in cnt), sum(min(c) for c in cnt))
    allq = lambda m, q: True                                                   # noqa: E731
    in_b = lambda m, q: not data[m]["S0"][q][1]                               # noqa: E731
    in_p = lambda m, q: not data[m]["S0"][q][1] and not data[m]["SE"][q][1]   # noqa: E731
    for label, qs in groups:
        print("\n" + "=" * 100 + "\n## %s (%d questions per model, %d pairs)" % (label, len(qs), len({pair_of[q] for q in qs})))
        print("-- every system: net (answers)")
        for lab, ms in rows:
            print("  %-12s %s" % (lab, " | ".join("%s %+d (%d)" % ((n,) + tot(ms, qs, n)) for n in SYS_ORDER)))
        print("-- fills inside B (S0 blank by the rule): count right / wrong / no information")
        for lab, ms in rows:
            out = []
            for f in ("fill EC", "fill E", "fill C", "fill prompt", "fill pE2"):
                got = [data[m][f][q] for m in ms for q in qs if not data[m]["S0"][q][1] and data[m][f][q][1]]
                out.append("%s %d: %d / %d / %d" % (f, len(got), sum(c[0] == 1 for c in got), sum(c[0] == -1 for c in got),
                                                    sum(c[0] == 0 for c in got)))
            print("  %-12s B %d | %s" % (lab, sum(in_b(m, q) for m in ms for q in qs), " | ".join(out)))
        print("-- 1 confirmation one: N1 - S0")
        for lab, ms in rows:
            print("  %-12s natural %s | equal answers %s" % (lab, fmt_ci(nat(ms, qs, "N1", "S0")), eq(ms, qs, "N1", "S0", allq)))
        print("-- 2 confirmation two: fill EC - S0")
        for lab, ms in rows:
            print("  %-12s %s" % (lab, fmt_ci(nat(ms, qs, "fill EC", "S0"))))
        print("-- 3 confirmation three: fill EC minus fill C / fill E / fill pE2; natural, then equal fills inside the pool")
        for lab, ms in rows:
            print("  %-12s fill EC - fill C   natural %s | equal in B %s" % (lab, fmt_ci(nat(ms, qs, "fill EC", "fill C")),
                                                                          eq(ms, qs, "fill EC", "fill C", in_b)))
            print("  %-12s fill EC - fill E   natural %s | equal in B %s" % ("", fmt_ci(nat(ms, qs, "fill EC", "fill E")),
                                                                          eq(ms, qs, "fill EC", "fill E", in_b)))
            print("  %-12s fill EC - fill pE2 natural %s | equal in B and SE blank (SE first answers kept) %s" % (
                "", fmt_ci(nat(ms, qs, "fill EC", "fill pE2")), eq(ms, qs, "fill EC", "fill pE2", in_p)))
        print("-- 4 the main sentence on new questions (no honest sentence): equal answer counts")
        for lab, ms in rows:
            print("  %-12s N1 - E %s | N1 - Conly %s || natural N1 - E %s, N1 - Conly %s" % (
                lab, eq(ms, qs, "N1", "E", allq), eq(ms, qs, "N1", "Conly", allq),
                fmt_ci(nat(ms, qs, "N1", "E")), fmt_ci(nat(ms, qs, "N1", "Conly"))))
        print("-- 5 SN1 - S0; E - S0 (natural, equal answers)")
        for lab, ms in rows:
            print("  %-12s SN1 - S0 %s | E - S0 natural %s, equal %s" % (lab, fmt_ci(nat(ms, qs, "SN1", "S0")),
                                                                     fmt_ci(nat(ms, qs, "E", "S0")), eq(ms, qs, "E", "S0", allq)))
        if label.startswith("held-out, premise valid"):
            print("-- 5 valid premise: every system minus the original model O (natural)")
            for lab, ms in rows:
                print("  %-12s %s" % (lab, " | ".join("%s %s" % (n, fmt_ci(nat(ms, qs, n, "O"))) for n in SYS_ORDER if n != "O")))
    sys.stdout.flush()


def run_table(models):
    ok, rows = check_bank()
    if not ok:
        raise SystemExit("bank check FAILED")
    dev = {r["id"] for r in rows if r["set"] == "dev"}
    held = {r["id"] for r in rows if r["set"] == "heldout"}
    qt = {r["id"]: r["question"] for r in rows}
    pair_of = {r["id"]: r["pair_id"] for r in rows}
    prem = {r["id"]: r["premise"] for r in rows}
    data, done, M = {}, [], {}
    for m in models:
        st = setting(m)
        try:
            R, R2 = load_rows(m, dev)
        except FileNotFoundError as err:
            print("# %s: development set not ready (%s)" % (m, str(err)[:200]))
            continue
        caps, lines = choose(R, R2, st)
        print("# %s: E%d main %s, s_sel %s, C%d; chosen on the %d development questions:" % (m, st[0], st[1], st[3], st[2], len(R.ids)))
        for x in lines:
            print("    " + x)
        try:
            H, H2 = load_rows(m, held)
        except FileNotFoundError as err:
            print("# %s: held-out set not judged yet (%s)" % (m, str(err)[:200]))
            continue
        data[m] = systems_of(H, H2, st, caps, H.ids)
        M[m] = margins_of(m)
        done.append(m)
    if not done:
        return
    ids = [r["id"] for r in rows if r["set"] == "heldout"]
    groups = [("held-out, premise false", [q for q in ids if prem[q] == "false"]),
              ("held-out, premise valid", [q for q in ids if prem[q] == "valid"]), ("held-out, all", ids)]
    readings(data, done, qt, pair_of, groups, M)


# ---- self-judge items --------------------------------------------------------------------------------------------- #
def handed(model, rows_by_name, ids):
    """[(name, id, text)] of every answered answer a system can hand in."""
    def blank(n, q):
        r = rows_by_name[n][q]
        return bool(r["rule_abstain"]) or not r["text"].strip()
    out = []
    for q in ids:
        for n in rows_by_name:
            if blank(n, q):
                continue
            if n.startswith("SEC"):
                if not blank("SE", q):
                    continue
            elif n.startswith("SC"):
                if not blank("S0", q):
                    continue
            elif n.startswith("EC"):
                if not blank("E", q):
                    continue
            elif n.startswith("C"):
                if not blank("O", q):
                    continue
            out.append((n, q, rows_by_name[n][q]["text"]))
    return out


def export_margins(model, phase):
    import numpy as np
    import selfjudge_probe as SP
    import selfjudge_stack_pick as SKP
    SP.SOURCES.update(SRC["smoke"] + SRC["full"])
    src = SRC[phase]
    root = os.path.join(SP.OUT, model)
    if os.path.exists(os.path.join(root, src[0] + "_items.json")):
        raise SystemExit("%s: %s already exported" % (model, src[0]))
    rows = load_bank()
    ids = [r["id"] for r in rows]
    bank = {r["id"]: r for r in rows}
    byname, n = {}, None
    for name, arm in roster(model).items():
        state, rr = file_state(model, arm, ids)
        if phase == "full" and state != "full" or phase == "smoke" and state in ("missing", "full"):
            raise SystemExit("%s: %s is %s, not what export-margins %s reads" % (model, arm, state, phase))
        if n is not None and len(rr) != n:
            raise SystemExit("%s: the files cover different questions" % model)
        n = len(rr)
        byname[name] = {r["id"]: r for r in rr}
    qs = ids[:n]
    have = SKP.lookup(model, extra_sources=(SRC["smoke"][0],) if phase == "full" else ())
    items, seen, tally = [], set(), collections.Counter()
    for name, q, text in handed(model, byname, qs):
        h = statement_sha(bank[q]["question"], text)
        if h in have:
            tally["already scored"] += 1
            continue
        if h in seen:
            tally["same statement again"] += 1
            continue
        seen.add(h)
        tally[name] += 1
        items.append(dict(id=h, split="falseqa %s %s" % (bank[q]["set"], name), question=bank[q]["question"], answer=text,
                          answer_sha1=SP.sha1(text), value=None, answered=True, neg=bool(SP.NEG_RE.match(text))))
    if not items:
        print("%s: nothing to score" % model)
        return
    SP.save_items(os.path.join(root, src[0] + "_items.json"), dict(model=model, source=src[0], items=items))
    k = min(16, len(items))
    pick = sorted(np.random.default_rng(20261122).choice(len(items), k, replace=False).tolist())
    chk = [items[j] for j in pick]
    perm = np.random.default_rng(20261123).permutation(k).tolist()
    for s_, xs in ((src[1], chk), (src[2], chk), (src[3], [chk[j] for j in perm])):
        SP.save_items(os.path.join(root, s_ + "_items.json"), dict(model=model, source=s_, items=xs))
    print("%s %s: %d questions, %d new statements | %s" % (model, phase, len(qs), len(items),
                                                          " | ".join("%s %d" % kv for kv in sorted(tally.items()))))


def check_margins(model, phase):
    import selfjudge_probe as SP
    SP.SOURCES.update(SRC["smoke"] + SRC["full"])
    src = SRC[phase]
    maps = {}
    for s_ in src:
        _, meta, z = SP.read_probe(model, source=s_)
        if meta.get("batch_size") != 1:
            raise SystemExit("%s: %s is not batch 1" % (model, s_))
        maps[s_] = {str(h): float(z["baseline__margin"][j]) for j, h in enumerate(z["statement_sha1"])}
    a, b, c = (maps[s_] for s_ in src[1:])
    if set(a) != set(b) or set(a) != set(c) or not set(a) <= set(maps[src[0]]):
        raise SystemExit("%s: check sets differ" % model)
    d1 = max(abs(a[h] - b[h]) for h in a)
    d2 = max(abs(a[h] - c[h]) for h in a)
    d3 = max(abs(a[h] - maps[src[0]][h]) for h in a)
    v = sorted(maps[src[0]].values())
    print("margins %s %s: %d statements scored (batch 1); check %d statements: second pass max |diff| %.6f, shuffled %.6f, "
          "vs the main pass %.6f -> %s | margin min %.2f median %.2f max %.2f" % (
              model, phase, len(v), len(a), d1, d2, d3, "OK" if d1 == d2 == d3 == 0 else "MISMATCH", v[0], v[len(v) // 2], v[-1]))
    if not d1 == d2 == d3 == 0:
        raise SystemExit(1)


def run_smoke(model, n):
    rows = load_bank()
    ids = [r["id"] for r in rows]
    bank = {r["id"]: r for r in rows}
    for name, arm in roster(model).items():
        state, rr = file_state(model, arm, ids)
        if rr is None:
            print("%-6s %-34s missing" % (name, arm))
            continue
        for r in rr[:n]:
            print("%-6s %-34s %s %s blank=%s fin=%s | %s" % (name, arm, r["id"], bank[r["id"]]["premise"],
                                                            bool(r["rule_abstain"]) or not r["text"].strip(), r["finished"],
                                                            r["text"].replace("\n", " ")[:160]))


# ---- synthetic test ------------------------------------------------------------------------------------------------ #
def fake(model, ids, arms):
    """A FRaw without files: arms = {arm: {q: (value, text, rule_abstain)}}."""
    R = object.__new__(FRaw)
    R.model, R.el, R.cl = model, 16, 19
    R.ids, R.all_ids = list(ids), list(ids)
    R.qtype = {q: "failure" for q in ids}
    R.rows, R.cond, R.val, R.pred = {}, {}, {}, {}
    for arm, d in arms.items():
        c = "cond_" + G.arm_token(arm)
        R.rows[arm] = {q: {"text": t, "rule_abstain": bl, "finished": True} for q, (v, t, bl) in d.items()}
        R.cond[arm] = c
        for q, (v, t, bl) in d.items():
            R.val[(q, c)] = v
            R.pred[(q, c)] = t
    return R


def selftest():
    import selfjudge_stack_pick as SKP
    qs = ["a", "b", "c", "d"]
    B_ = ""   # blank text

    def arm(vals):   # vals: per question (value, text) with text "" = blank
        return {q: (v, t, not t) for q, (v, t) in zip(qs, vals)}
    s, el, cl = "2", 16, 19
    e = "x:E16.w.p2"
    ec = lambda k: "%s+C19.w.m%s" % (e, k)   # noqa: E731
    c_ = lambda k: "x:C19.w.m%s" % k          # noqa: E731
    arms = {"x": arm([(1, "o-a"), (0, B_), (-1, "o-c"), (0, B_)]),            # O: b, d blank (own blanks)
            e: arm([(1, "e-a"), (0, B_), (0, B_), (0, B_)]),                   # E: b, c, d blank; c = "E's blank"
            ec("0.25"): arm([(1, "x"), (0, B_), (1, "o-c"), (-1, "w-d")]),     # c releases the no-write answer -> vetoed
            ec("0.5"): arm([(1, "x"), (1, "r-b"), (1, "r-c"), (-1, "w-d")]),
            ec("1"): arm([(1, "x"), (1, "r-b"), (1, "r-c"), (1, "r-d")]),
            ec("2"): arm([(1, "x"), (1, "r-b"), (1, "r-c"), (1, "r-d")]),
            c_("0.25"): arm([(1, "x"), (0, B_), (0, "x"), (0, B_)]),
            c_("0.5"): arm([(1, "x"), (1, "cr-b"), (0, "x"), (0, B_)]),
            c_("1"): arm([(1, "x"), (1, "cr-b"), (0, "x"), (1, "cr-d")]),
            c_("2"): arm([(1, "x"), (1, "cr-b"), (0, "x"), (1, "cr-d")])}
    R = fake("llama3.1-8b", qs, arms)
    checks = []
    # N1 at caps (E's blanks 0.5, own blanks 0.25): a = E; b own blank: 0.25 blank -> stays blank (E's cell);
    # c E's blank: 0.25 = no-write text -> vetoed (stops, E's cell kept, not answered) ; d own: 0.25 -> w-d (-1)
    n1 = n1_cells(R, s, ("0.5", "0.25"), qs)
    checks.append(("N1 caps 0.5/0.25", [n1[q][:2] + (n1[q][3],) for q in qs], [(1, True, "0"), (0, False, None), (0, False, None), (-1, True, "0.25")]))
    # N1 at caps (none, 1): c not re-asked; b own: 0.25 blank, 0.5 r-b (+1); d own: 0.25 w-d (-1, first answered rung)
    n1b = n1_cells(R, s, (None, "1"), qs)
    checks.append(("N1 caps none/1", [n1b[q][:2] + (n1b[q][3],) for q in qs], [(1, True, "0"), (1, True, "0.5"), (0, False, None), (-1, True, "0.25")]))
    # caps chosen: a +1 always; c (E's blank) is vetoed at its first answered rung under any cap -> 0; b / d own blanks:
    # own cap none 0 + 0, 0.25: 0 - 1, 0.5 / 1 / 2: +1 - 1 -> best net +1 at own none / 0.5 / 1 / 2 with any E cap; the
    # first of these in the grid (E's cap none, own cap none) is chosen
    R2 = fake("llama3.1-8b", qs, {k: v for k, v in arms.items()})
    caps, _ = choose(R, R2, (el, s, cl, s))
    checks.append(("N1 cap choice (ties to the earlier pair)", caps["N1"], (None, None)))
    checks.append(("N1 net at own cap 0.25", sum(R.n1_system(s, None, "0.25")[0][q] for q in qs), 0))
    # Conly: O first; b, d blank; 0.25: both blank -> 1 - 1 = 0; 0.5: b +1 -> +1; 1: b +1, d +1 -> +2; 2 = +2 -> 1 (smaller)
    lad = {k: sum(ladder_cells(R, k, qs)[q][0] for q in qs) for k in LAD}
    checks.append(("Conly nets by cap", lad, {"0.25": 0, "0.5": 1, "1": 2, "2": 2}))

    checks.append(("Conly cap", caps["Conly"], "1"))
    lc = ladder_cells(R, "1", qs)
    checks.append(("Conly cells at 1", [lc[q][:2] + (lc[q][3],) for q in qs], [(1, True, None), (1, True, "0.5"), (-1, True, None), (1, True, "1")]))
    # fill logic: S0 = x of R2 (b, d blank); SE = E (blank b, c, d); TE answers everything with -1
    st = (el, s, cl, s)
    R.rows["conf_E2"] = R.rows["x"]
    R.cond["conf_E2"] = R.cond["x"]
    te = "conf_E2:E16.w.p2"
    R.rows[te] = {q: {"text": "t-" + q, "rule_abstain": False, "finished": True} for q in qs}
    R.cond[te] = "cond_te"
    for q in qs:
        R.val[(q, "cond_te")] = -1
    d = systems_of(R, R2, st, {"N1": ("0.5", "0.25"), "SN1": (None, "1"), "Conly": "1", "SC": "0.25"}, qs)
    checks.append(("fill EC", [d["fill EC"][q][0] for q in qs], [1, 1, -1, -1]))      # a, c = S0; b, d = SN1 (+1, -1)
    checks.append(("fill pE2", [d["fill pE2"][q][0] for q in qs], [1, -1, -1, -1]))   # b, d: SE blank -> TE (-1)
    checks.append(("fill E", [d["fill E"][q][:2] for q in qs], [(1, True), (0, False), (-1, True), (0, False)]))
    checks.append(("fill C = SC at 0.25", [d["fill C"][q][:2] for q in qs], [(1, True), (0, False), (-1, True), (0, False)]))
    checks.append(("SN1 levels", [d["SN1"][q][3] for q in qs], ["0", "0.5", None, "0.25"]))
    # equal trimming: A answers 3 in the pool with margins 0.5 / -1 / 2, B answers 1 -> A keeps only the margin-2 answer
    M = {}
    qt = {q: "Q " + q for q in qs}
    A = {"a": (1, True, "p", None), "b": (-1, True, "q", None), "c": (1, True, "r", None), "d": (0, False, "", None)}
    Bc = {"a": (1, True, "p", None), "b": (0, False, "", None), "c": (0, False, "", None), "d": (-1, True, "s", None)}
    for q, mv in (("a", 0.5), ("b", -1.0), ("c", 2.0), ("d", 0.1)):
        M[statement_sha(qt[q], A[q][2] if A[q][1] else Bc[q][2])] = {"baseline": mv}
    dd, cnt, miss = equal_diff(qs, ["b", "c", "d"], A, Bc, qt, M)
    # pool b, c, d: A answers b (-1, margin -1), c (+1, margin 2); B answers d. A drops b -> A: b 0, c +1; diffs: a 0, b 0, c +1, d +1
    checks.append(("equal trimming in the pool (a outside kept)", ([dd[q] for q in qs], cnt, miss), ([0, 0, 1, 1], (2, 1), 0)))
    Mmiss = {k: v for k, v in M.items() if v["baseline"] != 2.0}
    dm, _, mm = equal_diff(qs, ["b", "c", "d"], A, Bc, qt, Mmiss)
    checks.append(("a missing margin on the bigger side -> n/a", (dm, mm), (None, 1)))
    # SKP.equal itself on the same input (the function the table uses)
    va, vb, _ = SKP.equal(["b", "c", "d"], {q: A[q][:2] + (margin(M, qt[q], A[q]),) for q in "bcd"},
                          {q: Bc[q][:2] + (margin(M, qt[q], Bc[q]),) for q in "bcd"})
    checks.append(("SKP.equal drops the lowest margin", [va[q][:2] for q in "bcd"], [(0, False), (1, True), (0, False)]))
    # pair resampling: pairs (a, b) and (c, d); per-question diffs +1 / -1 inside each pair -> every resample sums to 0
    pair_of = {"a": "p1", "b": "p1", "c": "p2", "d": "p2"}
    checks.append(("pair resampling, cancelling pairs", tuple(pair_ci({"a": 1, "b": -1, "c": 2, "d": -2}, pair_of, qs)), (0, 0, 0)))
    t, lo, hi = S.joint_ci([[1, -1, 2, -2]])
    checks.append(("question resampling of the same values is wider", lo < 0 < hi, True))
    t2 = pair_ci({"a": 1, "b": 1, "c": 0, "d": 1}, pair_of, qs)
    checks.append(("pair sums: total and bounds within [min, max] of n * pair sum", (t2[0], 2 <= t2[1] <= t2[2] <= 4), (3, True)))
    bad = 0
    for name, got, want in checks:
        ok = got == want
        bad += not ok
        print("selftest %-52s %s%s" % (name, "OK" if ok else "FAILED", "" if ok else " got %r want %r" % (got, want)))
    if bad:
        raise SystemExit("selftest FAILED (%d)" % bad)
    print("selftest PASS (%d checks)" % len(checks))


def main():
    a = sys.argv[1:]
    mode = a[0] if a else ""
    if mode == "check":
        if not run_check(a[1:], bool(a[1:])):
            raise SystemExit(1)
    elif mode == "status":
        run_status(a[1:] or ORDER)
    elif mode == "export-margins" and len(a) == 3 and a[1] in ORDER and a[2] in SRC:
        export_margins(a[1], a[2])
    elif mode == "check-margins" and len(a) == 3 and a[1] in ORDER and a[2] in SRC:
        check_margins(a[1], a[2])
    elif mode == "smoke" and len(a) in (2, 3) and a[1] in ORDER:
        run_smoke(a[1], int(a[2]) if len(a) == 3 else 3)
    elif mode == "table":
        run_table(a[1:] or ORDER)
    elif mode == "selftest":
        selftest()
    else:
        raise SystemExit(__doc__ if __doc__ else "usage: falseqa_read.py check|status|export-margins|check-margins|smoke|table|selftest")


if __name__ == "__main__":
    main()
