# Iteration_01 (Claude Opus 5.5, executor ren, 2026-11-24; plan discussion/work_queue/2026-11-24_queue_n.md sections 2
# and 3): where along the depth do the E write and the C re-ask work best? Read-only, no model, no judge call.
# Stage 1 (plan 2): single E at layer L, written at the core rule "prompt" (question tokens and assistant header; the
# rule of the current main table), plain prompt, tune split (192 questions), DeepSeek Pro. Per layer two strengths
# chosen on MMLU (1,000-item list, option probability, cap_mcq_sweep.py) by plan 2.1:
#   s1  = the largest strength before the first strength (from small to large) whose accuracy drop exceeds 1.0 point;
#   s05 = the same with 0.5 point. Ladder 0.5 0.75 1 1.5 2 2.5 3 3.5 4 5 6 8; a layer where 8 is still inside the
#   1.0 line gets 12 and 16 as well and the rule is applied again; a layer whose smallest strength is already outside
#   the line has "no strength inside the budget". Drops are compared in whole items (1.0 point = 10 of 1,000 items).
# Stage 2 (plan 3): E fixed at the main-table layer and strength, C at the current layer and at the new layers after
# E; N1 with its two caps chosen on tune by the chooser of the current main table (main_table_layers.n1_pick, through
# main_table_prompt's way of reading the prompt-rule work dirs).
# Scoring: right +1, no answer 0, wrong -1 from the Pro verdicts (policy_select2.load); answered = not blank (rule
# detector or empty), as final_records.answered; no-write cells come from the usual E+C dir of the model and must be
# identical in every new work dir (state, blank flag, text), as main_table_prompt.prompt_split checks. Paired bootstrap
# of the summed per-question differences: e44_round1_tables.boot (10,000 resamples, seed 20261111).
# Files (host paths, relative to /data/sanshizhao/Truth_Decoupling):
#   MMLU  Iteration_01/capability/depthscan/<model>_x.json (arm x), <model>_L<l>.json (12 strengths), <model>_L<l>_ext.json
#         (12 and 16), choices <model>_choice.json (written by `choose`)
#   TQA   <E run>/eval_extra/judge_tune_depthE_tune (stage 1), judge_tune_depthC_tune (stage 2, two-write conditions
#         with the suffix "-at-prompt"), next to the existing judge_tune_posprompt<tag>_tune (single E at the main-table
#         layer) and judge_tune_posprompt2<tag>_tune (E + current C)
# modes (host, from /data/sanshizhao/Truth_Decoupling; KEY = llama | q9):
#   python3 -B Iteration_01/diag/depth_scan.py checkx KEY        entry check 1: MMLU accuracy of x = stored reference
#   python3 -B Iteration_01/diag/depth_scan.py checkgen KEY      entry check 2: partial regeneration (t16 file) = archive
#   python3 -B Iteration_01/diag/depth_scan.py choose KEY        apply the rule of plan 2.1, write the choice file
#   python3 -B Iteration_01/diag/depth_scan.py need-ext KEY      layers that still need strengths 12 and 16
#   python3 -B Iteration_01/diag/depth_scan.py arms KEY          stage-1 generation arms (choice must be complete)
#   python3 -B Iteration_01/diag/depth_scan.py c-arms KEY        stage-2 generation arms
#   python3 -B Iteration_01/diag/depth_scan.py checkpos KEY E|C  every generation file of the stage: complete, 192 rows,
#                                                                positions "prompt"
#   python3 -B Iteration_01/diag/depth_scan.py plan KEY          dry listing of arms and file names
#   python3 -B Iteration_01/diag/depth_scan.py table KEY         stage-1 table (plan 2.2)
#   python3 -B Iteration_01/diag/depth_scan.py c KEY             stage-2 table (plan 3)
#   python3 -B Iteration_01/diag/depth_scan.py c-ref KEY         only the reproduction of N1's archived tune score at the
#                                                                current C layer (existing work dirs; run before the queue)
#   python3 -B Iteration_01/diag/depth_scan.py selftest          synthetic check of the rule and the tables
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/review_checks")

LADDER = ["0.5", "0.75", "1", "1.5", "2", "2.5", "3", "3.5", "4", "5", "6", "8"]
EXT = ["12", "16"]
LINES = (1.0, 0.5)
CAPD = "Iteration_01/capability/depthscan"
SUF = "-at-prompt"
NAME_E, NAME_C = "depthE_tune", "depthC_tune"
KEYS = {
    # first pass of 2026-11-24 (main session): fewer layers than plan 2.1 names, dense around the middle; the lists
    # must equal LAYERS in depth_queue2_1124.sh (llama 8-27 and q9 10-24 in full would hold card 1 for 8-12 hours)
    "llama": dict(name="Llama-3.1-8B", ng="llama3.1-8b", nl=32, layers=[10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 26], el=16, es="3.5", cl=19,
                  newc=[17, 18, 20, 21], run="results_e44_factorial_das/llama3.1-8b_side_E2", ref_x=640, tag="l16",
                  ref_n1_tune=27, window=(14, 18)),
    "q9": dict(name="Qwen3.5-9B", ng="qwen3.5-9b", nl=32, layers=[12, 13, 14, 15, 16, 17, 18, 20, 22, 24], el=15, es="2", cl=16,
               newc=[17, 18, 19], run="results_newmodels_ff_20261112/e44/qwen3.5-9b_E2", ref_x=776, tag="l15",
               ref_n1_tune=63, window=(13, 17)),
}


def hy(s):
    return s.replace(".", "-")


def arm_e(layer, s):
    return "x:E%d.w.p%s" % (layer, s)


def arm_ec(el, es, cl, k):
    return "x:E%d.w.p%s+C%d.w.m%s" % (el, es, cl, k)


def gen_file(arm, split="tune", partial=0):
    f = arm.replace(":", "--").replace("+", "-and-").replace(".", "-")
    return ("t%d." % partial if partial else "") + "%s%s_%s.json" % (f, SUF, split)


def cond_e(layer, s):
    return "E_L%d_x-w-p%s" % (layer, hy(s))


def mm_file(ng, what):
    return "%s/%s_%s.json" % (CAPD, ng, what)


def die(msg):
    raise SystemExit("depth_scan: " + msg)


# ----------------------------------------------------------------------------- MMLU and the rule of plan 2.1
def read_mm(path, want_arms, ref=None):
    """{arm: correct items}, n, (items sha, ids sha, identity sha) of one sweep file; every wanted arm present, none
    skipped, the same item list and identity as ref."""
    if not os.path.exists(path):
        die("no MMLU file %s" % path)
    d = json.load(open(path, encoding="utf-8"))
    if d.get("skipped"):
        die("%s skipped arms %s" % (path, sorted(d["skipped"])))
    ident = (d["items_file_sha256"], d["ids_file_sha256"], d["identity_sha256"])
    if ref is not None and ident != ref:
        die("%s: item list or identity differs from the x file" % path)
    out = {}
    for a in want_arms:
        if a not in d["arms"]:
            die("%s lacks arm %s" % (path, a))
        r = d["arms"][a]
        if len(r["rows"]) != d["n"]:
            die("%s arm %s: %d rows for n %d" % (path, a, len(r["rows"]), d["n"]))
        out[a] = int(round(r["accuracy"] * d["n"]))
    return out, d["n"], ident


def first_out(seq, cx, n, line):
    """seq = [(strength, correct)] from small to large. (s, first): s = largest strength before the first one whose drop
    exceeds `line` points (None: the smallest is already outside), first = that strength (None: all inside)."""
    prev = None
    for s, c in seq:
        if 100 * (cx - c) > line * n:
            return prev, s
        prev = s
    return prev, None


def choose_layer(acc, cx, n):
    """acc {strength: correct} (the ladder, plus 12 / 16 when present) -> choice record of one layer."""
    seq = [(s, acc[s]) for s in LADDER + EXT if s in acc]
    has_ext = all(s in acc for s in EXT)
    rec = {"acc": {s: c for s, c in seq}, "ext": has_ext}
    for line, key in ((1.0, "s1"), (0.5, "s05")):
        s, fo = first_out(seq, cx, n, line)
        rec[key], rec[key + "_first_out"] = s, fo
    rec["need_ext"] = rec["s1_first_out"] is None and not has_ext
    if rec["need_ext"]:
        rec["s1"] = rec["s05"] = None            # not final before 12 / 16 are measured
    return rec


def choose(key):
    k = KEYS[key]
    xs, n, ident = read_mm(mm_file(k["ng"], "x"), ["x"])
    cx = xs["x"]
    out = {"model": k["ng"], "n": n, "x_correct": cx, "ladder": LADDER, "ext_ladder": EXT, "lines": list(LINES),
           "rule": "plan 2.1: largest strength before the first (small to large) whose drop exceeds the line; drop "
                   "compared in items (100 * (x - arm) > line * n)", "layers": {}}
    for layer in k["layers"]:
        arms = [arm_e(layer, s) for s in LADDER]
        a, _, _ = read_mm(mm_file(k["ng"], "L%d" % layer), arms, ident)
        acc = {s: a[arm_e(layer, s)] for s in LADDER}
        ef = mm_file(k["ng"], "L%d_ext" % layer)
        if os.path.exists(ef):
            b, _, _ = read_mm(ef, [arm_e(layer, s) for s in EXT], ident)
            acc.update({s: b[arm_e(layer, s)] for s in EXT})
        out["layers"][str(layer)] = choose_layer(acc, cx, n)
    out["need_ext"] = [int(l) for l, r in out["layers"].items() if r["need_ext"]]
    out["complete"] = not out["need_ext"]
    return out


def load_choice(key, complete=True):
    p = mm_file(KEYS[key]["ng"], "choice")
    if not os.path.exists(p):
        die("no choice file %s (run choose)" % p)
    c = json.load(open(p, encoding="utf-8"))
    if complete and not c["complete"]:
        die("%s is not complete: layers %s still need 12 and 16" % (p, c["need_ext"]))
    return c


def stage1_arms(choice):
    """[(layer, line, strength or None)] and the distinct arms, in layer order."""
    rows, arms = [], []
    for l in sorted(choice["layers"], key=int):
        r = choice["layers"][l]
        for line, kk in ((1.0, "s1"), (0.5, "s05")):
            rows.append((int(l), line, r[kk]))
            if r[kk] is not None and arm_e(int(l), r[kk]) not in arms:
                arms.append(arm_e(int(l), r[kk]))
    return rows, arms


def stage2_arms(k):
    return [arm_ec(k["el"], k["es"], cl, m) for cl in k["newc"] for m in ("0.25", "0.5", "1", "2")]


# ----------------------------------------------------------------------------- generation files
def read_gen(path):
    if not os.path.exists(path):
        die("no generation file %s" % path)
    return json.load(open(path, encoding="utf-8"))


def check_gen_file(path, n=192, partial=False):
    d = read_gen(path)
    pos, cpos = d.get("positions"), d.get("generation_contract", {}).get("positions")
    bad = []
    if pos != "prompt" or cpos != "prompt":
        bad.append("positions %s / contract %s" % (pos, cpos))
    if bool(d.get("partial")) != partial or (not partial and not d.get("completed")):
        bad.append("partial %s completed %s" % (d.get("partial"), d.get("completed")))
    if len(d["rows"]) != n or d.get("split") != "tune":
        bad.append("rows %d split %s" % (len(d["rows"]), d.get("split")))
    return d, bad


def checkgen(key):
    k = KEYS[key]
    arm = arm_e(k["el"], k["es"])
    g = "%s/eval_extra/gen/" % k["run"]
    full, b1 = check_gen_file(g + gen_file(arm))
    part, b2 = check_gen_file(g + gen_file(arm, partial=16), n=16, partial=True)
    if b1 or b2:
        die("entry check 2 (%s): archive %s / regeneration %s" % (arm, b1 or "ok", b2 or "ok"))
    same = 0
    for a, b in zip(full["rows"][:16], part["rows"]):
        if (a["pair_id"], a["text"], a["token_ids"]) == (b["pair_id"], b["text"], b["token_ids"]):
            same += 1
        else:
            print("  differs: %s | archive %r | now %r" % (a["pair_id"], a["text"][:80], b["text"][:80]))
    print("entry check 2 %s %s: %d of 16 answers identical to the archived prompt-rule generation (positions prompt)" % (
        k["ng"], arm, same))
    return 0 if same == 16 else 1


def checkpos(key, stage):
    k = KEYS[key]
    arms = stage1_arms(load_choice(key))[1] if stage == "E" else stage2_arms(k)
    bad = []
    for a in arms:
        _, b = check_gen_file("%s/eval_extra/gen/%s" % (k["run"], gen_file(a)))
        if b:
            bad.append("%s: %s" % (a, "; ".join(b)))
    print("stage %s %s: %d generation files checked (complete, 192 rows, positions prompt in record and contract); bad %d" % (
        stage, k["ng"], len(arms), len(bad)))
    for x in bad:
        print("  " + x)
    return 1 if bad else 0


# ----------------------------------------------------------------------------- verdicts
def boot(d):
    import e44_round1_tables as T
    lo, hi = T.boot(d)
    return "%+d [%+d, %+d]" % (sum(d), lo, hi)


def val(x):
    return (x == "R") - (x == "W")


def work(k, name):
    return "%s/eval_extra/judge_tune_%s" % (k["run"], name)


def usual(k):
    """(no-write states, blanks, texts, usual question list) of the model's tune split."""
    import main_table_same_n as MT
    import final_records as fr
    D = MT.Data(k["name"], 0)
    utx = fr.raw_texts(D.m["ec"][0])
    st, bl, tx = {}, {}, {}
    for q in D.sp.qs:
        kk = (q, "baseline_x")
        st[kk], bl[kk], tx[kk] = D.sp.st[kk], D.sp.blank[kk], utx[kk]
    return st, bl, tx, list(D.sp.qs)


def add_dir(w, st, bl, tx, two, owner):
    """Merge the Pro verdicts of work dir w into st / bl / tx (main_table_prompt.prompt_split's checks)."""
    import policy_select2 as ps
    import final_records as fr
    if not os.path.exists(w + "/judge2/single_pro.jsonl"):
        die("no Pro verdicts yet in %s" % w)
    st1, bl1 = ps.load(w, "pro")
    t1 = fr.raw_texts(w)
    for (q, c), v in st1.items():
        if c == "baseline_x":
            if (q, c) in st and (v != st[(q, c)] or bl1[(q, c)] != bl[(q, c)] or t1.get((q, c)) != tx[(q, c)]):
                die("the no-write cell of %s differs from the usual one (%s)" % (os.path.basename(w), q))
            continue
        c0 = c
        if two:
            if not c.endswith(SUF):
                die("condition %s in %s lacks %s" % (c, os.path.basename(w), SUF))
            c = c[:-len(SUF)]
        elif c.endswith(SUF):
            die("a two-write condition %s in a single-write work dir" % c)
        if (q, c) in st and owner.get(c) != w:
            die("condition %s appears in two work dirs (%s, %s)" % (c, owner.get(c), w))
        owner[c] = w
        st[(q, c)], bl[(q, c)], tx[(q, c)] = v, bl1[(q, c0)], t1[(q, c0)]


def split_order(k, arms):
    """Tune question order of the generation files (all must agree)."""
    order = None
    for a in arms:
        ids = read_gen("%s/eval_extra/gen/%s" % (k["run"], gen_file(a)))["question_ids"]
        if order is None:
            order = ids
        elif ids != order:
            die("generation files disagree on the question order (%s)" % a)
    return order


def counts(states, ans):
    r = sum(x == "R" for x in states)
    w = sum(x == "W" for x in states)
    return sum(ans), r, w, r - w


def common_qs(st, conds, qs):
    return [q for q in qs if all(st.get((q, c), "?") != "?" for c in ["baseline_x"] + conds)]


def single_e(st, bl, tx, layer, s, qs):
    """(states, answered) of single E at layer / strength (hyphen form): the E cell of every question, answered = not
    blank -- what final_records.records(("E", s)) hands in (its E record is the E cell for every question; it only
    looks up the C ladder for a note, and single E has no C cells here)."""
    c = "E_L%d_x-w-p%s" % (layer, s)
    miss = [q for q in qs if st.get((q, c), "?") == "?"]
    if miss:
        die("layer %d strength %s: %d questions without a verdict" % (layer, s, len(miss)))
    return [st[(q, c)] for q in qs], [not bl[(q, c)] for q in qs]


def stage1_rows(choice, st, bl, tx, qs, n_mm):
    """[(layer, line, s, mmlu change, states, answered)] and the no-write (states, answered) on qs."""
    rows = []
    base = ([st[(q, "baseline_x")] for q in qs], [not bl[(q, "baseline_x")] for q in qs])
    cx = choice["x_correct"]
    for l in sorted(choice["layers"], key=int):
        r = choice["layers"][l]
        for line, kk in ((1.0, "s1"), (0.5, "s05")):
            s = r[kk]
            if s is None:
                rows.append((int(l), line, None, None, None, None))
                continue
            es, ea = single_e(st, bl, tx, int(l), hy(s), qs)
            rows.append((int(l), line, s, 100.0 * (r["acc"][s] - cx) / n_mm, es, ea))
    return rows, base


def print_stage1(k, rows, base, qs, label):
    nl = k["nl"]
    mid = nl // 2
    b_ans, b_r, b_w, b_net = counts(base[0], base[1])
    print("\n## %s, %s: %d questions; no write: answers %d | right %d wrong %d | net %+d" % (
        k["name"], label, len(qs), b_ans, b_r, b_w, b_net))
    print("| layer | position (l+1)/%d | line | strength | MMLU change (points, 1,000 items) | answers | right | wrong | net | net minus no write [95%%] | flag |" % nl)
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    by = {}
    for l, line, s, chg, es, ea in rows:
        flag = " ".join(x for x in ("main-table layer" if l == k["el"] else "", "middle layer (%d / 2)" % nl if l == mid else "") if x)
        if s is None:
            print("| %d | %.3f | %.1f | none | no strength inside the budget | | | | | | %s |" % (l, (l + 1.0) / nl, line, flag))
            continue
        a, r, w, net = counts(es, ea)
        print("| %d | %.3f | %.1f | %s | %+.1f | %d | %d | %d | %+d | %s | %s |" % (
            l, (l + 1.0) / nl, line, s, chg, a, r, w, net, boot([val(x) - val(y) for x, y in zip(es, base[0])]), flag))
        by[(l, line)] = (s, es, net)
    one = [(l, v) for (l, line), v in by.items() if line == 1.0]
    if not one:
        print("    no layer has a strength inside the 1.0 line")
        return
    lstar, (sstar, ststar, netstar) = max(one, key=lambda x: (x[1][2], -x[0]))
    gstar = netstar - b_net
    print("\n  best layer at the 1.0 line: L* = %d (strength %s, net %+d, gain over no write %+d; ties go to the lower layer)" % (
        lstar, sstar, netstar, gstar))
    print("| layer | position | strength | net | layer minus L* [95%] | gain over no write | flag |")
    print("|---|---|---|---|---|---|---|")
    for l, (s, es, net) in sorted(one):
        flag = " ".join(x for x in ("L*" if l == lstar else "", "main-table layer" if l == k["el"] else "",
                                    "middle layer" if l == mid else "") if x)
        print("| %d | %.3f | %s | %+d | %s | %+d | %s |" % (l, (l + 1.0) / nl, s, net,
                                                         boot([val(x) - val(y) for x, y in zip(es, ststar)]), net - b_net, flag))
    # the numbers the predictions of plan 2.3 are about (no verdict written here)
    print("\n  plan 2.3 quantities (%s):" % label)
    print("    L* = %d (position %.3f); predicted window for %s: layers %d-%d" % (lstar, (lstar + 1.0) / nl, k["name"], k["window"][0], k["window"][1]))
    if k["ng"].startswith("llama"):
        late = [(l, v[2] - b_net) for l, v in sorted(one) if l >= 21]
        print("    gain over no write of layers >= 21 at the 1.0 line: %s; half of L*'s gain = %.1f" % (
            ", ".join("L%d %+d" % x for x in late) or "none inside the budget", gstar / 2.0))
    if (mid, 1.0) in by:
        print("    middle layer %d minus L*: %+d points (net %+d against %+d)" % (mid, by[(mid, 1.0)][2] - netstar, by[(mid, 1.0)][2], netstar))
    else:
        print("    middle layer %d has no strength inside the 1.0 line" % mid)


def table(key):
    k = KEYS[key]
    choice = load_choice(key)
    rows0, arms = stage1_arms(choice)
    st, bl, tx, qs0 = usual(k)
    add_dir(work(k, NAME_E), st, bl, tx, False, {})
    conds = [cond_e(l, s) for l, _, s in rows0 if s is not None]
    miss = sorted({c for c in conds if not any(cc == c for _, cc in st)})
    if miss:
        die("conditions without cells in %s: %s" % (NAME_E, miss))
    qs = common_qs(st, conds, qs0)
    order = split_order(k, arms)
    print("# Plan N stage 1, %s: single E per layer at the prompt rule, tune split, DeepSeek Pro (depth_scan.py table)" % k["name"])
    print("MMLU 1,000 items: x correct %d of %d; rule of plan 2.1; score right +1 / no answer 0 / wrong -1; paired bootstrap "
          "10,000 resamples, seed 20261111. Questions: %d usual, %d with every verdict%s." % (
              choice["x_correct"], choice["n"], len(qs0), len(qs), "" if len(qs) == len(qs0) else " (the others are left out of every row)"))
    print("Layers numbered from 0; position = (layer + 1) / %d. Extended to 12 / 16: %s." % (
        k["nl"], ", ".join(l for l in sorted(choice["layers"], key=int) if choice["layers"][l]["ext"]) or "none"))
    for label, qq in (("tune, all questions", qs), ("tune, first 100 questions of the split order", [q for q in order[:100] if q in set(qs)])):
        rows, base = stage1_rows(choice, st, bl, tx, qq, choice["n"])
        print_stage1(k, rows, base, qq, label)
    return 0


# ----------------------------------------------------------------------------- stage 2
def c_table(key, ref_only=False):
    import main_table_layers as ML
    import newsets_score as S
    import final_records as fr
    import policy_select2 as ps
    import release_origin as ro
    k = KEYS[key]
    st, bl, tx, qs0 = usual(k)
    owner = {}
    add_dir(work(k, "posprompt%s_tune" % k["tag"]), st, bl, tx, False, owner)
    add_dir(work(k, "posprompt2%s_tune" % k["tag"]), st, bl, tx, True, owner)
    if not ref_only:
        add_dir(work(k, NAME_C), st, bl, tx, True, owner)
    s = hy(k["es"])
    txs = fr.Texts(tx, tx)
    cls = [k["cl"]] + k["newc"]

    def need(cl):
        sh = S.shell(k["el"], cl)
        return ["baseline_x", sh.cE(s)] + [sh.cEC(s, kk) for kk in ps.LAD]

    # reproduction of the archived main-table tune score of N1 at the current C layer (same chooser, same questions)
    sp0 = S.NewSplit(k["el"], k["cl"], st, bl, need(k["cl"]), qs=qs0)
    p0 = ML.n1_pick(sp0, txs, s)
    n0 = ps.score(S.system(sp0, txs, p0)[0], 1)
    if n0 != k["ref_n1_tune"]:
        die("N1 at the current C layer %d scores %+d on tune, the archived main table says %+d: reader or data differ" % (
            k["cl"], n0, k["ref_n1_tune"]))
    if ref_only:
        print("c-ref %s: N1 at E%d / %s with C%d on the %d usual tune questions = %+d (%s), the archived main table %+d" % (
            k["ng"], k["el"], k["es"], k["cl"], len(sp0.qs), n0, ro.name_of(p0), k["ref_n1_tune"]))
        return 0
    allneed = sorted({c for cl in cls for c in need(cl)})
    qs = common_qs(st, allneed, qs0)
    print("# Plan N stage 2, %s: C after E. E at layer %d, strength %s, prompt rule; N1's two caps chosen on tune by "
          "main_table_layers.n1_pick for each C layer (depth_scan.py c)" % (k["name"], k["el"], k["es"]))
    print("Check: N1 at the current C layer %d on the %d usual tune questions = %+d (archived main table %+d). Questions "
          "read: %d (every C layer has every verdict). Every C layer is chosen and read on the same tune questions: "
          "comparable with each other, optimistic in level. Paired bootstrap 10,000 resamples, seed 20261111.\n" % (
              k["cl"], len(qs0), n0, k["ref_n1_tune"], len(qs)))
    res = {}
    for cl in cls:
        sp = S.NewSplit(k["el"], cl, st, bl, need(cl), qs=qs)
        pick = ML.n1_pick(sp, txs, s)
        states, ans, recs = S.system(sp, txs, pick)
        rel = [r for r in recs if r["note"] == "released"]
        e = S.system(sp, txs, ("E", s))
        res[cl] = (pick, states, ans, rel, e)
    cur = res[k["cl"]][1]
    e_ans, e_r, e_w, e_net = counts(res[k["cl"]][4][0], res[k["cl"]][4][1])
    print("single E (layer %d, %s): answers %d | right %d wrong %d | net %+d\n" % (k["el"], k["es"], e_ans, e_r, e_w, e_net))
    print("| C layer | position (l+1)/%d | C minus E | N1 caps (chosen on tune) | answers | right | wrong | net | released right / wrong | minus current C layer %d [95%%] |" % (k["nl"], k["cl"]))
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cl in sorted(cls):
        pick, states, ans, rel, e = res[cl]
        a, r, w, net = counts(states, ans)
        rr = sum(x["state"] == "R" for x in rel)
        rw = sum(x["state"] == "W" for x in rel)
        d = "current" if cl == k["cl"] else boot([val(x) - val(y) for x, y in zip(states, cur)])
        print("| %d | %.3f | %+d | %s | %d | %d | %d | %+d | %d / %d | %s |" % (
            cl, (cl + 1.0) / k["nl"], cl - k["el"], ro.name_of(pick), a, r, w, net, rr, rw, d))
    return 0


# ----------------------------------------------------------------------------- listing
def plan(key):
    k = KEYS[key]
    print("%s (%s): E run %s" % (k["name"], k["ng"], k["run"]))
    print("  MMLU x: %s (reference %d of 1,000)" % (mm_file(k["ng"], "x"), k["ref_x"]))
    print("  MMLU layers %d-%d, one file each (%d arms): %s ... %s" % (k["layers"][0], k["layers"][-1], len(LADDER),
                                                                       mm_file(k["ng"], "L%d" % k["layers"][0]), mm_file(k["ng"], "L%d" % k["layers"][-1])))
    print("    arms of layer %d: %s" % (k["layers"][0], " ".join(arm_e(k["layers"][0], s) for s in LADDER)))
    print("    total %d arms + x; extension per layer (if needed): %s" % (len(LADDER) * len(k["layers"]), " ".join(arm_e(k["layers"][0], s) for s in EXT)))
    a = arm_e(k["el"], k["es"])
    print("  entry check 2: %s, first 16 tune questions -> gen/%s against gen/%s" % (a, gen_file(a, partial=16), gen_file(a)))
    print("  stage 1 generation: at most %d arms (two per layer), files gen/x--E<l>-w-p<s>%s_tune.json, export %s, work %s" % (
        2 * len(k["layers"]), SUF, NAME_E, work(k, NAME_E)))
    arms = stage2_arms(k)
    print("  stage 2 (C layers %s; selfcheck per pair x:E%d.w.p%s+C<l>.w.m1): %d arms, export %s, work %s" % (
        " ".join(map(str, k["newc"])), k["el"], k["es"], len(arms), NAME_C, work(k, NAME_C)))
    for x in arms:
        print("    %s -> %s" % (x, gen_file(x)))
    return 0


# ----------------------------------------------------------------------------- self-test (synthetic data, no file)
def selftest():
    ok = True

    def eq(what, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print("  %-62s %s%s" % (what, "ok" if good else "FAILED", "" if good else " (got %r, want %r)" % (got, want)))

    print("self-test of depth_scan.py")
    n, cx = 1000, 640
    # drops in items: 0,2,5,6,9,10,11,... -> 0.5 line (5 items) first exceeded at 6 items, 1.0 line (10) at 11
    acc = dict(zip(LADDER, [cx - d for d in (0, 2, 5, 6, 9, 10, 11, 20, 30, 40, 50, 60)]))
    r = choose_layer(acc, cx, n)
    eq("normal layer: s1 / first out", (r["s1"], r["s1_first_out"]), ("2.5", "3"))
    eq("normal layer: s05 / first out", (r["s05"], r["s05_first_out"]), ("1", "1.5"))
    eq("normal layer: no extension", r["need_ext"], False)
    acc = dict(zip(LADDER, [cx - 12] + [cx - 30] * 11))
    r = choose_layer(acc, cx, n)
    eq("smallest strength already outside both lines", (r["s1"], r["s05"]), (None, None))
    acc = dict(zip(LADDER, [cx + 3] * 6 + [cx - 4] * 6))
    r = choose_layer(acc, cx, n)
    eq("8 inside the 1.0 line: extension needed, no choice yet", (r["need_ext"], r["s1"], r["s05"]), (True, None, None))
    acc.update({"12": cx - 10, "16": cx - 25})
    r = choose_layer(acc, cx, n)
    eq("after 12 / 16: s1 = 12 (drop 10 items is inside), s05 = 8", (r["need_ext"], r["s1"], r["s05"]), (False, "12", "8"))
    acc["16"] = cx - 10
    r = choose_layer(acc, cx, n)
    eq("still inside at 16: s1 = 16, no first-out strength", (r["s1"], r["s1_first_out"]), ("16", None))
    acc = dict(zip(LADDER, [cx + 5, cx - 11] + [cx] * 10))
    r = choose_layer(acc, cx, n)
    eq("rule stops at the FIRST exceedance (later recovery ignored)", (r["s1"], r["s05"]), ("0.5", "0.5"))
    eq("arm and file names", (arm_e(16, "3.5"), gen_file(arm_e(16, "3.5")), gen_file(arm_e(8, "0.75"), partial=16)),
       ("x:E16.w.p3.5", "x--E16-w-p3-5-at-prompt_tune.json", "t16.x--E8-w-p0-75-at-prompt_tune.json"))
    eq("two-write file name", gen_file(arm_ec(16, "3.5", 17, "0.25")), "x--E16-w-p3-5-and-C17-w-m0-25-at-prompt_tune.json")
    eq("condition name", cond_e(16, "0.75"), "E_L16_x-w-p0-75")
    # verdict side: 6 questions, two layers; E blanks are "X"
    qs = ["q1", "q2", "q3", "q4", "q5", "q6"]
    st, bl, tx = {}, {}, {}

    def put(q, c, s, text):
        st[(q, c)], bl[(q, c)], tx[(q, c)] = s, s == "X", text

    for q, s in zip(qs, "RWWXWR"):
        put(q, "baseline_x", s, "base " + q)
    for q, s in zip(qs, "RXRXXR"):          # L10 at 1: right 3, wrong 0, answers 3 -> net +3; base net 2-3 = -1
        put(q, cond_e(10, "1"), s, "e10 " + q)
    for q, s in zip(qs, "WWRXWR"):          # L12 at 2: right 2, wrong 3, answers 5 -> net -1
        put(q, cond_e(12, "2"), s, "e12 " + q)
    choice = {"x_correct": 640, "n": 1000, "layers": {
        "10": {"s1": "1", "s05": "1", "acc": {"1": 635}, "ext": False},
        "12": {"s1": "2", "s05": None, "acc": {"2": 631}, "ext": False}}}
    rows, base = stage1_rows(choice, st, bl, tx, qs, 1000)
    eq("no-write counts (answers, right, wrong, net)", counts(base[0], base[1]), (5, 2, 3, -1))
    got = [(l, line, s, round(chg, 1) if chg is not None else None, counts(es, ea) if es else None) for l, line, s, chg, es, ea in rows]
    eq("stage-1 rows", got, [(10, 1.0, "1", -0.5, (3, 3, 0, 3)), (10, 0.5, "1", -0.5, (3, 3, 0, 3)),
                             (12, 1.0, "2", -0.9, (5, 2, 3, -1)), (12, 0.5, None, None, None)])
    eq("paired interval string (difference L10 minus no write = +4)", boot([val(x) - val(y) for x, y in zip(rows[0][4], base[0])]).split(" ")[0], "+4")
    k = dict(KEYS["llama"], name="synthetic", ng="llama-synthetic", el=12, cl=19)
    print("  printed stage-1 table of the synthetic case:")
    print_stage1(k, rows, base, qs, "synthetic")
    # N1 through the main-table chooser: E at layer 10 strength 1, C layer 19, the ladder 0.25 .. 2
    import newsets_score as S
    import main_table_layers as ML
    import policy_select2 as ps
    import final_records as fr
    sh = S.shell(10, 19)
    for kk in ps.LAD:                        # q2 (E's blank: base W) released right at 0.25; q4 (own blank) right at 0.5;
        c = sh.cEC("1", kk)                  # q5 (E's blank, base W) released with the ORIGINAL answer -> vetoed
        for q in qs:
            if q == "q2":
                put(q, c, "R", "c q2")
            elif q == "q4":
                put(q, c, "X" if kk == "0-25" else "R", "c q4")
            elif q == "q5":
                put(q, c, "W", "base q5")
            else:
                put(q, c, st[(q, cond_e(10, "1"))], tx[(q, cond_e(10, "1"))])
    sp = S.NewSplit(10, 19, st, bl, ["baseline_x", sh.cE("1")] + [sh.cEC("1", kk) for kk in ps.LAD], qs=qs)
    txs = fr.Texts(tx, tx)
    pick = ML.n1_pick(sp, txs, "1")
    states, ans, recs = S.system(sp, txs, pick)
    rel = [r["p"] for r in recs if r["note"] == "released"]
    eq("N1 pick releases q2 and q4 (net 3 + 2 = +5; q5 vetoed)", (counts(states, ans)[3], sorted(rel)), (5, ["q2", "q4"]))
    print("self-test %s" % ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    if len(sys.argv) < 2:
        print(__doc__ if __doc__ else "see the header of depth_scan.py")
        return 2
    mode = sys.argv[1]
    if mode == "selftest":
        return selftest()
    if len(sys.argv) < 3 or sys.argv[2] not in KEYS:
        print("usage: depth_scan.py MODE KEY (KEY = %s)" % " | ".join(KEYS))
        return 2
    key = sys.argv[2]
    k = KEYS[key]
    if mode == "checkx":
        xs, n, _ = read_mm(mm_file(k["ng"], "x"), ["x"])
        print("entry check 1 %s: MMLU x correct %d of %d, stored reference %d of 1000" % (k["ng"], xs["x"], n, k["ref_x"]))
        return 0 if (n == 1000 and xs["x"] == k["ref_x"]) else 1
    if mode == "checkgen":
        return checkgen(key)
    if mode == "choose":
        c = choose(key)
        p = mm_file(k["ng"], "choice")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(c, f, indent=1)
        os.replace(tmp, p)
        for l in sorted(c["layers"], key=int):
            r = c["layers"][l]
            print("  L%s: s1 %s (first out %s) | s05 %s (first out %s)%s%s" % (
                l, r["s1"], r["s1_first_out"], r["s05"], r["s05_first_out"], " | extended" if r["ext"] else "",
                " | NEEDS 12 / 16" if r["need_ext"] else ""))
        print("choice %s written: complete %s" % (p, c["complete"]))
        return 0
    if mode == "need-ext":
        print(" ".join(map(str, load_choice(key, complete=False)["need_ext"])))
        return 0
    if mode == "arms":
        print(" ".join(stage1_arms(load_choice(key))[1]))
        return 0
    if mode == "c-arms":
        print(" ".join(stage2_arms(k)))
        return 0
    if mode == "checkpos":
        if len(sys.argv) < 4 or sys.argv[3] not in ("E", "C"):
            print("usage: depth_scan.py checkpos KEY E|C")
            return 2
        return checkpos(key, sys.argv[3])
    if mode == "plan":
        return plan(key)
    if mode == "table":
        return table(key)
    if mode == "c":
        return c_table(key)
    if mode == "c-ref":
        return c_table(key, ref_only=True)
    print("unknown mode %s" % mode)
    return 2


if __name__ == "__main__":
    sys.exit(main())
