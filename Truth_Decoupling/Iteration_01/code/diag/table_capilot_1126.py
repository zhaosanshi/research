# Iteration_01 analysis (Claude Opus 5.5, executor si, 2026-11-26), read-only, no model, no judge call. Exploration.
# Plan discussion/work_queue/2026-11-26_queue_o.md, sections 2 and 3 (readings written before this program ran).
# TruthfulQA TUNE split only (192 questions), five models, the main table's settings at the write rule "prompt".
# Inputs per model: the main-table tune tables (diag/main_table_prompt.prompt_split: no write, single E, the N1 ladder
# E + C toward answering) and the new work dir <E run>/eval_extra/judge_tune_capilot_1126_tune with the two arms
# E + C toward CAUTION at 0.25 and 0.5 (conditions X_x--E<l>-w-p<s>-and-C<c>-w-p<k>-at-prompt; launcher
# launchers/explore_20261120/run_capilot_1126.sh). "Blank" = the rule detector or an empty answer (policy_select2.load),
# as everywhere in the project; net = right - wrong (hedge and blank 0).
#   N1 is reproduced first: its setting by main_table_layers.n1_pick on tune, its tune net against the reference of
#   results/write_positions/main_table_prompt_budget1_20261121.txt; a mismatch stops the table.
#   Reading 1: among the questions answered under single E with a right or wrong verdict, the share that becomes blank
#   under E + caution, wrong minus right (percentage points), separately for "changed" (the answer text under E differs
#   from the no-write text after release_origin.norm: lower case, punctuation to spaces) and "same"; 95% interval from
#   resampling the questions of the group (4000 resamples; a resample without a right or without a wrong answer is
#   skipped and counted).
#   Reading 2: rule A = N1, except that a question answered under E whose answer changed AND that is blank under
#   E + caution is handed in blank; rule B = the same without the condition "changed". Questions blank under E are
#   N1's own records (third pass). (rule minus N1) net with the paired interval of policy_select2.ci.
#   The five-model total of reading 2 is printed after the per-model lines.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/table_capilot_1126.py
import ast
import random
import sys

sys.path.insert(0, "Iteration_01/diag")
import iti_table as A  # noqa: E402

MP, ML, S, ps = A.MP, A.ML, A.S, A.ps
fr, ro = MP.fr, ML.ro
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")
# the settings of the task (E layer, E strength, C layer) and the N1 tune net of the reference file
EXPECT = {"llama3.1-8b": (16, "3.5", 19), "qwen3.5-9b": (15, "2", 16), "qwen3.8-27b": (28, "1.75", 33),
          "gemma3-12b": (24, "2.5", 25), "gemma3-27b": (28, "1", 31)}
REF_N1 = {"llama3.1-8b": 27, "qwen3.5-9b": 63, "qwen3.8-27b": 80, "gemma3-12b": 74, "gemma3-27b": 53}
KS = ("0-25", "0-5")
NAME = "capilot_1126"
SEED = 20261126
NB = 4000


def val(x):
    return (x == "R") - (x == "W")


def dash(x):
    return x.replace(".", "-")


def qaqa_set():
    """SET of diag/qaqa_pilot_table.py, read from its source (the module itself is not imported)."""
    tree = ast.parse(open("Iteration_01/diag/qaqa_pilot_table.py", encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SET" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("no SET in qaqa_pilot_table.py")


def share_diff(items):
    w = [b for s, b in items if s == "W"]
    r = [b for s, b in items if s == "R"]
    if not w or not r:
        return None
    return 100.0 * (sum(w) / len(w) - sum(r) / len(r))


def share_ci(items, seed):
    point = share_diff(items)
    if point is None:
        return None, None, None, NB
    rng = random.Random(seed)
    n, bs, skipped = len(items), [], 0
    for _ in range(NB):
        x = share_diff([items[rng.randrange(n)] for _ in range(n)])
        if x is None:
            skipped += 1
        else:
            bs.append(x)
    bs.sort()
    return point, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1], skipped


def counts(states, ans):
    return "answers %3d | right %3d wrong %3d | net %+4d" % (sum(ans), states.count("R"), states.count("W"), sum(map(val, states)))


def model_block(model, sset, tot):
    display, layer, s = A.KEY[model]
    el, es, cl = EXPECT[model]
    q = sset[model]
    if (layer, s.replace("-", "."), q[0], q[1], q[2]) != (el, es, el, es, cl):
        raise SystemExit("%s: settings differ: iti_table KEY %s / %s, qaqa_pilot_table SET %s, task %s" % (model, layer, s, q[:3], EXPECT[model]))
    T, why = MP.prompt_split(display, layer, 0, s)
    if T is None:
        raise SystemExit("%s: main-table tune tables not complete: %s" % (model, why))
    sp, tx, _, qs = T
    if sp.m["c"] != cl:
        raise SystemExit("%s: the main table's C layer is %s, the task says %s" % (model, sp.m["c"], cl))
    pick = ML.n1_pick(sp, tx, s)
    want = ("N1", s, dash(q[3]), dash(q[4]))
    n1 = S.system(sp, tx, pick)
    n1net = sum(map(val, n1[0]))
    print("\n## %s (display %s), E layer %d strength %s, C layer %d; tune questions %d" % (model, display, layer, s.replace("-", "."), cl, len(qs)))
    print("    N1 setting on tune %s (qaqa_pilot_table SET: %s) | N1 tune net %+d, reference %+d" % (pick, want, n1net, REF_N1[model]))
    if pick != want or n1net != REF_N1[model]:
        raise SystemExit("%s: N1 not reproduced (setting %s against %s, net %+d against %+d)" % (model, pick, want, n1net, REF_N1[model]))
    base = S.system(sp, tx, ("base",))
    e = S.system(sp, tx, ("E", s))
    cE = sp.cE(s)
    eblank = [sp.blank[(p, cE)] for p in qs]
    est = e[0]
    for i, p in enumerate(qs):
        if not eblank[i] and n1[0][i] != est[i]:
            raise SystemExit("%s: N1 differs from single E on a question E answers (%s)" % (model, p))
    changed = [ro.norm(tx.ec[(p, cE)]) != ro.norm(tx.ec[(p, "baseline_x")]) for p in qs]
    print("    %-34s %s" % ("no write", counts(base[0], base[1])))
    print("    %-34s %s" % ("single E", counts(e[0], e[1])))
    print("    %-34s %s" % ("N1", counts(n1[0], n1[1])))
    ans_e = [not b for b in eblank]
    print("    answered under E: %d (changed text %d, same text %d); blank under E %d" % (
        sum(ans_e), sum(a and c for a, c in zip(ans_e, changed)), sum(a and not c for a, c in zip(ans_e, changed)), sum(eblank)))
    # the new work dir
    _, run, _ = MP.usual(display, 0)
    w = "%s/eval_extra/judge_tune_%s_tune" % (run, NAME)
    if not MP.has(w):
        raise SystemExit("%s: no Pro verdicts in %s" % (model, w))
    st, bl = ps.load(w, "pro")
    ctx = fr.raw_texts(w)
    conds = {k: "X_x--E%d-w-p%s-and-C%d-w-p%s%s" % (layer, s, cl, k, MP.SUF) for k in KS}
    have = sorted({c for _, c in st})
    if have != sorted(["baseline_x"] + list(conds.values())):
        raise SystemExit("%s: %s holds conditions %s" % (model, w, have))
    for p in qs:
        k = (p, "baseline_x")
        if k not in st or st[k] != sp.st[k] or bl[k] != sp.blank[k] or ctx.get(k) != tx.ec[k]:
            raise SystemExit("%s: the no-write cell of %s differs from the usual one (%s)" % (model, w, p))
    out = {}
    for k in KS:
        c = conds[k]
        miss = [p for p in qs if (p, c) not in st]
        if miss:
            raise SystemExit("%s: %d tune questions have no cell of %s" % (model, len(miss), c))
        cb = [bl[(p, c)] for p in qs]
        cs = [st[(p, c)] for p in qs]
        nq = sum(x == "?" for x in cs)
        print("    E + C toward caution %-4s (as is) answers %3d | right %3d wrong %3d | net %+4d%s" % (
            k.replace("-", "."), sum(not b for b in cb), cs.count("R"), cs.count("W"), sum(map(val, cs)),
            "" if not nq else " | %d cells without a Pro verdict (counted 0 here; not used by the readings)" % nq))
        out[k] = cb
    # reading 1
    print("  reading 1: among answers of single E (right or wrong), share blank under E + caution, wrong minus right [95%, questions resampled]")
    for k in KS:
        cb = out[k]
        for gname, gsel in (("changed", True), ("same", False)):
            items = [(est[i], cb[i]) for i in range(len(qs)) if ans_e[i] and changed[i] == gsel and est[i] in "RW"]
            nh = sum(1 for i in range(len(qs)) if ans_e[i] and changed[i] == gsel and est[i] not in "RW")
            nw = sum(1 for x, _ in items if x == "W"); bw = sum(1 for x, b in items if x == "W" and b)
            nr = sum(1 for x, _ in items if x == "R"); br = sum(1 for x, b in items if x == "R" and b)
            pt, lo, hi, sk = share_ci(items, SEED)
            txt = "n/a (a group without right or without wrong answers)" if pt is None else "%+.1f [%+.1f, %+.1f] points%s" % (
                pt, lo, hi, "" if not sk else " (%d resamples skipped)" % sk)
            print("    caution %-4s %-7s wrong %3d -> blank %3d (%5.1f%%) | right %3d -> blank %3d (%5.1f%%) | other answered %2d | wrong minus right %s" % (
                k.replace("-", "."), gname, nw, bw, 100.0 * bw / nw if nw else 0.0, nr, br, 100.0 * br / nr if nr else 0.0, nh, txt))
    # reading 2
    print("  reading 2: rule minus N1, net [95%, paired, questions resampled]")
    same_as_n1 = rule_states(n1, eblank, changed, [False] * len(qs), True)
    if same_as_n1[0] != n1[0]:
        raise SystemExit("%s: a rule with an empty filter is not N1" % model)
    for k in KS:
        for rname, need_change in (("A (changed and blank)", True), ("B (blank)", False)):
            rs, ra, nf = rule_states(n1, eblank, changed, out[k], need_change)
            d = [val(x) - val(y) for x, y in zip(rs, n1[0])]
            fr_ = [n1[0][i] for i in range(len(qs)) if rs[i] == "X" and n1[0][i] != "X"]
            print("    caution %-4s rule %-22s %s | filtered %3d (right %3d, wrong %3d, other %2d) | minus N1 %s" % (
                k.replace("-", "."), rname, counts(rs, ra), nf, fr_.count("R"), fr_.count("W"), nf - fr_.count("R") - fr_.count("W"), ps.ci(d)))
            t = tot.setdefault((k, rname), {})
            for p, x in zip(qs, d):
                t[p] = t.get(p, 0) + x
            tot.setdefault(("models", k, rname), []).append(model)
    tot.setdefault("questions", []).append(set(qs))


def rule_states(n1, eblank, changed, cblank, need_change):
    states, ans, nf = list(n1[0]), list(n1[1]), 0
    for i in range(len(states)):
        if eblank[i]:
            continue
        if cblank[i] and (changed[i] or not need_change):
            states[i], ans[i] = "X", False
            nf += 1
    return states, ans, nf


def main():
    sset = qaqa_set()
    tot = {}
    bad = []
    print("# Plan O pilot (2026-11-26): E + C toward caution as a filter, TruthfulQA tune, DeepSeek Pro; net = right - wrong")
    for model in ORDER:
        try:
            model_block(model, sset, tot)
        except SystemExit as x:
            print("\n## %s: STOPPED: %s" % (model, x))
            bad.append(model)
    if bad:
        print("\n# incomplete: %s; no five-model total" % ", ".join(bad))
        return 1
    common = sorted(set.intersection(*tot["questions"]))
    print("\n## five models together (per question the differences are summed; %d questions all five have)" % len(common))
    for k in KS:
        for rname in ("A (changed and blank)", "B (blank)"):
            t = tot[(k, rname)]
            print("    caution %-4s rule %-22s minus N1 %s" % (k.replace("-", "."), rname, ps.ci([t[p] for p in common])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
