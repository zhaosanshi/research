# Iteration_01 analysis (Claude Opus 5.5, 2026-11-19), read-only, no model, no judge call.
# Work queue C2 / C3 on TriviaQA and PopQA (novelty/ours/2026-11-19_new_datasets_pilot_design.md). Scores the generations
# of scripts/newsets/newsets_gen.py with the per-question records of diag/final_records.py, UNCHANGED: the condition names
# are the engine's, so single E / single C / N1 / prompt B / the must-answer rows are exactly the TruthfulQA logic.
# Two judges, never mixed: DeepSeek Pro (main; tune = all 192 questions for the C2 prompts and must-answer + E, test = the
# fixed 100) and alias matching (auxiliary; every question; scripts/newsets/alias_match.py on the full alias list).
#   c2    per model and task (tune): the five prompts of step 0 (answers, right, wrong, hedge, net), must-answer + E,
#         step 0 = each sentence minus the must-answer prompt, alias-Pro agreement on answered cells, and D2: the gap of
#         scripts/newsets/newsets_gap.py with the sign predicted by the October rule P3 (+-0.03)
#   c3    per model and task (test): no write, single E, single C, N1 (TruthfulQA caps), prompt B, must-answer, must-answer
#         + E; the strong form at N1's answer count (must-answer + E, top N by mean log-prob); the extra row "caps
#         re-chosen on the new tune split by alias net"; five-model totals (questions resampled jointly) and the gate 1
#         reading of the design section 8; D2: the test gain of must-answer + E against the predicted sign
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/newsets_score.py c2|c3 [MODEL ...]
#                                                      python3 -B Iteration_01/diag/newsets_score.py --selftest
import glob
import json
import os
import random
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import policy_select2 as ps   # noqa: E402
import final_records as fr    # noqa: E402
import alias_match as AM      # noqa: E402

NS = "Iteration_01/newsets"
CFG = "Iteration_01/code/configs/newsets/frozen_models_20261119.json"
TASKS = ("triviaqa", "popqa")
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")   # first three = main text
LADNAME = {0.25: "0-25", 0.5: "0-5", 1: "1", 2: "2"}
C2_CONDS = ("baseline_x", "baseline_caut", "baseline_conf", "baseline_conf_E2", "baseline_conf_FR")
N_BOOT, SEED = 4000, 20261119


class NewSplit(ps.Split):
    """A Split over one judged question set of the new tasks: E at strength 2 only, single C at 0.25 only. Plain and
    must-answer conditions live in the same table (st = dst), as in the newsets judge inputs."""

    def __init__(self, el, cl, st, blank, need):
        self.m = dict(e=el, c=cl, order="EC")
        self.st, self.blank, self.dst, self.dblank = st, blank, st, blank
        self.cks = ["0-25"]
        miss = sorted(set(need) - {c for _, c in st})
        if miss:
            raise SystemExit("missing conditions %s" % miss)
        qs = sorted({p for p, _ in st})
        self.qs = [p for p in qs if all(st.get((p, c), "?") != "?" for c in need)]
        self.n_dropped = len(qs) - len(self.qs)


def need_test(sp):
    return ["baseline_x", sp.cE("2"), sp.cC("0-25"), "baseline_conf", sp.dE("2")] + [sp.cEC("2", k) for k in fr.LAD]


def models():
    with open(CFG, encoding="utf-8") as f:
        return json.load(f)["models"]


def bank(task):
    with open("%s/bank/%s_bank_20261119.jsonl" % (NS, task), encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def gen_rows(task, model, split):
    """{condition: {pair_id: row}} of the full generation files of one split."""
    out = {}
    for p in sorted(glob.glob("%s/gen/%s/%s/%s/*.json" % (NS, task, model, split))):
        if os.path.basename(p).split(".", 1)[0][1:].isdigit() and os.path.basename(p).startswith("t"):
            continue
        with open(p, encoding="utf-8") as f:
            g = json.load(f)
        if not g.get("full_pass") or g.get("split") != split or g.get("task") != task:
            raise SystemExit("%s is not a full %s %s file" % (p, task, split))
        out[g["condition"]] = {"%s:%s" % (task, r["id"]): r for r in g["rows"]}
    return out


def alias_tables(task, model, split, conds=None):
    """(st, blank, texts) from alias matching over the generation files."""
    b = bank(task)
    st, blank, tx = {}, {}, {}
    for c, rows in gen_rows(task, model, split).items():
        if conds and c not in conds:
            continue
        for pid, r in rows.items():
            iid = pid.split(":", 1)[1]
            bl = bool(r["rule_abstain"]) or not r["text"].strip()
            st[(pid, c)] = AM.verdict(r["text"], b[iid]["gold"], not bl)
            blank[(pid, c)] = bl
            tx[(pid, c)] = r["text"]
    return st, blank, tx


def pro_tables(task, model, work):
    w = "%s/judge/%s/%s/%s" % (NS, task, model, work)
    st, blank = ps.load(w, "pro")
    return st, blank, fr.raw_texts(w), w


def conf_rows(task, model, split, arm_token):
    p = "%s/conf/%s/%s/%s/%s.json" % (NS, task, model, split, arm_token)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return {"%s:%s" % (task, r["id"]): r for r in d["rows"]}


def tok(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def counts(states, ans):
    return dict(n=len(states), ans=sum(ans), R=sum(s == "R" for s in states), W=sum(s == "W" for s in states),
                H=sum(s == "H" for s in states), net=sum(s == "R" for s in states) - sum(s == "W" for s in states))


def vals(states):
    return [(s == "R") - (s == "W") for s in states]


def line(label, c):
    return "    %-58s answers %3d | right %3d wrong %3d hedge %2d | net %4d" % (label, c["ans"], c["R"], c["W"], c["H"], c["net"])


def system(sp, tx, cfg):
    recs = fr.records(sp, cfg, tx)
    return fr.states(recs), fr.answered(recs, sp, sp.dblank)


def top_n(sp, states, ans, rows, n):
    """Keep the n answered questions with the largest mean log-prob (ties by question order); the rest count as blank."""
    cand = [i for i, a in enumerate(ans) if a and rows[sp.qs[i]].get("mean_logp") is not None]
    order = sorted(cand, key=lambda i: (-rows[sp.qs[i]]["mean_logp"], i))
    kept = set(order[:n])
    return ["X" if i not in kept else s for i, s in enumerate(states)], [i in kept for i in range(len(states))]


def joint_ci(per_model, seed=SEED):
    """per_model: lists of per-question values on the same questions in the same order; total and 95% interval with the
    questions resampled jointly for all models."""
    n = len(per_model[0])
    if any(len(v) != n for v in per_model):
        raise ValueError("models are not on the same questions")
    q = [sum(v[i] for v in per_model) for i in range(n)]
    rng = random.Random(seed)
    bs = sorted(sum(q[rng.randrange(n)] for _ in range(n)) for _ in range(N_BOOT))
    return sum(q), bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1]


def holds(per_model_pro, per_model_alias):
    """Design section 8: five-model total positive with the 95% lower bound above 0 (Pro, 100 questions), at least three
    models positive, and the alias total over 500 questions of the same sign."""
    tot, lo, hi = joint_ci(per_model_pro)
    pos = sum(sum(v) > 0 for v in per_model_pro)
    atot = sum(sum(v) for v in per_model_alias)
    return (tot > 0 and lo > 0 and pos >= 3 and atot > 0), (tot, lo, hi, pos, atot)


def pick_caps(sp, tx):
    """(k1, k2) = (cap of E's blanks, cap of own blanks) maximising net on this split; ties keep the earlier (weaker)
    pair in final_records.CAPS order; (None, None) is E alone and not an N1 setting."""
    best, arg = None, None
    for k1 in fr.CAPS:
        for k2 in fr.CAPS:
            if k1 is None and k2 is None:
                continue
            v = sum(vals(system(sp, tx, ("N1", "2", k1, k2))[0]))
            if best is None or v > best:
                best, arg = v, (k1, k2)
    return arg, best


def d2_gap(task, model):
    p = "%s/gap/%s/%s/tune_gap.json" % (NS, task, model)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def predicted(gap):
    return "+" if gap > 0.03 else ("-" if gap < -0.03 else "0")


def report_c2(names):
    M = models()
    print("C2: tune 192, no write unless stated, DeepSeek Pro (main) and alias matching (auxiliary)")
    for task in TASKS:
        for name in names:
            m = M[name]
            st, blank, tx, w = pro_tables(task, name, "tune")
            ec = "E_L%d_conf-w-p2" % m["el"]
            need = list(C2_CONDS) + [ec]
            qs = sorted({p for p, c in st if c == "baseline_x"})
            qs = [p for p in qs if all(st.get((p, c), "?") != "?" for c in need)]
            with open(w + "/judge2/meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            print("\n== %s %s: %d questions judged (%d with refusal-only references excluded by the judge)" % (
                task, name, len(qs), meta.get("n_abstention_only_gold_questions", -1)))
            ast, ablank, _ = alias_tables(task, name, "tune", set(need))
            tab = {}
            for c in need:
                s = [st[(p, c)] for p in qs]
                tab[c] = s
                print(line(c, counts(s, [not blank[(p, c)] for p in qs])))
            for c in ("baseline_conf_E2", "baseline_conf_FR", ec):
                print("    step 0 / D2  %-30s minus baseline_conf: net %s" % (c, ps.ci([a - b for a, b in zip(vals(tab[c]), vals(tab["baseline_conf"]))])))
            agree = n = 0
            both = {("R", "R"): 0, ("R", "W"): 0, ("W", "R"): 0, ("W", "W"): 0}
            for c in need:
                for p in qs:
                    a, b = st[(p, c)], ast[(p, c)]
                    if not blank[(p, c)] and a in ("R", "W") and b in ("R", "W"):
                        both[(a, b)] += 1
                        n += 1
                        agree += a == b
            print("    alias vs Pro on answered right/wrong cells: %d / %d agree (%.1f%%); Pro R alias W %d, Pro W alias R %d" % (
                agree, n, 100.0 * agree / n if n else 0.0, both[("R", "W")], both[("W", "R")]))
            g = d2_gap(task, name)
            if g is None:
                print("    D2: no gap file yet")
            else:
                print("    D2: readout AUROC %.3f (flipped %.3f), confidence AUROC %.3f, gap %+.3f -> predicted sign of the test gain: %s" % (
                    g["readout_auroc"], 1 - g["readout_auroc"], g["confidence_auroc"], g["gap"], predicted(g["gap"])))


def report_c3(names):
    M = models()
    print("C3: test, frozen TruthfulQA setup; Pro on the fixed 100 questions, alias matching on all 500")
    gate = {}
    for task in TASKS:
        per = {"pro": {}, "alias": {}}
        for name in names:
            m = M[name]
            k1, k2 = LADNAME[m["caps"]["e_blank"]], LADNAME[m["caps"]["own_blank"]]
            cfgs = [("no write", ("base",)), ("single E (E 2)", ("E", "2")), ("single C (C 0.25)", ("C", "0-25")),
                    ("N1 (caps own %s / E's %s)" % (k2, k1), ("N1", "2", k1, k2)), ("prompt B (E 2)", ("B", "2")),
                    ("must-answer, no write", ("Dnone",)), ("must-answer + E 2", ("DE", "2"))]
            conf_e = conf_rows(task, name, "test", tok("conf:E%d.w.p2" % m["el"]))
            for judge in ("pro", "alias"):
                if judge == "pro":
                    st, blank, txt, _ = pro_tables(task, name, "test_pro100")
                else:
                    st, blank, txt = alias_tables(task, name, "test")
                sp = NewSplit(m["el"], m["cl"], st, blank, need_test(NewSplit(m["el"], m["cl"], st, blank, [])))
                tx = fr.Texts(txt, txt)
                print("\n== %s %s, %s: %d questions%s" % (task, name, judge, len(sp.qs), (" (%d dropped: a cell without a verdict)" % sp.n_dropped) if sp.n_dropped else ""))
                res = {}
                for label, cfg in cfgs:
                    s, a = system(sp, tx, cfg)
                    res[label] = (s, a)
                    print(line(label, counts(s, a)))
                s1, a1 = res["N1 (caps own %s / E's %s)" % (k2, k1)]
                s0, sE = res["no write"][0], res["single E (E 2)"][0]
                print("    N1 minus no write:  net %s" % ps.ci([x - y for x, y in zip(vals(s1), vals(s0))]))
                print("    N1 minus single E:  net %s" % ps.ci([x - y for x, y in zip(vals(s1), vals(sE))]))
                print("    single E minus no write: net %s" % ps.ci([x - y for x, y in zip(vals(sE), vals(s0))]))
                n = sum(a1)
                sd, ad = res["must-answer + E 2"]
                ks, ka = top_n(sp, sd, ad, conf_e, n)
                print(line("strong form: must-answer + E 2, top %d by mean log-prob" % n, counts(ks, ka)))
                print("    N1 minus strong form: net %s" % ps.ci([x - y for x, y in zip(vals(s1), vals(ks))]))
                print("    D2: test gain of must-answer + E 2 over must-answer: net %s" % ps.ci([x - y for x, y in zip(vals(sd), vals(res["must-answer, no write"][0]))]))
                per[judge][name] = {"n1_base": dict(zip(sp.qs, [x - y for x, y in zip(vals(s1), vals(s0))])),
                                    "n1_e": dict(zip(sp.qs, [x - y for x, y in zip(vals(s1), vals(sE))])),
                                    "e_base": dict(zip(sp.qs, [x - y for x, y in zip(vals(sE), vals(s0))]))}
                if judge == "alias":
                    ast, ablank, atx = alias_tables(task, name, "tune")
                    tsp = NewSplit(m["el"], m["cl"], ast, ablank, need_test(NewSplit(m["el"], m["cl"], ast, ablank, [])))
                    caps, tv = pick_caps(tsp, fr.Texts(atx, atx))
                    s2, a2 = system(sp, tx, ("N1", "2") + caps)
                    print(line("extra row: N1, caps re-chosen on the new tune split (E's %s, own %s; tune net %d)" % (caps[0], caps[1], tv), counts(s2, a2)))
        common = {j: sorted(set.intersection(*[set(per[j][x]["n1_base"]) for x in names])) for j in ("pro", "alias")}
        print("\n== %s: five-model totals on the questions every model has a verdict for (Pro %d, alias %d), resampled jointly" % (
            task, len(common["pro"]), len(common["alias"])))
        g = {}
        for key, label in (("n1_base", "N1 minus no write"), ("n1_e", "N1 minus single E"), ("e_base", "single E minus no write")):
            pro = [[per["pro"][x][key][q] for q in common["pro"]] for x in names]
            al = [[per["alias"][x][key][q] for q in common["alias"]] for x in names]
            ok, (tot, lo, hi, pos, atot) = holds(pro, al)
            g[key] = ok
            print("    %-26s Pro %+d [%+d, %+d], %d of %d models positive | alias total %+d | holds: %s" % (label, tot, lo, hi, pos, len(names), atot, ok))
        gate[task] = g
    if set(names) == set(ORDER):
        r1 = all(gate[t]["n1_base"] and gate[t]["n1_e"] for t in TASKS)
        r2 = (not any(gate[t]["e_base"] for t in TASKS)) and any(gate[t]["n1_e"] for t in TASKS)
        reading = 1 if r1 else (2 if r2 else 3)
        print("\ngate 1 reading (design section 8): %d" % reading)
    else:
        print("\ngate 1 reading needs all five models")


def selftest():
    # q1: E's blank, its first answer at C 0.25 is the no-write answer -> vetoed; q2: own blank released at C 0.25;
    # q3: not a candidate; q4: own blank above a cap of 0.25 (first answer at C 1)
    el, cl = 11, 19
    sp0 = NewSplit(el, cl, {("q", "baseline_x"): "R"}, {("q", "baseline_x"): False}, [])
    conds = need_test(sp0)
    st, blank, tx = {}, {}, {}

    def put(p, c, text, state):
        st[(p, c)], blank[(p, c)], tx[(p, c)] = state, text == "I don't know.", text
    for p, base, e, lad in (("q1", ("Paris.", "R"), ("I don't know.", "X"), [("Paris.", "R")] * 4),
                            ("q2", ("I don't know.", "X"), ("I don't know.", "X"), [("Lyon.", "W")] * 4),
                            ("q3", ("Rome.", "R"), ("Rome.", "R"), [("Rome.", "R")] * 4),
                            ("q4", ("I don't know.", "X"), ("I don't know.", "X"), [("I don't know.", "X")] * 2 + [("Nice.", "R")] * 2)):
        put(p, "baseline_x", *base)
        put(p, sp0.cE("2"), *e)
        for k, a in zip(fr.LAD, lad):
            put(p, sp0.cEC("2", k), *a)
        for c in conds:
            if (p, c) not in st:
                put(p, c, "Something.", "W")
    sp = NewSplit(el, cl, st, blank, conds)
    t = fr.Texts(tx, tx)
    recs = fr.records(sp, ("N1", "2", "0-25", "0-25"), t)
    got = [(r["p"], r["state"], r["note"]) for r in recs]
    want = [("q1", "X", "vetoed: the original answer came back"), ("q2", "W", "released"),
            ("q3", "R", "not a candidate"), ("q4", "X", "above the cap")]
    assert got == want, got
    assert [r["state"] for r in fr.records(sp, ("N1", "2", "0-25", "1"), t)] == ["X", "W", "R", "R"]
    assert fr.answered(recs, sp, sp.dblank) == [False, True, True, False]
    s, a = top_n(sp, ["R", "W", "R", "W"], [True, True, True, False],
                 {"q1": {"mean_logp": -0.1}, "q2": {"mean_logp": -0.5}, "q3": {"mean_logp": -0.2}, "q4": {"mean_logp": -0.01}}, 2)
    assert s == ["R", "X", "R", "X"] and a == [True, False, True, False], (s, a)
    tot, lo, hi = joint_ci([[1, 0, 1], [0, 1, 1]])
    assert tot == 4 and lo <= tot <= hi
    ok, info = holds([[1] * 10] * 5, [[1] * 10] * 5)
    assert ok and info[3] == 5
    ok, _ = holds([[1] * 10] * 2 + [[-1] * 10] * 3, [[1] * 10] * 5)
    assert not ok
    assert predicted(0.05) == "+" and predicted(-0.05) == "-" and predicted(0.01) == "0"
    print("selftest PASS")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    if len(sys.argv) < 2 or sys.argv[1] not in ("c2", "c3"):
        raise SystemExit(__doc__ or "usage: newsets_score.py c2|c3 [MODEL ...]")
    names = sys.argv[2:] or list(ORDER)
    (report_c2 if sys.argv[1] == "c2" else report_c3)(names)


if __name__ == "__main__":
    main()
