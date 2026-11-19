# Iteration_01 analysis (Claude Opus 5.5, 2026-11-19), read-only, no model, no judge call.
# Work queue C2 / C3 on TriviaQA and PopQA (novelty/ours/2026-11-19_new_datasets_pilot_design.md). Scores the generations
# of scripts/newsets/newsets_gen.py with the per-question records of diag/final_records.py, UNCHANGED: the condition names
# are the engine's, so single E / single C / N1 / prompt B / the must-answer rows are exactly the TruthfulQA logic.
# Two judges, never mixed: DeepSeek Pro (main; votes verified by scripts/newsets/newsets_votes.py, bound to the current
# answers) and alias matching (auxiliary; every question; scripts/newsets/alias_match.py on the full alias list).
# Inputs are checked against the frozen ID lists before anything is counted (GPT review 2026-11-19, section 2):
#   every generation file is a complete full pass of this model over the whole split, in bank order; every work dir
#   judges exactly the expected questions x conditions; exclusions are listed with their reason (refusal-only references,
#   Pro declined); a model or a five-model table below 90% of the expected questions is "insufficient evidence";
#   confidences are bound to the generation file (sha256) and to the exact answer text (sha1) before ranking; a
#   comparator that cannot reach the target answer count is "not comparable".
#   c2    tune: the six C2 conditions, step 0, alias-Pro agreement, D2 gap with its predicted sign
#   c3    test: the systems, generations per question, the strong form at N1's answer count, veto on / off, the extra
#         row (caps re-chosen on the new tune split by alias net), alias-Pro agreement on the Pro 100, D2 against the
#         prediction, five-model totals and the gate 1 reading of the design section 8
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/newsets_score.py c2|c3 [MODEL ...]
#                                                      python3 -B Iteration_01/diag/newsets_score.py --selftest
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import policy_select2 as ps   # noqa: E402
import final_records as fr    # noqa: E402
import alias_match as AM      # noqa: E402
import newsets_votes as NV    # noqa: E402

NS = "Iteration_01/newsets"
CFG = "Iteration_01/code/configs/newsets/frozen_models_20261119.json"
TASKS = ("triviaqa", "popqa")
ORDER = ("llama3.1-8b", "qwen3.5-9b", "qwen3.8-27b", "gemma3-12b", "gemma3-27b")   # first three = main text
LADNAME = {0.25: "0-25", 0.5: "0-5", 1: "1", 2: "2"}
C2_ARMS = ("x", "caut", "conf", "conf_E2", "conf_FR")
MIN_SHARE = 0.90
N_BOOT, SEED = 4000, 20261119
GEN_VERSION = 2
NEAR_ZERO = 2        # D2: a test gain within +-2 net points on the Pro 100 counts as "about zero" (design section 7)


class NewSplit(ps.Split):
    """A Split over one judged question set: E at strength 2 only, single C at 0.25 only; plain and must-answer
    conditions live in the same table (st = dst)."""

    def __init__(self, el, cl, st, blank, need, qs=None):
        self.m = dict(e=el, c=cl, order="EC")
        self.st, self.blank, self.dst, self.dblank = st, blank, st, blank
        self.cks = ["0-25"]
        miss = sorted(set(need) - {c for _, c in st})
        if miss:
            raise SystemExit("missing conditions %s" % miss)
        qs = sorted({p for p, _ in st}) if qs is None else list(qs)
        self.qs = [p for p in qs if all(st.get((p, c), "?") != "?" for c in need)]
        self.n_dropped = len(qs) - len(self.qs)


def shell(el, cl):
    return NewSplit(el, cl, {}, {}, [])


def need_test(sp):
    return ["baseline_x", sp.cE("2"), sp.cC("0-25"), "baseline_conf", sp.dE("2")] + [sp.cEC("2", k) for k in fr.LAD]


def tok(arm):
    return arm.replace(":", "--").replace("+", "-and-").replace(".", "-")


def test_arms(m):
    e, c = "E%d.w.p2" % m["el"], "C%d.w" % m["cl"]
    return ["x", "x:" + e] + ["x:%s+%s.m%s" % (e, c, k) for k in ("0.25", "0.5", "1", "2")] + ["x:%s.m0.25" % c, "conf", "conf:" + e]


def models():
    with open(CFG, encoding="utf-8") as f:
        return json.load(f)["models"]


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha1(text):
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def bank(task):
    p = "%s/bank/%s_bank_20261119.jsonl" % (NS, task)
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return {r["id"]: r for r in rows}, rows, p


def expected(task, split, pro100=False):
    _, rows, _ = bank(task)
    ids = [r["id"] for r in rows if r["split"] == split and (not pro100 or r["pro100"])]
    return ["%s:%s" % (task, i) for i in ids]


def gen_file(task, model, split, arm):
    """A complete full-pass generation of this model over the whole split, in bank order."""
    p = "%s/gen/%s/%s/%s/%s.json" % (NS, task, model, split, tok(arm))
    with open(p, encoding="utf-8") as f:
        g = json.load(f)
    c = g.get("contract") or {}
    _, _, bank_path = bank(task)
    want = [x.split(":", 1)[1] for x in expected(task, split)]
    ok = (g.get("completed") is True and c.get("gen_version") == GEN_VERSION and c.get("full_pass") is True
          and (c.get("task"), c.get("split"), c.get("model"), c.get("arm")) == (task, split, model, arm)
          and c.get("items_file_sha256") == sha_file(bank_path) and [r["id"] for r in g["rows"]] == want == c.get("ids"))
    if not ok:
        raise SystemExit("%s is not a complete full pass of %s over the %s %s split" % (p, model, task, split))
    return g, p


def alias_tables(task, model, split, arms):
    """(st, blank, texts, gen files) from alias matching over the verified generation files."""
    b, _, _ = bank(task)
    st, blank, tx, files = {}, {}, {}, {}
    for arm in arms:
        g, p = gen_file(task, model, split, arm)
        c = g["contract"]["condition"]
        files[arm] = p
        for r in g["rows"]:
            pid = "%s:%s" % (task, r["id"])
            bl = bool(r["rule_abstain"]) or not r["text"].strip()
            st[(pid, c)] = AM.verdict(r["text"], b[r["id"]]["gold"], not bl)
            blank[(pid, c)] = bl
            tx[(pid, c)] = r["text"]
    return st, blank, tx, files


def pro_tables(task, model, work, conds, want):
    """(st, blank, texts, exclusions, counts) from verified Pro votes; the work dir must judge exactly want x conds."""
    w = "%s/judge/%s/%s/%s" % (NS, task, model, work)
    split = "tune" if work == "tune" else "test"
    table, counts, meta = NV.load_votes(w, "%s/gen/%s/%s/%s" % (NS, task, model, split))
    if meta.get("model") != model:
        raise SystemExit("%s was built for %s" % (w, meta.get("model")))
    have = {(p, c) for p, c in table}
    exp = {(p, c) for p in want for c in conds}
    if have != exp:
        raise SystemExit("%s judges %d cells, expected %d questions x %d conditions" % (w, len(have), len(want), len(conds)))
    st = {k: v["state"] for k, v in table.items()}
    blank = {k: v["blank"] for k, v in table.items()}
    tx = {k: v["text"] for k, v in table.items()}
    excl = {}
    for (p, c), v in table.items():
        if v["state"] == "?":
            excl.setdefault(p, set()).add(v["why"])
    return st, blank, tx, excl, counts


def conf_rows(task, model, split, arm, judged_text):
    """Confidence rows of one arm, bound to the generation file that was judged and to each judged answer text."""
    p = "%s/conf/%s/%s/%s/%s.json" % (NS, task, model, split, tok(arm))
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    _, gp = gen_file(task, model, split, arm)
    if d.get("generation_sha256") != sha_file(gp) or d.get("arm") != arm or d.get("model") != model:
        raise SystemExit("%s was not computed on the generation that is scored" % p)
    out = {}
    for r in d["rows"]:
        pid = "%s:%s" % (task, r["id"])
        if pid in judged_text and r["text_sha1"] != sha1(judged_text[pid]):
            raise SystemExit("%s: the confidence of %s belongs to another answer text" % (p, pid))
        out[pid] = r
    return out


def counts(states, ans):
    return dict(n=len(states), ans=sum(ans), R=sum(s == "R" for s in states), W=sum(s == "W" for s in states),
                H=sum(s == "H" for s in states), net=sum(s == "R" for s in states) - sum(s == "W" for s in states))


def vals(states):
    return [(s == "R") - (s == "W") for s in states]


def line(label, c):
    return "    %-58s answers %3d | right %3d wrong %3d hedge %2d | net %4d" % (label, c["ans"], c["R"], c["W"], c["H"], c["net"])


def system(sp, tx, cfg):
    recs = fr.records(sp, cfg, tx)
    return fr.states(recs), fr.answered(recs, sp, sp.dblank), recs


def top_n(sp, states, ans, rows, n):
    """Keep the n answered questions with the largest mean log-prob (ties by question order); None when fewer than n
    answered questions carry a confidence (not comparable)."""
    cand = [i for i, a in enumerate(ans) if a and rows[sp.qs[i]].get("mean_logp") is not None]
    if len(cand) < n:
        return None
    order = sorted(cand, key=lambda i: (-rows[sp.qs[i]]["mean_logp"], i))
    kept = set(order[:n])
    out = ["X" if i not in kept else s for i, s in enumerate(states)], [i in kept for i in range(len(states))]
    if sum(out[1]) != n:
        raise RuntimeError("kept %d answers, wanted %d" % (sum(out[1]), n))
    return out


def joint_ci(per_model, seed=SEED):
    n = len(per_model[0])
    if any(len(v) != n for v in per_model):
        raise ValueError("models are not on the same questions")
    q = [sum(v[i] for v in per_model) for i in range(n)]
    rng = random.Random(seed)
    bs = sorted(sum(q[rng.randrange(n)] for _ in range(n)) for _ in range(N_BOOT))
    return sum(q), bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1]


def holds(per_model_pro, per_model_alias):
    """Design section 8: five-model total positive with the 95% lower bound above 0 (Pro), at least three models
    positive, and the alias total of the same sign."""
    tot, lo, hi = joint_ci(per_model_pro)
    pos = sum(sum(v) > 0 for v in per_model_pro)
    atot = sum(sum(v) for v in per_model_alias)
    return (tot > 0 and lo > 0 and pos >= 3 and atot > 0), (tot, lo, hi, pos, atot)


def n1_generations(recs, k1, k2):
    """Logical generations per question of N1: 1 for E, plus every C strength actually tried, the vetoed one included."""
    out = []
    for r in recs:
        n = 1
        if r["cand"] is not None:
            cap = k2 if r["cand"] == "own" else k1
            lad = fr.LAD[:fr.LAD.index(cap) + 1] if cap is not None else []
            if lad:
                n += len(lad) if r["note"] in ("above the cap", "blank at every dose") else lad.index(r["rung"]) + 1
        out.append(n)
    return out


def pick_caps(sp, tx):
    """(k1, k2) = (cap of E's blanks, cap of own blanks) maximising net; ties keep the earlier pair in CAPS order."""
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
    if gap is None:
        return "not estimable"
    return "+" if gap > 0.03 else ("-" if gap < -0.03 else "0")


def observed(gain):
    return "+" if gain > NEAR_ZERO else ("-" if gain < -NEAR_ZERO else "0")


def agreement(st_a, st_b, blank, keys):
    agree = n = 0
    for k in keys:
        a, b = st_a.get(k), st_b.get(k)
        if not blank.get(k) and a in ("R", "W") and b in ("R", "W"):
            n += 1
            agree += a == b
    return agree, n


def report_c2(names):
    M = models()
    print("C2: tune 192, Pro (main) and alias matching (auxiliary)")
    for task in TASKS:
        want = expected(task, "tune")
        for name in names:
            m = M[name]
            arms = list(C2_ARMS) + ["conf:E%d.w.p2" % m["el"]]
            conds = ["baseline_" + a for a in C2_ARMS] + ["E_L%d_conf-w-p2" % m["el"]]
            st, blank, tx, excl, cnt = pro_tables(task, name, "tune", conds, want)
            qs = [p for p in want if p not in excl]
            print("\n== %s %s: %d of %d questions (excluded: %s); votes %s" % (task, name, len(qs), len(want),
                  json.dumps({p: sorted(v) for p, v in excl.items()}) if excl else "none", json.dumps(cnt)))
            if len(qs) < MIN_SHARE * len(want):
                print("    insufficient evidence: fewer than %d%% of the questions are usable" % int(100 * MIN_SHARE))
                continue
            ast, ablank, _, _ = alias_tables(task, name, "tune", arms)
            tab = {}
            for c in conds:
                s = [st[(p, c)] for p in qs]
                tab[c] = s
                print(line(c, counts(s, [not blank[(p, c)] for p in qs])))
            for c in conds[3:]:
                print("    step 0 / D2  %-30s minus baseline_conf: net %s" % (c, ps.ci([a - b for a, b in zip(vals(tab[c]), vals(tab["baseline_conf"]))])))
            agree, n = agreement(st, ast, blank, [(p, c) for p in qs for c in conds])
            print("    alias vs Pro on answered right / wrong cells: %d / %d agree (%.1f%%)" % (agree, n, 100.0 * agree / n if n else 0.0))
            g = d2_gap(task, name)
            tune_gain = sum(vals(tab[conds[-1]])) - sum(vals(tab["baseline_conf"]))
            if g is None:
                print("    D2: no gap file yet")
            else:
                print("    D2: counts %s | readout AUROC %s | confidence AUROC %s | gap %s -> predicted sign of the test gain: %s"
                      " | reference predictor (tune gain of must-answer + E, %+d): %s" % (
                          json.dumps(g["counts"]), g["readout_auroc"], g["confidence_auroc"], g["gap"], predicted(g["gap"]),
                          tune_gain, observed(tune_gain * 100.0 / len(qs))))


def report_c3(names):
    M = models()
    print("C3: test, frozen TruthfulQA setup; Pro on the fixed 100 questions, alias matching on all 500")
    gate, enough = {}, True
    for task in TASKS:
        per = {"pro": {}, "alias": {}}
        same_n = {}
        for name in names:
            m = M[name]
            k1, k2 = LADNAME[m["caps"]["e_blank"]], LADNAME[m["caps"]["own_blank"]]
            arms = test_arms(m)
            sh = shell(m["el"], m["cl"])
            conds = need_test(sh)
            cfgs = [("no write", ("base",)), ("single E (E 2)", ("E", "2")), ("single C (C 0.25)", ("C", "0-25")),
                    ("N1 (caps own %s / E's %s)" % (k2, k1), ("N1", "2", k1, k2)), ("prompt B (E 2)", ("B", "2")),
                    ("must-answer, no write", ("Dnone",)), ("must-answer + E 2", ("DE", "2"))]
            pro_st = None
            for judge in ("pro", "alias"):
                if judge == "pro":
                    want = expected(task, "test", pro100=True)
                    st, blank, txt, excl, cnt = pro_tables(task, name, "test_pro100", conds, want)
                    pro_st, pro_blank = st, blank
                else:
                    want = expected(task, "test")
                    st, blank, txt, _ = alias_tables(task, name, "test", arms)
                    excl, cnt = {}, {}
                qs = [p for p in want if p not in excl]
                sp = NewSplit(m["el"], m["cl"], st, blank, conds, qs=qs)
                if sp.n_dropped:
                    raise SystemExit("%s %s %s: %d questions lack a verdict" % (task, name, judge, sp.n_dropped))
                tx = fr.Texts(txt, txt)
                ok = len(sp.qs) >= MIN_SHARE * len(want)
                enough &= ok
                print("\n== %s %s, %s: %d of %d questions%s%s" % (task, name, judge, len(sp.qs), len(want),
                      (" | excluded %s" % json.dumps({p: sorted(v) for p, v in excl.items()})) if excl else "",
                      "" if ok else " | INSUFFICIENT EVIDENCE"))
                res = {}
                for label, cfg in cfgs:
                    s, a, recs = system(sp, tx, cfg)
                    res[label] = (s, a, recs)
                    print(line(label, counts(s, a)))
                n1lab = "N1 (caps own %s / E's %s)" % (k2, k1)
                s1, a1, r1 = res[n1lab]
                s0, sE = res["no write"][0], res["single E (E 2)"][0]
                gens = n1_generations(r1, k1, k2)
                bgen = [1 + (r["cand"] is not None) for r in res["prompt B (E 2)"][2]]
                print("    generations per question: N1 %.3f | prompt B %.3f | every other system 1 (the strong form also needs a"
                      " scoring pass for its confidence; the offline grid is not counted)" % (sum(gens) / len(gens), sum(bgen) / len(bgen)))
                for lab, x, y in (("N1 minus no write", s1, s0), ("N1 minus single E", s1, sE), ("single E minus no write", sE, s0)):
                    print("    %-24s net %s" % (lab, ps.ci([a - b for a, b in zip(vals(x), vals(y))])))
                sv, _, _ = system(sp, tx, ("T2", "2", k1, k2))
                print("    veto on minus off (same caps, same questions): net %s" % ps.ci([a - b for a, b in zip(vals(s1), vals(sv))]))
                n = sum(a1)
                sd, ad, _ = res["must-answer + E 2"]
                judged = {p: txt[(p, sh.dE("2"))] for p in sp.qs}
                kept = top_n(sp, sd, ad, conf_rows(task, name, "test", "conf:E%d.w.p2" % m["el"], judged), n)
                if kept is None:
                    print("    strong form at %d answers: NOT COMPARABLE (must-answer + E has fewer answered questions with a confidence)" % n)
                    same_n.setdefault(judge, {})[name] = None
                else:
                    ks, ka = kept
                    print(line("strong form: must-answer + E 2, top %d by mean log-prob" % n, counts(ks, ka)))
                    d = [x - y for x, y in zip(vals(s1), vals(ks))]
                    print("    N1 minus strong form: net %s" % ps.ci(d))
                    same_n.setdefault(judge, {})[name] = dict(zip(sp.qs, d))
                dm = [x - y for x, y in zip(vals(sd), vals(res["must-answer, no write"][0]))]
                print("    D2: test gain of must-answer + E 2 over must-answer: net %s" % ps.ci(dm))
                if judge == "pro":
                    g = d2_gap(task, name)
                    print("    D2 cell: predicted %s (gap %s), observed %s (gain %+d on %d questions; about zero = within +-%d)" % (
                        predicted(None if g is None else g["gap"]), None if g is None else g["gap"], observed(sum(dm) * 100.0 / len(dm)),
                        sum(dm), len(dm), NEAR_ZERO))
                per[judge][name] = {"n1_base": dict(zip(sp.qs, [x - y for x, y in zip(vals(s1), vals(s0))])),
                                    "n1_e": dict(zip(sp.qs, [x - y for x, y in zip(vals(s1), vals(sE))])),
                                    "e_base": dict(zip(sp.qs, [x - y for x, y in zip(vals(sE), vals(s0))]))}
                if judge == "alias":
                    keys = [(p, c) for p in expected(task, "test", pro100=True) for c in conds if (p, c) in pro_st]
                    agree, nn = agreement(pro_st, st, pro_blank, keys)
                    print("    alias vs Pro on the Pro 100, answered right / wrong cells: %d / %d agree (%.1f%%)" % (agree, nn, 100.0 * agree / nn if nn else 0.0))
                    tarms = ["x", "x:E%d.w.p2" % m["el"]] + [a for a in arms if "+" in a] + ["x:C%d.w.m0.25" % m["cl"], "conf", "conf:E%d.w.p2" % m["el"]]
                    ast, ablank, atx, _ = alias_tables(task, name, "tune", tarms)
                    tsp = NewSplit(m["el"], m["cl"], ast, ablank, conds, qs=expected(task, "tune"))
                    caps, tv = pick_caps(tsp, fr.Texts(atx, atx))
                    s2, a2, _ = system(sp, tx, ("N1", "2") + caps)
                    print(line("extra row: N1, caps re-chosen on the new tune split (E's %s, own %s; tune net %d)" % (caps[0], caps[1], tv), counts(s2, a2)))
        print("\n== %s: five-model totals on the questions every model has a verdict for, resampled jointly" % task)
        g = {}
        for key, label in (("n1_base", "N1 minus no write"), ("n1_e", "N1 minus single E"), ("e_base", "single E minus no write")):
            common = {j: sorted(set.intersection(*[set(per[j][x][key]) for x in names])) for j in ("pro", "alias")}
            exp_n = {"pro": len(expected(task, "test", pro100=True)), "alias": len(expected(task, "test"))}
            ok_n = all(len(common[j]) >= MIN_SHARE * exp_n[j] for j in common)
            enough &= ok_n
            pro = [[per["pro"][x][key][q] for q in common["pro"]] for x in names]
            al = [[per["alias"][x][key][q] for q in common["alias"]] for x in names]
            ok, (tot, lo, hi, pos, atot) = holds(pro, al)
            g[key] = ok and ok_n
            print("    %-26s Pro %+d [%+d, %+d] on %d questions, %d of %d models positive | alias total %+d on %d | holds: %s%s" % (
                label, tot, lo, hi, len(common["pro"]), pos, len(names), atot, len(common["alias"]), g[key],
                "" if ok_n else " (INSUFFICIENT EVIDENCE)"))
        for j in ("pro", "alias"):
            comp = {x: v for x, v in same_n.get(j, {}).items() if v is not None}
            if len(comp) < len(names):
                print("    N1 minus strong form, %s: not comparable for %s" % (j, sorted(set(names) - set(comp))))
            if comp:
                common = sorted(set.intersection(*[set(v) for v in comp.values()]))
                tot, lo, hi = joint_ci([[v[q] for q in common] for v in comp.values()])
                print("    N1 minus strong form, %s, %d comparable models: %+d [%+d, %+d] on %d questions" % (j, len(comp), tot, lo, hi, len(common)))
        gate[task] = g
    if set(names) != set(ORDER):
        print("\ngate 1 reading needs all five models")
    elif not enough:
        print("\ngate 1 reading: INSUFFICIENT EVIDENCE (a model or a table has fewer than %d%% of the expected questions)" % int(100 * MIN_SHARE))
    else:
        r1 = all(gate[t]["n1_base"] and gate[t]["n1_e"] for t in TASKS)
        r2 = (not any(gate[t]["e_base"] for t in TASKS)) and any(gate[t]["n1_e"] for t in TASKS)
        print("\ngate 1 reading (design section 8): %d" % (1 if r1 else (2 if r2 else 3)))


def selftest():
    el, cl = 11, 19
    sp0 = shell(el, cl)
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
    s, a, recs = system(sp, t, ("N1", "2", "0-25", "0-25"))
    assert [(r["p"], r["state"], r["note"]) for r in recs] == [("q1", "X", "vetoed: the original answer came back"), ("q2", "W", "released"),
                                                             ("q3", "R", "not a candidate"), ("q4", "X", "above the cap")]
    assert n1_generations(recs, "0-25", "0-25") == [2, 2, 1, 2]
    assert [x for x in system(sp, t, ("N1", "2", "0-25", "1"))[0]] == ["X", "W", "R", "R"]
    assert system(sp, t, ("T2", "2", "0-25", "0-25"))[0] == ["R", "W", "R", "X"]      # veto off: q1's original answer counts
    conf = {"q1": {"mean_logp": -0.1}, "q2": {"mean_logp": -0.5}, "q3": {"mean_logp": -0.2}, "q4": {"mean_logp": -0.01}}
    ks, ka = top_n(sp, ["R", "W", "R", "W"], [True, True, True, False], conf, 2)
    assert ks == ["R", "X", "R", "X"] and ka == [True, False, True, False]
    assert top_n(sp, ["R", "W", "R", "W"], [True, False, False, False], conf, 2) is None        # GPT: target 2, one eligible
    tot, lo, hi = joint_ci([[1, 0, 1], [0, 1, 1]])
    assert tot == 4 and lo <= tot <= hi
    ok, info = holds([[1] * 10] * 5, [[1] * 10] * 5)
    assert ok and info[3] == 5
    assert not holds([[1] * 10] * 2 + [[-1] * 10] * 3, [[1] * 10] * 5)[0]
    assert predicted(0.05) == "+" and predicted(-0.05) == "-" and predicted(0.01) == "0" and predicted(None) == "not estimable"
    assert observed(3) == "+" and observed(-3) == "-" and observed(2) == "0"
    selftest_gate()
    print("selftest PASS")


def selftest_gate():
    """GPT review counter-example: one usable question per model and task must not yield a gate reading."""
    import contextlib
    import io
    from unittest.mock import patch
    mod = sys.modules[__name__]
    M = {n: dict(el=11, cl=19, caps=dict(e_blank=0.25, own_blank=0.25)) for n in ORDER}

    def one(task):
        q = task + ":q"
        sh = shell(11, 19)
        st = {(q, c): "R" for c in need_test(sh)}
        st[(q, "baseline_x")] = st[(q, sh.cE("2"))] = "X"
        return st, {k: v == "X" for k, v in st.items()}, {k: ("I don't know." if v == "X" else "gold") for k, v in st.items()}

    def exp(task, split, pro100=False):
        return ["%s:q" % task] + ["%s:z%d" % (task, i) for i in range(1, 100 if pro100 else (500 if split == "test" else 192))]

    def pro(task, model, work, conds, want):
        st, bl, tx = one(task)
        return st, bl, tx, {}, {}

    def alias(task, model, split, arms):
        st, bl, tx = one(task)
        return st, bl, tx, {}
    out = io.StringIO()
    with patch.object(mod, "models", return_value=M), patch.object(mod, "expected", exp), patch.object(mod, "pro_tables", pro), \
            patch.object(mod, "alias_tables", alias), patch.object(mod, "conf_rows", lambda *a: {}), patch.object(mod, "d2_gap", lambda *a: None):
        try:
            with contextlib.redirect_stdout(out):
                report_c3(list(ORDER))
            text = out.getvalue()
            assert "gate 1 reading (design" not in text and "INSUFFICIENT EVIDENCE" in text, text[-400:]
        except SystemExit as e:
            assert "lack a verdict" in str(e), e


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    if len(sys.argv) < 2 or sys.argv[1] not in ("c2", "c3"):
        raise SystemExit("usage: newsets_score.py c2|c3 [MODEL ...]")
    names = sys.argv[2:] or list(ORDER)
    (report_c2 if sys.argv[1] == "c2" else report_c3)(names)


if __name__ == "__main__":
    main()
