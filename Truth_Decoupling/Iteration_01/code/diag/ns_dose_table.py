# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# TriviaQA / PopQA with the E strength and the two C caps chosen on each dataset's own tune split (the lead, 2026-11-21:
# the method is fixed, the strengths may differ by model and by dataset; is N1 better than E alone and than C alone?).
# Answers: Iteration_01/newsets/gen/<task>/<model>/<split>/ (launchers newsets_c2c3_20261119.sh and
# explore_20261120/ns_dose_all.sh): for every E strength s a full pass of single E and of E + C at 0.25 / 0.5 / 1 / 2;
# the no-write pass and single C at 0.25. Verdicts by alias match (scripts/newsets/alias_match.py: right / wrong, blank
# by the rule detector), so every question of a split is scored and nothing is paid.
# Systems (records through diag/final_records.py unchanged): no write, single C 0.25, single E at s, N1 at s with the
# caps (E's blanks, own blanks) chosen on tune; "N1 without the veto" (final_records kind T2: an E's blank may also be
# released with the answer the model gave without any write) is printed as a diagnostic.
# Per model and task: the tune table over strengths; then two choices made on tune only -
#   (a) the strength with the best tune N1 (ties to the lower strength), and single E at that same strength;
#   (b) single E at ITS OWN best tune strength (the stronger single-direction baseline);
# and their test numbers with paired intervals (questions resampled).
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/ns_dose_table.py [TASK ...]
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_score as S     # noqa: E402
import final_records as fr    # noqa: E402
import alias_match as AM      # noqa: E402

NS = S.NS
STRENGTHS = ("0.5", "1", "1.5", "2", "3", "4")
KS = ("0.25", "0.5", "1", "2")
CAPS = [None, "0-25", "0-5", "1", "2"]


def dash(s):
    return s.replace(".", "-")


def bank(task, split):
    """Rows of one split; the second batch of test questions (test2) has its own bank file."""
    p = "%s/bank/%s_bank_%s.jsonl" % (NS, task, "test2_20261120" if split == "test2" else "20261119")
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return {r["id"]: r for r in rows if r["split"] == split}, p


def load(task, model, m, split):
    """(Split over alias verdicts, texts, strengths present) for one model, task and split."""
    b, bp = bank(task, split)
    want = list(b)
    sh = S.shell(m["el"], m["cl"])
    st, bl, tx, have = {}, {}, {}, []

    def take(arm):
        p = "%s/gen/%s/%s/%s/%s.json" % (NS, task, model, split, S.tok(arm))
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            g = json.load(f)
        k = g.get("contract") or {}
        if not (g.get("completed") and k.get("full_pass") and k.get("split") == split and k.get("model") == model
                and k.get("items_file_sha256") == S.sha_file(bp) and [r["id"] for r in g["rows"]] == want):
            raise SystemExit("%s is not a complete %s pass of %s" % (p, split, model))
        c = k["condition"]
        for r in g["rows"]:
            q = "%s:%s" % (task, r["id"])
            blank = bool(r["rule_abstain"]) or not r["text"].strip()
            st[(q, c)] = AM.verdict(r["text"], b[r["id"]]["gold"], not blank)
            bl[(q, c)] = blank
            tx[(q, c)] = r["text"]
        return c

    if take("x") is None:
        raise SystemExit("no no-write pass for %s %s %s" % (task, model, split))
    need = ["baseline_x"]
    c_alone = take("x:C%d.w.m0.25" % m["cl"])
    for s in STRENGTHS:
        arms = ["x:E%d.w.p%s" % (m["el"], s)] + ["x:E%d.w.p%s+C%d.w.m%s" % (m["el"], s, m["cl"], k) for k in KS]
        got = [take(a) for a in arms]
        if all(got):
            have.append(dash(s))
            need += got
    qs = ["%s:%s" % (task, i) for i in want]
    sp = S.NewSplit(m["el"], m["cl"], st, bl, need, qs=qs)
    return sp, fr.Texts(tx, tx), have, c_alone, st, bl


def net(s):
    return sum(S.vals(s))


def main():
    tasks = sys.argv[1:] or list(S.TASKS)
    M = S.models()
    for task in tasks:
        tot = {}
        print("\n# %s (alias match; tune 192 questions, test 500)" % task)
        for name in S.ORDER:
            m = M[name]
            try:
                tsp, ttx, th, tc, tst, tbl = load(task, name, m, "tune")
                xsp, xtx, xh, xc, xst, xbl = load(task, name, m, "test")
            except SystemExit as e:
                print("\n## %s: %s" % (name, e))
                continue
            have = [s for s in th if s in xh]
            if not have:
                print("\n## %s: no strength with all five arms on both splits yet" % name)
                continue
            base_t, base_x = S.system(tsp, ttx, ("base",)), S.system(xsp, xtx, ("base",))
            print("\n## %s: strengths %s | no write: tune net %+d (answers %d), test net %+d (answers %d)" % (
                name, " ".join(s.replace("-", ".") for s in have), net(base_t[0]), sum(base_t[1]), net(base_x[0]), sum(base_x[1])))
            if tc and xc:
                ct = [tst[(q, tc)] for q in tsp.qs]
                cx = [xst[(q, xc)] for q in xsp.qs]
                print("    single C 0.25: tune net %+d (answers %d) | test net %+d (answers %d)" % (
                    net(ct), sum(not tbl[(q, tc)] for q in tsp.qs), net(cx), sum(not xbl[(q, xc)] for q in xsp.qs)))
            rows = {}
            for s in have:
                cands = [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
                best = max(cands, key=lambda c: (net(S.system(tsp, ttx, c)[0]), -cands.index(c)))
                cands2 = [("T2", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
                best2 = max(cands2, key=lambda c: (net(S.system(tsp, ttx, c)[0]), -cands2.index(c)))
                e_t, e_x = S.system(tsp, ttx, ("E", s)), S.system(xsp, xtx, ("E", s))
                n_t, n_x = S.system(tsp, ttx, best), S.system(xsp, xtx, best)
                v_t, v_x = S.system(tsp, ttx, best2), S.system(xsp, xtx, best2)
                rows[s] = dict(best=best, best2=best2, e_t=e_t, e_x=e_x, n_t=n_t, n_x=n_x, v_t=v_t, v_x=v_x)
                print("    E %-4s tune: E %+4d (ans %3d) | N1 %+4d (ans %3d; caps E's %s / own %s) | no veto %+4d (caps %s / %s) || test: E %+4d (ans %3d) | N1 %+4d (ans %3d), N1 minus E %s | no veto %+4d, minus E %s" % (
                    s.replace("-", "."), net(e_t[0]), sum(e_t[1]), net(n_t[0]), sum(n_t[1]), best[2], best[3], net(v_t[0]), best2[2], best2[3],
                    net(e_x[0]), sum(e_x[1]), net(n_x[0]), sum(n_x[1]), S.ps.ci([a - b for a, b in zip(S.vals(n_x[0]), S.vals(e_x[0]))]),
                    net(v_x[0]), S.ps.ci([a - b for a, b in zip(S.vals(v_x[0]), S.vals(e_x[0]))])))
            sN = max(have, key=lambda s: (net(rows[s]["n_t"][0]), -have.index(s)))
            sE = max(have, key=lambda s: (net(rows[s]["e_t"][0]), -have.index(s)))
            sV = max(have, key=lambda s: (net(rows[s]["v_t"][0]), -have.index(s)))
            n1, e_same, e_best, nv = rows[sN]["n_x"][0], rows[sN]["e_x"][0], rows[sE]["e_x"][0], rows[sV]["v_x"][0]
            d = lambda a, b: [x - y for x, y in zip(S.vals(a), S.vals(b))]
            line = "    chosen on tune: N1 at E %s (test net %+d) | single E's own best strength %s (test net %+d)" % (
                sN.replace("-", "."), net(n1), sE.replace("-", "."), net(e_best))
            line += " || test: N1 minus no write %s | minus E at the same strength %s | minus E at its own best %s" % (
                S.ps.ci(d(n1, base_x[0])), S.ps.ci(d(n1, e_same)), S.ps.ci(d(n1, e_best)))
            if xc:
                line += " | minus single C %s" % S.ps.ci(d(n1, cx))
            print(line)
            print("    diagnostic, no veto chosen on tune: E %s, test net %+d | minus E at its own best %s | minus no write %s" % (
                sV.replace("-", "."), net(nv), S.ps.ci(d(nv, e_best)), S.ps.ci(d(nv, base_x[0]))))
            for key, v in (("N1 minus no write", d(n1, base_x[0])), ("N1 minus E at the same strength", d(n1, e_same)),
                           ("N1 minus E at its own best strength", d(n1, e_best)), ("no-veto N1 minus E at its own best strength", d(nv, e_best))):
                t = tot.setdefault(key, [])
                t.append(dict(zip(xsp.qs, v)))
        if tot and all(len(v) == len(S.ORDER) for v in tot.values()):
            print("\n## %s, five models together (each at its own tune-chosen strength; questions resampled jointly)" % task)
            for key, per in tot.items():
                common = sorted(set.intersection(*[set(p) for p in per]))
                vals = [[p[q] for q in common] for p in per]
                t_, lo, hi = S.joint_ci(vals)
                print("    %-46s %+d [%+d, %+d] on %d questions, %d of 5 models positive" % (key, t_, lo, hi, len(common), sum(sum(v) > 0 for v in vals)))


if __name__ == "__main__":
    main()
