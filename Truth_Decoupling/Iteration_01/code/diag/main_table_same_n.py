# Iteration_01 analysis (Claude Fable 5.2, 2026-11-19), read-only, no model, no judge call.
# The Dr's main comparison, with the frozen scheme N1 as the combination (asked for again by GPT, 2026-11-18 23:43,
# discussion/novelty_review/2026-11-18_n1_decision_for_claude.md): at the same number of answers, is E+C better than
# either single direction, and better than simply keeping the answers the model is most confident in?
# Free-form QA, test split, DeepSeek Pro, score = right - wrong. Settings are chosen on tune (Pro) exactly as before;
# nothing is re-chosen on test. For the two Gemma-3 models the full test split is used (confirm_fresh.py).
#   Part 1  every system at its own operating point: answers given (rule detector / empty, no correctness label), right,
#           wrong, hedge, net.
#   Part 2  at the SAME number N of answers as N1: selection systems that answer only their N most confident answers
#           (mean log-probability of the answer's own tokens under its own prompt and writes, ff_explore_ec_confidence.py;
#           ties by question order) and say "I don't know" elsewhere:
#             strong form  must-answer prompt + E at N1's strength (and at the other strength)
#             original     must-answer prompt, nothing written
#             weak form    plain prompt + C alone (each strength with at least N answers)
#           N1 minus each, paired bootstrap as policy_select2.ci (the selection is fixed on the full question set).
#   Part 3  the two must-answer selection systems at their OWN operating point: the confidence threshold with the best
#           net on tune (where the run has a tune confidence file), applied to test; and the best net any N could give on
#           test (an upper bound: it is chosen with the test labels).
#   Part 4  (2026-11-19) the arm the Dr named on 2026-11-15: plain prompt, E 2 and C 0.5 on every question, against the strong
#           form with the same number of answers.
# The confidence rows are checked to belong to the judged answers (sha1 of the answer text). Since 2026-11-19 every
# system's states and answer flags come from the per-question records of final_records.py (the same numbers as before).
# No prediction was written before this run; the earlier equal-N comparison (old E+C, counting right answers; for_Dr_review/
# 2026-11-11, section B.4) found the strong form undecided on Llama / Qwen3.5-9B / Qwen3.8-27B.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/main_table_same_n.py
import hashlib
import json
import sys

sys.path.insert(0, "Iteration_01/diag")
import policy_select2 as ps
import c_first as cf
import release_origin as ro
import final_records as fr

LAD, CAPS = ps.LAD, ro.CAPS
G3 = "Iteration_01/results/ff/gemma3/12b/e44/gemma3-12b"
G27 = "Iteration_01/results/ff/gemma3/27b/e44/gemma3-27b"
FULL = {"Gemma-3-12B": (G3 + "/eval_extra/judge_test_t25_test", G3 + "/judge_test_testD_E_L22"),
        "Gemma-3-27B": (G27 + "/eval_extra/judge_test_t31_test", G27 + "/judge_test_testD_E_L26")}
CONF = {  # confidence files (tune, test) of each run; None = this run has no such file
    "Llama-3.1-8B": (None, ps.LR + "/eval_extra/confidence_tll19.json"),
    "Qwen3.5-9B": (ps.NR + "/eval_extra/confidence_ec9b.json", ps.NR + "/eval_extra/confidence_t9b16.json"),
    "Qwen3.8-27B": (ps.QR + "/eval_extra/confidence_ec2733.json", ps.QR + "/eval_extra/confidence_t2733.json"),
    "Gemma-3-12B": (G3 + "/eval_extra/confidence_ec25.json", G3 + "/eval_extra/confidence_t25.json"),
    "Gemma-3-27B": (G27 + "/eval_extra/confidence_ec31.json", G27 + "/eval_extra/confidence_t31.json"),
}


class Data:
    def __init__(self, name, i):
        m = cf.MODELS[name]
        if i == 1 and name in FULL:
            m = dict(m, ec=(m["ec"][0], FULL[name][0]), d=(m["d"][0], FULL[name][1]), judge=(m["judge"][0], "pro"))
        self.m, self.i = m, i
        self.sp = ps.Split(m, i)
        self.tx = ro.texts(m["ec"][i])
        self.texts = fr.Texts.of(m, i)                          # raw answers: what each scheme hands in (final_records.py)
        self.dblank = ps.load(m["d"][i], m["judge"][i])[1]
        self.conf = None
        if CONF[name][i]:
            d = json.load(open(CONF[name][i], encoding="utf-8"))
            if d["split"] != ("tune", "test")[i] or d.get("partial"):
                raise SystemExit("%s is not a complete %s file" % (CONF[name][i], ("tune", "test")[i]))
            self.conf = {t: {r["pair_id"]: r for r in v["rows"]} for t, v in d["targets"].items()}
        self.checked = []

    def rows(self, target, work, cond):
        """Confidence rows of a target, after checking that they belong to the judged answers of (work, cond)."""
        rows = self.conf[target]
        raw = {}
        for l in open(work + "/judge2/cells.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["condition"] == cond:
                raw[r["pair_id"]] = r["prediction"]
        bad = [p for p in self.sp.qs if p not in rows or hashlib.sha1(raw[p].encode("utf-8")).hexdigest() != rows[p]["text_sha1"]]
        if bad:
            raise SystemExit("confidence rows of %s do not belong to the judged answers of %s %s: %d questions, e.g. %s" % (
                target, work, cond, len(bad), bad[:3]))
        self.checked.append(target)
        return rows


def handed_in(D, cfg):
    """(states, answered) of a system, both from the per-question records of what it hands in (final_records.py);
    answered = the answer handed in is not blank for the rule detector (nor empty)."""
    recs = fr.records(D.sp, cfg, D.texts)
    return fr.states(recs), fr.answered(recs, D.sp, D.dblank)


def plain(D, cfg):
    return handed_in(D, cfg)


def must(D, cond):
    cfg = {"baseline_conf": ("Dnone",), "baseline_conf_E2": ("DA",)}.get(cond)
    if cfg is None:
        cfg = next(("DE", s) for s in "12" if D.sp.dE(s) == cond)
    return handed_in(D, cfg)


def prompt_b(D, s):
    return handed_in(D, ("B", s))


def ranked(D, ans, rows, measure="mean_logp"):
    cand = [i for i, a in enumerate(ans) if a]
    return sorted(cand, key=lambda i: (-rows[D.sp.qs[i]][measure], i))


def keep(st, order, n):
    kept = set(order[:n])
    return ["X" if i not in kept else s for i, s in enumerate(st)], [i in kept for i in range(len(st))]


def net(st):
    return sum(x == "R" for x in st) - sum(x == "W" for x in st)


def val(st):
    return [(x == "R") - (x == "W") for x in st]


def line(label, st, ans):
    return "    %-62s answers %3d | right %3d wrong %3d hedge %2d | net %4d" % (
        label, sum(ans), sum(x == "R" for x in st), sum(x == "W" for x in st), sum(x == "H" for x in st), net(st))


def diff(a, b):
    return "net %s | right answers %s" % (ps.ci([x - y for x, y in zip(val(a), val(b))]), ps.ci([(x == "R") - (y == "R") for x, y in zip(a, b)]))


def best_cut(st, order):
    """(n, net) of the best prefix of the confidence order; ties go to fewer answers."""
    best, cur = (0, 0), 0
    for j, i in enumerate(order, 1):
        cur += (st[i] == "R") - (st[i] == "W")
        if cur > best[1]:
            best = (j, cur)
    return best


def report(name):
    tune, test = Data(name, 0), Data(name, 1)
    tsp, xsp = tune.sp, test.sp
    fam = {
        "no write": [("base",)],
        "single E": [("E", "1"), ("E", "2")],
        "single C": [("C", k) for k in LAD if k in tsp.cks and k in xsp.cks],
        "old E+C": [("E", s) for s in "12"] + [("EC", s, k) for s in "12" for k in LAD],
        "N1": [("E", s) for s in "12"] + [("N1", s, k1, k2) for s in "12" for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)],
    }
    pick = {f: max(c, key=lambda x: (ps.score(ro.states(tsp, tune.tx, x), 1), -c.index(x))) for f, c in fam.items() if c}
    sB = max("12", key=lambda s: (net(tsp.states(("B", s))), -int(s)))
    sD = max("12", key=lambda s: (net(tsp.states(("DE", s))), -int(s)))
    sN = pick["N1"][1]
    print("\n## %s: test %d questions with complete Pro verdicts (tune %d); judges %s" % (name, len(xsp.qs), len(tsp.qs), test.m["judge"]))
    print("  settings chosen on tune: " + "; ".join("%s = %s" % (f, ro.name_of(c)) for f, c in pick.items()) + "; prompt B at E %s; must-answer + E at E %s" % (sB, sD))
    print("  Part 1, each system at its own operating point")
    out = {f: plain(test, c) for f, c in pick.items()}
    out["prompt B"] = prompt_b(test, sB)
    out["must-answer, no write"] = must(test, "baseline_conf")
    out["must-answer + E"] = must(test, xsp.dE(sD))
    for f, (st, ans) in out.items():
        print(line(f, st, ans))
    n1, a1 = out["N1"]
    N = sum(a1)
    for f in ("no write", "single E", "single C", "old E+C", "prompt B", "must-answer, no write", "must-answer + E"):
        if f in out:
            print("    N1 minus %-24s %s" % (f, diff(n1, out[f][0])))
    print("  Part 2, the same number of answers as N1 (N = %d): keep the N most confident answers" % N)
    systems = []
    for s in (sN, "1" if sN == "2" else "2"):
        st, ans = must(test, xsp.dE(s))
        systems.append(("must-answer + E %s%s" % (s, " (strong form, N1's strength)" if s == sN else " (the other strength)"), st, ans,
                        test.rows("conf_E%s" % s, test.m["d"][1], xsp.dE(s))))
    st, ans = must(test, "baseline_conf")
    systems.append(("must-answer, no write (original model)", st, ans, test.rows("conf", test.m["d"][1], "baseline_conf")))
    for k in LAD:
        if k in xsp.cks and ("x_C%s" % ps.NAME[k]) in test.conf:
            st, ans = plain(test, ("C", k))
            systems.append(("plain + C %s alone (weak form)" % ps.NAME[k], st, ans, test.rows("x_C%s" % ps.NAME[k], test.m["ec"][1], xsp.cC(k))))
    st, ans = plain(test, ("base",))
    systems.append(("plain, no write", st, ans, test.rows("x", test.m["ec"][1], "baseline_x")))
    strong = None
    for label, st, ans, rows in systems:
        order = ranked(test, ans, rows)
        if len(order) < N:
            print("    %-62s only %d answers to choose from (fewer than N): not comparable" % (label, len(order)))
            continue
        ks, ka = keep(st, order, N)
        print(line(label + ", top N", ks, ka))
        print("    N1 minus it: %s" % diff(n1, ks))
        if "strong form" in label:
            strong = ks
            for measure in ("sum_logp", "first_logp", "min_logp"):  # other confidence measures of the same file (sensitivity)
                alt = keep(st, ranked(test, ans, rows, measure), N)[0]
                print("      the same with %-10s net %4d | N1 minus it: %s" % (measure + ":", net(alt), diff(n1, alt)))
    if name in FULL and strong is not None:
        old = set(ps.Split(cf.MODELS[name], 1).qs)              # the Dr's 98, the only test questions Pro had judged before N1 was frozen
        idx = [i for i, p in enumerate(xsp.qs) if p not in old]
        sub = lambda st: [st[i] for i in idx]
        print("  on the %d test questions outside the Dr's 100 (their Pro verdicts were made after N1 was frozen; N and the selection as above):" % len(idx))
        print("    N1 minus strong form: %s" % diff(sub(n1), sub(strong)))
        print("    N1 minus prompt B:    %s" % diff(sub(n1), sub(out["prompt B"][0])))
    print("  Part 3, the must-answer selection systems at their own operating point")
    for s, target, cond in ((sN, "conf_E%s" % sN, None), (None, "conf", "baseline_conf")):
        cond = cond or xsp.dE(s)
        label = "must-answer + E %s" % s if s else "must-answer, no write"
        st, ans = must(test, cond)
        rows = test.rows(target, test.m["d"][1], cond)
        order = ranked(test, ans, rows)
        bn, bv = best_cut(st, order)
        print("    %-30s best possible on test (chosen with the test labels, an upper bound): N %3d, net %4d" % (label, bn, bv))
        if tune.conf is None:
            print("    %-30s no tune confidence file for this run: no threshold chosen in advance" % label)
            continue
        tst, tans = must(tune, tsp.dE(s) if s else "baseline_conf")
        trows = tune.rows(target, tune.m["d"][0], tsp.dE(s) if s else "baseline_conf")
        torder = ranked(tune, tans, trows)
        tn, tv = best_cut(tst, torder)
        if tn == 0:
            print("    %-30s on tune no cut has a positive net: the system answers nothing" % label)
            continue
        tau = trows[tsp.qs[torder[tn - 1]]]["mean_logp"]
        n_test = sum(rows[xsp.qs[i]]["mean_logp"] >= tau for i in order)
        ks, ka = keep(st, order, n_test)
        print(line("%s, threshold from tune (tune: top %d of %d, net %d; mean logp >= %.3f)" % (label, tn, len(torder), tv, tau), ks, ka))
        print("    N1 minus it: %s" % diff(n1, ks))
    # Part 4 (added 2026-11-19): the arm the Dr named on 2026-11-15 for Qwen3.5-9B, printed for every model: plain prompt, E at
    # 2 and C at 0.5 written on every question; against the strong form with the same number of answers
    st4, ans4 = handed_in(test, ("ALL", "2", "0-5"))
    n4 = sum(ans4)
    st, ans = must(test, xsp.dE("2"))
    order = ranked(test, ans, test.rows("conf_E2", test.m["d"][1], xsp.dE("2")))
    print("  Part 4, the Dr's arm of 2026-11-15: plain prompt, E 2 and C 0.5 written on every question (N = %d)" % n4)
    print(line("plain, E 2 + C 0.5 on every question", st4, ans4))
    if len(order) >= n4:
        ks, ka = keep(st, order, n4)
        print(line("must-answer + E 2, top N", ks, ka))
        print("    the Dr's arm minus it: %s" % diff(st4, ks))
    else:
        print("    must-answer + E 2 has only %d answers to choose from: not comparable" % len(order))
    print("    N1 minus the Dr's arm (each at its own number of answers): %s" % diff(n1, st4))
    print("  confidence rows checked against the judged answers: test %s; tune %s" % (sorted(set(test.checked)), sorted(set(tune.checked))))
    other = {"N1 minus single E": out["single E"][0], "N1 minus old E+C": out["old E+C"][0], "N1 minus prompt B": out["prompt B"][0],
             "N1 minus must-answer + E (all answered)": out["must-answer + E"][0], "N1 minus strong form (same N)": strong}
    return {k: dict(zip(xsp.qs, [x - y for x, y in zip(val(n1), val(v))])) for k, v in other.items() if v is not None}


if __name__ == "__main__":
    MAIN = ("Llama-3.1-8B", "Qwen3.5-9B", "Qwen3.8-27B")        # the Dr (2026-11-15): free-form main text; Gemma in the appendix
    tot, tot3 = {}, {}
    for name in ro.FIVE:
        for k, d in report(name).items():
            for p, v in d.items():
                tot.setdefault(k, {})
                tot[k][p] = tot[k].get(p, 0) + v
                if name in MAIN:
                    tot3.setdefault(k, {})
                    tot3[k][p] = tot3[k].get(p, 0) + v
    # several models together: per question the differences are summed, and questions are resampled (one question carries
    # all its models), as GPT asked for joint statements (discussion/prompt_ladder/2026-11-18_gpt_review.md)
    for title, t in (("the five models together", tot), ("the three main-text models together (Llama-3.1-8B, Qwen3.5-9B, Qwen3.8-27B)", tot3)):
        print("\n## %s (net; per question the differences of the models are summed; questions resampled)" % title)
        for k, d in t.items():
            print("  %-42s %s (%d questions)" % (k, ps.ci([d[p] for p in sorted(d)]), len(d)))
