# Iteration_01 analysis (Claude Fable 5.2, 2026-11-21), read-only, no model, no judge call.
# TriviaQA / PopQA with every write at the core rule "prompt" and E at each model's MAIN-TABLE layer
# (launchers/explore_20261120/ns_prompt.sh; answers in Iteration_01/newsets/gen_prompt/<task>/<model>/<split>/, the
# no-write pass copied from newsets/gen). Alias match (scripts/newsets/alias_match.py), so every question is scored.
# Two readings, both fixed in ns_prompt.sh before any answer was read:
#   F  frozen transfer: E strength = the model's main-table strength; N1's caps (E's blanks, own blanks) and the
#      strength of single C = the ones chosen on the TruthfulQA tune split at "prompt"
#      (results/write_positions/main_table_prompt_budget1_20261121.txt); nothing is chosen on these datasets.
#   D  dataset-tuned: the E strength and the two caps with the best N1 net on the dataset's own tune split (ties to the
#      lower strength, then to the earlier caps), single C at its best tune strength, and single E also at ITS OWN best
#      tune strength (the stronger single-direction baseline).
# Printed per model: the tune table, then F and D on the chosen split (test by default; --split test2 for the repeat);
# then the five models together (questions resampled jointly). A third block per model counts what the re-ask releases
# at the main-table strength (the table of summary section 12.2 at the new rule): for the questions answered without a
# write and blank under E, and for the model's own blanks, whether the first released answer is the no-write answer
# again or a new one, and how many of each are right.
# Every write file must record positions == "prompt"; a file of another rule stops the program.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/ns_prompt_table.py [--split test2] [TASK ...]
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
sys.path.insert(0, "Iteration_01/code/scripts/newsets")
import newsets_score as S     # noqa: E402
import final_records as fr    # noqa: E402
import newsets_gen as G
from types import SimpleNamespace
import alias_match as AM      # noqa: E402
import release_origin as ro   # noqa: E402

NS = S.NS
KS = ("0.25", "0.5", "1", "2")
LAD = ["0-25", "0-5", "1", "2"]
CAPS = [None] + LAD
# model key: (E layer, main-table strength, sweep, TruthfulQA caps at "prompt" (E's blanks, own blanks), TruthfulQA single C)
MAIN = {
    "llama3.1-8b": (16, "3.5", ("1", "2", "3.5", "5"), ("1", "2"), "0.25"),
    "qwen3.5-9b": (15, "2", ("0.5", "1", "2", "3"), ("0-25", "0-5"), "0.25"),
    "qwen3.8-27b": (28, "1.75", ("0.5", "1", "1.75", "3"), ("0-5", "1"), "0.5"),
    "gemma3-12b": (24, "2.5", ("1", "2.5", "4"), ("2", "2"), "0.5"),
    "gemma3-27b": (28, "1", ("0.5", "1", "2"), ("1", "2"), "0.25"),
}


def dash(s):
    return s.replace(".", "-")


def bank(task, split):
    p = "%s/bank/%s_bank_%s.jsonl" % (NS, task, "test2_20261120" if split == "test2" else "20261119")
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return {r["id"]: r for r in rows if r["split"] == split}, p


def load(task, model, m, split):
    """(Split, texts, strengths with all five arms, {C strength: condition}, states, blanks) at the prompt rule."""
    el, main_s, sweep, _, _ = MAIN[model]
    b, bp = bank(task, split)
    want = list(b)
    st, bl, tx, have, cs = {}, {}, {}, [], {}

    def take(arm):
        p = "%s/gen_prompt/%s/%s/%s/%s.json" % (NS, task, model, split, S.tok(arm))
        if not os.path.exists(p):
            return None
        args = SimpleNamespace(task=task, split=split, model=model, items=bp)
        g = G.load_generation(p, args, want)
        k = g["contract"]
        if k["arm"] != arm:
            raise SystemExit("%s: expected %s, found %s" % (p, arm, k["arm"]))
        if (":" in arm) != (k.get("positions") == "prompt") or (":" not in arm and k.get("positions") is not None):
            raise SystemExit("%s: positions %r do not fit the arm %s (write arms must be at \"prompt\")" % (p, k.get("positions"), arm))
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
    for k in KS:
        c = take("x:C%d.w.m%s" % (m["cl"], k))
        if c:
            cs[dash(k)] = c
    for s in sweep:
        arms = ["x:E%d.w.p%s" % (el, s)] + ["x:E%d.w.p%s+C%d.w.m%s" % (el, s, m["cl"], k) for k in KS]
        got = [take(a) for a in arms]
        if all(got):
            have.append(dash(s))
            need += got
    qs = ["%s:%s" % (task, i) for i in want]
    sp = S.NewSplit(el, m["cl"], st, bl, need, qs=qs)
    return sp, fr.Texts(tx, tx), have, cs, st, bl


def net(s):
    return sum(S.vals(s))


def d(a, b):
    return [x - y for x, y in zip(S.vals(a), S.vals(b))]


def released(sp, tx, s):
    """Counts for the re-ask at strength s (ladder up to 2): {origin: {kind: [right, wrong]}}."""
    out = {"E's": {"same": [0, 0], "new": [0, 0], "none": 0, "orig": [0, 0]}, "own": {"new": [0, 0], "none": 0}}
    e = sp.cE(s)
    for q in sp.qs:
        if not sp.blank[(q, e)]:
            continue
        own = sp.blank[(q, "baseline_x")]
        o = out["own" if own else "E's"]
        if not own:
            o["orig"][0 if sp.st[(q, "baseline_x")] == "R" else 1] += 1
        c = next((sp.cEC(s, k) for k in LAD if not sp.blank[(q, sp.cEC(s, k))]), None)
        if c is None:
            o["none"] += 1
            continue
        same = (not own) and ro.norm(tx.ec[(q, c)]) == ro.norm(tx.ec[(q, "baseline_x")])
        o["same" if same else "new"][0 if sp.st[(q, c)] == "R" else 1] += 1
    return out


def main():
    argv = sys.argv[1:]
    split = "test"
    if argv[:1] == ["--split"]:
        split, argv = argv[1], argv[2:]
    if split not in ("test", "test2"):
        raise SystemExit("--split test or test2")
    tasks = argv or list(S.TASKS)
    M = S.models()
    for task in tasks:
        tot = {}
        print("\n# %s at the prompt rule (alias match; tune 192 questions, %s 500)" % (task, split))
        for name in S.ORDER:
            m = M[name]
            el, main_s, sweep, tq_caps, tq_c = MAIN[name]
            try:
                tsp, ttx, th, tcs, tst, tbl = load(task, name, m, "tune")
                xsp, xtx, xh, xcs, xst, xbl = load(task, name, m, split)
            except SystemExit as e:
                print("\n## %s: %s" % (name, e))
                continue
            have = [s for s in th if s in xh]
            ms = dash(main_s)
            if ms not in have or dash(tq_c) not in tcs or dash(tq_c) not in xcs or len(tcs) < len(KS) or len(xcs) < len(KS):
                print("\n## %s: not complete yet (strengths on both splits: %s; single C on tune %d, on %s %d)" % (
                    name, " ".join(have), len(tcs), split, len(xcs)))
                continue
            base_t, base_x = S.system(tsp, ttx, ("base",)), S.system(xsp, xtx, ("base",))
            print("\n## %s, E at layer %d: strengths %s | no write: tune net %+d (answers %d), %s net %+d (answers %d)" % (
                name, el, " ".join(s.replace("-", ".") for s in have), net(base_t[0]), sum(base_t[1]), split, net(base_x[0]), sum(base_x[1])))
            ct = {k: [tst[(q, c)] for q in tsp.qs] for k, c in tcs.items()}
            cx = {k: [xst[(q, c)] for q in xsp.qs] for k, c in xcs.items()}
            print("    single C, tune / %s net: %s" % (split, " | ".join("%s: %+d / %+d (answers %d)" % (
                k.replace("-", "."), net(ct[k]), net(cx[k]), sum(not xbl[(q, xcs[k])] for q in xsp.qs)) for k in LAD if k in ct)))
            rows = {}
            for s in have:
                cands = [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
                best = max(cands, key=lambda c: (net(S.system(tsp, ttx, c)[0]), -cands.index(c)))
                e_t, e_x = S.system(tsp, ttx, ("E", s)), S.system(xsp, xtx, ("E", s))
                n_t, n_x = S.system(tsp, ttx, best), S.system(xsp, xtx, best)
                rows[s] = dict(best=best, e_t=e_t, e_x=e_x, n_t=n_t, n_x=n_x)
                print("    E %-4s tune: E %+4d (ans %3d) | N1 %+4d (ans %3d; caps E's %s / own %s) || %s: E %+4d (ans %3d) | N1 %+4d (ans %3d), N1 minus E %s" % (
                    s.replace("-", "."), net(e_t[0]), sum(e_t[1]), net(n_t[0]), sum(n_t[1]), best[2], best[3], split,
                    net(e_x[0]), sum(e_x[1]), net(n_x[0]), sum(n_x[1]), S.ps.ci(d(n_x[0], e_x[0]))))
            # F: frozen transfer from TruthfulQA
            f_cfg = ("N1", ms, tq_caps[0], tq_caps[1])
            f_n, f_e, f_c = S.system(xsp, xtx, f_cfg), rows[ms]["e_x"], cx[dash(tq_c)]
            print("    F (frozen from TruthfulQA: E %s, caps E's %s / own %s, single C %s): no write %+d | single C %+d | single E %+d (ans %d) | N1 %+d (ans %d) || N1 minus single E %s | minus single C %s | minus no write %s | single E minus no write %s" % (
                main_s, tq_caps[0], tq_caps[1], tq_c, net(base_x[0]), net(f_c), net(f_e[0]), sum(f_e[1]), net(f_n[0]), sum(f_n[1]),
                S.ps.ci(d(f_n[0], f_e[0])), S.ps.ci(d(f_n[0], f_c)), S.ps.ci(d(f_n[0], base_x[0])), S.ps.ci(d(f_e[0], base_x[0]))))
            # D: tuned on the dataset's own tune split. The choice is made among the strengths complete on TUNE (th),
            # whatever exists on the read split, so that it cannot move with the batch that is read (GPT review of
            # 2026-11-22); a choice without a complete pass on the read split is reported as pending.
            tune_net = {}
            for s in th:
                cands = [("N1", s, k1, k2) for k1 in CAPS for k2 in CAPS if not (k1 is None and k2 is None)]
                best_t = max(cands, key=lambda c: (net(S.system(tsp, ttx, c)[0]), -cands.index(c)))
                tune_net[s] = (net(S.system(tsp, ttx, best_t)[0]), net(S.system(tsp, ttx, ("E", s))[0]))
            sN = max(th, key=lambda s: (tune_net[s][0], -th.index(s)))
            sE = max(th, key=lambda s: (tune_net[s][1], -th.index(s)))
            f_rows = (("F: N1 minus single E", d(f_n[0], f_e[0])), ("F: N1 minus single C", d(f_n[0], f_c)), ("F: N1 minus no write", d(f_n[0], base_x[0])),
                      ("F: single E minus no write", d(f_e[0], base_x[0])))
            if sN not in have or sE not in have:
                print("    D: pending (chosen on tune: N1 at E %s, single E at %s; no complete %s pass of that choice yet)" % (
                    sN.replace("-", "."), sE.replace("-", "."), split))
                for key, v in f_rows:
                    tot.setdefault(key, []).append(dict(zip(xsp.qs, v)))
                continue
            kC = max([k for k in LAD if k in ct], key=lambda k: (net(ct[k]), -LAD.index(k)))
            d_n, d_e, d_eb, d_c = rows[sN]["n_x"][0], rows[sN]["e_x"][0], rows[sE]["e_x"][0], cx[kC]
            print("    D (chosen on this dataset's tune: N1 at E %s with caps E's %s / own %s; single E's own best %s; single C %s): single C %+d | single E same strength %+d, own best %+d | N1 %+d || N1 minus single E same strength %s | minus single E own best %s | minus single C %s | minus no write %s" % (
                sN.replace("-", "."), rows[sN]["best"][2], rows[sN]["best"][3], sE.replace("-", "."), kC.replace("-", "."),
                net(d_c), net(d_e), net(d_eb), net(d_n), S.ps.ci(d(d_n, d_e)), S.ps.ci(d(d_n, d_eb)), S.ps.ci(d(d_n, d_c)), S.ps.ci(d(d_n, base_x[0]))))
            r = released(xsp, xtx, ms)
            print("    re-ask at E %s on %s: answered without a write and blank under E: %d (original answer right %d / wrong %d); released as the original answer right %d / wrong %d, as a new answer right %d / wrong %d, not released %d | own blanks under E: released right %d / wrong %d, not released %d" % (
                main_s, split, sum(r["E's"]["orig"]), r["E's"]["orig"][0], r["E's"]["orig"][1], r["E's"]["same"][0], r["E's"]["same"][1],
                r["E's"]["new"][0], r["E's"]["new"][1], r["E's"]["none"], r["own"]["new"][0], r["own"]["new"][1], r["own"]["none"]))
            for key, v in f_rows + (
                    ("D: N1 minus single E, same strength", d(d_n, d_e)), ("D: N1 minus single E at its own best", d(d_n, d_eb)),
                    ("D: N1 minus single C", d(d_n, d_c)), ("D: N1 minus no write", d(d_n, base_x[0]))):
                tot.setdefault(key, []).append(dict(zip(xsp.qs, v)))
        if tot and all(len(v) == len(S.ORDER) for v in tot.values()):
            print("\n## %s, five models together on %s (questions resampled jointly)" % (task, split))
            for key, per in tot.items():
                common = sorted(set.intersection(*[set(p) for p in per]))
                vals = [[p[q] for q in common] for p in per]
                t_, lo, hi = S.joint_ci(vals)
                print("    %-40s %+d [%+d, %+d] on %d questions, %d of 5 models positive" % (key, t_, lo, hi, len(common), sum(sum(v) > 0 for v in vals)))
        elif tot:
            print("\n## %s: %d of %d models complete; no total yet" % (task, min(len(v) for v in tot.values()), len(S.ORDER)))


if __name__ == "__main__":
    main()
