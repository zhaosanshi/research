# Iteration_01 analysis (Claude Fable 5.2, 2026-11-22), read-only, no model, no judge call. Exploration.
# The external baseline ITI (Li et al., NeurIPS 2023) for one model in this project's protocol: scripts/newsets/
# iti_baseline.py, launchers/explore_20261120/iti_grid.sh and iti_judge.sh. The rules below were written into
# discussion/work_queue/2026-11-22_queue.md (section 4) before any ITI answer was judged.
#   budget MODEL FIT   MMLU by option probability on the 14,042 items for every grid point (K heads, alpha) of the fit:
#                      accuracy change against the model itself on the SELECTION half (per_model_table.halves, the half
#                      on which the project's own strength is chosen) and on the report half. A grid point is within
#                      the budget when its selection-half change is not below -BUDGET points (default 1, the main
#                      table's budget). The K = 0 pass must agree with the project's usual no-write full-set file.
#   tune MODEL FIT     TruthfulQA tune split, DeepSeek Pro, score = right - wrong: every grid point within the budget
#                      (all of them must be judged), next to no write and the project's single E at the main-table
#                      setting; the choice = the highest net (ties: the smaller K, then the smaller alpha). With
#                      --write-choice it is saved to baselines/iti/<MODEL>/<FIT>/choice[_sp].json (never overwritten).
#   test MODEL FIT     TruthfulQA test split at the saved choice: ITI against no write, single E and N1 (main-table
#                      setting at the write rule "prompt"), paired intervals; MMLU of the chosen point on both halves.
# --all (added 2026-11-22 06 h, after the first three tune readings were seen; the reading most favourable to ITI):
#   tune --all   every grid point that has tune answers, whatever its MMLU change (shown when measured); the choice is
#                saved as choice_all[_sp].json;   test --all   the test split at that choice (work dir ..._all_test).
#   stems MODEL FIT    prints "K STEM" for every grid point within the budget (for launchers/.../iti_chain.sh).
#   unjudged MODEL FIT prints "K STEM" for every tune generation of the fit (and sigma) whose condition is in no
#                      judged tune work dir yet (for launchers/.../iti_free.sh).
#   point MODEL FIT STEM   prints the selection-half MMLU change of one grid point (file stem, e.g. k48_a15), for the
#                      launcher's early stop: once a K is far over the budget, its larger alphas are not run.
# Options: --sigma endq|pairs (which sigma of the fit; default endq, the official code's), --budget B.
# Work dirs: <run>/eval_extra/judge_tune_iti_<FIT>[_sp]_k<K>_tune (one per K) and judge_test_iti_<FIT>[_sp]_test.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/iti_table.py budget llama3.1-8b train_seed42
import glob
import json
import os
import sys

sys.path.insert(0, "Iteration_01/diag")
import main_table_prompt as MP
import posm_table as PT

ML, DM, S, PM, ps = MP.ML, MP.DM, MP.S, MP.PM, MP.ps
BASE = "Iteration_01/baselines/iti"
KEY = {PM.NG[d]: (d, layer, s) for d, layer, s in PT.SETS}     # model key: (display name, main-table E layer, strength)
val = PT.val


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def grid(model, fit, sigma):
    """[(K, alpha, file stem, {item id: correct})] of the fit's complete MMLU files, by K then alpha."""
    out = []
    for p in glob.glob("%s/%s/%s/mmlu/k*_a*.json" % (BASE, model, fit)):
        stem = os.path.basename(p)[:-5]
        if stem.endswith("_sp") != (sigma == "pairs"):
            continue
        d = read(p)
        st = d["setting"]
        if d.get("completed") is not True or d["model"] != model or st["sigma"] != sigma or d["n"] != 14042:
            raise SystemExit("%s is not a complete full-set MMLU pass of this model and sigma" % p)
        out.append((st["k"], st["alpha"], stem, {r["id"]: r["pick"] == r["gold"] for r in d["rows"]}))
    return sorted(out, key=lambda t: (t[0], t[1]))


def budget_rows(model, fit, sigma, budget, show=True):
    display, layer, _ = KEY[model]
    d0 = read("%s/%s/mmlu_k0.json" % (BASE, model))
    if d0.get("completed") is not True or d0["n"] != 14042 or d0["setting"] != {"k": 0}:
        raise SystemExit("the K = 0 pass is not a complete full-set pass")
    base = {r["id"]: r["pick"] == r["gold"] for r in d0["rows"]}
    usual = PM.mmlu(display, layer)["0"]
    if set(usual) != set(base):
        raise SystemExit("the K = 0 pass and the usual no-write file cover different items")
    differ = sum(usual[i] != base[i] for i in base)
    if differ > 0.001 * len(base):
        raise SystemExit("the K = 0 pass disagrees with the usual no-write picks on %d items" % differ)
    dev, rep = PM.halves(list(usual))
    rows = []
    for k, alpha, stem, cor in grid(model, fit, sigma):
        if set(cor) != set(base):
            raise SystemExit("%s covers other items" % stem)
        dd = PM.boot([cor[i] - base[i] for i in dev])
        dr = PM.boot([cor[i] - base[i] for i in rep])
        rows.append({"k": k, "alpha": alpha, "stem": stem, "dev": dd, "rep": dr, "within": dd[0] >= -budget})
    if show:
        print("# ITI, %s, fit %s, sigma %s: MMLU by option probability, 14,042 items; no intervention %.2f%% (selection half %.2f%%, report half %.2f%%); the K = 0 pass differs from the usual no-write file on %d items" % (
            model, fit, sigma, 100.0 * sum(base.values()) / len(base), 100.0 * sum(base[i] for i in dev) / len(dev),
            100.0 * sum(base[i] for i in rep) / len(rep), differ))
        print("# change in points against no intervention [95%% interval, items resampled]; within = selection-half change not below -%g" % budget)
        for r in rows:
            print("    K %3d alpha %5g | selection half %+6.2f [%+.2f, %+.2f] | report half %+6.2f [%+.2f, %+.2f] | %s" % (
                r["k"], r["alpha"], r["dev"][0], r["dev"][1], r["dev"][2], r["rep"][0], r["rep"][1], r["rep"][2],
                "within the budget" if r["within"] else "over the budget"))
    return rows


def point_change(model, fit, stem):
    """The selection-half MMLU change of one grid point, reading only the K = 0 pass and that point's file (the
    launcher's early stop calls this after every pass; budget_rows reads every file of the model and takes minutes).
    The halves are those of budget_rows: PM.halves depends only on the set of item ids, and budget mode checks that the
    K = 0 pass and the usual no-write file have the same set."""
    d0 = read("%s/%s/mmlu_k0.json" % (BASE, model))
    if d0.get("completed") is not True or d0["n"] != 14042 or d0["setting"] != {"k": 0}:
        raise SystemExit("the K = 0 pass is not a complete full-set pass")
    base = {r["id"]: r["pick"] == r["gold"] for r in d0["rows"]}
    d = read("%s/%s/%s/mmlu/%s.json" % (BASE, model, fit, stem))
    st = d["setting"]
    if (d.get("completed") is not True or d["model"] != model or d["n"] != 14042
            or st["sigma"] != ("pairs" if stem.endswith("_sp") else "endq")):
        raise SystemExit("%s is not a complete full-set MMLU pass of this model and sigma" % stem)
    cor = {r["id"]: r["pick"] == r["gold"] for r in d["rows"]}
    if set(cor) != set(base):
        raise SystemExit("%s covers other items" % stem)
    dev, _ = PM.halves(list(base))
    return PM.boot([cor[i] - base[i] for i in dev])[0]


def unjudged(model, fit, sigma):
    display, _, _ = KEY[model]
    _, run, _ = MP.usual(display, 0)
    tag = "iti_%s%s" % (fit, "_sp" if sigma == "pairs" else "")
    judged = set()
    for w in sorted(glob.glob("%s/eval_extra/judge_tune_%s_k*_tune" % (run, tag))):
        if MP.has(w):
            judged |= set(read(w + "/judge_input.json")["conditions"])
    out = []
    for p in glob.glob("%s/%s/%s/gen/tune_k*_a*.json" % (BASE, model, fit)):
        stem = os.path.basename(p)[len("tune_"):-5]
        if stem.endswith("_sp") != (sigma == "pairs"):
            continue
        g = read(p)
        if g.get("completed") is not True:
            raise SystemExit("%s is not complete" % p)
        if g["condition"] not in judged:
            out.append((g["setting"]["k"], g["setting"]["alpha"], stem))
    return sorted(out)


def condition_of(model, fit, split, stem):
    g = read("%s/%s/%s/gen/%s_%s.json" % (BASE, model, fit, split, stem))
    if g.get("completed") is not True or g["model"] != model or g["split"] != split:
        raise SystemExit("the %s generation of %s is not complete" % (split, stem))
    return g["condition"], g


def tune_table(model, fit, sigma, budget, write, every=False):
    display, layer, s = KEY[model]
    measured = budget_rows(model, fit, sigma, budget, show=False)
    rows = [r for r in measured if r["within"]]
    if every:
        mm = {r["stem"]: r for r in measured}
        rows = []
        for p in glob.glob("%s/%s/%s/gen/tune_k*_a*.json" % (BASE, model, fit)):
            stem = os.path.basename(p)[len("tune_"):-5]
            if stem.endswith("_sp") != (sigma == "pairs"):
                continue
            st0 = read(p)["setting"]
            rows.append(mm.get(stem) or {"k": st0["k"], "alpha": st0["alpha"], "stem": stem, "dev": None, "within": None})
        rows.sort(key=lambda r: (r["k"], r["alpha"]))
    if not rows:
        raise SystemExit("no grid point within the budget")
    T, why = MP.prompt_split(display, layer, 0, s)
    if T is None:
        raise SystemExit("the project's own tune split at the prompt rule is not complete (%s)" % why)
    tsp, ttx, _, qs = T
    _, run, _ = MP.usual(display, 0)
    st, bl = {}, {}
    tag = "iti_%s%s" % (fit, "_sp" if sigma == "pairs" else "")
    for w in sorted(glob.glob("%s/eval_extra/judge_tune_%s_k*_tune" % (run, tag))):
        if not MP.has(w):
            continue
        st1, bl1 = ps.load(w, "pro")
        st.update(st1)
        bl.update(bl1)
    base = S.system(tsp, ttx, ("base",))[0]
    e = S.system(tsp, ttx, ("E", s))
    print("# ITI, %s, fit %s, sigma %s: TruthfulQA tune split, %d questions, DeepSeek Pro, score = right - wrong; %s" % (
        model, fit, sigma, len(qs), "every grid point with tune answers, whatever its MMLU change" if every
        else "grid points within the MMLU budget of %g point(s)" % budget))
    print("    no write net %+d | the project's single E (layer %d, strength %s, prompt rule) net %+d (answers %d)" % (
        sum(map(val, base)), layer, DM.show(s), sum(map(val, e[0])), sum(e[1])))
    table, missing = [], []
    for r in rows:
        cond, _ = condition_of(model, fit, "tune", r["stem"])
        if any(st.get((q, cond), "?") == "?" for q in qs):
            missing.append(cond)
            continue
        net = sum(val(st[(q, cond)]) for q in qs)
        ans = sum(not bl[(q, cond)] for q in qs)
        right = sum(st[(q, cond)] == "R" for q in qs)
        wrong = sum(st[(q, cond)] == "W" for q in qs)
        table.append((net, r["k"], r["alpha"], r["stem"], cond, ans))
        print("    K %3d alpha %5g | net %+4d | answers %3d | right %3d | wrong %3d | MMLU selection half %s" % (
            r["k"], r["alpha"], net, ans, right, wrong, "not measured" if r["dev"] is None else "%+.2f%s" % (
                r["dev"][0], "" if r["within"] else " (over the budget)")))
    if missing:
        raise SystemExit("not judged yet: %s" % " ".join(missing))
    best = max(table, key=lambda t: (t[0], -t[1], -t[2]))
    print("    chosen on tune: K %d, alpha %g (net %+d, answers %d)" % (best[1], best[2], best[0], best[5]))
    if write:
        path = "%s/%s/%s/choice%s%s.json" % (BASE, model, fit, "_all" if every else "", "_sp" if sigma == "pairs" else "")
        with open(path, "x", encoding="utf-8") as f:
            json.dump({"model": model, "fit": fit, "sigma": sigma, "budget": None if every else budget, "k": best[1], "alpha": best[2],
                       "stem": best[3], "tune_condition": best[4], "tune_net": best[0], "tune_questions": len(qs),
                       "candidates": [[t[1], t[2], t[0]] for t in table]}, f, indent=1)
        print("    choice written to %s" % path)


def test_table(model, fit, sigma, every=False):
    display, layer, s = KEY[model]
    ch = read("%s/%s/%s/choice%s%s.json" % (BASE, model, fit, "_all" if every else "", "_sp" if sigma == "pairs" else ""))
    if ch["model"] != model or ch["fit"] != fit or ch["sigma"] != sigma or (ch["budget"] is None) != every:
        raise SystemExit("the choice file belongs to another setting")
    X, why = MP.prompt_split(display, layer, 1, s)
    T, why_t = MP.prompt_split(display, layer, 0, s)
    if X is None or T is None:
        raise SystemExit("the project's own split at the prompt rule is not complete (%s)" % (why or why_t))
    xsp, xtx, _, qs = X
    tsp, ttx, _, _ = T
    _, run, _ = MP.usual(display, 1)
    w = MP.work(run, "test", "iti_%s%s%s" % (fit, "_sp" if sigma == "pairs" else "", "_all" if every else ""))
    if not MP.has(w):
        raise SystemExit("no Pro verdicts yet in %s" % os.path.basename(w))
    cond, g = condition_of(model, fit, "test", ch["stem"])
    if g["setting"]["k"] != ch["k"] or g["setting"]["alpha"] != ch["alpha"] or g["setting"]["sigma"] != sigma:
        raise SystemExit("the test generation is not the chosen setting")
    st, bl = ps.load(w, "pro")
    if any(st.get((q, cond), "?") == "?" for q in qs):
        raise SystemExit("incomplete verdicts for %s" % cond)
    iti = [st[(q, cond)] for q in qs]
    base = S.system(xsp, xtx, ("base",))[0]
    e = S.system(xsp, xtx, ("E", s))
    n1 = S.system(xsp, xtx, ML.n1_pick(tsp, ttx, s))
    d = lambda a, b: [val(x) - val(y) for x, y in zip(a, b)]  # noqa: E731
    print("# ITI, %s, fit %s, sigma %s: TruthfulQA test split, %d questions, DeepSeek Pro, score = right - wrong" % (model, fit, sigma, len(qs)))
    print("    chosen on tune%s: K %d, alpha %g (tune net %+d); positions: the last prompt token and every generated token" % (
        " among all grid points, whatever the MMLU change" if every else " within the MMLU budget", ch["k"], ch["alpha"], ch["tune_net"]))
    print("    no write net %+d | ITI net %+d (answers %d, right %d, wrong %d) | single E (layer %d, strength %s, prompt rule) net %+d (answers %d) | N1 net %+d (answers %d)" % (
        sum(map(val, base)), sum(map(val, iti)), sum(not bl[(q, cond)] for q in qs), sum(x == "R" for x in iti), sum(x == "W" for x in iti),
        layer, DM.show(s), sum(map(val, e[0])), sum(e[1]), sum(map(val, n1[0])), sum(n1[1])))
    print("    ITI minus no write %s | single E minus ITI %s | N1 minus ITI %s | single E minus no write %s" % (
        ps.ci(d(iti, base)), ps.ci(d(e[0], iti)), ps.ci(d(n1[0], iti)), ps.ci(d(e[0], base))))
    for r in budget_rows(model, fit, sigma, 1.0 if every else ch["budget"], show=False):
        if (r["k"], r["alpha"]) == (ch["k"], ch["alpha"]):
            print("    MMLU of the chosen point: selection half %+.2f [%+.2f, %+.2f], report half %+.2f [%+.2f, %+.2f]" % (r["dev"] + r["rep"]))


def main():
    argv = sys.argv[1:]
    opt = {"--sigma": "endq", "--budget": "1"}
    write = every = False
    rest = []
    while argv:
        if argv[0] in opt:
            opt[argv[0]], argv = argv[1], argv[2:]
        elif argv[0] == "--write-choice":
            write, argv = True, argv[1:]
        elif argv[0] == "--all":
            every, argv = True, argv[1:]
        else:
            rest.append(argv[0])
            argv = argv[1:]
    if rest[:1] == ["point"] and len(rest) == 4 and rest[1] in KEY:
        try:
            print("%.3f" % point_change(rest[1], rest[2], rest[3]))
        except FileNotFoundError:
            raise SystemExit("no complete MMLU pass for %s" % rest[3])
        return
    if rest[:1] == ["unjudged"] and len(rest) == 3 and rest[1] in KEY and opt["--sigma"] in ("endq", "pairs"):
        for k, _, stem in unjudged(rest[1], rest[2], opt["--sigma"]):
            print(k, stem)
        return
    if rest[:1] == ["stems"] and len(rest) == 3 and rest[1] in KEY and opt["--sigma"] in ("endq", "pairs"):
        for r in budget_rows(rest[1], rest[2], opt["--sigma"], float(opt["--budget"]), show=False):
            if r["within"]:
                print(r["k"], r["stem"])
        return
    if len(rest) != 3 or rest[0] not in ("budget", "tune", "test") or rest[1] not in KEY or opt["--sigma"] not in ("endq", "pairs"):
        raise SystemExit("usage: iti_table.py budget|tune|test MODEL FIT [--sigma endq|pairs] [--budget B] [--write-choice]")
    mode, model, fit = rest
    if mode == "budget":
        budget_rows(model, fit, opt["--sigma"], float(opt["--budget"]))
    elif mode == "tune":
        tune_table(model, fit, opt["--sigma"], float(opt["--budget"]), write, every)
    else:
        test_table(model, fit, opt["--sigma"], every)


if __name__ == "__main__":
    main()
