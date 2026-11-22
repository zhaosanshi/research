# Iteration_01 analysis (Claude Opus 5.5, 2026-11-22), read-only, no model, no judge call. Exploration.
# The external baseline CAA (Rimsky / Panickssery et al., ACL 2024), adapted to this project's protocol
# (scripts/newsets/caa_vector.py; launchers/explore_20261120/caa_grid.sh, caa_judge.sh, caa_chain.sh; plan
# discussion/work_queue/2026-11-22_queue_f.md sections 1, 7 and 8). Same structure and rules as diag/iti_table.py; its
# verdict helpers (merged_verdicts, declined_cells) and the project's tables (main_table_prompt, per_model_table) are
# imported, not copied. A grid point is (layer, multiplier): the CAA vector of that layer times the multiplier, kind md,
# position rule gen, read from the mirror run directory baselines/caa/<MODEL>/run_caa/.
#   budget MODEL      MMLU by option probability on the 14,042 items (caa_grid.sh: one pass per layer with the arms x and
#                     x:E<l>.md.p<m>@gen): change against that pass's own no-write arm on the SELECTION half and the
#                     report half (per_model_table.halves, as ITI); within the budget = selection-half change not below
#                     -BUDGET (default 1). Every no-write arm must agree with the usual no-write full-set file (0.1 %).
#   tune MODEL        TruthfulQA tune split, DeepSeek Pro, score = right - wrong, every grid point within the budget that
#                     the early stop reached; choice = the highest net, ties to the lower layer, then to the smaller
#                     multiplier. --all: every point the early stop reached, whatever its MMLU change (printed as "the best
#                     among the measured points": points after a layer's early stop have no answers and are not part of
#                     it). --layers "L ..." --mults "M ..." declare the grid: every declared point needs its MMLU reading,
#                     and every point the reading needs (before the early stop of its layer; STOP = 3, caa_grid.sh's
#                     rule) needs its tune answers, or the table refuses. Judge-declined answers as in iti_table.py: such a
#                     point has no net, is never chosen, and the table refuses unless it stays below the chosen point even
#                     with every declined answer counted as right. --write-choice saves choice[_all].json (never
#                     overwritten).
#   test MODEL        TruthfulQA test split at the saved choice: CAA against no write, single E and N1 (main-table setting
#                     at the write rule "prompt"), paired intervals; the equal-count ranges (both systems keep exactly
#                     min(A, B) of their issued answers, the rest scores 0; the lowest and highest possible difference,
#                     the reading of discussion/validation_dataset/2026-11-22_iti_replay_b.py, rewritten here); the MMLU
#                     of the chosen point on both halves. --all reads the choice among all measured points.
#   neg MODEL         the direction check of plan section 7 at the chosen layer(s): multiplier -1 on tune next to no write
#                     and the chosen point, and the multiplier-0 check on the whole tune split (caa_grid.sh MODE=zero).
# Work dirs (in run_caa/eval_extra/): judge_tune_caa_L<l>_<TAG>_tune (one per layer), judge_tune_caa_neg_L<l>_<TAG>_tune,
# judge_test_caa_main_test and judge_test_caa_all_test.
# usage (host, from /data/sanshizhao/Truth_Decoupling): python3 -B Iteration_01/diag/caa_table.py budget llama3.1-8b
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, "Iteration_01/diag")
import iti_table as IT

MP, PM, S, ps, ML, DM = IT.MP, IT.PM, IT.S, IT.ps, IT.ML, IT.DM
KEY, val = IT.KEY, IT.val
BASE = "Iteration_01/baselines/caa"
STOP = 3.0
N_MMLU = 14042
ARM_RE = re.compile(r"^x:E(\d+)\.md\.p([0-9]+(?:\.[0-9]+)?)@gen$")


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mstr(m):
    return "%g" % m


def cond_name(layer, m):
    """The engine's single-write condition name (e44_eval_extra.condition_of)."""
    return "E_L%d_x-md-%s%s" % (layer, "p" if m > 0 else "m", mstr(abs(m)).replace(".", "-"))


def arm_of(layer, m):
    return "x:E%d.md.%s%s" % (layer, "p" if m > 0 else "m", mstr(abs(m)))


class Mirror:
    """The model's CAA files: vectors meta, mirror run directory, its listing; every cross-reference checked."""

    def __init__(self, model):
        self.model = model
        self.dir = "%s/%s" % (BASE, model)
        cfg = read(self.dir + "/models_run_caa.json")
        self.run = cfg["models"][model]["run_dir"]
        self.listing = read(self.run + "/MIRROR.json")
        self.meta = read(self.dir + "/vectors.pt.json")
        if self.listing.get("kind") != "caa" or self.listing.get("model") != model or self.meta.get("model") != model:
            raise SystemExit("the mirror run directory or the vectors belong to another model or kind")
        if self.listing.get("vectors_file_sha256") != sha_file(self.dir + "/vectors.pt"):
            raise SystemExit("the mirror records were not made from the vectors file next to them")
        self.norms = self.meta["norms"]

    def gen_path(self, split, layer, m):
        return "%s/eval_extra/gen/%s-at-gen_%s.json" % (self.run, arm_of(layer, m).replace(":", "--").replace(".", "-"), split)

    def amount_ok(self, layer, m, amount):
        want = m * self.norms[layer]
        return abs(amount - want) <= 1e-5 * max(1.0, abs(want))

    def generation(self, split, layer, m):
        """(condition, generation file) of one point, after checking that it is the CAA write of that point."""
        p = self.gen_path(split, layer, m)
        g = read(p)
        parts = g.get("parts") or []
        if (g.get("completed") is not True or g.get("partial") is not False or g.get("split") != split
                or g.get("positions") != "gen" or g.get("arm") != arm_of(layer, m) or len(parts) != 1):
            raise SystemExit("%s is not a complete %s generation of the point at the rule gen" % (os.path.basename(p), split))
        part = parts[0]
        rec = self.listing["layers"].get(str(layer)) or {}
        if (part.get("family") != "E" or part.get("kind") != "md" or part.get("layer") != layer or part.get("beta") != m
                or part.get("direction_run") != os.path.basename(self.run.rstrip("/"))
                or part.get("direction_file_sha256") != rec.get("file_sha256") or not self.amount_ok(layer, m, part["amount"])):
            raise SystemExit("%s does not write the CAA vector of layer %d times %g" % (os.path.basename(p), layer, m))
        return cond_name(layer, m), g


def usual_mmlu(model):
    display, layer, _ = KEY[model]
    return PM.mmlu(display, layer)["0"]


def budget_rows(model, budget, show=True):
    mi = Mirror(model)
    usual = usual_mmlu(model)
    dev, rep = PM.halves(list(usual))
    rows, notes = [], []
    for p in sorted(glob.glob("%s/mmlu/L*.json" % mi.dir)):
        filename = os.path.basename(p)
        if re.fullmatch(r"L[0-9]+\.receipt\.json", filename):
            continue
        match = re.fullmatch(r"L([0-9]+)\.json", filename)
        if match is None:
            raise SystemExit("unexpected MMLU file: " + filename)
        layer = int(match.group(1))
        receipt = read(p[:-5] + ".receipt.json")
        if receipt != {"direction_sha256": mi.listing["layers"][str(layer)]["file_sha256"],
                       "vectors_sha256": mi.listing["vectors_file_sha256"], "output_sha256": sha_file(p)}:
            raise SystemExit("MMLU receipt does not match this mirror and result: " + filename)
        d = read(p)
        if d.get("protocol") != "cap_mcq_sweep_v1" or d.get("model") != model or d.get("n") != N_MMLU or d.get("skipped"):
            raise SystemExit("%s is not a complete full-set MMLU pass of this model" % p)
        base = {r["id"]: r["pick"] == r["gold"] for r in d["arms"]["x"]["rows"]}
        if set(base) != set(usual):
            raise SystemExit("%s covers other items than the usual no-write file" % p)
        differ = sum(base[i] != usual[i] for i in base)
        if differ > 0.001 * len(base):
            raise SystemExit("%s: its no-write arm disagrees with the usual no-write file on %d items" % (p, differ))
        notes.append((layer, differ))
        for arm, a in d["arms"].items():
            if arm == "x":
                continue
            mm = ARM_RE.match(arm)
            if not mm or int(mm.group(1)) != layer:
                raise SystemExit("%s holds an arm %s of another layer or form" % (p, arm))
            m = float(mm.group(2))
            w = a.get("writes") or []
            if (a.get("positions") != "gen" or len(w) != 1 or w[0]["kind"] != "md" or w[0]["layer"] != layer
                    or w[0]["beta"] != m or not mi.amount_ok(layer, m, w[0]["amount"])):
                raise SystemExit("%s: %s is not the CAA write of layer %d at the rule gen" % (p, arm, layer))
            cor = {r["id"]: r["pick"] == r["gold"] for r in a["rows"]}
            if set(cor) != set(base):
                raise SystemExit("%s: %s covers other items" % (p, arm))
            dd = PM.boot([cor[i] - base[i] for i in dev])
            dr = PM.boot([cor[i] - base[i] for i in rep])
            rows.append({"layer": layer, "mult": m, "dev": dd, "rep": dr, "within": dd[0] >= -budget})
    rows.sort(key=lambda r: (r["layer"], r["mult"]))
    reached(rows)
    if show:
        print("# CAA, %s: MMLU by option probability, %d items; no write %.2f%% (selection half %.2f%%, report half %.2f%%); "
              "each layer's pass carries its own no-write arm (items that differ from the usual no-write file: %s)" % (
                  model, len(usual), 100.0 * sum(usual.values()) / len(usual), 100.0 * sum(usual[i] for i in dev) / len(dev),
                  100.0 * sum(usual[i] for i in rep) / len(rep), ", ".join("L%d %d" % t for t in notes) or "no file"))
        print("# change in points against no write [95%% interval, items resampled]; within = selection-half change not below -%g; "
              "stopped = after the layer's first point below -%g (no tune answers)" % (budget, STOP))
        for r in rows:
            print("    L%-3d x %-5s | selection half %+6.2f [%+.2f, %+.2f] | report half %+6.2f [%+.2f, %+.2f] | %s%s" % (
                r["layer"], mstr(r["mult"]), r["dev"][0], r["dev"][1], r["dev"][2], r["rep"][0], r["rep"][1], r["rep"][2],
                "within the budget" if r["within"] else "over the budget", "" if r["reached"] else ", stopped"))
    return rows


def reached(rows):
    """Mark r["reached"]: within a layer, multipliers in increasing order, False from the first point whose selection-half
    change is below -STOP on (caa_grid.sh generates no tune answers for those)."""
    by = {}
    for r in rows:
        by.setdefault(r["layer"], []).append(r)
    for rs in by.values():
        stop = False
        for r in sorted(rs, key=lambda r: r["mult"]):
            stop = stop or r["dev"][0] < -STOP
            r["reached"] = not stop


def grid_complete(mi, measured, layers, mults, every):
    have = {(r["layer"], r["mult"]): r for r in measured}
    missing = []
    for l in layers:
        for m in sorted(mults):
            r = have.get((l, m))
            if r is None:
                missing.append("mmlu L%d x %s" % (l, mstr(m)))
                continue
            if r["reached"] and (every or r["within"]) and not os.path.exists(mi.gen_path("tune", l, m)):
                missing.append("tune L%d x %s" % (l, mstr(m)))
    if missing:
        raise SystemExit("the declared grid is not complete: %s" % " ".join(missing))


def verify_work(mi, work, judged=True):
    """Bind the layer's scored cells to the current CAA generations before using its scores."""
    exp_path = os.path.join(work, "judge_input.json")
    exp = read(exp_path)
    source = exp["sources"]
    if exp["model"] != mi.model or not exp["completed"] or exp["identity_sha256"] != mi.meta["identity_sha256"]:
        raise SystemExit("judge input belongs to a different CAA model or identity: " + work)
    files = source["gen"]
    if not files or len(set(files)) != len(files):
        raise SystemExit("empty or repeated source files in " + work)
    for filename in files:
        if filename != os.path.basename(filename):
            raise SystemExit("unexpected generation basename in " + work)
        path = os.path.join(mi.run, "eval_extra", "gen", filename)
        digests = [h for p,h in source["sha256"].items() if os.path.basename(p) == filename]
        if digests != [sha_file(path)]:
            raise SystemExit("judge input is bound to an old generation: " + filename)
    folder = os.path.join(work, "judge2")
    meta = read(os.path.join(folder, "meta.json"))
    gen_sha = sha_file(exp_path)
    cells_sha = sha_file(os.path.join(folder, "cells.jsonl"))
    if meta["generation_sha256"] != gen_sha or meta["cells_sha256"] != cells_sha:
        raise SystemExit("judge input and generated cells differ in " + work)
    if judged:
        manifest = read(os.path.join(folder, "single_pro_manifest.json"))
        if (manifest["judge"] != "pro" or manifest["generation_sha256"] != gen_sha or
                manifest["rows_sha256"] != sha_file(os.path.join(folder, "single_pro.jsonl")) or
                manifest["sources"]["cells.jsonl"] != cells_sha):
            raise SystemExit("single-judge verdicts differ from the exported cells in " + work)


def tune_works(mi):
    works = sorted(glob.glob("%s/eval_extra/judge_tune_caa_L*_tune" % mi.run))
    for w in works:
        if MP.has(w):
            verify_work(mi, w)
    return works


def tune_table(model, budget, write, every=False, layers=None, mults=None):
    display, elayer, s = KEY[model]
    mi = Mirror(model)
    measured = budget_rows(model, budget, show=False)
    if bool(layers) != bool(mults):
        raise SystemExit("declare both --layers and --mults or neither")
    if layers:
        grid_complete(mi, measured, layers, mults, every)
        measured = [r for r in measured if r["layer"] in layers and r["mult"] in mults]
    rows = [r for r in measured if r["reached"] and (every or r["within"])]
    if not rows:
        raise SystemExit("no eligible points in the requested grid")
    T, why = MP.prompt_split(display, elayer, 0, s)
    if T is None:
        raise SystemExit("the project's own tune split at the prompt rule is not complete (%s)" % why)
    tsp, ttx, _, qs = T
    works = tune_works(mi)
    st, bl = IT.merged_verdicts(model, works)
    refused = IT.declined_cells(works)
    base = S.system(tsp, ttx, ("base",))[0]
    e = S.system(tsp, ttx, ("E", s))
    print("# CAA, %s: TruthfulQA tune split, %d questions, DeepSeek Pro, score = right - wrong; %s" % (
        model, len(qs), "the best among the measured points (every point the early stop reached), whatever its MMLU change"
        if every else "points within the MMLU budget of %g point(s)" % budget))
    print("    no write net %+d | the project's single E (layer %d, strength %s, prompt rule) net %+d (answers %d)" % (
        sum(map(val, base)), elayer, DM.show(s), sum(map(val, e[0])), sum(e[1])))
    table, missing, apart = [], [], []
    for r in rows:
        cond, g = mi.generation("tune", r["layer"], r["mult"])
        mmlu = "%+.2f%s" % (r["dev"][0], "" if r["within"] else " (over the budget)")
        unknown = [q for q in qs if st.get((q, cond), "?") == "?"]
        if any((q, cond) not in refused for q in unknown):
            missing.append(cond)
            continue
        if unknown:
            part = sum(val(st[(q, cond)]) for q in qs if st[(q, cond)] != "?")
            apart.append((r["layer"], r["mult"], part, len(unknown), cond))
            print("    L%-3d x %-5s | no net: the judge declined %d of the %d answers; net of the other %d %+d, so between %+d and %+d | answers %3d | MMLU selection half %s" % (
                r["layer"], mstr(r["mult"]), len(unknown), len(qs), len(qs) - len(unknown), part, part - len(unknown),
                part + len(unknown), sum(not bl[(q, cond)] for q in qs), mmlu))
            continue
        net = sum(val(st[(q, cond)]) for q in qs)
        ans = sum(not bl[(q, cond)] for q in qs)
        table.append((net, r["layer"], r["mult"], cond, ans))
        print("    L%-3d x %-5s | net %+4d | answers %3d | right %3d | wrong %3d | MMLU selection half %s" % (
            r["layer"], mstr(r["mult"]), net, ans, sum(st[(q, cond)] == "R" for q in qs), sum(st[(q, cond)] == "W" for q in qs), mmlu))
    if missing:
        raise SystemExit("not judged yet: %s" % " ".join(missing))
    if not table:
        raise SystemExit("no grid point with a verdict on every answer")
    best = max(table, key=lambda t: (t[0], -t[1], -t[2]))
    for layer, m, part, n, cond in apart:
        if part + n >= best[0]:
            raise SystemExit("the judge declined %d answers of %s; with all of them counted as right it reaches %+d, not below the best point with every verdict (%+d): the choice would depend on them" % (
                n, cond, part + n, best[0]))
    if apart:
        print("    %d point(s) with judge-declined answers are not eligible; each stays below the chosen point's net even with every declined answer counted as right" % len(apart))
    print("    chosen on tune: layer %d, multiplier %s (net %+d, answers %d)%s" % (
        best[1], mstr(best[2]), best[0], best[4], " -- the best among the measured points" if every else ""))
    if write:
        path = "%s/choice%s.json" % (mi.dir, "_all" if every else "")
        doc = {"model": model, "budget": None if every else budget, "stop": STOP, "layer": best[1], "mult": best[2],
               "arm": arm_of(best[1], best[2]), "tune_condition": best[3], "tune_net": best[0], "tune_answers": best[4],
               "tune_questions": len(qs), "candidates": [[t[1], t[2], t[0]] for t in table],
               "reading": "best among the measured points" if every else "best within the MMLU budget"}
        if apart:
            doc["declined"] = [[l, m, part, n] for l, m, part, n, _ in apart]
        doc["vectors_sha256"] = mi.listing["vectors_file_sha256"]
        if os.path.exists(path):
            prior = read(path)
            if prior != doc:
                raise SystemExit("existing choice disagrees with the recomputed tuning result: " + path)
            print("    choice reproduced from the current grid: %s" % path)
        else:
            with open(path, "x", encoding="utf-8") as f:
                json.dump(doc, f, indent=1)
            print("    choice written to %s" % path)


def read_choice(mi, every):
    ch = read("%s/choice%s.json" % (mi.dir, "_all" if every else ""))
    candidates = ch["candidates"]
    if not candidates or len({(l,m) for l,m,_ in candidates}) != len(candidates):
        raise SystemExit("choice has no valid tuning grid")
    layer, mult, net = max(candidates, key=lambda row: (row[2], -row[0], -row[1]))
    if ((layer, mult, net) != (ch["layer"], ch["mult"], ch["tune_net"]) or ch["arm"] != arm_of(layer, mult)
            or ch["tune_condition"] != cond_name(layer, mult) or ch.get("vectors_sha256") != mi.listing["vectors_file_sha256"]):
        raise SystemExit("choice is stale or disagrees with its tuning evidence")
    if ch["model"] != mi.model or (ch["budget"] is None) != every:
        raise SystemExit("the choice file belongs to another setting")
    return ch


def retained_bounds(states, flags, k):
    """(lowest, highest) net when exactly k of the issued answers (flags) are kept and every other row scores 0."""
    eligible = sorted(val(s) for s, f in zip(states, flags) if f)
    if not 0 <= k <= len(eligible):
        raise SystemExit("the requested number of kept answers is unreachable")
    return sum(eligible[:k]), sum(eligible[len(eligible) - k:])


def equal_count(name, st, ans, caa, caa_ans):
    n = min(sum(caa_ans), sum(ans))
    sl, sh = retained_bounds(st, ans, n)
    cl, chi = retained_bounds(caa, caa_ans, n)
    print("    %s minus CAA with both keeping %d answers: between %+d and %+d (%s keeps %d of %d, CAA %d of %d)" % (
        name, n, sl - chi, sh - cl, name, n, sum(ans), n, sum(caa_ans)))
    return {"kept": n, "low": sl - chi, "high": sh - cl}


def test_table(model, every=False):
    display, elayer, s = KEY[model]
    mi = Mirror(model)
    ch = read_choice(mi, every)
    X, why = MP.prompt_split(display, elayer, 1, s)
    T, why_t = MP.prompt_split(display, elayer, 0, s)
    if X is None or T is None:
        raise SystemExit("the project's own split at the prompt rule is not complete (%s)" % (why or why_t))
    xsp, xtx, _, qs = X
    tsp, ttx, _, _ = T
    w = MP.work(mi.run, "test", "caa_all" if every else "caa_main")
    if every and not MP.has(w):
        main = read_choice(mi, False)
        if (main["layer"], main["mult"]) == (ch["layer"], ch["mult"]):
            w = MP.work(mi.run, "test", "caa_main")
            print("# the choice among the measured points is the budget reading's own point: its test verdicts are read from %s" % os.path.basename(w))
    if not MP.has(w):
        raise SystemExit("no Pro verdicts yet in %s" % os.path.basename(w))
    cond, g = mi.generation("test", ch["layer"], ch["mult"])
    verify_work(mi, w)
    st, bl = ps.load(w, "pro")
    if any(st.get((q, cond), "?") == "?" for q in qs):
        raise SystemExit("incomplete verdicts for %s" % cond)
    caa = [st[(q, cond)] for q in qs]
    caa_ans = [not bl[(q, cond)] for q in qs]
    base = S.system(xsp, xtx, ("base",))[0]
    e = S.system(xsp, xtx, ("E", s))
    n1 = S.system(xsp, xtx, ML.n1_pick(tsp, ttx, s))
    d = lambda a, b: [val(x) - val(y) for x, y in zip(a, b)]  # noqa: E731
    print("# CAA, %s: TruthfulQA test split, %d questions, DeepSeek Pro, score = right - wrong" % (model, len(qs)))
    print("    chosen on tune%s: layer %d, multiplier %s (tune net %+d); positions: the last prompt token and every generated token" % (
        " as the best among the measured points, whatever the MMLU change" if every else " within the MMLU budget",
        ch["layer"], mstr(ch["mult"]), ch["tune_net"]))
    print("    no write net %+d | CAA net %+d (answers %d, right %d, wrong %d) | single E (layer %d, strength %s, prompt rule) net %+d (answers %d) | N1 net %+d (answers %d)" % (
        sum(map(val, base)), sum(map(val, caa)), sum(caa_ans), caa.count("R"), caa.count("W"),
        elayer, DM.show(s), sum(map(val, e[0])), sum(e[1]), sum(map(val, n1[0])), sum(n1[1])))
    print("    CAA minus no write %s | single E minus CAA %s | N1 minus CAA %s | single E minus no write %s" % (
        ps.ci(d(caa, base)), ps.ci(d(e[0], caa)), ps.ci(d(n1[0], caa)), ps.ci(d(e[0], base))))
    equal_count("N1", n1[0], n1[1], caa, caa_ans)
    equal_count("single E", e[0], e[1], caa, caa_ans)
    for r in budget_rows(model, 1.0, show=False):
        if (r["layer"], r["mult"]) == (ch["layer"], ch["mult"]):
            print("    MMLU of the chosen point: selection half %+.2f [%+.2f, %+.2f], report half %+.2f [%+.2f, %+.2f]" % (r["dev"] + r["rep"]))


def neg_table(model):
    display, elayer, s = KEY[model]
    mi = Mirror(model)
    T, why = MP.prompt_split(display, elayer, 0, s)
    if T is None:
        raise SystemExit("the project's own tune split at the prompt rule is not complete (%s)" % why)
    tsp, ttx, _, qs = T
    base = S.system(tsp, ttx, ("base",))[0]
    st, bl = IT.merged_verdicts(model, tune_works(mi))
    print("# CAA, %s: direction check on the tune split (%d questions), DeepSeek Pro, score = right - wrong; no write net %+d" % (
        model, len(qs), sum(map(val, base))))
    seen = set()
    for every in (False, True):
        if not os.path.exists("%s/choice%s.json" % (mi.dir, "_all" if every else "")):
            continue
        ch = read_choice(mi, every)
        if ch["layer"] in seen:
            continue
        seen.add(ch["layer"])
        works = sorted(glob.glob("%s/eval_extra/judge_tune_caa_neg_L%d_*_tune" % (mi.run, ch["layer"])))
        for w in works:
            if MP.has(w):
                verify_work(mi, w)
        sn, bn = IT.merged_verdicts(model, works)
        cond, _ = mi.generation("tune", ch["layer"], -1.0)
        if any(sn.get((q, cond), "?") == "?" for q in qs):
            raise SystemExit("multiplier -1 at layer %d is not judged yet" % ch["layer"])
        cc = ch["tune_condition"]
        print("    layer %d: multiplier -1 net %+d (answers %d) | chosen multiplier %s net %+d (answers %d) | -1 minus no write %s" % (
            ch["layer"], sum(val(sn[(q, cond)]) for q in qs), sum(not bn[(q, cond)] for q in qs), mstr(ch["mult"]),
            sum(val(st[(q, cc)]) for q in qs), sum(not bl[(q, cc)] for q in qs),
            ps.ci([val(sn[(q, cond)]) - val(b) for q, b in zip(qs, base)])))
    for p in sorted(glob.glob("%s/check_zero_full_L*.json" % mi.dir)):
        z = read(p)
        for layer, r in z["zero"].items():
            print("    multiplier 0 at layer %s on %d tune questions: %d answers equal to no write token for token (%s)" % (
                layer, r["of"], r["same_as_no_write"], os.path.basename(p)))


def main():
    argv = sys.argv[1:]
    opt = {"--budget": "1", "--layers": "", "--mults": ""}
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
    if len(rest) != 2 or rest[0] not in ("budget", "tune", "test", "neg") or rest[1] not in KEY:
        raise SystemExit("usage: caa_table.py budget|tune|test|neg MODEL [--budget B] [--layers \"L ...\" --mults \"M ...\"] [--all] [--write-choice]")
    mode, model = rest
    if mode == "budget":
        budget_rows(model, float(opt["--budget"]))
    elif mode == "tune":
        tune_table(model, float(opt["--budget"]), write, every,
                   [int(x) for x in opt["--layers"].split()], [float(x) for x in opt["--mults"].split()])
    elif mode == "test":
        test_table(model, every)
    else:
        neg_table(model)


if __name__ == "__main__":
    main()
