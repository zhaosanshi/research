#!/usr/bin/env python3
"""Selection-rule variants for the true/false E knob (queue M, progress entry 2026-11-25 22:11, part two). Executor "mao"
(Claude Opus 5.5), 2026-11-25. CPU only, read-only on every input; prints numbers, condition names, model names and pair
counts only (no statement text).

For one model, the DAS sweep (knob_sweep_<das-tag>) and the mean-difference sweep (knob_sweep_<md-tag>) under RES/E23/MODEL
are read. Candidates = the single-layer E conditions on the layers both sweeps cover, in the sweep's recorded order (as in
scripts/dr_tf_table.py). Every rule looks at the 192 tuning pairs only (the sweep json's split "A"):
  A  (frozen 2026-11-12, in use): feasible = |d answer rate| <= 0.03; max dJ; ties -> first in sweep order
  B  A restricted to settings whose tuning AUROC does not fall (d AUROC >= 0)
  C  feasible; max d AUROC; ties -> first
  D  feasible; max (d J + d AUROC); ties -> first
  E  A restricted to alpha = 2
  F  A restricted to alpha = 3
  G  the first five settings of rule A's ranking (dJ descending, sweep order within ties)
Comparisons use exact integer keys (core.metrics.sel_key): dJ * n_pos*n_neg, d AUROC * 2*n_pos*n_neg (AUROC with half
ties is a multiple of 1/(2*n_pos*n_neg)), the same "max(), first in sweep order" tie rule as e23_compare_estimators.rule_pick
(rule A, E, F call rule_pick itself).
Every picked setting is read on the Dr's 100 pairs and on all 529 test pairs: change against the original model of the
self-reported answer (Yes) rate, J and AUROC (p_true), with pair-bootstrap 95% intervals built exactly as dr_tf_table.py
builds them (whole pairs, 2000 resamples, numpy default_rng(seed 0) per report set, the same resamples for every arm), and
DAS minus mean difference under the same rule.
Self-check first: rule A must reproduce the existing three-method table json (dr_tf_table_<model>.json: picks, tuning
deltas, every point value and 95% interval of the three differences, sweep sha256); exit 1 otherwise.

usage: tfg3_rule_variants.py --res RES --model M [--table RES/dr_tf_table_M.json] [--pairs-file F]   (report on stdout)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.environ.get("PYTHONPATH", "").split(":")[0] or os.getcwd()
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, HERE)
from core.metrics import sel_key  # noqa: E402
import e23_compare_estimators as cmp  # noqa: E402  rule_pick, layers_of, E_LEAK
import dr_tf_table as drt  # noqa: E402  metrics, ci, sha, load, margin_of (the three-method table's own functions)

RULES = ("A", "B", "C", "D", "E", "F")
RULE_TEXT = {
    "A": "frozen 2026-11-12 (in use): |d ans| <= 0.03, max dJ, ties -> first in sweep order",
    "B": "A plus tuning d AUROC >= 0",
    "C": "|d ans| <= 0.03, max d AUROC, ties -> first",
    "D": "|d ans| <= 0.03, max (dJ + d AUROC), ties -> first",
    "E": "A restricted to alpha = 2",
    "F": "A restricted to alpha = 3",
    "G": "first five settings of rule A's ranking",
}
DK = ("yes_rate", "J", "AUROC")
TOL = 1e-12


def fail(msg):
    print("SELF-CHECK FAILED: %s" % msg)
    sys.exit(1)


def compress(xs):
    xs = sorted(xs)
    out, s = [], 0
    for i in range(1, len(xs) + 1):
        if i == len(xs) or xs[i] != xs[i - 1] + 1:
            out.append("%d" % xs[s] if s == i - 1 else "%d-%d" % (xs[s], xs[i - 1]))
            s = i
    return ", ".join(out)


def cell(point, arr):
    lo, hi = drt.ci(arr)
    return "%+.3f [%+.3f, %+.3f]" % (point, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--das-tag", default="das_tfce_full")
    ap.add_argument("--md-tag", default="meandiff_full")
    ap.add_argument("--table", default=None, help="existing three-method table json (default RES/dr_tf_table_MODEL.json)")
    ap.add_argument("--pairs-file", default=os.path.join(HERE, "configs", "splits", "dr_100pairs_20261110.json"))
    a = ap.parse_args()
    table = a.table or os.path.join(a.res, "dr_tf_table_%s.json" % a.model)
    if not os.path.exists(table):
        fail("no existing three-method table %s" % table)
    ref_all = json.load(open(table))
    ref = ref_all["models"].get(a.model)
    if ref is None:
        fail("model %s not in %s" % (a.model, table))
    n_boot, seed = int(ref_all["n_boot"]), int(ref_all["seed"])
    if (n_boot, seed) != (2000, 0) or ref_all["das_tag"] != a.das_tag or ref_all["md_tag"] != a.md_tag:
        fail("the table used n_boot %d seed %d tags %s / %s" % (n_boot, seed, ref_all["das_tag"], ref_all["md_tag"]))
    jd, zd, fd = drt.load(a.res, a.model, a.das_tag)
    jm, zm, fm = drt.load(a.res, a.model, a.md_tag)
    for est, f in (("DAS", fd), ("meandiff", fm)):
        for k in ("json_sha256", "npz_sha256"):
            if ref["files"][est][k] != f[k]:
                fail("%s %s %s differs from the table's record %s" % (est, k, f[k][:16], ref["files"][est][k][:16]))
    if drt.sha(a.pairs_file) != ref_all["pairs_file_sha256"]:
        fail("pairs file sha256 differs from the table's record")
    if jd.get("e_estimator") != "das" or jm.get("e_estimator") != "meandiff" or (jd.get("das") or {}).get("variant") != "tf-ce-bidir":
        fail("estimator / DAS variant not as expected")
    for k in ("ids", "half_a", "labels"):
        if not np.array_equal(zd[k], zm[k]):
            fail("the two sweeps differ in %s" % k)
    for k in ("model", "wording", "e_alphas", "n", "n_est", "tune_pairs"):
        if jd.get(k) != jm.get(k):
            fail("the two sweeps differ in %s" % k)
    for k in ("answer", "ptrue", "margin"):
        if not np.array_equal(zd["baseline__" + k], zm["baseline__" + k]):
            fail("baseline %s differs between the sweeps" % k)
    lab = zd["labels"].astype(bool)
    tune = zd["half_a"].astype(bool)
    pair = np.array([str(i).rsplit(":", 1)[0] for i in zd["ids"]])
    dr = set(json.load(open(a.pairs_file))["pair_ids"])
    if len(dr) != 100 or dr - set(pair[~tune]) or set(pair[tune]) & dr:
        fail("the Dr's 100 pairs are not 100 distinct test pairs")
    n_tune, n_test = len(set(pair[tune])), len(set(pair[~tune]))
    if n_tune != 192 or n_test != 529:
        fail("%d tuning / %d test pairs" % (n_tune, n_test))
    scale = int(lab[tune].sum()) * int((~lab[tune]).sum())

    # ---- candidates (as dr_tf_table.py) and tuning-set values from the sweep json, checked against the arrays ---- #
    layer_sets = [{tuple(cmp.layers_of(n)[0]) for n in j["conds"] if n.startswith("E_") and cmp.layers_of(n)[0]} for j in (jd, jm)]
    common = layer_sets[0] & layer_sets[1]
    J = {"DAS": jd, "meandiff": jm}
    Z = {"DAS": zd, "meandiff": zm}
    names, tun = {}, {}
    for est in ("DAS", "meandiff"):
        j, z = J[est], Z[est]
        names[est] = [n for n in j["conds"] if n.startswith("E_") and cmp.layers_of(n)[0] and len(cmp.layers_of(n)[0]) == 1
                      and tuple(cmp.layers_of(n)[0]) in common]
        bA = j["conds"]["baseline"]["A"]
        tun[est] = {}
        bt = drt.metrics(zd["baseline__answer"].astype(bool)[tune], zd["baseline__ptrue"].astype(float)[tune],
                         drt.margin_of(zd, "baseline")[tune], lab[tune])
        for n in names[est]:
            cA = j["conds"][n]["A"]
            mt = drt.metrics(z[n + "__answer"].astype(bool)[tune], z[n + "__ptrue"].astype(float)[tune], drt.margin_of(z, n)[tune], lab[tune])
            for km, kj in (("yes_rate", "ans"), ("J", "J"), ("AUROC", "AUROC")):
                if abs((mt[km] - bt[km]) - (cA[kj] - bA[kj])) > 1e-9:
                    fail("%s %s tuning %s recomputed from the arrays differs from the sweep json" % (est, n, km))
            tun[est][n] = {"ans": cA["ans"] - bA["ans"], "J": cA["J"] - bA["J"], "AUROC": cA["AUROC"] - bA["AUROC"]}

    def kJ(est, n):
        return sel_key(tun[est][n]["J"], scale)

    def kA(est, n):
        return sel_key(tun[est][n]["AUROC"], 2 * scale)

    def kD(est, n):
        return sel_key(tun[est][n]["J"], 2 * scale) + kA(est, n)

    def first_max(cands, key):
        return max(cands, key=key) if cands else None

    picks, info = {}, {}
    for est in ("DAS", "meandiff"):
        conds = J[est]["conds"]
        feas = [n for n in names[est] if abs(tun[est][n]["ans"]) <= cmp.E_LEAK + 1e-12]
        sets = {"A": feas, "B": [n for n in feas if kA(est, n) >= 0], "C": feas, "D": feas,
                "E": [n for n in feas if cmp.layers_of(n)[1] == 2.0], "F": [n for n in feas if cmp.layers_of(n)[1] == 3.0]}
        keys = {"A": lambda n: kJ(est, n), "B": lambda n: kJ(est, n), "C": lambda n: kA(est, n), "D": lambda n: kD(est, n),
                "E": lambda n: kJ(est, n), "F": lambda n: kJ(est, n)}
        for r in RULES:
            if r in ("A", "E", "F"):
                sub = names[est] if r == "A" else [n for n in names[est] if cmp.layers_of(n)[1] == (2.0 if r == "E" else 3.0)]
                p = cmp.rule_pick(conds, sub, scale)
            else:
                p = first_max(sets[r], keys[r])
            q = first_max(sets[r], keys[r])
            if p != q:
                fail("%s rule %s: rule_pick %s vs local %s" % (est, r, p, q))
            ties = sum(1 for n in sets[r] if keys[r](n) == keys[r](p)) if p else 0
            picks[(est, r)] = p
            info[(est, r)] = (len(sets[r]), ties)
        order = sorted(range(len(feas)), key=lambda i: (-kJ(est, feas[i]), i))
        picks[(est, "G")] = [feas[i] for i in order[:5]]
        if feas and picks[(est, "G")][0] != picks[(est, "A")]:
            fail("%s: rule G's first setting %s is not rule A's pick %s" % (est, picks[(est, "G")][0], picks[(est, "A")]))
        info[(est, "G")] = (len(feas), None)

    # ---- arrays and bootstrap (dr_tf_table.py's construction) ---- #
    arms = {"original": ("baseline", zd)}
    for est in ("DAS", "meandiff"):
        for r in RULES + ("G",):
            pl = picks[(est, r)] if r == "G" else [picks[(est, r)]]
            for n in pl:
                if n is not None:
                    arms["%s:%s" % (est, n)] = (n, Z[est])
    arr = {k: (z[n + "__answer"].astype(bool), z[n + "__ptrue"].astype(float), drt.margin_of(z, n)) for k, (n, z) in arms.items()}
    rep_sets = [("100 pairs", (~tune) & np.isin(pair, sorted(dr))), ("529 pairs", ~tune)]
    point, bs = {}, {}
    for sname, mask in rep_sets:
        pairs_here = np.array(sorted(set(pair[mask])))
        where = {p: np.where((pair == p) & mask)[0] for p in pairs_here}
        if any(len(v) != 2 for v in where.values()):
            fail("a report pair does not have exactly two statements")
        rng = np.random.default_rng(seed)
        boot = [np.concatenate([where[p] for p in rng.choice(pairs_here, len(pairs_here), replace=True)]) for _ in range(n_boot)]
        idx = np.where(mask)[0]
        point[sname] = {k: drt.metrics(*(x[idx] for x in v), lab[idx]) for k, v in arr.items()}
        bs[sname] = {}
        for k, v in arr.items():
            rows = [drt.metrics(*(x[ii] for x in v), lab[ii]) for ii in boot]
            bs[sname][k] = {m: np.array([r_[m] for r_ in rows]) for m in drt.KEYS}

    def diff(sname, k1, k0, m):
        return point[sname][k1][m] - point[sname][k0][m], bs[sname][k1][m] - bs[sname][k0][m]

    # ---- self-check against the existing three-method table ---- #
    if picks[("DAS", "A")] != ref["chosen"]["DAS"] or picks[("meandiff", "A")] != ref["chosen"]["meandiff"]:
        fail("rule A picks DAS %s / md %s, table %s" % (picks[("DAS", "A")], picks[("meandiff", "A")], ref["chosen"]))
    nchk = 0
    for est in ("DAS", "meandiff"):
        for kt, kj in (("J", "J"), ("ans", "ans"), ("AUROC", "AUROC")):
            if abs(tun[est][picks[(est, "A")]][kj] - ref["tune"][est][kt]) > TOL:
                fail("%s tuning %s differs from the table" % (est, kt))
            nchk += 1
    kd, km_ = "DAS:" + picks[("DAS", "A")], "meandiff:" + picks[("meandiff", "A")]
    for sname, _ in rep_sets:
        rs = ref["sets"][sname]
        for arm, k in (("original", "original"), ("DAS", kd), ("meandiff", km_)):
            for m in drt.KEYS:
                if abs(point[sname][k][m] - rs["point"][arm][m]) > TOL:
                    fail("%s %s %s point differs" % (sname, arm, m))
                nchk += 1
        for dn, k1, k0 in (("DAS - original", kd, "original"), ("meandiff - original", km_, "original"), ("DAS - meandiff", kd, km_)):
            for m in drt.KEYS:
                p_, a_ = diff(sname, k1, k0, m)
                lo, hi = drt.ci(a_)
                r_ = rs["diffs"][dn][m]
                if abs(p_ - r_["point"]) > TOL or abs(lo - r_["ci95"][0]) > TOL or abs(hi - r_["ci95"][1]) > TOL:
                    fail("%s %s %s differs from the table" % (sname, dn, m))
                nchk += 3

    # ---- report ---- #
    out = []
    P = out.append
    P("# Selection-rule variants for the true/false E knob: %s" % a.model)
    P("")
    P("Rule A was fixed on 2026-11-12, before Gemma-3-27B's results were seen; rules B to G were listed on 2026-11-25, after they were seen.")
    P("Every rule chooses on the 192 tuning pairs only; the Dr's 100 pairs and the 529 test pairs are only read. 'answer rate' = the")
    P("self-reported Yes rate of the separate 'Do you know the answer?' pass (the 'ans' of the sweep). Changes are against the original")
    P("model. 95%% intervals: pair bootstrap, %d resamples, seed %d, built as in dr_tf_table.py, the same resamples for every arm of this model." % (n_boot, seed))
    P("")
    P("## Header")
    P("- model: %s; results root: %s" % (a.model, a.res))
    for est, f in (("DAS", fd), ("meandiff", fm)):
        P("- %s sweep: %s (json sha256 %s, npz sha256 %s)" % (est, os.path.basename(f["npz"])[:-4], f["json_sha256"][:16], f["npz_sha256"][:16]))
    P("- existing three-method table: %s (sha256 %s); pairs file %s (sha256 %s)" % (
        os.path.basename(table), drt.sha(table)[:16], os.path.basename(a.pairs_file), drt.sha(a.pairs_file)[:16]))
    P("- pairs: tuning %d, test %d, Dr %d (part of the test pairs); statements per pair 2" % (n_tune, n_test, len(dr)))
    P("- alphas in the sweep: %s; layers: %s (%d single-layer E settings per estimator on the common layers)" % (
        ", ".join("%g" % x for x in jd["e_alphas"]), compress({l[0] for l in common}), len(names["DAS"])))
    btun = drt.metrics(*(x[tune] for x in arr["original"]), lab[tune])
    for sname, m in (("tuning 192 pairs", btun), ("Dr 100 pairs", point["100 pairs"]["original"]), ("test 529 pairs", point["529 pairs"]["original"])):
        P("- original model, %s: answer rate %.3f, J %+.3f, AUROC %.3f" % (sname, m["yes_rate"], m["J"], m["AUROC"]))
    P("")
    P("## Self-check")
    P("SELF-CHECK PASSED: rule A picks DAS %s and mean difference %s as the existing table does; %d numbers (tuning deltas, points, "
      "differences and 95%% interval ends of the three differences on both report sets) equal to within %g; sweep sha256 equal to the table's record."
      % (picks[("DAS", "A")], picks[("meandiff", "A")], nchk, TOL))
    for sname, _ in rep_sets:
        for dn, k1, k0 in (("DAS - original", kd, "original"), ("DAS - meandiff", kd, km_)):
            rs = ref["sets"][sname]["diffs"][dn]
            mine = " | ".join(cell(*diff(sname, k1, k0, m)) for m in ("J", "AUROC", "yes_rate"))
            theirs = " | ".join("%+.3f [%+.3f, %+.3f]" % (rs[m]["point"], rs[m]["ci95"][0], rs[m]["ci95"][1]) for m in ("J", "AUROC", "yes_rate"))
            P("- %s, %s (J | AUROC | answer rate): this program %s ; table %s" % (sname, dn, mine, theirs))
    P("")

    def row_cells(est, n):
        t = tun[est][n]
        c = ["%+.3f | %+.3f | %+.3f" % (t["ans"], t["J"], t["AUROC"])]
        for sname, _ in rep_sets:
            c.append(" | ".join(cell(*diff(sname, "%s:%s" % (est, n), "original", m)) for m in DK))
        return " | ".join(c)

    hd = ("| rule | picked | feasible | tied at max | tune d ans | tune dJ | tune dAUROC | Dr100 d ans | Dr100 dJ | Dr100 dAUROC "
          "| 529 d ans | 529 dJ | 529 dAUROC |")
    for est in ("DAS", "meandiff"):
        P("## %s: rules A to F" % est)
        P("")
        for r in RULES:
            P("- %s: %s" % (r, RULE_TEXT[r]))
        P("")
        P(hd)
        P("|" + "---|" * 13)
        for r in RULES:
            n = picks[(est, r)]
            nf, ties = info[(est, r)]
            if n is None:
                P("| %s | no feasible setting | %d | - | | | | | | | | | |" % (r, nf))
            else:
                P("| %s | %s | %d | %d | %s |" % (r, n, nf, ties, row_cells(est, n)))
        P("")
    for est in ("DAS", "meandiff"):
        P("## %s: rule G (first five of rule A's ranking; %d feasible)" % (est, info[(est, "G")][0]))
        P("")
        P("| rank | setting | tune d ans | tune dJ | tune dAUROC | Dr100 d ans | Dr100 dJ | Dr100 dAUROC | 529 d ans | 529 dJ | 529 dAUROC |")
        P("|" + "---|" * 11)
        for i, n in enumerate(picks[(est, "G")]):
            P("| %d | %s | %s |" % (i + 1, n, row_cells(est, n)))
        P("")
    P("## DAS minus mean difference under each rule (same resamples)")
    P("")
    P("| rule | DAS | mean difference | Dr100 J | Dr100 AUROC | 529 J | 529 AUROC |")
    P("|---|---|---|---|---|---|---|")
    for r in RULES:
        d_, m_ = picks[("DAS", r)], picks[("meandiff", r)]
        if d_ is None or m_ is None:
            P("| %s | %s | %s | no feasible setting on one side | | | |" % (r, d_ or "none", m_ or "none"))
            continue
        c = [cell(*diff(s, "DAS:" + d_, "meandiff:" + m_, m)) for s in ("100 pairs", "529 pairs") for m in ("J", "AUROC")]
        P("| %s | %s | %s | %s |" % (r, d_, m_, " | ".join(c)))
    P("")
    P("## DAS per layer (points only; * = feasible, |tune d ans| <= 0.03)")
    test_idx = np.where(~tune)[0]
    bte = point["529 pairs"]["original"]
    for alpha in jd["e_alphas"]:
        P("")
        P("### alpha %g: layer | tune d ans | tune dJ | tune dAUROC | 529 dJ | 529 dAUROC" % alpha)
        for n in names["DAS"]:
            ls_, al = cmp.layers_of(n)
            if al != float(alpha):
                continue
            t = tun["DAS"][n]
            z = zd
            m = drt.metrics(z[n + "__answer"].astype(bool)[test_idx], z[n + "__ptrue"].astype(float)[test_idx], drt.margin_of(z, n)[test_idx], lab[test_idx])
            P("L%d%s %+.3f %+.3f %+.3f | %+.3f %+.3f" % (ls_[0], "*" if abs(t["ans"]) <= cmp.E_LEAK + 1e-12 else " ", t["ans"], t["J"], t["AUROC"],
                                                       m["J"] - bte["J"], m["AUROC"] - bte["AUROC"]))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
