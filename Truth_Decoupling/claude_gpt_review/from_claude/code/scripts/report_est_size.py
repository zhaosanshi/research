#!/usr/bin/env python3
"""Estimation-set size experiment (赵磊 2026-11-02): does the DAS direction (or the mean-difference one) get better when it is
estimated on 288 pairs instead of 96?  1:2:7 (96 est : 192 tune : 529 test)  ->  288 est : 192 tune : 337 test.

Four E23 sweeps of ONE model: {DAS, mean-diff} x {small estimation set (--root-old), large (--root-new)}. The new sweeps
define the split (e23_knob_sweep.py --est-extra-pairs 192 --tune-pairs 192: the old tuning pairs moved into the estimation
pool, a fresh tuning set drawn from the old test pairs, the remaining pairs = test). The old sweeps' per-item arrays are
re-cut to the new ids, so all four are selected on the SAME tuning pairs (E rule of e23: max dJ s.t. |d ans| <= 0.03,
tie-break = sweep order) and reported on the SAME test pairs; every difference is a pair-bootstrap on identical resamples.
Metric definitions as in e23_compare_estimators.py (pred = p_true > 0.5; J = TPR - FPR; tie-aware AUROC; B = mean pred).
CPU / numpy only.

usage: python3 scripts/report_est_size.py --root-old /w/results_das_tf --root-new /w/results_das_tf_r288 --model llama3.1-8b \
           --out /w/results_das_tf_r288/compare/llama3.1-8b_est_size.md
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import auroc, sel_key  # noqa: E402  (the sweeps' tie-aware AUROC and exact-count selection key)

E_LEAK = 0.03
K = ("J", "AUROC", "ans", "B")


def j_stat(pred, lab):
    return float(pred[lab].mean() - pred[~lab].mean())


def metrics(ans, p, lab):
    pred = p > 0.5
    return {"ans": float(ans.mean()), "J": j_stat(pred, lab), "AUROC": auroc(p, lab), "B": float(pred.mean())}


def layers_of(name):
    m = re.match(r"^E_L([\d-]+)_a(.+)$", name)
    return ([int(x) for x in m.group(1).split("-")], float(m.group(2))) if m else (None, None)


class Sweep:
    """One E23 sweep re-cut to a given id order: per-condition answer / p_true arrays on exactly those items."""

    def __init__(self, estimator, root, model, tag, ids, labels):
        d = os.path.join(root, "E23", model)
        self.j = json.load(open(os.path.join(d, "knob_sweep_%s.json" % tag)))
        z = np.load(os.path.join(d, "knob_sweep_%s.npz" % tag), allow_pickle=True)
        pos = {i: k for k, i in enumerate(z["ids"].tolist())}
        missing = [i for i in ids if i not in pos]
        if missing:
            raise SystemExit("%s/%s: %d items of the split are missing (e.g. %s)" % (root, tag, len(missing), missing[:3]))
        idx = np.array([pos[i] for i in ids])
        if not np.array_equal(z["labels"][idx].astype(bool), labels):
            raise SystemExit("%s/%s: labels differ on the common items" % (root, tag))
        self.lab = labels
        self.ids_all = z["ids"].tolist()
        self.n_items_total = int(len(z["ids"]))
        self.n_est = int(self.j.get("n_est") or 0)
        self.n_est_pairs = self.n_est // 2
        self.label = "%s-%d" % (estimator, self.n_est_pairs)
        self.names = [n for n in self.j["conds"] if n.startswith("E_")]        # sweep order == the tie-break order
        self.ans = {n: z[n + "__answer"][idx].astype(bool) for n in ["baseline"] + self.names}
        self.p = {n: z[n + "__ptrue"][idx].astype(float) for n in ["baseline"] + self.names}
        self.das = (self.j.get("das") or {}).get("per_layer") or {}

    def m(self, name, mask):
        return metrics(self.ans[name][mask], self.p[name][mask], self.lab[mask])

    def delta(self, name, mask, key):
        return self.m(name, mask)[key] - self.m("baseline", mask)[key]

    def rule_pick(self, names, mask):
        """E rule on `mask`: feasible = |d ans| <= E_LEAK; best = max dJ; ties -> first in sweep order. dJ is compared as an exact
        integer count (core.metrics.sel_key; dJ is a multiple of 1/(n_pos*n_neg) on a fixed split): the raw float let a 1.1e-16
        rounding difference pick Qwen3-8B MD-288 L35 a2 over the tied, earlier L34 a3 (GPT review round 13, 2026-11-04)."""
        ok = [n for n in names if abs(self.delta(n, mask, "ans")) <= E_LEAK + 1e-12]
        scale = int(self.lab[mask].sum()) * int((~self.lab[mask]).sum())
        return max(ok, key=lambda n: sel_key(self.delta(n, mask, "J"), scale)) if ok else None

    def est_delta_J(self, name):
        """In-sample: dJ of `name` on the sweep's own estimation pairs (from the sweep json; 96 or 288 pairs)."""
        c = self.j["conds"]
        return c[name]["est"]["J"] - c["baseline"]["est"]["J"] if "est" in c.get(name, {}) else float("nan")


def ci_cell(v, arr):
    return "%+.3f [%+.3f, %+.3f]" % (v, np.percentile(arr, 2.5), np.percentile(arr, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-old", required=True, help="results root of the small-estimation-set sweeps (96 pairs)")
    ap.add_argument("--root-new", required=True, help="results root of the large-estimation-set sweeps (defines the split)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag-das", default="das_tf_full")
    ap.add_argument("--tag-md", default="meandiff_full")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="markdown report (refuses to overwrite); a .json with the numbers is written next to it")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("refusing to overwrite %s" % args.out)

    dn = os.path.join(args.root_new, "E23", args.model)
    z_das = np.load(os.path.join(dn, "knob_sweep_%s.npz" % args.tag_das), allow_pickle=True)
    z_md = np.load(os.path.join(dn, "knob_sweep_%s.npz" % args.tag_md), allow_pickle=True)
    if not (np.array_equal(z_das["ids"], z_md["ids"]) and np.array_equal(z_das["half_a"], z_md["half_a"])):
        raise SystemExit("the two new sweeps have different items or a different tuning split")
    ids = z_das["ids"].tolist()
    lab = z_das["labels"].astype(bool)
    tune = z_das["half_a"].astype(bool)
    test = ~tune
    j_new = json.load(open(os.path.join(dn, "knob_sweep_%s.json" % args.tag_das)))
    n_extra = int(j_new.get("est_extra_pairs") or 0)
    if n_extra == 0:
        print("warning: the new sweeps record no moved estimation pairs (est_extra_pairs = 0)", flush=True)

    sweeps = [Sweep("DAS", args.root_old, args.model, args.tag_das, ids, lab),
              Sweep("MD", args.root_old, args.model, args.tag_md, ids, lab),
              Sweep("DAS", args.root_new, args.model, args.tag_das, ids, lab),
              Sweep("MD", args.root_new, args.model, args.tag_md, ids, lab)]
    if len({s.label for s in sweeps}) != 4:
        raise SystemExit("the four sweeps must differ in estimator and estimation-set size: %s" % [s.label for s in sweeps])
    old_das, old_md, new_das, new_md = sweeps
    pair = np.array([i.rsplit(":", 1)[0] for i in ids])
    moved = set(pair.tolist()) & set(j_new.get("est_extra_ids") or [])
    if moved:
        raise SystemExit("pairs moved into the estimation set are still in the new sweep's evaluation pool: %s" % sorted(moved)[:3])
    if n_extra and len(set(pair.tolist())) + n_extra != len(set(p.rsplit(":", 1)[0] for p in old_das.ids_all)):
        raise SystemExit("evaluation pairs (%d) + moved pairs (%d) do not add up to the old evaluation pool" % (len(set(pair.tolist())), n_extra))
    test_pairs = np.array(sorted(set(pair[test])))
    tune_pairs = sorted(set(pair[tune]))
    rng = np.random.default_rng(args.seed)
    where = {p: np.where((pair == p) & test)[0] for p in test_pairs}
    boot = [np.concatenate([where[p] for p in rng.choice(test_pairs, len(test_pairs), replace=True)]) for _ in range(args.n_boot)]

    out = ["# %s: estimation-set size, %d vs %d pairs, DAS and mean-difference (same %d tuning pairs, same %d test pairs)" % (
               args.model, old_das.n_est_pairs, new_das.n_est_pairs, len(tune_pairs), len(test_pairs)), "",
           "Old sweeps `%s/E23/%s/knob_sweep_{%s,%s}` (%d items) re-cut to the new split defined by `%s/E23/%s` (%d items; %d pairs moved into "
           "the estimation set). Rule on the tuning pairs: E = max dJ s.t. |d ans| <= %.2f, tie-break = sweep order (dJ compared as an exact integer count). Test pairs reported with "
           "pair-bootstrap 95%% CIs (%d resamples, seed %d); differences are paired on the same resamples." % (
               args.root_old, args.model, args.tag_das, args.tag_md, old_das.n_items_total, args.root_new, args.model, len(ids), n_extra,
               E_LEAK, args.n_boot, args.seed), ""]

    # ---- baseline agreement between the old and new runs on the common items ----------------------------- #
    out += ["## Baseline agreement on the common items (same model and prompts; batches differ, so tiny numerical drift is expected)", "",
            "| pair of runs | max abs d p_true | flipped True/False predictions | flipped Yes/No self-reports |", "|---|---|---|---|"]
    for a, b in ((old_das, new_das), (old_md, new_md), (old_das, old_md), (new_das, new_md)):
        dp = np.abs(a.p["baseline"] - b.p["baseline"]).max()
        fl = int(((a.p["baseline"] > 0.5) != (b.p["baseline"] > 0.5)).sum())
        fa = int((a.ans["baseline"] != b.ans["baseline"]).sum())
        out.append("| %s vs %s | %.2e | %d / %d | %d / %d |" % (a.label, b.label, dp, fl, len(ids), fa, len(ids)))

    # ---- rule-chosen condition per sweep, test deltas with CIs ------------------------------------------- #
    chosen, full, arrs = {}, {}, {}
    out += ["", "## Rule-chosen condition of each sweep (selected on the %d tuning pairs, reported on the %d test pairs)" % (len(tune_pairs), len(test_pairs)), "",
            "| sweep | chosen | tune dJ | test dJ [CI] | test dAUROC [CI] | test d ans [CI] | test dB [CI] | in-sample dJ on its own est pairs | oracle on test (dJ) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in sweeps:
        ch = s.rule_pick(s.names, tune)
        chosen[s.label] = ch
        rec = (s.j.get("selection") or {}).get("E", {}).get("chosen")
        if s in (new_das, new_md) and rec is not None and rec != ch:
            out.append("> note: %s: recomputed choice %s differs from the sweep's recorded choice %s" % (s.label, ch, rec))
        if ch is None:
            out.append("| %s | none feasible | | | | | | | |" % s.label)
            continue
        bA, bP = s.ans["baseline"], s.p["baseline"]
        a, p = s.ans[ch], s.p[ch]
        full[s.label] = {k: metrics(a[test], p[test], lab[test])[k] - metrics(bA[test], bP[test], lab[test])[k] for k in K}
        arrs[s.label] = {k: np.array([metrics(a[ii], p[ii], lab[ii])[k] - metrics(bA[ii], bP[ii], lab[ii])[k] for ii in boot]) for k in K}
        orc = s.rule_pick(s.names, test)                     # the same rule applied on the test pairs themselves (oracle)
        out.append("| %s | %s | %+.3f | %s | %s | %s | %s | %+.3f (%d pairs) | %s (%+.3f) |" % (
            s.label, ch, s.delta(ch, tune, "J"), ci_cell(full[s.label]["J"], arrs[s.label]["J"]), ci_cell(full[s.label]["AUROC"], arrs[s.label]["AUROC"]),
            ci_cell(full[s.label]["ans"], arrs[s.label]["ans"]), ci_cell(full[s.label]["B"], arrs[s.label]["B"]),
            s.est_delta_J(ch), s.n_est_pairs, orc, s.delta(orc, test, "J") if orc else float("nan")))

    # ---- paired differences ------------------------------------------------------------------------------- #
    out += ["", "## Paired differences on the same resamples", "",
            "| difference | dJ | dAUROC | d ans | dB | P(dJ > 0) | P(dAUROC > 0) |", "|---|---|---|---|---|---|---|"]
    pairs_to_compare = [(new_das, old_das, "more data for DAS"), (new_md, old_md, "more data for mean-diff"),
                        (new_das, new_md, "DAS vs mean-diff, large estimation set"), (old_das, old_md, "DAS vs mean-diff, small estimation set")]
    diffs = {}
    for a, b, why in pairs_to_compare:
        if a.label not in full or b.label not in full:
            out.append("| %s - %s (%s) | n/a | | | | | |" % (a.label, b.label, why))
            continue
        fa, fb, aa, ab = full[a.label], full[b.label], arrs[a.label], arrs[b.label]
        diffs["%s - %s" % (a.label, b.label)] = {k: {"value": fa[k] - fb[k], "ci": [float(np.percentile(aa[k] - ab[k], 2.5)), float(np.percentile(aa[k] - ab[k], 97.5))]} for k in K}
        out.append("| %s - %s (%s) | %s | %s | %s | %s | %.3f | %.3f |" % (
            a.label, b.label, why, ci_cell(fa["J"] - fb["J"], aa["J"] - ab["J"]), ci_cell(fa["AUROC"] - fb["AUROC"], aa["AUROC"] - ab["AUROC"]),
            ci_cell(fa["ans"] - fb["ans"], aa["ans"] - ab["ans"]), ci_cell(fa["B"] - fb["B"], aa["B"] - ab["B"]),
            (aa["J"] > ab["J"]).mean(), (aa["AUROC"] > ab["AUROC"]).mean()))

    # ---- per layer ----------------------------------------------------------------------------------------- #
    out += ["", "## Per layer: alpha picked by the rule on the tuning pairs, test dJ (4 decimals); 'leak' = every alpha moves the answer rate by more than %.2f on tune" % E_LEAK, "",
            "| L | " + " | ".join("%s cfg | dJ | dAUROC" % s.label for s in sweeps) + " | DAS loss (small / large) | cos(DAS, meandiff) (small / large) |",
            "|---|" + "---|" * (3 * len(sweeps)) + "---|---|"]
    layers = sorted({layers_of(n)[0][0] for s in sweeps for n in s.names if layers_of(n)[0] and len(layers_of(n)[0]) == 1})
    per_layer = {}
    for l in layers:
        cells = [str(l)]
        per_layer[l] = {}
        for s in sweeps:
            names = [n for n in s.names if layers_of(n)[0] == [l]]
            ch = s.rule_pick(names, tune) if names else None
            if not names:
                cells += ["-", "", ""]
            elif ch is None:
                cells += ["leak", "", ""]
            else:
                dj, da = s.delta(ch, test, "J"), s.delta(ch, test, "AUROC")
                per_layer[l][s.label] = {"chosen": ch, "test_dJ": dj, "test_dAUROC": da}
                cells += [ch, "%+.4f" % dj, "%+.4f" % da]
        fit = []
        for s in (old_das, new_das):
            d = s.das.get(str(l))
            fit.append(("%.3f" % d["final_losses"][0], "%+.3f" % d["cos_with_align_to"]) if d else ("-", "-"))
        cells += ["%s / %s" % (fit[0][0], fit[1][0]), "%s / %s" % (fit[0][1], fit[1][1])]
        out.append("| " + " | ".join(cells) + " |")

    txt = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "n_tune_pairs": len(tune_pairs), "n_test_pairs": len(test_pairs), "moved_pairs": n_extra,
                   "chosen": chosen, "test_deltas": full, "paired_differences": diffs, "per_layer": per_layer}, f, indent=1)
    print(txt)


if __name__ == "__main__":
    main()
