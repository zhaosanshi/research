"""Merge E23 knob-sweep chunks (same tune/test split, different layer ranges) and do the
pre-declared selection once over ALL layers (赵磊 2026-10-30: scan every layer, select on the
192-pair tuning set, report on the 529-pair test set, then compare with the original plan's LA-Score).

Outputs results_tqa/E23/<model>/knob_sweep_<out-tag>.{md,json}:
  * selection on the tuning set -> effect on the test set with pair-bootstrap 95% CIs
  * test-set oracle under the same rule -> selection error in layers (the plan's E3 quantity)
  * LA-Score comparison: W*/R* vs the chosen layer, Spearman rho between S(l) (and sep(l)) and the
    measured per-layer effect, and the full per-layer table
usage: python3 scripts/e23_merge.py --model llama3.1-8b --tags all_a all_b --out-tag all
Runs on CPU (numpy only); results dirs are read relative to --root.
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import auroc, sel_key  # Share the sweep's tie-aware AUROC and exact-count selection key.

E_LEAK, C_J, C_B = 0.03, 0.02, 0.05          # the pre-declared rule


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def layers_of(name):
    m = re.search(r"_L([\d-]+)_", name)
    return [int(x) for x in m.group(1).split("-")] if m else []


def centre(name):
    L = layers_of(name)
    return float(np.mean(L)) if L else float("nan")


def j_stat(pred, lab):
    return float(pred[lab].mean() - pred[~lab].mean())


def metrics(ans, p, lab):
    pred = p > 0.5
    return {"ans": float(ans.mean()), "J": j_stat(pred, lab), "AUROC": auroc(p, lab), "B": float(pred.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out-tag", default="all")
    ap.add_argument("--la-score", default=None, help="layer_scores.json (default results_tqa/E1/<model>/layer_scores.json)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c-prefix", default="Cp", help="which C estimator's conditions to select over: Cp (prompt contrast) or Cs (self-report)")
    args = ap.parse_args()
    d = os.path.join(args.root, "results_tqa", "E23", args.model)

    # ---- load and check chunks ------------------------------------------------ #
    conds, per, ids, lab, half_a = {}, {}, None, None, None
    base_check = None
    for t in args.tags:
        j = json.load(open(os.path.join(d, "knob_sweep_%s.json" % t)))
        z = np.load(os.path.join(d, "knob_sweep_%s.npz" % t), allow_pickle=True)
        if ids is None:
            ids, lab, half_a = z["ids"], z["labels"], z["half_a"]
        else:
            if not (np.array_equal(ids, z["ids"]) and np.array_equal(half_a, z["half_a"])):
                raise SystemExit("chunk %s has a different item order or tuning split; refusing to merge" % t)
        b = j["conds"]["baseline"]
        if base_check is None:
            base_check = b
        else:
            for h in ("A", "B"):
                for k in ("ans", "J", "B"):
                    if abs(b[h][k] - base_check[h][k]) > 1e-6:
                        raise SystemExit("baseline differs between chunks (%s %s): %.4f vs %.4f" % (h, k, b[h][k], base_check[h][k]))
        for name, m in j["conds"].items():
            if name in conds and name != "baseline":
                raise SystemExit("duplicate condition %s across chunks" % name)
            conds[name] = m
            per[name] = (z[name + "__answer"].astype(bool), z[name + "__ptrue"].astype(float))
    pair = np.array([i.rsplit(":", 1)[0] for i in ids])
    tune, test = half_a.astype(bool), ~half_a.astype(bool)
    n_tune, n_test = len(set(pair[tune])), len(set(pair[test]))

    def delta(name, h, k):
        return conds[name][h][k] - conds["baseline"][h][k]

    # ---- selection on the tuning set (A), oracle on the test set (B) ----------- #
    # Review round 14 (2026-11-04): conditions are ordered numerically (layer centre, then alpha; a string sort put L10 before
    # L2), ties are compared on exact integer counts (core.metrics.sel_key), and an E rule with no feasible condition
    # returns None instead of a candidate that violates the answer-rate constraint.
    def alpha_of(name):
        m = re.search(r"_a([0-9.]+)$", name)
        return float(m.group(1)) if m else float("nan")

    def sweep_order(name):   # the sweep records single-layer conditions first (layer, alpha ascending), then the multi-layer spans
        return (len(layers_of(name)) > 1, centre(name), alpha_of(name), name)
    E = sorted((n for n in conds if n.startswith("E_")), key=sweep_order)
    cp = args.c_prefix + "_"
    Cab = sorted((n for n in conds if n.startswith(cp) and "_abst_" in n), key=sweep_order)
    Can = sorted((n for n in conds if n.startswith(cp) and "_ans_" in n), key=sweep_order)
    ys = {"A": lab.astype(bool)[tune], "B": lab.astype(bool)[test]}

    def scale_J(h):
        return int(ys[h].sum()) * int((~ys[h]).sum())

    def pick_E(h):
        ok = [n for n in E if abs(delta(n, h, "ans")) <= E_LEAK + 1e-12]
        return (max(ok, key=lambda n: sel_key(delta(n, h, "J"), scale_J(h))), True) if ok else (None, False)

    def pick_C(h, names, sign):
        ok = [n for n in names if abs(delta(n, h, "J")) <= C_J + 1e-12 and abs(delta(n, h, "B")) <= C_B + 1e-12 and sign * delta(n, h, "ans") > 0]
        return max(ok, key=lambda n: sel_key(sign * delta(n, h, "ans"), int(len(ys[h])))) if ok else None

    # ---- pair bootstrap on the test set ---------------------------------------- #
    rng = np.random.default_rng(args.seed)
    test_pairs = np.array(sorted(set(pair[test])))
    where = {p: np.where((pair == p) & test)[0] for p in test_pairs}
    boot_idx = [np.concatenate([where[p] for p in rng.choice(test_pairs, len(test_pairs), replace=True)]) for _ in range(args.n_boot)]
    bA, bP = per["baseline"]

    def ci(name):
        a, p = per[name]
        out = {}
        for k in ("ans", "J", "AUROC", "B"):
            vals = np.array([metrics(a[ii], p[ii], lab[ii])[k] - metrics(bA[ii], bP[ii], lab[ii])[k] for ii in boot_idx])
            out[k] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        return out

    sel = {}
    e_ch, e_or, e_ok = None, None, False
    if E:
        e_ch, e_ok = pick_E("A"); e_or, _ = pick_E("B")
    if e_ch is not None:
        sel["E"] = {"chosen": e_ch, "feasible": e_ok, "oracle": e_or, "test": {k: delta(e_ch, "B", k) for k in ("ans", "J", "AUROC", "B")}, "ci": ci(e_ch),
                    "oracle_test": {k: delta(e_or, "B", k) for k in ("ans", "J", "AUROC", "B")} if e_or else None,
                    "layer_error": abs(centre(e_ch) - centre(e_or)) if e_or else float("nan")}
    else:   # no E conditions, or none within the answer-rate constraint: report none, never a violating fallback
        sel["E"] = {"chosen": None, "feasible": False, "oracle": e_or, "test": None, "ci": None,
                    "oracle_test": {k: delta(e_or, "B", k) for k in ("ans", "J", "AUROC", "B")} if e_or else None, "layer_error": float("nan")}
    for key, names, sign in (("C_abst", Cab, -1), ("C_ans", Can, 1)):
        ch, orc = pick_C("A", names, sign), pick_C("B", names, sign)
        sel[key] = {"chosen": ch, "oracle": orc,
                    "test": {k: delta(ch, "B", k) for k in ("ans", "J", "AUROC", "B")} if ch else None, "ci": ci(ch) if ch else None,
                    "oracle_test": {k: delta(orc, "B", k) for k in ("ans", "J", "AUROC", "B")} if orc else None,
                    "layer_error": abs(centre(ch) - centre(orc)) if (ch and orc) else float("nan")}

    # ---- LA-Score comparison --------------------------------------------------- #
    la_path = args.la_score or os.path.join(args.root, "results_tqa", "E1", args.model, "layer_scores.json")
    la = json.load(open(la_path)) if os.path.exists(la_path) else None
    layers = sorted({layers_of(n)[0] for n in conds if n.startswith(("E_", cp)) and len(layers_of(n)) == 1})
    rows = []
    for l in layers:
        e3 = "E_L%d_a3" % l; c2 = "%sL%d_abst_a2" % (cp, l); c2a = "%sL%d_ans_a2" % (cp, l)
        r = {"layer": l,
             "E_dJ_a3": delta(e3, "all", "J") if e3 in conds else float("nan"), "E_dans_a3": delta(e3, "all", "ans") if e3 in conds else float("nan"),
             "C_dans_abst_a2": delta(c2, "all", "ans") if c2 in conds else float("nan"), "C_dJ_abst_a2": delta(c2, "all", "J") if c2 in conds else float("nan"),
             "C_dB_abst_a2": delta(c2, "all", "B") if c2 in conds else float("nan"), "C_dans_ans_a2": delta(c2a, "all", "ans") if c2a in conds else float("nan")}
        if la:
            rE, rC = la["E"]["rows"].get(str(l), {}), la["C"]["rows"].get(str(l), {})
            r.update({"S_E": rE.get("S", float("nan")), "sep_E": rE.get("sep", float("nan")), "S_C": rC.get("S", float("nan")), "dact_C": rC.get("dact", float("nan"))})
        rows.append(r)
    la_cmp = None
    if la:
        la_cmp = {"E_write_layer": la["E"]["write_layer"], "E_read_layer": la["E"]["read_layer"], "C_write_layer": la["C"]["write_layer"],
                  "rho_S_E_vs_dJ": spearman([r["S_E"] for r in rows], [r["E_dJ_a3"] for r in rows]),
                  "rho_sep_E_vs_dJ": spearman([r["sep_E"] for r in rows], [r["E_dJ_a3"] for r in rows]),
                  "rho_S_C_vs_dans": spearman([r["S_C"] for r in rows], [-r["C_dans_abst_a2"] for r in rows]),
                  "rho_dact_C_vs_dans": spearman([r["dact_C"] for r in rows], [-r["C_dans_abst_a2"] for r in rows]),
                  "E_chosen_centre": centre(e_ch) if e_ch else float("nan"), "C_abst_chosen_centre": centre(sel["C_abst"]["chosen"]) if sel["C_abst"]["chosen"] else float("nan")}

    res = {"model": args.model, "tags": args.tags, "n_tune_pairs": n_tune, "n_test_pairs": n_test, "rule": {"E_leak": E_LEAK, "C_J": C_J, "C_B": C_B},
           "selection": sel, "la_score": la_cmp, "per_layer": rows, "conds": conds}
    stem = os.path.join(d, "knob_sweep_%s" % args.out_tag)
    json.dump(res, open(stem + ".json", "w"), indent=1)

    def fmt(v, ci_):
        return "%+.3f [%+.3f, %+.3f]" % (v, ci_[0], ci_[1])
    L = ["# E23 all-layer knob sweep, %s (tune %d pairs -> test %d pairs; chunks %s)" % (args.model, n_tune, n_test, " ".join(args.tags)), "",
         "Rule (pre-declared, applied on the tuning set): E = max dJ s.t. |d ans| <= %.2f; C = max |d ans| s.t. |dJ| <= %.2f, |dB| <= %.2f. Reported on the test set with pair-bootstrap 95%% CIs (%d resamples). Oracle = the same rule applied on the test set itself; layer error = |chosen centre - oracle centre|." % (E_LEAK, C_J, C_B, args.n_boot), "",
         "## Selection and the 2x2 on the test set", "",
         "| knob | chosen on tune | test: d ans | d J | d AUROC | d B | oracle on test | layer error |", "|---|---|---|---|---|---|---|---|"]
    for key in ("E", "C_abst", "C_ans"):
        s = sel[key]
        if s["chosen"]:
            t, c = s["test"], s["ci"]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                key, s["chosen"], fmt(t["ans"], c["ans"]), fmt(t["J"], c["J"]), fmt(t["AUROC"], c["AUROC"]), fmt(t["B"], c["B"]),
                ("%s (%+.3f ans, %+.3f J)" % (s["oracle"], s["oracle_test"]["ans"], s["oracle_test"]["J"])) if s["oracle"] else "none feasible",
                ("%.1f" % s["layer_error"]) if s["oracle"] else ""))
        else:
            L.append("| %s | none feasible | | | | | %s | |" % (key, s["oracle"] or "none feasible"))
    if la_cmp:
        L += ["", "## LA-Score (the original plan's training-free layer score) vs measured effect", "",
              "| quantity | value |", "|---|---|",
              "| LA-Score E write layer W* / read layer R* | %s / %s |" % (la_cmp["E_write_layer"], la_cmp["E_read_layer"]),
              "| effect-selected E (centre layer) | %s (%.1f) |" % (e_ch, la_cmp["E_chosen_centre"]) if e_ch else
              "| effect-selected E | (%s) |" % ("no E condition within the answer-rate limit" if E else "no E conditions in this run"),
              "| LA-Score C write layer | %s |" % la_cmp["C_write_layer"],
              "| effect-selected C toward abstention (centre layer) | %s (%.1f) |" % (sel["C_abst"]["chosen"], la_cmp["C_abst_chosen_centre"]),
              "| Spearman rho: S_E(l) vs dJ(E, a3) over layers | %.2f |" % la_cmp["rho_S_E_vs_dJ"],
              "| Spearman rho: sep_E(l) (probe AUROC) vs dJ(E, a3) | %.2f |" % la_cmp["rho_sep_E_vs_dJ"],
              "| Spearman rho: S_C(l) vs |d ans|(C abst, a2) | %.2f |" % la_cmp["rho_S_C_vs_dans"],
              "| Spearman rho: dact_C(l) vs |d ans|(C abst, a2) | %.2f |" % la_cmp["rho_dact_C_vs_dans"]]
    L += ["", "## Per layer (all 721 pairs; d = change vs baseline)", "",
          "| layer | S_E | sep_E | E a3: dJ | E a3: d ans | S_C | C abst a2: d ans | dJ | dB | C ans a2: d ans |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %d | %.2f | %.3f | %+.3f | %+.3f | %.2f | %+.3f | %+.3f | %+.3f | %+.3f |" % (
            r["layer"], r.get("S_E", float("nan")), r.get("sep_E", float("nan")), r["E_dJ_a3"], r["E_dans_a3"], r.get("S_C", float("nan")), r["C_dans_abst_a2"], r["C_dJ_abst_a2"], r["C_dB_abst_a2"], r["C_dans_ans_a2"]))
    L += ["", "## All conditions (tune A / test B; d = change vs baseline)", "", "| condition | tune: d ans | d J | d B | test: d ans | d J | d AUROC | d B |", "|---|---|---|---|---|---|---|---|"]
    for n in sorted(conds, key=lambda n: (n.split("_")[0], centre(n) if layers_of(n) else -1, n)):
        if n == "baseline":
            continue
        L.append("| %s | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f |" % (n, delta(n, "A", "ans"), delta(n, "A", "J"), delta(n, "A", "B"), delta(n, "B", "ans"), delta(n, "B", "J"), delta(n, "B", "AUROC"), delta(n, "B", "B")))
    open(stem + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:14]))
    print("written", stem + ".md")


if __name__ == "__main__":
    main()
