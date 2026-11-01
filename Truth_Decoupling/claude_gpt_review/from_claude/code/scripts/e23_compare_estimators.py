"""Paired comparison of two E23 knob sweeps that differ only in the E estimator (e.g. DAS vs mean-difference).

Both sweeps must come from the same model, the same items and the same tuning/test split (checked). For each
sweep the pre-declared rule is applied on the tuning half (E: max dJ s.t. |d ans| <= 0.03) and the chosen
condition is reported on the test half with pair-bootstrap 95% CIs. Because the two sweeps score the SAME test
items, the difference "estimator A - estimator B" is bootstrapped on the same resamples (paired), which is the
honest way to ask "is A better than B here".

Metric definitions are copied from e23_merge.py (pred = p_true > 0.5; J = TPR - FPR; AUROC by ranks; B = mean pred).
Runs on CPU (numpy only).

usage: python3 scripts/e23_compare_estimators.py --root /path/to/results_das_tf --model llama3.1-8b \
           --tag-a das_tf_full --label-a DAS --tag-b meandiff_full --label-b meandiff --out /path/out.md
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import auroc  # Same tie-aware implementation as the E23 sweep.

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


def rule_pick(conds, names):
    """Pre-declared E rule on the tuning half: feasible = |d ans| <= E_LEAK; best = max dJ.
    Tie-break = the sweep's own: plain max over the conditions in the sweep's recorded order (layer ascending, alpha
    ascending), so equal tune dJ -- quantised at 1/192 on 192 pairs, exact ties happen (Mistral-7B: L11/L12/L13 a3) --
    resolves to the first (lowest layer, smaller alpha) exactly as e23_knob_sweep.py recorded it. An earlier version
    broke ties by name and could pick a different layer than the sweep."""
    b = conds["baseline"]
    ok = [n for n in names if abs(conds[n]["A"]["ans"] - b["A"]["ans"]) <= E_LEAK + 1e-12]
    if not ok:
        return None
    return max(ok, key=lambda n: conds[n]["A"]["J"] - b["A"]["J"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results root holding E23/<model>/knob_sweep_<tag>.{json,npz}")
    ap.add_argument("--root-b", default=None, help="results root of sweep B when it lives elsewhere (e.g. the old-anchor DAS run under results_das_tf_cur); default: --root")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag-a", required=True); ap.add_argument("--label-a", default=None)
    ap.add_argument("--tag-b", required=True); ap.add_argument("--label-b", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="markdown report path (refuses to overwrite)")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("refusing to overwrite %s" % args.out)
    la, lb = args.label_a or args.tag_a, args.label_b or args.tag_b
    d = os.path.join(args.root, "E23", args.model)
    d_b = os.path.join(args.root_b, "E23", args.model) if args.root_b else d

    def load(dir_, tag):
        j = json.load(open(os.path.join(dir_, "knob_sweep_%s.json" % tag)))
        z = np.load(os.path.join(dir_, "knob_sweep_%s.npz" % tag), allow_pickle=True)
        return j, z
    ja, za = load(d, args.tag_a); jb, zb = load(d_b, args.tag_b)
    if not (np.array_equal(za["ids"], zb["ids"]) and np.array_equal(za["half_a"], zb["half_a"])):
        raise SystemExit("the two sweeps have different items or a different tuning split; a paired comparison is meaningless")
    if ja["model"] != jb["model"] or ja.get("wording") != jb.get("wording") or ja.get("e_alphas") != jb.get("e_alphas"):
        raise SystemExit("model / wording / alphas differ between the sweeps")
    lab = za["labels"].astype(bool); test = ~za["half_a"].astype(bool)
    pair = np.array([i.rsplit(":", 1)[0] for i in za["ids"]])
    base = {}
    for tag, z in (("a", za), ("b", zb)):
        base[tag] = (z["baseline__answer"].astype(bool), z["baseline__ptrue"].astype(float))
    same_base = np.array_equal(base["a"][0], base["b"][0]) and np.allclose(base["a"][1], base["b"][1], atol=1e-6)

    rng = np.random.default_rng(args.seed)
    test_pairs = np.array(sorted(set(pair[test])))
    where = {p: np.where((pair == p) & test)[0] for p in test_pairs}
    boot = [np.concatenate([where[p] for p in rng.choice(test_pairs, len(test_pairs), replace=True)]) for _ in range(args.n_boot)]

    def deltas(z, name, bA, bP):
        a, p = z[name + "__answer"].astype(bool), z[name + "__ptrue"].astype(float)
        full = {k: metrics(a[test], p[test], lab[test])[k] - metrics(bA[test], bP[test], lab[test])[k] for k in K}
        arr = {k: np.array([metrics(a[ii], p[ii], lab[ii])[k] - metrics(bA[ii], bP[ii], lab[ii])[k] for ii in boot]) for k in K}
        return full, arr

    def ci_cell(v, arr):
        return "%+.3f [%+.3f, %+.3f]" % (v, np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    out = ["# %s: %s vs %s (paired, same %d test pairs)" % (args.model, la, lb, len(test_pairs)), "",
           "Sweeps `%s` / `%s` under `%s`%s. Rule on the tuning half (%d pairs): E = max dJ s.t. |d ans| <= %.2f. "
           "Test half reported with pair-bootstrap 95%% CIs (%d resamples, seed %d). Baseline arrays identical across the two sweeps: %s."
           % (args.tag_a, args.tag_b, d, (" / `%s`" % d_b) if d_b != d else "", int(za["half_a"].sum()) // 2, E_LEAK, args.n_boot, args.seed, same_base), ""]
    e = ja["conds"]["baseline"].get("est")
    if e:
        tpr, fpr = e["B"] + e["J"] / 2, e["B"] - e["J"] / 2
        out += ["Baseline on the estimation pairs (the DAS training pairs; balanced classes): J=%.3f, B=%.3f -> TPR=%.3f, FPR=%.3f "
                "(the model answers False to %.1f%% of true statements and True to %.1f%% of false ones)." % (e["J"], e["B"], tpr, fpr, 100 * (1 - tpr), 100 * fpr), ""]

    # ---- chosen conditions, CIs, paired difference ------------------------------------------ #
    res = {}
    out += ["## Rule-chosen condition of each estimator (test half)", "",
            "| estimator | chosen on tune | tune dJ | test dJ [CI] | test dAUROC [CI] | test d ans [CI] | test dB [CI] | oracle on test (dJ) |", "|---|---|---|---|---|---|---|---|"]
    for lbl, j, z, bt in ((la, ja, za, base["a"]), (lb, jb, zb, base["b"])):
        names = [n for n in j["conds"] if n.startswith("E_")]
        ch = rule_pick(j["conds"], names)
        rec = j.get("selection", {}).get("E", {}).get("chosen")
        if rec is not None and ch != rec:
            out.append("> note: recomputed choice %s differs from the sweep's recorded choice %s" % (ch, rec))
        b = j["conds"]["baseline"]
        orc = max((n for n in names if abs(j["conds"][n]["B"]["ans"] - b["B"]["ans"]) <= E_LEAK), key=lambda n: j["conds"][n]["B"]["J"] - b["B"]["J"])
        full, arr = deltas(z, ch, *bt); res[lbl] = (ch, full, arr)
        out.append("| %s | %s | %+.3f | %s | %s | %s | %s | %s (%+.3f) |" % (
            lbl, ch, j["conds"][ch]["A"]["J"] - b["A"]["J"], ci_cell(full["J"], arr["J"]), ci_cell(full["AUROC"], arr["AUROC"]),
            ci_cell(full["ans"], arr["ans"]), ci_cell(full["B"], arr["B"]), orc, j["conds"][orc]["B"]["J"] - b["B"]["J"]))
    (cha, fa, aa), (chb, fb, ab) = res[la], res[lb]
    out += ["", "## Paired difference %s(%s) - %s(%s) on the same resamples" % (la, cha, lb, chb), "",
            "| dJ | dAUROC | d ans | dB | P(dJ_%s > dJ_%s) | P(dAUROC_%s > dAUROC_%s) |" % (la, lb, la, lb), "|---|---|---|---|---|---|",
            "| %s | %s | %s | %s | %.3f | %.3f |" % (ci_cell(fa["J"] - fb["J"], aa["J"] - ab["J"]), ci_cell(fa["AUROC"] - fb["AUROC"], aa["AUROC"] - ab["AUROC"]),
                                                   ci_cell(fa["ans"] - fb["ans"], aa["ans"] - ab["ans"]), ci_cell(fa["B"] - fb["B"], aa["B"] - ab["B"]),
                                                   (aa["J"] > ab["J"]).mean(), (aa["AUROC"] > ab["AUROC"]).mean()), ""]

    # ---- per layer (single-layer conditions) -------------------------------------------------- #
    def per_layer(j):
        b = j["conds"]["baseline"]; tab = {}
        for n in j["conds"]:
            L, a = layers_of(n)
            if L and len(L) == 1:
                tab.setdefault(L[0], []).append(n)
        rows = {}
        for l, names in tab.items():
            ch = rule_pick(j["conds"], names)
            rows[l] = (ch, {k: j["conds"][ch]["B"][k] - b["B"][k] for k in K} if ch else None,
                       [(n, j["conds"][n]["A"]["ans"] - b["A"]["ans"]) for n in sorted(names)])
        return rows
    ra, rb = per_layer(ja), per_layer(jb)
    pla = (ja.get("das") or {}).get("per_layer") or {}; plb = (jb.get("das") or {}).get("per_layer") or {}
    out += ["## Per layer, single-layer conditions (alpha picked by the rule on tune; test-half deltas; 'none' = every alpha leaks > %.2f on tune, tune d ans shown)" % E_LEAK, "",
            "| L | %s cfg | dJ | dAUROC | d ans | dB | %s cfg | dJ | dAUROC | d ans | dB | DAS loss / cos(DAS, meandiff) |" % (la, lb), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in sorted(set(ra) | set(rb)):
        cells = [str(l)]
        for rows in (ra, rb):
            r = rows.get(l)
            if r is None:
                cells += ["-", "", "", "", ""]
            elif r[0] is None:
                cells += ["none (tune d ans %s)" % ", ".join("%+.3f" % v for _, v in r[2]), "", "", "", ""]
            else:
                cells += [r[0]] + ["%+.4f" % r[1][k] for k in K]
        extra = []
        for pl in (pla, plb):
            if str(l) in pl:
                p = pl[str(l)]; extra.append("%.3f / %+.3f" % (p["final_losses"][0], p["cos_with_align_to"]))
        cells.append("; ".join(extra))
        out.append("| " + " | ".join(cells) + " |")

    # ---- multi-layer spans, if any -------------------------------------------------------------- #
    for lbl, j in ((la, ja), (lb, jb)):
        b = j["conds"]["baseline"]
        sp = [n for n in j["conds"] if layers_of(n)[0] and len(layers_of(n)[0]) > 1]
        if sp:
            out += ["", "## Multi-layer E conditions, %s (tune / test deltas; feasible = |tune d ans| <= %.2f)" % (lbl, E_LEAK), "",
                    "| condition | feasible | tune dJ | tune d ans | test dJ | test dAUROC | test d ans | test dB |", "|---|---|---|---|---|---|---|---|"]
            for n in sorted(sp, key=lambda n: (layers_of(n)[0], layers_of(n)[1])):
                c = j["conds"][n]; fe = abs(c["A"]["ans"] - b["A"]["ans"]) <= E_LEAK + 1e-12
                out.append("| %s | %s | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f |" % (
                    n, "yes" if fe else "no", c["A"]["J"] - b["A"]["J"], c["A"]["ans"] - b["A"]["ans"],
                    c["B"]["J"] - b["B"]["J"], c["B"]["AUROC"] - b["B"]["AUROC"], c["B"]["ans"] - b["B"]["ans"], c["B"]["B"] - b["B"]["B"]))
    txt = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
