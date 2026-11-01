#!/usr/bin/env python3
"""The 2x2 of one model from two E23 sweeps: an E sweep (E conditions) and a C sweep (Cp / Cs conditions).

Both sweeps must score the same items with the same tuning / test split (checked), e.g. the DAS-tf E sweep
(knob_sweep_das_tf_spans) and the C sweep whose C directions were orthogonalised against that same E
(knob_sweep_das_tf_C). The pre-declared rules of e23_knob_sweep.py are re-applied on the tuning half:
  E          : max dJ  s.t. |d ans| <= 0.03
  C (abst)   : max -d ans s.t. |dJ| <= 0.02, |dB| <= 0.05, d ans < 0      (separately for Cp and Cs)
  C (ans)    : max +d ans s.t. |dJ| <= 0.02, |dB| <= 0.05, d ans > 0
and every chosen condition is reported on the test half with pair-bootstrap 95% CIs. The 2x2 reads: E should move
J / AUROC and not the answer rate; C should move the answer rate and not J / AUROC / B (anti-diagonal ~ 0).
Runs on CPU (numpy only).

usage: python3 scripts/report_2x2.py --e-root results_das_tf --e-tag das_tf_spans --c-root results_das_tf --c-tag das_tf_C \
           --model llama3.1-8b --out results_das_tf/compare/llama3.1-8b_2x2_das_tf.md
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics import auroc  # tie-aware, same as the sweep

E_LEAK, C_J, C_B = 0.03, 0.02, 0.05
K = ("ans", "J", "AUROC", "B")


def metrics(ans, p, lab):
    pred = p > 0.5
    return {"ans": float(ans.mean()), "J": float(pred[lab].mean() - pred[~lab].mean()), "AUROC": auroc(p, lab), "B": float(pred.mean())}


def alpha_of(name):
    m = re.search(r"_a([\d.]+)$", name)
    return float(m.group(1)) if m else float("nan")


def layer_of(name):
    m = re.match(r"^C[ps]_L(\d+)_", name)
    return int(m.group(1)) if m else None


def pick_e(conds):
    b = conds["baseline"]
    names = [n for n in conds if n.startswith("E_")]
    ok = [n for n in names if abs(conds[n]["A"]["ans"] - b["A"]["ans"]) <= E_LEAK + 1e-12]
    return (max(ok, key=lambda n: (conds[n]["A"]["J"] - b["A"]["J"], -alpha_of(n), n)) if ok else None), len(ok), len(names)


def pick_c(conds, kind, dname):
    sign = -1 if dname == "abst" else 1
    b = conds["baseline"]
    names = [n for n in conds if n.startswith(kind + "_") and ("_%s_" % dname) in n]
    ok = [n for n in names if abs(conds[n]["A"]["J"] - b["A"]["J"]) <= C_J + 1e-12 and abs(conds[n]["A"]["B"] - b["A"]["B"]) <= C_B + 1e-12
          and sign * (conds[n]["A"]["ans"] - b["A"]["ans"]) > 0]
    return (max(ok, key=lambda n: (sign * (conds[n]["A"]["ans"] - b["A"]["ans"]), -alpha_of(n), n)) if ok else None), len(ok), len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-root", required=True); ap.add_argument("--e-tag", required=True)
    ap.add_argument("--c-root", required=True); ap.add_argument("--c-tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="markdown report path (refuses to overwrite)")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit("refusing to overwrite %s" % args.out)

    def load(root, tag):
        d = os.path.join(root, "E23", args.model)
        return json.load(open(os.path.join(d, "knob_sweep_%s.json" % tag))), np.load(os.path.join(d, "knob_sweep_%s.npz" % tag), allow_pickle=True)
    je, ze = load(args.e_root, args.e_tag)
    jc, zc = load(args.c_root, args.c_tag)
    if not (np.array_equal(ze["ids"], zc["ids"]) and np.array_equal(ze["half_a"], zc["half_a"]) and np.array_equal(ze["labels"], zc["labels"])):
        raise SystemExit("the E and C sweeps have different items, labels or tuning split")
    if je["model"] != jc["model"] or je.get("wording") != jc.get("wording"):
        raise SystemExit("model / wording differ between the sweeps")
    lab = ze["labels"].astype(bool); test = ~ze["half_a"].astype(bool)
    pair = np.array([i.rsplit(":", 1)[0] for i in ze["ids"]])
    same_base = (np.array_equal(ze["baseline__answer"], zc["baseline__answer"]) and np.allclose(ze["baseline__ptrue"], zc["baseline__ptrue"], atol=1e-6))
    rng = np.random.default_rng(args.seed)
    test_pairs = np.array(sorted(set(pair[test])))
    where = {p: np.where((pair == p) & test)[0] for p in test_pairs}
    boot = [np.concatenate([where[p] for p in rng.choice(test_pairs, len(test_pairs), replace=True)]) for _ in range(args.n_boot)]

    def deltas(z, name):
        a, p = z[name + "__answer"].astype(bool), z[name + "__ptrue"].astype(float)
        bA, bP = z["baseline__answer"].astype(bool), z["baseline__ptrue"].astype(float)
        full = {k: metrics(a[test], p[test], lab[test])[k] - metrics(bA[test], bP[test], lab[test])[k] for k in K}
        arr = {k: np.array([metrics(a[ii], p[ii], lab[ii])[k] - metrics(bA[ii], bP[ii], lab[ii])[k] for ii in boot]) for k in K}
        return full, arr

    def cell(v, arr):
        return "%+.3f [%+.3f, %+.3f]" % (v, np.percentile(arr, 2.5), np.percentile(arr, 97.5))

    bt = metrics(ze["baseline__answer"].astype(bool)[test], ze["baseline__ptrue"].astype(float)[test], lab[test])
    out = ["# %s: the 2x2 from E sweep `%s` and C sweep `%s`" % (args.model, args.e_tag, args.c_tag), "",
           "E sweep under `%s`, C sweep under `%s` (E estimator of the C sweep: %s%s). Rules re-applied on the tuning half (%d pairs); test half (%d pairs) "
           "with pair-bootstrap 95%% CIs (%d resamples, seed %d). Baseline arrays identical across the two sweeps: %s. Test baseline: answer rate %.3f, J %+.3f, AUROC %.3f, B %.3f." % (
               os.path.join(args.e_root, "E23", args.model), os.path.join(args.c_root, "E23", args.model), jc.get("e_estimator"),
               (" variant %s" % jc["das"]["variant"]) if jc.get("das") else "", int(ze["half_a"].sum()) // 2, len(test_pairs), args.n_boot, args.seed, same_base,
               bt["ans"], bt["J"], bt["AUROC"], bt["B"]), ""]

    # ---- chosen knobs ---- #
    rows = []
    chE, okE, nE = pick_e(je["conds"])
    rec = (je.get("selection") or {}).get("E", {}).get("chosen")
    if chE is None:
        rows.append(("E (amplify)", None, "none feasible of %d" % nE, None, None))
    else:
        f, a = deltas(ze, chE)
        rows.append(("E (amplify)", chE, "%d feasible of %d" % (okE, nE), f, a))
        if rec is not None and rec != chE:
            out.append("> note: recomputed E choice %s differs from the sweep's recorded choice %s" % (chE, rec))
    for kind, kname in (("Cp", "C prompt-contrast"), ("Cs", "C self-report")):
        for dname, dlabel in (("abst", "toward abstention"), ("ans", "toward answering")):
            ch, ok, n = pick_c(jc["conds"], kind, dname)
            if n == 0:
                continue
            rec = ((jc.get("selection") or {}).get("%s_%s" % (kind, dname)) or {}).get("chosen")
            if rec is not None and rec != ch:
                out.append("> note: recomputed %s_%s choice %s differs from the sweep's recorded choice %s" % (kind, dname, ch, rec))
            if ch is None:
                rows.append(("%s, %s" % (kname, dlabel), None, "none feasible of %d" % n, None, None))
            else:
                f, a = deltas(zc, ch)
                rows.append(("%s, %s" % (kname, dlabel), ch, "%d feasible of %d" % (ok, n), f, a))
    out += ["## Rule-chosen knobs (test half)", "",
            "| knob | chosen on tune | feasibility | d answer rate | dJ | dAUROC | dB |", "|---|---|---|---|---|---|---|"]
    for name, ch, feas, f, a in rows:
        if ch is None:
            out.append("| %s | - | %s | | | | |" % (name, feas))
        else:
            out.append("| %s | %s | %s | %s | %s | %s | %s |" % (name, ch, feas, cell(f["ans"], a["ans"]), cell(f["J"], a["J"]), cell(f["AUROC"], a["AUROC"]), cell(f["B"], a["B"])))
    out += ["", "Reading: the E row should move J / AUROC and leave the answer rate inside +-0.03; the C rows should move the answer rate and leave J within +-0.02 and B within +-0.05 (anti-diagonal). "
            "Both C estimators are shown; the protocol picks the estimator per model on the tuning set.", ""]

    # ---- per layer C table ---- #
    bc = jc["conds"]["baseline"]
    layers = sorted({layer_of(n) for n in jc["conds"] if layer_of(n) is not None})
    out += ["## C per layer (alpha picked by the rule on tune; test-half deltas; 'leak' = no alpha satisfies the J / B constraints on tune)", "",
            "| L | Cp abst | d ans | dJ | dB | Cp ans | d ans | dJ | dB | Cs abst | d ans | dJ | dB | Cs ans | d ans | dJ | dB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in layers:
        cells = [str(l)]
        for kind in ("Cp", "Cs"):
            for dname in ("abst", "ans"):
                sign = -1 if dname == "abst" else 1
                names = [n for n in jc["conds"] if n.startswith("%s_L%d_%s_" % (kind, l, dname))]
                if not names:
                    cells += ["-", "", "", ""]
                    continue
                ok = [n for n in names if abs(jc["conds"][n]["A"]["J"] - bc["A"]["J"]) <= C_J + 1e-12 and abs(jc["conds"][n]["A"]["B"] - bc["A"]["B"]) <= C_B + 1e-12
                      and sign * (jc["conds"][n]["A"]["ans"] - bc["A"]["ans"]) > 0]
                if not ok:
                    worst = max(names, key=lambda n: sign * (jc["conds"][n]["A"]["ans"] - bc["A"]["ans"]))
                    cells += ["leak (a%g: d ans %+.3f, dJ %+.3f, dB %+.3f on tune)" % (alpha_of(worst), jc["conds"][worst]["A"]["ans"] - bc["A"]["ans"],
                                                                                     jc["conds"][worst]["A"]["J"] - bc["A"]["J"], jc["conds"][worst]["A"]["B"] - bc["A"]["B"]), "", "", ""]
                    continue
                ch = max(ok, key=lambda n: (sign * (jc["conds"][n]["A"]["ans"] - bc["A"]["ans"]), -alpha_of(n), n))
                m = jc["conds"][ch]["B"]
                cells += ["a%g" % alpha_of(ch), "%+.3f" % (m["ans"] - bc["B"]["ans"]), "%+.3f" % (m["J"] - bc["B"]["J"]), "%+.3f" % (m["B"] - bc["B"]["B"])]
        out.append("| " + " | ".join(cells) + " |")
    # cos(C, E) per layer if the sweep logged it is not in the json; note the orthogonalisation instead
    out += ["", "C directions were orthogonalised against the E direction of the same layer before use (sweep setting: E estimator %s%s)." % (
        jc.get("e_estimator"), (", DAS variant %s" % jc["das"]["variant"]) if jc.get("das") else ""), ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out[:16]))
    print("written %s" % args.out)


if __name__ == "__main__":
    main()
