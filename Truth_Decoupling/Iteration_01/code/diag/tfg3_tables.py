#!/usr/bin/env python3
"""Readers for the Gemma-3 true/false run of queue M section 2 (executor G, Claude Opus 5.5, 2026-11-24). CPU only, reads
the sweep files of Iteration_01/logs/newsets_20261119/tfg3_model.sh; prints numbers, condition names and pair counts only
(no statement text).

  smoke  --res RES --model M --layer L
         the smoke sweeps knob_sweep_smoke_das / smoke_md: same items and bit-equal baselines in both; every multiplier-0
         condition (E_L<l>_a0, Cp/Cs_L<l>_*_a0) equal to the original model; baseline numbers; deltas of the other
         conditions; the DAS training record and the cosines of the layer. Exit 1 when a check fails.
  tables --res RES --model M [--old ROOT] [--pairs-file F]
         per-layer tables on the tuning pairs (192), the remaining test pairs (529) and the Dr's 100 pairs: E (DAS and mean
         difference, alpha 2 and 3): change of the self-reported Yes rate, J, AUROC (p_true); C (Cp and Cs, toward abstention /
         answering, alpha 1 and 2): change of the Yes rate, J, AUROC, B; cos(Cp, E) and cos(Cs, E) per layer before the
         orthogonalisation (from the C sweep logs) and cos(DAS, mean difference); then the three-method table and the 2x2
         report as written by dr_tf_table.py / report_2x2.py; then the October side line (old method) from ROOT/results_tqa.
         Every recomputed test-set metric is checked against the sweep json (exit 1 on a mismatch).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.environ.get("PYTHONPATH", "").split(":")[0] or os.getcwd()
sys.path.insert(0, os.path.join(HERE, "scripts"))
from core.metrics import auroc  # noqa: E402
from e22_two_stage import j_stat  # noqa: E402

COS = re.compile(r"^L(\d+)\s+gap_Cprompt\s+(\S+)\s+cos\(Cp,E\)\s+(\S+)\s+\|\s+gap_Cself\s+(\S+)\s+cos\(Cs,E\)\s+(\S+)")


def metrics(ans, p, mg, lab):
    pred = p > 0.5
    return {"yes": float(ans.mean()), "J": j_stat(pred, lab), "AUROC": float(auroc(p, lab)),
            "AUROC_margin": float(auroc(mg, lab)), "B": float(pred.mean())}


def load(res, model, tag):
    stem = os.path.join(res, "E23", model, "knob_sweep_%s" % tag)
    if not os.path.exists(stem + ".npz"):
        return None, None
    return json.load(open(stem + ".json")), np.load(stem + ".npz", allow_pickle=True)


def arrays(z, name):
    return (z[name + "__answer"].astype(bool), z[name + "__ptrue"].astype(float), z[name + "__margin"].astype(float))


def cond_names(z):
    return [k[: -len("__answer")] for k in z.files if k.endswith("__answer")]


def cosines(logdir, pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(logdir, pattern))):
        for line in open(f, encoding="utf-8", errors="replace"):
            m = COS.match(line.strip())
            if m:
                out[int(m.group(1))] = (float(m.group(3)), float(m.group(5)))
    return out


def smoke(args):
    ok = True
    jd, zd = load(args.res, args.model, "smoke_das")
    jm, zm = load(args.res, args.model, "smoke_md")
    if zd is None or zm is None:
        print("SMOKE CHECK FAILED: missing smoke sweep files"); return 1
    for k in ("ids", "labels", "half_a"):
        if not np.array_equal(zd[k], zm[k]):
            print("SMOKE CHECK FAILED: %s differ between the DAS and mean-diff sweeps" % k); ok = False
    for k in ("answer", "ptrue", "margin"):
        same = np.array_equal(zd["baseline__" + k], zm["baseline__" + k])
        print("baseline %s bit-equal in the two sweeps: %s" % (k, same)); ok &= same
    lab = zd["labels"].astype(bool)
    pairs = sorted(set(str(i).rsplit(":", 1)[0] for i in zd["ids"]))
    print("items: %d statements, %d pairs (%d true), tuning statements %d" % (len(lab), len(pairs), int(lab.sum()), int(zd["half_a"].sum())))
    base = arrays(zd, "baseline")
    b = metrics(*base, lab)
    print("original model on these %d statements: AUROC %.3f  AUROC_margin %.3f  J %+.3f  self-reported Yes %.3f  B %.3f" % (
        len(lab), b["AUROC"], b["AUROC_margin"], b["J"], b["yes"], b["B"]))
    for tag, z in (("DAS", zd), ("meandiff", zm)):
        for n in cond_names(z):
            if n == "baseline":
                continue
            a = arrays(z, n)
            if n.endswith("_a0"):
                bit = all(np.array_equal(x, y) for x, y in zip(a, base))
                dmax = float(np.max(np.abs(a[1] - base[1])))
                flips = int((a[0] != base[0]).sum())
                print("[%s] %-22s multiplier 0: bit-equal to the original %s; max |d p_true| %.2e; Yes flips %d" % (tag, n, bit, dmax, flips))
                if flips or dmax > 1e-6:
                    ok = False
            else:
                m = metrics(*a, lab)
                print("[%s] %-22s d Yes %+.3f  d J %+.3f  d AUROC %+.3f  d B %+.3f" % (
                    tag, n, m["yes"] - b["yes"], m["J"] - b["J"], m["AUROC"] - b["AUROC"], m["B"] - b["B"]))
    meta = ((jd.get("das") or {}).get("per_layer") or {}).get(str(args.layer))
    if meta:
        print("DAS layer %d: pairs %s, final loss %s, cos(DAS, mean difference) %+.3f" % (
            args.layer, meta.get("n_pairs"), ["%.4f" % x for x in meta.get("final_losses", [])], meta.get("cos_with_align_to", float("nan"))))
    else:
        print("SMOKE CHECK FAILED: no DAS record for layer %d" % args.layer); ok = False
    for l, (cp, cs) in sorted(cosines(os.path.join(args.res, "logs"), "e23_das_smoke.log").items()):
        print("layer %d: cos(Cp, E) %+.3f, cos(Cs, E) %+.3f before orthogonalisation (C is orthogonalised against E in the sweep)" % (l, cp, cs))
    print("SMOKE CHECK %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def fmt(d, keys):
    return " | ".join("%+.3f" % d[k] for k in keys)


def tables(args):
    jd, zd = load(args.res, args.model, "das_tfce_full")
    jm, zm = load(args.res, args.model, "meandiff_full")
    jc, zc = load(args.res, args.model, "das_tfce_C")
    if zd is None or zm is None:
        print("no E sweeps yet under %s" % args.res); return 1
    for z in (zm,) + ((zc,) if zc is not None else ()):
        for k in ("ids", "labels", "half_a"):
            if not np.array_equal(zd[k], z[k]):
                print("TABLES REFUSED: %s differ between sweeps" % k); return 1
    lab = zd["labels"].astype(bool)
    tune = zd["half_a"].astype(bool)
    pair = np.array([str(i).rsplit(":", 1)[0] for i in zd["ids"]])
    dr = set(json.load(open(args.pairs_file))["pair_ids"])
    sets = [("tuning pairs (%d)" % (int(tune.sum()) // 2), tune), ("test pairs (%d)" % (int((~tune).sum()) // 2), ~tune),
            ("Dr 100 pairs", np.array([p in dr for p in pair]) & ~tune)]
    if int(sets[2][1].sum()) != 200:
        print("TABLES REFUSED: the Dr's 100 pairs give %d test statements" % int(sets[2][1].sum())); return 1
    bad = 0
    for j, z in ((jd, zd), (jm, zm)) + (((jc, zc),) if zc is not None else ()):
        for n in cond_names(z):
            m = metrics(*[x[~tune] for x in arrays(z, n)], lab[~tune])
            ref = j["conds"][n]["B"]
            for k_m, k_r in (("yes", "ans"), ("J", "J"), ("AUROC", "AUROC"), ("B", "B")):
                if abs(m[k_m] - ref[k_r]) > 1e-9:
                    bad += 1
    if bad:
        print("TABLES REFUSED: %d recomputed test metrics differ from the sweep json" % bad); return 1
    print("check: every recomputed test-set metric equals the sweep json")
    n_layers = int(jd.get("num_layers"))
    cosC = cosines(os.path.join(args.res, "logs"), "e23_C_c*.log")
    perl = (jd.get("das") or {}).get("per_layer") or {}
    print("\n# %s: original model" % args.model)
    print("| set | AUROC | AUROC_margin | J | self-reported Yes | B |\n|---|---|---|---|---|---|")
    base = {}
    for sname, mask in sets:
        base[sname] = metrics(*[x[mask] for x in arrays(zd, "baseline")], lab[mask])
        b = base[sname]
        print("| %s | %.3f | %.3f | %+.3f | %.3f | %.3f |" % (sname, b["AUROC"], b["AUROC_margin"], b["J"], b["yes"], b["B"]))
    K = ("yes", "J", "AUROC")
    for sname, mask in sets:
        print("\n## E row per layer, %s: d self-reported Yes | d J | d AUROC (vs original)" % sname)
        hd = ["%s %s" % (arm, k) for arm in ("DAS a2", "DAS a3", "md a2", "md a3") for k in ("dYes", "dJ", "dAUROC")]
        print("| layer | %s | cos(DAS, md) |\n|%s" % (" | ".join(hd), "---|" * (len(hd) + 2)))
        for l in range(n_layers):
            cells = []
            for z, a in ((zd, 2), (zd, 3), (zm, 2), (zm, 3)):
                n = "E_L%d_a%g" % (l, a)
                if n + "__answer" in z.files:
                    m = metrics(*[x[mask] for x in arrays(z, n)], lab[mask])
                    cells.append(fmt({k: m[k] - base[sname][k] for k in K}, K))
                else:
                    cells.append("-")
            c = (perl.get(str(l)) or {}).get("cos_with_align_to")
            print("| %d | %s | %s |" % (l, " | ".join(cells), "%+.3f" % c if c is not None else "-"))
    if zc is None:
        print("\n(no C sweep yet)")
    else:
        KC = ("yes", "J", "AUROC", "B")
        for est in ("Cp", "Cs"):
            for sname, mask in sets:
                print("\n## C row per layer, %s, %s: d self-reported Yes | d J | d AUROC | d B (C orthogonalised against DAS-E)" % (est, sname))
                hd = ["%s %s" % (arm, k) for arm in ("abst a1", "abst a2", "ans a1", "ans a2") for k in ("dYes", "dJ", "dAUROC", "dB")]
                print("| layer | cos(%s, E) before | %s |\n|%s" % (est, " | ".join(hd), "---|" * (len(hd) + 2)))
                for l in range(n_layers):
                    cells = []
                    for d, a in (("abst", 1), ("abst", 2), ("ans", 1), ("ans", 2)):
                        n = "%s_L%d_%s_a%g" % (est, l, d, a)
                        if n + "__answer" in zc.files:
                            m = metrics(*[x[mask] for x in arrays(zc, n)], lab[mask])
                            cells.append(fmt({k: m[k] - base[sname][k] for k in KC}, KC))
                        else:
                            cells.append("-")
                    c = cosC.get(l, (float("nan"), float("nan")))[0 if est == "Cp" else 1]
                    print("| %d | %+.3f | %s |" % (l, c, " | ".join(cells)))
    for f in ("dr_tf_table_%s.md" % args.model, "E23/%s/compare_das_tfce_vs_meandiff_full.md" % args.model,
              "E23/%s/report_2x2_das_tfce.md" % args.model):
        p = os.path.join(args.res, f)
        print("\n# file %s" % f)
        print(open(p, encoding="utf-8").read() if os.path.exists(p) else "(not written yet)")
    old = os.path.join(args.old, "results_tqa")
    print("\n# October side line (old method; E22 = two-stage readout with mean-difference E at fixed layers, n = 1442 statements;"
          " E23 = the old knob sweep, mean-difference / probe E; directions estimated in October, not the DAS of this run)")
    p = os.path.join(old, "E22", args.model, "two_stage_pik4.md")
    if os.path.exists(p):
        lines = open(p, encoding="utf-8").read().splitlines()
        print(lines[0]); print("\n".join(x for x in lines if x.startswith("|")))
    else:
        print("(no E22 file)")
    for f in sorted(glob.glob(os.path.join(old, "E23", args.model, "knob_sweep_*.md"))):
        lines = open(f, encoding="utf-8").read().splitlines()
        cut = next((i for i, x in enumerate(lines) if x.startswith("## All conditions")), len(lines))
        print("\n## %s" % os.path.basename(f)); print("\n".join(lines[:cut]))
    if not glob.glob(os.path.join(old, "E23", args.model, "knob_sweep_*.md")):
        print("\n(no October E23 sweep for %s)" % args.model)
    p = os.path.join(old, "E1", args.model, "summary.json")
    if os.path.exists(p):
        rows = json.load(open(p))["rows"]
        print("\n## October E1 cos(E, C) per layer (mean difference, October prompts): " +
              ", ".join("L%s %+.2f" % (k, rows[k]["cos_EC"]) for k in sorted(rows, key=int)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--res", required=True); s.add_argument("--model", required=True)
    s.add_argument("--layer", type=int, required=True)
    t = sub.add_parser("tables"); t.add_argument("--res", required=True); t.add_argument("--model", required=True)
    t.add_argument("--old", default="/workspace/Truth_Decoupling")
    t.add_argument("--pairs-file", default=os.path.join(HERE, "configs", "splits", "dr_100pairs_20261110.json"))
    a = ap.parse_args()
    sys.exit(smoke(a) if a.cmd == "smoke" else tables(a))


if __name__ == "__main__":
    main()
