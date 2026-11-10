#!/usr/bin/env python3
"""Read-only audit of saved E43 direction records (CPU only, no model, no GPU).
For each family x layer: coordinate scale at the read position, side separation (paired), write size at training,
cosine to the mean-difference direction, cosine between E and C, between adjacent layers, and (optionally) cosine
to the same direction trained in another run (stability under tiny numeric / target perturbations).
usage: e43_dir_audit.py RUN_DIR [--other OTHER_RUN_DIR]
"""
import argparse, json, math
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--other", type=Path, default=None)
a = ap.parse_args()


def load(run, fam, L):
    p = run / "dirs" / ("%s_L%d.pt" % (fam, L))
    return torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None


def stats(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))


recs = {(f, L): load(a.run_dir, f, L) for f in ("E", "C") for L in range(32)}
recs = {k: v for k, v in recs.items() if v is not None}
print("loaded %d direction records from %s" % (len(recs), a.run_dir))
other = {}
if a.other:
    other = {k: load(a.other, *k) for k in recs}
    other = {k: v for k, v in other.items() if v is not None}
    print("loaded %d direction records from %s" % (len(other), a.other))

for fam in ("E", "C"):
    print("\n## %s  (z = coordinate of the source prompt's last token along the learned w)" % fam)
    print("| L | n | z plus mean±sd | z minus mean±sd | gap | paired diff mean±sd (d, %right sign) | md: gap, paired d | cos(w,md) | CE ep1 +/- | CE final +/- | write rel train +/- | cos(w, other run) | cos(w_L, w_L-1) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    prev = None
    for L in range(32):
        r = recs.get((fam, L))
        if r is None:
            continue
        zp = torch.tensor(r["endpoints"]["plus"]["values"], dtype=torch.float64)
        zm = torch.tensor(r["endpoints"]["minus"]["values"], dtype=torch.float64)
        d = zp - zm
        sign = 1.0 if d.mean() >= 0 else -1.0
        dm, ds = float(d.mean()), float(d.std(unbiased=False))
        right = float(((d * sign) > 0).double().mean())
        mp = torch.tensor(r["endpoints_md"]["plus"]["values"], dtype=torch.float64)
        mm = torch.tensor(r["endpoints_md"]["minus"]["values"], dtype=torch.float64)
        md = mp - mm
        h = r["training"]["history"]
        pp = r["per_pair_ce_after"]
        cep = sum(x["ce_plus"] for x in pp) / len(pp); cem = sum(x["ce_minus"] for x in pp) / len(pp)
        wr_p = sum(x["write_rel_mean"][0] for x in pp) / len(pp); wr_m = sum(x["write_rel_mean"][1] for x in pp) / len(pp)
        w = r["w"].double()
        co = other.get((fam, L))
        cos_other = "%.2f" % abs(float(torch.dot(w, co["w"].double()))) if co is not None else "—"
        cos_prev = "%.2f" % abs(float(torch.dot(w, prev))) if prev is not None else "—"
        prev = w
        print("| %d | %d | %+.2f±%.2f | %+.2f±%.2f | %+.2f | %+.2f±%.2f (d=%.2f, %.0f%%) | %+.2f, d=%.2f | %+.2f | %.2f / %.2f | %.2f / %.2f | %.2f / %.2f | %s | %s |" % (
            L, len(zp), float(zp.mean()), float(zp.std(unbiased=False)), float(zm.mean()), float(zm.std(unbiased=False)),
            dm, dm, ds, abs(dm) / max(ds, 1e-12), 100 * right,
            float(md.mean()), abs(float(md.mean())) / max(float(md.std(unbiased=False)), 1e-12),
            r["cosine_w_md"], h[0]["ce_plus"], h[0]["ce_minus"], cep, cem, wr_p, wr_m, cos_other, cos_prev))

print("\n## cosine between the E and C learned directions at the same layer (|cos|), and between their mean-difference directions")
print("| L | |cos(w_E, w_C)| | |cos(md_E, md_C)| |")
print("|---|---|---|")
for L in range(32):
    e, c = recs.get(("E", L)), recs.get(("C", L))
    if e is None or c is None:
        continue
    print("| %d | %.2f | %.2f |" % (L, abs(float(torch.dot(e["w"].double(), c["w"].double()))),
                               abs(float(torch.dot(e["w_md"].double(), c["w_md"].double())))))

print("\n## training curves (mean CE per epoch), selected layers")
for fam in ("E", "C"):
    for L in (4, 12, 16, 20, 24, 28, 31):
        r = recs.get((fam, L))
        if r is None:
            continue
        h = r["training"]["history"]
        print("%s L%d plus : %s" % (fam, L, " ".join("%.2f" % x["ce_plus"] for x in h)))
        print("%s L%d minus: %s" % (fam, L, " ".join("%.2f" % x["ce_minus"] for x in h)))
