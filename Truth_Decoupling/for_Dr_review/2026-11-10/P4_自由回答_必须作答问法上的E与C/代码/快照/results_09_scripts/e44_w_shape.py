# -*- coding: utf-8 -*-
"""Read-only (CPU): shape of the trained E44 directions. If |w_i| is nearly constant across coordinates, the Adam + renormalise
loop has driven w to a sign pattern; then cos(w, sign(md)/sqrt(d)) is close to 1 and cos(w, md) ~ mean|md_i| / rms(md_i)."""
import sys, glob, os, torch
R = sys.argv[1]
print("file | cos(w, md) | cos(w, sign(md)) | mean|md|/rms(md) | |w_i|*sqrt(d): p10 p50 p90 max | share of coords with |w_i|*sqrt(d) in [0.8, 1.2] | sign agreement with md")
for p in sorted(glob.glob(R + "/dirs/*.pt")):
    if os.path.basename(p).startswith("t."): continue
    r = torch.load(p, map_location="cpu", weights_only=False)
    w, md = r["w"].float(), r["w_md"].float(); d = w.numel()
    s = torch.sign(md) / d ** 0.5
    a = w.abs() * d ** 0.5
    q = torch.quantile(a, torch.tensor([0.1, 0.5, 0.9]))
    print("%s | %.3f | %.3f | %.3f | %.2f %.2f %.2f %.2f | %.2f | %.3f" % (
        os.path.basename(p), float(w @ md), float(w @ s), float(md.abs().mean() / md.pow(2).mean().sqrt()), q[0], q[1], q[2], a.max(),
        float(((a > 0.8) & (a < 1.2)).float().mean()), float((torch.sign(w) == torch.sign(md)).float().mean())))
