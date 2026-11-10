# -*- coding: utf-8 -*-
"""Read-only, CPU: for saved E44 direction records, the size of what each fixed-shift arm writes per unit strength.
learned arm = gap along w; mean-difference arm = md_norm along w_md; random arm = gap along a random direction.
Also the spread, over the training rows, of the per-question coordinate that the per-question interchange writes."""
import sys, os, torch

run = sys.argv[1]
print("# What each arm writes per unit strength, from the saved direction records (read-only)\n")
print("gap = mean over training rows of the prompt difference projected on the learned direction w (the learned arm writes strength x gap along w; the random arm writes the same amount along a random direction).")
print("md_norm = norm of the mean prompt difference (the mean-difference arm writes strength x md_norm along the mean difference, i.e. the mean difference itself).")
print("Because gap = md_norm x cos(w, mean difference), the mean-difference write equals the learned write plus a part orthogonal to w.\n")
print("| family | layer | gap | md_norm | cos(w, mean difference) | md_norm x cos | mean-difference write / learned write | part of the mean-difference write orthogonal to w | per-row coordinate along w: mean, sd, min, max, rows, share positive |")
print("|---|---|---|---|---|---|---|---|---|")
for fam in ("E", "C"):
    for layer in range(32):
        f = os.path.join(run, "dirs", "%s_L%d.pt" % (fam, layer))
        if not os.path.exists(f):
            continue
        r = torch.load(f, map_location="cpu", weights_only=False)
        w = torch.as_tensor(r["w"]).float(); wmd = torch.as_tensor(r["w_md"]).float()
        gap, mdn = float(r["gap"]), float(r["md_norm"])
        c = float(w @ wmd) / float(w.norm() * wmd.norm())
        d = r["difference_along_w"]; v = torch.as_tensor(d["values"]).float()
        print("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.2f | %.3f | %+.3f, %.3f, %+.3f, %+.3f, %d, %.2f |" % (
            fam, layer, gap, mdn, c, mdn * c, mdn / gap if gap else float("nan"), mdn * max(0.0, 1 - c * c) ** 0.5,
            float(v.mean()), float(v.std()), float(v.min()), float(v.max()), v.numel(), float((v > 0).float().mean())))
