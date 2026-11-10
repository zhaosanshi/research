# -*- coding: utf-8 -*-
"""Read-only (CPU): how much of each trained E44 direction is still its random initialisation.
The initial vector is reproduced exactly as in core/qa_factorial_das.train_direction (seed * 1000 + layer)."""
import sys, glob, os, torch
R = sys.argv[1]
unit = lambda v: v / v.norm()  # noqa: E731
print("file | cos(w, init) | cos(w, md) | cos(init, md) | energy: init part / md part / rest | final train cause loss vs no write")
for p in sorted(glob.glob(R + "/dirs/*.pt"), key=lambda x: (os.path.basename(x)[0], int(os.path.basename(x).split("_L")[1].split(".")[0]))):
    if os.path.basename(p).startswith("t."): continue
    r = torch.load(p, map_location="cpu", weights_only=False)
    w, md = r["w"].double(), r["w_md"].double()
    seed = r["training_contract"]["seed"]; layer = r["layer"]
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + int(layer))
    w0 = unit(torch.randn(w.numel(), generator=g)).double()
    sgn = 1.0
    a, b, c = float(w @ w0), float(w @ md), float(w0 @ md)
    # orthonormalise (init, md) and split the energy of w
    e1 = w0; e2 = unit(md - (md @ w0) * w0)
    p1, p2 = float(w @ e1) ** 2, float(w @ e2) ** 2
    tr = r["losses"]["train"]
    print("%s | %+.3f | %.3f | %+.3f | %.2f / %.2f / %.2f | %.4f -> %.4f" % (os.path.basename(p), a, b, c, p1, p2, 1 - p1 - p2, tr["no_write"]["cause"], tr["trained"]["cause"]))
