# -*- coding: utf-8 -*-
"""Read-only (CPU): is the part of each trained direction that is NOT along the mean difference systematic or optimiser noise?
Adjacent layers have nearly the same mean-difference direction. If the rest were independent noise of equal energy,
cos(w_L, w_L+1) could not exceed about half of cos(md_L, md_L+1); if it is systematic, the two cosines are comparable."""
import sys, os, torch
R = sys.argv[1]
def load(f, l):
    p = "%s/dirs/%s_L%d.pt" % (R, f, l)
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None
print("family layers | cos(md_L, md_L+1) | cos(w_L, w_L+1) | ratio | cos(rest_L, rest_L+1)  (rest = w minus its md component, renormalised)")
for f in ("E", "C"):
    for l in range(31):
        a, b = load(f, l), load(f, l + 1)
        if a is None or b is None: continue
        wa, wb, ma, mb = a["w"].double(), b["w"].double(), a["w_md"].double(), b["w_md"].double()
        ra = wa - (wa @ ma) * ma; rb = wb - (wb @ mb) * mb
        cm, cw = float(ma @ mb), float(wa @ wb)
        print("%s L%d-L%d | %.3f | %.3f | %.2f | %.3f" % (f, l, l + 1, cm, cw, cw / cm if cm else float("nan"), float((ra / ra.norm()) @ (rb / rb.norm()))))
# E versus C at the same layer
print()
for l in range(32):
    a, b = load("E", l), load("C", l)
    if a is None or b is None: continue
    print("L%d: cos(w_E, w_C) %+.3f | cos(md_E, md_C) %+.3f" % (l, float(a["w"].double() @ b["w"].double()), float(a["w_md"].double() @ b["w_md"].double())))
