# -*- coding: utf-8 -*-
"""Read-only, CPU: cosines between the E directions of the main run and of the declared side branches, same layer,
and each branch's write sizes. usage: e44_branch_cosines.py RESULTS_ROOT"""
import sys, os, torch

root = sys.argv[1]
runs = {"main": "llama3.1-8b", "M": "llama3.1-8b_side_M", "E2": "llama3.1-8b_side_E2", "MD": "llama3.1-8b_side_initmd"}


def rec(branch, layer, family="E"):
    f = os.path.join(root, runs[branch], "dirs", "%s_L%d.pt" % (family, layer))
    return torch.load(f, map_location="cpu", weights_only=False) if os.path.exists(f) else None


def unit(v):
    v = torch.as_tensor(v).double()
    return v / v.norm()


print("# E directions of the main run and of the side branches, same layer (read-only, CPU)\n")
print("main = truthful sentence E1 against no sentence; M = E1 against the popular-belief sentence M; E2 = the other truthful sentence against no sentence; MD = main, initialised at the mean difference (3 layers).\n")
print("| layer | cos(w main, w M) | cos(w main, w E2) | cos(w M, w E2) | cos(w main, w MD) | cos(w M, w C main) | cos(md main, md M) | gap main | gap M | md_norm M | cos(w, md) M | gap E2 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for layer in range(32):
    a, m, e2, d, c = rec("main", layer), rec("M", layer), rec("E2", layer), rec("MD", layer), rec("main", layer, "C")
    if a is None or m is None:
        continue
    f = lambda x, y: "—" if x is None or y is None else "%+.3f" % float(unit(x) @ unit(y))  # noqa: E731
    print("| %d | %s | %s | %s | %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %s |" % (
        layer, f(a["w"], m["w"]), f(a["w"], e2 and e2["w"]), f(m["w"], e2 and e2["w"]), f(a["w"], d and d["w"]),
        f(m["w"], c and c["w"]), f(a["w_md"], m["w_md"]), float(a["gap"]), float(m["gap"]), float(m["md_norm"]),
        float(m["cosine_w_md"]), "—" if e2 is None else "%.3f" % float(e2["gap"])))
