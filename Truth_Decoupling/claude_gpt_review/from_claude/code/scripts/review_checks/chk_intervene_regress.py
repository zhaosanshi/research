"""Old-vs-new core/intervene.py regression (review of GPT round 12, E path).

usage: chk_intervene_regress.py OLD_intervene.py NEW_intervene.py OUT.json
Same random inputs through both LayerSteerHook implementations for every calling pattern the existing scripts use
(position last/all, scalar/per-sample alpha, add/amplify/interchange/clamp, max_calls None/1, fp32/bf16, tuple/tensor
layer output, rank 1/2). Reports bitwise value mismatches and audit-counter differences. Then checks the new-only
features against the old semantics: position='mask' (last column / all) and decode_center on call 2. CPU only.
"""
import importlib.util
import json
import sys

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = load("old_intervene", sys.argv[1])
new = load("new_intervene", sys.argv[2])
torch.manual_seed(0)
B, T, d = 4, 7, 16
rep = {"cases": 0, "value_mismatch": [], "count_diffs": [], "errors": [], "new_only": {}}


def run(mod, kw, calls, as_tuple):
    spec = mod.SteerSpec(**kw)
    hook = mod.LayerSteerHook([spec])
    outs = []
    for h in calls:
        out = (h.clone(), torch.zeros(1)) if as_tuple else h.clone()
        res = hook(None, None, out)
        outs.append((res[0] if isinstance(res, tuple) else res).clone())
    return outs, [hook.call_count, hook.applied_count, spec.n_applied, list(hook.first_shape)]


for r in (1, 2):
    W = torch.randn(d, r)
    W = W / W.norm(dim=0, keepdim=True)
    center, coef = torch.randn(d), torch.randn(r)
    src_shared, src_rows = torch.randn(d), torch.randn(B, d)
    zt_shared, zt_rows = torch.randn(r), torch.randn(B * r)
    for mode in ("add", "amplify", "interchange", "clamp"):
        for position in ("last", "all"):
            variants = [("plain", None)]
            if mode == "interchange":
                variants = [("src_shared", src_shared)] + ([("src_rows", src_rows)] if position == "last" else [])
            if mode == "clamp":
                variants = [("zt_shared", zt_shared)] + ([("zt_rows", zt_rows)] if position == "last" else [])
            for vname, vval in variants:
                for aname, alpha in (("scalar", 1.5), ("per_sample", torch.tensor([0.5, 1.0, -1.0, 2.0]))):
                    for max_calls in (None, 1):
                        for dtype in (torch.float32, torch.bfloat16):
                            for as_tuple in (True, False):
                                case = [r, mode, position, vname, aname, max_calls, str(dtype), as_tuple]
                                kw = dict(W=W, layer=3, mode=mode, alpha=alpha, position=position, max_calls=max_calls)
                                if mode == "add":
                                    kw["coef"] = coef
                                if mode == "amplify":
                                    kw["center"] = center
                                if mode == "interchange":
                                    kw["source_act"] = vval
                                if mode == "clamp":
                                    kw["z_target"] = vval
                                calls = [torch.randn(B, T, d).to(dtype), torch.randn(B, 1, d).to(dtype)]
                                rep["cases"] += 1
                                try:
                                    o_out, o_cnt = run(old, dict(kw), calls, as_tuple)
                                except Exception as ex:  # noqa: BLE001
                                    rep["errors"].append({"who": "old", "case": case, "err": repr(ex)[:200]})
                                    continue
                                try:
                                    n_out, n_cnt = run(new, dict(kw), calls, as_tuple)
                                except Exception as ex:  # noqa: BLE001
                                    rep["errors"].append({"who": "new", "case": case, "err": repr(ex)[:200]})
                                    continue
                                for i, (a, b) in enumerate(zip(o_out, n_out)):
                                    if not torch.equal(a, b):
                                        rep["value_mismatch"].append({"case": case, "call": i + 1,
                                                                      "max_abs_diff": float((a.float() - b.float()).abs().max())})
                                if o_cnt != n_cnt:
                                    rep["count_diffs"].append({"case": case, "old[calls,applied,n_applied,first_shape]": o_cnt, "new": n_cnt})

# ---- new-only features vs old semantics ------------------------------------------------------------------------- #
W = torch.randn(d, 1)
W = W / W.norm()
c1, c2 = torch.randn(d), torch.randn(d)
h1, h2 = torch.randn(B, T, d), torch.randn(B, 1, d)
alpha = torch.tensor([0.5, 1.0, -1.0, 2.0])
# mask selecting the last column == old position='last'
mask_last = torch.zeros(B, T, dtype=torch.bool)
mask_last[:, -1] = True
o, _ = run(old, dict(W=W, layer=0, mode="amplify", alpha=alpha, center=c1, position="last"), [h1], False)
n, _ = run(new, dict(W=W, layer=0, mode="amplify", alpha=alpha, center=c1, position="mask", position_mask=mask_last), [h1], False)
rep["new_only"]["mask_last_equals_old_last"] = bool(torch.equal(o[0], n[0]))
# mask selecting everything == old position='all'
o, _ = run(old, dict(W=W, layer=0, mode="amplify", alpha=alpha, center=c1, position="all"), [h1], False)
n, _ = run(new, dict(W=W, layer=0, mode="amplify", alpha=alpha, center=c1, position="mask", position_mask=torch.ones(B, T, dtype=torch.bool)), [h1], False)
rep["new_only"]["mask_all_equals_old_all"] = bool(torch.equal(o[0], n[0]))
# mask with a single selected row and per-sample alpha: row 2 must get alpha[2], not alpha[0]
mask_row2 = torch.zeros(B, T, dtype=torch.bool)
mask_row2[2, 3] = True
n, _ = run(new, dict(W=W, layer=0, mode="amplify", alpha=alpha, center=c1, position="mask", position_mask=mask_row2), [h1], False)
expect = h1.clone()
expect[2, 3] = h1[2, 3] + alpha[2] * (((h1[2, 3] - c1) @ W) @ W.T)
rep["new_only"]["mask_single_row_uses_own_alpha"] = bool(torch.allclose(n[0], expect, atol=1e-6))
rep["new_only"]["mask_single_row_untouched_elsewhere"] = bool(torch.equal(n[0][mask_row2.logical_not()], h1[mask_row2.logical_not()]))
# decode_center: call 1 == old with c1 on the prompt pass; call 2 == old with c2 on the decode step
n, n_cnt = run(new, dict(W=W, layer=0, mode="amplify", alpha=alpha, center=c1, decode_center=c2, position="last"), [h1, h2], False)
o1, _ = run(old, dict(W=W, layer=0, mode="amplify", alpha=alpha, center=c1, position="last"), [h1], False)
o2, _ = run(old, dict(W=W, layer=0, mode="amplify", alpha=alpha, center=c2, position="last"), [h2], False)
rep["new_only"]["decode_center_call1_equals_old_prompt_center"] = bool(torch.equal(n[0], o1[0]))
rep["new_only"]["decode_center_call2_equals_old_decode_center"] = bool(torch.equal(n[1], o2[0]))
# prefill-only (max_calls=1) with decode_center: call 2 must be untouched
n, n_cnt = run(new, dict(W=W, layer=0, mode="amplify", alpha=alpha, center=c1, decode_center=c2, position="last", max_calls=1), [h1, h2], False)
rep["new_only"]["prefill_only_leaves_decode_untouched"] = bool(torch.equal(n[1], h2))
rep["new_only"]["prefill_only_counters[calls,applied,n_applied]"] = n_cnt[:3]
# E25 teacher-forced pattern: prompt-end spec (center c1) + decode-predictor spec (center c2) on one full-sequence pass
P = 4
pe = torch.zeros(B, T, dtype=torch.bool)
pe[:, P - 1] = True
pred = torch.zeros(B, T, dtype=torch.bool)
pred[:, P - 1:T - 1] = True
specs = [new.SteerSpec(W, 0, mode="amplify", alpha=alpha, center=c1, position="mask", position_mask=pe),
         new.SteerSpec(W, 0, mode="amplify", alpha=alpha, center=c2, position="mask", position_mask=pred & ~pe)]
hook = new.LayerSteerHook(specs)
out = hook(None, None, h1.clone())
expect = h1.clone()
for b in range(B):
    expect[b, P - 1] = h1[b, P - 1] + alpha[b] * (((h1[b, P - 1] - c1) @ W) @ W.T)
    for t in range(P, T - 1):
        expect[b, t] = h1[b, t] + alpha[b] * (((h1[b, t] - c2) @ W) @ W.T)
rep["new_only"]["tf_two_spec_pattern_matches_manual"] = bool(torch.allclose(out, expect, atol=1e-6))
rep["new_only"]["tf_two_spec_last_position_untouched"] = bool(torch.equal(out[:, T - 1], h1[:, T - 1]))

print(json.dumps({k: (v if k != "value_mismatch" else len(v)) for k, v in rep.items() if k != "count_diffs"}, indent=1)[:3000])
print("count_diffs:", len(rep["count_diffs"]), "value_mismatch:", len(rep["value_mismatch"]), "errors:", len(rep["errors"]))
if rep["count_diffs"]:
    print("first count diff:", json.dumps(rep["count_diffs"][0]))
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(rep, f, indent=1)
