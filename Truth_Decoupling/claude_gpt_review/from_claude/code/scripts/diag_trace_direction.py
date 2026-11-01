#!/usr/bin/env python3
"""Trace one existing E direction from the write position to the first output token (GPT review round 5, section 5).

Question: a direction trained by interchange (write the source's coordinate into the base) is used by amplification
(scale the base's own coordinate). Where does that stop working? Three places can break:
  (a) the write itself   -- wrong position, or the float32 edit is lost when it is cast back to the model dtype
                            (bfloat16 keeps 8 significant bits; some residual coordinates are in the tens of thousands);
  (b) generalisation     -- the swap reproduces the source's first token on the training pairs but not on unseen pairs;
  (c) operator mismatch  -- the swap works on unseen pairs, but amplifying the coordinate does not move the true/false
                            judgement in the right direction.

For every requested layer, for the mean-difference direction and the DAS direction of that layer (same files, same
sign alignment and the same centre as e23_knob_sweep.py), the script
  1. TRACE: runs the first --n-train-pairs DAS training pairs and the first --n-tune-pairs tuning pairs (fixed order,
     never chosen by effect) through: identity (zero edit; must reproduce the baseline bit for bit), swap along w with
     the paired statement as source (alpha = 1), amplify along w (alpha in --e-alphas, centre = estimation-set
     midpoint). Per forward it records the written position and its token, |h|, |delta|, |delta| / |h|, how much of
     delta survives the cast to the model dtype, the projection on w before / intended / kept, the first-token readout
     (P(True), P(False), p_true, argmax, top-k), KL(p_source || p_intervened) at the first token (the t = 0 term of the
     DAS loss) and KL(p_base || p_intervened).
  2. STATS: repeats swap and amplify over ALL training pairs and ALL tuning pairs: first-token swap success on
     informative pairs (base and source disagree at baseline), KL closed, p_true moved towards the source; amplify:
     J / AUROC / B over the set, compared with the sweep's own tune-half numbers when --sweep-json is given (end-to-end
     check that this script intervenes exactly like the sweep).
  3. SCALE FACTS: mean |h|, the largest coordinates and the share of |h|^2, of w^2 and of the applied edits on them,
     the bfloat16 spacing at those magnitudes, prompt token lengths against the sliding window, library versions.
Nothing here is a method result; it only locates the break.

usage: python3 scripts/diag_trace_direction.py --model gemma3-12b --layers 20 25 \
           --das-dir /path/results_das_tf/E1/gemma3-12b --das-variant tf \
           --sweep-json /path/results_das_tf/E23/gemma3-12b/knob_sweep_das_tf_full.json \
           --out /path/results_das_tf/diag/gemma3-12b/trace_direction
output: <out>.json, <out>.md
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e22_two_stage as e22                                           # noqa: E402
from common import load_split, tf_prompt                              # noqa: E402
from core.config import load_experiment, load_model_config            # noqa: E402
from core.directions import load_das_direction, mean_diff_direction, teacher_tokens  # noqa: E402
from core.intervene import SteerSpec                                  # noqa: E402
from core.io_utils import set_seed                                    # noqa: E402
from core.metrics import auroc                                        # noqa: E402
from core.model_io import capture_hooks, capture_last_token, forward_logits, load_model  # noqa: E402
from core.readout import build_label_token_map, label_logprobs_from_logits  # noqa: E402


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def lse(x):
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def kl(logp_p, logp_q):
    """KL(p || q) over the full vocabulary from log-probabilities."""
    return float(np.sum(np.exp(logp_p) * (logp_p - logp_q)))


def readout(lg, mB, tok, topk=0):
    """First-token readout from last-position logits: label masses, renormalised p_true, argmax, optional top-k."""
    lg = np.asarray(lg, dtype=np.float64)
    logp = lg - lse(lg)
    lp = label_logprobs_from_logits(lg, mB, use_abstain=False)
    am = int(np.argmax(lg))
    ro = {"P_True": math.exp(lp["True"]), "P_False": math.exp(lp["False"]),
          "p_true": float(e22.tf_scores(lg[None], mB)[0]), "argmax_id": am, "argmax": tok.decode([am])}
    if topk:
        idx = np.argpartition(-lg, topk)[:topk]
        idx = idx[np.argsort(-lg[idx])]
        ro["top"] = [[tok.decode([int(i)]), round(math.exp(logp[i]), 4)] for i in idx]
    return ro, logp


def bf16_spacing(x):
    """Distance between neighbouring bfloat16 values at magnitude x (7 explicit significand bits)."""
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 0.0


def j_b(p, lab):
    pred = p > 0.5
    return {"J": float(pred[lab].mean() - pred[~lab].mean()), "AUROC": float(auroc(p, lab)), "B": float(pred.mean())}


def clean(o):
    """JSON-safe copy: numpy scalars/arrays to Python, NaN/inf to null, dict keys to str."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


# --------------------------------------------------------------------------- #
# the instrumented hook: same arithmetic and the same cast as core.intervene.LayerSteerHook
# --------------------------------------------------------------------------- #

class TraceHook:
    """Applies one SteerSpec at the last position exactly like LayerSteerHook (float32 edit, cast back to the model
    dtype) and records what happened to the vector. spec=None applies a zero edit (identity control)."""

    def __init__(self, spec, w_ref, massive_idx):
        self.spec, self.w_ref, self.massive_idx = spec, w_ref, massive_idx
        self.calls, self.rec = 0, None

    def __call__(self, module, inp, out):
        self.calls += 1
        h = out[0] if isinstance(out, tuple) else out
        pos = h.shape[1] - 1                                       # LayerSteerHook writes h[:, -1, :]
        h_pos = h[:, pos, :].float()
        delta = self.spec.delta(h_pos) if self.spec is not None else torch.zeros_like(h_pos)
        h_new = h.clone()
        h_new[:, pos, :] = (h_pos + delta).to(h.dtype)
        kept = h_new[:, pos, :].float() - h_pos                    # the part of the edit the model dtype keeps
        w = self.w_ref.to(h_pos.device)
        mi = self.massive_idx.to(h_pos.device)

        def n(x):
            return float(torch.linalg.vector_norm(x))
        h_n, d_n, k_n = n(h_pos), n(delta), n(kept)
        self.rec = {
            "pos": int(pos), "seq_len": int(h.shape[1]), "h_norm": h_n, "delta_norm": d_n, "delta_over_h": d_n / max(h_n, 1e-12),
            "kept_norm": k_n, "kept_over_delta": (k_n / d_n) if d_n > 0 else None,
            "cos_kept_delta": float((kept * delta).sum() / (k_n * d_n)) if (d_n > 0 and k_n > 0) else None,
            "proj_before": float(h_pos @ w), "proj_intended": float((h_pos + delta) @ w), "proj_kept": float(h_new[:, pos, :].float() @ w),
            "n_changed": int((h_new[:, pos, :] != h[:, pos, :]).sum()), "n_delta_nonzero": int((delta != 0).sum()),
            "delta_share_massive": float((delta[:, mi] ** 2).sum() / (delta ** 2).sum()) if d_n > 0 else None,
            "kept_share_massive": float((kept[:, mi] ** 2).sum() / (kept ** 2).sum()) if k_n > 0 else None,
            "h_share_massive": float((h_pos[:, mi] ** 2).sum() / (h_pos ** 2).sum()),
        }
        return (h_new,) + tuple(out[1:]) if isinstance(out, tuple) else h_new


def forward_traced(bundle, inputs, layer, spec, w_ref, massive_idx):
    hk = TraceHook(spec, w_ref, massive_idx)
    handle = bundle.layer(layer).register_forward_hook(hk)
    try:
        lg = forward_logits(bundle, inputs, grad=False)[:, -1, :].float().cpu().numpy()[0]
    finally:
        handle.remove()
    if hk.calls != 1:
        raise RuntimeError("layer hook ran %d times during one prompt forward" % hk.calls)
    return lg, hk.rec


def forward_capture(bundle, inputs, layer):
    """Un-intervened forward: last-position logits and the layer's residual at the last (write) position, [1, d] float32."""
    with capture_hooks(bundle, [layer], ("resid",), position="all") as store:
        lg = forward_logits(bundle, inputs, grad=False)[:, -1, :].float().cpu().numpy()[0]
    h = store[("resid", layer)][:, -1, :].detach().float()
    return lg, h


# --------------------------------------------------------------------------- #
# 1. trace on fixed pairs
# --------------------------------------------------------------------------- #

def trace_pair(bundle, d, layer, fams, cB, mB, tok, massive_idx, alphas, prompts, topk):
    enc = {lab: bundle.encode(tf_prompt(bundle, prompts, d[lab]["text"], allow_idk=False)) for lab in (True, False)}
    base = {}
    for lab in (True, False):
        lg, h = forward_capture(bundle, enc[lab], layer)
        ro, logp = readout(lg, mB, tok, topk)
        teach = teacher_tokens(bundle, enc[lab], 8)
        base[lab] = {"lg": lg, "h": h, "ro": ro, "logp": logp, "teacher": repr(tok.decode(teach, skip_special_tokens=False))}
    rec = {"pair_id": d[True].get("pair_id") or d[True]["id"],
           "baseline_argmax_differs": base[True]["ro"]["argmax_id"] != base[False]["ro"]["argmax_id"],
           "baseline_p_true_gap": base[True]["ro"]["p_true"] - base[False]["ro"]["p_true"]}
    rec["informative_for_swap"] = bool(rec["baseline_argmax_differs"] or abs(rec["baseline_p_true_gap"]) >= 0.2)
    rec["statements"] = {}
    w_das = torch.as_tensor(fams["das"], dtype=torch.float32)
    for lab in (True, False):
        other = not lab
        rows = []

        def add(cond, lg, hr, extra=None):
            ro, logp = readout(lg, mB, tok, topk)
            row = {"cond": cond}
            row.update(extra or {})
            row.update(hr)
            row.update(ro)
            row["kl_src_int"] = kl(base[other]["logp"], logp)
            row["kl_base_int"] = kl(base[lab]["logp"], logp)
            rows.append(row)
        lg, hr = forward_traced(bundle, enc[lab], layer, None, w_das, massive_idx)
        add("identity", lg, hr, {"bit_identical_to_baseline": bool(np.array_equal(lg, base[lab]["lg"]))})
        for name, w in fams.items():
            wt = torch.as_tensor(w, dtype=torch.float32)
            src_h = base[other]["h"].cpu()[0]
            spec = SteerSpec(w, layer, mode="interchange", alpha=1.0, source_act=src_h)
            lg, hr = forward_traced(bundle, enc[lab], layer, spec, wt, massive_idx)
            add("swap_%s" % name, lg, hr, {"source_proj": float(src_h.double().numpy() @ w)})
            for a in alphas:
                spec = SteerSpec(w, layer, mode="amplify", alpha=a, center=cB)
                lg, hr = forward_traced(bundle, enc[lab], layer, spec, wt, massive_idx)
                add("amp_%s_a%g" % (name, a), lg, hr, {"centre_proj": float(cB @ w)})
        rec["statements"][str(lab)] = {
            "label": lab, "id": d[lab]["id"], "text": d[lab]["text"], "n_tokens": int(enc[lab]["input_ids"].shape[1]),
            "pos_token": tok.decode([int(enc[lab]["input_ids"][0, -1])]), "baseline": base[lab]["ro"],
            "kl_src_base": kl(base[other]["logp"], base[lab]["logp"]), "teacher_tokens": base[lab]["teacher"], "rows": rows}
    return rec


# --------------------------------------------------------------------------- #
# 2. stats over all pairs of a set
# --------------------------------------------------------------------------- #

def swap_amp_stats(bundle, pairs, layer, fams, cB, mB, tok, massive_idx, alphas, prompts):
    labels, p_base, informative, kl_src_base = [], [], [], []
    p_swap = {f: [] for f in fams}
    ok_swap = {f: [] for f in fams}
    kl_src_swap = {f: [] for f in fams}
    kept_swap = {f: [] for f in fams}
    doh_swap = {f: [] for f in fams}
    p_amp = {(f, a): [] for f in fams for a in alphas}
    kept_amp = {(f, a): [] for f in fams for a in alphas}
    doh_amp = {(f, a): [] for f in fams for a in alphas}
    ids = []
    for d in pairs:
        enc = {lab: bundle.encode(tf_prompt(bundle, prompts, d[lab]["text"], allow_idk=False)) for lab in (True, False)}
        b = {}
        for lab in (True, False):
            lg, h = forward_capture(bundle, enc[lab], layer)
            ro, logp = readout(lg, mB, tok, 0)
            b[lab] = (h, ro, logp)
        inf = b[True][1]["argmax_id"] != b[False][1]["argmax_id"]
        for lab in (True, False):
            other = not lab
            ids.append(d[lab]["id"])
            labels.append(lab)
            p_base.append(b[lab][1]["p_true"])
            informative.append(inf)
            kl_src_base.append(kl(b[other][2], b[lab][2]))
            for name, w in fams.items():
                wt = torch.as_tensor(w, dtype=torch.float32)
                spec = SteerSpec(w, layer, mode="interchange", alpha=1.0, source_act=b[other][0].cpu()[0])
                lg, hr = forward_traced(bundle, enc[lab], layer, spec, wt, massive_idx)
                ro, logp = readout(lg, mB, tok, 0)
                p_swap[name].append(ro["p_true"])
                ok_swap[name].append(ro["argmax_id"] == b[other][1]["argmax_id"])
                kl_src_swap[name].append(kl(b[other][2], logp))
                kept_swap[name].append(hr["kept_over_delta"] if hr["kept_over_delta"] is not None else float("nan"))
                doh_swap[name].append(hr["delta_over_h"])
                for a in alphas:
                    spec = SteerSpec(w, layer, mode="amplify", alpha=a, center=cB)
                    lg, hr = forward_traced(bundle, enc[lab], layer, spec, wt, massive_idx)
                    ro, _ = readout(lg, mB, tok, 0)
                    p_amp[(name, a)].append(ro["p_true"])
                    kept_amp[(name, a)].append(hr["kept_over_delta"] if hr["kept_over_delta"] is not None else float("nan"))
                    doh_amp[(name, a)].append(hr["delta_over_h"])
    lab = np.array(labels, dtype=bool)
    pb = np.array(p_base)
    inf = np.array(informative, dtype=bool)
    out = {"n_pairs": len(pairs), "n_statements": int(len(lab)), "n_informative_statements": int(inf.sum()),
           "baseline": j_b(pb, lab), "baseline_TPR": float((pb[lab] > 0.5).mean()), "baseline_FPR": float((pb[~lab] > 0.5).mean()),
           "mean_kl_src_base": float(np.mean(kl_src_base)), "swap": {}, "amplify": {}, "ids": ids}
    for name in fams:
        ps = np.array(p_swap[name])
        toward = np.where(lab, pb - ps, ps - pb)          # true<-false source: p_true should fall; false<-true: rise
        ok = np.array(ok_swap[name], dtype=bool)
        out["swap"][name] = {
            "first_token_success_informative": float(ok[inf].mean()) if inf.any() else None,
            "n_success_informative": int(ok[inf].sum()), "first_token_match_all": float(ok.mean()),
            "mean_kl_src_swapped": float(np.mean(kl_src_swap[name])),
            "kl_closed_fraction": float(1.0 - np.mean(kl_src_swap[name]) / max(np.mean(kl_src_base), 1e-12)),
            "mean_p_true_moved_towards_source": float(toward.mean()),
            "p_true_moved_towards_source_informative": float(toward[inf].mean()) if inf.any() else None,
            "after_swap": j_b(ps, lab),
            "kept_over_delta_mean": float(np.nanmean(kept_swap[name])), "kept_over_delta_min": float(np.nanmin(kept_swap[name])),
            "delta_over_h_mean": float(np.mean(doh_swap[name]))}
    for (name, a), ps_l in p_amp.items():
        ps = np.array(ps_l)
        after = j_b(ps, lab)
        base_ok, after_ok = (pb > 0.5) == lab, (ps > 0.5) == lab
        out["amplify"]["%s_a%g" % (name, a)] = {
            "after": after, "delta": {k: after[k] - out["baseline"][k] for k in after},
            "mean_dp_true_on_true": float((ps - pb)[lab].mean()), "mean_dp_true_on_false": float((ps - pb)[~lab].mean()),
            "wrong_to_right": int((~base_ok & after_ok).sum()), "right_to_wrong": int((base_ok & ~after_ok).sum()),
            "kept_over_delta_mean": float(np.nanmean(kept_amp[(name, a)])), "kept_over_delta_min": float(np.nanmin(kept_amp[(name, a)])),
            "delta_over_h_mean": float(np.mean(doh_amp[(name, a)])), "delta_over_h_max": float(np.max(doh_amp[(name, a)]))}
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def f3(x, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return ("%+.3f" if signed else "%.3f") % x


def fmt_sci(x):
    return "n/a" if x is None else ("%.3g" % x)


def trace_table(rec, tok_topk):
    lines = []
    for lab_key in ("False", "True"):
        s = rec["statements"][lab_key]
        b = s["baseline"]
        lines += ["", "**%s statement** `%s` (%d tokens; write position token %r): %s" % (
                      "True" if s["label"] else "False", s["id"], s["n_tokens"], s["pos_token"], s["text"].replace("|", "\\|")),
                  "", "baseline: P(True) %.3f, P(False) %.3f, p_true %.3f, argmax %r, top-%d %s; greedy continuation %s; KL(p_source \\|\\| p_base) %.3f" % (
                      b["P_True"], b["P_False"], b["p_true"], b["argmax"], tok_topk, b.get("top"), s["teacher_tokens"], s["kl_src_base"]), "",
                  "| condition | pos | \\|h\\| | \\|delta\\| | delta/\\|h\\| | kept/delta | cos(kept, delta) | proj before -> intended -> kept | changed dims | P(True) | P(False) | p_true | argmax | KL(src\\|\\|int) | KL(base\\|\\|int) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in s["rows"]:
            extra = ""
            if "source_proj" in r:
                extra = " (source %.2f)" % r["source_proj"]
            if "centre_proj" in r:
                extra = " (centre %.2f)" % r["centre_proj"]
            lines.append("| %s%s | %d/%d | %.1f | %s | %s | %s | %s | %.2f -> %.2f -> %.2f%s | %d/%d | %.3f | %.3f | %.3f | %r | %.3f | %.3f |" % (
                r["cond"], (" (bit-identical: %s)" % r["bit_identical_to_baseline"]) if "bit_identical_to_baseline" in r else "",
                r["pos"], r["seq_len"], r["h_norm"], fmt_sci(r["delta_norm"]), fmt_sci(r["delta_over_h"]), f3(r["kept_over_delta"]), f3(r["cos_kept_delta"]),
                r["proj_before"], r["proj_intended"], r["proj_kept"], extra, r["n_changed"], r["n_delta_nonzero"],
                r["P_True"], r["P_False"], r["p_true"], r["argmax"], r["kl_src_int"], r["kl_base_int"]))
    return lines


def stats_table(st, fams, alphas, sweep_cmp):
    b = st["baseline"]
    lines = ["%d pairs, %d statements, %d statements in informative pairs (baseline first tokens differ). Baseline J %.3f, AUROC %.3f, B %.3f (TPR %.3f, FPR %.3f); mean KL(p_source \\|\\| p_base) %.3f." % (
                 st["n_pairs"], st["n_statements"], st["n_informative_statements"], b["J"], b["AUROC"], b["B"], st["baseline_TPR"], st["baseline_FPR"], st["mean_kl_src_base"]), "",
             "| swap along | first-token success on informative | match, all | mean KL(src\\|\\|swapped) | KL closed | p_true moved towards source (all / informative) | J after | AUROC after | B after | kept/delta mean (min) | delta/\\|h\\| mean |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in fams:
        s = st["swap"][name]
        lines.append("| %s | %s (%d/%d) | %.3f | %.3f | %s | %s / %s | %s | %s | %s | %.3f (%.3f) | %s |" % (
            name, f3(s["first_token_success_informative"]), s["n_success_informative"], st["n_informative_statements"], s["first_token_match_all"],
            s["mean_kl_src_swapped"], f3(s["kl_closed_fraction"], True), f3(s["mean_p_true_moved_towards_source"], True), f3(s["p_true_moved_towards_source_informative"], True),
            f3(s["after_swap"]["J"], True), f3(s["after_swap"]["AUROC"]), f3(s["after_swap"]["B"]), s["kept_over_delta_mean"], s["kept_over_delta_min"], fmt_sci(s["delta_over_h_mean"])))
    lines += ["", "| amplify | dJ | dAUROC | dB | mean dp_true on true | on false | wrong->right | right->wrong | kept/delta mean (min) | delta/\\|h\\| mean (max) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name in fams:
        for a in alphas:
            s = st["amplify"]["%s_a%g" % (name, a)]
            lines.append("| %s a%g | %s | %s | %s | %s | %s | %d | %d | %.3f (%.3f) | %s (%s) |" % (
                name, a, f3(s["delta"]["J"], True), f3(s["delta"]["AUROC"], True), f3(s["delta"]["B"], True),
                f3(s["mean_dp_true_on_true"], True), f3(s["mean_dp_true_on_false"], True), s["wrong_to_right"], s["right_to_wrong"],
                s["kept_over_delta_mean"], s["kept_over_delta_min"], fmt_sci(s["delta_over_h_mean"]), fmt_sci(s["delta_over_h_max"])))
    if sweep_cmp:
        lines += ["", "Cross-check with the sweep's own tune-half numbers (same 192 pairs, batch 8 there, batch 1 here):", ""]
        for k, v in sweep_cmp.items():
            lines.append("- %s: sweep J %s / AUROC %s / B %s; here J %s / AUROC %s / B %s" % (
                k, f3(v["sweep"]["J"], True), f3(v["sweep"]["AUROC"]), f3(v["sweep"]["B"]), f3(v["here"]["J"], True), f3(v["here"]["AUROC"]), f3(v["here"]["B"])))
    return lines


def reading(layer_rep, alphas):
    """Heuristic location of the break, from the numbers above; stated as a reading, not a conclusion."""
    out = []
    tr = layer_rep["traces"]
    ident = [r for t in tr for s in t["statements"].values() for r in s["rows"] if r["cond"] == "identity"]
    if ident and not all(r["bit_identical_to_baseline"] for r in ident):
        out.append("identity control is NOT bit-identical to the baseline: the hook itself perturbs the forward -- fix before reading anything else")
    st = layer_rep.get("stats", {})
    for name in ("das", "meandiff"):
        kept = [st[s]["swap"][name]["kept_over_delta_mean"] for s in st] + [st[s]["amplify"]["%s_a%g" % (name, a)]["kept_over_delta_mean"] for s in st for a in alphas]
        if kept and min(kept) < 0.9:
            out.append("%s: on average only %.0f%% of the intended edit survives the cast to the model dtype -> branch (a), the write is partly lost" % (name, 100 * min(kept)))
        if "train" in st and "tune" in st:
            s_tr, s_tu = st["train"]["swap"][name], st["tune"]["swap"][name]
            if s_tr["first_token_success_informative"] is not None and s_tu["first_token_success_informative"] is not None:
                if s_tr["first_token_success_informative"] >= 0.5 and s_tu["first_token_success_informative"] < 0.5:
                    out.append("%s: swap reproduces the source's first token on %.0f%% of informative training statements but only %.0f%% of tuning statements -> branch (b), the direction does not generalise" % (
                        name, 100 * s_tr["first_token_success_informative"], 100 * s_tu["first_token_success_informative"]))
                elif s_tu["first_token_success_informative"] >= 0.5:
                    for a in alphas:
                        d = st["tune"]["amplify"]["%s_a%g" % (name, a)]["delta"]
                        if d["AUROC"] <= 0.0:
                            out.append("%s: swap works on %.0f%% of informative tuning statements, yet amplify a%g changes tune AUROC by %+.3f (J %+.3f, B %+.3f) -> branch (c), interchange training and amplification use differ" % (
                                name, 100 * s_tu["first_token_success_informative"], a, d["AUROC"], d["J"], d["B"]))
                else:
                    out.append("%s: swap fails on most informative statements in BOTH sets (train %.0f%%, tune %.0f%%) -> the trained direction does not carry the first-token swap even where it was fitted" % (
                        name, 100 * s_tr["first_token_success_informative"], 100 * s_tu["first_token_success_informative"]))
    return out or ["no automatic reading triggered; see the tables"]


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_tqa", help="evaluation experiment (prompts, eval dataset, seed); same as the sweep")
    ap.add_argument("--e1-exp", default="E1_directions_tqa", help="E1 experiment whose dataset/split order defined the DAS training pairs")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--das-dir", required=True)
    ap.add_argument("--das-variant", default="tf")
    ap.add_argument("--das-pairs", type=int, default=96, help="number of DAS training pairs the E1 run used (--das-pairs there)")
    ap.add_argument("--tune-pairs", type=int, default=192, help="the sweep's --tune-pairs (same seeded draw)")
    ap.add_argument("--n-train-pairs", type=int, default=1, help="training pairs to trace in detail (first ones, fixed order)")
    ap.add_argument("--n-tune-pairs", type=int, default=2, help="tuning pairs to trace in detail (first ones in test order)")
    ap.add_argument("--e-alphas", nargs="+", type=float, default=[3.0])
    ap.add_argument("--sweep-json", default=None, help="knob_sweep_*.json of the DAS sweep, for the tune-half cross-check")
    ap.add_argument("--n-massive", type=int, default=8, help="how many largest coordinates to report")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--skip-stats", action="store_true", help="only the detailed traces, no all-pair statistics")
    ap.add_argument("--out", required=True, help="output stem; writes <out>.json and <out>.md (refuses to overwrite)")
    args = ap.parse_args()
    for ext in (".json", ".md"):
        if os.path.exists(args.out + ext):
            raise SystemExit("refusing to overwrite %s" % (args.out + ext))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()

    exp = load_experiment(args.exp)
    exp1 = load_experiment(args.e1_exp)
    mcfg = load_model_config(args.model)
    prompts = exp["prompts"]
    st = prompts["statement"]
    if exp1["prompts"]["statement"]["user_tf"] != st["user_tf"] or exp1["prompts"]["statement"]["system"] != st["system"]:
        raise SystemExit("the E1 experiment and the evaluation experiment use different statement prompts; the training pairs would not be reproduced")
    seed = exp.get("seed", 42)
    set_seed(seed)
    bs = exp.get("batch_size", 8)
    bundle = load_model(mcfg)
    tok = bundle.tokenizer
    import transformers
    tc = getattr(bundle.model.config, "text_config", None) or bundle.model.config
    env = {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
           "model_dtype": str(next(bundle.model.parameters()).dtype), "architectures": list(getattr(bundle.model.config, "architectures", []) or []),
           "hidden_size": bundle.hidden_size, "num_layers": bundle.num_layers, "sliding_window": getattr(tc, "sliding_window", None),
           "layer_types": (lambda lt: {t: lt.count(t) for t in sorted(set(lt))} if lt else None)(list(getattr(tc, "layer_types", None) or [])),
           "attn_implementation": getattr(bundle.model.config, "_attn_implementation", None)}
    for l in args.layers:
        if not (0 <= l < bundle.num_layers):
            raise ValueError("layer %d out of range" % l)

    # ---- sets: estimation (calib + dev, as the sweep), test, tune draw, DAS training pairs in E1's order ---- #
    est = []
    for ds in exp["eval_datasets"]:
        est += load_split(ds, "calib") + load_split(ds, "dev")
    y_est = np.array([bool(r["label"]) for r in est])
    eB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
    recs = []
    for ds in exp["eval_datasets"]:
        recs += load_split(ds, exp.get("eval_split", "test"))
    tB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in recs]
    mB = build_label_token_map(tok, tB[0], {"True": st["label_variants"]["True"], "False": st["label_variants"]["False"]})
    pair_ids = [str(r.get("pair_id") or r["id"]) for r in recs]
    uniq = sorted(set(pair_ids))
    tune = set(np.random.default_rng(seed).choice(uniq, args.tune_pairs, replace=False).tolist())
    tp, order = {}, []
    for r, pid in zip(recs, pair_ids):
        if pid in tune:
            if pid not in tp:
                order.append(pid)
            tp.setdefault(pid, {})[bool(r["label"])] = r
    tune_pairs = [tp[p] for p in order if True in tp[p] and False in tp[p]]
    e_recs = []
    for ds in exp1["e_datasets"]:
        for split in exp1.get("direction_splits", ["calib", "dev"]):
            e_recs += load_split(ds, split)
    by_pair = {}
    for r in e_recs:
        if r.get("pair_id"):
            by_pair.setdefault(r["pair_id"], {})[bool(r["label"])] = r
    train_pairs = [d for d in by_pair.values() if True in d and False in d][:args.das_pairs]
    lens_est = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in eB]
    lens_test = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in tB]
    print("estimation %d statements; test %d statements, tune %d pairs (first traced: %s); DAS training pairs %d (first traced: %s)" % (
        len(est), len(recs), len(tune_pairs), [p[True].get("pair_id") for p in tune_pairs[:args.n_tune_pairs]], len(train_pairs),
        [p[True].get("pair_id") for p in train_pairs[:args.n_train_pairs]]), flush=True)
    print("label tokens: %s" % {k: [tok.decode([i]) for i in v] for k, v in mB.items()}, flush=True)
    sweep = json.load(open(args.sweep_json)) if args.sweep_json else None

    # ---- estimation-set activations (as the sweep: left-padded batches) ---- #
    aB, _ = capture_last_token(bundle, eB, args.layers, components=("resid",), batch_size=bs)

    report = {"model": mcfg["name"], "layers": args.layers, "das_dir": args.das_dir, "das_variant": args.das_variant, "e_alphas": args.e_alphas,
              "env": env, "label_tokens": {k: [tok.decode([i]) for i in v] for k, v in mB.items()},
              "prompt_tokens": {"est": {"min": int(min(lens_est)), "mean": float(np.mean(lens_est)), "max": int(max(lens_est))},
                                "test": {"min": int(min(lens_test)), "mean": float(np.mean(lens_test)), "max": int(max(lens_test))},
                                "n_test_over_sliding_window": int(sum(x > (env["sliding_window"] or 10 ** 9) for x in lens_test))},
              "sets": {"n_est": len(est), "n_test": len(recs), "n_tune_pairs": len(tune_pairs), "n_train_pairs": len(train_pairs),
                       "traced_train": [p[True].get("pair_id") for p in train_pairs[:args.n_train_pairs]],
                       "traced_tune": [p[True].get("pair_id") for p in tune_pairs[:args.n_tune_pairs]]},
              "per_layer": {}}
    for l in args.layers:
        HB = aB[("resid", l)].numpy().astype(np.float64)
        wMD, mMD = mean_diff_direction(HB[y_est], HB[~y_est])
        wDAS, meta = load_das_direction(args.das_dir, "E", l, variant=args.das_variant, align_to=wMD)
        if wDAS is None:
            raise FileNotFoundError("no DAS direction for layer %d (variant %r) in %s" % (l, args.das_variant, args.das_dir))
        cB = (HB[y_est].mean(0) + HB[~y_est].mean(0)) / 2
        fams = {"meandiff": wMD, "das": wDAS}
        mean_abs = np.abs(HB).mean(0)
        top = np.argsort(-mean_abs)[:args.n_massive]
        massive_idx = torch.as_tensor(top, dtype=torch.long)
        scale = {"mean_h_norm": float(np.linalg.norm(HB, axis=1).mean()), "median_mean_abs_coord": float(np.median(mean_abs)),
                 "h_share_massive": float((HB[:, top] ** 2).sum(1).mean() / (HB ** 2).sum(1).mean()),
                 "massive": [{"dim": int(i), "mean_abs": float(mean_abs[i]), "bf16_spacing": bf16_spacing(float(mean_abs[i])),
                              "w_meandiff": float(wMD[i]), "w_das": float(wDAS[i])} for i in top]}
        dstat = {}
        for name, w in fams.items():
            z = HB @ w
            dstat[name] = {"readout_auroc_est": float(auroc(z, y_est)), "gap_est": float(z[y_est].mean() - z[~y_est].mean()),
                           "within_sd_est": float(math.sqrt(0.5 * (z[y_est].var() + z[~y_est].var()))),
                           "w_share_massive": float((w[top] ** 2).sum()), "cos_with_meandiff": float(w @ wMD)}
        dstat["das"]["final_losses"] = meta["final_losses"]
        dstat["das"]["files"] = meta["files"]
        print("L%02d  mean|h| %.1f, top coords %s (share of |h|^2 %.3f); w^2 share on them meandiff %.3f / das %.3f; readout AUROC est meandiff %.3f / das %.3f; cos %.3f; DAS loss %s" % (
            l, scale["mean_h_norm"], [m["dim"] for m in scale["massive"][:4]], scale["h_share_massive"], dstat["meandiff"]["w_share_massive"], dstat["das"]["w_share_massive"],
            dstat["meandiff"]["readout_auroc_est"], dstat["das"]["readout_auroc_est"], dstat["das"]["cos_with_meandiff"], meta["final_losses"]), flush=True)
        lrep = {"scale": scale, "directions": dstat, "centre_proj": {n: float(cB @ w) for n, w in fams.items()}, "traces": [], "stats": {}}
        for set_name, pairs in (("train", train_pairs[:args.n_train_pairs]), ("tune", tune_pairs[:args.n_tune_pairs])):
            for d in pairs:
                tr = trace_pair(bundle, d, l, fams, cB, mB, tok, massive_idx, args.e_alphas, prompts, args.topk)
                tr["set"] = set_name
                lrep["traces"].append(tr)
                for lab_key, s in tr["statements"].items():
                    print("  L%02d %s pair %s %s stmt: base p_true %.3f %r | %s" % (
                        l, set_name, tr["pair_id"], "T" if s["label"] else "F", s["baseline"]["p_true"], s["baseline"]["argmax"],
                        "; ".join("%s p_true %.3f %r kept %s" % (r["cond"], r["p_true"], r["argmax"], f3(r["kept_over_delta"])) for r in s["rows"])), flush=True)
        if not args.skip_stats:
            for set_name, pairs in (("train", train_pairs), ("tune", tune_pairs)):
                lrep["stats"][set_name] = swap_amp_stats(bundle, pairs, l, fams, cB, mB, tok, massive_idx, args.e_alphas, prompts)
                s = lrep["stats"][set_name]
                print("  L%02d %s stats: swap success informative das %s / meandiff %s; amplify a%g dAUROC das %+.3f meandiff %+.3f; kept/delta min das %.3f" % (
                    l, set_name, f3(s["swap"]["das"]["first_token_success_informative"]), f3(s["swap"]["meandiff"]["first_token_success_informative"]),
                    args.e_alphas[0], s["amplify"]["das_a%g" % args.e_alphas[0]]["delta"]["AUROC"], s["amplify"]["meandiff_a%g" % args.e_alphas[0]]["delta"]["AUROC"],
                    s["amplify"]["das_a%g" % args.e_alphas[0]]["kept_over_delta_min"]), flush=True)
            if sweep and "tune" in lrep["stats"]:
                cmp = {}
                sb = sweep["conds"]["baseline"]["A"]
                cmp["baseline"] = {"sweep": {k: sb[k] for k in ("J", "AUROC", "B")}, "here": lrep["stats"]["tune"]["baseline"]}
                for a in args.e_alphas:
                    key = "E_L%d_a%g" % (l, a)
                    if key in sweep["conds"]:
                        cmp[key + " (das)"] = {"sweep": {k: sweep["conds"][key]["A"][k] for k in ("J", "AUROC", "B")}, "here": lrep["stats"]["tune"]["amplify"]["das_a%g" % a]["after"]}
                lrep["sweep_cross_check"] = cmp
        lrep["reading"] = reading(lrep, args.e_alphas)
        report["per_layer"][str(l)] = lrep

    # ---- write ---- #
    report["elapsed_s"] = time.time() - t0
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(clean(report), f, indent=1, ensure_ascii=False)
    pt = report["prompt_tokens"]
    lines = ["# Direction trace: write position -> first token, %s" % mcfg["name"], "",
             "Layers %s; DAS files `%s` (variant %s); amplify alphas %s; estimation set %d statements; traced training pairs %s, traced tuning pairs %s (fixed order). "
             "Identity = zero edit through the same hook (must be bit-identical). Swap = write the paired statement's coordinate along w (alpha 1). Amplify = the sweep's operator with the sweep's centre." % (
                 args.layers, args.das_dir, args.das_variant, args.e_alphas, len(est), report["sets"]["traced_train"], report["sets"]["traced_tune"]), "",
             "Environment: torch %s, transformers %s, CUDA %s, model dtype %s, %s, hidden %d, %d layers, sliding window %s, layer types %s, attention %s." % (
                 env["torch"], env["transformers"], env["cuda"], env["model_dtype"], env["architectures"], env["hidden_size"], env["num_layers"], env["sliding_window"], env["layer_types"], env["attn_implementation"]),
             "Prompt lengths (forced True/False prompt): estimation %d-%d tokens (mean %.1f), test %d-%d (mean %.1f); %d test prompts exceed the sliding window." % (
                 pt["est"]["min"], pt["est"]["max"], pt["est"]["mean"], pt["test"]["min"], pt["test"]["max"], pt["test"]["mean"], pt["n_test_over_sliding_window"]),
             "Label tokens: %s" % report["label_tokens"], ""]
    for l in args.layers:
        lr = report["per_layer"][str(l)]
        sc, ds = lr["scale"], lr["directions"]
        lines += ["## Layer %d" % l, "", "### Scale and direction facts (estimation set)", "",
                  "Mean |h| at the write position %.1f; median mean|coordinate| %.3f; the %d largest coordinates carry %.1f%% of |h|^2." % (
                      sc["mean_h_norm"], sc["median_mean_abs_coord"], args.n_massive, 100 * sc["h_share_massive"]), "",
                  "| coordinate | mean abs | bf16 spacing there | w_meandiff | w_das |", "|---|---|---|---|---|"]
        for m in sc["massive"]:
            lines.append("| %d | %.1f | %.3g | %+.4f | %+.4f |" % (m["dim"], m["mean_abs"], m["bf16_spacing"], m["w_meandiff"], m["w_das"]))
        lines += ["", "| direction | cos with meandiff | DAS final loss | readout AUROC of w.h (est) | gap along w (est) | within-class sd | w^2 share on the largest coordinates | centre projection w.c |", "|---|---|---|---|---|---|---|---|"]
        for name in ("meandiff", "das"):
            d = ds[name]
            lines.append("| %s | %.3f | %s | %.3f | %.2f | %.2f | %.3f | %.2f |" % (
                name, d["cos_with_meandiff"], ("%.4f" % d["final_losses"][0]) if name == "das" else "-", d["readout_auroc_est"], d["gap_est"], d["within_sd_est"], d["w_share_massive"], lr["centre_proj"][name]))
        lines += ["", "### Trace on the fixed pairs", ""]
        for tr in lr["traces"]:
            lines += ["#### %s pair `%s` -- baseline first tokens differ: %s; p_true gap %.3f; %s" % (
                          tr["set"], tr["pair_id"], tr["baseline_argmax_differs"], tr["baseline_p_true_gap"],
                          "informative for the swap test" if tr["informative_for_swap"] else "NOT informative: the two statements already give the same first-token distribution, the swap cannot be tested here")]
            lines += trace_table(tr, args.topk)
            lines.append("")
        if lr["stats"]:
            for set_name in ("train", "tune"):
                if set_name in lr["stats"]:
                    lines += ["### Statistics over all %s pairs" % set_name, ""]
                    lines += stats_table(lr["stats"][set_name], ("meandiff", "das"), args.e_alphas, lr.get("sweep_cross_check") if set_name == "tune" else None)
                    lines.append("")
        lines += ["### Reading (heuristic)", ""] + ["- " + r for r in lr["reading"]] + [""]
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwritten %s.{json,md} (%.0fs)" % (args.out, report["elapsed_s"]), flush=True)


if __name__ == "__main__":
    main()
