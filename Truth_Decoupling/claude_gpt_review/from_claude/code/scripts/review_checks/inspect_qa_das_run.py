#!/usr/bin/env python3
"""Compact read-out of one QA-DAS v1 pilot run directory (bank, training, check artifacts). JSON only, no model."""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])


def load(name):
    p = run / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def short(x, n=160):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "...(%d chars)" % len(s)


bank = load("bank.json")
print("#### BANK", run)
if bank:
    for k in ("model", "num_layers", "hidden_size", "layers", "md_layers", "das_candidate_layers", "structural_null_layers",
              "n_E_pairs", "n_prompt_center_questions", "training_pair_ids", "pilot"):
        print("  %-28s %s" % (k, bank.get(k)))
    print("  rows                         %s" % bank.get("rows"))
    pr = bank.get("precision") or {}
    print("  precision: head=%s dtype=%s proj=%s matmul=%s" % (pr.get("head_module"), pr.get("parameter_dtype"),
                                                               pr.get("projection_dtype"), pr.get("matmul_precision")))
    for item in bank.get("token_audit", [])[:1]:
        for side in ("positive", "negative"):
            s = item[side]
            ta = s["text_audit"]
            print("  token_audit[%s/%s]: answer_end=%s label_predictor=%s label_ids=%s" % (
                item["pair_id"], side, s["answer_end"], s["label_predictor"], s["label_ids"]))
            print("     display_text: %s" % short(ta["display_text"], 600))
            print("     answer=%r suffix=%r labels=%s" % (ta["answer"], ta["suffix"], ta["labels"]))
tr = load("training.json")
print("#### TRAINING")
if tr:
    print("  completed=%s structural_null=%s" % (tr.get("completed"), tr.get("structural_null_layers")))
    for l, row in tr["layers"].items():
        ls = row["losses"]
        ps = row["per_seed"]
        print("  L%s: epochs=%d loss first=%.4f last=%.4f | grad norm min=%.3e max=%.3e mean=%.3e zero=%s updates=%s cache=%s" % (
            l, len(ls), ls[0], ls[-1], ps["gradient_norm_min"], ps["gradient_norm_max"], ps["gradient_norm_mean"],
            ps["zero_gradient_updates"], ps["optimizer_updates"], ps["source_cache_supplied"]))
        for t in row["position_trace"]:
            print("    trace %-20s label=%-5s tok=%s len=%s base_end=%s src_end=%s first_pred=%s cur_pred=%s | h=%.1f intended=%.3f kept=%.3f coord before=%.3f after=%.3f src=%s op=%s dtype=%s" % (
                t.get("direction"), t.get("label"), t.get("label_token_id"), t.get("input_length"), t.get("base_answer_end"),
                t.get("source_answer_end"), t.get("first_label_predictor"), t.get("current_label_predictor"),
                t.get("h_norm", float("nan")), t.get("intended_delta_norm", float("nan")), t.get("kept_delta_norm", float("nan")),
                t.get("coordinate_before", float("nan")), t.get("coordinate_after", float("nan")),
                None if t.get("source_coordinate") is None else round(t["source_coordinate"], 3), t.get("operation"), t.get("residual_dtype")))
ck = load("check_selfcheck.json")
print("#### CHECK_SELFCHECK")
if ck:
    for rep in ck["reports"]:
        res = rep["result"]
        print("  context=%s layer=%s alpha=%s structural_null=%s" % (rep["context"], res["layer"], res["alpha"], res["structural_null"]))
        for pair in res["pairs"]:
            for d in ("negative_to_positive", "positive_to_negative"):
                r = pair[d]
                cells = []
                for cond in ("baseline", "self_swap", "cross_swap", "amplify"):
                    c = r[cond]
                    s = c["logp_sum"]
                    cells.append("%s: pred=%s exp=%s ok=%s (T %.2f F %.2f)" % (cond, c["predicted_label"], c["expected_label"],
                                                                                c["correct"], s["true"], s["false"]))
                wa = r["cross_swap"]["write_audit"]["true"][0] if r["cross_swap"]["write_audit"].get("true") else {}
                print("    %s %-20s end=%s/%s pred_pos=%s | %s | cross kept=%.3f coord %.3f->%.3f src=%.3f" % (
                    pair["pair_id"], d, r["base_answer_end"], r["source_answer_end"], r["label_predictor"], " | ".join(cells),
                    wa.get("kept_delta_norm", float("nan")), wa.get("coordinate_before", float("nan")),
                    wa.get("coordinate_after", float("nan")), wa.get("source_coordinate", float("nan"))))
    pd = ck["precision_diagnostics"]
    for row in pd["same_hidden_projection"]:
        p = row["projection"]
        sel = ["tok%s native=%s fp32=%s diff=%s" % (t["token_id"], [round(v, 4) for v in t["native_raw_scores"]],
                                                     [round(v, 4) for v in t["fp32_raw_scores"]], [round(v, 5) for v in t["abs_diff"]])
               for t in p["selected_tokens"]]
        print("  proj %s/%s native_dtype=%s max_abs_diff=%.5f mean=%.6f hidden_dtype=%s | %s" % (
            row["pair_id"], row["side"], p["native_parameter_dtype"], p["max_abs_diff"], p["mean_abs_diff"], p["hidden"]["dtype"], "; ".join(sel)))
    bp = pd["batch_projection"]
    print("  batch proj: shape=%s max_abs_diff=%.5f | single_vs_batch_hidden_max_abs_diff=%.5f per_row=%s" % (
        bp["score_shape"], bp["max_abs_diff"], pd["single_vs_batch_hidden_max_abs_diff"],
        [round(v, 4) for v in pd["single_vs_batch_hidden_per_row_max_abs_diff"]]))
gen = load("check_generations.json")
print("#### CHECK_GENERATIONS")
if gen:
    print("  completed=%s split=%s pilot=%s conditions=%s" % (gen.get("completed"), gen.get("split"), gen.get("pilot"), gen["conditions"]))
    print("  prompt=%s" % short(gen["prompt"], 400))
    for e in gen["evaluations"]:
        print("  Q %s: %s" % (e["pair_id"], e["question"]))
        print("     gold=%s" % short(e["gold"], 200))
        for c in gen["conditions"]:
            print("     %-34s fin=%s abst=%s -> %r" % (c, e["finished"].get(c), e["rule_abstain"].get(c), e["outputs"].get(c)))
    def flat(v):
        return [x[0] if isinstance(x, list) else x for x in v]

    for a in gen["audit"]:
        disp = a.get("displacement") or []
        if disp:
            k = [d["kept_delta_norm"] for d in disp]
            rel = [d["kept_relative_norm"] for d in disp]
            summ = "steps=%d kept_delta_norm step0=%s last=%s | kept_rel step0=%s max=%.3f | proj before->after step0=%s->%s" % (
                len(disp), [round(v, 2) for v in k[0]], [round(v, 2) for v in k[-1]], [round(v, 4) for v in rel[0]],
                max(max(r) for r in rel),
                [round(v, 2) for v in flat(disp[0]["projection_before"])], [round(v, 2) for v in flat(disp[0]["projection_after"])])
        else:
            summ = "no displacement trace"
        print("  audit %-34s new_tokens=%s edits=%s %s" % (a["condition"], a["new_tokens"], a["edits"], summ))
        if disp and len(disp) > 1 and "continuous" in a["condition"]:
            # per-step implied (h - c_decode)·w = (proj_after - proj_before) / alpha  (alpha = 1 in the pilot)
            for i, d in enumerate(disp[:8]):
                pb, pa = flat(d["projection_before"]), flat(d["projection_after"])
                print("      step %d h_norm=%s proj_before=%s shift=%s kept_rel=%s" % (
                    i, [round(v, 1) for v in d["h_norm"]], [round(v, 1) for v in pb],
                    [round(y - x, 1) for x, y in zip(pb, pa)], [round(v, 3) for v in d["kept_relative_norm"]]))
