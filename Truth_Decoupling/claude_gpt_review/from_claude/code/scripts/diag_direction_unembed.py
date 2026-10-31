#!/usr/bin/env python3
"""Direction diagnostic: does a steering direction act through the direct path ("mouth") or downstream ("evidence")?

First-order effect of a direction w written at layer l (CDAS, ICLR 2026, Appendix J, eq. 17):
    dlogits = W_U . Norm(a . w)
i.e. what the unembedding reads off w if nothing downstream reacts. RMSNorm is scale-free, so |a| does not
matter and only the sign does; both +w and -w are reported.

Reading: a direction whose top tokens are the answer words (True / False, Yes / No) works through the direct path,
so it is a "mouth" lever: swapping it flips the answer perfectly (low DAS loss) but amplifying it cannot make the
answer more correct. A direction whose top tokens are uninterpretable fragments works through downstream
components, which is what an "evidence" lever has to do. This is the cheap, model-only check of the explanation
"deep-layer DAS finds the mouth, not the evidence".

Directions analysed per layer: the mean-difference E and C directions (E1 directions_meandiff.npz) and every DAS
direction found in --das-dir (all tags, variants, seeds; seeds averaged after sign alignment).

usage: python3 scripts/diag_direction_unembed.py --model llama3.1-8b --e1-dir results_tqa/E1/llama3.1-8b \
           [--das-dir results_das/E1/llama3.1-8b] [--layers 8 12 16] [--topk 10]
output: <e1-dir>/unembed_diag.{json,md}
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e22_two_stage as e22                                       # noqa: E402
from common import load_split, tf_prompt                          # noqa: E402
from core.config import load_experiment, load_model_config        # noqa: E402
from core.directions import cosine, load_das_direction, unit      # noqa: E402
from core.io_utils import atomic_json_dump                        # noqa: E402
from core.model_io import _resolve_attr, load_model               # noqa: E402
from core.readout import build_label_token_map                    # noqa: E402

FINAL_NORM_CANDIDATES = ("model.norm", "model.language_model.norm", "language_model.model.norm",
                         "model.text_model.norm", "transformer.ln_f")
DAS_FILE = re.compile(r"^das_(E|C)_L(\d+)_r(\d+)_s(\d+)(?:_(.+))?\.pt$")


def final_norm(bundle):
    cands = ([bundle.cfg["final_norm_attr"]] if bundle.cfg.get("final_norm_attr") else []) + list(FINAL_NORM_CANDIDATES)
    for a in cands:
        try:
            return _resolve_attr(bundle.model, a), a
        except AttributeError:
            continue
    raise SystemExit("final norm module not found; set final_norm_attr in the model config (tried %s)" % ", ".join(cands))


def direct_logits(bundle, norm, w):
    """W_U . Norm(w) as float32 numpy [V] (logit softcapping, if any, is monotone and does not change ranks)."""
    p = next(norm.parameters())
    x = torch.as_tensor(np.asarray(w, dtype=np.float32)).to(p.device, p.dtype).view(1, 1, -1)
    lm = bundle.model.get_output_embeddings()
    with torch.no_grad():
        h = norm(x)
        out = lm(h.to(lm.weight.device, lm.weight.dtype))
    return out.view(-1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", default="E2b_tqa", help="experiment config (prompts, eval dataset) used to build the label token maps")
    ap.add_argument("--e1-dir", required=True, help="E1 output dir with directions_meandiff.npz; the report is written there")
    ap.add_argument("--das-dir", default=None, help="directory with das_*_L*_r*_s*.pt files (optional)")
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="layers to report (default: every layer in the npz)")
    ap.add_argument("--wording", default="pik")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--mouth-rank", type=int, default=20, help="a label token within this rank (either sign) marks the direction as a direct-path ('mouth') lever")
    args = ap.parse_args()
    t0 = time.time()

    exp = load_experiment(args.exp)
    mcfg = load_model_config(args.model)
    prompts = exp["prompts"]; st = prompts["statement"]
    e22.SPLIT_A = dict(e22.SPLIT_WORDINGS[args.wording])   # rebind a copy; SPLIT_A is the same object as SPLIT_WORDINGS["know_yn"] (GPT review round 4)
    bundle = load_model(mcfg)
    norm, norm_attr = final_norm(bundle)
    tok = bundle.tokenizer
    print("final norm: %s; unembedding %s" % (norm_attr, tuple(bundle.model.get_output_embeddings().weight.shape)), flush=True)

    # label token ids in the two readout contexts (same construction as e23_knob_sweep)
    rec = load_split(exp["eval_datasets"][0], "calib")[0]
    tB = tf_prompt(bundle, prompts, rec["text"], allow_idk=True)
    labB = build_label_token_map(tok, tB, st["label_variants"])                       # True / False / Abstain
    tA = bundle.build_chat(st["system"], e22.split_user(rec, e22.SPLIT_A["instr"]))
    labA = build_label_token_map(tok, tA, e22.SPLIT_A["variants"])                   # Answer (Yes) / Abstain (No)
    labels = {"True": labB.get("True", []), "False": labB.get("False", []), "Unknown": labB.get("Abstain", []),
              "Yes": labA.get("Answer", []), "No": labA.get("Abstain", [])}
    print("label tokens: %s" % {k: [tok.decode([i]) for i in v] for k, v in labels.items()}, flush=True)
    U = bundle.model.get_output_embeddings().weight.detach().float().cpu().numpy()
    u_tf = U[labels["True"]].mean(0) - U[labels["False"]].mean(0) if labels["True"] and labels["False"] else None
    u_yn = U[labels["Yes"]].mean(0) - U[labels["No"]].mean(0) if labels["Yes"] and labels["No"] else None

    def describe(w):
        out = {}
        for sign, key in ((1.0, "+w"), (-1.0, "-w")):
            lg = direct_logits(bundle, norm, sign * np.asarray(w))
            order = np.argsort(-lg)
            rank = np.empty(len(lg), dtype=np.int64); rank[order] = np.arange(len(lg))
            z = (lg - lg.mean()) / (lg.std() + 1e-9)
            d = {"top": [{"id": int(i), "token": tok.decode([int(i)]), "z": round(float(z[i]), 2)} for i in order[: args.topk]], "labels": {}}
            for name, ids in labels.items():
                if ids:
                    best = min(ids, key=lambda i: rank[i])
                    d["labels"][name] = {"rank": int(rank[best]), "z": round(float(z[best]), 2)}
            out[key] = d
        out["cos_w_uTrue_minus_uFalse"] = cosine(w, u_tf) if u_tf is not None else None
        out["cos_w_uYes_minus_uNo"] = cosine(w, u_yn) if u_yn is not None else None
        tf_ranks = [out[k]["labels"][n]["rank"] for k in ("+w", "-w") for n in ("True", "False") if n in out[k]["labels"]]
        yn_ranks = [out[k]["labels"][n]["rank"] for k in ("+w", "-w") for n in ("Yes", "No") if n in out[k]["labels"]]
        out["min_rank_TF"] = min(tf_ranks) if tf_ranks else None
        out["min_rank_YesNo"] = min(yn_ranks) if yn_ranks else None
        out["mouth_TF"] = bool(tf_ranks) and min(tf_ranks) < args.mouth_rank
        out["mouth_YesNo"] = bool(yn_ranks) and min(yn_ranks) < args.mouth_rank
        return out

    # ---- directions ------------------------------------------------------- #
    z = np.load(os.path.join(args.e1_dir, "directions_meandiff.npz"))
    layers_npz = sorted(int(k[3:]) for k in z.files if k.startswith("E_L"))
    layers = args.layers or layers_npz
    report = {"model": mcfg["name"], "final_norm": norm_attr, "labels": labels, "layers": layers, "directions": {}}
    families = {}                                                # family name -> {layer: direction}
    for l in layers:
        families.setdefault("meandiff_E", {})[l] = unit(z["E_L%d" % l])
        families.setdefault("meandiff_C", {})[l] = unit(z["C_L%d" % l])
    if args.das_dir:
        seen = set()
        for f in glob.glob(os.path.join(args.das_dir, "das_*_L*_r*_s*.pt")):
            m = DAS_FILE.match(os.path.basename(f))
            if not m:
                continue
            tag, l, rank, variant = m.group(1), int(m.group(2)), int(m.group(3)), m.group(5) or ""
            if l not in layers or (tag, l, rank, variant) in seen:
                continue
            seen.add((tag, l, rank, variant))
            ref = unit(z["%s_L%d" % (tag, l)]) if ("%s_L%d" % (tag, l)) in z.files else None
            w, meta = load_das_direction(args.das_dir, tag, l, variant=variant, align_to=ref, rank=rank)
            if w is None:
                continue
            fam = "das_%s%s" % (tag, ("_" + variant) if variant else "")
            families.setdefault(fam, {})[l] = w
            report["directions"].setdefault(fam, {})[str(l)] = {"das_meta": meta}
    for fam, by_layer in families.items():
        for l, w in sorted(by_layer.items()):
            d = report["directions"].setdefault(fam, {}).setdefault(str(l), {})
            d.update(describe(w))
            print("  %-22s L%02d  +w top: %-40s | -w top: %-40s | minrank T/F %s  Yes/No %s  cos(uT-uF) %s" % (
                fam, l, " ".join(repr(t["token"]) for t in d["+w"]["top"][:4]), " ".join(repr(t["token"]) for t in d["-w"]["top"][:4]),
                d["min_rank_TF"], d["min_rank_YesNo"], ("%+.3f" % d["cos_w_uTrue_minus_uFalse"]) if d["cos_w_uTrue_minus_uFalse"] is not None else "n/a"), flush=True)

    # ---- report ---------------------------------------------------------- #
    def esc(s):
        return s.replace("|", "\\|").replace("\n", "\\n")

    lines = ["# Direct-path (unembedding) diagnostic, %s" % mcfg["name"], "",
             "dlogits = W_U . %s(w). A label token within rank %d for either sign marks a direct-path ('mouth') lever." % (norm_attr, args.mouth_rank),
             "Label tokens: %s" % esc(str({k: [tok.decode([i]) for i in v] for k, v in labels.items()})), ""]
    for fam in families:
        lines += ["## %s" % fam, "", "| layer | +w top-%d | -w top-%d | min rank True/False | min rank Yes/No | cos(w, uTrue-uFalse) | cos(w, uYes-uNo) | mouth? |" % (args.topk, args.topk),
                  "|---|---|---|---|---|---|---|---|"]
        mouth_layers = []
        for l in sorted(int(k) for k in report["directions"].get(fam, {})):
            d = report["directions"][fam][str(l)]
            if "+w" not in d:
                continue
            flag = ("TF" if d["mouth_TF"] else "") + ("+YN" if d["mouth_YesNo"] else "")
            if flag:
                mouth_layers.append(l)
            lines.append("| %d | %s | %s | %s | %s | %s | %s | %s |" % (
                l, esc(" ".join("`%s`" % t["token"] for t in d["+w"]["top"])), esc(" ".join("`%s`" % t["token"] for t in d["-w"]["top"])),
                d["min_rank_TF"], d["min_rank_YesNo"],
                ("%+.3f" % d["cos_w_uTrue_minus_uFalse"]) if d["cos_w_uTrue_minus_uFalse"] is not None else "n/a",
                ("%+.3f" % d["cos_w_uYes_minus_uNo"]) if d["cos_w_uYes_minus_uNo"] is not None else "n/a", flag or "-"))
        lines += ["", "mouth layers (%s): %s" % (fam, mouth_layers if mouth_layers else "none"), ""]
        report.setdefault("mouth_layers", {})[fam] = mouth_layers
    report["elapsed_s"] = time.time() - t0
    atomic_json_dump(report, os.path.join(args.e1_dir, "unembed_diag.json"))
    with open(os.path.join(args.e1_dir, "unembed_diag.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwritten %s/unembed_diag.{json,md} (%.0fs)" % (args.e1_dir, report["elapsed_s"]), flush=True)


if __name__ == "__main__":
    main()
