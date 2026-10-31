#!/usr/bin/env python3
"""E22: readout protocols for the 2x2 -- one-shot three-way (True/False/Unknown) vs two-stage ("sure/unsure" or
"know/unknown" first, then True/False with the first word teacher-forced) vs forced two-way.

For every protocol and steering condition we report
  answer_rate  : stage-1 share of items the model is willing to judge (three-way: P(True)+P(False) > P(Unknown))
  J_all, AUROC_all, B_all : discrimination / ranking / bias of the True-vs-False decision on ALL items
                            (two-stage: stage 2 is read for every item with the stage-1 word teacher-forced;
                             three-way: True vs False renormalised; forced2: the plain two-way prompt)
  J_sel, acc_sel, B_sel   : the same restricted to the items the model was willing to judge (selective view)
Conditions: baseline, evidence amplify (E_a2, E_a3 at --e-layers), commitment add toward abstention (C_abst_a*) and
toward answering (C_ans_a*), and E+C.  Directions come from E1 under the results root (E1/<model>).
usage: python3 scripts/e22_two_stage.py --exp E2b_tqa --model llama3.1-8b --e-layers 12 13 14 [--limit N]
output: results/E22/<model>/two_stage.{json,md}
"""
import argparse
import os
import time

import numpy as np

from common import RESULTS_DIR, load_split, tf_prompt
from core.config import load_experiment, load_model_config
from core.intervene import SteerSpec, generate, next_token_logits
from core.io_utils import atomic_json_dump, load_json, set_seed
from core.metrics import auroc
from core.directions import mean_diff_direction
from core.model_io import capture_last_token, load_model
from core.readout import build_label_token_map, label_logprobs_from_logits

# pass A wording chosen on 300 Llama + 200 Qwen3-8B items (2026-10-29): 'know/unknown' -> 8% of Llama answers were 'true'
# (the model blurts the judgement when it is sure the statement is true); 'Yes/No' -> 0% non-compliant on both models.
SPLIT_WORDINGS = {
    "know_yn": {"instr": "Do you know whether this statement is true or false? Answer with exactly one word: Yes or No.",
                "variants": {"Answer": ["Yes", " Yes", "yes", " yes"], "Abstain": ["No", " No", "no", " no"]}, "gen": ("yes", "no")},
    "sure_yn": {"instr": "Are you sure whether this statement is true or false? Answer with exactly one word: Yes or No.",
                "variants": {"Answer": ["Yes", " Yes", "yes", " yes"], "Abstain": ["No", " No", "no", " no"]}, "gen": ("yes", "no")},
    "pik": {"instr": "Do you know the answer to this question? Answer with exactly one word: Yes or No.", "question_level": True,
            "variants": {"Answer": ["Yes", " Yes", "yes", " yes"], "Abstain": ["No", " No", "no", " no"]}, "gen": ("yes", "no")},
    "qsure": {"instr": "Are you sure you can accurately answer this question based on your internal knowledge? Answer with exactly one word: sure or unsure.", "question_level": True,
              "variants": {"Answer": ["sure", " sure", "Sure"], "Abstain": ["unsure", " unsure", "Unsure"]}, "gen": ("sure", "unsure")},
    "rtuning": {"instr": "Are you sure you can accurately judge this statement based on your internal knowledge? Answer with exactly one word: sure or unsure.",
                "variants": {"Answer": ["sure", " sure", "Sure"], "Abstain": ["unsure", " unsure", "Unsure"]}, "gen": ("sure", "unsure")},
}
SPLIT_A = SPLIT_WORDINGS["know_yn"]

STAGE1 = {
    "sure": {"instr": "First, say whether you are sure whether this statement is true or false: answer 'sure' or 'unsure'. "
                      "If you are sure, then say whether the statement is true or false. "
                      "Reply in the form 'sure, True' or 'sure, False' or 'unsure'.",
             "prefix": "sure,", "variants": {"Answer": ["sure", " sure", "Sure"], "Abstain": ["unsure", " unsure", "Unsure"]}},
    "sure_yn": {"instr": "First, say whether you are sure whether this statement is true or false: answer 'Yes' or 'No'. "
                         "If Yes, then say whether the statement is true or false. "
                         "Reply in the form 'Yes, True' or 'Yes, False' or 'No'.",
                "prefix": "Yes,", "variants": {"Answer": ["Yes", " Yes", "yes"], "Abstain": ["No", " No", "no"]}},
    "know": {"instr": "First, say whether you know whether this statement is true or false: answer 'know' or 'unknown'. "
                      "If you know, then say whether the statement is true or false. "
                      "Reply in the form 'know, True' or 'know, False' or 'unknown'.",
             "prefix": "know,", "variants": {"Answer": ["know", " know", "Know"], "Abstain": ["unknown", " unknown", "Unknown"]}},
}


def split_user(r, instr):
    """pass-A user text: question-level (P(IK)) if the wording asks for it and the record has a question, else statement-level."""
    if SPLIT_A.get("question_level") and r.get("meta", {}).get("question"):
        return "Question: %s\n%s" % (r["meta"]["question"], instr)
    return "Statement: %s\n%s" % (r["text"], instr)


def logits_for(bundle, texts, specs, bs):
    out = []
    for i in range(0, len(texts), bs):
        out.append(next_token_logits(bundle, bundle.encode(texts[i:i + bs]), specs))
    return np.concatenate([o.numpy() for o in out])


def tf_scores(logits, tf_map):
    """p_true renormalised over True/False (no abstain), per item."""
    p = []
    for lg in logits:
        lp = label_logprobs_from_logits(lg, tf_map, use_abstain=False)
        a, b = lp["True"], lp["False"]
        m = max(a, b)
        p.append(float(np.exp(a - m) / (np.exp(a - m) + np.exp(b - m))))
    return np.array(p)


def j_stat(pred_true, labels):
    labels = np.asarray(labels, dtype=bool)
    pred_true = np.asarray(pred_true, dtype=bool)
    tpr = pred_true[labels].mean() if labels.any() else float("nan")
    fpr = pred_true[~labels].mean() if (~labels).any() else float("nan")
    return float(tpr - fpr)


def metrics(answer, p_true, labels):
    labels = np.asarray(labels, dtype=bool)
    pred = p_true > 0.5
    m = {"answer_rate": float(np.mean(answer)), "J_all": j_stat(pred, labels), "AUROC_all": float(auroc(p_true, labels)),
         "B_all": float(pred.mean()), "n_answered": int(np.sum(answer))}
    if answer.sum() > 0:
        a = answer.astype(bool)
        m.update({"J_sel": j_stat(pred[a], labels[a]), "acc_sel": float((pred[a] == labels[a]).mean()), "B_sel": float(pred[a].mean()),
                  "AUROC_sel": float(auroc(p_true[a], labels[a])) if labels[a].any() and (~labels[a]).any() else float("nan")})
    else:
        m.update({"J_sel": float("nan"), "acc_sel": float("nan"), "B_sel": float("nan"), "AUROC_sel": float("nan")})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2b_tqa")
    ap.add_argument("--model", default=None)
    ap.add_argument("--e-layers", nargs="+", type=int, default=[12, 13, 14])
    ap.add_argument("--c-layer", type=int, default=None)
    ap.add_argument("--e-alphas", nargs="+", type=float, default=[2.0, 3.0])
    ap.add_argument("--c-alphas", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--protocols", nargs="+", default=["forced2", "split"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--c-from", default="s1", choices=["s1", "s2", "e1"], help="estimate the commitment direction C under pass-A text (s1, default) or under the statement-level forced text (s2)")
    ap.add_argument("--c-orth", action="store_true", help="remove from C its component along the evidence direction at the C layer (estimated under the statement prompt)")
    ap.add_argument("--e-source", default="meandiff", choices=["meandiff", "meandiff_orth", "das"],
                    help="evidence direction: mean-diff; mean-diff with its component along C (same layer) removed; or the rank-1 DAS direction from --das-dir")
    ap.add_argument("--das-dir", default=None, help="directory with das_E_L<l>_r1_s*.pt (e.g. results_das/E1/<model>)")
    ap.add_argument("--dir-datasets", nargs="+", default=None, help="estimate E and C on these datasets' calib+dev instead of the eval datasets (transfer test)")
    ap.add_argument("--split-wording", default="know_yn", choices=sorted(SPLIT_WORDINGS), help="pass-A question wording for the split protocol")
    ap.add_argument("--no-reestimate", action="store_true", help="use the E1 directions for every protocol instead of re-estimating them under each protocol's own prompt")
    args = ap.parse_args()
    chosen = dict(SPLIT_WORDINGS[args.split_wording])   # copy first: SPLIT_A *is* SPLIT_WORDINGS["know_yn"]; clear() would empty both (GPT review round 4)
    SPLIT_A.clear(); SPLIT_A.update(chosen)

    exp = load_experiment(args.exp)
    if args.model:
        exp["model"] = load_model_config(args.model)
    mcfg, prompts = exp["model"], exp["prompts"]
    seed = exp.get("seed", 42)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    bs = exp.get("batch_size", 8)
    e1_dir = os.path.join(RESULTS_DIR, "E1", mcfg["name"])
    e1 = load_json(os.path.join(e1_dir, "summary.json"))
    z = np.load(os.path.join(e1_dir, "directions_meandiff.npz"))
    cL = args.c_layer if args.c_layer is not None else int(e1["C_write_layer"])
    gap_C = float(e1["rows"][str(cL)]["gap_C"])

    bundle = load_model(mcfg)
    recs = []
    for ds in exp["eval_datasets"]:
        recs += load_split(ds, exp.get("eval_split", "test"))
    if args.limit:
        recs = recs[: args.limit]
    labels = np.array([bool(r["label"]) for r in recs])
    pair_ids = [r.get("pair_id") or r["id"] for r in recs]
    st = prompts["statement"]
    tf_variants = {"True": st["label_variants"]["True"], "False": st["label_variants"]["False"]}

    # ---- prompt texts per protocol -------------------------------------- #
    texts = {}
    for proto in args.protocols:
        if proto == "threeway":
            texts[proto] = {"s1": [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in recs]}
        elif proto == "forced2":
            texts[proto] = {"s1": [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in recs]}
        elif proto == "split":
            texts[proto] = {"s1": [bundle.build_chat(st["system"], split_user(r, SPLIT_A["instr"])) for r in recs],
                            "s2": [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in recs]}
        else:
            spec = STAGE1[proto]
            s1 = [bundle.build_chat(st["system"], "Statement: %s\n%s" % (r["text"], spec["instr"])) for r in recs]
            texts[proto] = {"s1": s1, "s2": [t + spec["prefix"] for t in s1]}
    maps = {}
    for proto in args.protocols:
        if proto == "threeway":
            maps[proto] = {"s1": build_label_token_map(bundle.tokenizer, texts[proto]["s1"][0], st["label_variants"])}
        elif proto == "forced2":
            maps[proto] = {"s1": build_label_token_map(bundle.tokenizer, texts[proto]["s1"][0], tf_variants)}
        elif proto == "split":
            maps[proto] = {"s1": build_label_token_map(bundle.tokenizer, texts[proto]["s1"][0], SPLIT_A["variants"]),
                           "s2": build_label_token_map(bundle.tokenizer, texts[proto]["s2"][0], tf_variants)}
        else:
            tf_sp = {"True": [v for v in tf_variants["True"] if v.startswith(" ")], "False": [v for v in tf_variants["False"] if v.startswith(" ")]}
            maps[proto] = {"s1": build_label_token_map(bundle.tokenizer, texts[proto]["s1"][0], STAGE1[proto]["variants"]),
                           "s2": build_label_token_map(bundle.tokenizer, texts[proto]["s2"][0], tf_sp)}
        print("%s label tokens: %s" % (proto, {k: [bundle.tokenizer.decode([i]) for i in v] for m in maps[proto].values() for k, v in m.items()}), flush=True)

    # ---- format check: a few greedy continuations per protocol (baseline) -- #
    samples = {}
    for proto in args.protocols:
        outs, _ = generate(bundle, bundle.encode(texts[proto]["s1"][:4]), None, max_new_tokens=6)
        samples[proto] = outs
        print("%s greedy samples: %s" % (proto, [o.strip()[:30] for o in outs]), flush=True)

    # ---- directions per protocol (estimated on calib+dev under that protocol's own prompts) ------------- #
    est = []
    for ds in (args.dir_datasets or exp["eval_datasets"]):
        est += load_split(ds, "calib") + load_split(ds, "dev")
    y_est = np.array([bool(r["label"]) for r in est])
    cc = prompts["commitment_contrast"]
    dirs = {}
    for proto in args.protocols:
        if args.no_reestimate:
            dirs[proto] = {"E": {l: (z["E_L%d" % l], z["E_center_L%d" % l]) for l in args.e_layers}, "C": (z["C_L%d" % cL], gap_C)}
            dirs[proto]["E_s1"] = dirs[proto]["E"]
            continue
        if proto == "threeway":
            tf_texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=True) for r in est]
            s1_user = lambda r: st["user_tf_idk"].format(statement=r["text"])
        elif proto == "forced2":
            tf_texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
            s1_user = lambda r: st["user_tf"].format(statement=r["text"])
        elif proto == "split":
            tf_texts = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est]
            s1_user = lambda r: split_user(r, SPLIT_A["instr"])
        else:
            spec = STAGE1[proto]
            tf_texts = [bundle.build_chat(st["system"], "Statement: %s\n%s" % (r["text"], spec["instr"])) + spec["prefix"] for r in est]
            s1_user = lambda r, spec=spec: "Statement: %s\n%s" % (r["text"], spec["instr"])
        acts, _ = capture_last_token(bundle, tf_texts, sorted(set(list(args.e_layers) + [cL])), components=("resid",), batch_size=bs)
        H_cL = acts[("resid", cL)].numpy().astype(np.float64)
        wE_cL, _ = mean_diff_direction(H_cL[y_est], H_cL[~y_est])
        Ed = {}
        for l in args.e_layers:
            H = acts[("resid", l)].numpy().astype(np.float64)
            w, _ = mean_diff_direction(H[y_est], H[~y_est])
            Ed[l] = (w, (H[y_est].mean(0) + H[~y_est].mean(0)) / 2)
        Ed_s1 = Ed
        if proto in STAGE1 or proto == "split":   # the know/unknown position has its own centre along the same direction
            s1_texts = [bundle.build_chat(st["system"], s1_user(r)) for r in est]
            acts1, _ = capture_last_token(bundle, s1_texts, args.e_layers, components=("resid",), batch_size=bs)
            Ed_s1 = {l: (Ed[l][0], acts1[("resid", l)].numpy().astype(np.float64).mean(0)) for l in args.e_layers}
        c_user = (lambda r: st["user_tf"].format(statement=r["text"])) if args.c_from == "s2" else s1_user
        conf = [bundle.build_chat(cc["confident_system"], c_user(r)) for r in est]
        caut = [bundle.build_chat(cc["cautious_system"], c_user(r)) for r in est]
        c_layers = sorted(set([cL] + list(args.e_layers)))
        a_conf, _ = capture_last_token(bundle, conf, c_layers, components=("resid",), batch_size=bs)
        a_caut, _ = capture_last_token(bundle, caut, c_layers, components=("resid",), batch_size=bs)
        wC, mC = mean_diff_direction(a_conf[("resid", cL)].numpy().astype(np.float64), a_caut[("resid", cL)].numpy().astype(np.float64))
        if args.c_from == "e1":   # commitment direction from E1 (estimated under the statement prompt WITH the Unknown option)
            wC, mC = z["C_L%d" % cL].astype(np.float64), {"gap": gap_C}
            print("%s: C taken from E1 (idk-prompt estimate), gap %.2f" % (proto, gap_C), flush=True)
        if args.c_orth:
            cosCE = float(np.dot(wC, wE_cL))
            wC = wC - cosCE * wE_cL; wC /= np.linalg.norm(wC)
            print("%s: C orthogonalised against E at L%d (cos before %.3f)" % (proto, cL, cosCE), flush=True)
        if args.e_source == "das":
            import glob, torch
            for l in args.e_layers:
                files = glob.glob(os.path.join(args.das_dir, "das_E_L%d_r1_s*.pt" % l))
                if not files:
                    raise SystemExit("no DAS direction for layer %d in %s" % (l, args.das_dir))
                w_das = torch.load(files[0], map_location="cpu")["W"][:, 0].numpy().astype(np.float64)
                w_das /= np.linalg.norm(w_das)
                Ed[l] = (w_das, Ed[l][1]); Ed_s1[l] = (w_das, Ed_s1[l][1])
            print("%s: DAS E directions loaded from %s" % (proto, args.das_dir), flush=True)
        elif args.e_source == "meandiff_orth":
            for l in args.e_layers:
                wC_l, _ = mean_diff_direction(a_conf[("resid", l)].numpy().astype(np.float64), a_caut[("resid", l)].numpy().astype(np.float64))
                w = Ed[l][0]; cos_before = float(np.dot(w, wC_l))
                w_o = w - np.dot(w, wC_l) * wC_l; w_o /= np.linalg.norm(w_o)
                Ed[l] = (w_o, Ed[l][1]); Ed_s1[l] = (w_o, Ed_s1[l][1])
                print("%s: L%d cos(E, C) before orthogonalisation %.3f -> 0" % (proto, l, cos_before), flush=True)
        dirs[proto] = {"E": Ed, "E_s1": Ed_s1, "C": (wC, float(mC["gap"]))}
        cosE = {l: float(np.dot(Ed[l][0], z["E_L%d" % l]) / (np.linalg.norm(Ed[l][0]) * np.linalg.norm(z["E_L%d" % l]))) for l in args.e_layers}
        print("%s directions re-estimated: cos(E, E1) per layer %s | gap_C %.2f (E1 %.2f) cos(C, E1) %.2f" % (
            proto, {l: round(c, 2) for l, c in cosE.items()}, mC["gap"], gap_C, float(np.dot(wC, z["C_L%d" % cL]) / np.linalg.norm(z["C_L%d" % cL]))), flush=True)

    # ---- conditions ------------------------------------------------------ #
    def E(a, proto, stage="s2"):
        key = "E_s1" if stage == "s1" else "E"
        return [SteerSpec(dirs[proto][key][l][0], l, mode="amplify", alpha=a, center=dirs[proto][key][l][1]) for l in args.e_layers]

    def C(a, proto):   # negative alpha = toward abstention (as in E11)
        return [SteerSpec(dirs[proto]["C"][0], cL, mode="add", alpha=-a, coef=[dirs[proto]["C"][1]])]

    def conds_for(proto, stage="s2"):
        c = [("baseline", None)]
        c += [("E_a%g" % a, E(a, proto, stage)) for a in args.e_alphas]
        c += [("C_abst_a%g" % a, C(a, proto)) for a in args.c_alphas]
        c += [("C_ans_a%g" % a, C(-a, proto)) for a in args.c_alphas]
        c += [("E_a%g+C_abst_a%g" % (args.e_alphas[-1], args.c_alphas[0]), E(args.e_alphas[-1], proto, stage) + C(args.c_alphas[0], proto))]
        return c
    conds = conds_for(args.protocols[0])

    # ---- run -------------------------------------------------------------- #
    t0 = time.time()
    res = {"model": mcfg["name"], "n": len(recs), "e_layers": args.e_layers, "c_layer": cL, "protocols": {}, "samples": samples}
    per_item = {}
    for proto in args.protocols:
        res["protocols"][proto] = {}
        per_item[proto] = {}
        conds = conds_for(proto)
        conds_s1 = dict(conds_for(proto, "s1"))
        for name, specs in conds:
            lg1 = logits_for(bundle, texts[proto]["s1"], conds_s1[name] if (proto in STAGE1 or proto == "split") else specs, bs)
            if proto == "threeway":
                lps = [label_logprobs_from_logits(lg, maps[proto]["s1"]) for lg in lg1]
                answer = np.array([max(lp["True"], lp["False"]) > lp["Abstain"] for lp in lps])
                p_true = tf_scores(lg1, maps[proto]["s1"])
            elif proto == "forced2":
                answer = np.ones(len(recs), dtype=bool)
                p_true = tf_scores(lg1, maps[proto]["s1"])
            elif proto == "split":
                lps = [label_logprobs_from_logits(lg, maps[proto]["s1"], use_abstain=True) for lg in lg1]
                answer = np.array([lp["Answer"] > lp["Abstain"] for lp in lps])
                p_true = tf_scores(logits_for(bundle, texts[proto]["s2"], specs, bs), maps[proto]["s2"])
            else:
                lps = [label_logprobs_from_logits(lg, maps[proto]["s1"], use_abstain=True) for lg in lg1]
                answer = np.array([lp["Answer"] > lp["Abstain"] for lp in lps])
                lg2 = logits_for(bundle, texts[proto]["s2"], specs, bs)
                p_true = tf_scores(lg2, maps[proto]["s2"])
                # Refusal-Index style second pass: ONLY the items that said unknown are re-asked with the forced two-way prompt
                # (same steering); answered items keep their own True/False from the first pass.
                idk = np.where(~answer)[0]
                p_true_ri = p_true.copy()
                if len(idk):
                    f_texts = [tf_prompt(bundle, prompts, recs[i]["text"], allow_idk=False) for i in idk]
                    f_map = build_label_token_map(bundle.tokenizer, f_texts[0], tf_variants)
                    # the forced prompt gets the directions/centres estimated under the forced prompt (forced2), if available
                    specs_f = dict(conds_for("forced2"))[name] if "forced2" in dirs else specs
                    p_true_ri[idk] = tf_scores(logits_for(bundle, f_texts, specs_f, bs), f_map)
                m_ri = metrics(answer, p_true_ri, labels)
            m = metrics(answer, p_true, labels)
            if proto in STAGE1:
                m.update({"J_all_RI": m_ri["J_all"], "AUROC_all_RI": m_ri["AUROC_all"], "B_all_RI": m_ri["B_all"], "n_idk_reasked": int(len(idk))})
                per_item[proto][name + "__RI"] = (answer, p_true_ri)
            per_item[proto][name] = (answer, p_true)
            res["protocols"][proto][name] = m
            if name == "baseline" and proto == "split":
                gen = []
                for i in range(0, len(recs), bs):
                    outs, _ = generate(bundle, bundle.encode(texts[proto]["s1"][i:i + bs]), None, max_new_tokens=3)
                    gen += outs
                ga, gb = SPLIT_A["gen"]
                g_abst = np.array([g.strip().lower().startswith(gb) for g in gen])
                g_first = np.array([g.strip().lower().startswith(ga) for g in gen]) & ~g_abst
                cons = {"n": len(gen), "gen_know": int(g_first.sum()), "gen_unknown": int(g_abst.sum()), "gen_unparsed": int(len(gen) - g_first.sum() - g_abst.sum()),
                        "readout_agrees_with_generation": int(np.sum(answer[g_first | g_abst] == g_first[g_first | g_abst])), "examples": [g.strip()[:16] for g in gen[:8]]}
                res["protocols"][proto]["consistency"] = cons
                print("  [%s] pass-A consistency vs generation: %s" % (proto, cons), flush=True)
            if name == "baseline" and proto in STAGE1:
                # does the model's own generation agree with the two readouts?  (赵磊: could the T/F it says after the first
                # word differ from the T/F we read with the prefix forced?)
                gen = []
                for i in range(0, len(recs), bs):
                    outs, _ = generate(bundle, bundle.encode(texts[proto]["s1"][i:i + bs]), None, max_new_tokens=5)
                    gen += outs
                ans_w, abst_w = STAGE1[proto]["variants"]["Answer"][0].strip().lower(), STAGE1[proto]["variants"]["Abstain"][0].strip().lower()
                def after_comma(g):
                    gl = g.strip().lower()
                    return gl.split(",", 1)[1] if ("," in gl and gl.startswith(ans_w)) else ""
                g_first = np.array([g.strip().lower().startswith(ans_w) for g in gen])
                g_abst = np.array([g.strip().lower().startswith(abst_w) for g in gen])
                g_tf = np.array([("true" in after_comma(g)) for g in gen])                       # True iff the model itself said True
                g_has_tf = np.array([("true" in after_comma(g)) or ("false" in after_comma(g)) for g in gen])
                cons = {"n": len(gen), "gen_first_is_answer": int(g_first.sum()), "gen_first_is_abstain": int(g_abst.sum()),
                        "gen_unparsed_first": int(len(gen) - g_first.sum() - g_abst.sum()),
                        "stage1_readout_agrees_with_generation": int(np.sum(answer[g_first | g_abst] == g_first[g_first | g_abst])),
                        "answered_with_parsable_tf": int(g_has_tf.sum()),
                        "stage2_readout_agrees_with_generated_tf": int(np.sum((p_true[g_has_tf] > 0.5) == g_tf[g_has_tf])),
                        "examples": [g.strip()[:24] for g in gen[:6]]}
                res["protocols"][proto]["consistency"] = cons
                print("  [%s] consistency vs greedy generation: %s" % (proto, cons), flush=True)
            print("  [%s] %-18s ans %.3f | all: J %+.3f AUROC %.3f B %.3f | sel: J %+.3f acc %.3f B %.3f (n=%d)%s  (%.0fs)" % (
                proto, name, m["answer_rate"], m["J_all"], m["AUROC_all"], m["B_all"], m["J_sel"], m["acc_sel"], m["B_sel"], m["n_answered"],
                (" | RI: J %+.3f B %.3f (re-asked %d)" % (m["J_all_RI"], m["B_all_RI"], m["n_idk_reasked"])) if "J_all_RI" in m else "", time.time() - t0), flush=True)
        # bootstrap CIs for deltas vs baseline (pairs resampled)
        pairs = {}
        for i, pid in enumerate(pair_ids):
            pairs.setdefault(str(pid), []).append(i)
        plist = list(pairs.values())
        base_a, base_p = per_item[proto]["baseline"]
        for name, _ in conds[1:]:
            a, p = per_item[proto][name]
            dJ, dA, dR = [], [], []
            for _ in range(args.n_boot):
                idx = np.concatenate([plist[k] for k in rng.integers(0, len(plist), len(plist))])
                dJ.append(j_stat(p[idx] > 0.5, labels[idx]) - j_stat(base_p[idx] > 0.5, labels[idx]))
                dA.append(float(auroc(p[idx], labels[idx]) - auroc(base_p[idx], labels[idx])))
                dR.append(float(a[idx].mean() - base_a[idx].mean()))
            res["protocols"][proto][name].update({
                "dJ_all_ci": [float(np.quantile(dJ, .025)), float(np.quantile(dJ, .975))],
                "dAUROC_all_ci": [float(np.quantile(dA, .025)), float(np.quantile(dA, .975))],
                "dAnswer_ci": [float(np.quantile(dR, .025)), float(np.quantile(dR, .975))]})

    out_dir = os.path.join(RESULTS_DIR, "E22", mcfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    stem = "two_stage" + (("_" + args.tag) if args.tag else "")
    np.savez(os.path.join(out_dir, stem + ".npz"), ids=np.array([r["id"] for r in recs]), labels=labels, pair_ids=np.array(pair_ids),
             **{"%s__%s__answer" % (pr, nm): v[0] for pr in per_item for nm, v in per_item[pr].items()},
             **{"%s__%s__ptrue" % (pr, nm): v[1] for pr in per_item for nm, v in per_item[pr].items()})
    res["elapsed_s"] = time.time() - t0
    atomic_json_dump(res, os.path.join(out_dir, stem + ".json"))
    lines = ["# E22 readout protocols, %s (n=%d; E layers %s; C layer %d)" % (mcfg["name"], len(recs), args.e_layers, cL), ""]
    for proto in args.protocols:
        lines += ["## %s" % proto, "", "greedy samples (baseline): %s" % [o.strip()[:30] for o in samples[proto]], "",
                  ("consistency vs generation: %s" % res["protocols"][proto]["consistency"]) if "consistency" in res["protocols"][proto] else "", "",
                  "| condition | answer rate (d) | J_all (d [CI]) | AUROC_all (d [CI]) | B_all | J_sel | acc_sel | B_sel | n_ans | J_all RI-style | AUROC_all RI | B_all RI | n re-asked |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        b = res["protocols"][proto]["baseline"]
        for name, _ in conds:
            m = res["protocols"][proto][name]
            dJ = m.get("dJ_all_ci"); dA = m.get("dAUROC_all_ci"); dR = m.get("dAnswer_ci")
            lines.append("| %s | %.3f (%+.3f%s) | %+.3f (%+.3f%s) | %.3f (%+.3f%s) | %.3f | %+.3f | %.3f | %.3f | %d |" % (
                name, m["answer_rate"], m["answer_rate"] - b["answer_rate"], (" [%+.3f,%+.3f]" % tuple(dR)) if dR else "",
                m["J_all"], m["J_all"] - b["J_all"], (" [%+.3f,%+.3f]" % tuple(dJ)) if dJ else "",
                m["AUROC_all"], m["AUROC_all"] - b["AUROC_all"], (" [%+.3f,%+.3f]" % tuple(dA)) if dA else "",
                m["B_all"], m["J_sel"], m["acc_sel"], m["B_sel"], m["n_answered"])
                + (" %+.3f | %.3f | %.3f | %d |" % (m["J_all_RI"], m["AUROC_all_RI"], m["B_all_RI"], m["n_idk_reasked"]) if "J_all_RI" in m else " – | – | – | – |"))
        lines.append("")
    with open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
