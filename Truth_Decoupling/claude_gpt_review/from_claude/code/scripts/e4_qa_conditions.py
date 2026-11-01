#!/usr/bin/env python3
"""E4/E5 (fixed-alpha part): QA-level generation under conditions, then LLM-judge grading.

Two phases, like the v12 workflow (GPU generation on the server, API judging anywhere):

  --phase generate   greedy answers for each QA dataset under each condition, saved to
                     results/E4/<model>/<dataset>_<condition>.json  (resumable)
  --phase judge      rule abstention + LLM judge (SimpleQA-style A/B/C) per answer, cached;
                     then per-condition rates, three-way decomposition vs baseline,
                     known/unknown stratification, SimpleQA F-score; summary JSON.

Conditions: baseline | cautious_prompt (system swap, prompt-only reference) | C_add (abstention
steering) | E_amp (evidence gain) | Both. Directions from E1 (write layers by default).

Usage:
  python3 scripts/e4_qa_conditions.py --phase generate --model qwen3-8b --datasets simpleqa truthfulqa --split dev
  DEEPSEEK_API_KEY=... python3 scripts/e4_qa_conditions.py --phase judge --model qwen3-8b --datasets simpleqa truthfulqa --split dev
"""
import argparse
import glob
import os
import time

from common import RESULTS_DIR, load_split

import numpy as np

from core.config import load_experiment, load_model_config
from core.io_utils import ResumableResults, atomic_json_dump, load_json, set_seed
from core.judge import combine_outcome, judge_from_config
from core.metrics import QA_ABSTAIN, QA_CORRECT, QA_WRONG, qa_rates, stratified_report
from core.readout import rule_abstains

CONDITIONS = ["baseline", "cautious_prompt", "C_add", "E_amp", "Both"]


def build_specs(exp, e1_dir, e1, cond):
    from core.intervene import SteerSpec
    z = np.load(os.path.join(e1_dir, "directions_meandiff.npz"))
    eL = exp.get("e_layer") or e1["E_write_layer"]
    cL = exp.get("c_layer") or e1["C_write_layer"]
    aE, aC = exp.get("alpha_e", 1.0), exp.get("alpha_c", 1.0)
    E = SteerSpec(z["E_L%d" % eL], eL, mode="amplify", alpha=aE, center=z["E_center_L%d" % eL], name="E_amp")
    C = SteerSpec(z["C_L%d" % cL], cL, mode="add", alpha=-aC, coef=[e1["rows"][str(cL)]["gap_C"]], name="C_add")
    return {"baseline": [], "cautious_prompt": [], "C_add": [C], "E_amp": [E], "Both": [E, C]}[cond]


def phase_generate(exp, args):
    from core.intervene import generate
    from core.model_io import load_model
    mcfg, prompts = exp["model"], exp["prompts"]
    e1_dir = os.path.join(RESULTS_DIR, "E1", mcfg["name"])
    e1 = load_json(os.path.join(e1_dir, "summary.json"))
    bundle = load_model(mcfg)
    qa = prompts["qa"]
    bs = exp.get("batch_size", 8)
    for ds in args.datasets:
        recs = load_split(ds, args.split)
        if args.limit:
            recs = recs[: args.limit]
        for cond in args.conditions:
            path = os.path.join(RESULTS_DIR, "E4", mcfg["name"], "%s_%s_%s.json" % (ds, args.split, cond))
            rr = ResumableResults(path, header={"model": mcfg["name"], "dataset": ds, "split": args.split, "condition": cond,
                                                "e_layer": e1["E_write_layer"], "c_layer": e1["C_write_layer"]}, flush_every=2)
            specs = build_specs(exp, e1_dir, e1, cond)
            system = qa["system_cautious"] if cond == "cautious_prompt" else qa["system"]
            t0 = time.time()
            for i in range(0, len(recs), bs):
                idxs = list(range(i, min(i + bs, len(recs))))
                if all(rr.done(j) for j in idxs):
                    continue
                texts = [bundle.build_chat(system, qa["user"].format(question=recs[j]["text"])) for j in idxs]
                outs, hooks = generate(bundle, bundle.encode(texts), specs=specs, max_new_tokens=qa.get("max_new_tokens", 48))
                for j, o in zip(idxs, outs):
                    rr.add(j, {"id": recs[j]["id"], "question": recs[j]["text"], "gold": recs[j]["gold"], "incorrect": recs[j].get("incorrect", []),
                               "meta": recs[j].get("meta", {}), "output": o, "rule_abstain": rule_abstains(o)})
                if (i // bs) % 10 == 0:
                    print("  [%s/%s] %d/%d (%.0fs)" % (ds, cond, min(i + bs, len(recs)), len(recs), time.time() - t0), flush=True)
            rr.finalize({"n": len(rr.evaluations), "rule_abstain_rate": float(np.mean([e["rule_abstain"] for e in rr.evaluations]))})
            print("  saved", path, flush=True)


def phase_judge(exp, args):
    mcfg = exp["model"]
    judge = judge_from_config(exp["judge"], cache_dir=os.path.join(RESULTS_DIR, "judge_cache"))
    second = judge_from_config(exp["judge_second"], cache_dir=os.path.join(RESULTS_DIR, "judge_cache")) if exp.get("judge_second") and args.second_judge else None
    summary = {"model": mcfg["name"], "split": args.split, "judge": judge.name, "datasets": {}}
    for ds in args.datasets:
        outcomes = {}
        for cond in args.conditions:
            path = os.path.join(RESULTS_DIR, "E4", mcfg["name"], "%s_%s_%s.json" % (ds, args.split, cond))
            if not os.path.exists(path):
                print("  missing", path, "(skip)")
                continue
            data = load_json(path)
            evs = sorted(data["evaluations"], key=lambda e: e["idx"])
            labs = []
            for e in evs:
                if "judge" not in e or args.rejudge:
                    g = judge.qa_grade(e["question"], e["gold"], e["output"], incorrect=e.get("incorrect") or None)
                    e["judge"] = g
                    if second is not None:
                        e["judge2"] = second.qa_grade(e["question"], e["gold"], e["output"], incorrect=e.get("incorrect") or None)
                    e["outcome"] = combine_outcome(g["label"], e["rule_abstain"])
                labs.append(e["outcome"])
            data["evaluations"] = evs
            data["judge_summary"] = {"rates": qa_rates([l for l in labs if l != "unresolved"]), "n_unresolved": sum(l == "unresolved" for l in labs)}
            if second is not None:
                agree = [e["judge"]["label"] == e["judge2"]["label"] for e in evs if "judge2" in e]
                data["judge_summary"]["second_judge_agreement"] = float(np.mean(agree)) if agree else None
            atomic_json_dump(data, path)
            outcomes[cond] = labs
            print("  %s/%s: %s" % (ds, cond, data["judge_summary"]["rates"]), flush=True)
        if "baseline" in outcomes:
            per_cond = {}
            base = outcomes["baseline"]
            for cond, labs in outcomes.items():
                rep = stratified_report(base, labs) if cond != "baseline" else {"all": {"rates": qa_rates(base)}}
                c = sum(l == QA_CORRECT for l in labs)
                w = sum(l == QA_WRONG for l in labs)
                a = sum(l == QA_ABSTAIN for l in labs)
                rep["simpleqa_F"] = (2 * c) / (2 * c + 2 * w + a) if (2 * c + 2 * w + a) else float("nan")
                rep["penalised_score"] = (c - w) / len(labs) if labs else float("nan")  # abstain = 0, wrong = -1
                per_cond[cond] = rep
            summary["datasets"][ds] = per_cond
    out = os.path.join(RESULTS_DIR, "E4", mcfg["name"], "summary_%s.json" % args.split)
    atomic_json_dump(summary, out)
    print("judge calls: %d (cache hits %d) ->" % (judge.n_calls, judge.n_cache_hits), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E4_qa")
    ap.add_argument("--model", default=None)
    ap.add_argument("--phase", choices=["generate", "judge"], required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--second-judge", action="store_true")
    ap.add_argument("--rejudge", action="store_true")
    args = ap.parse_args()
    exp = load_experiment(args.exp)
    if args.model:
        exp["model"] = load_model_config(args.model)
    if not args.datasets:
        args.datasets = exp.get("qa_datasets", ["simpleqa", "truthfulqa", "halueval_qa"])
    set_seed(exp.get("seed", 42))
    if args.phase == "generate":
        phase_generate(exp, args)
    else:
        phase_judge(exp, args)


if __name__ == "__main__":
    main()
