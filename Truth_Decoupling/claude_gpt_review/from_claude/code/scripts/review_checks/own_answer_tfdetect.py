#!/usr/bin/env python3
"""The TF-protocol E coordinate as a detector of the model's OWN wrong answers (Claude, 2026-11-05).

usage: own_answer_tfdetect.py MODEL V1_RUN_DIR OUT_JSON [--code-dir DIR] [--n-boot N]
Own answers = the v1 baseline free-generation answers with their semantic judge labels (192 tuning, 523 main-table test
questions; abstentions and refusal-only-gold items excluded). Each answer is written as the TF statement and read under the
TF prompt (exactly as verify_own_answers.py). Scores:
  coord_<est>_L  projection of the last-prompt-token block output on the layer-L E direction of estimator <est>:
                 dasce (gold-label CE DAS, results_das_tfce E1), daskl (round-4 KL DAS, results_das_tf E1), md (mean
                 difference from the 192 estimation statements, rebuilt as e23 does); sign oriented on tuning data;
  verdict        the unsteered TF verdict margin log P(True) - log P(False) (the model's own judgement).
Per estimator the layer with the best TUNING AUROC is picked; TEST AUROC with paired bootstrap differences against the
verdict and between estimators.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("v1_run")
ap.add_argument("out")
ap.add_argument("--code-dir", default="/workspace/Truth_Decoupling/review_e25/claude_gpt_review/from_claude/code")
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()
sys.path[:0] = [args.code_dir + "/scripts", args.code_dir]
import common  # noqa: F401,E402
from common import load_split, tf_prompt  # noqa: E402
from core.config import load_experiment, load_model_config  # noqa: E402
from core.directions import load_das_direction, mean_diff_direction  # noqa: E402
from core.model_io import capture_last_token, load_model  # noqa: E402
from core.readout import build_label_token_map, label_logprobs_from_logits  # noqa: E402

ROOT = "/workspace/Truth_Decoupling"
exp = load_experiment("E2b_tqa")
prompts = exp["prompts"]
st = prompts["statement"]
run = Path(args.v1_run)
bundle = load_model(load_model_config(args.model))
tok = bundle.tokenizer
L = list(range(bundle.num_layers))

est = []
for ds in exp["eval_datasets"]:
    est += load_split(ds, "calib") + load_split(ds, "dev")
y_est = np.array([bool(r["label"]) for r in est])
aE, _ = capture_last_token(bundle, [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in est], L, batch_size=8)
dirs = {"md": {}, "dasce": {}, "daskl": {}}
for l in L:
    HB = aE[("resid", l)].numpy().astype(np.float64)
    wmd, _ = mean_diff_direction(HB[y_est], HB[~y_est])
    dirs["md"][l] = wmd
    for name, d, var in (("dasce", "%s/results_das_tfce/E1/%s" % (ROOT, args.model), "tf-ce-bidir"),
                         ("daskl", "%s/results_das_tf/E1/%s" % (ROOT, args.model), "tf")):
        w, _ = load_das_direction(d, "E", l, variant=var, align_to=wmd)
        if w is not None:
            dirs[name][l] = np.asarray(w, dtype=np.float64)


def statement(q, a):
    a = " ".join(a.split())
    return 'The answer to the question "%s" is: %s' % (q.strip().rstrip("?").strip() + "?", a.rstrip(".") + ".")


data = {}
for split, fname in (("tune", "tune_scored_items.json"), ("test", "test_scored_items.json")):
    rows = [e for e in json.loads((run / fname).read_text())["evaluations"]
            if not e.get("gold_is_abstention") and e["outcome"]["baseline"] in ("correct", "wrong") and e["outputs"]["baseline"].strip()]
    texts = [tf_prompt(bundle, prompts, statement(e["question"], e["outputs"]["baseline"]), allow_idk=False) for e in rows]
    acts, lg = capture_last_token(bundle, texts, L, batch_size=8, return_logits=True)
    mB = build_label_token_map(tok, texts[0], {"True": st["label_variants"]["True"], "False": st["label_variants"]["False"]})
    verdict = []
    for x in lg.numpy():
        lp = label_logprobs_from_logits(x, mB, use_abstain=False)
        verdict.append(lp["True"] - lp["False"])
    data[split] = {"y": np.array([1 if e["outcome"]["baseline"] == "correct" else 0 for e in rows]),
                   "H": {l: acts[("resid", l)].numpy().astype(np.float64) for l in L}, "verdict": np.array(verdict),
                   "ids": [e["pair_id"] for e in rows]}
    print(split, "answered", len(rows), "correct", int(data[split]["y"].sum()), flush=True)


def auroc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


yt, ys = data["tune"]["y"], data["test"]["y"]
scores, table = {"verdict": (data["tune"]["verdict"], data["test"]["verdict"])}, {}
for est_name, by_layer in dirs.items():
    for l, w in by_layer.items():
        st_, ss_ = data["tune"]["H"][l] @ w, data["test"]["H"][l] @ w
        sign = 1.0 if auroc(st_, yt) >= 0.5 else -1.0
        scores["%s_L%d" % (est_name, l)] = (sign * st_, sign * ss_)
picks = {"verdict": "verdict"}
for est_name in dirs:
    names = [k for k in scores if k.startswith(est_name + "_L")]
    if names:
        picks[est_name] = max(names, key=lambda k: (auroc(scores[k][0], yt), -int(k.split("_L")[1])))
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ys), len(ys)) for _ in range(args.n_boot)]
for fam, name in picks.items():
    a = auroc(scores[name][1], ys)
    bd = [auroc(scores[name][1][i], ys[i]) for i in boot]
    table[fam] = {"pick": name, "tune_auroc": auroc(scores[name][0], yt), "test_auroc": a,
                  "test_auroc_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
paired = {}
for a_f, b_f in (("dasce", "verdict"), ("daskl", "verdict"), ("md", "verdict"), ("dasce", "md"), ("dasce", "daskl")):
    if a_f in picks and b_f in picks:
        sa, sb = scores[picks[a_f]][1], scores[picks[b_f]][1]
        d = [auroc(sa[i], ys[i]) - auroc(sb[i], ys[i]) for i in boot]
        paired["%s_minus_%s" % (a_f, b_f)] = {"delta": auroc(sa, ys) - auroc(sb, ys),
                                              "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
per_layer_test = {est_name: {str(l): auroc(scores["%s_L%d" % (est_name, l)][1], ys) for l in dirs[est_name]} for est_name in dirs}
json.dump({"model": args.model, "n": {"tune": int(len(yt)), "test": int(len(ys)), "test_correct": int(ys.sum())},
           "picked": table, "paired_test_auroc_differences": paired, "per_layer_test_auroc": per_layer_test}, open(args.out, "w"), indent=1)
for fam, t in table.items():
    print("%-8s %-10s tune AUROC %.3f | TEST AUROC %.3f [%.3f, %.3f]" % (fam, t["pick"], t["tune_auroc"], t["test_auroc"], *t["test_auroc_ci95"]))
for k, v in paired.items():
    print("paired TEST AUROC %-20s %+.3f [%+.3f, %+.3f]" % (k, v["delta"], *v["ci95"]))
print("saved", args.out)
