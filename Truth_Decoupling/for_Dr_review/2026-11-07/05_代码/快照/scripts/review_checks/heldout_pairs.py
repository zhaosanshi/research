#!/usr/bin/env python3
"""Held-out paired-answer preference (MC-style) for the QA-DAS v1 directions (Claude, 2026-11-05).

usage: heldout_pairs.py MODEL RUN_DIR OUT_JSON [--code-dir DIR] [--n-boot N]
The v1 proxy (tune-e) scores, on the 192 TUNING pairs only, whether the summed conditional log-probability of the
correct answer exceeds that of one incorrect answer under each intervention. This script applies exactly the same
scoring code path (e25_qa_das.score: encode_answer, interleaved pairs, pack_answers, legacy.specs_for masks, fp32 head)
to the cleaned TEST pairs, for the baseline and every shortlisted condition, without changing any selection:
the pre-specified test pick per estimator is shortlist[estimator][0] (best tuning proxy). Paired bootstrap over pairs.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("run_dir")
p.add_argument("out")
p.add_argument("--code-dir", default="/workspace/Truth_Decoupling/review_e25/claude_gpt_review/from_claude/code")
p.add_argument("--n-boot", type=int, default=2000)
args = p.parse_args()
sys.path[:0] = [args.code_dir + "/scripts", args.code_dir]
import common  # noqa: F401,E402
import e25_qa_das as pipeline  # noqa: E402
import e25_qa_context as legacy  # noqa: E402
from core.intervene import steer  # noqa: E402
from core.model_io import forward_logits  # noqa: E402
from core.qa_context import answer_logprobs, encode_answer, pack_answers, prepare_e_pairs  # noqa: E402
from core.qa_precision import fp32_output_projection  # noqa: E402

run = Path(args.run_dir)
ns = SimpleNamespace(run_dir=run, model=args.model)
meta, bank, prepared = pipeline.load_bank(ns)
meta, bank = pipeline.runtime_bank(ns, meta, bank)
short = json.loads((run / "shortlist.json").read_text())
proxy = json.loads((run / "proxy_scores.json").read_text())
cfg = meta["config"]
test_pairs, audit = prepare_e_pairs(prepared["splits"]["test"])
choices = [dict(pipeline.BASELINE)] + [c for est in pipeline.METHODS for c in short["candidates"][est]]
bundle = pipeline.open_bundle(meta, prepared)
encoded = [encode_answer(bundle, cfg["qa_system"], r["question"], r[k], cfg["qa_user"])
           for r in test_pairs for k in ("positive_answer", "negative_answer")]
res = {}
with fp32_output_projection(bundle), torch.no_grad():
    for c in choices:
        vals = []
        for s in range(0, len(encoded), cfg["batch_size"]):
            packed = pack_answers(encoded[s:s + cfg["batch_size"]], bundle.tokenizer.pad_token_id, bundle.device)
            with steer(bundle, legacy.specs_for(meta, bank, c, packed)):
                v = answer_logprobs(forward_logits(bundle, packed["inputs"]), packed).detach().cpu().numpy()
            if not np.isfinite(v).all():
                raise ValueError("non-finite paired score")
            vals.extend(v.tolist())
        pos, neg = np.array(vals[0::2]), np.array(vals[1::2])
        res[c["name"]] = {"condition": c, "correct": (pos > neg).tolist(), "margin": (pos - neg).tolist()}
        print("TEST-PAIRS %-36s acc %.4f" % (c["name"], float((pos > neg).mean())), flush=True)
base = np.array(res["baseline"]["correct"], bool)
n = len(base)
rng = np.random.default_rng(0)
boot = [rng.integers(0, n, n) for _ in range(args.n_boot)]
summary = {"model": args.model, "n_test_pairs": n, "test_pairs_excluded": audit["excluded"],
           "baseline_acc": float(base.mean()), "pre_specified_pick": {est: short["candidates"][est][0]["name"] if short["candidates"][est] else None for est in pipeline.METHODS},
           "conditions": {}}
for name, r in res.items():
    if name == "baseline":
        continue
    x = np.array(r["correct"], bool)
    d = [float(x[i].mean() - base[i].mean()) for i in boot]
    tune = proxy["conditions"][name]
    summary["conditions"][name] = {
        "test_acc": float(x.mean()), "test_delta": float(x.mean() - base.mean()),
        "test_delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "test_corrected": int((~base & x).sum()), "test_harmed": int((base & ~x).sum()),
        "tune_acc": tune["accuracy"], "tune_baseline_acc": proxy["conditions"]["baseline"]["accuracy"],
        "tune_corrected": tune["corrected"], "tune_harmed": tune["harmed"]}
    s = summary["conditions"][name]
    print("  %-36s tune %.3f->%.3f | TEST %.3f->%.3f d %+.4f [%+.4f, %+.4f] corrected %d harmed %d%s" % (
        name, s["tune_baseline_acc"], s["tune_acc"], summary["baseline_acc"], s["test_acc"], s["test_delta"],
        s["test_delta_ci95"][0], s["test_delta_ci95"][1], s["test_corrected"], s["test_harmed"],
        "   <== pre-specified pick" if name in summary["pre_specified_pick"].values() else ""), flush=True)
json.dump({"summary": summary, "per_pair": {k: {"correct": v["correct"], "margin": v["margin"]} for k, v in res.items()},
           "pair_ids": [r["pair_id"] for r in test_pairs]}, open(args.out, "w"), indent=1)
print("saved", args.out)
