#!/usr/bin/env python3
"""Held-out MC1-style accuracy (best answer vs ALL incorrect answers) for the QA-DAS v1 directions (Claude, 2026-11-05).

usage: heldout_mc.py MODEL RUN_DIR OUT_JSON [--code-dir DIR] [--n-boot N]
Same scoring path as e25_qa_das.score / heldout_pairs.py (encode_answer, pack_answers, legacy.specs_for masks,
summed conditional answer log-probability, fp32 head), but each TEST question is scored against every usable incorrect
answer of TruthfulQA (deduplicated; empty, personal-refusal and bare Unknown answers removed as in prepare_e_pairs):
correct iff the best answer has a strictly higher summed log-probability than every incorrect answer. Conditions are the
baseline and the six shortlisted v1 candidates; the pre-specified pick per estimator is shortlist[estimator][0].
This is TruthfulQA-MC1-like on the generation-set answer lists, not the official mc1_targets file.
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
from core.qa_context import _answer_key, _unusable_answer, answer_logprobs, encode_answer, pack_answers, prepare_e_pairs  # noqa: E402
from core.qa_precision import fp32_output_projection  # noqa: E402

run = Path(args.run_dir)
ns = SimpleNamespace(run_dir=run, model=args.model)
meta, bank, prepared = pipeline.load_bank(ns)
meta, bank = pipeline.runtime_bank(ns, meta, bank)
short = json.loads((run / "shortlist.json").read_text())
cfg = meta["config"]
test_pairs, _ = prepare_e_pairs(prepared["splits"]["test"])
questions = []
for r in test_pairs:
    best = r["positive_answer"]
    seen, inc = {_answer_key(best)}, []
    for a in r.get("incorrect") or []:
        if _unusable_answer(a) is None and _answer_key(a.strip()) not in seen:
            seen.add(_answer_key(a.strip()))
            inc.append(a.strip())
    if inc:
        questions.append({"pair_id": r["pair_id"], "question": r["question"], "answers": [best] + inc})
choices = [dict(pipeline.BASELINE)] + [c for est in pipeline.METHODS for c in short["candidates"][est]]
bundle = pipeline.open_bundle(meta, prepared)
flat, owner = [], []
for qi, q in enumerate(questions):
    for ai, a in enumerate(q["answers"]):
        flat.append(encode_answer(bundle, cfg["qa_system"], q["question"], a, cfg["qa_user"]))
        owner.append((qi, ai))
res = {}
with fp32_output_projection(bundle), torch.no_grad():
    for c in choices:
        vals = []
        for s in range(0, len(flat), cfg["batch_size"]):
            packed = pack_answers(flat[s:s + cfg["batch_size"]], bundle.tokenizer.pad_token_id, bundle.device)
            with steer(bundle, legacy.specs_for(meta, bank, c, packed)):
                v = answer_logprobs(forward_logits(bundle, packed["inputs"]), packed).detach().cpu().numpy()
            vals.extend(v.tolist())
        per_q = [[] for _ in questions]
        for (qi, ai), v in zip(owner, vals):
            per_q[qi].append(v)
        correct = [bool(sc[0] > max(sc[1:])) for sc in per_q]
        res[c["name"]] = correct
        print("MC1-STYLE %-36s acc %.4f" % (c["name"], float(np.mean(correct))), flush=True)
base = np.array(res["baseline"], bool)
n = len(base)
rng = np.random.default_rng(0)
boot = [rng.integers(0, n, n) for _ in range(args.n_boot)]
picks = {est: short["candidates"][est][0]["name"] if short["candidates"][est] else None for est in pipeline.METHODS}
summary = {"model": args.model, "n_questions": n, "mean_incorrect_per_question": float(np.mean([len(q["answers"]) - 1 for q in questions])),
           "baseline_acc": float(base.mean()), "pre_specified_pick": picks, "conditions": {}}
for name, x in res.items():
    if name == "baseline":
        continue
    x = np.array(x, bool)
    d = [float(x[i].mean() - base[i].mean()) for i in boot]
    summary["conditions"][name] = {"acc": float(x.mean()), "delta": float(x.mean() - base.mean()),
                                   "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                   "corrected": int((~base & x).sum()), "harmed": int((base & ~x).sum())}
    s = summary["conditions"][name]
    print("  %-36s TEST MC1-style %.3f->%.3f d %+.4f [%+.4f, %+.4f] corrected %d harmed %d%s" % (
        name, summary["baseline_acc"], s["acc"], s["delta"], *s["delta_ci95"], s["corrected"], s["harmed"],
        "   <== pre-specified pick" if name in picks.values() else ""), flush=True)
json.dump({"summary": summary, "per_question": res, "pair_ids": [q["pair_id"] for q in questions]}, open(args.out, "w"), indent=1)
print("saved", args.out)
