#!/usr/bin/env python3
"""Baseline discriminability of the frozen v1 self-check readout (Claude, round 16 follow-up, 2026-11-05).

usage: selfcheck_baseline.py MODEL OUT_JSON [CODE_DIR]
For every tuning pair (192), teacher-force the correct and the wrong answer with the frozen suffix
("Is the answer above factually correct? ... Verdict:") and read log P(" True") - log P(" False") at the label
predictor, fp32 output projection, no intervention. Reports AUROC of that margin (correct vs wrong answer), how often the
verdict is True for each side, and what the model's actual top token at "Verdict:" is. This is the supervision signal
the QA-DAS objective is trained against; if it is at chance, a direction that flips it cannot carry truth.
"""
from collections import Counter
import json
import sys

import numpy as np
import torch

CODE = sys.argv[3] if len(sys.argv) > 3 else "/workspace/Truth_Decoupling/review_e25/claude_gpt_review/from_claude/code"
sys.path[:0] = [CODE + "/scripts", CODE]
import common  # noqa: F401,E402
import e25_qa_context as legacy  # noqa: E402
from core.config import load_json_config, load_model_config, resolve_config_path  # noqa: E402
from core.model_io import forward_logits, load_model  # noqa: E402
from core.qa_das import prepare_selfcheck_pairs  # noqa: E402
from core.qa_precision import fp32_output_projection  # noqa: E402

model, out = sys.argv[1], sys.argv[2]
cfg = load_json_config(resolve_config_path("experiments", "E25_qa_das_v1"))
prep = legacy.split_questions(cfg)
bundle = load_model(load_model_config(model))
pairs = prepare_selfcheck_pairs(bundle, cfg, prep["e_tune"])
rows = []
with fp32_output_projection(bundle), torch.no_grad():
    for p in pairs:
        r = {"pair_id": p["pair_id"]}
        for side in ("positive", "negative"):
            item = p[side]
            logits = forward_logits(bundle, item["inputs"], logits_to_keep=1)[:, -1, :].float()
            lp = torch.log_softmax(logits, -1)[0]
            t, f = item["label_ids"]["true"][0], item["label_ids"]["false"][0]
            top = int(logits.argmax())
            r[side] = {"logp_true": float(lp[t]), "logp_false": float(lp[f]), "margin": float(lp[t] - lp[f]),
                       "p_true_plus_false": float(lp[t].exp() + lp[f].exp()),
                       "top_token_id": top, "top_token": bundle.tokenizer.decode([top]), "answer": item["text_audit"]["answer"]}
        rows.append(r)
pos = np.array([r["positive"]["margin"] for r in rows])
neg = np.array([r["negative"]["margin"] for r in rows])
auroc = float(np.mean([(1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg]))
summary = {
    "model": model, "n_pairs": len(rows), "suffix": cfg["selfcheck_suffix"], "labels": [cfg["selfcheck_label_true"], cfg["selfcheck_label_false"]],
    "auroc_margin_correct_vs_wrong": auroc, "paired_correct_answer_has_higher_true_margin": int((pos > neg).sum()),
    "says_true_for_correct": int((pos > 0).sum()), "says_true_for_wrong": int((neg > 0).sum()),
    "mean_margin_correct": float(pos.mean()), "mean_margin_wrong": float(neg.mean()),
    "mean_p_true_plus_false": {side: float(np.mean([r[side]["p_true_plus_false"] for r in rows])) for side in ("positive", "negative")},
    "top_token_counts": {side: dict(Counter(r[side]["top_token"] for r in rows).most_common(8)) for side in ("positive", "negative")},
}
json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, ensure_ascii=False))
