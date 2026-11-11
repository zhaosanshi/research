#!/usr/bin/env python3
"""Pass-A readout check for a new model (Claude, 2026-11-12; red-team note 7 of from_claude/41).

On the ESTIMATION statements only (calib + dev; no test item is scored): with the sweep's pass-A wording (default 'pik',
"Do you know the answer to this question? ... Yes or No") and pass-B prompt, report the Yes/No (pass A) and True/False
(pass B) label mass -- the probability the model puts on the label words at the first answer token -- plus the self-report
Yes rate and the five most likely first tokens of a few items. A low mass would make the answer-rate and J readouts meaningless.

usage: newmodel_passA_check.py --model qwen3.5-9b [--wording pik]
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE) + "/..")
import e22_two_stage as e22                                       # noqa: E402
from common import load_split, tf_prompt                          # noqa: E402
from core.config import load_model_config, load_prompt_config     # noqa: E402
from core.intervene import next_token_logits                      # noqa: E402
from core.model_io import load_model                              # noqa: E402
from core.readout import build_label_token_map                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wording", default="pik")
    args = ap.parse_args()
    e22.SPLIT_A = dict(e22.SPLIT_WORDINGS[args.wording])
    prompts = load_prompt_config("default_v1")
    st = prompts["statement"]
    bundle = load_model(load_model_config(args.model))
    recs = load_split("truthfulqa_tf", "calib") + load_split("truthfulqa_tf", "dev")
    tA = [bundle.build_chat(st["system"], e22.split_user(r, e22.SPLIT_A["instr"])) for r in recs]
    tB = [tf_prompt(bundle, prompts, r["text"], allow_idk=False) for r in recs]
    mA = build_label_token_map(bundle.tokenizer, tA[0], e22.SPLIT_A["variants"])
    mB = build_label_token_map(bundle.tokenizer, tB[0], {"True": st["label_variants"]["True"], "False": st["label_variants"]["False"]})
    print("pass A label ids", {k: [[i, bundle.tokenizer.decode([i])] for i in v] for k, v in mA.items()}, flush=True)
    print("pass A prompt tail %r" % tA[0][-160:], flush=True)
    for name, texts, lm in (("A", tA, mA), ("B", tB, mB)):
        masses, firsts, yes = [], [], []
        for i in range(0, len(texts), 8):
            lg = next_token_logits(bundle, bundle.encode(texts[i:i + 8]))
            lp = torch.log_softmax(lg.float(), -1)
            for row in lp:
                p = {k: float(torch.exp(row[v]).sum()) for k, v in lm.items()}
                masses.append(sum(p.values()))
                if name == "A":
                    yes.append(p["Answer"] > p["Abstain"])
                if len(firsts) < 4:
                    top = torch.topk(row, 5)
                    firsts.append([[bundle.tokenizer.decode([int(j)]), round(float(v), 3)] for v, j in zip(top.values, top.indices)])
        m = np.array(masses)
        print("pass %s: label mass median %.4f, min %.4f, share < 0.5: %.3f%s" % (
            name, float(np.median(m)), float(m.min()), float((m < 0.5).mean()),
            ("; self-report Yes rate %.3f" % float(np.mean(yes))) if name == "A" else ""), flush=True)
        for f in firsts:
            print("  top5 %s" % f, flush=True)


if __name__ == "__main__":
    main()
