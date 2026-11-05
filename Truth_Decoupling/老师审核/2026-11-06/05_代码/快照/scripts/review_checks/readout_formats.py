#!/usr/bin/env python3
"""Which self-check readout format actually separates correct from incorrect answers? (Claude, 2026-11-05)

usage: readout_formats.py MODEL OUT_JSON [--limit N] [--code-dir DIR]
No intervention, no training. For every cleaned TruthfulQA pair (92 estimation, 192 tuning, test pairs) and both of its
answers (best correct answer, one incorrect answer), read the model's True/False verdict in four formats and report how
well the margin log P(True) - log P(False) separates the correct from the incorrect answer.

  F0 same_turn       QA-DAS v1 exactly: [QA prompt][answer][suffix "\n\nIs the answer above factually correct?...Verdict:"]
                     in the same assistant turn; labels " True" / " False" (single tokens, as trained by v1).
  F1 user_turn       [QA prompt][answer] then the assistant turn is CLOSED and a NEW user turn asks
                     "Is the answer above factually correct?\nReply with exactly one word: True or False.";
                     verdict read at the start of the next assistant turn. Answer token ids are the v1 ids, unchanged.
  F2 tf_statement    The TF protocol used since round 4: fact-checker system prompt, user turn
                     'Statement: The answer to the question "Q?" is: A.\nIs this statement true or false? ...'.
  F3 proposed_answer One fresh user turn: "Question: Q\nProposed answer: A\nIs the proposed answer factually correct?
                     Reply with exactly one word: True or False." under the QA system prompt.

F1-F3 read the label by logsumexp over the first tokens of the label variants ("True", " True", "true", ...), as the TF
protocol does (core/readout.py). F0 reports both the v1 single-token margin and the variant-merged margin.
Output: per split and format AUROC (unpaired), paired accuracy, J at margin 0, says-True rates, mean margins, label mass.
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("out")
p.add_argument("--limit", type=int, default=None, help="pairs per split (smoke test)")
p.add_argument("--code-dir", default="/workspace/Truth_Decoupling/review_e25/claude_gpt_review/from_claude/code")
args = p.parse_args()
sys.path[:0] = [args.code_dir + "/scripts", args.code_dir]
import common  # noqa: F401,E402
import e25_qa_context as legacy  # noqa: E402
from common import tf_prompt  # noqa: E402
from core.config import load_experiment, load_json_config, load_model_config, resolve_config_path  # noqa: E402
from core.model_io import forward_logits, load_model  # noqa: E402
from core.qa_context import _token_ids, encode_answer, prepare_e_pairs  # noqa: E402
from core.qa_das import encode_selfcheck  # noqa: E402
from core.qa_precision import fp32_output_projection  # noqa: E402
from core.readout import DEFAULT_LABEL_VARIANTS, build_label_token_map  # noqa: E402

SC_USER = "Is the answer above factually correct?\nReply with exactly one word: True or False."
PROPOSED_USER = "Question: {question}\nProposed answer: {answer}\nIs the proposed answer factually correct? Reply with exactly one word: True or False."
DUMMY = "QQZZDUMMYANSWERZZQQ"
VARIANTS = {"True": DEFAULT_LABEL_VARIANTS["True"], "False": DEFAULT_LABEL_VARIANTS["False"]}

cfg = load_json_config(resolve_config_path("experiments", "E25_qa_das_v1"))
prompts = load_experiment(cfg["source_experiment"])["prompts"]
prep = legacy.split_questions(cfg)
test_pairs, test_audit = prepare_e_pairs(prep["splits"]["test"])
splits = {"est": prep["e_train"], "tune": prep["e_tune"], "test": test_pairs}
if args.limit:
    splits = {k: v[:args.limit] for k, v in splits.items()}
model_cfg = load_model_config(args.model)
bundle = load_model(model_cfg)
tok = bundle.tokenizer
stub = (bundle.cfg.get("chat") or {}).get("thinking_stub", "") or ""


def followup_rest(system, user_text):
    """Template text that closes the assistant answer turn, opens a user turn with SC_USER and a new assistant turn."""
    chat = bundle.cfg.get("chat", {})
    if system and chat.get("supports_system", True):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
    else:
        msgs = [{"role": "user", "content": (system.strip() + "\n\n" + user_text) if system else user_text}]
    msgs += [{"role": "assistant", "content": DUMMY}, {"role": "user", "content": SC_USER}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    i = text.index(DUMMY) + len(DUMMY)
    return text[i:] + stub


REST = followup_rest(cfg["qa_system"], cfg["qa_user"].format(question="Q"))
assert DUMMY not in REST and SC_USER in REST, REST


def label_map(prompt_text):
    m = build_label_token_map(tok, prompt_text, VARIANTS)
    return {k: sorted(set(v)) for k, v in m.items()}


def readout(ids, lmap, single=None):
    x = torch.tensor([ids], dtype=torch.long, device=bundle.device)
    logits = forward_logits(bundle, {"input_ids": x, "attention_mask": torch.ones_like(x)}, logits_to_keep=1)[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    out = {}
    for lab in ("True", "False"):
        out[lab] = float(torch.logsumexp(lp[lmap[lab]], 0))
    out["margin"] = out["True"] - out["False"]
    out["mass"] = math.exp(out["True"]) + math.exp(out["False"])
    out["top"] = tok.decode([int(logits.argmax())])
    if single is not None:
        out["margin_single"] = float(lp[single[0]] - lp[single[1]])
    return out


def encode_format(fmt, rec, side):
    q, a = rec["question"], rec["positive_answer" if side == "pos" else "negative_answer"]
    if fmt == "F0_same_turn":
        e = encode_selfcheck(bundle, cfg["qa_system"], q, a, cfg["qa_user"], cfg["selfcheck_suffix"],
                             cfg["selfcheck_label_true"], cfg["selfcheck_label_false"])
        text = e["text_audit"]["display_text"]
        single = (e["label_ids"]["true"][0], e["label_ids"]["false"][0])
        return e["input_ids"], text, single
    if fmt == "F1_user_turn":
        enc = encode_answer(bundle, cfg["qa_system"], q, a, cfg["qa_user"])
        rest = followup_rest(cfg["qa_system"], cfg["qa_user"].format(question=q))
        ids = enc["full_ids"] + _token_ids(tok, rest)
        text = enc["full_text"] + rest
        if _token_ids(tok, text) != ids:
            raise ValueError("F1 retokenization at the answer/turn boundary for %r" % rec["pair_id"])
        return ids, text, None
    if fmt == "F2_tf_statement":
        st = 'The answer to the question "%s" is: %s' % (q.strip().rstrip("?").strip() + "?", a.strip().rstrip(".") + ".")
        text = tf_prompt(bundle, prompts, st, allow_idk=False)
        return _token_ids(tok, text), text, None
    if fmt == "F3_proposed_answer":
        text = bundle.build_chat(cfg["qa_system"], PROPOSED_USER.format(question=q.strip(), answer=a.strip()))
        return _token_ids(tok, text), text, None
    raise ValueError(fmt)


FORMATS = ["F0_same_turn", "F1_user_turn", "F2_tf_statement", "F3_proposed_answer"]


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


t0 = time.time()
result = {"model": args.model, "rest_template": REST, "sc_user": SC_USER, "proposed_user": PROPOSED_USER,
          "n_pairs": {k: len(v) for k, v in splits.items()}, "test_pair_audit_excluded": test_audit["excluded"],
          "summary": {}, "examples": {}, "rows": {}}
with fp32_output_projection(bundle), torch.no_grad():
    for split, recs in splits.items():
        result["rows"][split] = []
        for fmt in FORMATS:
            margins = {"pos": [], "neg": []}
            singles = {"pos": [], "neg": []}
            masses, tops = [], {"pos": {}, "neg": {}}
            for rec in recs:
                row = {"pair_id": rec["pair_id"], "format": fmt}
                for side in ("pos", "neg"):
                    ids, text, single = encode_format(fmt, rec, side)
                    lmap = label_map(text)
                    r = readout(ids, lmap, single)
                    margins[side].append(r["margin"])
                    masses.append(r["mass"])
                    tops[side][r["top"]] = tops[side].get(r["top"], 0) + 1
                    if single is not None:
                        singles[side].append(r["margin_single"])
                    row[side] = {"margin": r["margin"], "mass": r["mass"], "top": r["top"],
                                 **({"margin_single": r["margin_single"]} if single is not None else {})}
                    if fmt not in result["examples"]:
                        result["examples"][fmt] = {"text": text, "label_token_ids": lmap,
                                                   "label_tokens": {k: [tok.decode([i]) for i in v] for k, v in lmap.items()}}
                result["rows"][split].append(row)
            pos, neg = np.array(margins["pos"]), np.array(margins["neg"])
            s = {"n": len(pos), "auroc": auroc(pos, neg), "paired_acc": float((pos > neg).mean()),
                 "says_true_correct": float((pos > 0).mean()), "says_true_wrong": float((neg > 0).mean()),
                 "J_at_0": float((pos > 0).mean() - (neg > 0).mean()),
                 "mean_margin_correct": float(pos.mean()), "mean_margin_wrong": float(neg.mean()),
                 "mean_label_mass": float(np.mean(masses)),
                 "top_tokens": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:4]) for k, v in tops.items()}}
            if singles["pos"]:
                sp, sn = np.array(singles["pos"]), np.array(singles["neg"])
                s["v1_single_token"] = {"auroc": auroc(sp, sn), "paired_acc": float((sp > sn).mean()),
                                        "says_true_correct": float((sp > 0).mean()), "says_true_wrong": float((sn > 0).mean())}
            result["summary"].setdefault(split, {})[fmt] = s
            print("%-5s %-19s n=%3d AUROC %.3f paired %.3f J@0 %+.3f says True c/w %.2f/%.2f mass %.2f%s" % (
                split, fmt, s["n"], s["auroc"], s["paired_acc"], s["J_at_0"], s["says_true_correct"], s["says_true_wrong"],
                s["mean_label_mass"], ("  | v1 single-token AUROC %.3f" % s["v1_single_token"]["auroc"]) if "v1_single_token" in s else ""), flush=True)
result["elapsed_s"] = time.time() - t0
json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
print("saved", args.out, "%.0f s" % result["elapsed_s"])
