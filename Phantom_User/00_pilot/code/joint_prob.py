#!/usr/bin/env python3
"""joint_prob.py -- exact probability of a whole role WORD (all its tokens) at the slot, not just its first token.

For multi-token role words (pinyin YONGHU = 'Y','ONG','HU'; '[User]' = '[','User',']'; Benutzer = 'Ben','utzer' ...)
the first-token probability counts every other word that starts the same way.  Here the word is teacher-forced at
the slot and P(word) = prod_i P(token_i | context, token_<i).  The contaminant in the model's own earlier answers uses
the same word (as CONTAM_ROLE_WORD does), and a greedy continuation after the word is recorded for A/B classification.

  PROBE_LANG=en CONTAM_STYLE=plain python joint_prob.py --model $MODEL_ROOT/Qwen3-8B --no-think \
      --filler-coding results/filler_en_coding.jsonl --filler-meta results/filler_en_general.jsonl \
      --cells m00 m03 --seeds 1 2 3 --words "[User]" YONGHU Operator U Prompt Me --out results/joint_Qwen3-8B.jsonl
"""
import argparse
import json
import math
import random
import time

import turn_boundary_probe as tbp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--filler-coding", required=True)
    ap.add_argument("--filler-meta", required=True)
    ap.add_argument("--cells", nargs="+", default=["m00", "m03"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--words", nargs="+", required=True)
    ap.add_argument("--target-tokens", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.no_think:
        tbp.CHAT_KWARGS["enable_thinking"] = False
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = tbp.load_model(args.model, torch.bfloat16, args.device)
    device = next(model.parameters()).device
    eot_str, eot_id, hdr_str, hdr_id, eom_id = tbp.detect_special(tok)
    base_contam = tbp.FAKE_USER_TURN_TEXT
    assert base_contam.startswith("\n\nuser\n"), "run with CONTAM_ROLE_WORD unset"
    rest = base_contam[len("\n\nuser"):]
    fill = {}
    for kind, path in (("coding", args.filler_coding), ("meta", args.filler_meta)):
        fill[kind] = tbp.load_filler(path)
    out = open(args.out, "a", encoding="utf-8")
    for cell in args.cells:
        kind = {"m": "meta", "c": "coding"}[cell[0]]
        contam_k = int(cell[2])
        for s in args.seeds:
            pairs, toks = fill[kind]
            pairs, toks = list(pairs), list(toks)
            if s:
                idx = list(range(len(pairs)))
                random.Random(s).shuffle(idx)
                pairs = [pairs[i] for i in idx]
                toks = [toks[i] for i in idx]
            n_turns = tbp.fit_n_turns(toks, args.target_tokens)
            for w in args.words:
                t0 = time.time()
                tbp.FAKE_USER_TURN_TEXT = "\n\n" + w + rest          # contaminant carries the same word
                msgs, mp = tbp.build_conv(kind=kind, n_turns=n_turns, wrapper=False, contam_k=contam_k,
                                          thin_input=True, pre_tool_signoff=True, filler=pairs)
                text = tbp.render_chat(msgs, tok, mp, eot_str)
                ids = tbp.encode(tok, text).input_ids.to(device)
                wids = tok.encode(w, add_special_tokens=False)
                full = torch.cat([ids, torch.tensor([wids], device=device)], dim=1)
                with torch.no_grad():
                    logits = model(full, attention_mask=torch.ones_like(full)).logits[0].float()
                lp = torch.log_softmax(logits, dim=-1)
                n0 = ids.shape[1]
                tok_lps = [float(lp[n0 - 1 + i, wids[i]]) for i in range(len(wids))]
                with torch.no_grad():
                    gen = model.generate(full, attention_mask=torch.ones_like(full), max_new_tokens=args.max_new,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
                cont = w + tok.decode(gen[0, full.shape[1]:], skip_special_tokens=False)
                rec = {"condition": f"X20k_{cell}", "cell": cell, "kind": kind, "contam_k": contam_k, "filler_seed": s,
                       "word": w, "tokens": [tok.decode([t]) for t in wids], "n_tokens": len(wids),
                       "p_first": math.exp(tok_lps[0]), "p_joint": math.exp(sum(tok_lps)), "token_logps": tok_lps,
                       "continuation": cont, "n_ctx_tokens": int(n0), "model": args.model, "seconds": round(time.time() - t0, 1)}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{cell} s{s} {w!r:16s} toks={rec['tokens']} P(first)={rec['p_first']:.1e} P(joint)={rec['p_joint']:.1e} "
                      f"cont={cont[:80]!r} ({rec['seconds']}s)", flush=True)
                del logits, lp
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
