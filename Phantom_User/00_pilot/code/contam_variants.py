#!/usr/bin/env python3
"""contam_variants.py -- which part of the fake turn in the model's OWN earlier answers opens the gate?

Hypothesis from head_attrib.py: the promoting heads behave like induction heads -- at the slot the current token is
the '\\n\\n' that ends the assistant header, and they attend to the 'user' tokens that followed '\\n\\n' inside the
model's earlier answers (the fake turns start with "\\n\\nuser\\n").  If so, the bigram ('\\n\\n' -> 'user') is what
matters: changing the token before 'user', or the role word itself, should move P(user) much more than removing
the <USER_REQUEST> wrapper does.

Replaces FAKE_USER_TURN_TEXT with variants and measures the slot distribution on X20k_m13 contexts (3 seeds).

  python contam_variants.py --model $MODEL_ROOT/Llama-3.1-8B-Instruct --filler-meta results/filler_meta.jsonl \
      --filler-coding results/filler_coding.jsonl --seeds 1 2 3 --out results/contam_variants_Llama-3.1-8B-Instruct.jsonl
"""
import argparse
import json
import math
import random
import statistics
import time

import turn_boundary_probe as tbp

WRAP = ("<USER_REQUEST>\n你又这样了：自己编了一个我的提问，然后自己回答了。\n</USER_REQUEST>\n"
        "<ADDITIONAL_METADATA>\nThe current local time is: 2026-10-19T15:20:00+08:00.\n</ADDITIONAL_METADATA>")
PLAIN = "你又这样了：自己编了一个我的提问，然后自己回答了。"
VARIANTS = {
    "orig: \\n\\n user wrap":      ("\n\nuser\n" + WRAP, "user"),
    "\\n user wrap":               ("\nuser\n" + WRAP, "user"),
    "\\n\\n\\n user wrap":         ("\n\n\nuser\n" + WRAP, "user"),
    "\\n\\n _user wrap":           ("\n\n user\n" + WRAP, " user"),
    "\\n\\n User wrap":            ("\n\nUser\n" + WRAP, "User"),
    "\\n\\n 用户 wrap":            ("\n\n用户\n" + WRAP, "用户"),
    "\\n\\n assistant wrap":       ("\n\nassistant\n" + WRAP, "assistant"),
    "\\n\\n user plain":           ("\n\nuser\n" + PLAIN, "user"),
    "\\n\\n wrap only":            ("\n\n" + WRAP, None),
    "none (clean)":                ("", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--condition", default="X20k_m13")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--filler-coding", default=None)
    ap.add_argument("--filler-meta", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.no_think:
        tbp.CHAT_KWARGS["enable_thinking"] = False
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = tbp.load_model(args.model, torch.bfloat16, args.device)
    device = next(model.parameters()).device
    eot_str, eot_id, hdr_str, hdr_id, eom_id = tbp.detect_special(tok)
    user_id = tok.convert_tokens_to_ids("user")
    if user_id is None or user_id < 0 or user_id == tok.unk_token_id:
        user_id = tok.encode("user", add_special_tokens=False)[0]

    def first_id(s):
        enc = tok.encode(s, add_special_tokens=False)
        return enc[0] if enc else None

    filler0 = {"coding": tbp.load_filler(args.filler_coding), "meta": tbp.load_filler(args.filler_meta)}
    desc, kw = tbp.CONDITIONS[args.condition]
    kind = kw.get("kind", "coding")
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    results = {}
    print(f"model={args.model} condition={args.condition} user_id={user_id}", flush=True)
    for vname, (vtext, role_word) in VARIANTS.items():
        tbp.FAKE_USER_TURN_TEXT = vtext
        role_id = first_id(role_word) if role_word else None
        vtoks = [tok.decode([i]) for i in tok.encode(vtext[:16], add_special_tokens=False)] if vtext else []
        pus, prs, tops = [], [], []
        for seed in args.seeds:
            pairs, toks = filler0[kind]
            if seed:
                idx = list(range(len(pairs)))
                random.Random(seed).shuffle(idx)
                pairs, toks = [pairs[i] for i in idx], [toks[i] for i in idx]
            kw2 = dict(kw)
            target = kw2.pop("target_tokens", 0)
            kw2["filler"] = pairs
            kw2["n_turns"] = tbp.fit_n_turns(toks, target)
            if not vtext:
                kw2["contam_k"] = 0
            msgs, mp = tbp.build_conv(final_answer=None, **kw2)
            text = tbp.render_chat(msgs, tok, mp, eot_str)
            ids = tbp.encode(tok, text).input_ids.to(device)
            id_list = ids[0].tolist()
            n_free_user = sum(1 for p in range(1, len(id_list)) if id_list[p] == user_id and id_list[p - 1] != hdr_id)
            n_role = sum(1 for p in range(1, len(id_list)) if role_id is not None and id_list[p] == role_id and id_list[p - 1] != hdr_id)
            t0 = time.time()
            with torch.no_grad():
                lg = model(ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[0, -1].float()
            pr = torch.softmax(lg, dim=-1)
            top = torch.topk(pr, 5)
            rec = {"variant": vname, "variant_text_head": vtext[:24], "variant_tokens": vtoks, "role_word": role_word,
                   "role_id": role_id, "seed": seed, "condition": args.condition, "model": args.model,
                   "n_ctx_tokens": len(id_list), "n_free_user_tokens": n_free_user, "n_free_role_tokens": n_role,
                   "p_user": float(pr[user_id]), "p_role": float(pr[role_id]) if role_id is not None else None,
                   "p_eot": float(pr[eot_id]),
                   "top5": [(tok.decode([int(j)]), round(float(p), 5)) for p, j in zip(top.values, top.indices)],
                   "seconds": round(time.time() - t0, 1)}
            if out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            pus.append(rec["p_user"])
            prs.append(rec["p_role"] if rec["p_role"] is not None else float("nan"))
            tops.append(rec["top5"][0][0])
        gm = lambda xs: 10 ** statistics.mean(math.log10(max(x, 1e-30)) for x in xs)
        results[vname] = (gm(pus), gm(prs) if role_word else None, n_free_user, n_role, vtoks, tops)
        print(f"  {vname:28s} tokens={vtoks!r:40s} free 'user'={n_free_user} role={n_role}  "
              f"P(user)={gm(pus):.2e}  P(role word)={(gm(prs) if role_word else float('nan')):.2e}  top1={tops}", flush=True)
    print("\nsummary (geometric mean over seeds):")
    for vname, (pu, prw, nfu, nr, vt, tops) in results.items():
        print(f"  {vname:28s} P(user)={pu:.2e}" + (f"  P({VARIANTS[vname][1]!r})={prw:.2e}" if prw is not None else ""))
    if out:
        out.close()


if __name__ == "__main__":
    main()
