#!/usr/bin/env python3
"""logit_lens.py -- mechanism block, probe 1: where in the network does the role token 'user' show up at the slot?

For each condition x filler seed the context is built exactly as in turn_boundary_probe.py (same filler, same chat
template, slot_start = first assistant token after the tool result).  One forward pass with hidden states; the
residual stream at the measured position is decoded after every layer with the model's own final norm +
unembedding (the logit lens).  Per layer we record P(user), rank(user), P(role class), P(eot), P(header) and the
top-3 tokens.  A base checkpoint can be run on the same rendering with --tokenizer pointing at the instruct
tokenizer (same vocabulary, chat template available).

  python logit_lens.py --model $MODEL_ROOT/Llama-3.1-8B-Instruct \
      --filler-meta results/filler_meta.jsonl --filler-coding results/filler_coding.jsonl \
      --conditions X20k_m10 X20k_m13 X20k_c10 X20k_c13 --seeds 1 2 3 --out results/lens_v1_Llama-3.1-8B-Instruct.jsonl
"""
import argparse
import json
import random
import time

import turn_boundary_probe as tbp


def find_final_norm(model):
    """The final RMSNorm of the language model (Llama/Qwen: model.model.norm; Gemma3 multimodal: ...language_model.norm)."""
    for path in ("model.norm", "model.language_model.norm", "language_model.model.norm", "model.model.norm"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise RuntimeError("final norm not found; add its path to find_final_norm()")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="tokenizer path (default: --model); use the instruct tokenizer for a base checkpoint")
    ap.add_argument("--conditions", nargs="+", default=["X20k_m10", "X20k_m13", "X20k_c10", "X20k_c13"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--filler-coding", default=None)
    ap.add_argument("--filler-meta", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--tag", default="")
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
    cats = tbp.classify_vocab(tok, eot_id, hdr_id, eom_id)
    user_id = tok.convert_tokens_to_ids("user")
    if user_id is None or user_id < 0 or user_id == tok.unk_token_id:
        user_id = tok.encode("user", add_special_tokens=False)[0]
    norm = find_final_norm(model)
    head = model.get_output_embeddings()
    role_idx = torch.tensor(cats["role"], device=device)
    print(f"model={args.model} tokenizer={args.tokenizer or args.model} eot={eot_str!r}({eot_id}) header={hdr_id} "
          f"user_id={user_id} n_role_tokens={len(cats['role'])} device={device}", flush=True)

    filler0 = {"coding": tbp.load_filler(args.filler_coding), "meta": tbp.load_filler(args.filler_meta)}
    out = open(args.out, "a", encoding="utf-8") if args.out else None

    for seed in args.seeds:
        filler = {}
        for k, (p, t) in filler0.items():
            if p and seed:
                idx = list(range(len(p)))
                random.Random(seed).shuffle(idx)
                filler[k] = ([p[i] for i in idx], [t[i] for i in idx])
            else:
                filler[k] = (p, t)
        for name in args.conditions:
            desc, kw = tbp.CONDITIONS[name]
            kind = kw.get("kind", "coding")
            kw2 = dict(kw)
            target = kw2.pop("target_tokens", 0)
            if target:
                pairs, toks = filler[kind]
                if not pairs:
                    raise SystemExit(f"{name}: no filler for kind={kind}")
                kw2["filler"] = pairs
                kw2["n_turns"] = tbp.fit_n_turns(toks, target)
            msgs, mp = tbp.build_conv(final_answer=None, **kw2)
            if mp != "slot_start":
                raise SystemExit(f"{name}: logit lens is for slot_start conditions (got {mp})")
            text = tbp.render_chat(msgs, tok, mp, eot_str)
            ids = tbp.encode(tok, text).input_ids.to(device)
            t0 = time.time()
            with torch.no_grad():
                o = model(ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, logits_to_keep=1)
                true_logits = o.logits[0, -1].float()
                hs = [h[0, -1] for h in o.hidden_states]          # residual stream at the slot, per layer
                # is the last recorded state already normed?  decide by matching the true logits
                last = hs[-1]
                d_raw = (head(last).float() - true_logits).abs().max().item()
                d_norm = (head(norm(last)).float() - true_logits).abs().max().item()
                last_is_normed = d_raw < d_norm
                layers = []
                for i, h in enumerate(hs):
                    hn = h if (i == len(hs) - 1 and last_is_normed) else norm(h)
                    lg = head(hn).float()
                    pr = torch.softmax(lg, dim=-1)
                    rank_user = int((pr > pr[user_id]).sum().item()) + 1
                    top = torch.topk(pr, 3)
                    layers.append({
                        "layer": i,
                        "p_user": float(pr[user_id]),
                        "rank_user": rank_user,
                        "p_role": float(pr[role_idx].sum()),
                        "p_eot": float(pr[eot_id]),
                        "p_hdr": float(pr[hdr_id]) if hdr_id is not None else None,
                        "entropy": float(-(pr * torch.log(pr + 1e-30)).sum()),
                        "top3": [(tok.decode([int(j)]), round(float(p), 4)) for p, j in zip(top.values, top.indices)],
                    })
                del o, hs
            torch.cuda.empty_cache()
            rec = {"condition": name, "desc": desc, "kind": kind, "kwargs": kw, "filler_seed": seed, "model": args.model,
                   "tokenizer": args.tokenizer or args.model, "tag": args.tag, "n_ctx_tokens": int(ids.shape[1]),
                   "n_states": len(layers), "last_is_normed": last_is_normed, "lens_vs_true_maxdiff": min(d_raw, d_norm),
                   "layers": layers, "seconds": round(time.time() - t0, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            final = layers[-1]
            print(f"\n== {name} seed={seed} ctx={rec['n_ctx_tokens']} states={len(layers)} last_normed={last_is_normed} "
                  f"lens-vs-true maxdiff={rec['lens_vs_true_maxdiff']:.3g} | final P(user)={final['p_user']:.2e} "
                  f"rank={final['rank_user']} P(eot)={final['p_eot']:.3f} top3={final['top3']}")
            print(f"   {'layer':>5s} {'P(user)':>9s} {'rank':>6s} {'P(role)':>9s} {'P(eot)':>8s} {'H':>5s}  top3")
            for L in layers:
                print(f"   {L['layer']:5d} {L['p_user']:9.2e} {L['rank_user']:6d} {L['p_role']:9.2e} {L['p_eot']:8.4f} "
                      f"{L['entropy']:5.2f}  {L['top3']}")
            if out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
    if out:
        out.close()


if __name__ == "__main__":
    main()
