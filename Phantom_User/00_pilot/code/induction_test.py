#!/usr/bin/env python3
"""induction_test.py -- induction and previous-token scores for every attention head.

Random token sequence of length L repeated once (BOS + x_1..x_L + x_1..x_L).  For a position i in the repeat whose
token x_i also occurred at i-L, an induction head attends to i-L+1 (the token that followed the earlier occurrence);
a previous-token head attends to i-1.  Scores are the mean attention mass on those positions over the repeat, over
several random sequences.  Attention is recomputed from q/k with RoPE (same code path as head_attrib.py).

  python induction_test.py --model $MODEL_ROOT/Llama-3.1-8B-Instruct --out results/induction_Llama-3.1-8B-Instruct.json
"""
import argparse
import json

import turn_boundary_probe as tbp
from head_attrib import find_layers, rotate_half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--length", type=int, default=60)
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = tbp.load_model(args.model, torch.bfloat16, args.device)
    device = next(model.parameters()).device
    layers = find_layers(model)
    attn0 = layers[0].self_attn
    hd = attn0.head_dim
    n_heads = attn0.q_proj.out_features // hd
    n_kv = attn0.k_proj.out_features // hd
    n_rep = n_heads // n_kv
    L = args.length
    cap = {}

    def make_pre(l):
        def h(mod, args_, kwargs):
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args_[0]
            cos, sin = kwargs["position_embeddings"]
            T = hs.shape[1]
            q = mod.q_proj(hs).view(1, T, n_heads, hd)
            k = mod.k_proj(hs).view(1, T, n_kv, hd)
            if hasattr(mod, "q_norm"):
                q = mod.q_norm(q)
                k = mod.k_norm(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
            k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
            qf = q[0].float()                                   # [nH, T, hd]
            kf = k[0].float().repeat_interleave(n_rep, dim=0)   # [nH, T, hd]
            scores = torch.einsum("hid,hjd->hij", qf, kf) * mod.scaling
            mask = torch.full((T, T), float("-inf"), device=scores.device).triu(1)
            a = torch.softmax(scores + mask, dim=-1)            # [nH, T, T]
            cap[l] = a.detach()
        return h

    handles = [layer.self_attn.register_forward_pre_hook(make_pre(l), with_kwargs=True) for l, layer in enumerate(layers)]
    g = torch.Generator().manual_seed(0)
    bos = tok.bos_token_id
    ind = torch.zeros(len(layers), n_heads, device=device)
    prev = torch.zeros(len(layers), n_heads, device=device)
    for s in range(args.n_seq):
        x = torch.randint(1000, min(len(tok), 100000), (L,), generator=g)
        seq = torch.cat([torch.tensor([bos]) if bos is not None else torch.tensor([], dtype=torch.long), x, x]).unsqueeze(0).to(device)
        off = 1 if bos is not None else 0
        cap.clear()
        with torch.no_grad():
            model(seq, attention_mask=torch.ones_like(seq), logits_to_keep=1)
        for l in range(len(layers)):
            a = cap[l]
            idx_i = torch.arange(off + L, off + 2 * L - 1, device=device)   # positions in the repeat (not the last)
            ind[l] += a[:, idx_i, idx_i - L + 1].mean(-1)
            prev[l] += a[:, idx_i, idx_i - 1].mean(-1)
    ind /= args.n_seq
    prev /= args.n_seq
    for hh in handles:
        hh.remove()
    scores = {f"L{l}.H{h}": {"induction": round(float(ind[l, h]), 4), "prev_token": round(float(prev[l, h]), 4)}
              for l in range(len(layers)) for h in range(n_heads)}
    top = sorted(scores.items(), key=lambda kv: -kv[1]["induction"])[:20]
    print(f"model={args.model} L={L} n_seq={args.n_seq}  top induction heads:")
    for k, v in top:
        print(f"  {k:8s} induction={v['induction']:.3f} prev_token={v['prev_token']:.3f}")
    if args.out:
        json.dump({"model": args.model, "length": L, "n_seq": args.n_seq, "scores": scores}, open(args.out, "w"), indent=0)
        print("saved", args.out)


if __name__ == "__main__":
    main()
