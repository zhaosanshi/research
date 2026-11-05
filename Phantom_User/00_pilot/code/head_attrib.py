#!/usr/bin/env python3
"""head_attrib.py -- mechanism block, probe 2: which attention heads push 'user' at the slot, and where do they look?

Same contexts as turn_boundary_probe / logit_lens (chat template, slot_start).  One forward pass with hooks:
  * direct logit attribution (DLA): the final residual r = embed + sum_l (attn_l + mlp_l); with RMSNorm the logit of
    token t is (W_U[t] * gamma) . r / rms(r), linear in r, so every component gets an exact share of logit(user)
    and of the margin logit(user) - logit(top1).  Attention is split per head through o_proj's column blocks.
  * attention pattern of every head from the slot query, recomputed from q/k (RoPE applied) and verified against the
    head outputs the model actually produced; mass aggregated over spans: BOS, system prompt, earlier real user
    headers, non-header 'user' tokens (the fake turns' role word), the fake-turn spans, the tool result, the current
    assistant header, everything else.
  * ablation: zero the top-k heads (by DLA(user)) at the slot position only, re-run, read P(user); random-k control.

  python head_attrib.py --model $MODEL_ROOT/Llama-3.1-8B-Instruct --filler-meta ... --filler-coding ... \
      --conditions X20k_m10 X20k_m13 X20k_c10 X20k_c13 --seeds 1 2 3 --out results/heads_v1_Llama-3.1-8B-Instruct.jsonl
"""
import argparse
import json
import random
import time

import turn_boundary_probe as tbp


def rotate_half(x):
    import torch
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def find_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise RuntimeError("decoder layers not found")


def find_module(model, names):
    for path in names:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise RuntimeError(f"none of {names} found")


def spans_from_ids(ids, user_id, asst_id, header_id, eot_id):
    """Token-position sets for the attention aggregation.  ids: python list."""
    T = len(ids)
    pos = {k: set() for k in ("bos", "sys", "hdr_user_prev", "tool_hdr_user", "tool_body", "fake_user_tok",
                              "fake_span", "asst_hdr", "cur_hdr", "other")}
    pos["bos"].add(0)
    try:
        first_eot = ids.index(eot_id)
    except ValueError:
        first_eot = 0
    pos["sys"].update(range(1, first_eot + 1))
    # user role words: header ones vs free-text ones
    hdr_users = [p for p in range(1, T) if ids[p] == user_id and header_id is not None and ids[p - 1] == header_id]
    free_users = [p for p in range(1, T) if ids[p] == user_id and not (header_id is not None and ids[p - 1] == header_id)]
    if header_id is None:  # no header token in this template: treat the first 'user' after an eot as header
        hdr_users = [p for p in range(1, T) if ids[p] == user_id and ids[p - 1] == eot_id]
        free_users = [p for p in range(1, T) if ids[p] == user_id and ids[p - 1] != eot_id]
    last_eot = max(p for p in range(T) if ids[p] == eot_id)
    pos["cur_hdr"].update(range(last_eot + 1, T))
    if hdr_users:
        tool_hdr = hdr_users[-1]
        pos["tool_hdr_user"].add(tool_hdr)
        pos["tool_body"].update(range(tool_hdr + 1, last_eot + 1))
        pos["hdr_user_prev"].update(hdr_users[:-1])
    pos["fake_user_tok"].update(free_users)
    for p in free_users:  # from the free 'user' word to the end of that message
        q = p
        while q < T and ids[q] != eot_id:
            q += 1
        pos["fake_span"].update(range(p, q))
    pos["asst_hdr"].update(p for p in range(1, T) if ids[p] == asst_id and header_id is not None and ids[p - 1] == header_id)
    taken = set()
    order = ("bos", "sys", "cur_hdr", "tool_hdr_user", "tool_body", "fake_user_tok", "fake_span", "hdr_user_prev", "asst_hdr")
    for k in order:   # make the spans disjoint in priority order (fake_user_tok before fake_span, etc.)
        pos[k] -= taken
        taken |= pos[k]
    pos["other"] = set(range(T)) - taken
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--conditions", nargs="+", default=["X20k_m10", "X20k_m13", "X20k_c10", "X20k_c13"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--filler-coding", default=None)
    ap.add_argument("--filler-meta", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--ablate-top", nargs="*", type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument("--top-print", type=int, default=12)
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
    user_id = tok.convert_tokens_to_ids("user")
    if user_id is None or user_id < 0 or user_id == tok.unk_token_id:
        user_id = tok.encode("user", add_special_tokens=False)[0]
    asst_id = tok.convert_tokens_to_ids("assistant")
    if asst_id is None or asst_id < 0:
        asst_id = tok.encode("assistant", add_special_tokens=False)[0]
    layers = find_layers(model)
    norm = find_module(model, ["model.norm", "model.language_model.norm", "language_model.model.norm"])
    embed = model.get_input_embeddings()
    W_U = model.get_output_embeddings().weight          # [V, d]
    n_layers = len(layers)
    attn0 = layers[0].self_attn
    n_heads = attn0.q_proj.out_features // attn0.head_dim
    n_kv = attn0.k_proj.out_features // attn0.head_dim
    hd = attn0.head_dim
    n_rep = n_heads // n_kv
    print(f"model={args.model} layers={n_layers} heads={n_heads} kv={n_kv} head_dim={hd} eot={eot_str!r}({eot_id}) "
          f"header={hdr_id} user_id={user_id} asst_id={asst_id} device={device}", flush=True)

    # ---------------- hooks ----------------
    cap = {}

    def embed_hook(m, i, o):
        cap["embed"] = o[0, -1].detach().float()

    def make_oproj_pre(l):
        def h(m, args_):
            cap.setdefault("attn_in", {})[l] = args_[0][0, -1].detach().float()
        return h

    def make_mlp_hook(l):
        def h(m, i, o):
            cap.setdefault("mlp_out", {})[l] = o[0, -1].detach().float()
        return h

    def norm_pre(m, args_):
        cap["final_resid"] = args_[0][0, -1].detach().float()

    def make_attn_pre(l):
        def h(mod, args_, kwargs):
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args_[0]
            cos, sin = kwargs["position_embeddings"]
            T = hs.shape[1]
            q = mod.q_proj(hs[:, -1:, :]).view(1, 1, n_heads, hd)
            k = mod.k_proj(hs).view(1, T, n_kv, hd)
            v = mod.v_proj(hs).view(1, T, n_kv, hd)
            if hasattr(mod, "q_norm"):
                q = mod.q_norm(q)
                k = mod.k_norm(k)
            q = q.transpose(1, 2)            # [1, nH, 1, hd]
            k = k.transpose(1, 2)            # [1, nKV, T, hd]
            v = v.transpose(1, 2)
            c_last, s_last = cos[:, -1:].unsqueeze(1), sin[:, -1:].unsqueeze(1)
            q = q * c_last + rotate_half(q) * s_last
            k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
            qf = q[0, :, 0, :].float()                       # [nH, hd]
            kf = k[0].float()                                # [nKV, T, hd]
            kf = kf.repeat_interleave(n_rep, dim=0)          # [nH, T, hd]
            scores = torch.einsum("hd,htd->ht", qf, kf) * mod.scaling
            a = torch.softmax(scores, dim=-1)                # [nH, T]
            vf = v[0].float().repeat_interleave(n_rep, dim=0)  # [nH, T, hd]
            o = torch.einsum("ht,htd->hd", a, vf)            # recomputed head outputs
            cap.setdefault("attn_w", {})[l] = a.detach()
            cap.setdefault("attn_o_recomp", {})[l] = o.detach()
        return h

    handles = [embed.register_forward_hook(embed_hook), norm.register_forward_pre_hook(norm_pre)]
    for l, layer in enumerate(layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_oproj_pre(l)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))
        handles.append(layer.self_attn.register_forward_pre_hook(make_attn_pre(l), with_kwargs=True))

    filler0 = {"coding": tbp.load_filler(args.filler_coding), "meta": tbp.load_filler(args.filler_meta)}
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    span_keys = ("bos", "sys", "hdr_user_prev", "tool_hdr_user", "tool_body", "fake_user_tok", "fake_span", "asst_hdr", "cur_hdr", "other")

    for seed in args.seeds:
        filler = {}
        for kname, (p, t) in filler0.items():
            if p and seed:
                idx = list(range(len(p)))
                random.Random(seed).shuffle(idx)
                filler[kname] = ([p[i] for i in idx], [t[i] for i in idx])
            else:
                filler[kname] = (p, t)
        for name in args.conditions:
            desc, kw = tbp.CONDITIONS[name]
            kind = kw.get("kind", "coding")
            kw2 = dict(kw)
            target = kw2.pop("target_tokens", 0)
            if target:
                pairs, toks = filler[kind]
                kw2["filler"] = pairs
                kw2["n_turns"] = tbp.fit_n_turns(toks, target)
            msgs, mp = tbp.build_conv(final_answer=None, **kw2)
            assert mp == "slot_start"
            text = tbp.render_chat(msgs, tok, mp, eot_str)
            ids = tbp.encode(tok, text).input_ids.to(device)
            id_list = ids[0].tolist()
            T = len(id_list)
            spans = spans_from_ids(id_list, user_id, asst_id, hdr_id, eot_id)
            t0 = time.time()
            cap.clear()
            with torch.no_grad():
                logits = model(ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            top1 = int(torch.argmax(logits))
            p_user, p_top1 = float(probs[user_id]), float(probs[top1])

            # ---- DLA ----
            r = cap["final_resid"]
            eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
            rms = torch.sqrt(r.pow(2).mean() + eps)
            gamma = norm.weight.detach().float()
            def u_vec(t):
                return (W_U[t].detach().float() * gamma) / rms
            u_user, u_top1 = u_vec(user_id), u_vec(top1)
            u_diff = u_user - u_top1
            logit_user_check = float(u_user @ r)
            comps_sum = cap["embed"].clone()
            per_layer = []
            heads = []
            for l, layer in enumerate(layers):
                x = cap["attn_in"][l]                                   # [nH*hd]
                W_O = layer.self_attn.o_proj.weight.detach().float()    # [d, nH*hd]
                hc = torch.stack([W_O[:, h * hd:(h + 1) * hd] @ x[h * hd:(h + 1) * hd] for h in range(n_heads)])  # [nH, d]
                mlp = cap["mlp_out"][l]
                comps_sum += hc.sum(0) + mlp
                dla_u = hc @ u_user
                dla_d = hc @ u_diff
                a = cap["attn_w"][l]                                    # [nH, T]
                # verification of the recomputed attention against the real head outputs (o_proj input slices)
                o_real = x.view(n_heads, hd)
                o_rec = cap["attn_o_recomp"][l]
                rel_err = float((o_rec - o_real).norm() / (o_real.norm() + 1e-9))
                masses = {k: a[:, sorted(spans[k])].sum(-1) if spans[k] else torch.zeros(n_heads, device=device) for k in span_keys}
                topk = torch.topk(a, 5, dim=-1)
                per_layer.append({"layer": l, "attn_dla_user": float(dla_u.sum()), "mlp_dla_user": float(mlp @ u_user),
                                  "attn_dla_diff": float(dla_d.sum()), "mlp_dla_diff": float(mlp @ u_diff),
                                  "attn_recomp_rel_err": rel_err,
                                  "mean_mass_fake_user_tok": float(masses["fake_user_tok"].mean())})
                for h in range(n_heads):
                    heads.append({"layer": l, "head": h, "dla_user": float(dla_u[h]), "dla_diff": float(dla_d[h]),
                                  "mass": {k: float(masses[k][h]) for k in span_keys},
                                  "top5": [(int(j), tok.decode([id_list[int(j)]]), round(float(w), 4))
                                           for w, j in zip(topk.values[h], topk.indices[h])]})
            decomp_err = float((comps_sum - r).abs().max())
            embed_dla = float(cap["embed"] @ u_user)
            heads_sorted = sorted(heads, key=lambda d: -d["dla_user"])
            heads_sorted_diff = sorted(heads, key=lambda d: -d["dla_diff"])

            # ---- ablations (slot position only) ----
            def run_ablated(head_list):
                slices = [(d["layer"], d["head"]) for d in head_list]
                by_layer = {}
                for l, h in slices:
                    by_layer.setdefault(l, []).append(h)
                hs_ = []
                for l, hlist in by_layer.items():
                    def pre(m, args_, _hl=hlist):
                        xx = args_[0].clone()
                        for h in _hl:
                            xx[:, -1, h * hd:(h + 1) * hd] = 0
                        return (xx,)
                    hs_.append(layers[l].self_attn.o_proj.register_forward_pre_hook(pre))
                with torch.no_grad():
                    lg = model(ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[0, -1].float()
                for hh in hs_:
                    hh.remove()
                pr = torch.softmax(lg, dim=-1)
                return {"p_user": float(pr[user_id]), "p_top1": float(pr[top1]), "logit_user": float(lg[user_id]),
                        "new_top1": tok.decode([int(torch.argmax(lg))])}
            ablations = []
            rng = random.Random(1000 + seed)
            late = [d for d in heads if d["layer"] >= n_layers * 2 // 3]
            for k in args.ablate_top:
                ablations.append({"kind": "top_dla_user", "k": k, "heads": [(d["layer"], d["head"]) for d in heads_sorted[:k]],
                                  **run_ablated(heads_sorted[:k])})
                rnd = rng.sample(late, k)
                ablations.append({"kind": "random_late", "k": k, "heads": [(d["layer"], d["head"]) for d in rnd], **run_ablated(rnd)})
            ablations.append({"kind": "top_dla_diff", "k": 10, "heads": [(d["layer"], d["head"]) for d in heads_sorted_diff[:10]],
                              **run_ablated(heads_sorted_diff[:10])})

            rec = {"condition": name, "desc": desc, "kind": kind, "kwargs": kw, "filler_seed": seed, "model": args.model,
                   "tokenizer": args.tokenizer or args.model, "tag": args.tag, "n_ctx_tokens": T,
                   "n_spans": {k: len(spans[k]) for k in span_keys},
                   "p_user": p_user, "p_top1": p_top1, "top1": tok.decode([top1]), "logit_user": float(logits[user_id]),
                   "logit_top1": float(logits[top1]), "logit_user_from_resid": logit_user_check,
                   "decomp_max_abs_err": decomp_err, "embed_dla_user": embed_dla,
                   "per_layer": per_layer, "heads": heads, "ablations": ablations,
                   "seconds": round(time.time() - t0, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if out:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()

            # ---- report ----
            print(f"\n==== {name} seed={seed} ctx={T} P(user)={p_user:.2e} top1={rec['top1']!r} P={p_top1:.3f} "
                  f"logit(user)={rec['logit_user']:.3f} (from resid {logit_user_check:.3f}) decomp_err={decomp_err:.2e} "
                  f"spans={rec['n_spans']} max attn recomp rel err={max(pl['attn_recomp_rel_err'] for pl in per_layer):.3f}")
            print(f"   embed DLA(user)={embed_dla:+.3f} | per layer: attn / mlp DLA(user), attn / mlp DLA(user-top1), mean head mass on fake 'user' tokens")
            for pl in per_layer:
                if pl["layer"] >= n_layers // 2:
                    print(f"   L{pl['layer']:2d}  attn {pl['attn_dla_user']:+7.3f}  mlp {pl['mlp_dla_user']:+7.3f} | "
                          f"attn {pl['attn_dla_diff']:+7.3f}  mlp {pl['mlp_dla_diff']:+7.3f} | fake-tok mass {pl['mean_mass_fake_user_tok']:.4f}")
            print(f"   top {args.top_print} heads by DLA(user):  L.H   dla_user  dla_diff | mass: fakeTok fakeSpan hdrUserPrev toolHdr toolBody curHdr bos sys other | top attended")
            for d in heads_sorted[:args.top_print]:
                m = d["mass"]
                print(f"   L{d['layer']:2d}.H{d['head']:<2d} {d['dla_user']:+8.3f} {d['dla_diff']:+8.3f} | "
                      f"{m['fake_user_tok']:.3f} {m['fake_span']:.3f} {m['hdr_user_prev']:.3f} {m['tool_hdr_user']:.3f} "
                      f"{m['tool_body']:.3f} {m['cur_hdr']:.3f} {m['bos']:.3f} {m['sys']:.3f} {m['other']:.3f} | "
                      + " ".join(f"{t!r}@{j}:{w}" for j, t, w in d["top5"][:3]))
            print("   ablations (slot position only):")
            for ab in ablations:
                print(f"     {ab['kind']:13s} k={ab['k']:2d}  P(user) {p_user:.2e} -> {ab['p_user']:.2e}  "
                      f"P(top1) {p_top1:.3f} -> {ab['p_top1']:.3f}  new top1={ab['new_top1']!r}")
            print(flush=True)
    for hh in handles:
        hh.remove()
    if out:
        out.close()


if __name__ == "__main__":
    main()
