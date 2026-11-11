#!/usr/bin/env python3
"""Pre-flight probe for a newly added model config (Claude, 2026-11-12). Reads data, steers nothing that is kept.

Checks, in order (any failure exits non-zero):
  1 load through core.model_io.load_model; class, layer count, hidden size agree with the model config
  2 weights really loaded: several tensors compared bit-for-bit with the checkpoint's safetensors shards
  3 chat prompt: build_chat equals the tokenizer's own template call with the config's template kwargs; thinking closed
  4 true/false readout: label token ids, top-5 next tokens and label mass on a few calibration statements
  5 left-padded batch == one-by-one forward (last-token block output at sampled layers, label log-probs)
  6 hooks: zero direction reproduces the baseline; an amplify write changes the logits and is applied once per forward
  7 gradient through the stack from a vector added at a middle layer (what DAS training needs)
  8 greedy generation on two questions (sanity) and timing

usage: newmodel_probe.py --model qwen3.5-9b [--n 6] [--out probe.json]
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load_split, tf_prompt                         # noqa: E402
from core.config import load_model_config, load_prompt_config     # noqa: E402
from core.intervene import SteerSpec, next_token_logits, steer    # noqa: E402
from core.model_io import capture_last_token, chat_template_kwargs, forward_logits, load_model  # noqa: E402
from core.readout import build_label_token_map, label_logprobs_from_logits  # noqa: E402


def fail(msg):
    print("PROBE FAILED: " + msg, flush=True)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=6, help="calibration statements used by checks 4-5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", default=None, help="override the config dtype (float32: is a batch-vs-single gap numerical or a padding leak?)")
    ap.add_argument("--only-batch-check", action="store_true", help="run checks 1, 3 and 5 only (also: same-length batch without padding)")
    args = ap.parse_args()
    rep = {"model": args.model}
    cfg = load_model_config(args.model)
    if args.dtype:
        cfg["dtype"] = args.dtype
        rep["dtype_override"] = args.dtype
    prompts = load_prompt_config("default_v1")
    import transformers
    rep["versions"] = {"transformers": transformers.__version__, "torch": torch.__version__}
    print("versions", rep["versions"], flush=True)

    # 1 load
    t0 = time.time()
    bundle = load_model(cfg)
    rep["load_s"] = round(time.time() - t0, 1)
    rep["class"] = type(bundle.model).__name__
    rep["num_layers"], rep["hidden"] = bundle.num_layers, bundle.hidden_size
    print("1 loaded %s in %.0f s: %d layers, hidden %d" % (rep["class"], rep["load_s"], bundle.num_layers, bundle.hidden_size), flush=True)
    if bundle.num_layers != cfg["num_layers"] or bundle.hidden_size != cfg["hidden_size"]:
        fail("config says %d layers / hidden %d" % (cfg["num_layers"], cfg["hidden_size"]))
    if len(bundle.layers()) != bundle.num_layers:
        fail("layers_attr %s gives %d modules" % (cfg["layers_attr"], len(bundle.layers())))
    types = [getattr(bundle.layer(i), "layer_type", getattr(bundle.layer(i), "block_type", "?")) for i in range(bundle.num_layers)]
    rep["layer_types"] = types
    rep["attn_modules"] = [type(bundle.attn(i)).__name__ for i in range(bundle.num_layers)]
    print("  layer types:", "".join("L" if t == "linear_attention" else ("F" if t == "full_attention" else "?") for t in types), flush=True)

    # 2 weights against the checkpoint
    from safetensors import safe_open
    if args.only_batch_check:
        idx = {}
    else:
        idx = None
    L = bundle.num_layers
    if idx is None:
        idx = json.load(open(glob.glob(os.path.join(cfg["path"], "*.safetensors.index.json"))[0]))["weight_map"]
    sd = dict(bundle.model.named_parameters())
    picks = ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight",
             "model.layers.0.linear_attn.in_proj_qkv.weight", "model.layers.3.self_attn.q_proj.weight",
             "model.layers.%d.mlp.down_proj.weight" % (L - 1), "model.layers.%d.input_layernorm.weight" % (L // 2)]
    checked = []
    for name in ([] if args.only_batch_check else picks):
        if name not in sd:
            fail("parameter %s not in the loaded model" % name)
        cands = [k for k in (name, name.replace("model.", "model.language_model.", 1)) if k in idx]
        if not cands:
            fail("parameter %s not found in the checkpoint index" % name)
        key = cands[0]
        with safe_open(os.path.join(cfg["path"], idx[key]), framework="pt") as f:
            ref = f.get_tensor(key)
        got = sd[name].detach().cpu()
        if tuple(ref.shape) != tuple(got.shape) or not torch.equal(ref.to(got.dtype), got):
            fail("parameter %s differs from checkpoint key %s" % (name, key))
        checked.append([name, key])
    rep["weights_checked"] = checked
    print("2 %d tensors equal to the checkpoint" % len(checked), flush=True)

    # 3 chat prompt
    st = prompts["statement"]
    recs = load_split("truthfulqa_tf", "calib")[: args.n]
    text0 = tf_prompt(bundle, prompts, recs[0]["text"])
    msgs = [{"role": "system", "content": st["system"]}, {"role": "user", "content": st["user_tf"].format(statement=recs[0]["text"])}]
    own = bundle.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs(cfg))
    own += cfg["chat"].get("thinking_stub", "")
    if own != text0:
        fail("build_chat differs from the template call")
    if chat_template_kwargs(cfg).get("enable_thinking") is False and not text0.endswith("<think>\n\n</think>\n\n"):
        fail("thinking is not closed at the end of the prompt: %r" % text0[-60:])
    rep["prompt_example"] = text0
    print("3 prompt ok; tail %r" % text0[-90:], flush=True)

    # 4 readout
    lab_map = build_label_token_map(bundle.tokenizer, text0, st.get("label_variants"))
    rep["label_map"] = {k: [[i, bundle.tokenizer.decode([i])] for i in v] for k, v in lab_map.items()}
    print("4 label tokens", rep["label_map"], flush=True)
    texts = [tf_prompt(bundle, prompts, r["text"]) for r in recs]
    rows = []
    for r, t in ([] if args.only_batch_check else zip(recs, texts)):
        lg = next_token_logits(bundle, bundle.encode([t]))[0]
        top = torch.topk(torch.log_softmax(lg, -1), 5)
        lp = label_logprobs_from_logits(lg, lab_map, use_abstain=False)
        mass = float(torch.exp(torch.tensor([lp["True"], lp["False"]])).sum())
        rows.append({"id": r["id"], "gold": r["label"], "lpT": lp["True"], "lpF": lp["False"], "mass": mass,
                     "top5": [[bundle.tokenizer.decode([int(i)]), round(float(v), 3)] for v, i in zip(top.values, top.indices)]})
        print("  %s gold %-5s lpT %7.3f lpF %7.3f mass %.3f top5 %s" % (r["id"], r["label"], lp["True"], lp["False"], mass, rows[-1]["top5"]), flush=True)
    rep["readout"] = rows
    if rows and min(x["mass"] for x in rows) < 0.5:
        print("  WARNING: True/False mass below 0.5 on some statement", flush=True)

    # 5 batch vs single
    layers = sorted(set([0, 1, 2, 3, L // 4, L // 2, (3 * L) // 4, L - 1]))
    aB, lgB = capture_last_token(bundle, texts, layers, components=("resid",), batch_size=len(texts), return_logits=True)
    a1, lg1 = capture_last_token(bundle, texts, layers, components=("resid",), batch_size=1, return_logits=True)
    worst = {}
    for l in layers:
        x, y = aB[("resid", l)], a1[("resid", l)]
        worst[l] = float(((x - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-6)).max())
    lp_diff = 0.0
    for i in range(len(texts)):
        pB = label_logprobs_from_logits(lgB[i], lab_map, use_abstain=False)
        p1 = label_logprobs_from_logits(lg1[i], lab_map, use_abstain=False)
        lp_diff = max(lp_diff, abs((pB["True"] - pB["False"]) - (p1["True"] - p1["False"])))
    lens = [len(bundle.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    rep["batch_vs_single"] = {"prompt_lengths": lens, "max_rel_diff_by_layer": worst, "max_margin_diff": lp_diff}
    print("5 batch vs single: lengths %s; max relative diff by layer %s; max True-False margin diff %.4f"
          % (lens, {k: round(v, 4) for k, v in worst.items()}, lp_diff), flush=True)
    # same-length statements batched WITHOUT padding: the pure batch-shape part of the gap
    allrecs = load_split("truthfulqa_tf", "calib") + load_split("truthfulqa_tf", "dev")
    by_len = {}
    for r in allrecs:
        t = tf_prompt(bundle, prompts, r["text"])
        by_len.setdefault(len(bundle.tokenizer(t, add_special_tokens=False)["input_ids"]), []).append(t)
    same = max(by_len.values(), key=len)[:6]
    sB, _ = capture_last_token(bundle, same, layers, components=("resid",), batch_size=len(same))
    s1, _ = capture_last_token(bundle, same, layers, components=("resid",), batch_size=1)
    worst_np = {l: float(((sB[("resid", l)] - s1[("resid", l)]).norm(dim=1) / s1[("resid", l)].norm(dim=1).clamp_min(1e-6)).max())
                for l in layers}
    rep["batch_vs_single_same_length_no_padding"] = {"n": len(same), "max_rel_diff_by_layer": worst_np}
    print("5b same length, no padding (n=%d): max relative diff by layer %s" % (len(same), {k: round(v, 4) for k, v in worst_np.items()}), flush=True)
    if args.only_batch_check:
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rep, f, indent=1, ensure_ascii=False)
        print("PROBE OK (batch check only)", flush=True)
        return
    if max(worst.values()) > 0.05 or lp_diff > 0.25:
        fail("left-padded batch does not reproduce one-by-one forwards")

    # 6 hooks
    inp = bundle.encode(texts[:2])
    base = next_token_logits(bundle, inp)
    zero = SteerSpec(torch.zeros(bundle.hidden_size), L // 2, mode="add", alpha=1.0)
    z = next_token_logits(bundle, inp, [zero])
    if not torch.equal(base, z):
        fail("zero direction changed the logits")
    u = torch.randn(bundle.hidden_size, generator=torch.Generator().manual_seed(0))
    u = u / u.norm()
    amp = SteerSpec(u, L // 2, mode="amplify", alpha=50.0)
    with steer(bundle, [amp]) as hooks:
        lg = forward_logits(bundle, inp)[:, -1, :].float().cpu()
    d = float((lg - base).abs().max())
    rep["hooks"] = {"zero_equal": True, "amplify_max_logit_change": d, "calls": hooks[L // 2].call_count, "applied": hooks[L // 2].applied_count}
    print("6 hooks: zero direction equal; amplify changes logits by up to %.3f; calls %d applied %d"
          % (d, hooks[L // 2].call_count, hooks[L // 2].applied_count), flush=True)
    if d <= 0 or hooks[L // 2].call_count != 1 or hooks[L // 2].applied_count != 1:
        fail("steering hook not applied as expected")

    # 7 gradient through the stack
    v = torch.zeros(bundle.hidden_size, device=bundle.device, requires_grad=True)

    def add_v(module, inp_, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + v.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hd = bundle.layer(L // 2).register_forward_hook(add_v)
    try:
        lg = forward_logits(bundle, bundle.encode(texts[:2]), grad=True)[:, -1, :].float()
        tid = lab_map["True"][0]
        loss = -torch.log_softmax(lg, -1)[:, tid].mean()
        loss.backward()
    finally:
        hd.remove()
    gn = float(v.grad.norm()) if v.grad is not None else float("nan")
    rep["grad_norm"] = gn
    print("7 gradient norm at layer %d: %.4g" % (L // 2, gn), flush=True)
    if not (gn > 0 and gn < float("inf")):
        fail("no finite gradient through the stack")

    # 8 generation + timing
    from core.model_io import greedy_generate
    qs = ["What is the capital of France?", "What happens if you crack your knuckles a lot?"]
    qa = prompts["qa"]
    gin = bundle.encode([bundle.build_chat(qa.get("system", ""), qa["user"].format(question=q)) for q in qs])
    t1 = time.time()
    outs, _ = greedy_generate(bundle, gin, 48)
    rep["generation"] = {"questions": qs, "outputs": outs, "seconds": round(time.time() - t1, 1)}
    for q, o in zip(qs, outs):
        print("8 Q: %s\n  A: %r" % (q, o), flush=True)
    t2 = time.time()
    capture_last_token(bundle, texts * 4, [L // 2], batch_size=8)
    rep["seconds_per_statement_forward"] = round((time.time() - t2) / (4 * len(texts)), 4)
    rep["max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    print("timing: %.3f s per statement (batch 8); peak memory %.1f GB" % (rep["seconds_per_statement_forward"], rep["max_memory_gb"]), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
    print("PROBE OK", flush=True)


if __name__ == "__main__":
    main()
