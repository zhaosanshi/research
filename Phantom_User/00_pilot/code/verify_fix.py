#!/usr/bin/env python3
"""verify_fix.py -- after deregistering the torch._native Triton override, do cuda:1 and a 2-GPU split reproduce
the cuda:0 text at ~14k tokens, every time?  Also checks that the Triton kernel module never gets imported."""
import os
import sys

import torch

try:
    import torch._native.registry as reg
    reg.deregister_op_overrides(disable_dsl_names="triton")
    print("torch._native triton override disabled; filter:", vars(reg._filter_state), flush=True)
except Exception as e:  # noqa: BLE001
    print("no torch._native registry to disable:", type(e).__name__, e)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PATH = os.path.join(os.environ.get("MODEL_ROOT", "models"), "Qwen3-4B")
tok = AutoTokenizer.from_pretrained(PATH)
LINE = "我们讨论了项目进度、预算和下季度的招聘计划，大家同意先把核心模块做完再扩招。\n"
msgs = [{"role": "user", "content": "下面是一份很长的会议记录，请读完后用一句话总结，并说明重复了多少次。\n" + LINE * 600}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
print("prompt tokens", ids.shape[1], flush=True)


def gen(m):
    x = ids.to(next(m.parameters()).device)
    with torch.no_grad():
        out = m.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=30, do_sample=False)
    return tok.decode(out[0, x.shape[1]:], skip_special_tokens=False)


def load(**kw):
    return AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, **kw).eval()


m = load(device_map="cuda:0")
ref = gen(m)
print("cuda:0 ref:", repr(ref[:70]), flush=True)
del m
torch.cuda.empty_cache()

bad = 0
m = load(device_map="cuda:1")
for t in range(4):
    s = gen(m)
    bad += s != ref
    print(f"cuda:1 trial {t}: {'OK ' if s == ref else 'BAD'} {s[:60]!r}", flush=True)
del m
torch.cuda.empty_cache()

m = load(device_map="auto", max_memory={0: "14GiB", 1: "14GiB"})
print("split devices:", sorted(set(str(v) for v in m.hf_device_map.values())))
for t in range(4):
    s = gen(m)
    bad += s != ref
    print(f"split  trial {t}: {'OK ' if s == ref else 'BAD'} {s[:60]!r}", flush=True)
del m

print("triton kernel module imported:", "torch._native.ops.bmm_outer_product.triton_kernels" in sys.modules)
print(f"VERIFY FIX DONE: {bad} bad of 8", flush=True)
