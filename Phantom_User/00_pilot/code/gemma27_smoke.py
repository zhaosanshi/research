#!/usr/bin/env python3
"""gemma27_smoke.py -- is gemma-3-27b-it coherent under our loader?  Greedy answers to a short and a ~10k-token
prompt, with sdpa and eager attention, on one GPU.  Run inside the docker image (CUDA_VISIBLE_DEVICES=<one gpu>)."""
import os
import sys
import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

PATH = os.path.join(os.environ.get("MODEL_ROOT", "models"), "gemma-3-27b-it")
tok = AutoTokenizer.from_pretrained(PATH)

short = [{"role": "user", "content": "用三句话介绍一下你自己，然后把这句话翻译成英文：今天北京下雨了，我们把会议改到了明天上午十点。"}]
long_user = ("下面是一份会议记录，请读完后用一句话总结。\n"
             + "我们讨论了项目进度、预算和下季度的招聘计划，大家同意先把核心模块做完再扩招。\n" * 300)
long = [{"role": "user", "content": long_user}]


def gen(model, msgs, max_new):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False)
    return ids.shape[1], tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)


for impl in (sys.argv[1:] or ["sdpa", "eager"]):
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                        attn_implementation=impl).eval()
    print(f"\n===== {impl}: loaded in {time.time() - t0:.0f}s, mem {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    n, s = gen(model, short, 120)
    print(f"[short, {n} tok] {s!r}", flush=True)
    n, s = gen(model, long, 80)
    print(f"[long, {n} tok] {s!r}", flush=True)
    del model
    torch.cuda.empty_cache()
print("SMOKE DONE", flush=True)
