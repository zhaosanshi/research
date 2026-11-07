#!/usr/bin/env python3
"""Check the saved random initial vectors of GPT's early E27 adapters (Claude, 2026-11-07).

usage: r19_random_init_check.py EARLY_ROOT
For <EARLY_ROOT>/<model>/<ce_alpha1_ep3|ce_alpha3_ep3>/random/adapter.pt: recompute
normalize(randn(hidden, Generator(cpu).manual_seed(42 + layer))) exactly as core/qa_generation_adapter.py does and compare it
bit for bit with initial_W; also print the cosine of initial_W with the DAS-init adapter's initial_W (the original DAS direction)
and with the trained random W. CPU only; read-only.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

root = Path(sys.argv[1])
for model in ("llama3.1-8b", "qwen3-8b"):
    for run in ("ce_alpha1_ep3", "ce_alpha3_ep3"):
        rnd = torch.load(root / model / run / "random" / "adapter.pt", map_location="cpu", weights_only=False)
        das = torch.load(root / model / run / "das" / "adapter.pt", map_location="cpu", weights_only=False)
        for key, init in rnd["initial_W"].items():
            layer = int(key)
            g = torch.Generator(device="cpu").manual_seed(42 + layer)
            expect = F.normalize(torch.randn(init.numel(), generator=g), dim=0)
            cos = lambda a, b: float(F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))
            print("%-12s %-14s L%-3d alpha %-4s exact %-5s max|diff| %.1e | cos(init, DAS) %+.5f | cos(init, trained) %+.4f"
                  % (model, run, layer, rnd.get("alpha"), torch.equal(init, expect), (init - expect).abs().max().item(),
                     cos(init, das["initial_W"][key]), cos(init, rnd["W"][key])))
