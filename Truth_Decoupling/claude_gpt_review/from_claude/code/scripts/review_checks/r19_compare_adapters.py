#!/usr/bin/env python3
"""Compare every tensor of GPT's early E27 adapters with Claude's release-code rerun (Claude, 2026-11-07).

usage: r19_compare_adapters.py EARLY_ROOT RELEASE_ROOT
Pairs <EARLY_ROOT>/<model>/ce_alpha3_ep3/{das,md,random}/adapter.pt with <RELEASE_ROOT>/<model>/ce_alpha3_ep3/
{das_s42,md_s42,random_s42}/adapter.pt for qwen3-8b and llama3.1-8b; prints, per pair, how many tensors are bit-identical
and the largest absolute difference otherwise. CPU only; read-only.
"""
import sys
from pathlib import Path

import torch


def flat(obj, prefix=""):
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from flat(v, "%s/%s" % (prefix, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flat(v, "%s[%d]" % (prefix, i))


early_root, release_root = map(Path, sys.argv[1:3])
for model in ("qwen3-8b", "llama3.1-8b"):
    for early, release in (("das", "das_s42"), ("md", "md_s42"), ("random", "random_s42")):
        a = dict(flat(torch.load(early_root / model / "ce_alpha3_ep3" / early / "adapter.pt", map_location="cpu", weights_only=False)))
        b = dict(flat(torch.load(release_root / model / "ce_alpha3_ep3" / release / "adapter.pt", map_location="cpu", weights_only=False)))
        keys = sorted(set(a) | set(b))
        same, diffs = 0, []
        for k in keys:
            if k not in a or k not in b:
                diffs.append((k, "missing in %s" % ("early" if k not in a else "release")))
            elif a[k].shape != b[k].shape or a[k].dtype != b[k].dtype:
                diffs.append((k, "shape/dtype %s %s vs %s %s" % (tuple(a[k].shape), a[k].dtype, tuple(b[k].shape), b[k].dtype)))
            elif torch.equal(a[k], b[k]):
                same += 1
            else:
                diffs.append((k, "max |diff| %.3e" % (a[k].double() - b[k].double()).abs().max().item()))
        print("%-12s %-7s vs %-11s tensors %d, bit-identical %d%s" % (model, early, release, len(keys), same,
              "" if not diffs else "; differ: " + "; ".join("%s (%s)" % d for d in diffs[:6])))
