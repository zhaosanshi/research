#!/usr/bin/env python3
"""Gradient agreement of the two ways of back-propagating a training question, on the real model, at a fixed direction
(main session, Claude Fable 5.2, 2026-11-25; plan discussion/work_queue/2026-11-24_queue_m.md, progress entry of
2026-11-25 23:5x). It runs the patched trainer unchanged and only listens.

  record OUT.pt -- <arguments of scripts/e44_factorial_das.py, e.g. train --source-run ... --layers 31 --epochs 1 --lr 1e-12>
      Runs scripts/e44_factorial_das.py of the code copy on PYTHONPATH with these arguments. Every optimizer step hands
      the direction's gradient to torch.nn.utils.clip_grad_norm_; a wrapper around that function stores a copy of the
      gradient of each step. With a learning rate of 1e-12 the direction does not move (the update is far below
      float32 resolution), so every step's gradient is taken at the same direction: the seeded random start. The
      gradients are saved as one float32 tensor [steps, hidden] in OUT.pt, also when the trainer ends with an error.
  compare A.pt B.pt [LABEL]
      Step by step (the same question in both files: the order comes from the same seed): cosine of the two gradients
      and the ratio of their norms; then the cosine of the two sums over all steps. Prints numbers only.

record needs the card of the trainer; compare runs on the CPU.
"""
import os
import runpy
import sys

import torch


def record(out, argv):
    code = os.environ.get("PYTHONPATH", "").split(":")[0]
    script = os.path.join(code, "scripts", "e44_factorial_das.py")
    if not os.path.isfile(script):
        print("no scripts/e44_factorial_das.py under PYTHONPATH %r" % code)
        return 2
    grads = []
    original = torch.nn.utils.clip_grad_norm_

    def listening(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        if len(parameters) != 1 or parameters[0].grad is None:
            raise RuntimeError("expected the one direction parameter with a gradient")
        grads.append(parameters[0].grad.detach().float().cpu().clone())
        return original(parameters, max_norm, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = listening
    sys.argv = [script] + list(argv)
    rc = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        torch.nn.utils.clip_grad_norm_ = original
        if grads:
            torch.save(torch.stack(grads), out)
        print("recorded %d step gradients -> %s" % (len(grads), out))
    return rc


def compare(path_a, path_b, label):
    a = torch.load(path_a, map_location="cpu").double()
    b = torch.load(path_b, map_location="cpu").double()
    if a.shape != b.shape or a.ndim != 2:
        print("FAILED: shapes differ or are not [steps, hidden]: %s vs %s" % (tuple(a.shape), tuple(b.shape)))
        return 1
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        print("FAILED: non-finite gradients")
        return 1
    na, nb = a.norm(dim=1), b.norm(dim=1)
    cos = (a * b).sum(1) / (na * nb)
    ratio = nb / na
    rel = (a - b).norm(dim=1) / na
    q = lambda t, p: float(torch.quantile(t, p))
    total = float((a.sum(0) * b.sum(0)).sum() / (a.sum(0).norm() * b.sum(0).norm()))
    print("%s: steps %d, hidden %d" % (label, a.shape[0], a.shape[1]))
    print("  cosine per step: min %.6f, 5%% %.6f, median %.6f, mean %.6f, max %.6f; steps below 0.99: %d, below 0.9: %d" % (
        float(cos.min()), q(cos, 0.05), q(cos, 0.5), float(cos.mean()), float(cos.max()),
        int((cos < 0.99).sum()), int((cos < 0.9).sum())))
    print("  norm ratio second / first per step: min %.6f, median %.6f, max %.6f; mean norm first %.6f, second %.6f, ratio of means %.6f" % (
        float(ratio.min()), q(ratio, 0.5), float(ratio.max()), float(na.mean()), float(nb.mean()), float(nb.mean() / na.mean())))
    print("  relative difference |a - b| / |a| per step: median %.6f, max %.6f" % (q(rel, 0.5), float(rel.max())))
    print("  cosine of the two sums over all steps: %.6f" % total)
    return 0


def main():
    if len(sys.argv) >= 5 and sys.argv[1] == "record" and sys.argv[3] == "--":
        return record(sys.argv[2], sys.argv[4:])
    if len(sys.argv) in (4, 5) and sys.argv[1] == "compare":
        return compare(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) == 5 else "comparison")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
