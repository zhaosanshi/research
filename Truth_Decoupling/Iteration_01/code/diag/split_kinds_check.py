#!/usr/bin/env python3
"""CPU check of backward_by_kind (launchers/explore_20261120/split_kinds_1125.patch; main session, Claude Fable 5.2,
2026-11-25). No model, no card.

With rows_loss replaced by a small differentiable stand-in (float64), for many random questions (random numbers and
orders of cause / isolate rows):
  * the gradient that backward_by_kind leaves in w.grad equals the gradient of question_loss on all rows at once;
  * the per-row values come back in the order of the rows, and the total equals question_loss;
  * a question without a cause row, a row with an unknown kind and a non-finite loss are refused, as in the single pass.
It also checks that train_direction keeps its old behaviour by default (the keyword split_kinds defaults to False).

Run with PYTHONPATH = the patched code copy:
  python3 -B Iteration_01/diag/split_kinds_check.py
Exit 0 and a line "OK ..." when every check holds; exit 1 with the first difference otherwise.
"""
import inspect
import sys

import torch

import core.qa_factorial_das as fd


def stand_in(bundle, layer, rows, *, w=None, mode="direction", loss="jsd", positions="content", grad=False):
    a = torch.stack([r["a"] for r in rows])
    b = torch.stack([r["b"] for r in rows])
    z = a @ w
    return torch.tanh(z) ** 2 + b * z, None


def main():
    if not hasattr(fd, "backward_by_kind"):
        print("FAILED: this code copy has no backward_by_kind (not the patched copy)")
        return 1
    default = inspect.signature(fd.train_direction).parameters["split_kinds"].default
    if default is not False:
        print("FAILED: train_direction's split_kinds must default to False, found %r" % (default,))
        return 1
    real = fd.rows_loss
    fd.rows_loss = stand_in
    try:
        g = torch.Generator().manual_seed(20261125)
        worst_grad = worst_row = worst_value = 0.0
        trials = 500
        for _ in range(trials):
            n_cause = int(torch.randint(1, 6, (1,), generator=g))
            n_isolate = int(torch.randint(0, 6, (1,), generator=g))
            kinds = ["cause"] * n_cause + ["isolate"] * n_isolate
            order = torch.randperm(len(kinds), generator=g).tolist()
            kinds = [kinds[i] for i in order]
            a = torch.randn(len(kinds), 16, dtype=torch.float64, generator=g)
            b = torch.randn(len(kinds), dtype=torch.float64, generator=g)
            rows = [{"a": a[i], "b": b[i]} for i in range(len(kinds))]
            w0 = torch.randn(16, dtype=torch.float64, generator=g)
            w1 = torch.nn.Parameter(w0.clone())
            per_row1, _ = fd.rows_loss(None, 0, rows, w=w1)
            value1 = fd.question_loss(per_row1, kinds)
            value1.backward()
            w2 = torch.nn.Parameter(w0.clone())
            per_row2, value2 = fd.backward_by_kind(None, 0, rows, kinds, w2)
            if per_row2.requires_grad or value2.requires_grad:
                print("FAILED: backward_by_kind must return detached values")
                return 1
            worst_grad = max(worst_grad, float((w1.grad - w2.grad).abs().max()))
            worst_row = max(worst_row, float((per_row1.detach() - per_row2).abs().max()))
            worst_value = max(worst_value, abs(float(value1.detach()) - float(value2)))
        if max(worst_grad, worst_row, worst_value) > 1e-10:
            print("FAILED: largest differences: gradient %.3e, per-row %.3e, total %.3e" % (worst_grad, worst_row, worst_value))
            return 1
        refused = 0
        w = torch.nn.Parameter(torch.randn(16, dtype=torch.float64, generator=g))
        row = {"a": torch.randn(16, dtype=torch.float64, generator=g), "b": torch.tensor(0.5, dtype=torch.float64)}
        for kinds, rows in ((["isolate"], [row]), (["cause", "other"], [row, row]), (["cause"], [row, row])):
            try:
                fd.backward_by_kind(None, 0, rows, kinds, w)
            except ValueError:
                refused += 1
        bad = {"a": torch.full((16,), float("nan"), dtype=torch.float64), "b": torch.tensor(0.5, dtype=torch.float64)}
        try:
            fd.backward_by_kind(None, 0, [bad], ["cause"], w)
        except RuntimeError:
            refused += 1
        if refused != 4:
            print("FAILED: %d of 4 invalid inputs were refused" % refused)
            return 1
    finally:
        fd.rows_loss = real
    print("OK: %d random questions; largest differences: gradient %.3e, per-row %.3e, total %.3e; 4 invalid inputs refused; "
          "split_kinds defaults to False" % (trials, worst_grad, worst_row, worst_value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
