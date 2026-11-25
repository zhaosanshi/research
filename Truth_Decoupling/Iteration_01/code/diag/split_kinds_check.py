#!/usr/bin/env python3
"""CPU check of backward_by_kind (launchers/explore_20261120/split_kinds_1125.patch; main session, Claude Fable 5.2,
2026-11-25). No model, no card.

With rows_loss replaced by a small differentiable stand-in (float64), for many random questions (random numbers and
orders of cause / isolate rows):
  * the gradient that backward_by_kind leaves in w.grad equals the gradient of question_loss on all rows at once;
  * the per-row values come back in the order of the rows, and the total equals question_loss;
  * a question without a cause row, a row with an unknown kind and a non-finite loss are refused, as in the single pass.
It also checks that train_direction keeps its old behaviour by default (the keyword split_kinds defaults to False).

Second version of the patch (split_kinds2_1126.patch, 2026-11-26: both parts are padded to the width of the full batch).
When the code copy's packing functions take a width, two more things are checked: backward_by_kind hands the same
width, the longest row of ALL rows, to both parts; and pack_with_content_mask with a larger width returns the same
batch shifted to the right (token ids, attention mask and the three position masks), refuses a width below the longest
row, and is unchanged without the argument. With the first version (no width) these two checks are reported as absent.

Run with PYTHONPATH = the patched code copy:
  python3 -B Iteration_01/diag/split_kinds_check.py
Exit 0 and a line "OK ..." when every check holds; exit 1 with the first difference otherwise.
"""
import inspect
import sys

import torch

import core.qa_factorial_das as fd


WIDTHS = []


def stand_in(bundle, layer, rows, *, w=None, mode="direction", loss="jsd", positions="content", grad=False, width=None):
    WIDTHS.append(width)
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
        with_width = 0
        for _ in range(trials):
            n_cause = int(torch.randint(1, 6, (1,), generator=g))
            n_isolate = int(torch.randint(0, 6, (1,), generator=g))
            kinds = ["cause"] * n_cause + ["isolate"] * n_isolate
            order = torch.randperm(len(kinds), generator=g).tolist()
            kinds = [kinds[i] for i in order]
            a = torch.randn(len(kinds), 16, dtype=torch.float64, generator=g)
            b = torch.randn(len(kinds), dtype=torch.float64, generator=g)
            lengths = torch.randint(3, 40, (len(kinds), 2), generator=g).tolist()
            rows = [{"a": a[i], "b": b[i], "encoded": {"prompt_length": lengths[i][0], "answer_length": lengths[i][1]}}
                    for i in range(len(kinds))]
            longest = max(p + m for p, m in lengths)
            w0 = torch.randn(16, dtype=torch.float64, generator=g)
            w1 = torch.nn.Parameter(w0.clone())
            per_row1, _ = fd.rows_loss(None, 0, rows, w=w1)
            value1 = fd.question_loss(per_row1, kinds)
            value1.backward()
            w2 = torch.nn.Parameter(w0.clone())
            del WIDTHS[:]
            per_row2, value2 = fd.backward_by_kind(None, 0, rows, kinds, w2)
            if any(x is not None for x in WIDTHS):
                with_width += 1
                if any(x != longest for x in WIDTHS):
                    print("FAILED: backward_by_kind handed widths %s, the longest of all rows is %d" % (WIDTHS, longest))
                    return 1
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
        enc = {"prompt_length": 5, "answer_length": 3}
        row = {"a": torch.randn(16, dtype=torch.float64, generator=g), "b": torch.tensor(0.5, dtype=torch.float64), "encoded": enc}
        for kinds, rows in ((["isolate"], [row]), (["cause", "other"], [row, row]), (["cause"], [row, row])):
            try:
                fd.backward_by_kind(None, 0, rows, kinds, w)
            except ValueError:
                refused += 1
        bad = {"a": torch.full((16,), float("nan"), dtype=torch.float64), "b": torch.tensor(0.5, dtype=torch.float64), "encoded": enc}
        try:
            fd.backward_by_kind(None, 0, [bad], ["cause"], w)
        except RuntimeError:
            refused += 1
        if refused != 4:
            print("FAILED: %d of 4 invalid inputs were refused" % refused)
            return 1
    finally:
        fd.rows_loss = real
    if with_width not in (0, trials):
        print("FAILED: a width was handed to the parts in %d of %d questions" % (with_width, trials))
        return 1
    note = "width of the full batch handed to both parts in all questions" if with_width else "no width handed to the parts (first version of the patch)"
    packing = pack_check()
    if packing is None:
        return 1
    print("OK: %d random questions; largest differences: gradient %.3e, per-row %.3e, total %.3e; 4 invalid inputs refused; "
          "split_kinds defaults to False; %s; %s" % (trials, worst_grad, worst_row, worst_value, note, packing))
    return 0


def pack_check():
    """The packing with a given width (second version). Returns a note, or None after printing a failure."""
    pack = fd.pd.pack_with_content_mask
    if "width" not in inspect.signature(pack).parameters:
        return "packing takes no width (first version of the patch)"
    g = torch.Generator().manual_seed(7)
    for _ in range(200):
        items = []
        for _ in range(int(torch.randint(1, 6, (1,), generator=g))):
            p = int(torch.randint(2, 12, (1,), generator=g))
            m = int(torch.randint(1, 8, (1,), generator=g))
            prefix = [int(x) for x in torch.randint(1, 50, (p,), generator=g)]
            answer = [int(x) for x in torch.randint(50, 99, (m,), generator=g)]
            items.append({"prompt_length": p, "answer_length": m, "prefix_ids": prefix, "answer_ids": answer,
                          "full_ids": prefix + answer, "content_start": int(torch.randint(0, p, (1,), generator=g))})
        longest = max(len(i["full_ids"]) for i in items)
        extra = int(torch.randint(1, 9, (1,), generator=g))
        plain = pack(items, 0, "cpu")
        same = pack(items, 0, "cpu", None)
        wide = pack(items, 0, "cpu", longest + extra)
        exact = pack(items, 0, "cpu", longest)
        pairs = [("input_ids", plain["inputs"]["input_ids"], same["inputs"]["input_ids"], wide["inputs"]["input_ids"], exact["inputs"]["input_ids"]),
                 ("attention_mask", plain["inputs"]["attention_mask"], same["inputs"]["attention_mask"], wide["inputs"]["attention_mask"], exact["inputs"]["attention_mask"])]
        for key in sorted(k for k, v in plain.items() if torch.is_tensor(v)):
            pairs.append((key, plain[key], same[key], wide[key], exact[key]))
        if len(pairs) < 5:
            print("FAILED: fewer packed tensors than expected: %s" % [p[0] for p in pairs])
            return None
        for key, a, b, c, d in pairs:
            if not (torch.equal(a, b) and torch.equal(a, d)):
                print("FAILED: %s changes when width is None or the longest row" % key)
                return None
            if tuple(c.shape) != (a.shape[0], longest + extra) or not torch.equal(c[:, extra:], a) or bool(c[:, :extra].any()):
                print("FAILED: %s with a larger width is not the plain batch shifted right behind empty columns" % key)
                return None
        try:
            pack(items, 0, "cpu", longest - 1)
        except ValueError:
            continue
        print("FAILED: a width below the longest row was accepted")
        return None
    return "packing with a width: 200 random batches shifted right exactly, too small a width refused"


if __name__ == "__main__":
    sys.exit(main())
