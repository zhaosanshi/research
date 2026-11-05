#!/usr/bin/env python3
"""Unit checks for train_das(loss="ce") -- the v1-style gold-label objective in the statement protocol (Claude, 2026-11-05).

usage: chk_das_ce.py CODE_DIR [OUT_JSON]
CPU only, a 2-block random toy model (the fake bundle of chk_das_loader.py). Asked for by the red team of the ce change:
  * no teacher tokens are generated (teacher_tokens patched to raise);
  * the epoch-0 loss of a W that cannot change anything equals the hand-computed -log sum_v p(v) of the gold label
    (a rank-1 W orthogonal to every source-base difference is emulated by identical base and source prompts);
  * W receives a nonzero gradient and the optimiser moves it when the swap can matter;
  * forward (True) and reverse (False) targets are the ones intended, and swapping the label sets changes the loss;
  * overlapping label sets, missing label_ids and positions="all" are refused;
  * T is recorded as 1 and label_ids are stored; the KL default is unchanged (still asks for teacher tokens).
Exit 1 on any failed expectation.
"""
import json
import math
import sys
import types
from unittest.mock import patch

import torch
from torch import nn

CODE = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, CODE)
import core.directions as d  # noqa: E402

fails, rep = [], {}


def expect(name, ok, detail=None):
    rep[name] = {"ok": bool(ok), "detail": detail}
    print(("ok   " if ok else "FAIL ") + name, "" if detail is None else detail)
    if not ok:
        fails.append(name)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(4, 4, bias=False)

    def forward(self, h):
        n = torch.arange(1, h.shape[1] + 1).view(1, -1, 1)
        return h + torch.tanh(self.mix(h.cumsum(1) / n))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 4)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.head = nn.Linear(4, 17, bias=False)

    def forward(self, input_ids, **kw):
        h = self.embedding(input_ids)
        for block in self.blocks:
            h = block(h)
        return types.SimpleNamespace(logits=self.head(h))


def enc(ids):
    x = torch.tensor([ids])
    return {"input_ids": x, "attention_mask": torch.ones_like(x)}


torch.manual_seed(7)
model = Tiny().eval()
model.requires_grad_(False)
bundle = types.SimpleNamespace(model=model, device=torch.device("cpu"), hidden_size=4, layer=lambda i: model.blocks[i])
LAB = {"True": [9, 10], "False": [11, 12]}


def no_teacher(*a, **k):
    raise AssertionError("teacher_tokens must not be called for loss='ce'")


def gold_logp(inputs, ids):
    with torch.no_grad():
        lp = torch.log_softmax(model(inputs["input_ids"]).logits[0, -1].float(), -1)
    return float(torch.logsumexp(lp[ids], 0))


# 1. identical base/source: the swap cannot change anything, so the epoch-0 loss is the plain gold-label CE of the prompt.
same = [{"base": enc([1, 2, 3]), "source": enc([1, 2, 3])}]
with patch.object(d, "teacher_tokens", no_teacher):
    r = d.train_das(bundle, same, 0, epochs=1, log_fn=None, anchor="prompt", loss="ce", bidirectional=True, label_ids=LAB)
want = 0.5 * (-gold_logp(same[0]["base"], LAB["True"]) - gold_logp(same[0]["source"], LAB["False"]))
expect("no_teacher_tokens_and_epoch0_loss_is_handcomputed_gold_ce", abs(r["losses"][0] - want) < 1e-5,
       {"loss": r["losses"][0], "hand": want})
expect("T_recorded_1_and_label_ids_stored", r["T"] == 1 and r["label_ids"] == {"True": [9, 10], "False": [11, 12]} and r["loss_kind"] == "ce",
       {"T": r["T"], "label_ids": r["label_ids"]})

# 2. different prompts: W gets a gradient and moves; forward-only loss equals the True-target CE under the swap at init.
pairs = [{"base": enc([1, 2, 3]), "source": enc([4, 5, 6, 3])}, {"base": enc([2, 1, 3]), "source": enc([6, 5, 4, 3])}]
basis = d.OrthoBasis(4, 1, seed=42)
l_fwd, n = d.gold_label_ce_loss(bundle, basis, pairs[0], 0, LAB["True"])
l_fwd.backward()
g = basis.raw.grad
expect("W_receives_nonzero_gradient", g is not None and float(g.norm()) > 0, {"grad_norm": None if g is None else float(g.norm()), "steps": n})
with patch.object(d, "teacher_tokens", no_teacher):
    a = d.train_das(bundle, pairs, 0, epochs=3, log_fn=None, anchor="prompt", loss="ce", bidirectional=True, label_ids=LAB)
    b = d.train_das(bundle, pairs, 0, epochs=3, log_fn=None, anchor="prompt", loss="ce", bidirectional=True,
                    label_ids={"True": LAB["False"], "False": LAB["True"]})
w0 = d.OrthoBasis(4, 1, seed=42).detached()
expect("W_moves_from_init", float((a["W"] - w0).norm()) > 1e-6, {"moved": float((a["W"] - w0).norm())})
expect("swapping_label_sets_changes_the_loss", abs(a["losses"][0] - b["losses"][0]) > 1e-6, {"normal": a["losses"][0], "swapped": b["losses"][0]})
expect("loss_decreases_over_epochs", a["losses"][-1] < a["losses"][0], {"losses": a["losses"]})
expect("position_trace_reads_and_writes_last_prompt_tokens",
       a["position_trace"] and a["position_trace"][0]["read_src_pos"] == 3 and a["position_trace"][0]["write_base_pos"] == 2,
       {"trace": a["position_trace"][:1]})

# 3. refusals
for name, kw in (("missing_label_ids_refused", dict(label_ids=None)),
                 ("overlapping_label_sets_refused", dict(label_ids={"True": [9, 10], "False": [10, 11]})),
                 ("positions_all_refused", dict(label_ids=LAB, positions="all"))):
    try:
        with patch.object(d, "teacher_tokens", no_teacher):
            d.train_das(bundle, pairs, 0, epochs=1, log_fn=None, anchor="prompt", loss="ce", **kw)
        expect(name, False, "accepted")
    except ValueError as e:
        expect(name, True, str(e)[:80])

# 4. the KL default still asks for teacher tokens (unchanged path)
asked = []


def fake_teacher(bundle_, inputs, T):
    asked.append(T)
    return torch.arange(T) + 8


with patch.object(d, "teacher_tokens", fake_teacher):
    k = d.train_das(bundle, [{"base": enc([1, 2, 3]), "source": enc([4, 5, 6, 3])}], 0, T=2, epochs=1, log_fn=None, anchor="prompt")
expect("kl_default_unchanged_uses_teacher_and_no_label_ids", asked == [2] and k["loss_kind"] == "kl" and "label_ids" not in k and k["T"] == 2,
       {"teacher_requests": asked, "keys": sorted(k)})
print("failures:", fails)
if OUT:
    json.dump({"checks": rep, "failures": fails}, open(OUT, "w"), indent=1)
sys.exit(1 if fails else 0)
