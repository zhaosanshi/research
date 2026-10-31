"""Small CPU check of the submitted DAS anchor fix; never loads model weights.

Uses the real directions.py and model_io.py with a tiny frozen causal network.
The uploaded snapshot omits io_utils.py: only its set_seed utility is supplied
here, in a private test package. This is not a full experiment-driver test.
Run: python3 claude_gpt_review/from_gpt/verify_20261101_03.py
"""
import importlib
import json
from pathlib import Path
import random
import sys
import types
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


CODE = Path(__file__).resolve().parent.parent / "from_claude" / "code"
package = types.ModuleType("_das_anchor_review")
package.__path__ = [str(CODE / "core")]
sys.modules[package.__name__] = package
directions = importlib.import_module(package.__name__ + ".directions")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


io = types.ModuleType(package.__name__ + ".io_utils")
io.set_seed = set_seed
sys.modules[io.__name__] = io


class CausalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(4, 4, bias=False)

    def forward(self, h):
        counts = torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
        return h + torch.tanh(self.mix(h.cumsum(1) / counts))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 4)
        self.blocks = nn.ModuleList([CausalBlock(), CausalBlock()])
        self.head = nn.Linear(4, 17, bias=False)

    def forward(self, input_ids, **kwargs):
        h = self.embedding(input_ids)
        for block in self.blocks:
            h = block(h)
        return types.SimpleNamespace(logits=self.head(h))


def encoded(ids):
    x = torch.tensor([ids], dtype=torch.long)
    return {"input_ids": x, "attention_mask": torch.ones_like(x)}


def main():
    set_seed(7)
    model = TinyModel().eval()
    model.requires_grad_(False)
    bundle = types.SimpleNamespace(model=model, device=torch.device("cpu"), hidden_size=4,
                                   layer=lambda i: model.blocks[i])
    pair = {"base": encoded([1, 2, 3]), "source": encoded([4, 5, 6, 3])}
    # Fixed teacher prefix tests the intervention path; it is not task data.
    teacher = torch.tensor([8, 9, 10])
    original_factory = directions._interchange_hook_factory
    report = {}

    for anchor, expected in (("prompt", [2, 2, 2]), ("current", [2, 3, 4])):
        observed = []

        def recording_factory(basis_fn, holder, positions="last", write_pos=-1):
            original = original_factory(basis_fn, holder, positions, write_pos)

            def hook(module, inp, out):
                result = original(module, inp, out)
                changed = ((result - out).abs().amax(dim=-1)[0] > 1e-7).nonzero().flatten().tolist()
                observed.append({"changed": changed, "source": holder["h"].detach().clone()})
                return result
            return hook

        basis = directions.OrthoBasis(4, 1, seed=42)
        trace = []
        with patch.object(directions, "_interchange_hook_factory", recording_factory):
            loss, steps = directions.multistep_kl_loss(bundle, basis, pair, 0, teacher, 3,
                                                      anchor=anchor, trace=trace)
        assert steps == 3
        assert [x["write_base_pos"] for x in trace] == expected
        assert [x["read_src_pos"] for x in trace] == [i + 1 for i in expected]
        assert [x["changed"] for x in observed] == [[i] for i in expected]
        if anchor == "prompt":
            for item in observed[1:]:
                torch.testing.assert_close(item["source"], observed[0]["source"])
        loss.backward()
        assert torch.isfinite(basis.raw.grad).all() and basis.raw.grad.norm() > 0
        # Verify the actual gradient numerically, beyond merely logging indices.
        flat_index = int(basis.raw.grad.abs().argmax())
        analytical = float(basis.raw.grad.flatten()[flat_index])
        epsilon = 1e-3
        values = []
        for sign in (1, -1):
            candidate = directions.OrthoBasis(4, 1, seed=42)
            with torch.no_grad():
                candidate.raw.flatten()[flat_index] += sign * epsilon
            value, _ = directions.multistep_kl_loss(bundle, candidate, pair, 0, teacher, 3, anchor=anchor)
            values.append(float(value.detach()))
        numerical = (values[0] - values[1]) / (2 * epsilon)
        assert abs(analytical - numerical) < 2e-4, (analytical, numerical)
        assert len(model.blocks[0]._forward_hooks) == 0
        report[anchor] = {"trace": trace, "actual_changed_positions": [x["changed"] for x in observed],
                          "loss": float(loss.detach()), "gradient": analytical, "finite_difference": numerical}

    # The first prediction has identical positions under the old and new paths.
    first = []
    for anchor in ("prompt", "current"):
        basis = directions.OrthoBasis(4, 1, seed=42)
        loss, _ = directions.multistep_kl_loss(bundle, basis, pair, 0, teacher, 1, anchor=anchor)
        first.append(loss)
    torch.testing.assert_close(first[0], first[1])

    # Isolate a later prediction: changing the fixed prompt position must really
    # affect downstream logits, not merely an index printed by the training loop.
    model_io = importlib.import_module(package.__name__ + ".model_io")
    base_later = directions._append_prefix(pair["base"], teacher[:1])
    source_h = model.blocks[0](model.embedding(pair["source"]["input_ids"]))[:, -1, :]
    before = model_io.forward_logits(bundle, base_later)[:, -1, :]
    basis = directions.OrthoBasis(4, 1, seed=42)
    handle = model.blocks[0].register_forward_hook(original_factory(basis, {"h": source_h}, write_pos=2))
    try:
        after = model_io.forward_logits(bundle, base_later, grad=True)[:, -1, :]
    finally:
        handle.remove()
    later_change = float((after - before).abs().max().detach())
    assert later_change > 1e-7
    report["later_prediction_max_logit_change"] = later_change

    # Exercise the optimizer and metadata path on two synthetic pairs.
    # Pre-cached teacher tokens avoid pretending to validate real-model generation.
    pairs = [dict(pair, y_src=teacher),
             {"base": encoded([2, 1, 3]), "source": encoded([6, 5, 4, 3]), "y_src": teacher}]
    frozen = [p.detach().clone() for p in model.parameters()]
    result = directions.train_das(bundle, pairs, 0, T=3, epochs=2, seed=42,
                                  anchor="prompt", log_fn=None)
    initial = directions.OrthoBasis(4, 1, seed=42).detached()
    assert not torch.allclose(result["W"], initial)
    assert torch.isfinite(result["W"]).all()
    torch.testing.assert_close(result["W"].norm(), torch.tensor(1.0))
    assert result["anchor"] == "prompt" and result["n_pairs"] == 2
    assert len(result["position_trace"]) == 3
    for before, after in zip(frozen, model.parameters()):
        torch.testing.assert_close(before, after)
        assert after.grad is None
    report["optimizer"] = {"losses": result["losses"], "direction_updated": True, "model_unchanged": True}
    report["scope"] = "CPU synthetic causal network only; no Llama/Qwen/Gemma run or full driver verification"
    print(json.dumps(report, indent=2))
    print("PASS: fixed read/write positions, actual edited tensors, source invariance, gradients, old mode, optimizer")


if __name__ == "__main__":
    main()
