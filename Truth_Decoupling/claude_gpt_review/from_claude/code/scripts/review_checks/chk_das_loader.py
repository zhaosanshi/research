"""DAS loader / teacher-cache checks for review round 14 (D1-D5). CPU only; tiny vectors, a 4-d frozen toy network,
temporary checkpoints written with the production file-name helper.

usage: chk_das_loader.py CODE_DIR OUT.json [OLD_directions.py]
  CODE_DIR/core/directions.py is the fixed version. OLD_directions.py (optional) is the previous version, loaded
  standalone, to show that the single-seed path is unchanged and what the multi-seed cases used to return.
Exit status 1 if any expectation fails.
"""
import importlib.util
import itertools
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

CODE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(CODE))
import core.directions as d  # noqa: E402


def load_standalone(path):
    spec = importlib.util.spec_from_file_location("old_directions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = load_standalone(sys.argv[3]) if len(sys.argv) > 3 else None
rep, fails = {}, []


def expect(name, cond, detail):
    rep[name] = {"ok": bool(cond), "detail": detail}
    if not cond:
        fails.append(name)


def ckpt(folder, seed, W, gaps, rank=1, name_rank=None, **extra):
    torch.save({"W": torch.tensor(W, dtype=torch.float32), "mean_gap": gaps, "losses": [0.1], "seed": seed, "rank": rank,
                "n_pairs": 96, **extra}, Path(folder) / d.das_file_name("E", 0, name_rank or rank, seed))


def load(mod, files, **kw):
    """files: list of dicts for ckpt(). Returns (w, meta) or ("ValueError", message). Test the first element with is_err():
    an ndarray compared to a string with != gives an element-wise array, not a bool (red-team finding)."""
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            ckpt(td, **f)
        try:
            return mod.load_das_direction(td, "E", 0, **kw)
        except ValueError as e:
            return ("ValueError", str(e))


def is_err(res):
    return res is not None and isinstance(res[0], str)


def both(files, **kw):
    new = load(d, files, **kw)
    o = load(old, files, **kw) if old else None
    return new, o


def summary(res):
    if res is None:
        return None
    if is_err(res):
        return {"error": res[1][:120]}
    w, meta = res
    return {"W": [round(float(x), 6) for x in w], "norm": round(float(np.linalg.norm(w)), 6), "mean_gap": round(float(meta["mean_gap"]), 6),
            "n_seeds": meta["n_seeds"]}


def f32(v):
    """The checkpoint stores W as float32; the loader renormalises the float32-rounded vector in float64. Expected values are
    computed from the same rounded vector so the comparison is exact to ~1e-15 (red-team finding on the D3 tolerance)."""
    v = np.asarray(v, dtype=np.float32).astype(np.float64)
    return v / np.linalg.norm(v)


# D1: identical subspace, opposite signs, external near-orthogonal reference -> must not cancel to zero
new, o = both([dict(seed=42, W=[[1.0], [0.0]], gaps=[1.0]), dict(seed=43, W=[[-1.0], [0.0]], gaps=[-1.0])], align_to=[0.0, 1.0])
expect("D1_identical_subspaces_do_not_cancel", not is_err(new) and abs(abs(new[0][0]) - 1) < 1e-9 and abs(new[1]["mean_gap"] - 1) < 1e-9,
       {"new": summary(new), "old": summary(o)})

# D1: nearly identical seeds, external near-orthogonal reference -> merged direction stays with the seeds
eps = 0.001
a = np.array([1.0, eps]); a /= np.linalg.norm(a)
c = np.array([1.0, -eps]); c /= np.linalg.norm(c)
new, o = both([dict(seed=42, W=a[:, None].tolist(), gaps=[1.0]), dict(seed=43, W=c[:, None].tolist(), gaps=[1.0])], align_to=[0.0, 1.0])
expect("D1_nearly_identical_seeds_stay", not is_err(new) and abs(float(a @ new[0])) > 0.999,
       {"seed_cos": float(a @ c), "new": summary(new), "new_cos_with_seed42": None if is_err(new) else float(abs(a @ new[0])),
        "old": summary(o), "old_cos_with_seed42": None if (o is None or is_err(o)) else float(abs(a @ o[0]))})

# D1: disagreeing seeds are refused, not averaged
new, o = both([dict(seed=42, W=[[1.0], [0.0]], gaps=[1.0]), dict(seed=43, W=[[0.0], [1.0]], gaps=[1.0])], align_to=[1.0, 1.0])
expect("D1_orthogonal_seeds_refused", is_err(new), {"new": summary(new), "old": summary(o)})

# D3: merged mean_gap is the gap along the merged unit direction
a = np.array([1.0, 0.3]); a /= np.linalg.norm(a)
c = np.array([1.0, -0.3]); c /= np.linalg.norm(c)
new, o = both([dict(seed=42, W=a[:, None].tolist(), gaps=[1.0]), dict(seed=43, W=c[:, None].tolist(), gaps=[1.0])])
# a and c (as stored, float32) both see gap 1 for the activation difference delta with delta.a = delta.c = 1, i.e.
# delta = (1/a[0], 0); along the merged w = (1, 0) that is 1/a[0]
expected_gap = 1.0 / f32(a)[0]
expect("D3_mean_gap_rescaled", not is_err(new) and abs(new[1]["mean_gap"] - expected_gap) < 1e-9 and abs(abs(new[0][0]) - 1) < 1e-9,
       {"expected_gap": expected_gap, "new": summary(new), "old": summary(o)})

# D2: higher rank is refused by the rank-1 interface, and a rank-1 file holding two columns is refused
new, o = both([dict(seed=42, W=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], gaps=[1.0, 2.0], rank=2)], rank=2)
expect("D2_rank2_interface_refused", is_err(new), {"new": summary(new), "old": summary(o)})
new, o = both([dict(seed=42, W=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], gaps=[1.0, 2.0], rank=2, name_rank=1)])
expect("D2_two_column_file_refused", is_err(new), {"new": summary(new), "old": summary(o)})

# D5: seeds with different training protocols are refused
new, o = both([dict(seed=42, W=[[1.0], [0.0]], gaps=[1.0], anchor="prompt", loss_kind="kl"),
               dict(seed=43, W=[[0.9], [0.1]], gaps=[1.0], anchor="current", loss_kind="jsd")], align_to=[1.0, 1.0])
expect("D5_incompatible_protocol_refused", is_err(new), {"new": summary(new), "old": summary(o)})

# Single seed: identical to the previous loader (direction, sign from align_to, gap, cosines)
rng = np.random.default_rng(0)
w = rng.normal(size=8); w /= np.linalg.norm(w)
ref = rng.normal(size=8)
files = [dict(seed=42, W=w[:, None].tolist(), gaps=[0.7], anchor="prompt", loss_kind="kl")]
new = load(d, files, align_to=ref.tolist())
if old is not None:
    o = load(old, files, align_to=ref.tolist())
    same = (not is_err(new) and not is_err(o) and np.allclose(new[0], o[0], atol=1e-12) and abs(new[1]["mean_gap"] - o[1]["mean_gap"]) < 1e-12
            and abs(new[1]["cos_with_align_to"] - o[1]["cos_with_align_to"]) < 1e-12 and new[1]["cos_seed_vs_mean"] == o[1]["cos_seed_vs_mean"])
    expect("single_seed_identical_to_old", same, {"new": summary(new), "old": summary(o)})
else:
    expect("single_seed_sign_and_gap", not is_err(new) and np.dot(new[0], ref) >= 0 and abs(abs(new[1]["mean_gap"]) - 0.7) < 1e-9, {"new": summary(new)})

# D4: a cached teacher made for one T is regenerated when train_das is called with another T on the same pair dicts.


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
pairs = [{"base": enc([1, 2, 3]), "source": enc([4, 5, 6, 3])}, {"base": enc([2, 1, 3]), "source": enc([6, 5, 4, 3])}]
requests = []


def fake_teacher(bundle_, inputs, T):
    requests.append(T)
    return torch.arange(T) + 8


with patch.object(d, "teacher_tokens", fake_teacher):
    r1 = d.train_das(bundle, pairs, 0, T=1, epochs=1, log_fn=None, anchor="prompt")
    r3 = d.train_das(bundle, pairs, 0, T=3, epochs=1, log_fn=None, anchor="prompt")
    r3b = d.train_das(bundle, pairs, 0, T=3, epochs=1, seed=43, log_fn=None, anchor="prompt")   # same T again: cache reused
expect("D4_teacher_regenerated_on_T_change", requests == [1, 1, 3, 3] and len(r3["position_trace"]) == 3 and len(pairs[0]["y_src"]) == 3,
       {"teacher_requests": requests, "r1_T": r1["T"], "r3_T": r3["T"], "r3_trace_steps": len(r3["position_trace"]), "y_src_len": int(len(pairs[0]["y_src"]))})

rep["failures"] = fails
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("PASSED: %d loader / teacher-cache expectations" % (len(rep) - 1))
