"""DAS loader / teacher-cache checks for review rounds 14 (D1-D5) and 15 (§3.1-3.5, §4). CPU only; tiny vectors, a 4-d
frozen toy network, temporary checkpoints written with the production file-name helper.

usage: chk_das_loader.py CODE_DIR OUT.json [OLD_directions.py]
  CODE_DIR/core/directions.py is the fixed version. OLD_directions.py (optional) is the pre-round-14 version (cb3f67b), loaded
  standalone, to show that the single-seed path is bit-identical and what the multi-seed cases used to return.
Round 15 additions: a first seed lacking a protocol field no longer hides a conflict between later seeds; a multi-seed merge
needs the full protocol in every file; identity fields (model, pair ids, ...) are compared key by key; NaN / Inf directions and
gaps are refused; several trimmed seeds report the gap as unavailable; one seed returns direction AND gap bit for bit (200
random files); the teacher cache is keyed by T, model object and prompt ids.
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


FULL = dict(anchor="prompt", loss_kind="kl", positions="last", T=8, bidirectional=False, epochs=10, lr=1e-3, trim=0.0, weighted=False)


def ckpt(folder, seed, W, gaps, rank=1, name_rank=None, omit=(), **extra):
    """A checkpoint with the full training protocol recorded (FULL + rank + n_pairs); `extra` overrides / adds fields, `omit`
    drops fields (legacy files written before a field existed)."""
    row = {"W": torch.tensor(W, dtype=torch.float32), "mean_gap": gaps, "losses": [0.1], "seed": seed, "rank": rank, "n_pairs": 96, **FULL, **extra}
    for k in omit:
        row.pop(k, None)
    torch.save(row, Path(folder) / d.das_file_name("E", 0, name_rank or rank, seed))


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
    return {"W": [round(float(x), 6) for x in w], "norm": round(float(np.linalg.norm(w)), 6),
            "mean_gap": None if meta["mean_gap"] is None else round(float(meta["mean_gap"]), 6), "n_seeds": meta["n_seeds"]}


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

# Round 15 §3.1: a first seed LACKING a protocol field must not hide a conflict between the next two seeds
new, o = both([dict(seed=42, W=[[1.0], [0.0]], gaps=[1.0], omit=("anchor",)), dict(seed=43, W=[[1.0], [0.0]], gaps=[1.0], anchor="prompt"),
               dict(seed=44, W=[[1.0], [0.0]], gaps=[1.0], anchor="current")])
expect("D5_missing_first_key_conflict_refused", is_err(new), {"new": summary(new), "old": summary(o)})

# Round 15 §3.1: a multi-seed merge needs the full protocol in every file, even without a visible conflict
new, o = both([dict(seed=42, W=[[1.0], [0.0]], gaps=[1.0], omit=("anchor",)), dict(seed=43, W=[[1.0], [0.0]], gaps=[1.0])])
expect("D5_missing_protocol_field_multiseed_refused", is_err(new) and "lacks the protocol fields" in new[1], {"new": summary(new), "old": summary(o)})

# Round 15 §3.2: identity fields (model, training pair ids, template ...) are compared like the protocol; a key only one side has is a conflict
new, o = both([dict(seed=42, W=[[1.0], [0.0]], gaps=[1.0], model_name="llama", training_pair_ids=["a"]),
               dict(seed=43, W=[[1.0], [0.0]], gaps=[1.0], model_name="qwen", training_pair_ids=["b"])])
expect("D5_identity_fields_compared", is_err(new) and "model_name" in new[1], {"new": summary(new), "old": summary(o)})
new, o = both([dict(seed=42, W=[[1.0], [0.0]], gaps=[1.0], model_name="llama"), dict(seed=43, W=[[1.0], [0.0]], gaps=[1.0])])
expect("D5_identity_key_on_one_side_refused", is_err(new) and "model_name" in new[1], {"new": summary(new), "old": summary(o)})
new, o = both([dict(seed=42, W=[[1.0], [0.0]], gaps=[1.0], model_name="llama", training_pair_ids=["a", "b"], per_seed={"wall_s": 1.0}),
               dict(seed=43, W=[[1.0], [0.0]], gaps=[1.0], model_name="llama", training_pair_ids=["a", "b"], per_seed={"wall_s": 2.0})])
expect("D5_same_identity_merges_per_seed_bookkeeping_ignored", not is_err(new) and new[1]["n_seeds"] == 2, {"new": summary(new)})

# Round 15 §3.3: NaN / Inf pass a `norm < eps` test; they must be refused by name
new, o = both([dict(seed=42, W=[[float("nan")], [0.0]], gaps=[1.0])])
expect("nonfinite_direction_refused", is_err(new) and "non-finite direction" in new[1], {"new": summary(new), "old": summary(o)})
new, o = both([dict(seed=42, W=[[1.0], [0.0]], gaps=[float("inf")])])
expect("nonfinite_gap_refused", is_err(new) and "non-finite mean_gap" in new[1], {"new": summary(new), "old": summary(o)})

# Round 15 §3.4: several seeds trained with trim > 0 -> direction merged, gap reported unavailable (not a linear merge)
a = np.array([1.0, 0.0]); c = np.array([0.8, 0.6])
new, o = both([dict(seed=42, W=a[:, None].tolist(), gaps=[0.875], trim=0.1), dict(seed=43, W=c[:, None].tolist(), gaps=[2.225], trim=0.1)])
expect("D3_multiseed_trimmed_gap_unavailable", not is_err(new) and new[1]["mean_gap"] is None and new[1]["mean_gap_available"] is False
       and abs(float(np.linalg.norm(new[0])) - 1) < 1e-12 and new[1]["n_seeds"] == 2, {"new": summary(new), "note": None if is_err(new) else new[1]["mean_gap_note"], "old": summary(o)})

# Single seed: identical to the previous loader (direction, sign from align_to, gap, cosines) -- exact, not allclose (round 15 §4)
rng = np.random.default_rng(0)
w = rng.normal(size=8); w /= np.linalg.norm(w)
ref = rng.normal(size=8)
files = [dict(seed=42, W=w[:, None].tolist(), gaps=[0.7], anchor="prompt", loss_kind="kl")]
new = load(d, files, align_to=ref.tolist())
if old is not None:
    o = load(old, files, align_to=ref.tolist())
    same = (not is_err(new) and not is_err(o) and np.array_equal(new[0], o[0]) and new[1]["mean_gap"] == o[1]["mean_gap"]
            and new[1]["cos_with_align_to"] == o[1]["cos_with_align_to"] and new[1]["cos_seed_vs_mean"] == o[1]["cos_seed_vs_mean"])
    expect("single_seed_identical_to_old", same, {"new": summary(new), "old": summary(o)})
else:
    expect("single_seed_sign_and_gap", not is_err(new) and np.dot(new[0], ref) >= 0 and abs(abs(new[1]["mean_gap"]) - 0.7) < 1e-9, {"new": summary(new)})

# Round 15 §4: 200 random single-seed files, with and without align_to -- direction and gap bit for bit against the old loader
rng = np.random.default_rng(15)
n_w = n_gap = 0
with tempfile.TemporaryDirectory() as td:
    for k in range(200):
        w = rng.normal(size=16); w /= np.linalg.norm(w)
        gap = float(rng.normal())
        ref = None if k % 3 == 0 else rng.normal(size=16).tolist()
        ckpt(td, 42, w[:, None].tolist(), [gap])
        wn, mn = d.load_das_direction(td, "E", 0, align_to=ref)
        if old is not None:
            wo, mo = old.load_das_direction(td, "E", 0, align_to=ref)
            n_w += np.array_equal(wn, wo); n_gap += (mn["mean_gap"] == mo["mean_gap"])
        else:   # no old module: the gap must be the stored one up to the display sign, exactly
            # wn has already been sign-aligned. Use the original stored W to
            # predict whether the loader should negate the stored gap.
            stored = f32(w)
            sign = 1.0 if ref is None or float(np.dot(stored, np.asarray(ref) / np.linalg.norm(ref))) >= 0 else -1.0
            expected_w = sign * (stored / np.linalg.norm(stored))
            n_w += np.array_equal(wn, expected_w); n_gap += (mn["mean_gap"] == sign * gap)
expect("single_seed_200_bit_identical", n_w == 200 and n_gap == 200, {"direction_equal": n_w, "gap_equal": n_gap, "vs": "old loader" if old else "stored gap"})

# Legacy single-seed file without the newer protocol fields still loads; meta says the protocol is incomplete
new = load(d, [dict(seed=42, W=[[1.0], [0.0]], gaps=[0.7], omit=("anchor", "loss_kind", "positions", "T", "bidirectional", "epochs", "lr", "trim", "weighted"))])
expect("single_seed_legacy_fields_accepted", not is_err(new) and new[1]["protocol_complete"] is False and new[1]["mean_gap"] == 0.7, {"new": summary(new)})

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

# Round 15 §3.5: the cache is keyed by (T, model object, prompt ids). Sequence from GPT's reproduce_loader_edges.py: T=1 bidirectional ->
# T=2 one-way (both teachers dropped) -> T=2 bidirectional (base regenerated) -> pair 0's source changed at the same T (its teachers
# regenerated, pair 1 untouched) -> another model object (all four regenerated) -> same call again (nothing regenerated).


def bundle_of(seed):
    torch.manual_seed(seed)
    m = Tiny().eval()
    m.requires_grad_(False)
    return types.SimpleNamespace(model=m, device=torch.device("cpu"), hidden_size=4, layer=lambda i, m=m: m.blocks[i], tag=seed)


log = []


def teacher_by_input(bundle_, inputs, T):
    log.append((T, id(bundle_.model), inputs["input_ids"].reshape(-1).tolist()))
    return torch.arange(T) + (int(inputs["input_ids"].sum()) + bundle_.tag) % 10 + 1


b7 = bundle_of(7)
pairs2 = [{"base": enc([1, 2, 3]), "source": enc([4, 5, 6, 3])}, {"base": enc([2, 1, 3]), "source": enc([6, 5, 4, 3])}]
with patch.object(d, "teacher_tokens", teacher_by_input):
    d.train_das(b7, pairs2, 0, T=1, epochs=1, anchor="prompt", bidirectional=True, log_fn=None); n1 = len(log)
    d.train_das(b7, pairs2, 0, T=2, epochs=1, anchor="prompt", bidirectional=False, log_fn=None); n2 = len(log)
    base_cleared = all("y_base" not in p for p in pairs2)
    d.train_das(b7, pairs2, 0, T=2, epochs=1, anchor="prompt", bidirectional=True, log_fn=None); n3 = len(log)
    pairs2[0]["source"] = enc([9, 10, 3])
    d.train_das(b7, pairs2, 0, T=2, epochs=1, anchor="prompt", bidirectional=True, log_fn=None); n4 = len(log)
    src_after = pairs2[0]["y_src"].tolist()
    src_expected = (torch.arange(2) + (int(pairs2[0]["source"]["input_ids"].sum()) + 7) % 10 + 1).tolist()
    pair1_untouched = log[n3:n4] and all(ids != [6, 5, 4, 3] and ids != [2, 1, 3] for _, _, ids in log[n3:n4])
    # The three model objects below are kept alive (b8a, b8b, b8): the cache key uses id(model), which is unique only among LIVE
    # objects -- a released model's address could be reused by the next one (red-team finding).
    b8a = bundle_of(8)
    d.train_das(b8a, pairs2, 0, T=2, epochs=1, anchor="prompt", bidirectional=True, log_fn=None); n5 = len(log)
    b8b = bundle_of(8)
    d.train_das(b8b, pairs2, 0, T=2, epochs=1, anchor="prompt", bidirectional=True, log_fn=None); n6 = len(log)            # another object -> regenerated
    b8 = bundle_of(8)
    d.train_das(b8, pairs2, 0, T=2, epochs=1, anchor="prompt", bidirectional=True, log_fn=None); n7 = len(log)
    d.train_das(b8, pairs2, 0, T=2, epochs=1, anchor="prompt", bidirectional=True, log_fn=None); n8 = len(log)             # same object: cache reused
expect("D4_cache_keyed_by_T_model_and_prompt",
       (n1, n2, n3, n4, n5, n6, n7, n8) == (4, 6, 8, 10, 14, 18, 22, 22) and base_cleared and src_after == src_expected and bool(pair1_untouched),
       {"requests_after_each_call": [n1, n2, n3, n4, n5, n6, n7, n8], "expected_requests": [4, 6, 8, 10, 14, 18, 22, 22],
        "base_cleared_after_T_change": base_cleared, "source_teacher_after_input_change": src_after, "expected_source_teacher": src_expected,
        "T_sequence": [t for t, _, _ in log]})

def clean(x):
    """Non-finite floats (the OLD loader returns NaN / Inf for the non-finite cases) as strings, so the JSON stays strictly parseable."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


rep["failures"] = fails
rep = clean(rep)
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[2]).write_text(json.dumps(rep, indent=1, allow_nan=False))
print(json.dumps(rep, indent=1, allow_nan=False))
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("PASSED: %d loader / teacher-cache expectations" % (len(rep) - 1))
