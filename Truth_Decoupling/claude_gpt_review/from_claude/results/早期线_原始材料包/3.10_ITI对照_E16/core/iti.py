"""Inference-Time Intervention (Li et al., NeurIPS 2023) as a same-protocol baseline.

ITI = (1) a linear probe per attention head on the last-token head output (the input of o_proj, split by head),
      (2) the K heads with the highest held-out probe accuracy,
      (3) at inference, shift each selected head's output by alpha * sigma_h * theta_h where theta_h is the unit
          mass-mean-shift direction (mean true - mean false) and sigma_h the std of the projections on it,
          at the last position of every forward (prompt pass + each decode step), as in the official code.
Only the capture / hook plumbing lives here; head selection data and the evaluation are the driver's business.
"""
import contextlib

import numpy as np
import torch

from .directions import fit_logreg, logreg_predict_proba


def head_geometry(bundle):
    cfg = bundle.model.config
    n_heads = int(cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or (int(cfg.hidden_size) // n_heads))
    o_proj = getattr(bundle.attn(0), "o_proj")
    if o_proj.in_features != n_heads * head_dim:
        raise ValueError("o_proj.in_features %d != n_heads %d * head_dim %d" % (o_proj.in_features, n_heads, head_dim))
    return n_heads, head_dim


@contextlib.contextmanager
def capture_head_inputs(bundle, layers, n_heads, head_dim):
    """Yields store[l] -> [B, n_heads, head_dim] float32 cpu: last-token input of o_proj at layer l."""
    store, handles = {}, []

    def make(l):
        def pre(module, args):
            x = args[0]
            store[l] = x[:, -1, :].detach().float().cpu().reshape(x.shape[0], n_heads, head_dim)
        return pre
    for l in layers:
        handles.append(bundle.attn(l).o_proj.register_forward_pre_hook(make(l)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def capture_heads(bundle, texts, layers, batch_size=8):
    """Returns acts[l] -> np.ndarray [N, n_heads, head_dim] (float32)."""
    from .model_io import forward_logits
    n_heads, head_dim = head_geometry(bundle)
    acts = {l: [] for l in layers}
    for i in range(0, len(texts), batch_size):
        inputs = bundle.encode(texts[i:i + batch_size])
        with capture_head_inputs(bundle, layers, n_heads, head_dim) as store:
            forward_logits(bundle, inputs, grad=False)
        for l in layers:
            acts[l].append(store[l].numpy())
    return {l: np.concatenate(v, 0) for l, v in acts.items()}


def select_heads(acts, y, groups, k, n_folds=4, l2=1.0, seed=0):
    """Grouped K-fold held-out accuracy of a logistic probe per (layer, head); returns the top-k list of
    (layer, head, val_acc) and the full accuracy table {layer: [acc per head]}."""
    y = np.asarray(y).astype(int)
    keys = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    fold_of = {g: i % n_folds for i, g in enumerate(keys)}
    folds = np.array([fold_of[g] for g in groups])
    table, cands = {}, []
    for l, A in acts.items():
        n_heads = A.shape[1]
        accs = []
        for h in range(n_heads):
            X = A[:, h, :].astype(np.float64)
            correct = 0
            for f in range(n_folds):
                tr, va = folds != f, folds == f
                if va.sum() == 0 or len(set(y[tr])) < 2:
                    continue
                fit = fit_logreg(X[tr], y[tr], l2=l2, standardize=True)
                p = logreg_predict_proba(fit, X[va])
                correct += int(((p > 0.5).astype(int) == y[va]).sum())
            acc = correct / len(y)
            accs.append(acc)
            cands.append((l, h, acc))
        table[l] = accs
    cands.sort(key=lambda t: (-t[2], t[0], t[1]))
    return cands[:k], table


def head_directions(acts, y, heads):
    """Unit mass-mean-shift direction and projection std per selected head (fit on the estimation statements)."""
    y = np.asarray(y).astype(bool)
    dirs, sigmas = [], []
    for (l, h, _) in heads:
        X = acts[l][:, h, :].astype(np.float64)
        d = X[y].mean(0) - X[~y].mean(0)
        d = d / (np.linalg.norm(d) + 1e-8)
        dirs.append(d.astype(np.float32))
        sigmas.append(float((X @ d).std()))
    return np.stack(dirs), np.array(sigmas, dtype=np.float32)


class ITISpec:
    def __init__(self, heads, directions, sigmas, alpha=15.0):
        self.heads = [(int(l), int(h)) for (l, h, *_) in heads]
        self.directions = torch.as_tensor(np.asarray(directions), dtype=torch.float32)
        self.sigmas = torch.as_tensor(np.asarray(sigmas), dtype=torch.float32)
        self.alpha = alpha
        self.by_layer = {}
        for i, (l, h) in enumerate(self.heads):
            self.by_layer.setdefault(l, []).append((h, i))


class _ITIHook:
    def __init__(self, spec, layer, head_dim, device):
        self.spec, self.layer, self.head_dim, self.device = spec, layer, head_dim, device
        self.calls = 0

    def __call__(self, module, args):
        x = args[0]
        B = x.shape[0]
        a = self.spec.alpha
        a = a.to(self.device).float().reshape(-1) if torch.is_tensor(a) else torch.full((B,), float(a), device=self.device)
        if a.numel() == 1 and B > 1:
            a = a.expand(B)
        x2 = x.clone()
        last = x2[:, -1, :].float()
        for (h, i) in self.spec.by_layer[self.layer]:
            shift = (a * float(self.spec.sigmas[i])).view(B, 1) * self.spec.directions[i].to(self.device).view(1, -1)
            last[:, h * self.head_dim:(h + 1) * self.head_dim] += shift
        x2[:, -1, :] = last.to(x.dtype)
        self.calls += 1
        return (x2,) + tuple(args[1:])


@contextlib.contextmanager
def iti_steer(bundle, spec, alpha=None):
    """Register ITI pre-hooks on o_proj of every layer that owns a selected head, for the duration of the context."""
    if spec is None:
        yield {}
        return
    if alpha is not None:
        spec.alpha = alpha
    _, head_dim = head_geometry(bundle)
    hooks, handles = {}, []
    try:
        for l in spec.by_layer:
            hk = _ITIHook(spec, l, head_dim, bundle.device)
            hooks[l] = hk
            handles.append(bundle.attn(l).o_proj.register_forward_pre_hook(hk))
        yield hooks
    finally:
        for h in handles:
            h.remove()
