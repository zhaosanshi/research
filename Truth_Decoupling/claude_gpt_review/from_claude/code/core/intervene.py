"""Forward-hook steering (inference time).

A SteerSpec describes one direction applied at one layer:

  mode="add"          h += alpha * (W @ coef)                 constant shift (CAA / ITI style);
                      coef defaults to the direction's mean_gap so alpha=1 ~ one
                      training-set interchange gap (source-free, as v12 Sec. source-free).
  mode="amplify"      h += alpha * W W^T (h - center)         gain control along W: alpha>0 sharpens
                      the model's own signal, alpha=-1 erases it, alpha=0 no-op.
  mode="interchange"  h += alpha * W W^T (h_src - h)          paired DAS interchange (needs source act).
  mode="clamp"        h += alpha * (z_target - W^T h) W       pin the projection to a target value.

`alpha` may be a float or a per-sample tensor [B] (per-input gating).
Several specs may share a layer; a LayerSteerHook applies them in order. The hook
acts on the last position of every forward call (prompt pass + every decode step),
i.e. the "continuous" mode of v12, or on all positions if position="all".
"""
import contextlib

import torch


class SteerSpec:
    def __init__(self, W, layer, mode="add", alpha=1.0, coef=None, center=None, source_act=None,
                 z_target=None, position="last", name=None, max_calls=None):
        W = torch.as_tensor(W, dtype=torch.float32)
        if W.dim() == 1:
            W = W.unsqueeze(1)
        self.W = W                      # [d, r]
        self.layer = int(layer)
        self.mode = mode
        self.alpha = alpha
        self.max_calls = max_calls      # e.g. 1 = act on the prompt pass only, not on the decode steps
        self.coef = None if coef is None else torch.as_tensor(coef, dtype=torch.float32).reshape(-1)
        self.center = None if center is None else torch.as_tensor(center, dtype=torch.float32).reshape(-1)
        self.source_act = None if source_act is None else torch.as_tensor(source_act, dtype=torch.float32)
        self.z_target = None if z_target is None else torch.as_tensor(z_target, dtype=torch.float32).reshape(-1)
        self.position = position
        self.name = name or mode
        if mode == "add" and self.coef is None:
            self.coef = torch.ones(self.W.shape[1])
        if mode == "interchange" and self.source_act is None:
            raise ValueError("interchange needs source_act")
        if mode == "clamp" and self.z_target is None:
            raise ValueError("clamp needs z_target")

    def to(self, device):
        self.W = self.W.to(device)
        for attr in ("coef", "center", "source_act", "z_target"):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, v.to(device))
        if torch.is_tensor(self.alpha):
            self.alpha = self.alpha.to(device)
        return self

    def _alpha(self, B, device):
        a = self.alpha
        if torch.is_tensor(a):
            a = a.to(device).float().reshape(-1)
            if a.numel() == 1:
                a = a.expand(B)
            return a.view(B, 1)
        return torch.full((B, 1), float(a), device=device)

    def delta(self, h_pos):
        """h_pos: [B, d] float32 -> delta [B, d]."""
        B, device = h_pos.shape[0], h_pos.device
        if self.W.device != device:                       # layers may live on another GPU under device_map
            self.to(device)
        a = self._alpha(B, device)
        W = self.W
        if self.mode == "add":
            return a * (W @ self.coef).unsqueeze(0)
        if self.mode == "amplify":
            c = self.center if self.center is not None else torch.zeros(h_pos.shape[1], device=device)
            return a * (((h_pos - c) @ W) @ W.T)
        if self.mode == "interchange":
            src = self.source_act
            if src.dim() == 1:
                src = src.unsqueeze(0)
            if src.shape[0] == 1 and B > 1:
                src = src.expand(B, -1)
            return a * (((src - h_pos) @ W) @ W.T)
        if self.mode == "clamp":
            z = h_pos @ W                                   # [B, r]
            zt = self.z_target
            # per-sample targets ([B*r] values) or one shared target ([r] values)
            zt = zt.view(B, -1) if zt.numel() == B * W.shape[1] and B > 1 else zt.view(1, -1)
            return a * ((zt - z) @ W.T)
        raise ValueError("unknown mode %s" % self.mode)


class LayerSteerHook:
    """Applies a list of SteerSpecs (all at the same layer) to the layer output."""

    def __init__(self, specs):
        self.specs = list(specs)
        self.call_count = 0
        self.applied_count = 0

    def __call__(self, module, inp, out):
        self.call_count += 1
        h = out[0] if isinstance(out, tuple) else out
        if not self.specs:
            return out
        h_new = h.clone()
        for spec in self.specs:
            if spec.max_calls is not None and self.call_count > spec.max_calls:
                continue
            if spec.position == "last":
                h_pos = h_new[:, -1, :].float()
                h_new[:, -1, :] = (h_pos + spec.delta(h_pos)).to(h.dtype)
            else:  # all positions
                B, T, d = h_new.shape
                flat = h_new.reshape(B * T, d).float()
                # per-sample alpha broadcast over positions
                saved_alpha = spec.alpha
                if torch.is_tensor(saved_alpha) and saved_alpha.numel() == B:
                    spec.alpha = saved_alpha.reshape(B, 1).expand(B, T).reshape(-1)
                h_new = (flat + spec.delta(flat)).to(h.dtype).reshape(B, T, d)
                spec.alpha = saved_alpha
        self.applied_count += 1
        if isinstance(out, tuple):
            return (h_new,) + tuple(out[1:])
        return h_new


@contextlib.contextmanager
def steer(bundle, specs):
    """Register hooks for `specs` (any layers) for the duration of the context."""
    by_layer = {}
    for s in specs or []:
        s.to(bundle.device)
        by_layer.setdefault(s.layer, []).append(s)
    hooks, handles = {}, []
    try:
        for layer, ss in by_layer.items():
            hk = LayerSteerHook(ss)
            hooks[layer] = hk
            handles.append(bundle.layer(layer).register_forward_hook(hk))
        yield hooks
    finally:
        for hd in handles:
            hd.remove()


def next_token_logits(bundle, inputs, specs=None):
    """Last-position logits under steering (prompt forward only)."""
    from .model_io import forward_logits
    with steer(bundle, specs):
        logits = forward_logits(bundle, inputs, grad=False)
    return logits[:, -1, :].detach().float().cpu()


def generate(bundle, inputs, specs=None, max_new_tokens=64):
    """Greedy generation with steering applied at every decode step. Returns (texts, hooks)."""
    from .model_io import greedy_generate
    with steer(bundle, specs) as hooks:
        texts, _ = greedy_generate(bundle, inputs, max_new_tokens)
    return texts, {l: {"calls": h.call_count, "applied": h.applied_count} for l, h in hooks.items()}


def sanity_zero_direction(bundle, inputs, layer, max_new_tokens=32):
    """v12-style sanity check: a zero direction must reproduce the baseline exactly."""
    from .model_io import greedy_generate
    base, _ = greedy_generate(bundle, inputs, max_new_tokens)
    zero = SteerSpec(torch.zeros(bundle.hidden_size), layer, mode="add", alpha=1.0)
    steered, hooks = generate(bundle, inputs, [zero], max_new_tokens)
    return {"match": base == steered, "baseline": base, "steered": steered, "hooks": hooks}
