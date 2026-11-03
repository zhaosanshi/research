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
i.e. the "continuous" mode of v12, on all positions if position="all", or on an
explicit boolean [batch, sequence] position_mask if position="mask". Masks name
the positions producing the scored next-token logits, not the target-token
positions. A caller must exclude padding from an explicit mask.
"""
import contextlib

import torch


class SteerSpec:
    def __init__(self, W, layer, mode="add", alpha=1.0, coef=None, center=None, source_act=None,
                 z_target=None, position="last", name=None, max_calls=None,
                 position_mask=None, decode_center=None):
        if position not in ("last", "all", "mask"):
            raise ValueError("position must be 'last', 'all', or 'mask'")
        if position == "mask" and position_mask is None:
            raise ValueError("position='mask' requires position_mask")
        if position != "mask" and position_mask is not None:
            raise ValueError("position_mask is only valid with position='mask'")
        if decode_center is not None and position != "last":
            raise ValueError("decode_center is only valid with position='last'")
        W = torch.as_tensor(W, dtype=torch.float32)
        if W.dim() == 1:
            W = W.unsqueeze(1)
        if W.dim() != 2:
            raise ValueError("W must have shape [hidden] or [hidden, rank]")
        self.W = W                      # [d, r]
        self.layer = int(layer)
        self.mode = mode
        self.alpha = alpha
        self.max_calls = max_calls      # e.g. 1 = act on the prompt pass only, not on the decode steps
        self.coef = None if coef is None else torch.as_tensor(coef, dtype=torch.float32).reshape(-1)
        self.center = None if center is None else torch.as_tensor(center, dtype=torch.float32).reshape(-1)
        self.decode_center = (None if decode_center is None
                              else torch.as_tensor(decode_center, dtype=torch.float32).reshape(-1))
        self.source_act = None if source_act is None else torch.as_tensor(source_act, dtype=torch.float32)
        self.z_target = None if z_target is None else torch.as_tensor(z_target, dtype=torch.float32).reshape(-1)
        self.position = position
        self.position_mask = None if position_mask is None else torch.as_tensor(position_mask)
        if self.position_mask is not None and (self.position_mask.dtype != torch.bool or self.position_mask.dim() != 2):
            raise ValueError("position_mask must be a boolean [batch, sequence] tensor")
        self.name = name or mode
        self.n_applied = 0              # how many forward calls this spec actually edited (audit of max_calls / prefill-only use)
        if mode == "add" and self.coef is None:
            self.coef = torch.ones(self.W.shape[1])
        if mode == "interchange" and self.source_act is None:
            raise ValueError("interchange needs source_act")
        if mode == "clamp" and self.z_target is None:
            raise ValueError("clamp needs z_target")

    def to(self, device):
        self.W = self.W.to(device)
        for attr in ("coef", "center", "decode_center", "source_act", "z_target", "position_mask"):
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
            elif a.numel() != B:
                raise ValueError("alpha must be scalar or contain one value per original batch row")
            return a.view(B, 1)
        return torch.full((B, 1), float(a), device=device)

    def delta(self, h_pos, *, batch_rows=None, batch_size=None):
        """Return edits for [N, hidden] positions, preserving their batch rows.

        The default remains one position per batch row. For several selected
        positions per example, batch_rows maps every selected position back to
        its original sample, and batch_size specifies that original batch size.
        Per-sample alpha, source activations and clamping targets follow this
        map rather than being interpreted as position-level batch parameters.
        """
        if h_pos.dim() != 2 or h_pos.shape[1] != self.W.shape[0]:
            raise ValueError("h_pos must have shape [positions, hidden]")
        N, device = h_pos.shape[0], h_pos.device
        if batch_rows is None:
            if batch_size is not None and batch_size != N:
                raise ValueError("batch_size differs from h_pos rows without batch_rows")
            B = N
            rows = torch.arange(N, device=device)
        else:
            if batch_size is None:
                raise ValueError("batch_rows requires original batch_size")
            B = int(batch_size)
            rows = torch.as_tensor(batch_rows, dtype=torch.long, device=device)
            if rows.dim() != 1 or rows.numel() != N or bool(((rows < 0) | (rows >= B)).any()):
                raise ValueError("batch_rows must map every selected position to a valid original batch row")
        if self.W.device != device:                       # layers may live on another GPU under device_map
            self.to(device)
        a = self._alpha(B, device)[rows]
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
            if src.dim() != 2 or src.shape[1] != h_pos.shape[1] or src.shape[0] not in (1, B):
                raise ValueError("source_act must be shared [hidden] or [batch, hidden]")
            src = src.expand(N, -1) if src.shape[0] == 1 else src[rows]
            return a * (((src - h_pos) @ W) @ W.T)
        if self.mode == "clamp":
            z = h_pos @ W                                   # [B, r]
            zt = self.z_target
            # Per-sample targets ([B*r] values) or one shared target ([r] values).
            if zt.numel() == W.shape[1]:
                zt = zt.view(1, -1).expand(N, -1)
            elif zt.numel() == B * W.shape[1]:
                zt = zt.view(B, -1)[rows]
            else:
                raise ValueError("z_target must be shared [rank] or contain [batch, rank] values")
            return a * ((zt - z) @ W.T)
        raise ValueError("unknown mode %s" % self.mode)


class LayerSteerHook:
    """Applies a list of SteerSpecs (all at the same layer) to the layer output."""

    def __init__(self, specs):
        self.specs = list(specs)
        self.call_count = 0
        self.applied_count = 0
        self.first_shape = None         # [B, T, d] of the first forward call: lets a caller assert that call 1 was the full prompt

    def __call__(self, module, inp, out):
        self.call_count += 1
        h = out[0] if isinstance(out, tuple) else out
        if self.call_count == 1:
            self.first_shape = tuple(h.shape)
        if not self.specs:
            return out
        if h.dim() != 3:
            raise ValueError("layer output must have shape [batch, sequence, hidden]")
        B, T, _ = h.shape
        h_new = None
        for spec in self.specs:
            if spec.max_calls is not None and self.call_count > spec.max_calls:
                continue
            if spec.position not in ("last", "all", "mask"):
                raise ValueError("position must be 'last', 'all', or 'mask'")
            if spec.decode_center is not None and spec.position != "last":
                raise ValueError("decode_center is only valid with position='last'")
            # Move both centres before saving/restoring either one: the layer
            # may live on a different device from the bundle's input device.
            spec.to(h.device)
            if spec.position == "last":
                selected = torch.zeros((B, T), dtype=torch.bool, device=h.device)
                if T:
                    selected[:, -1] = True
            elif spec.position == "all":
                selected = torch.ones((B, T), dtype=torch.bool, device=h.device)
            else:
                selected = spec.position_mask
                if (selected is None or selected.dtype != torch.bool
                        or tuple(selected.shape) != (B, T)):
                    raise ValueError("position_mask must be boolean and match layer output [batch, sequence]")
            if not bool(selected.any()):
                continue
            if h_new is None:
                h_new = h.clone()
            # Advanced indexing packs positions, so retain each position's
            # original row explicitly. Never broadcast B-row parameters as if
            # the packed N positions were N independent examples.
            batch_rows = torch.arange(B, device=h.device)[:, None].expand(B, T)[selected]
            h_pos = h_new[selected].float()
            saved_center = spec.center
            try:
                if spec.position == "last" and self.call_count > 1 and spec.decode_center is not None:
                    spec.center = spec.decode_center
                delta = spec.delta(h_pos, batch_rows=batch_rows, batch_size=B)
                h_new[selected] = (h_pos + delta).to(h.dtype)
            finally:
                spec.center = saved_center
            spec.n_applied += 1
        if h_new is None:
            return out
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
