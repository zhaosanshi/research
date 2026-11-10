"""FP32 output projection for frozen QA models, including tied output weights.

Only the output Linear.forward is replaced. The model still performs its own
normalization, scaling, soft-capping and other processing around that layer.
The FP32 weight/bias copies live for the outermost enabled context on the head;
wrap a training/generation phase, rather than each token, to reuse those copies.
No model parameter is moved, cast, updated, or registered by this helper.

Matmul precision is process-global. These contexts serialize their own use and
restore the previous settings, including on exceptions; unrelated concurrent
code must not change those settings while a context is active. Keep backward()
inside the context when its output-head matmul should use the same policy.
"""
from contextlib import contextmanager
import hashlib
import operator
import threading
import weakref

import torch
from torch import nn
import torch.nn.functional as F


FP32_OUTPUT_PROJECTION_VERSION = "qa_fp32_output_projection_v1"
_LOCK = threading.RLock()
_ACTIVE = weakref.WeakKeyDictionary()
_MISSING = object()


def _output_head(bundle):
    model = bundle.model
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    if head is None:
        for path in ("lm_head", "language_model.lm_head", "model.language_model.lm_head"):
            candidate = model
            for part in path.split("."):
                candidate = getattr(candidate, part, None)
                if candidate is None:
                    break
            if candidate is not None:
                head = candidate
                break
    if not isinstance(head, nn.Linear):
        raise TypeError("FP32 output projection requires an actual nn.Linear output head")
    if head.weight.is_meta or not head.weight.is_floating_point():
        raise TypeError("FP32 output projection requires materialized floating-point head weights")
    name = next((name for name, module in model.named_modules() if module is head), "<output_head>")
    return head, name


def output_projection_metadata(bundle, enabled=True):
    """Describe the requested projection policy without changing the model."""
    if not enabled:
        return {"version": FP32_OUTPUT_PROJECTION_VERSION, "enabled": False,
                "projection_policy": "native"}
    head, name = _output_head(bundle)
    return {"version": FP32_OUTPUT_PROJECTION_VERSION, "enabled": True,
            "head_module": name, "head_type": type(head).__name__,
            "parameter_dtype": str(head.weight.dtype), "parameter_device": str(head.weight.device),
            "projection_dtype": "torch.float32", "projection_autocast": False,
            "input_device_policy": "align_to_materialized_output_weight_preserving_gradient",
            "matmul_precision": "highest", "allow_tf32": False,
            "weight_policy": "detached_fp32_copy_per_outer_context",
            "model_postprocessing": "preserved"}


@contextmanager
def _strict_matmul_precision():
    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        # The legacy TF32 setter also changes the matmul policy. Restore the
        # named policy last so an original "medium" does not become "high".
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)


@contextmanager
def fp32_output_projection(bundle, enabled=True):
    """Temporarily compute the output Linear in FP32 and yield policy metadata.

    The detached cache is for a frozen base model. Gradients still propagate
    through input.float() into hidden states / intervention parameters. Nested
    contexts for the same head share one cache; disabled contexts are no-ops.
    """
    if not enabled:
        yield output_projection_metadata(bundle, enabled=False)
        return
    with _LOCK:
        head, _ = _output_head(bundle)
        metadata = output_projection_metadata(bundle)
        with _strict_matmul_precision():
            state = _ACTIVE.get(head)
            if state is None:
                state = {"depth": 0, "original_forward": head.__dict__.get("forward", _MISSING),
                         "weight": head.weight.detach().to(dtype=torch.float32, copy=True),
                         "bias": (head.bias.detach().to(dtype=torch.float32, copy=True)
                                  if head.bias is not None else None)}

                def project(input):
                    # Casting logits after a BF16 matmul cannot undo rounding.
                    # Replacing Linear.forward also replaces an Accelerate
                    # dispatch wrapper. The preceding block and this head can
                    # live on different GPUs, so align explicitly to the cached
                    # weight. Tensor.to preserves gradients back to the edit.
                    projected_input = input.to(device=state["weight"].device, dtype=torch.float32)
                    with torch.autocast(device_type=projected_input.device.type, enabled=False):
                        return F.linear(projected_input, state["weight"], state["bias"])

                head.forward = project
                _ACTIVE[head] = state
            state["depth"] += 1
            try:
                yield metadata
            finally:
                state["depth"] -= 1
                if state["depth"] == 0:
                    if state["original_forward"] is _MISSING:
                        del head.forward
                    else:
                        head.forward = state["original_forward"]
                    del _ACTIVE[head]


def projection_diagnostic(bundle, inputs, token_ids=None, *, return_hidden=False):
    """Compare two raw output projections of ONE captured final hidden state.

    One original model forward captures only the head input at the last sequence
    position (left-padded inputs are required). The original model still runs its
    normalization and postprocessing; the reported scores are independently
    recomputed F.linear outputs BEFORE any model softcap/scaling afterward.

    "Native" here means head-parameter dtype with autocast/TF32 disabled, not a
    second native-precision model run. The FP32 arm uses the same H and FP32 copies
    of the same stored parameters. This isolates projection arithmetic at that H;
    it does not measure full-model differences or attribute batching drift.

    token_ids is an optional iterable of vocabulary indices. The default return
    is a JSON-safe dict. With return_hidden=True, return (report, H_last_cpu),
    where H is detached, retains its captured dtype, and has shape [batch, dim].
    No model weights are loaded, generated, changed, or trained by this helper.
    """
    from .model_io import forward_logits

    head, name = _output_head(bundle)
    selected_ids = []
    for token_id in (() if token_ids is None else token_ids):
        if isinstance(token_id, bool):
            raise TypeError("diagnostic token IDs must be integers, not booleans")
        try:
            token_id = operator.index(token_id)
        except TypeError as exc:
            raise TypeError("diagnostic token IDs must be integers") from exc
        if not 0 <= token_id < head.weight.shape[0]:
            raise ValueError("diagnostic token ID is outside the output vocabulary")
        if token_id not in selected_ids:
            selected_ids.append(token_id)
    mask = inputs["attention_mask"]
    if mask.ndim != 2 or not mask.shape[0] or not mask.shape[1] or not mask[:, -1].bool().all():
        raise ValueError("projection diagnostic requires nonempty left-padded inputs")

    captured = []

    def capture_input(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("input")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim not in (2, 3):
            raise ValueError("expected output-head input [batch, dim] or [batch, sequence, dim]")
        if hidden.ndim == 3:
            hidden = hidden[:, -1, :]
        if hidden.shape[0] != mask.shape[0]:
            raise ValueError("output-head input does not preserve the diagnostic batch dimension")
        captured.append(hidden.detach().clone())

    with fp32_output_projection(bundle), torch.no_grad():
        handle = head.register_forward_pre_hook(capture_input, with_kwargs=True)
        try:
            # This calls the original model forward, retaining its norm/softcap.
            # The output is deliberately not used as a raw projection score.
            outputs = forward_logits(bundle, inputs, grad=False, logits_to_keep=1)
            del outputs
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("projection diagnostic expected one output-head call, got %d" % len(captured))
        hidden = captured[0]
        state = _ACTIVE[head]
        with torch.autocast(device_type=head.weight.device.type, enabled=False):
            native = F.linear(hidden.to(device=head.weight.device, dtype=head.weight.dtype), head.weight.detach(),
                              head.bias.detach() if head.bias is not None else None)
            fp32 = F.linear(hidden.to(device=state["weight"].device, dtype=torch.float32), state["weight"], state["bias"])
        if not (torch.isfinite(hidden).all() and torch.isfinite(native).all() and torch.isfinite(fp32).all()):
            raise ValueError("non-finite hidden states or scores in projection diagnostic")
        difference = (native.float() - fp32).abs().double()
        hidden_cpu = hidden.cpu().contiguous()
        hidden_stats = hidden_cpu.double()
        native_cpu, fp32_cpu = native.float().cpu(), fp32.cpu()
        selected = [{"token_id": int(token_id),
                     "native_raw_scores": native_cpu[:, token_id].tolist(),
                     "fp32_raw_scores": fp32_cpu[:, token_id].tolist(),
                     "abs_diff": difference[:, token_id].cpu().tolist()}
                    for token_id in selected_ids]
        report = {
            "version": "qa_same_hidden_projection_diagnostic_v1",
            "scope": "pre_model_postprocessing",
            "comparison": "parameter_dtype_vs_fp32_on_one_captured_hidden",
            "model_forward_calls": 1,
            "head_module": name,
            "native_parameter_dtype": str(head.weight.dtype),
            "native_input_dtype": str(head.weight.dtype),
            "native_input_cast": hidden.dtype != head.weight.dtype,
            "fp32_projection_dtype": str(fp32.dtype),
            "projection_autocast": False,
            "allow_tf32": False,
            "score_shape": list(fp32.shape),
            "max_abs_diff": float(difference.max().item()),
            "mean_abs_diff": float(difference.mean().item()),
            "per_example_max_abs_diff": difference.max(dim=-1).values.cpu().tolist(),
            "per_example_mean_abs_diff": difference.mean(dim=-1).cpu().tolist(),
            "selected_tokens": selected,
            "hidden": {
                "position": "last_output_head_input_position",
                "shape": list(hidden_cpu.shape),
                "dtype": str(hidden_cpu.dtype),
                "row_sha256": [hashlib.sha256(row.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                               for row in hidden_cpu],
                "row_mean": hidden_stats.mean(dim=-1).tolist(),
                "row_std": hidden_stats.std(dim=-1, unbiased=False).tolist(),
                "row_l2_norm": hidden_stats.norm(dim=-1).tolist(),
            },
            "interpretation": (
                "Raw projection arithmetic at the same H; model postprocessing is excluded from these scores. "
                "This is not a full-model precision comparison or a proof of the cause of batching drift."
            ),
        }
    return (report, hidden_cpu) if return_hidden else report
