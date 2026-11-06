"""Generation-aligned direction adaptation; this is not the original DAS objective.

The frozen model is trained through the exact source-free centered-amplification
operator used at deployment. Only one unit vector per explicitly chosen layer
is updated. Correct-answer token CE is primary; paired answer ranking is optional.
"""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .intervene import SteerSpec
from .model_io import forward_logits
from .qa_context import answer_logprobs, encode_answer, pack_answers
from .qa_precision import fp32_output_projection


PROTOCOL = "qa_generation_adapter_v1"
OPERATOR = "centered_amplification"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vector(value, hidden, name, unit=False):
    x = torch.as_tensor(value).detach().float().cpu().reshape(-1)
    if x.shape != (hidden,) or not bool(torch.isfinite(x).all()):
        raise ValueError("%s must be a finite hidden-size vector" % name)
    if unit and abs(float(x.norm()) - 1.0) > 1e-4:
        raise ValueError("%s must be unit length" % name)
    return x.clone()


def resolve_stop_token_id(bundle, stop_token_id=None):
    """Use a declared tokenizer EOS or an explicit generation stop ID; never guess from a list."""
    generation = getattr(getattr(bundle.model, "generation_config", None), "eos_token_id", None)
    stops = list(generation) if isinstance(generation, (list, tuple)) else [generation]
    if not stops or any(type(x) is not int or x < 0 for x in stops):
        raise ValueError("include_eos requires a valid model.generation_config.eos_token_id")
    token = getattr(bundle.tokenizer, "eos_token_id", None) if stop_token_id is None else stop_token_id
    if type(token) is not int or token < 0:
        raise ValueError("tokenizer.eos_token_id is missing/invalid; supply an explicit --stop-token-id from the generation stop set")
    if token not in stops:
        raise ValueError("selected EOS ID %r is not in generation_config.eos_token_id %r; supply an explicit --stop-token-id" % (token, stops))
    getter = getattr(bundle.model, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    vocab_size = getattr(getattr(bundle.model, "config", None), "vocab_size", None)
    if head is not None and hasattr(head, "weight"):
        vocab_size = int(head.weight.shape[0])
    if vocab_size is not None and token >= vocab_size:
        raise ValueError("selected EOS ID is outside the model output vocabulary")
    return token


def stop_training_config(adapter):
    """Old content-only artifacts omitted both keys; read them without mutating the artifact."""
    cfg = adapter.get("train_config", {})
    include_eos, stop = cfg.get("include_eos", False), cfg.get("stop_token_id")
    if type(include_eos) is not bool:
        raise ValueError("stored include_eos must be boolean")
    if include_eos and (type(stop) is not int or stop < 0):
        raise ValueError("EOS-trained adapter needs an explicit valid stored stop_token_id")
    if not include_eos and stop is not None:
        raise ValueError("content-only adapter must not declare a supervised stop token")
    return include_eos, stop


def prepare_generation_pairs(bundle, cfg, records, include_eos=False, stop_token_id=None):
    """Encode raw answers; optionally append exactly one validated stop ID to each target sequence."""
    if type(include_eos) is not bool:
        raise ValueError("include_eos must be boolean")
    if not include_eos and stop_token_id is not None:
        raise ValueError("stop_token_id requires include_eos=True")
    stop = resolve_stop_token_id(bundle, stop_token_id) if include_eos else None
    if not records:
        raise ValueError("generation adaptation needs at least one training pair")
    seen, pairs = set(), []
    for r in records:
        pid = r.get("pair_id")
        if not isinstance(pid, str) or not pid or pid in seen:
            raise ValueError("training pair IDs must be unique nonempty strings")
        seen.add(pid)
        pair = {"pair_id": pid}
        for side, key in (("positive", "positive_answer"), ("negative", "negative_answer")):
            pair[side] = encode_answer(bundle, cfg["qa_system"], r["question"], r[key], cfg["qa_user"])
            if include_eos:
                item = pair[side]
                item["content_answer_length"] = item["answer_length"]
                item["answer_ids"] = item["answer_ids"] + [stop]
                item["full_ids"] = item["full_ids"] + [stop]
                item["answer_length"] += 1
                item["text_audit"].update(appended_eos=True, stop_token_id=stop,
                                          terminal_token_construction="append one validated generation stop ID after content IDs",
                                          full_text_excludes_terminal_special_token=True)
        if pair["positive"]["prefix_ids"] != pair["negative"]["prefix_ids"]:
            raise ValueError("paired training answers must share their deployment prompt")
        pairs.append(pair)
    return pairs


@contextlib.contextmanager
def frozen_model(model):
    """Preserve pre-existing requires_grad flags and train/eval states, also on failure."""
    parameters = [(p, p.requires_grad, p._version) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        for p, _, _ in parameters:
            p.requires_grad_(False)
        yield
    finally:
        mutated = any(p._version != version for p, _, version in parameters)
        for p, flag, _ in parameters:
            p.requires_grad_(flag)
        for m, training in modes:
            m.training = training
        if mutated:
            raise RuntimeError("a frozen base-model parameter was modified during adaptation")


@contextlib.contextmanager
def differentiable_amplification(bundle, raw_vectors, centers, packed, alpha, trace=None):
    """Torch-only differentiable equivalent of the legacy prompt/decode mask specs."""
    handles = []
    predictor = packed["predictor_mask"]
    prompt = packed["prompt_end_mask"]
    if predictor.dtype != torch.bool or prompt.dtype != torch.bool or predictor.shape != prompt.shape:
        raise ValueError("predictor and prompt masks must be aligned booleans")
    if bool((prompt & ~predictor).any()):
        raise ValueError("prompt-end positions must be answer-token predictors")

    def make_hook(layer):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.ndim != 3 or tuple(h.shape[:2]) != tuple(predictor.shape):
                raise ValueError("training layer output does not align with packed answer positions")
            w = F.normalize(raw_vectors[layer], dim=0).to(h.device, torch.float32)
            updated = h.clone()
            observations = []
            for name, selected in (("prompt", prompt), ("decode", predictor & ~prompt)):
                selected = selected.to(h.device)
                if not bool(selected.any()):
                    continue
                c = centers[layer][name].to(h.device, torch.float32)
                before = updated[selected].float()
                delta = float(alpha) * ((before - c) @ w).unsqueeze(-1) * w.unsqueeze(0)
                updated[selected] = (before + delta).to(h.dtype)
                if trace is not None:
                    observations.append({"center": name, "positions": selected.nonzero().detach().cpu().tolist(),
                                         "intended_delta_norm": float(delta.detach().norm()),
                                         "kept_delta_norm": float((updated[selected].float() - before).detach().norm())})
            if trace is not None:
                trace.append({"layer": int(layer), "residual_dtype": str(h.dtype), "writes": observations})
            return (updated,) + tuple(out[1:]) if isinstance(out, tuple) else updated
        return hook

    try:
        for layer in sorted(raw_vectors):
            handles.append(bundle.layer(layer).register_forward_hook(make_hook(layer)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def train_generation_adapter(bundle, records, cfg, initial_directions, centers, *, alpha=1.0,
                             init="das", epochs=3, lr=1e-3, clip=1.0, ranking_weight=0.0,
                             seed=42, identity=None, log_fn=print, include_eos=False, stop_token_id=None):
    if init not in ("das", "md", "random"):
        raise ValueError("init must be das, md or random")
    if type(epochs) is not int or epochs < 1 or type(seed) is not int:
        raise ValueError("epochs must be positive and seed must be an integer")
    for name, v in (("alpha", alpha), ("lr", lr), ("clip", clip)):
        if not math.isfinite(v) or v <= 0:
            raise ValueError(name + " must be finite and positive")
    if not math.isfinite(ranking_weight) or ranking_weight < 0:
        raise ValueError("ranking_weight must be finite and nonnegative")
    layers = sorted(initial_directions)
    if not layers or any(type(l) is not int or not 0 <= l < bundle.num_layers for l in layers):
        raise ValueError("explicit valid layers are required")
    if set(layers) != set(centers):
        raise ValueError("centers must cover exactly the adapted layers")
    pairs = prepare_generation_pairs(bundle, cfg, records, include_eos=include_eos, stop_token_id=stop_token_id)
    stop = pairs[0]["positive"]["text_audit"]["stop_token_id"] if include_eos else None
    checked_centers = {l: {k: _vector(centers[l][k], bundle.hidden_size, "%s L%d" % (k, l))
                           for k in ("prompt", "decode")} for l in layers}
    raw, initial = {}, {}
    for l in layers:
        v = _vector(initial_directions[l], bundle.hidden_size, "initial L%d" % l, unit=True)
        if init == "random":
            generator = torch.Generator(device="cpu").manual_seed(seed + l)
            v = F.normalize(torch.randn(bundle.hidden_size, generator=generator), dim=0)
        initial[str(l)] = v.clone()
        raw[l] = torch.nn.Parameter(v.to(bundle.device))
    optimizer = torch.optim.Adam(list(raw.values()), lr=lr)
    norms = {str(l): [] for l in layers}
    losses, traces = [], []

    def mean_logp(item, trace=None):
        packed = pack_answers([item], bundle.tokenizer.pad_token_id, bundle.device)
        with differentiable_amplification(bundle, raw, checked_centers, packed, alpha, trace):
            logits = forward_logits(bundle, packed["inputs"], grad=True)
            result = answer_logprobs(logits, packed)[0] / item["answer_length"]
        return result

    with frozen_model(bundle.model), fp32_output_projection(bundle):
        for epoch in range(epochs):
            totals = {"loss": 0.0, "positive_token_ce": 0.0, "pair_ranking": 0.0}
            for index, pair in enumerate(pairs):
                optimizer.zero_grad(set_to_none=True)
                trace = [] if epoch == 0 and index == 0 else None
                positive = mean_logp(pair["positive"], trace)
                ce = -positive
                ranking = ce.new_zeros(())
                if ranking_weight:
                    negative = mean_logp(pair["negative"])
                    ranking = F.softplus(negative - positive)
                loss = ce + ranking_weight * ranking
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite generation-adapter loss at " + pair["pair_id"])
                loss.backward()
                for l in layers:
                    grad = raw[l].grad
                    if grad is None or not bool(torch.isfinite(grad).all()):
                        raise RuntimeError("missing/nonfinite adapter gradient at L%d" % l)
                    norms[str(l)].append(float(grad.norm()))
                torch.nn.utils.clip_grad_norm_(list(raw.values()), clip, error_if_nonfinite=True)
                optimizer.step()
                for l in layers:
                    if not bool(torch.isfinite(raw[l]).all()) or float(raw[l].detach().norm()) < 1e-12:
                        raise RuntimeError("invalid adapter direction after optimization")
                for key, value in (("loss", loss), ("positive_token_ce", ce), ("pair_ranking", ranking)):
                    totals[key] += float(value.detach())
                if trace is not None:
                    for item in trace:
                        item.update(pair_id=pair["pair_id"], prompt_length=pair["positive"]["prompt_length"],
                                    answer_length=pair["positive"]["answer_length"],
                                    final_answer_position=len(pair["positive"]["full_ids"]) - 1)
                        if include_eos:
                            item.update(include_eos=True, stop_token_id=stop,
                                        final_content_position=len(pair["positive"]["full_ids"]) - 2,
                                        stop_token_position=len(pair["positive"]["full_ids"]) - 1)
                    traces.extend(trace)
            losses.append({k: v / len(pairs) for k, v in totals.items()})
            if log_fn:
                log_fn("[%s_generation_adapter %s] epoch %d/%d loss %.6f CE %.6f ranking %.6f" %
                       (init, layers, epoch + 1, epochs, losses[-1]["loss"],
                        losses[-1]["positive_token_ce"], losses[-1]["pair_ranking"]))
    for l, values in norms.items():
        if not any(v > 0 for v in values):
            raise RuntimeError("all adapter updates had zero gradient at layer " + l)
    result = {
        "protocol": PROTOCOL, "method": init + "_generation_adapter", "init": init,
        "model": getattr(bundle, "name", bundle.cfg.get("name")), "model_config": bundle.cfg,
        "layers": layers, "hidden_size": bundle.hidden_size, "alpha": float(alpha),
        "operator": OPERATOR, "mode": "continuous", "precision": "float32_output_projection",
        "config": {k: cfg[k] for k in ("qa_system", "qa_user", "max_new_tokens", "batch_size")},
        "W": {str(l): F.normalize(raw[l].detach().float().cpu(), dim=0) for l in layers},
        "initial_W": initial,
        "centers": {str(l): checked_centers[l] for l in layers},
        "training_pair_ids": [p["pair_id"] for p in pairs],
        "training_tokens_sha256": fingerprint([{ "pair_id": p["pair_id"], **{
            side: {k: p[side][k] for k in ("prefix_ids", "full_ids", "answer_ids")}
            for side in ("positive", "negative")}} for p in pairs]),
        "train_config": {"epochs": epochs, "lr": float(lr), "clip": float(clip), "seed": seed,
                         "ranking_weight": float(ranking_weight), "optimizer": "Adam", "alpha": float(alpha),
                         "include_eos": include_eos, "stop_token_id": stop},
        "objective": "correct-answer mean token CE + ranking_weight * softplus(negative_mean_logp - positive_mean_logp)",
        "positions": ("prompt_end and later answer predictors including the final content token predicting EOS; terminal EOS state excluded"
                      if include_eos else "prompt_end and later answer predictors; final answer token excluded"),
        "losses_measurement": "online pre-update epoch means", "losses": losses, "position_trace": traces,
        "gradient_audit": {l: {"minimum": min(v), "maximum": max(v), "zero_updates": sum(x == 0 for x in v),
                               "updates": len(v)} for l, v in norms.items()},
        "base_parameter_versions_unchanged": True,
    }
    for key, value in (identity or {}).items():
        if key in result:
            raise ValueError("identity cannot override training result field " + key)
        result[key] = value
    return result


def validate_adapter(adapter, expected_model=None, expected_alpha=None):
    stop_training_config(adapter)
    if adapter.get("protocol") != PROTOCOL or adapter.get("init") not in ("das", "md", "random"):
        raise ValueError("unsupported generation adapter")
    if adapter.get("method") != adapter["init"] + "_generation_adapter":
        raise ValueError("adapter method name does not match initialization")
    if adapter.get("operator") != OPERATOR or adapter.get("mode") != "continuous":
        raise ValueError("adapter deployment operator differs from training")
    if expected_model is not None and adapter.get("model") != expected_model:
        raise ValueError("adapter belongs to another model")
    alpha = adapter.get("alpha")
    if not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid adapter alpha")
    if expected_alpha is not None and alpha != expected_alpha:
        raise ValueError("deployment alpha must equal the trained alpha")
    if adapter.get("train_config", {}).get("alpha") != alpha:
        raise ValueError("stored deployment and training alpha disagree")
    layers = adapter.get("layers", [])
    if not layers or any(type(l) is not int or l < 0 for l in layers) or layers != sorted(set(layers)):
        raise ValueError("adapter layers must be unique sorted nonnegative integers")
    if set(adapter.get("W", {})) != {str(l) for l in layers} or set(adapter.get("centers", {})) != {str(l) for l in layers}:
        raise ValueError("adapter layer vectors are incomplete")
    hidden = adapter["hidden_size"]
    for l in layers:
        _vector(adapter["W"][str(l)], hidden, "W", unit=True)
        for center in ("prompt", "decode"):
            _vector(adapter["centers"][str(l)][center], hidden, center)
    ids = adapter.get("training_pair_ids", [])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("adapter training identities are empty/duplicated")
    return adapter


def save_adapter(adapter, directory):
    validate_adapter(adapter)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path, manifest = directory / "adapter.pt", directory / "adapter.json"
    if path.exists() or manifest.exists():
        raise FileExistsError("adapter output exists; use a new run directory")
    torch.save(adapter, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)
    metadata = {k: v for k, v in adapter.items() if k not in ("W", "initial_W", "centers")}
    metadata.update(checkpoint_file=path.name, checkpoint_sha256=digest(path))
    tmp = Path(str(manifest) + ".tmp")
    tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, manifest)
    return metadata


def load_adapter(path, expected_model=None, expected_alpha=None):
    path = Path(path)
    if path.is_dir():
        path = path / "adapter.pt"
    manifest = json.loads(path.with_suffix(".json").read_text())
    if manifest.get("checkpoint_file") != path.name or manifest.get("checkpoint_sha256") != digest(path):
        raise ValueError("adapter checkpoint differs from its manifest")
    adapter = torch.load(path, map_location="cpu", weights_only=True)
    metadata = {k: v for k, v in adapter.items() if k not in ("W", "initial_W", "centers")}
    if any(manifest.get(k) != v for k, v in metadata.items()):
        raise ValueError("adapter metadata and manifest disagree")
    return validate_adapter(adapter, expected_model, expected_alpha)


def adapter_specs(adapter, packed=None, expected_alpha=None, name=None):
    validate_adapter(adapter, expected_alpha=expected_alpha)
    out = []
    for l in adapter["layers"]:
        w, c = adapter["W"][str(l)], adapter["centers"][str(l)]
        label = "%s_L%d" % (name or adapter["method"], l)
        if packed is None:
            out.append(SteerSpec(w, l, mode="amplify", alpha=adapter["alpha"], center=c["prompt"],
                                 decode_center=c["decode"], name=label))
        else:
            for kind, mask in (("prompt", packed["prompt_end_mask"]),
                               ("decode", packed["predictor_mask"] & ~packed["prompt_end_mask"])):
                out.append(SteerSpec(w, l, mode="amplify", alpha=adapter["alpha"], center=c[kind],
                                     position="mask", position_mask=mask, name=label + "_" + kind))
    return out
