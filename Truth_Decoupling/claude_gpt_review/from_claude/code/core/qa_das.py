"""QA-DAS v1: answer-end interchange, supervised by later self-check labels.

Only the rank-one subspace is trained. The frozen causal LM receives an
unchanged, explicitly tokenized question/answer prefix and self-check suffix.
Each direction minimizes mean full-vocabulary CE over ALL its label tokens;
the two directions receive equal weight before one optimizer step per pair.
This objective is distinct from the older source-distribution KL objective.
"""
import hashlib
import inspect
import json
import math

import torch
import torch.nn.functional as F

from .directions import OrthoBasis
from .io_utils import set_seed
from .model_io import forward_logits
from .qa_context import encode_answer, _token_ids

PROTOCOL = "qa_das_ptrue_ce_bidir_v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _inputs(ids, device):
    values = torch.tensor([ids], dtype=torch.long, device=device)
    return {"input_ids": values, "attention_mask": torch.ones_like(values)}


def encode_selfcheck(bundle, system, question, answer, user_template, suffix,
                     label_true=" True", label_false=" False"):
    """Keep encode_answer IDs intact; append separately encoded suffix IDs.

    Label IDs are the increment of tokenizing suffix+label relative to suffix,
    with exact prefix preservation checked. No label or EOS is in `inputs`.
    `label_predictor` is the position predicting the FIRST label token, not the
    position at which that label token would be appended.
    """
    if not isinstance(suffix, str) or not suffix.strip():
        raise ValueError("self-check suffix must be nonempty")
    encoded = encode_answer(bundle, system, question, answer, user_template)
    suffix_ids = _token_ids(bundle.tokenizer, suffix)
    if not suffix_ids:
        raise ValueError("self-check suffix has no tokens")
    special = set(getattr(bundle.tokenizer, "all_special_ids", []) or [])
    if special.intersection(suffix_ids):
        raise ValueError("self-check suffix contains special token IDs")
    label_ids = {}
    labels = {"true": label_true, "false": label_false}
    for name, label in labels.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("self-check labels must be nonempty strings")
        combined = _token_ids(bundle.tokenizer, suffix + label)
        if combined[:len(suffix_ids)] != suffix_ids:
            raise ValueError("suffix/label boundary retokenizes the suffix (%s)" % name)
        ids = combined[len(suffix_ids):]
        if not ids or special.intersection(ids):
            raise ValueError("%s label is empty or contains special token IDs" % name)
        label_ids[name] = ids
    if label_ids["true"] == label_ids["false"]:
        raise ValueError("True and False labels tokenize identically")
    ids = encoded["full_ids"] + suffix_ids
    end = len(encoded["full_ids"]) - 1
    return {
        "inputs": _inputs(ids, bundle.device), "input_ids": ids,
        "answer_end": end, "label_predictor": len(ids) - 1,
        "label_ids": label_ids, "suffix_ids": suffix_ids, "encoded": encoded,
        "text_audit": {
            "system": system, "question": question, "answer": answer,
            "user_template": user_template, "suffix": suffix, "labels": labels,
            "display_text": encoded["full_text"] + suffix,
            "token_construction": "encode_answer.full_ids + tokenize(suffix); label increment from suffix+label",
            "answer_prefix_ids_preserved": True, "labels_in_inputs": False,
            "appended_eos": False,
        },
    }


def prepare_selfcheck_pairs(bundle, cfg, records):
    """Encode already-cleaned E records without adding or removing questions."""
    pairs, seen = [], set()
    for record in records:
        pid = record.get("pair_id")
        if pid is None or not str(pid).strip() or str(pid) in seen:
            raise ValueError("self-check records require unique, nonempty pair_id values")
        seen.add(str(pid))
        pair = {"pair_id": pid}
        for side, key in (("positive", "positive_answer"), ("negative", "negative_answer")):
            try:
                pair[side] = encode_selfcheck(
                    bundle, cfg["qa_system"], record["question"], record[key],
                    cfg["qa_user"], cfg["selfcheck_suffix"],
                    cfg.get("selfcheck_label_true", " True"),
                    cfg.get("selfcheck_label_false", " False"))
            except (KeyError, ValueError) as exc:
                raise ValueError("self-check pair %r, %s: %s" % (pid, side, exc)) from exc
        pairs.append(pair)
    return pairs


def _validate(bundle, pairs, layer, training=False):
    if type(layer) is not int or not 0 <= layer < bundle.num_layers:
        raise ValueError("layer must index a loaded decoder block")
    if training and layer == bundle.num_layers - 1:
        raise ValueError("last block OUTPUT at answer_end has no path to later label logits; structural null, not a training result")
    if not pairs:
        raise ValueError("QA-DAS requires at least one pair")
    ids = [str(p["pair_id"]) for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate self-check pair IDs")
    labels = pairs[0]["positive"]["label_ids"]
    for pair in pairs:
        for side in ("positive", "negative"):
            item = pair[side]
            x, mask = item["inputs"]["input_ids"], item["inputs"]["attention_mask"]
            if x.ndim != 2 or x.shape[0] != 1 or mask.shape != x.shape or not bool(mask.all()):
                raise ValueError("self-check inputs must be single unpadded sequences")
            if not 0 <= item["answer_end"] < item["label_predictor"] == x.shape[1] - 1:
                raise ValueError("answer_end must precede the final self-check label predictor")
            if item["label_ids"] != labels:
                raise ValueError("all pairs must use the same tokenized self-check labels")


def _forward_last(bundle, inputs, grad=False):
    # The shared forward helper optionally avoids a full sequence vocabulary
    # projection; any externally installed high-precision output context stays
    # active. Compatibility with the older helper is deliberate.
    kwargs = {"grad": grad}
    if "logits_to_keep" in inspect.signature(forward_logits).parameters:
        kwargs["logits_to_keep"] = 1
    return forward_logits(bundle, inputs, **kwargs)[:, -1, :]


def _source_states(bundle, pairs, layer, source_activations=None):
    result = {}
    for side in ("positive", "negative"):
        if source_activations is not None:
            if side not in source_activations:
                raise ValueError("source cache needs positive and negative arrays")
            states = torch.as_tensor(source_activations[side]).detach().float().cpu().clone()
        else:
            captured = []
            for pair in pairs:
                item, holder = pair[side], {}
                def capture(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    holder["h"] = h[:, item["answer_end"], :].detach().float().cpu().clone()
                handle = bundle.layer(layer).register_forward_hook(capture)
                try:
                    _forward_last(bundle, item["inputs"], grad=False)
                finally:
                    handle.remove()
                captured.append(holder["h"])
            states = torch.cat(captured, dim=0)
        if states.shape != (len(pairs), bundle.hidden_size) or not bool(torch.isfinite(states).all()):
            raise ValueError("%s source cache must be finite [n_pairs, hidden_size]" % side)
        result[side] = states
    return result


def _write_hook(W_fn, position, source=None, center=None, alpha=None,
                audit=None, apply=True):
    """Optionally observe the single-example rank1 edit without retaining a graph.

    `kept_delta_norm` measures the actual output after conversion back to the
    residual dtype, whereas `intended_delta_norm` measures the float32 edit.
    With apply=False the hook only observes and returns the original output.
    """
    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        before = h[:, position, :].float()
        W = W_fn().to(before.device)
        donor, c = None, None
        if not apply:
            delta = torch.zeros_like(before)
        elif source is not None:
            # Detach the frozen source STATE, never its projection through W.
            donor = source.detach().to(before.device, torch.float32).reshape(1, -1)
            delta = ((donor - before) @ W) @ W.T
        else:
            c = torch.as_tensor(center, dtype=torch.float32, device=before.device).reshape(1, -1)
            delta = float(alpha) * (((before - c) @ W) @ W.T)
        if not apply:
            changed = h
        else:
            changed = h.clone()
            changed[:, position, :] = (before + delta).to(h.dtype)
        if audit is not None:
            with torch.no_grad():
                after = changed[:, position, :].detach().float()
                kept = after - before.detach()
                before_coordinate = float((before.detach() @ W.detach()).item())
                after_coordinate = float((after @ W.detach()).item())
                audit.append({
                    "operation": "baseline" if not apply else ("interchange" if donor is not None else "amplify"),
                    "residual_dtype": str(h.dtype),
                    "measurement": "kept = float32(actual dtype-cast output) - float32(original output)",
                    "h_norm": float(before.detach().norm()),
                    "intended_delta_norm": float(delta.detach().norm()),
                    "kept_delta_norm": float(kept.norm()),
                    "coordinate_before": before_coordinate,
                    "coordinate_after": after_coordinate,
                    "intended_coordinate_delta": float((delta.detach() @ W.detach()).item()),
                    "kept_coordinate_delta": after_coordinate - before_coordinate,
                    "source_coordinate": None if donor is None else float((donor @ W.detach()).item()),
                    "center_coordinate": None if c is None else float((c @ W.detach()).item()),
                })
        if not apply:
            return output
        return (changed,) + output[1:] if isinstance(output, tuple) else changed
    return hook


def _label_loss(bundle, item, target, layer, W_fn=None, source=None,
                center=None, alpha=None, grad=False, trace=None, audit_W_fn=None):
    """Mean full-vocabulary CE, with every token in the target label scored."""
    label = item["label_ids"][target]
    base = item["inputs"]
    losses = []
    for index, token in enumerate(label):
        prefix = torch.tensor([label[:index]], dtype=torch.long, device=base["input_ids"].device)
        inputs = {"input_ids": torch.cat([base["input_ids"], prefix], dim=1),
                  "attention_mask": torch.cat([base["attention_mask"], torch.ones_like(prefix)], dim=1)}
        handle = None
        write_audit = [] if trace is not None else None
        if W_fn is not None:
            handle = bundle.layer(layer).register_forward_hook(
                _write_hook(W_fn, item["answer_end"], source=source, center=center, alpha=alpha,
                            audit=write_audit))
        elif trace is not None and audit_W_fn is not None:
            handle = bundle.layer(layer).register_forward_hook(
                _write_hook(audit_W_fn, item["answer_end"], audit=write_audit, apply=False))
        try:
            logits = _forward_last(bundle, inputs, grad=grad).float()
            loss = F.cross_entropy(logits, torch.tensor([token], dtype=torch.long, device=logits.device))
        finally:
            if handle is not None:
                handle.remove()
        losses.append(loss)
        if trace is not None:
            if (W_fn is not None or audit_W_fn is not None) and len(write_audit) != 1:
                raise RuntimeError("self-check write audit requires exactly one block-output observation per label token")
            trace.append({"label": target, "label_token_index": index, "label_token_id": token,
                          "input_length": inputs["input_ids"].shape[1],
                          "base_answer_end": item["answer_end"],
                          "first_label_predictor": item["label_predictor"],
                          "current_label_predictor": inputs["input_ids"].shape[1] - 1,
                          **(write_audit[0] if write_audit else {})})
    return torch.stack(losses).mean(), len(label)


def _identity_metadata(bundle, pairs):
    ids = [p["pair_id"] for p in pairs]
    templates, inputs = [], []
    for pair in pairs:
        row = {"pair_id": pair["pair_id"]}
        for side in ("positive", "negative"):
            item, audit = pair[side], pair[side]["text_audit"]
            templates.append({k: audit[k] for k in ("system", "user_template", "suffix", "labels")})
            row[side] = {"ids": item["inputs"]["input_ids"].detach().cpu().tolist()[0],
                         "answer_end": item["answer_end"], "label_predictor": item["label_predictor"],
                         "label_ids": item["label_ids"]}
        inputs.append(row)
    if any(t != templates[0] for t in templates[1:]):
        raise ValueError("self-check pairs use different templates")
    return {"model_name": getattr(bundle, "name", (getattr(bundle, "cfg", {}) or {}).get("name", "model")),
            "model_config": getattr(bundle, "cfg", {}), "training_pair_ids": ids,
            "training_pair_ids_sha256": _hash(ids), "selfcheck_template": templates[0],
            "selfcheck_template_sha256": _hash(templates[0]), "selfcheck_inputs_sha256": _hash(inputs)}


def train_qa_das(bundle, pairs, layer, epochs=10, lr=.001, seed=42, clip=1,
                 source_activations=None, identity=None, log_fn=print):
    """Train rank1 with negative<-positive/True and positive<-negative/False.

    The label CE is averaged across its tokens, the directions are weighted
    0.5 each, and Adam updates once per pair. `losses` records online epoch
    means before each pair update, not a re-evaluation of the final checkpoint.
    Cached source rows must follow the exact order of `pairs` at this layer.
    """
    _validate(bundle, pairs, layer, training=True)
    if type(epochs) is not int or epochs < 1 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(clip) or clip <= 0:
        raise ValueError("epochs, learning rate and gradient clip must be positive")
    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    set_seed(seed)
    source = _source_states(bundle, pairs, layer, source_activations)
    metadata = _identity_metadata(bundle, pairs)
    metadata.update({"protocol": PROTOCOL, "estimator": "qa_das", "layer": layer, "rank": 1,
                     "n_pairs": len(pairs), "T": max(len(v) for v in pairs[0]["positive"]["label_ids"].values()),
                     "epochs": epochs, "lr": float(lr), "clip": float(clip), "optimizer": "Adam",
                     "loss_kind": "full_vocabulary_label_cross_entropy",
                     "label_reduction": "mean_over_all_label_tokens_per_direction; directions_weight_0.5_each",
                     "losses_measurement": "online_pre_update_epoch_mean", "positions": "answer_end",
                     "anchor": "answer_end", "bidirectional": True, "trim": 0.0, "weighted": False,
                     "label_ids": pairs[0]["positive"]["label_ids"], "structural_null": False})
    for key, value in (identity or {}).items():
        if key in {"W", "losses", "mean_gap", "gap_std", "seed", "position_trace", "per_seed"}:
            raise ValueError("identity must not override per-seed field %s" % key)
        if key in metadata and value != metadata[key]:
            raise ValueError("identity disagrees with actual training field %s" % key)
        metadata[key] = value
    basis = OrthoBasis(bundle.hidden_size, 1, seed=seed, device=bundle.device)
    optimizer = torch.optim.Adam(basis.parameters(), lr=lr)
    losses, norms, traces = [], [], []
    zero_grad_steps = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, pair in enumerate(pairs):
            optimizer.zero_grad()
            values = []
            for base_side, source_side, target in (("negative", "positive", "true"),
                                                   ("positive", "negative", "false")):
                trace = [] if epoch == 0 and i == 0 else None
                loss, count = _label_loss(bundle, pair[base_side], target, layer, basis,
                                          source=source[source_side][i], grad=True, trace=trace)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite QA-DAS loss at pair %r" % pair["pair_id"])
                (.5 * loss).backward()
                values.append(float(loss.detach()))
                if trace is not None:
                    for row in trace:
                        row.update({"pair_id": pair["pair_id"], "direction": base_side + "_to_" + source_side,
                                    "source_answer_end": pair[source_side]["answer_end"],
                                    "source_input_length": pair[source_side]["inputs"]["input_ids"].shape[1]})
                    traces.extend(trace)
            gradient = basis.raw.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("missing/non-finite QA-DAS direction gradient")
            norm = float(gradient.norm())
            norms.append(norm)
            zero_grad_steps += norm == 0.0
            torch.nn.utils.clip_grad_norm_(basis.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            if not bool(torch.isfinite(basis.raw).all()):
                raise RuntimeError("non-finite QA-DAS parameters after optimizer step")
            epoch_loss += .5 * sum(values)
        losses.append(epoch_loss / len(pairs))
        if log_fn:
            log_fn("[QA-DAS L%d s%d] epoch %d/%d mean CE %.6f" % (layer, seed, epoch + 1, epochs, losses[-1]))
    if not any(value > 0 for value in norms):
        raise RuntimeError("QA-DAS had zero direction gradient on every update; do not report an untrained direction")
    W = basis.detached()
    gaps = (source["positive"] - source["negative"]) @ W
    return {**metadata, "W": W, "seed": int(seed), "losses": losses,
            "mean_gap": gaps.mean(0).tolist(), "gap_std": gaps.std(0, unbiased=False).tolist(),
            "position_trace": traces,
            "per_seed": {"source_cache_supplied": source_activations is not None, "optimizer_updates": len(norms),
                         "gradient_norm_min": min(norms), "gradient_norm_max": max(norms),
                         "gradient_norm_mean": sum(norms) / len(norms), "zero_gradient_updates": zero_grad_steps}}


def evaluate_selfcheck(bundle, pairs, W, layer, center, alpha):
    """Diagnostic only; counterfactual matching is not real-question accuracy.

    Decisions compare SUMMED full-label log probabilities. Training uses mean
    token CE so the two target labels receive equal directional weight even if
    their tokenizer lengths differ. Both sums and means are reported here.
    """
    _validate(bundle, pairs, layer)
    W = torch.as_tensor(W, dtype=torch.float32).detach()
    if W.ndim == 1:
        W = W[:, None]
    if W.shape != (bundle.hidden_size, 1) or not bool(torch.isfinite(W).all()) or abs(float(W.norm()) - 1) > 1e-4:
        raise ValueError("diagnostic W must be a finite unit rank1 direction")
    center = torch.as_tensor(center, dtype=torch.float32).detach()
    if center.shape != (bundle.hidden_size,) or not bool(torch.isfinite(center).all()) or not math.isfinite(alpha):
        raise ValueError("diagnostic center/alpha must be finite with matching hidden size")
    bundle.model.eval()
    source = _source_states(bundle, pairs, layer)
    output = {"protocol": PROTOCOL, "eligible_for_selection": False, "layer": layer, "alpha": float(alpha),
              "structural_null": layer == bundle.num_layers - 1,
              "readout": "sum conditional log probabilities of every label token; ties have no prediction",
              "pairs": []}
    for i, pair in enumerate(pairs):
        record = {"pair_id": pair["pair_id"], "eligible_for_selection": False}
        for base_side, source_side, base_label, source_label in (("negative", "positive", "false", "true"),
                                                               ("positive", "negative", "true", "false")):
            conditions = {}
            for condition in ("baseline", "self_swap", "cross_swap", "amplify"):
                kwargs = {}
                if condition != "baseline":
                    kwargs["W_fn"] = lambda: W
                if condition in ("self_swap", "cross_swap"):
                    kwargs["source"] = source[base_side if condition == "self_swap" else source_side][i]
                if condition == "amplify":
                    kwargs.update(center=center, alpha=alpha)
                means, sums, write_audit = {}, {}, {}
                for label in ("true", "false"):
                    write_audit[label] = []
                    loss, count = _label_loss(bundle, pair[base_side], label, layer, **kwargs,
                                               trace=write_audit[label], audit_W_fn=lambda: W)
                    means[label] = -float(loss.detach())
                    sums[label] = means[label] * count
                if not all(math.isfinite(v) for v in sums.values()):
                    raise RuntimeError("non-finite self-check diagnostic scores")
                pred = "true" if sums["true"] > sums["false"] else ("false" if sums["false"] > sums["true"] else None)
                expected = source_label if condition == "cross_swap" else base_label
                conditions[condition] = {"logp_sum": sums, "logp_mean": means, "predicted_label": pred,
                                         "write_audit": write_audit,
                                         "expected_label": expected, "correct": pred == expected,
                                         "correct_for_base": pred == base_label, "matches_source_label": pred == source_label}
            record[base_side + "_to_" + source_side] = {
                "base_answer_end": pair[base_side]["answer_end"],
                "source_answer_end": pair[source_side]["answer_end"],
                "label_predictor": pair[base_side]["label_predictor"], **conditions}
        output["pairs"].append(record)
    return output
