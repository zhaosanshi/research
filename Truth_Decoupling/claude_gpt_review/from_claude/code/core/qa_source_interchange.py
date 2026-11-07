"""Inference-only, question-conditioned rank-one source interchange.

The source is the same frozen model under a fixed instruction and the exact
already-generated token IDs. There is no reference answer, optimizer, fixed
activation centre or extrapolation coefficient in this module. The selected
coordinate is replaced: h' = h + w * ((s - h) @ w), with unit w.

Cached decoding keeps independent source/base KV states. The uncached reference
recomputes *all* previous predictor interventions, not only the newest one.
That distinction matters: earlier writes affect later attention in the base.
"""
import contextlib
import hashlib
import inspect
import json

import torch
import torch.nn.functional as F


PROTOCOL = 'same_prefix_source_interchange_v1'
MODES = ('baseline', 'source_only', 'equal_logit_mix', 'source_interchange')


def token_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def unit_direction(w, hidden):
    w = torch.as_tensor(w).detach().float().reshape(-1)
    if w.shape != (hidden,) or not bool(torch.isfinite(w).all()) or abs(float(w.norm()) - 1.) > 1e-4:
        raise ValueError('source interchange requires a finite unit hidden-size vector')
    return F.normalize(w, dim=0)


def interchange(base, source, direction):
    """Return the cast write and separate ideal-FP32 versus stored-dtype audits."""
    if base.shape != source.shape or base.ndim != 2 or not base.is_floating_point():
        raise ValueError('base and source must be aligned [predictor, hidden] matrices')
    w = unit_direction(direction, base.shape[-1]).to(base.device)
    b, s = base.float(), source.detach().to(base.device).float()
    if not bool(torch.isfinite(b).all() and torch.isfinite(s).all()):
        raise ValueError('nonfinite source/base activation')
    gap = s - b
    delta = (gap @ w).unsqueeze(-1) * w
    ideal = b + delta
    stored = ideal.to(base.dtype)
    kept = stored.float() - b
    quantization = stored.float() - ideal
    ideal_coord_error = (ideal @ w - s @ w).abs()
    ideal_orthogonal_error = (delta - (delta @ w).unsqueeze(-1) * w).norm(dim=-1)
    stored_coord_error = (stored.float() @ w - s @ w).abs()
    stored_orthogonal_error = (kept - (kept @ w).unsqueeze(-1) * w).norm(dim=-1)
    # No false hard norm guarantee after a BF16 cast. Bound ideal errors using
    # FP32 arithmetic, then record the actual cast error separately.
    tol = 32 * torch.finfo(torch.float32).eps * (1 + b.norm(dim=-1) + s.norm(dim=-1))
    if bool((ideal_coord_error > tol).any() or (ideal_orthogonal_error > tol).any()
            or (delta.norm(dim=-1) > gap.norm(dim=-1) + tol).any()):
        raise RuntimeError('rank-one coordinate exchange invariant failed before dtype cast')
    if not bool(torch.isfinite(stored).all()):
        raise RuntimeError('interchange write overflowed residual dtype')
    audit = []
    for i in range(len(base)):
        audit.append({'source_base_distance': float(gap[i].norm()),
                      'ideal_delta_norm': float(delta[i].norm()), 'stored_delta_norm': float(kept[i].norm()),
                      'base_norm': float(b[i].norm()),
                      'stored_delta_over_base': float(kept[i].norm() / b[i].norm().clamp_min(1e-12)),
                      'ideal_coordinate_error': float(ideal_coord_error[i]),
                      'stored_coordinate_error': float(stored_coord_error[i]),
                      'ideal_orthogonal_error': float(ideal_orthogonal_error[i]),
                      'stored_orthogonal_error': float(stored_orthogonal_error[i]),
                      'cast_error_norm': float(quantization[i].norm()),
                      'float32_tolerance': float(tol[i]), 'residual_dtype': str(base.dtype)})
    return stored, audit


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextlib.contextmanager
def capture_source(bundle, layer, start):
    """Capture before installing any base intervention hook."""
    state = {'calls': 0}
    def hook(module, inputs, output):
        h = _hidden(output)
        if h.ndim != 3 or h.shape[0] != 1 or not 0 <= start < h.shape[1]:
            raise ValueError('source predictor positions do not match the current prefix')
        state['calls'] += 1
        state['hidden'] = h[0, start:].detach().clone()
    handle = bundle.layer(layer).register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


@contextlib.contextmanager
def write_base(bundle, layer, start, source, direction):
    state = {'calls': 0, 'writes': []}
    def hook(module, inputs, output):
        h = _hidden(output)
        if h.ndim != 3 or h.shape[0] != 1 or h.shape[1] - start != len(source):
            raise ValueError('base and source predictor indices are misaligned')
        state['calls'] += 1
        updated = h.clone()
        replacement, audit = interchange(h[0, start:], source, direction)
        updated[0, start:] = replacement
        state['writes'] = [{**a, 'physical_base_position': start + i} for i, a in enumerate(audit)]
        return (updated,) + tuple(output[1:]) if isinstance(output, tuple) else updated
    handle = bundle.layer(layer).register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def _cache_tensors(cache):
    if torch.is_tensor(cache):
        return [cache]
    if isinstance(cache, (list, tuple)):
        return [t for part in cache for t in _cache_tensors(part)]
    if isinstance(cache, dict):
        return [t for part in cache.values() for t in _cache_tensors(part)]
    # Transformers DynamicCache stores per-layer keys and values. Avoid the
    # deprecated key_cache property, and never mutate/cache-copy these objects.
    if hasattr(cache, 'layers'):
        return [t for layer in cache.layers for name in ('keys', 'values')
                for t in _cache_tensors(getattr(layer, name, None))]
    return []


class SourceInterchangeDecoder:
    """A single-question decoder; both streams consume exactly accepted IDs."""
    def __init__(self, bundle, base_prompt_ids, source_prompt_ids, *, mode,
                 layer=None, direction=None, use_cache=True):
        if mode not in MODES:
            raise ValueError('unknown source-interchange mode')
        self.bundle, self.mode, self.use_cache = bundle, mode, bool(use_cache)
        self.base_ids, self.source_ids = list(base_prompt_ids), list(source_prompt_ids)
        for ids in (self.base_ids, self.source_ids):
            if not ids or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError('each prompt must contain valid token IDs')
        if mode == 'source_interchange':
            if type(layer) is not int or not 0 <= layer < bundle.num_layers:
                raise ValueError('interchange needs an explicit valid layer')
            direction = unit_direction(direction, bundle.hidden_size)
        self.layer, self.direction = layer, direction
        self.base_cache = self.source_cache = None
        self.history, self.traces = [], []
        self.pending = False
        self.vocab_size = None

    def _forward(self, prompt, cache):
        ids = prompt + self.history
        active = [self.history[-1]] if self.use_cache and cache is not None else ids
        start = len(ids) - len(active)
        inputs = torch.tensor([active], dtype=torch.long, device=self.bundle.device)
        mask = torch.ones((1, len(ids)), dtype=torch.long, device=self.bundle.device)
        position = torch.arange(start, len(ids), device=self.bundle.device).unsqueeze(0)
        extra = {}
        parameter = inspect.signature(self.bundle.model.forward).parameters.get('logits_to_keep')
        if parameter is not None and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            extra['logits_to_keep'] = 1
        output = self.bundle.model(input_ids=inputs, attention_mask=mask, position_ids=position,
                                   past_key_values=cache if self.use_cache else None,
                                   use_cache=self.use_cache, **extra)
        new_cache = output.past_key_values if self.use_cache else None
        if self.use_cache and new_cache is None:
            raise RuntimeError('model did not return a requested KV cache')
        logits = output.logits[0, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError('nonfinite next-token logits')
        return logits, new_cache

    @torch.no_grad()
    def next_logits(self):
        if self.pending:
            raise RuntimeError('accept exactly one chosen token before the next forward')
        trace = {'step': len(self.history), 'accepted_prefix_ids': list(self.history),
                 'accepted_prefix_sha256': token_digest(self.history),
                 'base_prompt_tokens': len(self.base_ids), 'source_prompt_tokens': len(self.source_ids),
                 'source_predictor_logical_position': len(self.source_ids) - 1 + len(self.history),
                 'base_predictor_logical_position': len(self.base_ids) - 1 + len(self.history),
                 'uses_reference_answer': False, 'use_cache': self.use_cache, 'writes': []}
        source_logits = source_states = None
        if self.mode != 'baseline':
            if self.mode == 'source_interchange':
                start = 0 if self.use_cache and self.source_cache is not None else len(self.source_ids) - 1
                with capture_source(self.bundle, self.layer, start) as captured:
                    source_logits, self.source_cache = self._forward(self.source_ids, self.source_cache)
                if captured['calls'] != 1:
                    raise RuntimeError('source layer was not called exactly once')
                source_states = captured['hidden']
                expected = 1 if self.use_cache else len(self.history) + 1
                if len(source_states) != expected:
                    raise RuntimeError('source contains a future token or misses a predictor')
            else:
                source_logits, self.source_cache = self._forward(self.source_ids, self.source_cache)
        if self.mode == 'source_only':
            logits = source_logits
        else:
            if self.mode == 'source_interchange':
                start = 0 if self.use_cache and self.base_cache is not None else len(self.base_ids) - 1
                with write_base(self.bundle, self.layer, start, source_states, self.direction) as written:
                    logits, self.base_cache = self._forward(self.base_ids, self.base_cache)
                if written['calls'] != 1:
                    raise RuntimeError('base layer was not called exactly once')
                trace['writes'] = written['writes']
                trace['source_predictors_captured'] = len(source_states)
            else:
                logits, self.base_cache = self._forward(self.base_ids, self.base_cache)
                if self.mode == 'equal_logit_mix':
                    logits = (logits + source_logits) * .5
        if self.use_cache and self.base_cache is not None and self.source_cache is not None:
            a = {t.untyped_storage().data_ptr() for t in _cache_tensors(self.base_cache)}
            b = {t.untyped_storage().data_ptr() for t in _cache_tensors(self.source_cache)}
            if not a or not b:
                raise RuntimeError('unknown or empty KV cache representation; storage isolation cannot be verified')
            if self.base_cache is self.source_cache or a & b:
                raise RuntimeError('base/source KV streams share mutable storage')
            trace['cache_storage_disjoint'] = True
            trace['base_cache_tensors'] = len(a); trace['source_cache_tensors'] = len(b)
        trace['next_top1_id'] = int(logits.argmax())
        trace['next_top1_probability'] = float(logits.softmax(-1).max())
        self.traces.append(trace)
        self.vocab_size, self.pending = logits.numel(), True
        return logits, trace

    def accept(self, token_id):
        if not self.pending or type(token_id) is not int or not 0 <= token_id < self.vocab_size:
            raise ValueError('accept needs one valid token following a next_logits call')
        self.history.append(token_id)
        self.pending = False


def greedy_source_generate(bundle, base_prompt_ids, source_prompt_ids, *, mode, layer=None,
                           direction=None, max_new_tokens=48, use_cache=True):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    stops = getattr(bundle.model.generation_config, 'eos_token_id', None)
    stops = set(stops if isinstance(stops, (tuple, list)) else [stops])
    if not stops or any(type(t) is not int or t < 0 for t in stops):
        raise ValueError('model must declare valid generation stop IDs')
    decoder = SourceInterchangeDecoder(bundle, base_prompt_ids, source_prompt_ids, mode=mode,
                                        layer=layer, direction=direction, use_cache=use_cache)
    finished = False
    for _ in range(max_new_tokens):
        logits, _ = decoder.next_logits()
        token = int(logits.argmax())
        decoder.accept(token)
        if token in stops:
            finished = True
            break
    return {'token_ids': decoder.history, 'output': bundle.tokenizer.decode(decoder.history, skip_special_tokens=True),
            'finished': finished, 'trace': decoder.traces}
