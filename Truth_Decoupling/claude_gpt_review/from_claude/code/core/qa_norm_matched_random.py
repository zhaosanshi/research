"""Fixed random orientation, matched to the target's actual stored write norm.

This null borrows only the learned direction's scalar write size at the SAME
current state/source. It is not a wholly information-free random source swap.
Natural trajectories differ; matching is conditional on each observed state.
"""
import contextlib
import torch
import torch.nn.functional as F
from . import qa_source_interchange as source

PROTOCOL = 'random_orientation_stored_write_norm_match_v1'


def fixed_random_direction(hidden_size, seed, layer):
    g = torch.Generator(device='cpu').manual_seed(int(seed) + int(layer))
    return F.normalize(torch.randn(hidden_size, generator=g), dim=0)


def matched_random_write(before, donor, target_direction, random_direction):
    target, _ = source.interchange(before, donor, target_direction)
    b = before.float()
    v = source.unit_direction(random_direction, b.shape[-1]).to(b.device)
    sign = torch.sign((donor.detach().to(b.device).float() - b) @ v)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    orientation = sign[:, None] * v
    target_norm = (target.float() - b).norm(dim=-1)
    active = target_norm > 0

    def candidate(amplitude):
        stored = (b + amplitude[:, None] * orientation).to(before.dtype)
        return stored, (stored.float() - b).norm(dim=-1)

    low = torch.zeros_like(target_norm)
    high = torch.where(active, 2 * target_norm, torch.zeros_like(target_norm))
    _, high_norm = candidate(high)
    for _ in range(12):
        grow = active & (high_norm < target_norm)
        if not bool(grow.any()): break
        high = torch.where(grow, high * 2, high)
        _, high_norm = candidate(high)
    if bool((high_norm < target_norm).any()):
        raise RuntimeError('could not bracket requested actual random write size')
    for _ in range(28):
        middle = (low + high) / 2
        _, magnitude = candidate(middle)
        below = magnitude < target_norm
        low = torch.where(below, middle, low)
        high = torch.where(below, high, middle)
    lv, ln = candidate(low); hv, hn = candidate(high)
    choose_low = (ln-target_norm).abs() <= (hn-target_norm).abs()
    stored = torch.where(choose_low[:, None], lv, hv)
    norm = (stored.float()-b).norm(dim=-1)
    error = (norm-target_norm).abs()
    # Predeclared BF16 quantization allowance, not chosen from outcomes.
    tolerance = .01 * target_norm + .0001 * b.norm(dim=-1)
    if not bool(torch.isfinite(stored).all()) or bool((error > tolerance+1e-7).any()):
        raise RuntimeError('stored norm mismatch exceeds declared cast tolerance')
    audit = []
    for i in range(len(b)):
        audit.append({'target_stored_delta_norm': float(target_norm[i]),
            'random_stored_delta_norm': float(norm[i]), 'absolute_norm_error': float(error[i]),
            'cast_match_tolerance': float(tolerance[i]), 'within_tolerance': True,
            'h_norm': float(b[i].norm()),
            'random_relative_write': float(norm[i]/b[i].norm().clamp_min(1e-12)),
            'target_relative_write_same_state': float(target_norm[i]/b[i].norm().clamp_min(1e-12)),
            'cast_lower_bracket_norm': float(ln[i]), 'cast_upper_bracket_norm': float(hn[i]),
            'fixed_random_orientation': True, 'learned_direction_used_for_size_only': True,
            'fixed_random_axis': True, 'source_gap_projection_sign': float(sign[i]),
            'sign_rule': 'sign of donor-minus-current-state projected onto the fixed random axis; never the learned sign',
            'is_unscaled_source_interchange': False, 'residual_dtype': str(before.dtype)})
    return stored, audit


@contextlib.contextmanager
def matched_write_hook(bundle, layer, start, donor, random_direction, target_direction):
    state = {'calls': 0, 'writes': []}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.ndim != 3 or h.shape[0] != 1 or h.shape[1]-start != len(donor):
            raise ValueError('norm-control predictor positions differ')
        value, audit = matched_random_write(h[0, start:], donor, target_direction, random_direction)
        changed = h.clone(); changed[0, start:] = value
        state['calls'] += 1
        state['writes'] = [{**a, 'physical_base_position': start+i} for i,a in enumerate(audit)]
        return (changed,)+tuple(output[1:]) if isinstance(output,tuple) else changed
    handle = bundle.layer(layer).register_forward_hook(hook)
    try: yield state
    finally: handle.remove()


class NormMatchedRandomDecoder(source.SourceInterchangeDecoder):
    def __init__(self, bundle, base_prompt_ids, source_prompt_ids, *, layer, target_direction,
                 random_direction, use_cache=True):
        super().__init__(bundle, base_prompt_ids, source_prompt_ids, mode='source_interchange',
                         layer=layer, direction=random_direction, use_cache=use_cache)
        self.target_direction = source.unit_direction(target_direction, bundle.hidden_size)

    @torch.no_grad()
    def next_logits(self):
        original = source.write_base
        def replacement(bundle, layer, start, donor, direction):
            return matched_write_hook(bundle, layer, start, donor, direction, self.target_direction)
        # Source capture precedes the parent installing this base-only hook.
        # One process, one decoder, no concurrent model calls.
        source.write_base = replacement
        try: logits, trace = super().next_logits()
        finally: source.write_base = original
        trace['control_protocol'] = PROTOCOL
        return logits, trace


def generate(bundle, base_prompt_ids, source_prompt_ids, *, layer, target_direction,
             random_direction, max_new_tokens=48):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('positive generation budget required')
    stops = bundle.model.generation_config.eos_token_id
    stops = set(stops if isinstance(stops,(list,tuple)) else [stops])
    if not stops or any(type(x) is not int or x < 0 for x in stops):
        raise ValueError('valid native stop IDs required')
    decoder = NormMatchedRandomDecoder(bundle, base_prompt_ids, source_prompt_ids, layer=layer,
        target_direction=target_direction, random_direction=random_direction)
    finished = False
    for _ in range(max_new_tokens):
        logits, _ = decoder.next_logits()
        token = int(logits.argmax()); decoder.accept(token)
        if token in stops:
            finished = True; break
    return {'token_ids': decoder.history, 'output': bundle.tokenizer.decode(decoder.history,skip_special_tokens=True),
            'finished': finished, 'trace': decoder.traces}
