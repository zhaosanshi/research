"""Fixed source E plus prompt-end C, confined to the base forward stream.

Physical layer depth determines execution order. Different C layers also have
different estimated vectors/scales: this is not a pure permutation of two
identical operators. The source never receives the C intervention.
"""
import contextlib
import math

import torch

from .qa_source_interchange import SourceInterchangeDecoder, _hidden


PROTOCOL = 'source_e_commitment_layer_order_v1'


def stored_c_addition(base, direction, coefficient, alpha):
    """Exactly the main C add operator: FP32 delta/addition, native-dtype store."""
    w = direction.to(base.device).float()
    scale = torch.tensor([coefficient],device=base.device,dtype=torch.float32)
    delta = float(alpha) * (w[:,None] @ scale)
    return (base.float()+delta).to(base.dtype), delta


def relative_write_profile(states, direction, coefficient):
    norms = states.float().norm(dim=-1)
    if states.ndim != 2 or not bool(torch.isfinite(states).all()) or bool((norms<=0).any()):
        raise ValueError('calibration requires finite nonzero unmodified prompt states')
    values = []
    for alpha in (-1.,1.):
        stored,_ = stored_c_addition(states,direction,coefficient,alpha)
        values.append((stored.float()-states.float()).norm(dim=-1)/norms)
    stacked = torch.stack(values)
    return {'median':float(torch.quantile(stacked.flatten(),.5)),
            'negative':stacked[0].cpu().tolist(),'positive':stacked[1].cpu().tolist()}


def calibrate_coefficients(states, directions, original_scales):
    """Only unmodified training h enters; never model outputs or semantic labels."""
    if set(states) != {'before','after'} or set(directions)!=set(states) or set(original_scales)!=set(states):
        raise ValueError('calibration needs exactly the two predeclared C layers')
    sizes={tuple(v.shape) for v in states.values()}
    if len(sizes)!=1 or len(next(iter(sizes)))!=2 or next(iter(sizes))[0]<2:
        raise ValueError('calibration requires the same multi-row cohort at both C layers')
    n,hidden=next(iter(sizes))
    for side in states:
        w=directions[side]
        if (w.shape!=(hidden,) or not bool(torch.isfinite(w).all()) or abs(float(w.float().norm())-1)>1e-4
                or not math.isfinite(float(original_scales[side])) or float(original_scales[side])<=0):
            raise ValueError('calibration needs unit C directions and finite positive original scales')
    original = {side:relative_write_profile(states[side],directions[side],original_scales[side]) for side in states}
    rho = min(v['median'] for v in original.values())
    if not math.isfinite(rho) or rho<=0: raise ValueError('original C dose has no measurable positive target')
    result = {}
    for side in ('before','after'):
        low,high = 0.,float(original_scales[side]);best=(abs(original[side]['median']-rho),high,original[side])
        search=[]
        for _ in range(48):
            mid=(low+high)/2.;profile=relative_write_profile(states[side],directions[side],mid)
            error=abs(profile['median']-rho);search.append({'coefficient':mid,'median':profile['median']})
            if error<best[0]: best=(error,mid,profile)
            if profile['median']<rho:low=mid
            else:high=mid
        if best[0] > .005*rho+1e-7: raise RuntimeError('stored-C training median cannot match the fixed target tolerance')
        result[side]={'original_coefficient':float(original_scales[side]),'coefficient':best[1],
                      'original_profile':original[side],'calibrated_profile':best[2],
                      'absolute_median_error':best[0],'search':search}
    return {'protocol':'train_prompt_actual_relative_C_norm_v1','rho':rho,
            'n_training_states':n,'n_signed_measurements_per_layer':2*n,
            'target_rule':'minimum across layers of median over the same training-state cohort and both signs',
            'median_rule':'quantile 0.5 with linear interpolation over both signs and all training states',
            'tolerance':.005*rho+1e-7,'layers':result}


class SourceCommitmentDecoder(SourceInterchangeDecoder):
    def __init__(self, bundle, base_prompt_ids, source_prompt_ids, *, e_layer=None,
                 e_direction=None, c_layer=None, c_direction=None, c_scale=None,
                 c_alpha=None, use_cache=True):
        mode = 'baseline' if e_direction is None else 'source_interchange'
        super().__init__(bundle, base_prompt_ids, source_prompt_ids, mode=mode,
                         layer=e_layer, direction=e_direction, use_cache=use_cache)
        if c_direction is None:
            if any(x is not None for x in (c_layer, c_scale, c_alpha)):
                raise ValueError('absent C requires absent C parameters')
            self.c_direction = None
        else:
            c = torch.as_tensor(c_direction).detach().float().reshape(-1).clone()
            if (type(c_layer) is not int or not 0 <= c_layer < bundle.num_layers
                    or c.shape != (bundle.hidden_size,) or not bool(torch.isfinite(c).all())
                    or abs(float(c.norm()) - 1.) > 1e-4
                    or not math.isfinite(float(c_scale)) or float(c_scale) <= 0
                    or c_alpha not in (-1., 1.) or c_layer == e_layer):
                raise ValueError('C needs a different valid layer, unit vector, positive scale and fixed +/-1 dose')
            self.c_direction = c
        self.c_layer, self.c_scale, self.c_alpha = c_layer, c_scale, c_alpha
        self._base_events, self._c_writes = [], []
        self._source_passes, self._base_passes = 0, 0

    def _forward(self, prompt, cache):
        # Identity, not token-value equality: equal-length/equal-value source
        # and base prompts are still separate streams on the same model.
        if prompt is self.source_ids:
            self._source_passes += 1
            return super()._forward(prompt, cache)
        if prompt is not self.base_ids:
            raise RuntimeError('unknown forward-stream object')
        self._base_passes += 1
        phase = 'prefill' if cache is None else 'decode'
        replay_c = self.c_direction is not None and (not self.use_cache or cache is None)
        logical = len(self.base_ids) - 1
        expected_length = len(self.base_ids) + len(self.history)
        if self.use_cache and cache is not None:
            expected_length = 1

        def c_hook(module, inputs, output):
            h = _hidden(output)
            if h.shape != (1, expected_length, self.bundle.hidden_size) or logical >= h.shape[1]:
                raise RuntimeError('C prompt-end position does not match base forward')
            before = h[0, logical].float()
            w = self.c_direction.to(h.device)
            stored,delta = stored_c_addition(h[0,logical],w,self.c_scale,self.c_alpha)
            actual = stored.float() - before
            if not bool(torch.isfinite(stored).all()):
                raise RuntimeError('C write overflowed residual dtype')
            out = h.clone(); out[0, logical] = stored
            self._base_events.append({'operator': 'C', 'layer': self.c_layer,
                'phase': phase, 'stream': 'base', 'physical_position': logical,
                'logical_position': logical, 'replayed_prefill': not self.use_cache and bool(self.history)})
            self._c_writes.append({'layer': self.c_layer, 'alpha': float(self.c_alpha),
                'scale': float(self.c_scale), 'base_norm': float(before.norm()),
                'ideal_delta_norm': float(delta.norm()), 'stored_delta_norm': float(actual.norm()),
                'stored_delta_over_base': float(actual.norm() / before.norm().clamp_min(1e-12)),
                'projection_before': float(before @ w), 'projection_after': float(stored.float() @ w),
                'cast_error_norm': float((stored.float() - before - delta).norm()),
                'residual_dtype': str(h.dtype), 'physical_position': logical,
                'logical_position': logical, 'source_intervened': False})
            return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out

        def e_observer(module, inputs, output):
            self._base_events.append({'operator': 'E', 'layer': self.layer,
                'phase': phase, 'stream': 'base',
                'physical_first_position': 0 if self.use_cache and cache is not None else logical,
                'logical_last_position': logical + len(self.history)})

        with contextlib.ExitStack() as stack:
            if self.mode == 'source_interchange':
                handle = self.bundle.layer(self.layer).register_forward_hook(e_observer)
                stack.callback(handle.remove)
            if replay_c:
                handle = self.bundle.layer(self.c_layer).register_forward_hook(c_hook)
                stack.callback(handle.remove)
            return super()._forward(prompt, cache)

    @torch.no_grad()
    def next_logits(self):
        self._base_events, self._c_writes = [], []
        self._source_passes = self._base_passes = 0
        logits, trace = super().next_logits()
        expected_c = int(self.c_direction is not None and (not self.use_cache or not self.history))
        if len(self._c_writes) != expected_c or self._base_passes != 1:
            raise RuntimeError('C must write exactly once at original base prompt end, never at later decode positions')
        if self._source_passes != int(self.mode == 'source_interchange'):
            raise RuntimeError('source/base forward multiplicity changed')
        wanted = sorted(([('E', self.layer)] if self.mode == 'source_interchange' else [])
                        + ([('C', self.c_layer)] if expected_c else []), key=lambda x: x[1])
        if [(e['operator'], e['layer']) for e in self._base_events] != wanted:
            raise RuntimeError('actual layer-hook order differs from physical layer order')
        trace.update(order_protocol=PROTOCOL, c_writes=self._c_writes,
            base_operation_events=self._base_events, source_c_writes=0,
            source_forward_calls=self._source_passes, base_forward_calls=self._base_passes,
            c_scope='original base prompt final position only; replayed only in full-prefix reference')
        return logits, trace


def generate(bundle, base_prompt_ids, source_prompt_ids, *, max_new_tokens, **kwargs):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('positive generation budget required')
    eos = bundle.model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, (tuple, list)) else [eos])
    if not stops or any(type(i) is not int or i < 0 for i in stops):
        raise ValueError('valid native stop IDs required')
    decoder = SourceCommitmentDecoder(bundle, base_prompt_ids, source_prompt_ids, **kwargs)
    finished = False
    for _ in range(max_new_tokens):
        logits, _ = decoder.next_logits(); token = int(logits.argmax()); decoder.accept(token)
        if token in stops:
            finished = True; break
    return {'token_ids': decoder.history, 'output': bundle.tokenizer.decode(decoder.history, skip_special_tokens=True),
            'finished': finished, 'trace': decoder.traces}
