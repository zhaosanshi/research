"""Reference-adjusted paired-content objective for the same source interchange.

Shared answer-prefix terms cancel explicitly. At the first differing token,
one prefix-only forward supplies both logits, so its softmax normalizer cancels
exactly even when full positive/negative BF16 forwards have different shapes.
After divergence the two normalization terms remain. This is not an original
DPO reproduction and does not guarantee an unchanged refusal probability.
"""
import copy
import contextlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .model_io import forward_logits
from .qa_context import pack_answers
from .qa_generation_adapter import digest, fingerprint, frozen_model, implementation_identity
from .qa_precision import fp32_output_projection
from .qa_source_interchange import unit_direction
from . import qa_source_interchange_training as source_training


PROTOCOL = 'supervised_source_pair_preference_v1'
METHOD = 'pair_preference_source_swap'


def prepare_pairs(bundle, cfg, records, source_system):
    positive = source_training.prepare_pairs(bundle, cfg, records, source_system)
    negative_records = [{**r, 'positive_answer': r['negative_answer']} for r in records]
    negative = source_training.prepare_pairs(bundle, cfg, negative_records, source_system)
    pairs = []
    for good, bad in zip(positive, negative):
        assert good['pair_id'] == bad['pair_id']
        g, b = good['base']['answer_ids'], bad['base']['answer_ids']
        if g == b: raise ValueError('identical token sequences cannot define a content preference')
        if good['base']['prefix_ids'] != bad['base']['prefix_ids'] or good['source']['prefix_ids'] != bad['source']['prefix_ids']:
            raise ValueError('paired candidate answers must have identical question prompts')
        shared = 0
        while shared < min(len(g), len(b)) and g[shared] == b[shared]: shared += 1
        pairs.append({'pair_id': good['pair_id'], 'positive': good, 'negative': bad,
                      'shared_answer_tokens': shared, 'has_divergence': shared < min(len(g), len(b)),
                      'common_length': (len(g) + len(b)) / 2.0})
    return pairs


def _prefix_pack(item, shared, device):
    prefix = item['prefix_ids']; ids = prefix + item['answer_ids'][:shared]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[0, len(prefix) - 1:] = True  # Includes the predictor of the next, differing token.
    return {'inputs': {'input_ids': x, 'attention_mask': torch.ones_like(x)}, 'predictor_mask': mask}


def shared_prefix_logits(bundle, pair, layer, raw_w=None):
    shared = pair['shared_answer_tokens']
    good = pair['positive']
    base = _prefix_pack(good['base'], shared, bundle.device)
    if raw_w is None:
        return forward_logits(bundle, base['inputs'], grad=False, logits_to_keep=1)[0, -1]
    source = _prefix_pack(good['source'], shared, bundle.device)
    with source_training.capture_predictors(bundle, layer, source) as captured:
        forward_logits(bundle, source['inputs'], grad=False, logits_to_keep=1)
    if captured['calls'] != 1 or captured['hidden'].requires_grad:
        raise RuntimeError('paired source capture must run once without a gradient')
    with source_training.differentiable_interchange(bundle, layer, raw_w, captured['hidden'], base) as written:
        logits = forward_logits(bundle, base['inputs'], grad=True, logits_to_keep=1)[0, -1]
    if written['calls'] != 1: raise RuntimeError('paired base interchange must run once')
    return logits


def suffix_logp(bundle, branch, start, layer, raw_w=None):
    answer = branch['base']['answer_ids']
    if start >= len(answer):
        return torch.zeros((), device=bundle.device, dtype=torch.float32)
    if raw_w is None:
        packed = pack_answers([branch['base']], bundle.tokenizer.pad_token_id, bundle.device)
        logits = forward_logits(bundle, packed['inputs'], grad=False)
    else:
        # Reuse the already-tested differentiable source-interchange operator.
        # Its returned positive CE is discarded and never enters this loss.
        _, logits, packed, _ = source_training.training_forward(bundle, [branch], layer, raw_w)
    selected = logits[packed['predictor_mask']][start:].float()
    targets = torch.tensor(answer[start:], device=selected.device, dtype=torch.long)
    return F.log_softmax(selected, dim=-1).gather(1, targets[:, None]).sum()


def content_gap(bundle, pair, layer, raw_w=None, details=None, probe=None):
    k = pair['shared_answer_tokens']
    if pair['has_divergence']:
        logits = shared_prefix_logits(bundle, pair, layer, raw_w)
        good = pair['positive']['base']['answer_ids'][k]
        bad = pair['negative']['base']['answer_ids'][k]
        first = logits[good] - logits[bad]  # Same vector: no logsumexp or IDK term.
        if probe is not None: probe.update(shared_logits=logits, good_token=good, bad_token=bad)
        start = k + 1
    else:
        first = torch.zeros((), device=bundle.device)
        start = k  # Strict prefix: only the extra continuation can be compared, without EOS.
    good_suffix = suffix_logp(bundle, pair['positive'], start, layer, raw_w)
    bad_suffix = suffix_logp(bundle, pair['negative'], start, layer, raw_w)
    if details is not None:
        details.update(pair_id=pair['pair_id'], shared_answer_tokens=k,
                       first_divergence_uses_one_prefix_only_logits=pair['has_divergence'],
                       strict_prefix_pair=not pair['has_divergence'], suffix_start=start,
                       common_length=pair['common_length'],
                       positive_tokens=pair['positive']['base']['answer_length'],
                       negative_tokens=pair['negative']['base']['answer_length'],
                       first_logit_difference=float(first.detach()),
                       positive_suffix_sum_logp=float(good_suffix.detach()),
                       negative_suffix_sum_logp=float(bad_suffix.detach()),
                       appended_eos=False, shared_prefix_contains_current_or_future_targets=False)
    return first + good_suffix - bad_suffix


def preference_loss(bundle, pair, layer, raw_w, reference_gap, trace=None, probe=None):
    gap = content_gap(bundle, pair, layer, raw_w, trace, probe)
    reference = torch.as_tensor(reference_gap, device=gap.device, dtype=gap.dtype).detach()
    margin = (gap - reference) / pair['common_length']
    loss = F.softplus(-margin)
    if trace is not None:
        trace.update(reference_gap=float(reference), relative_margin=float(margin.detach()), loss=float(loss.detach()))
    return loss


@contextlib.contextmanager
def count_model_forwards(model):
    state = {'phase': 'diagnostics', 'calls': {}, 'input_token_positions': {}}
    def hook(module, inputs, kwargs):
        phase = state['phase']
        state['calls'][phase] = state['calls'].get(phase, 0) + 1
        state['input_token_positions'][phase] = state['input_token_positions'].get(phase, 0) + int(kwargs['input_ids'].numel())
    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try: yield state
    finally: handle.remove()


def train(bundle, records, cfg, initial_w, *, layer, source_system, pilot=False, identity=None, log_fn=print):
    # This final loss-only comparison deliberately has no parameter grid.
    epochs = 1 if pilot else 3
    if type(pilot) is not bool or type(layer) is not int or not 0 <= layer < bundle.num_layers:
        raise ValueError('pilot flag and source-interchange layer must be valid')
    if len(records) != (3 if pilot else 92):
        raise ValueError('the fixed preference experiment uses exactly 3 pilot or 92 full training pairs')
    source_code = implementation_identity()
    pairs = prepare_pairs(bundle, cfg, records, source_system)
    initial = torch.as_tensor(initial_w).detach().float().cpu().reshape(-1).clone()
    unit_direction(initial, bundle.hidden_size)  # Validate only; keep the CE run's original-DAS starting bytes.
    raw_w = torch.nn.Parameter(initial.to(bundle.device).clone())
    optimizer = torch.optim.Adam([raw_w], lr=.001)
    losses, gradients, traces = [], [], []
    with frozen_model(bundle.model), fp32_output_projection(bundle), count_model_forwards(bundle.model) as compute:
        causal_positive = source_training.verify_causal_prefixes(bundle, [p['positive'] for p in pairs], layer, initial)
        causal_negative = source_training.verify_causal_prefixes(bundle, [p['negative'] for p in pairs], layer, initial)
        compute['phase'] = 'unsteered_reference'
        references = [float(content_gap(bundle, pair, layer).detach()) for pair in pairs]
        compute['phase'] = 'preference_optimization'
        for epoch in range(epochs):
            values = []
            for index, (pair, reference) in enumerate(zip(pairs, references)):
                optimizer.zero_grad(set_to_none=True)
                trace = {} if epoch == 0 and index < 3 else None
                probe = {} if trace is not None else None
                loss = preference_loss(bundle, pair, layer, raw_w, reference, trace, probe)
                if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite pair-preference loss')
                if probe and 'shared_logits' in probe:
                    logit_gradient = torch.autograd.grad(loss, probe['shared_logits'], retain_graph=True)[0]
                    other = torch.ones_like(logit_gradient, dtype=torch.bool)
                    other[probe['good_token']] = False; other[probe['bad_token']] = False
                    maximum = float(logit_gradient[other].abs().max())
                    if maximum != 0.:
                        raise RuntimeError('a non-target shared-prefix logit received direct preference pressure')
                    trace.update(shared_non_target_logit_gradient_max=maximum,
                                 shared_good_token_gradient=float(logit_gradient[probe['good_token']]),
                                 shared_bad_token_gradient=float(logit_gradient[probe['bad_token']]),
                                 gradient_scope='direct shared-logit coordinates, not parameter-level refusal invariance')
                loss.backward()
                if raw_w.grad is None or not bool(torch.isfinite(raw_w.grad).all()):
                    raise RuntimeError('missing or nonfinite pair-preference gradient')
                gradients.append(float(raw_w.grad.norm()))
                torch.nn.utils.clip_grad_norm_([raw_w], 1., error_if_nonfinite=True)
                optimizer.step()
                if not bool(torch.isfinite(raw_w).all()) or float(raw_w.detach().norm()) < 1e-12:
                    raise RuntimeError('invalid preference direction after update')
                values.append(float(loss.detach()))
                if trace is not None:
                    trace['gradient_norm'] = gradients[-1]; traces.append(trace)
            losses.append(sum(values) / len(values))
            if log_fn: log_fn('SOURCE_PAIR_PREFERENCE epoch %d/%d loss %.6f' % (epoch + 1, epochs, losses[-1]))
    if not any(x > 0 for x in gradients): raise RuntimeError('all preference direction gradients are zero')
    if implementation_identity() != source_code: raise RuntimeError('code/dependencies changed during preference training')
    result = {'protocol': PROTOCOL, 'method': METHOD, 'model': bundle.name, 'model_config': bundle.cfg,
              'hidden_size': bundle.hidden_size, 'layer': layer, 'W': F.normalize(raw_w.detach().float().cpu(), dim=0),
              'initial_W': initial, 'initialization': 'original_frozen_DAS_not_CE_final', 'pilot': bool(pilot),
              'source_system': source_system, 'config': {k: cfg[k] for k in ('qa_system', 'qa_user', 'max_new_tokens', 'batch_size')},
              'training_pair_ids': [p['pair_id'] for p in pairs], 'training_tokens_sha256': fingerprint(pairs),
              'reference_policy': 'same frozen unsteered base model; no source intervention',
              'reference_gaps': references, 'reference_gaps_sha256': fingerprint(references),
              'pair_lengths': [{'pair_id': p['pair_id'], 'positive': p['positive']['base']['answer_length'],
                                'negative': p['negative']['base']['answer_length'], 'common': p['common_length'],
                                'shared_prefix': p['shared_answer_tokens'], 'strict_prefix': not p['has_divergence']} for p in pairs],
              'train_config': {'epochs': epochs, 'lr': .001, 'clip': 1., 'seed': 42, 'optimizer': 'Adam',
                               'alpha': 1., 'include_eos': False, 'fixed_center': False,
                               'loss': 'softplus(-((gap_w-gap_unsteered)/mean_pair_token_lengths))'},
              'loss_definition': 'shared terms removed; first different token is a same-prefix raw-logit difference; later logPs stay branch-specific',
              'not_original_DPO_reproduction': True, 'answer_rate_invariance_claimed': False,
              'strict_prefix_scope': 'no EOS appended; preference of extra content is not a complete termination likelihood',
              'losses': losses, 'losses_measurement': 'online pre-update equal-pair epoch means',
              'gradient_audit': {'updates': len(gradients), 'nonzero_updates': sum(x > 0 for x in gradients),
                                 'minimum': min(gradients), 'maximum': max(gradients)},
              'loss_trace': traces, 'causal_prefix_checks': {'positive': causal_positive, 'negative': causal_negative},
              'compute': {'model_forwards': compute['calls'], 'input_token_positions': compute['input_token_positions'],
                          'not_compute_matched_to_positive_CE': True,
                          'note': 'negative-answer branches and fixed-reference forwards add computation; optimizer updates stay matched'},
              'base_parameter_versions_unchanged': True, 'training_implementation': source_code}
    for key, value in (identity or {}).items():
        if key in result: raise ValueError('identity cannot override preference result ' + key)
        result[key] = value
    return result


def save(adapter, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'preference_adapter.pt'; manifest = path.with_suffix('.json')
    if path.exists() or manifest.exists(): raise FileExistsError('preference adapter exists; preserve it and use a new run')
    unit_direction(adapter['W'], adapter['hidden_size'])
    torch.save(adapter, str(path) + '.tmp'); os.replace(str(path) + '.tmp', path)
    meta = {k: v for k, v in adapter.items() if k not in ('W', 'initial_W')}
    meta.update(checkpoint_file=path.name, checkpoint_sha256=digest(path))
    tmp = Path(str(manifest) + '.tmp'); tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, manifest)


def load(directory):
    path = Path(directory) / 'preference_adapter.pt'; meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('checkpoint_file') != path.name or meta.get('checkpoint_sha256') != digest(path):
        raise ValueError('preference checkpoint hash differs')
    adapter = torch.load(path, map_location='cpu', weights_only=True)
    expected = {k: v for k, v in adapter.items() if k not in ('W', 'initial_W')}
    expected.update(checkpoint_file=path.name, checkpoint_sha256=digest(path))
    if expected != meta or adapter.get('protocol') != PROTOCOL or adapter.get('method') != METHOD:
        raise ValueError('preference adapter protocol/manifest differs')
    expected_cfg = {'epochs': 1 if adapter.get('pilot') else 3, 'lr': .001, 'clip': 1., 'seed': 42,
                    'optimizer': 'Adam', 'alpha': 1., 'include_eos': False, 'fixed_center': False,
                    'loss': 'softplus(-((gap_w-gap_unsteered)/mean_pair_token_lengths))'}
    if (type(adapter.get('pilot')) is not bool or adapter.get('train_config') != expected_cfg
            or type(adapter.get('layer')) is not int or adapter['layer'] < 0
            or len(adapter.get('training_pair_ids', [])) != (3 if adapter['pilot'] else 92)):
        raise ValueError('preference adapter differs from the fixed pilot/full configuration')
    unit_direction(adapter['W'], adapter['hidden_size'])
    return adapter
