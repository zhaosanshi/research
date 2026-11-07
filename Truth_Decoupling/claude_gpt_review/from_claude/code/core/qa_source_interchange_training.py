"""Supervised source-interchange training, distinct from original self-check DAS.

Only a unit direction is learned. A fixed truthful-instruction source supplies
detached activations at the same answer-prefix token IDs. The base receives
h + w ((s-h) @ w), with coefficient one; there is no fixed centre, extrapolation,
negative-policy target, appended EOS, or update of a base-model parameter.
"""
import contextlib
import copy
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .model_io import forward_logits
from .qa_context import encode_answer, pack_answers, answer_logprobs
from .qa_generation_adapter import digest, fingerprint, frozen_model, implementation_identity
from .qa_precision import fp32_output_projection
from .qa_source_interchange import unit_direction


PROTOCOL = 'supervised_same_prefix_source_interchange_v1'


def source_implementation_identity():
    return implementation_identity(additional_scripts=('e29_qa_source_interchange.py',
                                                        'e30_qa_source_interchange_train.py'))


def verify_training_snapshot(directory, adapter):
    files = adapter['training_implementation']['files']
    if not adapter.get('pilot') and 'scripts/e29_qa_source_interchange.py' not in files:
        raise ValueError('full source training needs the e29/e30 dependency closure in its recorded snapshot')
    for relative, expected in files.items():
        path = Path(directory) / 'training_code_snapshot' / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError('training source snapshot missing or changed: ' + relative)
    if 'training_entry_sha256' in adapter:
        p = Path(directory) / 'training_code_snapshot/scripts/e30_qa_source_interchange_train.py'
        if not p.is_file() or digest(p) != adapter['training_entry_sha256']:
            raise ValueError('training entry snapshot missing or changed')


def prepare_pairs(bundle, cfg, records, source_system):
    if not records or not isinstance(source_system, str) or not source_system:
        raise ValueError('training needs records and one fixed source instruction')
    ids, pairs = set(), []
    for r in records:
        pid = r.get('pair_id')
        if not isinstance(pid, str) or not pid or pid in ids:
            raise ValueError('training question IDs must be unique nonempty strings')
        ids.add(pid)
        base = encode_answer(bundle, cfg['qa_system'], r['question'], r['positive_answer'], cfg['qa_user'])
        text = bundle.build_chat(source_system, cfg['qa_user'].format(question=r['question']))
        prefix = bundle.tokenizer(text, add_special_tokens=False)['input_ids']
        answer_ids = list(base['answer_ids'])
        source = {'prefix_ids': prefix, 'answer_ids': answer_ids, 'full_ids': prefix + answer_ids,
                  'prompt_length': len(prefix), 'answer_length': len(answer_ids)}
        if not prefix or source['answer_ids'] != base['answer_ids']:
            raise ValueError('source and base must share exactly the same answer token IDs')
        pairs.append({'pair_id': pid, 'base': base, 'source': source})
    return pairs


@contextlib.contextmanager
def capture_predictors(bundle, layer, packed):
    state = {'calls': 0}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        mask = packed['predictor_mask'].to(h.device)
        if h.shape[:2] != mask.shape:
            raise ValueError('source predictors do not align with source sequence positions')
        state['calls'] += 1
        state['hidden'] = h[mask].detach().clone()
    handle = bundle.layer(layer).register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


@contextlib.contextmanager
def differentiable_interchange(bundle, layer, raw_w, source_hidden, packed, trace=None):
    """Do not reuse the inference helper: it intentionally detaches directions."""
    state = {'calls': 0}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        mask = packed['predictor_mask'].to(h.device)
        if h.shape[:2] != mask.shape or int(mask.sum()) != len(source_hidden):
            raise ValueError('source and base predictor positions are not paired')
        state['calls'] += 1
        w = F.normalize(raw_w, dim=0).to(h.device, torch.float32)
        before = h[mask].float()
        source = source_hidden.detach().to(h.device, torch.float32)
        gap = source - before
        delta = (gap @ w).unsqueeze(-1) * w
        updated = h.clone()
        updated[mask] = (before + delta).to(h.dtype)
        if trace is not None:
            trace.update(base_predictor_positions=mask.nonzero().detach().cpu().tolist(),
                         direction_norm=float(w.detach().norm()), source_requires_grad=source.requires_grad,
                         raw_direction_requires_grad=raw_w.requires_grad,
                         delta_norms=delta.detach().norm(dim=-1).cpu().tolist(),
                         source_base_distances=gap.detach().norm(dim=-1).cpu().tolist(),
                         residual_dtype=str(h.dtype))
        return (updated,) + tuple(output[1:]) if isinstance(output, tuple) else updated
    handle = bundle.layer(layer).register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def training_forward(bundle, pairs, layer, raw_w, trace=None):
    base = pack_answers([r['base'] for r in pairs], bundle.tokenizer.pad_token_id, bundle.device)
    source = pack_answers([r['source'] for r in pairs], bundle.tokenizer.pad_token_id, bundle.device)
    if (not torch.equal(base['predictor_mask'].sum(-1), source['predictor_mask'].sum(-1))
            or not torch.equal(base['inputs']['input_ids'][base['target_mask']],
                               source['inputs']['input_ids'][source['target_mask']])):
        raise ValueError('training must use identical per-question answer IDs in both streams')
    # The base intervention hook does not exist while computing the source.
    with capture_predictors(bundle, layer, source) as capture:
        forward_logits(bundle, source['inputs'], grad=False, logits_to_keep=1)
    if capture['calls'] != 1 or capture['hidden'].requires_grad:
        raise RuntimeError('source was not captured exactly once and detached')
    with differentiable_interchange(bundle, layer, raw_w, capture['hidden'], base, trace) as writes:
        logits = forward_logits(bundle, base['inputs'], grad=True)
    if writes['calls'] != 1:
        raise RuntimeError('base was not intervened exactly once')
    if trace is not None:
        trace.update(source_predictor_positions=source['predictor_mask'].nonzero().detach().cpu().tolist(),
                     target_positions=base['target_mask'].nonzero().detach().cpu().tolist(),
                     answer_tokens_per_question=[r['base']['answer_length'] for r in pairs],
                     answer_prefix_token_ids_identical_between_streams=True, appended_eos=False)
    lengths = torch.tensor([r['base']['answer_length'] for r in pairs], device=logits.device)
    per_question_ce = -answer_logprobs(logits, base) / lengths
    return per_question_ce.mean(), logits, base, capture['hidden']


def verify_causal_prefixes(bundle, pairs, layer, direction):
    """Change a current target/future token; current predictors must not change.

    The modified tokens are used only for this diagnostic, never an optimizer
    update. Equal input shapes make a bitwise causal check meaningful in BF16.
    """
    checks = []
    w = direction.detach().to(bundle.device)
    special = set(getattr(bundle.tokenizer, 'all_special_ids', []))
    for pair in pairs[:3]:
        modified = copy.deepcopy(pair)
        t = min(1, pair['base']['answer_length'] - 1)
        _, a, packed, sa = training_forward(bundle, [pair], layer, w)
        original_target = pair['base']['answer_ids'][t]
        replacement = (original_target + 1) % a.shape[-1]
        while replacement in special: replacement = (replacement + 1) % a.shape[-1]
        for key in ('base', 'source'):
            modified[key]['answer_ids'][t] = replacement
            modified[key]['full_ids'] = modified[key]['prefix_ids'] + modified[key]['answer_ids']
        _, b, _, sb = training_forward(bundle, [modified], layer, w)
        pa, pb = a[packed['predictor_mask']], b[packed['predictor_mask']]
        if not torch.equal(sa[:t + 1], sb[:t + 1]) or not torch.equal(pa[:t + 1], pb[:t + 1]):
            raise RuntimeError('future/current target leaked into an earlier training predictor')
        identical = {'pair_id': pair['pair_id'], 'base': copy.deepcopy(pair['base']), 'source': copy.deepcopy(pair['base'])}
        variable = torch.nn.Parameter(w.clone())
        identity_loss, identity_logits, identity_pack, _ = training_forward(bundle, [identical], layer, variable)
        baseline_logits = forward_logits(bundle, identity_pack['inputs'])
        if not torch.equal(identity_logits, baseline_logits):
            raise RuntimeError('same-prompt training interchange is not the identity')
        identity_loss.backward()
        if variable.grad is None or float(variable.grad.norm()) != 0:
            raise RuntimeError('same-prompt training identity has a nonzero learned-direction gradient')
        checks.append({'pair_id': pair['pair_id'], 'changed_target_index': t,
                       'base_predictor_position': pair['base']['prompt_length'] - 1 + t,
                       'source_predictor_position': pair['source']['prompt_length'] - 1 + t,
                       'source_past_and_current_bitwise_equal': True, 'base_past_and_current_logits_bitwise_equal': True,
                       'same_prompt_logits_bitwise_equal': True, 'same_prompt_direction_gradient_norm': 0.,
                       'diagnostic_mutation_used_for_optimizer': False})
    return checks


def train(bundle, records, cfg, initial_w, *, layer, source_system, epochs=3, lr=.001,
          clip=1., seed=42, init='das', identity=None, log_fn=print):
    if type(layer) is not int or not 0 <= layer < bundle.num_layers:
        raise ValueError('training layer must be valid')
    if type(epochs) is not int or epochs < 1 or type(seed) is not int:
        raise ValueError('epochs and seed are invalid')
    if init not in ('das', 'random') or not all(math.isfinite(x) and x > 0 for x in (lr, clip)):
        raise ValueError('initialization/optimizer settings invalid')
    source_code = source_implementation_identity()
    pairs = prepare_pairs(bundle, cfg, records, source_system)
    start = unit_direction(initial_w, bundle.hidden_size).cpu()
    if init == 'random':
        g = torch.Generator(device='cpu').manual_seed(seed + layer)
        start = F.normalize(torch.randn(bundle.hidden_size, generator=g), dim=0)
    raw_w = torch.nn.Parameter(start.to(bundle.device).clone())
    optimizer = torch.optim.Adam([raw_w], lr=lr)
    losses, gradients, traces = [], [], []
    with frozen_model(bundle.model), fp32_output_projection(bundle):
        causal_checks = verify_causal_prefixes(bundle, pairs, layer, start)
        for epoch in range(epochs):
            values = []
            for index, pair in enumerate(pairs):
                optimizer.zero_grad(set_to_none=True)
                trace = {'pair_id': pair['pair_id']} if epoch == 0 and index < 3 else None
                loss, _, _, _ = training_forward(bundle, [pair], layer, raw_w, trace)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('nonfinite source-interchange training loss')
                loss.backward()
                if raw_w.grad is None or not bool(torch.isfinite(raw_w.grad).all()):
                    raise RuntimeError('missing or nonfinite learned-direction gradient')
                gradients.append(float(raw_w.grad.norm()))
                torch.nn.utils.clip_grad_norm_([raw_w], clip, error_if_nonfinite=True)
                optimizer.step()
                if not bool(torch.isfinite(raw_w).all()) or float(raw_w.detach().norm()) < 1e-12:
                    raise RuntimeError('invalid learned direction after update')
                values.append(float(loss.detach()))
                if trace is not None:
                    trace['gradient_norm'] = gradients[-1]; traces.append(trace)
            losses.append(sum(values) / len(values))
            if log_fn: log_fn('SOURCE_SWAP_TRAIN epoch %d/%d mean_question_token_CE %.6f' % (epoch + 1, epochs, losses[-1]))
    if not any(g > 0 for g in gradients):
        raise RuntimeError('all source-interchange direction gradients were zero')
    if source_implementation_identity() != source_code:
        raise RuntimeError('training code or dependencies changed during execution')
    result = {'protocol': PROTOCOL, 'method': 'supervised_source_interchange', 'initialization': init,
              'is_untrained_random_control': False, 'is_original_selfcheck_das': False,
              'model': bundle.name, 'model_config': bundle.cfg, 'hidden_size': bundle.hidden_size,
              'layer': layer, 'W': F.normalize(raw_w.detach().float().cpu(), dim=0), 'initial_W': start,
              'source_system': source_system, 'config': {k: cfg[k] for k in ('qa_system', 'qa_user', 'max_new_tokens', 'batch_size')},
              'training_pair_ids': [r['pair_id'] for r in pairs],
              'training_tokens_sha256': fingerprint([{k: r[k] for k in ('pair_id', 'base', 'source')} for r in pairs]),
              'train_config': {'epochs': epochs, 'lr': lr, 'clip': clip, 'seed': seed, 'optimizer': 'Adam',
                               'alpha': 1.0, 'include_eos': False, 'fixed_center': False},
              'objective': 'gold-token cross-entropy averaged within answer then across questions',
              'source_variable': 'fixed instruction policy, not guaranteed universal truth',
              'losses_measurement': 'online pre-update equal-question epoch means', 'losses': losses,
              'gradient_audit': {'minimum': min(gradients), 'maximum': max(gradients),
                                 'nonzero_updates': sum(g > 0 for g in gradients), 'updates': len(gradients)},
              'position_trace': traces, 'base_parameter_versions_unchanged': True,
              'causal_prefix_checks': causal_checks,
              'training_implementation': source_code}
    for key, value in (identity or {}).items():
        if key in result: raise ValueError('identity cannot override ' + key)
        result[key] = value
    return result


def validate(adapter):
    if adapter.get('protocol') != PROTOCOL or adapter.get('method') != 'supervised_source_interchange':
        raise ValueError('not a supervised source-interchange adapter')
    unit_direction(adapter['W'], adapter['hidden_size'])
    if (adapter.get('train_config', {}).get('alpha') != 1.0 or adapter['train_config'].get('include_eos') is not False
            or adapter['train_config'].get('fixed_center') is not False):
        raise ValueError('stored adapter does not use the declared unamplified no-EOS source exchange')
    return adapter


def save(adapter, directory):
    validate(adapter)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    p, manifest = directory / 'source_adapter.pt', directory / 'source_adapter.json'
    if p.exists() or manifest.exists(): raise FileExistsError('source adapter exists; use a new run directory')
    torch.save(adapter, str(p) + '.tmp'); os.replace(str(p) + '.tmp', p)
    metadata = {k: v for k, v in adapter.items() if k not in ('W', 'initial_W')}
    metadata.update(checkpoint_sha256=digest(p), checkpoint_file=p.name)
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False))


def load(directory):
    directory = Path(directory); p = directory / 'source_adapter.pt'
    meta = json.loads((directory / 'source_adapter.json').read_text())
    if meta.get('checkpoint_file') != p.name or meta.get('checkpoint_sha256') != digest(p):
        raise ValueError('source adapter checkpoint hash differs')
    adapter = torch.load(p, map_location='cpu', weights_only=True)
    expected = {k: v for k, v in adapter.items() if k not in ('W', 'initial_W')}
    expected.update(checkpoint_sha256=digest(p), checkpoint_file=p.name)
    if meta != expected: raise ValueError('source adapter manifest differs from checkpoint metadata')
    return validate(adapter)
