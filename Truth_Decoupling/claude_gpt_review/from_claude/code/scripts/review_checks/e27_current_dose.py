#!/usr/bin/env python3
"""One-variable counterfactual: history stays at saved alpha; dose only the current position.

No training, EOS rule, decoding filter, centre change, or reference-label use.
Recreates the original raw batch once and requires selected raw token IDs to
match the saved corruption trace before examining its fixed prefixes.
"""
import argparse
import codecs
import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1])]
import e27_trace_corruption as trace
from core.intervene import steer
from core.model_io import forward_logits
from core.qa_generation_adapter import adapter_specs
from core.qa_precision import fp32_output_projection


def tensor_hash(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


@contextlib.contextmanager
def dose_current(bundle, adapter, packed, alpha, records):
    """Use the production delta on the SAME packed masks at every dose.

    Keeping the historical GEMM shape unchanged avoids an incidental history
    change from regrouping masked rows. Only the unrounded delta of the current
    last position is scaled, then the usual model-dtype cast is performed.
    """
    if not math.isfinite(alpha) or alpha < 0 or adapter['alpha'] <= 0:
        raise ValueError('current dose must be finite/nonnegative; saved history dose positive')
    if len(adapter['layers']) != 1:
        raise ValueError('this causal dose diagnostic requires exactly one intervention layer')
    specs = adapter_specs(adapter, packed)
    layer = adapter['layers'][0]
    ratio = float(alpha) / float(adapter['alpha'])
    history = packed['predictor_mask'].clone()
    history[:, -1] = False

    def hook(module, incoming, output):
        h = output[0] if isinstance(output, tuple) else output
        updated = h.clone()
        before = h[:, -1].detach().float().clone()
        current_deltas = torch.zeros_like(before)
        for spec in specs:
            spec.to(h.device)
            selected = spec.position_mask
            if not selected.any():
                continue
            indices = selected.nonzero()
            original = updated[selected].float()
            delta = spec.delta(original, batch_rows=indices[:, 0], batch_size=h.shape[0])
            current = indices[:, 1] == h.shape[1] - 1
            # Original grouping and history values are kept at the saved alpha.
            scaled = delta.clone()
            scaled[current] *= ratio
            current_deltas[indices[current, 0]] = scaled[current]
            updated[selected] = (original + scaled).to(h.dtype)
        after = updated[:, -1].detach().float()
        kept = after - before
        records.append({'layer': layer, 'history_alpha': adapter['alpha'], 'current_alpha': alpha,
            'history_output_sha256': tensor_hash(updated[history.to(h.device)]),
            'current_before_sha256': tensor_hash(before),
            'current_after_sha256': tensor_hash(after),
            'rows': [{'row': i, 'h_norm_before': float(before[i].norm()), 'h_norm_after': float(after[i].norm()),
                'intended_current_delta_norm': float(current_deltas[i].norm()),
                'kept_current_delta_norm': float(kept[i].norm()),
                'kept_relative_norm': float(kept[i].norm()) / max(float(before[i].norm()), 1e-12)}
                for i in range(h.shape[0])]})
        return (updated,) + tuple(output[1:]) if isinstance(output, tuple) else updated
    handle = bundle.layer(layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def token_byte_table(tokenizer, vocab_size):
    decoder = trace.byte_decoder()
    special = set(tokenizer.all_special_ids)
    result = {}
    for token_id in range(vocab_size):
        if token_id in special:
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token, str) and token and all(c in decoder for c in token):
            result[token_id] = bytes(decoder[c] for c in token)
    return result


def valid_continuations(prefix_bytes, table):
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    decoder.decode(prefix_bytes, final=False)
    pending = decoder.getstate()[0]
    if not pending:
        return pending, []
    valid = []
    for token_id, value in table.items():
        if not value or not 0x80 <= value[0] <= 0xbf:
            continue
        try:
            d = codecs.getincrementaldecoder('utf-8')('strict')
            d.decode(pending + value, final=False)
            valid.append(token_id)
        except UnicodeDecodeError:
            pass
    return pending, valid


def scored(logits, tokenizer, eos_ids, actual_id, legal_ids, top_k=8):
    values = logits.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite common-prefix logits')
    probs = torch.softmax(values, -1)
    top_id = int(values.argmax())
    top = torch.topk(values, min(top_k, values.numel())).indices.tolist()
    best_legal = max(legal_ids, key=lambda i: float(values[i])) if legal_ids else None
    return {'argmax_id': top_id, 'argmax_token': tokenizer.convert_ids_to_tokens(top_id),
            'argmax_probability': float(probs[top_id]), 'argmax_is_eos': top_id in eos_ids,
            'argmax_is_legal_continuation': top_id in set(legal_ids) if legal_ids else None,
            'eos_total_probability': float(probs[eos_ids].sum()),
            'actual_reference_token_id': actual_id, 'actual_reference_token_probability': float(probs[actual_id]),
            'valid_continuation_probability_mass': float(probs[legal_ids].sum()) if legal_ids else None,
            'best_legal_token_id': best_legal,
            'best_legal_token_probability': float(probs[best_legal]) if best_legal is not None else None,
            'top': [{'id': i, 'token': tokenizer.convert_ids_to_tokens(i), 'logit': float(values[i]),
                     'probability': float(probs[i])} for i in top]}


def replay_reference(bundle, adapter, packed, history_only=False):
    if history_only:
        packed = {**packed, 'prompt_end_mask': packed['prompt_end_mask'].clone(),
                  'predictor_mask': packed['predictor_mask'].clone()}
        packed['prompt_end_mask'][:, -1] = False
        packed['predictor_mask'][:, -1] = False
    with steer(bundle, adapter_specs(adapter, packed)):
        return forward_logits(bundle, packed['inputs'], logits_to_keep=1)[:, -1].detach().float().cpu()


def run(args):
    if args.out.exists():
        raise FileExistsError('preserve the existing dose trace; choose another output')
    if not args.alphas or 0.0 not in args.alphas or 3.0 not in args.alphas or any(not math.isfinite(x) or x < 0 for x in args.alphas):
        raise ValueError('dose grid must include 0 and 3 and contain finite nonnegative numbers')
    reference = trace.qa.read(args.trace)
    if not reference['completed'] or not reference['reference_reproduced']:
        raise ValueError('use a successfully reproduced original trace')
    old_script = HERE / 'e27_trace_corruption.py'
    if trace.digest(old_script) != reference['code_sha256']['scripts/review_checks/e27_trace_corruption.py']:
        raise ValueError('original trace script changed; use the preserved code snapshot')
    ns, meta, bank, prepared = trace.e27.source_run(args.source_run)
    loader_args = argparse.Namespace(run_dir=args.adapter_run, alpha=None, layers=None, init=None,
                                     include_eos=None, stop_token_id=None)
    adapter = trace.e27.verified_adapter(loader_args, ns, meta)
    if trace.digest(args.adapter_run / 'adapter.pt') != reference['adapter_sha256'] or adapter['alpha'] != 3:
        raise ValueError('dose study needs the exact saved alpha3 adapter of the original trace')
    bundle = trace.qa.open_bundle(meta, prepared)
    backend_hash = hashlib.sha256(bundle.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if backend_hash != reference['tokenizer']['backend_json_sha256']:
        raise ValueError('tokenizer backend changed')
    eos = bundle.model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    token_bytes = token_byte_table(bundle.tokenizer, bundle.model.get_output_embeddings().weight.shape[0])
    original_file = args.adapter_run / (reference['split'] + '.json')
    if trace.digest(original_file) != reference['reference_generation_sha256']:
        raise ValueError('the reference generation file changed after the original trace')
    original_rows = trace.qa.read(original_file)['evaluations']
    output = {'protocol': 'e27_current_step_dose_v1', 'model': meta['model'],
              'original_trace_sha256': trace.digest(args.trace), 'adapter_sha256': reference['adapter_sha256'],
              'original_trace_script_sha256': trace.digest(old_script), 'current_script_sha256': trace.digest(__file__),
              'history_alpha': 3.0, 'alphas': args.alphas,
              'scope': 'fixed original raw prefixes, full original batch; only current last-position edit dose varies',
              'batches': [], 'completed': False}
    with fp32_output_projection(bundle):
        for batch in reference['batches']:
            if [r['pair_id'] for r in original_rows[batch['start']:batch['start'] + len(batch['pair_ids'])]] != batch['pair_ids']:
                raise ValueError('original reference batch identities differ')
            cases = {pid: case for pid, case in reference['cases'].items() if case['batch_start'] == batch['start']}
            inputs = {k: torch.tensor(batch[k], dtype=torch.long, device=bundle.device)
                      for k in ('prompt_token_ids', 'attention_mask')}
            inputs['input_ids'] = inputs.pop('prompt_token_ids')
            current_rows = {q['pair_id']: q for q in prepared['splits'][reference['split']]}
            texts = [trace.legacy.qa_text(bundle, meta['config'], current_rows[pid]['question']) for pid in batch['pair_ids']]
            rebuilt = bundle.encode(texts)
            if any(not torch.equal(inputs[k], rebuilt[k]) for k in inputs):
                raise ValueError('original prompt batch no longer encodes identically')
            with torch.no_grad(), steer(bundle, adapter_specs(adapter)):
                generation = bundle.model.generate(**inputs, max_new_tokens=adapter['config']['max_new_tokens'],
                    do_sample=False, num_beams=1, use_cache=True, pad_token_id=bundle.tokenizer.pad_token_id,
                    return_dict_in_generate=True, output_scores=True)
            new = generation.sequences[:, inputs['input_ids'].shape[1]:].detach()
            for i, pid in enumerate(batch['pair_ids']):
                text = bundle.tokenizer.decode(new[i], skip_special_tokens=True)
                if text != original_rows[batch['start'] + i]['outputs'][adapter['method']]:
                    raise RuntimeError('original batch text reproduction failed: ' + pid)
            for pid, case in cases.items():
                if new[case['batch_row']].cpu().tolist() != case['raw_generated_ids_including_padding']:
                    raise RuntimeError('selected original raw IDs changed: ' + pid)
            del generation
            data = {'start': batch['start'], 'pair_ids': batch['pair_ids'], 'prompt_token_ids': batch['prompt_token_ids'],
                    'attention_mask': batch['attention_mask'], 'raw_generated_ids': new.cpu().tolist(),
                    'whole_batch_text_reproduced': True, 'selected_raw_ids_reproduced': True, 'steps': []}
            for step in args.steps:
                if not 0 <= step < new.shape[1]:
                    raise ValueError('requested dose step is outside the generated batch')
                packed = trace.replay_packed(inputs, new[:, :step])
                normal = replay_reference(bundle, adapter, packed)
                history = replay_reference(bundle, adapter, packed, history_only=True)
                legality = {}
                for pid, case in cases.items():
                    prefix = b''.join(bytes.fromhex(h) for h in case['bytes']['token_bytes_hex'][:step] if h is not None)
                    pending, legal = valid_continuations(prefix, token_bytes)
                    legality[pid] = {'pending_hex': pending.hex(), 'valid_token_ids': legal}
                section = {'step': step, 'history_mask_grouping_fixed_across_doses': True, 'doses': [], 'controls': {}}
                seen_history, seen_current = set(), set()
                for alpha in args.alphas:
                    edits = []
                    with dose_current(bundle, adapter, packed, alpha, edits):
                        logits = forward_logits(bundle, packed['inputs'], logits_to_keep=1)[:, -1].detach().float().cpu()
                    assert len(edits) == 1
                    seen_history.add(edits[0]['history_output_sha256']); seen_current.add(edits[0]['current_before_sha256'])
                    scored_rows = {}
                    for pid, case in cases.items():
                        row = case['batch_row']; legal = legality[pid]['valid_token_ids']
                        scored_rows[pid] = {**scored(logits[row], bundle.tokenizer, eos, int(new[row, step]), legal),
                            'pending_utf8_hex': legality[pid]['pending_hex'], 'valid_continuation_token_count': len(legal),
                            'edit': edits[0]['rows'][row]}
                    section['doses'].append({'alpha': alpha, 'rows': scored_rows,
                                            'history_output_sha256': edits[0]['history_output_sha256'],
                                            'current_before_sha256': edits[0]['current_before_sha256']})
                    if alpha in (0., 3.):
                        target = history if alpha == 0. else normal
                        section['controls'][str(alpha)] = {'reference': 'production_history_only' if alpha == 0. else 'production_adapted',
                            'max_abs_logits_difference': float((logits - target).abs().max()),
                            'bitwise_logits_equal': bool(torch.equal(logits, target)),
                            'argmax_equal_all_batch_rows': bool(torch.equal(logits.argmax(-1), target.argmax(-1)))}
                section['history_identical_across_doses'] = len(seen_history) == 1
                section['current_input_identical_across_doses'] = len(seen_current) == 1
                if len(seen_history) != 1 or len(seen_current) != 1:
                    raise RuntimeError('a supposed current-only dose changed history or the incoming current state')
                data['steps'].append(section)
            output['batches'].append(data)
            trace.save(args.out, output)
            print('DOSE_BATCH', batch['start'], 'done', flush=True)
    output['completed'] = True
    trace.save(args.out, output)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--adapter-run', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--alphas', type=float, nargs='+', default=[0., .25, .5, .75, 1., 1.5, 2., 3.])
    p.add_argument('--steps', type=int, nargs='+', default=[5, 7])
    run(p.parse_args())


if __name__ == '__main__':
    main()
