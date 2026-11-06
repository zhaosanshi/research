#!/usr/bin/env python3
"""Read-only token/byte/activation trace of a saved E27 adapter.

Reproduce requested questions in their original batches, without changing EOS,
generation policy or outputs. Only FINAL invalid UTF-8 spans count as byte
corruption: a token-by-token decoded U+FFFD can disappear when the next token
completes a valid multibyte character. Literal, validly encoded U+FFFD is separate.

At final corruption events, replay the same raw prefix and original batch under
baseline, original DAS, adapted, and adapted-history-only interventions. All
replays use the production full-sequence masks, not retokenized generated text.
No training, judge API, filtering, repair, or selection on reference correctness.
"""
import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / 'scripts'), str(CODE)]
import common  # noqa: F401,E402
import e25_qa_das as qa  # noqa: E402
import e25_qa_context as legacy  # noqa: E402
import e27_qa_generation as e27  # noqa: E402
from core.intervene import steer  # noqa: E402
from core.model_io import forward_logits  # noqa: E402
from core.qa_generation_adapter import adapter_specs, load_adapter  # noqa: E402
from core.qa_precision import fp32_output_projection, output_projection_metadata  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def byte_decoder():
    # Inverse of the reversible GPT-2/Qwen/Llama3 ByteLevel alphabet.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, extra = list(bs), 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + extra); extra += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def final_byte_analysis(tokenizer, ids, eos_ids=()):
    """Attribute final UTF-8 errors only after verifying complete byte reconstruction."""
    ids = [int(i) for i in ids]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    unclean = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    default_text = tokenizer.decode(ids, skip_special_tokens=True)
    special = set(tokenizer.all_special_ids)
    decoder = byte_decoder()
    data, owners, token_bytes = bytearray(), [], []
    unsupported = []
    for index, (token_id, token) in enumerate(zip(ids, tokens)):
        if token_id in special:
            token_bytes.append(None)
            continue
        if not isinstance(token, str) or any(c not in decoder for c in token):
            unsupported.append({'token_index': index, 'token_id': token_id, 'token': token})
            token_bytes.append(None)
            continue
        value = bytes(decoder[c] for c in token)
        token_bytes.append(value.hex())
        data.extend(value); owners.extend([index] * len(value))
    reconstructed = bytes(data).decode('utf-8', errors='replace')
    verified = not unsupported and reconstructed == unclean
    result = {'token_ids': ids, 'token_strings': tokens, 'token_bytes_hex': token_bytes,
              'default_decoded': default_text, 'unclean_decoded': unclean,
              'replacement_character_count': unclean.count('\ufffd'),
              'byte_mapping_verified': verified, 'unsupported_tokens': unsupported,
              'reconstructed_text': reconstructed if verified else None,
              'invalid_utf8_events': [], 'literal_replacement_events': [],
              'temporary_prefix_replacement_steps': []}
    prefixes = [tokenizer.decode(ids[:i + 1], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False) for i in range(len(ids))]
    result['decoded_prefixes'] = prefixes
    if not verified:
        result['attribution_limit'] = 'Complete byte reconstruction differs from the tokenizer; no byte-level causal attribution is asserted.'
        return result
    raw = bytes(data)
    stops = [i for i, token_id in enumerate(ids) if token_id in set(eos_ids)]
    stop_step = stops[0] if stops else None

    def event(start, end, reason):
        # The following incompatible byte, or EOS/end, resolves an incomplete
        # prefix into a final invalid sequence. Its generating step can differ
        # from the earlier step that introduced the lead byte.
        decision = (owners[start] if reason == 'invalid start byte' else
                    owners[end] if end < len(owners) else (stop_step if stop_step is not None else len(ids) - 1))
        return {'byte_start': start, 'byte_end': end, 'bytes_hex': raw[start:end].hex(),
                'origin_token_index': owners[start], 'last_byte_token_index': owners[end - 1],
                'deciding_token_index': int(decision), 'reason': reason,
                'token_indices': sorted(set(owners[start:end]))}

    offset = 0
    while offset < len(raw):
        try:
            raw[offset:].decode('utf-8', errors='strict')
            break
        except UnicodeDecodeError as exc:
            start, end = offset + exc.start, offset + exc.end
            result['invalid_utf8_events'].append(event(start, end, exc.reason))
            offset = end
    offset = 0
    while True:
        start = raw.find(b'\xef\xbf\xbd', offset)
        if start < 0:
            break
        item = event(start, start + 3, 'valid UTF-8 encoding of the literal replacement character')
        item['deciding_token_index'] = item['last_byte_token_index']
        result['literal_replacement_events'].append(item)
        offset = start + 3
    final_origins = {e['origin_token_index'] for e in result['invalid_utf8_events'] + result['literal_replacement_events']}
    # This list is diagnostic only; final event attribution above is authoritative.
    result['temporary_prefix_replacement_steps'] = [i for i, text in enumerate(prefixes)
        if '\ufffd' in text and i not in final_origins and unclean.count('\ufffd') < text.count('\ufffd')]
    result['raw_bytes_hex'] = raw.hex()
    return result


def replay_packed(inputs, generated_prefix):
    """Preserve raw IDs, left padding and all original batch rows."""
    ids = torch.cat([inputs['input_ids'], generated_prefix], dim=1)
    mask = torch.cat([inputs['attention_mask'], torch.ones_like(generated_prefix)], dim=1)
    prompt_end = torch.zeros_like(ids, dtype=torch.bool)
    prompt_end[:, inputs['input_ids'].shape[1] - 1] = True
    predictors = torch.zeros_like(prompt_end)
    predictors[:, inputs['input_ids'].shape[1] - 1:] = True
    return {'inputs': {'input_ids': ids, 'attention_mask': mask},
            'prompt_end_mask': prompt_end, 'predictor_mask': predictors}


@contextlib.contextmanager
def observed_steer(bundle, specs, selected_rows, records):
    """Observe around the actual production hooks, without reimplementing edits."""
    before, handles = {}, []
    by_layer = {s.layer: s for s in specs}
    def capture_before(layer):
        def hook(module, incoming, output):
            h = output[0] if isinstance(output, tuple) else output
            before[layer] = (h[:, -1].detach().float().clone(), tuple(h.shape), str(h.dtype))
        return hook
    def capture_after(layer, hooks):
        def hook(module, incoming, output):
            h = output[0] if isinstance(output, tuple) else output
            old, shape, dtype = before.pop(layer)
            new = h[:, -1].detach().float()
            step = hooks[layer].call_count - 1
            spec = by_layer[layer]
            w = spec.W.to(new.device).float().reshape(-1)
            c = spec.center if step == 0 or spec.decode_center is None else spec.decode_center
            c = c.to(new.device).float()
            delta = new - old
            for row in selected_rows:
                norm = float(old[row].norm())
                intended = float(spec.alpha) * torch.dot(old[row] - c, w) * w
                top = torch.topk(delta[row].abs(), min(5, len(w))).indices.tolist()
                records.append({'step': step, 'layer': layer, 'batch_row': row, 'forward_shape': list(shape),
                    'residual_dtype': dtype, 'center_kind': 'prompt' if step == 0 else 'decode',
                    'h_norm': norm, 'kept_delta_norm': float(delta[row].norm()),
                    'kept_relative_norm': float(delta[row].norm()) / max(norm, 1e-12),
                    'intended_delta_norm': float(intended.norm()),
                    'kept_vs_recomputed_formula_error_norm': float((delta[row] - intended).norm()),
                    'coordinate_before': float(torch.dot(old[row], w)), 'coordinate_after': float(torch.dot(new[row], w)),
                    'center_coordinate': float(torch.dot(c, w)), 'weight_norm': float(w.norm()),
                    'largest_changed_dimensions': [{'dimension': i, 'before': float(old[row, i]),
                        'after': float(new[row, i]), 'weight': float(w[i]), 'center': float(c[i])} for i in top]})
        return hook
    try:
        handles += [bundle.layer(l).register_forward_hook(capture_before(l)) for l in by_layer]
        with steer(bundle, specs) as hooks:
            handles += [bundle.layer(l).register_forward_hook(capture_after(l, hooks)) for l in by_layer]
            yield hooks
    finally:
        for handle in handles:
            handle.remove()


def token_scores(logits, tokenizer, actual_id=None, eos_ids=(), top_k=12):
    x = torch.as_tensor(logits).detach().float().cpu()
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite logits in corruption trace')
    lp = torch.log_softmax(x, dim=-1)
    top = torch.topk(x, min(top_k, x.numel())).indices.tolist()
    watched = sorted(set(top + ([] if actual_id is None else [int(actual_id)]) + list(eos_ids)))
    return {'argmax_id': int(x.argmax()), 'top_ids': top,
            'tokens': [{'id': i, 'token_string': tokenizer.convert_ids_to_tokens(i),
                        'logit': float(x[i]), 'log_probability': float(lp[i]),
                        'single_token_decode_not_a_corruption_test': tokenizer.decode([i], skip_special_tokens=False)}
                       for i in watched]}


def common_prefix_replays(bundle, inputs, generated, step, selected_rows, adapter, original, cached_scores, top_k):
    packed = replay_packed(inputs, generated[:, :step])
    outputs = {}
    for name in ('baseline', 'original_das', 'adapted', 'adapted_history_only'):
        current = packed
        if name == 'adapted_history_only':
            current = {**packed, 'prompt_end_mask': packed['prompt_end_mask'].clone(),
                       'predictor_mask': packed['predictor_mask'].clone()}
            current['prompt_end_mask'][:, -1] = False
            current['predictor_mask'][:, -1] = False
        specs = [] if name == 'baseline' else adapter_specs(original if name == 'original_das' else adapter, packed=current)
        with steer(bundle, specs):
            logits = forward_logits(bundle, current['inputs'], logits_to_keep=1)[:, -1].detach().float().cpu()
        outputs[name] = {}
        eos = bundle.model.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        for row in selected_rows:
            info = token_scores(logits[row], bundle.tokenizer, int(generated[row, step]), eos, top_k)
            if name == 'adapted':
                scores = cached_scores[step][row].detach().float().cpu()
                info['cached_processed_score_max_abs_difference'] = float((scores - logits[row]).abs().max())
                info['cached_top1_match'] = int(scores.argmax()) == info['argmax_id']
                info['comparison_limit'] = 'Full-sequence vs cached kernels can differ numerically; cached scores may include declared generation processors.'
            outputs[name][str(row)] = info
    return {'step': step, 'prefix_width': int(packed['inputs']['input_ids'].shape[1]),
            'same_raw_prefix_for_every_variant': True, 'original_batch_preserved': True, 'variants': outputs}


def run(args):
    if args.out.exists():
        raise FileExistsError('trace output already exists; preserve it and choose another path')
    ns, meta, bank, prepared = e27.source_run(args.source_run)
    check_args = argparse.Namespace(run_dir=args.adapter_run, alpha=None, layers=None, init=None,
                                    include_eos=None, stop_token_id=None)
    adapter = e27.verified_adapter(check_args, ns, meta)
    reference_path = args.reference_generation or args.adapter_run / (args.split + '.json')
    reference = qa.read(reference_path)
    if (not reference.get('completed') or reference.get('model') != meta['model']
            or reference.get('split') != args.split
            or reference.get('training_sha256') != digest(args.adapter_run / 'adapter.pt')):
        raise ValueError('reference generation is incomplete or belongs to another adapter/model/split')
    rows = prepared['splits'][args.split]
    qa.verify_question_rows(reference['evaluations'], rows)
    wanted = set(args.pair_ids)
    if len(wanted) != len(args.pair_ids) or not wanted <= {r['pair_id'] for r in rows}:
        raise ValueError('pair IDs must be unique and belong to the requested split')
    batch_size = adapter['config']['batch_size']
    groups = sorted({i // batch_size for i, r in enumerate(rows) if r['pair_id'] in wanted})
    bundle = qa.open_bundle(meta, prepared)
    original = copy.deepcopy(adapter)
    original['W'] = {str(l): torch.as_tensor(bank['E_qa_das_L%d' % l]).float() for l in adapter['layers']}
    eos = bundle.model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    if any(type(i) is not int for i in eos):
        raise ValueError('generation EOS IDs are not explicit integers')
    result = {'protocol': 'e27_corruption_trace_v1', 'model': meta['model'], 'split': args.split,
        'adapter_sha256': digest(args.adapter_run / 'adapter.pt'), 'source': adapter['source'],
        'reference_generation_sha256': digest(reference_path), 'generation_config': bundle.model.generation_config.to_dict(),
        'projection': output_projection_metadata(bundle),
        'formula_error_limit': 'Kept-versus-recomputed-formula error includes dtype casting and dot/matmul accumulation-order differences; it is not an isolated quantization attribution.',
        'model_identity': {'loaded_class': type(bundle.model).__name__, 'bundle_config': bundle.cfg,
            'loaded_config': bundle.model.config.to_dict(),
            'hidden_size': bundle.hidden_size, 'num_layers': bundle.num_layers,
            'output_head_shape': list(bundle.model.get_output_embeddings().weight.shape)},
        'tokenizer': {'class': type(bundle.tokenizer).__name__,
            'name_or_path': bundle.tokenizer.name_or_path, 'vocab_size': len(bundle.tokenizer),
            'special_ids': bundle.tokenizer.all_special_ids,
            'backend_json_sha256': hashlib.sha256(bundle.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            'decoder_state': bundle.tokenizer.backend_tokenizer.decoder.__getstate__().decode()},
        'code_sha256': {str(p.relative_to(CODE)): digest(p) for p in [Path(__file__), CODE/'core/intervene.py',
            CODE/'core/model_io.py', CODE/'core/qa_precision.py', CODE/'core/qa_generation_adapter.py', CODE/'scripts/e27_qa_generation.py']},
        'cases': {}, 'batches': [], 'reference_reproduced': True, 'completed': False}
    with fp32_output_projection(bundle):
        for group in groups:
            start = group * batch_size
            chunk = rows[start:start + batch_size]
            selected = [i for i, r in enumerate(chunk) if r['pair_id'] in wanted]
            texts = [legacy.qa_text(bundle, meta['config'], r['question']) for r in chunk]
            inputs = bundle.encode(texts)
            if not inputs['attention_mask'][:, -1].all():
                raise ValueError('original generation needs left padding')
            specs = adapter_specs(adapter, name=adapter['method'])
            edits = []
            with torch.no_grad(), observed_steer(bundle, specs, selected, edits) as hooks:
                generated = bundle.model.generate(**inputs, max_new_tokens=adapter['config']['max_new_tokens'],
                    do_sample=False, num_beams=1, use_cache=True, pad_token_id=bundle.tokenizer.pad_token_id,
                    return_dict_in_generate=True, output_scores=True)
            new = generated.sequences[:, inputs['input_ids'].shape[1]:]
            if len(generated.scores) != new.shape[1] or any(s.n_applied != new.shape[1] for s in specs):
                raise RuntimeError('generation scores/write counts differ from the observed steps')
            if any(h.first_shape[1] != inputs['input_ids'].shape[1] or h.call_count != new.shape[1] for h in hooks.values()):
                raise RuntimeError('prefill/decode call trace differs from production contract')
            replay_steps = set()
            for i in selected:
                record = chunk[i]; token_ids = new[i].detach().cpu().tolist()
                terminal = next((j for j, token in enumerate(token_ids) if token in eos), None)
                effective = token_ids[:terminal + 1] if terminal is not None else token_ids
                analysis = final_byte_analysis(bundle.tokenizer, effective, eos)
                expected = reference['evaluations'][start + i]['outputs'][adapter['method']]
                identical = analysis['default_decoded'] == expected
                result['reference_reproduced'] &= identical
                events = (analysis['invalid_utf8_events'] + analysis['literal_replacement_events'])[:args.max_events]
                for event in events:
                    replay_steps.update([event['origin_token_index'], event['deciding_token_index']])
                result['cases'][record['pair_id']] = {'question': record['question'], 'batch_row': i,
                    'batch_start': start, 'raw_generated_ids_including_padding': token_ids, 'effective_ids': effective,
                    'stopped_at_step': terminal, 'original_output': expected, 'reference_output_identical': identical,
                    'bytes': analysis, 'edits': [e for e in edits if e['batch_row'] == i],
                    'steps': [{'step': j, 'token': token_scores(scores[i], bundle.tokenizer, token_ids[j], eos, args.top_k)}
                              for j, scores in enumerate(generated.scores) if terminal is None or j <= terminal]}
                print('TRACE', record['pair_id'], 'matches_reference=', identical,
                      'byte_mapping_verified=', analysis['byte_mapping_verified'],
                      'final_bad_utf8=', len(analysis['invalid_utf8_events']),
                      'literal_replacement=', len(analysis['literal_replacement_events']),
                      repr(analysis['default_decoded']), flush=True)
            replays = [common_prefix_replays(bundle, inputs, new, step, selected, adapter, original,
                                             generated.scores, args.top_k) for step in sorted(replay_steps)]
            result['batches'].append({'start': start, 'pair_ids': [r['pair_id'] for r in chunk],
                'prompt_token_ids': inputs['input_ids'].detach().cpu().tolist(),
                'attention_mask': inputs['attention_mask'].detach().cpu().tolist(), 'common_prefix_replays': replays})
            save(args.out, result)
            del generated, new
    result['completed'] = True
    save(args.out, result)
    if not result['reference_reproduced'] and not args.allow_reference_mismatch:
        raise RuntimeError('trace saved, but at least one output differs from the reference; do not attribute its token event to the old run')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--adapter-run', type=Path, required=True)
    p.add_argument('--split', choices=['tune', 'test'], default='tune')
    p.add_argument('--pair-ids', nargs='+', required=True)
    p.add_argument('--reference-generation', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--top-k', type=int, default=12)
    p.add_argument('--max-events', type=int, default=2)
    p.add_argument('--allow-reference-mismatch', action='store_true')
    args = p.parse_args()
    if args.top_k < 1 or args.max_events < 1:
        p.error('top-k and max-events must be positive')
    run(args)


if __name__ == '__main__':
    main()
