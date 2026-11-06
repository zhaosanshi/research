#!/usr/bin/env python3
"""Read-only E27 controls: frozen Gaussian before training versus after training.

No optimizer, training or parameter update is called by this script. Conditions:
baseline, original DAS, saved frozen random initial_W, trained random, trained
DAS. Layers, alpha, centres, prompts, precision and source batch boundaries are
matched. --limit restricts stored/scored questions; model batches replay the
source run's complete batches to avoid changing bf16 batch numerical context.
Reference answers are only consumed by the separate judge phase.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / 'scripts'), str(CODE)]
import e25_qa_das as qa
import e25_qa_context as legacy
import e27_qa_generation as e27
from core.qa_generation_adapter import adapter_specs, load_adapter
from core.qa_precision import fp32_output_projection
from core.readout import rule_abstains

PROTOCOL = 'e27_frozen_initial_controls_v1'
CONDITIONS = ('baseline', 'original_das', 'frozen_random_initial', 'trained_random', 'trained_das')
CODE_FILES = ('core/qa_generation_adapter.py', 'core/intervene.py', 'core/model_io.py',
              'core/qa_precision.py', 'core/qa_context.py', 'scripts/e25_qa_das.py',
              'scripts/e27_qa_generation.py', 'scripts/review_checks/e27_frozen_initial_controls.py')


def tensor_hash(value):
    x = value.detach().float().cpu().contiguous()
    return hashlib.sha256(json.dumps({'shape': list(x.shape), 'dtype': 'float32'}, sort_keys=True).encode()
                          + x.numpy().tobytes()).hexdigest()


def inputs(args):
    ns, meta, bank, prepared = e27.source_run(args.source_run)
    r = load_adapter(args.random_run, expected_model=meta['model'])
    d = load_adapter(args.das_run, expected_model=meta['model'])
    if r['init'] != 'random' or d['init'] != 'das':
        raise ValueError('random-run/das-run initialization identities are wrong')
    for key in ('model_config', 'config', 'source', 'layers', 'alpha', 'operator', 'mode',
                'positions', 'training_pair_ids', 'training_tokens_sha256', 'train_config'):
        if r[key] != d[key]:
            raise ValueError('control setup differs: ' + key)
    if r['source'] != e27.source_identity(ns, meta, r['layers']):
        raise ValueError('source artifacts changed since the saved adapters')
    if r['config'] != {k: meta['config'][k] for k in r['config']}:
        raise ValueError('source prompts or generation budget differ')
    vectors = {}
    for name in CONDITIONS[1:]:
        a = copy.deepcopy(r)
        if name == 'original_das':
            a['W'] = {str(l): torch.as_tensor(bank['E_qa_das_L%d' % l]).float().clone() for l in r['layers']}
        elif name == 'frozen_random_initial':
            a['W'] = {str(l): r['initial_W'][str(l)].detach().float().clone() for l in r['layers']}
        elif name == 'trained_das':
            a['W'] = {str(l): d['W'][str(l)].detach().float().clone() for l in r['layers']}
        for l in r['layers']:
            for phase in ('prompt', 'decode'):
                if not torch.equal(r['centers'][str(l)][phase], d['centers'][str(l)][phase]):
                    raise ValueError('DAS and random centres differ')
        vectors[name] = a
    identity = {'source': r['source'], 'source_run': str(ns.run_dir.resolve()),
        'random_adapter_sha256': qa.digest(Path(args.random_run) / 'adapter.pt'),
        'das_adapter_sha256': qa.digest(Path(args.das_run) / 'adapter.pt'),
        'layers': r['layers'], 'alpha': r['alpha'], 'config': r['config'],
        'vector_sha256': {n: {str(l): tensor_hash(a['W'][str(l)]) for l in a['layers']} for n, a in vectors.items()},
        'code_sha256': {name: qa.digest(CODE / name) for name in CODE_FILES},
        'frozen_random_definition': 'Saved Gaussian initial_W, zero optimizer updates for the evaluated vector; shares DAS-selected layers and data-estimated centres.',
        'trained_random_definition': 'Same saved initial_W after correct-answer content training; this is not an untrained null.',
        'training_called_by_this_script': False}
    identity['cached_reference_generations'] = {}
    for tag, directory, adapter in [('random', Path(args.random_run), r), ('das', Path(args.das_run), d)]:
        path = directory / (args.split + '.json')
        raw = qa.read(path)
        if (not raw.get('completed') or raw.get('model') != meta['model'] or raw.get('split') != args.split
                or raw.get('training_sha256') != qa.digest(directory / 'adapter.pt')
                or raw.get('source') != adapter['source']):
            raise ValueError('cached reference generation does not belong to this adapter/source')
        qa.verify_question_rows(raw['evaluations'], prepared['splits'][args.split])
        identity['cached_reference_generations'][tag] = {'path': str(path.resolve()), 'sha256': qa.digest(path)}
    return ns, meta, bank, prepared, vectors, identity


def generation_meta(meta, identity):
    return {**meta, 'protocol_fingerprint': qa.fingerprint(identity),
            'training_sha256': qa.fingerprint(identity['vector_sha256'])}


def cached_references(args, split):
    references = {}
    for tag, directory in [('random', Path(args.random_run)), ('das', Path(args.das_run))]:
        path = directory / (split + '.json')
        if not path.exists():
            raise ValueError('matched source generation missing: ' + str(path))
        data = qa.read(path)
        if not data.get('completed'):
            raise ValueError('matched source generation incomplete')
        references[tag] = {r['pair_id']: r for r in data['evaluations']}
    return references


def generate(args, ns, meta, prepared, vectors, identity, bundle=None):
    cfg = meta['config']; all_questions = prepared['splits'][args.split]
    limit = len(all_questions) if args.limit is None else args.limit
    if not 1 <= limit <= len(all_questions):
        raise ValueError('limit must fit source split')
    questions = all_questions[:limit]
    names = list(CONDITIONS)
    signature = qa.fingerprint([identity, args.split, [r['pair_id'] for r in questions],
                                'full_source_batches', cfg['batch_size']])
    path = args.run_dir / (args.split + '.json')
    choices = [dict(qa.BASELINE)] + [{'name': n, 'kind': 'control', 'uses_reference_answer': False,
        'eligible_for_selection': False, 'layers': identity['layers'], 'alpha': identity['alpha'],
        'mode': 'continuous'} for n in names[1:]]
    emeta = generation_meta(meta, identity)
    if path.exists():
        data = qa.read(path)
        if data.get('protocol_fingerprint') != signature:
            raise ValueError('resume code, vector, setup or cohort changed; use a new run')
        qa.verify_question_rows(data['evaluations'], questions)
        if any(not x['exact_text_equal'] or not x['finished_equal'] for x in data.get('cached_reproduction', [])):
            raise ValueError('a saved reference failed reproduction; inspect it and use a new run after resolving the cause')
    else:
        data = {'model': meta['model'], 'protocol': PROTOCOL, 'protocol_fingerprint': signature,
            'bank_protocol_fingerprint': emeta['protocol_fingerprint'], 'training_sha256': emeta['training_sha256'],
            'questions_sha256': meta['questions_sha256'], 'split': args.split,
            'pilot': limit < len(all_questions), 'diagnostic_only': False, 'identity': identity,
            'precision': meta['precision'], 'prompt': {k: cfg[k] for k in ('qa_system', 'qa_user', 'max_new_tokens')},
            'E': {'control': 'frozen versus content-trained vectors; no training in this script'}, 'C': {'enabled': False},
            'intervention_position': 'prefill last prompt token and every later predictor; same centres and alpha in all vector arms',
            'conditions': names, 'condition_specs': choices, 'audit': [], 'cached_reproduction': [],
            'evaluations': [{**q, 'idx': i, 'outputs': {}, 'finished': {}, 'rule_abstain': {},
                             'prompt_token_ids': {}, 'generation_token_ids': {}} for i, q in enumerate(questions)],
            'completed': False}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for rel, digest in identity['code_sha256'].items():
        dest = args.run_dir / 'code_snapshot' / rel
        if dest.exists() and qa.digest(dest) != digest:
            raise ValueError('saved code snapshot changed')
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(CODE / rel, dest)
    qa.save(path, data)
    if any(qa.digest(row['path']) != row['sha256'] for row in identity['cached_reference_generations'].values()):
        raise ValueError('cached reference generation changed since identity was loaded')
    references = cached_references(args, args.split)
    if bundle is None:
        bundle = qa.open_bundle(meta, prepared)
    else:
        qa.verify_bundle(bundle, meta, prepared)
    versions = [(p, p._version, p.requires_grad) for p in bundle.model.parameters()]
    eos = getattr(bundle.model.generation_config, 'eos_token_id', bundle.tokenizer.eos_token_id)
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    mismatch = []
    with fp32_output_projection(bundle):
        for name in names:
            for start in range(0, limit, cfg['batch_size']):
                stored = data['evaluations'][start:min(start + cfg['batch_size'], limit)]
                if all(name in r['outputs'] for r in stored):
                    continue
                # Replay complete original batches even when only three rows are audited.
                context = all_questions[start:start + cfg['batch_size']]
                texts = [legacy.qa_text(bundle, cfg, row['question']) for row in context]
                packed = bundle.encode(texts)
                if not packed['attention_mask'][:, -1].all():
                    raise ValueError('left-padded source prompts required')
                specs = [] if name == 'baseline' else adapter_specs(vectors[name], expected_alpha=identity['alpha'], name=name)
                width = packed['input_ids'].shape[1]
                generated, hooks, trace = qa.generate_with_audit(bundle, packed, specs, cfg,
                                                                 trace_displacement=bool(data['pilot']))
                steps = generated.shape[1] - width
                if any(s.n_applied != steps for s in specs) or any(h.call_count != steps or h.first_shape[1] != width for h in hooks.values()):
                    raise RuntimeError('generation write audit failed')
                for row, tokens, ids, mask in zip(stored, generated[:, width:], packed['input_ids'], packed['attention_mask']):
                    answer = bundle.tokenizer.decode(tokens, skip_special_tokens=True)
                    row['outputs'][name] = answer
                    row['finished'][name] = any(int(t) in eos for t in tokens)
                    row['rule_abstain'][name] = bool(rule_abstains(answer))
                    row['generation_token_ids'][name] = tokens.detach().cpu().tolist()
                    row['prompt_token_ids'][name] = ids[mask.bool()].detach().cpu().tolist()
                    if name != 'frozen_random_initial':
                        group, old_name = ('das', 'das_generation_adapter') if name == 'trained_das' else (
                            'random', 'random_generation_adapter' if name == 'trained_random' else name)
                        old = references[group][row['pair_id']]
                        equal = answer == old['outputs'][old_name]
                        data['cached_reproduction'].append({'pair_id': row['pair_id'], 'condition': name,
                            'exact_text_equal': equal, 'finished_equal': row['finished'][name] == old['finished'][old_name]})
                        if not equal or row['finished'][name] != old['finished'][old_name]:
                            mismatch.append({'pair_id': row['pair_id'], 'condition': name,
                                'saved_text': old['outputs'][old_name], 'reproduced_text': answer})
                    if data['pilot']:
                        print('PILOT', row['pair_id'], name, repr(answer), flush=True)
                data['audit'].append({'condition': name, 'batch_start': start,
                    'source_batch_ids': [r['pair_id'] for r in context], 'stored_ids': [r['pair_id'] for r in stored],
                    'prompt_len': width, 'new_tokens': steps, 'edits': {s.name: s.n_applied for s in specs},
                    'displacement': trace})
                qa.save(path, data)
    if any(p._version != version or p.requires_grad != flag for p, version, flag in versions):
        raise RuntimeError('base parameter state changed during read-only generation')
    data['base_parameter_versions_and_flags_unchanged'] = True
    data['reproduction_mismatches'] = mismatch
    data['completed'] = not mismatch
    qa.save(path, data)
    if mismatch:
        raise RuntimeError('saved controls did not reproduce; raw outputs saved for diagnosis; do not interpret the new null yet')
    print('FROZEN_INITIAL_CONTROL_GENERATION_DONE', args.split, len(questions), flush=True)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['generate', 'judge'])
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--random-run', type=Path, required=True)
    p.add_argument('--das-run', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--split', choices=['tune', 'test'], default='tune')
    p.add_argument('--limit', type=int, default=3, help='stored/scored questions; full original batches are replayed')
    p.add_argument('--all-questions', action='store_true', help='explicitly expand after the three-question acceptance')
    p.add_argument('--judge-exp', default='E4_qa')
    args = p.parse_args()
    if args.all_questions:
        args.limit = None
    ns, meta, bank, prepared, vectors, identity = inputs(args)
    args.model = meta['model']
    if args.phase == 'generate':
        generate(args, ns, meta, prepared, vectors, identity)
    else:
        e27.validate_judge_experiment(args.judge_exp)
        qa.judge(args, generation_meta(meta, identity), args.split + '.json', args.split + '_scored')


if __name__ == '__main__':
    main()
