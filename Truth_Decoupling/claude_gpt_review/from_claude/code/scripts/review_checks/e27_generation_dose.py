#!/usr/bin/env python3
"""Frozen-weight deployment-dose diagnostic, tuning questions only.

The adapter was trained at alpha=3. Its weights/layer/centres stay unchanged;
deployment at alpha=1/2 is an explicitly labelled diagnostic override, NOT a
newly trained method. No EOS adjustment, token filtering or repair is used.
Original batches are preserved even when only three selected rows are retained.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / 'scripts'), str(CODE)]
import common  # noqa: F401,E402
import e25_qa_das as qa  # noqa: E402
import e25_qa_context as legacy  # noqa: E402
import e27_qa_generation as e27  # noqa: E402
from core.qa_generation_adapter import adapter_specs, digest, fingerprint  # noqa: E402
from core.qa_precision import fp32_output_projection  # noqa: E402
from core.readout import rule_abstains  # noqa: E402

PROTOCOL = 'e27_frozen_generation_dose_diagnostic_v1'
DEPLOYMENT_ALPHAS = (1., 2., 3.)


def snapshot_code(run_dir):
    """Bind executable source bytes, without claiming an unstored Git revision."""
    paths = list((CODE / 'core').glob('*.py')) + [CODE / 'scripts' / p for p in (
        'common.py', 'e24_real_qa.py', 'e25_qa_context.py', 'e25_qa_das.py', 'e27_qa_generation.py')]
    paths.append(Path(__file__).resolve())
    files = {str(p.relative_to(CODE)): digest(p) for p in sorted(set(paths))}
    manifest = {'sha256': files, 'fingerprint': fingerprint(files)}
    out = Path(run_dir) / 'code_snapshot'
    existing = out / 'manifest.json'
    if existing.exists() and qa.read(existing) != manifest:
        raise ValueError('diagnostic code changed; preserve this run and use a new directory')
    for p in paths:
        dst = out / p.relative_to(CODE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if digest(dst) != files[str(p.relative_to(CODE))]:
                raise ValueError('stored executable snapshot changed')
        else:
            shutil.copy2(p, dst)
    qa.save(existing, manifest)
    return manifest


def condition_specs(adapter):
    if adapter['alpha'] != 3.:
        raise ValueError('this diagnostic fixes the saved training alpha to 3')
    return [dict(qa.BASELINE), {
        'name': 'original_das_alpha3', 'kind': 'E', 'context': 'original_das',
        'training_alpha': None, 'deployment_alpha': 3., 'mode': 'continuous',
        'uses_reference_answer': False, 'eligible_for_selection': False}] + [
        {'name': 'adapted_deploy_alpha%d' % a, 'kind': 'E', 'context': 'generation_adapter',
         'training_alpha': 3., 'deployment_alpha': a, 'mode': 'continuous',
         'diagnostic_alpha_override': a != 3., 'uses_reference_answer': False,
         'eligible_for_selection': False} for a in DEPLOYMENT_ALPHAS]


def runtime_specs(adapter, bank, condition):
    if condition['name'] == 'baseline':
        return []
    view = adapter
    if condition['context'] == 'original_das':
        view = copy.deepcopy(adapter)
        view['W'] = {str(l): torch.as_tensor(bank['E_qa_das_L%d' % l]).float() for l in adapter['layers']}
    # Validate/create using its actual saved training identity. The only
    # deliberate override is on fresh, ephemeral inference SteerSpecs.
    specs = adapter_specs(view, expected_alpha=3., name=condition['name'])
    for spec in specs:
        spec.alpha = condition['deployment_alpha']
    return specs


def target_questions(prepared, pair_ids=None, all_tune=False):
    if all_tune and pair_ids or not all_tune and not pair_ids:
        raise ValueError('select explicit pair IDs or all tuning questions')
    rows = prepared['splits']['tune']
    if all_tune:
        return list(rows)
    wanted = set(pair_ids)
    if len(wanted) != len(pair_ids) or not wanted <= {r['pair_id'] for r in rows}:
        raise ValueError('pair IDs must be unique and belong to tuning, never test')
    return [r for r in rows if r['pair_id'] in wanted]


def identity(args, ns, meta, adapter, questions, code):
    source = e27.source_identity(ns, meta, adapter['layers'])
    checkpoint_hash = digest(args.adapter_run / 'adapter.pt')
    signature = fingerprint([PROTOCOL, source, checkpoint_hash, condition_specs(adapter),
                             [q['pair_id'] for q in questions], code['fingerprint']])
    return {'protocol_fingerprint': signature, 'training_sha256': checkpoint_hash,
            'source': source, 'code_snapshot_fingerprint': code['fingerprint']}


def generate(args, ns, meta, bank, prepared, adapter, bundle=None):
    questions = target_questions(prepared, args.pair_ids, args.all_tune)
    code = snapshot_code(args.run_dir)
    ident = identity(args, ns, meta, adapter, questions, code)
    choices = condition_specs(adapter)
    filename = 'tune.json' if args.all_tune else 'pilot.json'
    path = args.run_dir / filename
    reference_path = args.adapter_run / 'tune.json'
    reference = qa.read(reference_path)
    if (not reference.get('completed') or reference['training_sha256'] != ident['training_sha256']
            or reference['model'] != meta['model'] or reference['split'] != 'tune'):
        raise ValueError('reference is not the completed tuning generation of this adapter')
    qa.verify_question_rows(reference['evaluations'], prepared['splits']['tune'])
    if path.exists():
        data = qa.read(path)
        if (data['protocol_fingerprint'] != ident['protocol_fingerprint']
                or data['condition_specs'] != choices
                or data['reference_generation_sha256'] != digest(reference_path)):
            raise ValueError('resume weights, code, source, conditions or cohort changed')
        qa.verify_question_rows(data['evaluations'], questions)
    else:
        data = {'model': meta['model'], 'protocol': PROTOCOL, **ident,
            'bank_protocol_fingerprint': ident['protocol_fingerprint'], 'bank_sha256': meta['bank_sha256'],
            'questions_sha256': meta['questions_sha256'], 'split': 'tune', 'pilot': not args.all_tune,
            'diagnostic_only': False, 'diagnostic_experiment': True,
            'diagnostic_scope': 'frozen-weight deployment-dose diagnostic, not a trained-at-new-alpha method',
            'precision': meta['precision'], 'prompt': adapter['config'],
            'reference_generation_sha256': digest(reference_path),
            'intervention_position': 'same continuous prompt/decode operator; saved layer/W/centres; deliberate deployment alpha override only',
            'E': {'method': 'frozen-generation-adapter-dose-diagnostic', 'training_alpha': 3.,
                  'deployment_alphas': list(DEPLOYMENT_ALPHAS), 'not_a_retrained_method': True},
            'C': {'enabled': False}, 'conditions': [c['name'] for c in choices], 'condition_specs': choices,
            'audit': [], 'reference_reproduced': True, 'completed': False,
            'evaluations': [{**q, 'idx': i, 'outputs': {}, 'finished': {}, 'rule_abstain': {},
                            'raw_token_ids': {}, 'quality': {}, 'edit_summary': {}} for i, q in enumerate(questions)]}
    if bundle is None:
        bundle = qa.open_bundle(meta, prepared)
    else:
        qa.verify_bundle(bundle, meta, prepared)
    all_rows = prepared['splits']['tune']; cfg = {**meta['config'], **adapter['config']}
    wanted = {q['pair_id'] for q in questions}; saved = {r['pair_id']: r for r in data['evaluations']}
    eos = getattr(bundle.model.generation_config, 'eos_token_id', None)
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos if eos is not None else bundle.tokenizer.eos_token_id])
    reference_alias = {'baseline': 'baseline', 'original_das_alpha3': 'original_das',
                       'adapted_deploy_alpha3': adapter['method']}
    bs = cfg['batch_size']
    with fp32_output_projection(bundle):
        for condition in choices:
            name = condition['name']
            for start in range(0, len(all_rows), bs):
                chunk = all_rows[start:start + bs]
                selected = [i for i, q in enumerate(chunk) if q['pair_id'] in wanted]
                if not selected or all(name in saved[chunk[i]['pair_id']]['outputs'] for i in selected):
                    continue
                texts = [legacy.qa_text(bundle, cfg, q['question']) for q in chunk]
                inputs = bundle.encode(texts); plen = inputs['input_ids'].shape[1]
                specs = runtime_specs(adapter, bank, condition)
                tokens, hooks, displacement = qa.generate_with_audit(bundle, inputs, specs, cfg, trace_displacement=True)
                new_tokens = tokens[:, plen:]; n_new = new_tokens.shape[1]
                if any(s.n_applied != n_new for s in specs) or any(h.first_shape[1] != plen or h.call_count != n_new for h in hooks.values()):
                    raise RuntimeError('generation edit count/position mismatch')
                for i in selected:
                    q = chunk[i]; row = saved[q['pair_id']]
                    ids = new_tokens[i].detach().cpu().tolist()
                    answer = bundle.tokenizer.decode(ids, skip_special_tokens=True)
                    if name in row['outputs'] and row['outputs'][name] != answer:
                        raise RuntimeError('resumed batch changed an existing output')
                    row['outputs'][name] = answer
                    row['raw_token_ids'][name] = ids
                    stop = next((j for j, token in enumerate(ids) if token in eos), None)
                    row['finished'][name] = stop is not None
                    row['rule_abstain'][name] = bool(rule_abstains(answer))
                    row['quality'][name] = {'replacement_character': '\ufffd' in answer,
                                            'unfinished': stop is None, 'characters': len(answer)}
                    actual = [r['kept_relative_norm'][i] for r in displacement if stop is None or r['step'] <= stop]
                    row['edit_summary'][name] = {'max_actual_relative_edit': max(actual) if actual else 0.,
                        'mean_actual_relative_edit': sum(actual) / len(actual) if actual else 0.,
                        'observations': len(actual)}
                    if name in reference_alias:
                        expected = reference['evaluations'][start + i]['outputs'][reference_alias[name]]
                        matches = answer == expected
                        data['reference_reproduced'] &= matches
                        row.setdefault('reference_matches', {})[name] = matches
                    if not args.all_tune:
                        print('DOSE_PILOT', q['pair_id'], name, repr(answer), row['quality'][name], flush=True)
                data['audit'].append({'condition': name, 'batch_start': start,
                    'original_batch_pair_ids': [q['pair_id'] for q in chunk], 'prompt_length': plen,
                    'new_tokens': n_new, 'edits': {s.name: s.n_applied for s in specs},
                    'selected_rows': selected, 'displacement': displacement})
                qa.save(path, data)
    data['completed'] = True
    qa.save(path, data)
    if not data['reference_reproduced']:
        raise RuntimeError('saved diagnostic, but baseline/original/alpha3 failed exact reference text reproduction')
    return data


def judge(args, ns, meta, prepared, adapter):
    questions = target_questions(prepared, args.pair_ids, args.all_tune)
    code = snapshot_code(args.run_dir)
    ident = identity(args, ns, meta, adapter, questions, code)
    name = 'tune' if args.all_tune else 'pilot'
    data = qa.read(args.run_dir / (name + '.json'))
    if not data['reference_reproduced'] or data['protocol_fingerprint'] != ident['protocol_fingerprint']:
        raise ValueError('only matching successfully reproduced generation can be graded')
    e27.validate_judge_experiment(args.judge_exp)
    qa.judge(args, {**meta, **ident}, name + '.json', name + '_scored')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['generate', 'judge'])
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--adapter-run', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    cohort = p.add_mutually_exclusive_group(required=True)
    cohort.add_argument('--pair-ids', nargs='+')
    cohort.add_argument('--all-tune', action='store_true')
    p.add_argument('--judge-exp', default='E4_qa')
    args = p.parse_args()
    ns, meta, bank, prepared = e27.source_run(args.source_run)
    args.model = meta['model']
    loader = argparse.Namespace(run_dir=args.adapter_run, alpha=None, layers=None, init=None,
                                include_eos=None, stop_token_id=None)
    adapter = e27.verified_adapter(loader, ns, meta)
    if args.phase == 'generate':
        generate(args, ns, meta, bank, prepared, adapter)
    else:
        judge(args, ns, meta, prepared, adapter)


if __name__ == '__main__':
    main()
