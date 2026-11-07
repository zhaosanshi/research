#!/usr/bin/env python3
"""Inference-only source-interchange controls on the existing tuning split.

This is a new deployment diagnostic of frozen original-DAS directions, not a
claim of reproducing CDAS or of having trained generation-specific DAS. The
truth-oriented source instruction never receives a reference answer. All five
conditions and all requested questions are reported; no test-set phase exists.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import torch

import common  # noqa: F401
import e25_qa_das as qa
import e27_qa_generation as adapter_script
from core.qa_generation_adapter import implementation_identity
from core.qa_precision import fp32_output_projection
from core.qa_source_interchange import (PROTOCOL, SourceInterchangeDecoder, greedy_source_generate,
                                        token_digest)
from core.qa_source_layer_plan import resolve_layer_plan, freeze_plan, plan_identity
from core.readout import rule_abstains


SOURCE_SYSTEM = ("Answer factual questions truthfully. Correct false premises rather than accepting them. "
                 "Give a short, direct answer of a few words. If you do not know the answer, say 'I don't know'.")
CONDITIONS = ('baseline', 'source_only', 'equal_logit_mix', 'source_das_swap', 'source_random_swap')


def prompt_tokens(bundle, cfg, question):
    """The sole text input to either forward stream is a question, never gold."""
    if not isinstance(question, str) or not question:
        raise ValueError('a nonempty question is required')
    user = cfg['qa_user'].format(question=question)
    base = bundle.build_chat(cfg['qa_system'], user)
    source = bundle.build_chat(SOURCE_SYSTEM, user)
    ids = []
    for text in (base, source):
        inputs = bundle.encode(text)
        if inputs['input_ids'].shape[0] != 1 or not bool(inputs['attention_mask'].all()):
            raise ValueError('single-question prompts must be unpadded')
        ids.append(inputs['input_ids'][0].tolist())
    return ids


def cache_reference_check(bundle, base, source, layer, w, accepted_tokens, steps=3):
    """Replay identical actual past IDs; compare two implementations before scaling up."""
    cached = SourceInterchangeDecoder(bundle, base, source, mode='source_interchange', layer=layer,
                                      direction=w, use_cache=True)
    full = SourceInterchangeDecoder(bundle, base, source, mode='source_interchange', layer=layer,
                                    direction=w, use_cache=False)
    dtype = next(bundle.model.parameters()).dtype
    # BF16 cached/single-token and full-sequence GEMMs differ numerically even
    # without any intervention. Real-model controls measured 0.21--0.32 logit
    # differences while the corresponding FP32 differences were < 3.2e-5.
    # Record low-precision shape sensitivity, rather than mislabel it a failed
    # causal/cache invariant. Strict real-model validation is performed in FP32.
    strict = dtype == torch.float32
    tolerance = 1e-4 if strict else None
    rows = []
    for token in accepted_tokens[:steps]:
        a, ta = cached.next_logits(); b, tb = full.next_logits()
        diff = float((a - b).abs().max())
        same_top = int(a.argmax()) == int(b.argmax())
        rows.append({'step': len(cached.history), 'max_absolute_logit_difference': diff,
                     'tolerance': tolerance, 'top1_equal': same_top,
                     'strict_float32_check': strict, 'residual_dtype': str(dtype),
                     'accepted_prefix_ids_equal': ta['accepted_prefix_ids'] == tb['accepted_prefix_ids'],
                     'cached_writes': len(ta['writes']), 'full_recomputed_writes': len(tb['writes'])})
        if strict and (diff > tolerance or not same_top):
            raise RuntimeError('cached/full source interchange disagrees: ' + json.dumps(rows[-1]))
        cached.accept(token); full.accept(token)
    return rows


def same_prompt_identity_check(bundle, base, layer, w, accepted_tokens, steps=3):
    baseline = SourceInterchangeDecoder(bundle, base, base, mode='baseline')
    exchange = SourceInterchangeDecoder(bundle, base, base, mode='source_interchange', layer=layer, direction=w)
    checks = []
    for token in accepted_tokens[:steps]:
        a, _ = baseline.next_logits(); b, trace = exchange.next_logits()
        if not torch.equal(a, b) or any(x['stored_delta_norm'] != 0 for x in trace['writes']):
            raise RuntimeError('identical base/source prompts did not yield an exact identity intervention')
        checks.append({'step': len(baseline.history), 'logits_bitwise_equal': True, 'stored_delta_norm': 0.0})
        baseline.accept(token); exchange.accept(token)
    return checks


def inputs(args):
    ns, meta, bank, prepared = adapter_script.source_run(args.source_run)
    frozen = qa.selection(ns, meta)
    selected = frozen['selected']['qa_das']
    if selected is None:
        raise ValueError('source run has no previously frozen original-DAS layer')
    existing = None
    run_dir = getattr(args, 'run_dir', None)
    if run_dir is not None:
        stem = 'check.json' if args.phase in ('check', 'judge-check') else 'tune.json'
        if (Path(run_dir) / stem).exists(): existing = qa.read(Path(run_dir) / stem)
    layer, layer_plan = resolve_layer_plan(args, meta, selected['layer'], SOURCE_SYSTEM,
                                           existing['identity']['layer'] if existing else None)
    if run_dir is not None: freeze_plan(run_dir, layer_plan)
    questions = prepared['splits']['tune']
    if args.pair_ids:
        wanted = list(args.pair_ids)
        if len(wanted) != len(set(wanted)) or not set(wanted) <= {q['pair_id'] for q in questions}:
            raise ValueError('pilot IDs must be unique tuning questions')
        questions = [q for q in questions if q['pair_id'] in set(wanted)]
    elif args.phase in ('check', 'judge-check'):
        questions = questions[:3]
    if not questions:
        raise ValueError('empty tuning cohort')
    w = torch.as_tensor(bank['E_qa_das_L%d' % layer]).float().clone()
    generator = torch.Generator(device='cpu').manual_seed(args.seed + layer)
    random_w = torch.nn.functional.normalize(torch.randn(meta['hidden_size'], generator=generator), dim=0)
    code = implementation_identity()
    code['entry_script_sha256'] = qa.digest(Path(__file__))
    identity = {'protocol': PROTOCOL, 'model': meta['model'], 'source': adapter_script.source_identity(ns, meta, [layer]),
                'source_selection_sha256': qa.digest(ns.run_dir / 'selection.json'),
                'layer': layer, 'layer_choice': 'unchanged existing tuning-frozen original-DAS layer',
                'source_system': SOURCE_SYSTEM, 'base_system': meta['config']['qa_system'],
                'user_template': meta['config']['qa_user'], 'max_new_tokens': meta['config']['max_new_tokens'],
                'batch_size': 1, 'alpha': 1.0, 'fixed_global_center': False,
                'conditions': list(CONDITIONS), 'seed': args.seed, 'random_seed': args.seed + layer,
                'question_ids': [q['pair_id'] for q in questions], 'split': 'tune',
                'training_called': False, 'extra_content_adaptation': False, 'gold_used_at_inference': False,
                'same_generated_prefix_token_ids': True, 'independent_kv_streams': True,
                'direction_sha256': qa.fingerprint(w.tolist()), 'random_direction_sha256': qa.fingerprint(random_w.tolist()),
                'implementation': code, 'exploratory_previously_exposed_tuning_set': True,
                'source_variable': 'fixed instruction policy; source answer is not guaranteed true'}
    if layer_plan is not None:
        identity['layer_plan'] = plan_identity(layer_plan)
        identity['layer_choice'] = 'one of two fixed 50%/75% generation-layer follow-up locations'
    parent_path = getattr(args, 'parent_controls', None)
    if parent_path is None and existing is not None:
        parent_path = existing['identity'].get('parent_control_file')
    if parent_path is not None:
        parent_path = Path(parent_path)
        if layer_plan is not None and qa.digest(parent_path) != layer_plan['plan']['parent_control_raw_sha256'][meta['model']]:
            raise ValueError('parent controls differ from the hash fixed before the layer follow-up')
        parent = qa.read(parent_path); pi = parent['identity']
        current_source = identity['source']
        if (not parent.get('completed') or parent['model'] != meta['model']
                or not set(('baseline', 'source_only', 'equal_logit_mix')) <= set(parent['conditions'])
                or pi['base_system'] != meta['config']['qa_system'] or pi['source_system'] != SOURCE_SYSTEM
                or pi['user_template'] != meta['config']['qa_user'] or pi['batch_size'] != 1
                or pi['max_new_tokens'] != meta['config']['max_new_tokens']
                or any(pi['source'][k] != v for k, v in current_source.items() if k != 'checkpoints')):
            raise ValueError('parent unsteered/source/mix outputs have different model, data or prompt settings')
        for name in ('core/qa_source_interchange.py', 'core/model_io.py', 'core/qa_precision.py'):
            if pi['implementation']['files'][name] != qa.digest(Path(__file__).resolve().parents[1] / name):
                raise ValueError('reused control inference implementation changed: ' + name)
        by_id = {r['pair_id']: r for r in parent['evaluations']}
        if len(by_id) != len(parent['evaluations']) or not {q['pair_id'] for q in questions} <= set(by_id):
            raise ValueError('parent controls do not cover unique requested questions')
        for q in questions:
            if any(by_id[q['pair_id']].get(k) != q.get(k) for k in ('question', 'gold', 'incorrect', 'best_answer')):
                raise ValueError('parent question identity/references differ')
        identity.update(parent_control_file=str(parent_path.resolve()), parent_control_sha256=qa.digest(parent_path),
                        reused_conditions=['baseline', 'source_only', 'equal_logit_mix'],
                        executed_conditions=['source_das_swap', 'source_random_swap'])
    return ns, meta, bank, prepared, questions, layer, w, random_w, identity


def generate(args, state, bundle=None):
    ns, meta, bank, prepared, questions, layer, w, random_w, identity = state
    args.run_dir.mkdir(parents=True, exist_ok=True)
    filename = 'check.json' if args.phase == 'check' else 'tune.json'
    path = args.run_dir / filename
    if path.exists():
        existing = qa.read(path)
        if existing.get('identity') != identity:
            raise ValueError('existing source-interchange run differs; preserve it and use a new directory')
        if existing.get('completed'):
            return existing
        raise ValueError('partial diagnostic exists; preserve it and use a new directory')
    if bundle is None:
        bundle = qa.open_bundle(meta, prepared)
    qa.verify_bundle(bundle, meta, prepared)
    versions = [(p, p._version, p.requires_grad) for p in bundle.model.parameters()]
    choices = [{'name': name, 'kind': 'baseline' if name == 'baseline' else 'E',
                'uses_reference_answer': False, 'eligible_for_selection': False} for name in CONDITIONS]
    result = {'protocol': PROTOCOL, 'model': meta['model'], 'identity': identity,
              'bank_protocol_fingerprint': qa.fingerprint(identity), 'protocol_fingerprint': qa.fingerprint(identity),
              'split': 'tune', 'pilot': args.phase == 'check' or bool(args.pair_ids),
              'wiring_pilot': args.phase == 'check', 'diagnostic_only': False,
              'conditions': list(CONDITIONS), 'condition_specs': choices, 'evaluations': [],
              'intervention_position': 'same-prefix source coordinate at prompt-end and every subsequent answer predictor; see per-row trace',
              'E': {'method': 'original-DAS frozen direction with live same-prefix source coordinate'},
              'C': {'enabled': False}, 'completed': False}
    parent_rows = None
    if 'parent_control_file' in identity:
        parent_path = Path(identity['parent_control_file'])
        if qa.digest(parent_path) != identity['parent_control_sha256']:
            raise ValueError('reused parent control artifact changed')
        parent_rows = {r['pair_id']: r for r in qa.read(parent_path)['evaluations']}
    snapshots = args.run_dir / 'code_snapshot'
    code_root = Path(__file__).resolve().parents[1]
    for relative, expected in identity['implementation']['files'].items():
        target = snapshots / relative; target.parent.mkdir(parents=True, exist_ok=True)
        if qa.digest(code_root / relative) != expected:
            raise RuntimeError('source code changed before diagnostic execution')
        shutil.copyfile(code_root / relative, target)
    shutil.copyfile(__file__, snapshots / 'scripts/e29_qa_source_interchange.py')
    with fp32_output_projection(bundle):
        for q in questions:
            base, source = prompt_tokens(bundle, meta['config'], q['question'])
            row = {**copy.deepcopy(q), 'idx': len(result['evaluations']), 'outputs': {}, 'finished': {}, 'rule_abstain': {},
                   'generation_token_ids': {}, 'trace': {}, 'cache_reference_checks': {},
                   'prompt_token_ids': {'base': base, 'source': source}}
            for name in CONDITIONS:
                mode = 'source_interchange' if name.endswith('_swap') else name
                direction = random_w if name == 'source_random_swap' else w
                if parent_rows is not None and name in identity['reused_conditions']:
                    parent = parent_rows[q['pair_id']]
                    if parent['prompt_token_ids'] != {'base': base, 'source': source}:
                        raise ValueError('reused control prompt token IDs differ from the current model')
                    output = {'output': parent['outputs'][name], 'finished': parent['finished'][name],
                              'token_ids': copy.deepcopy(parent['generation_token_ids'][name]),
                              'trace': copy.deepcopy(parent['trace'][name])}
                    row.setdefault('reused_control_provenance', {})[name] = {
                        'parent_raw_sha256': identity['parent_control_sha256'], 'pair_id': q['pair_id'],
                        'prompt_token_ids_verified_identical': True, 'generated_again': False}
                else:
                    output = greedy_source_generate(bundle, base, source, mode=mode, layer=layer, direction=direction,
                                                     max_new_tokens=identity['max_new_tokens'])
                row['outputs'][name] = output['output']; row['finished'][name] = output['finished']
                row['rule_abstain'][name] = bool(rule_abstains(output['output']))
                row['generation_token_ids'][name] = output['token_ids']; row['trace'][name] = output['trace']
                if name == 'baseline' and args.phase == 'check':
                    row['same_prompt_identity_checks'] = same_prompt_identity_check(bundle, base, layer, w, output['token_ids'])
                if name.endswith('_swap') and args.phase == 'check':
                    row['cache_reference_checks'][name] = cache_reference_check(
                        bundle, base, source, layer, direction, output['token_ids'])
                print('SOURCE_INTERCHANGE', meta['model'], q['pair_id'], name, repr(output['output']), flush=True)
            result['evaluations'].append(row)
            qa.save(path, result)
    if any(p._version != version or p.requires_grad != flag for p, version, flag in versions):
        raise RuntimeError('inference-only diagnostic changed base-model parameters')
    if qa.fingerprint(w.tolist()) != identity['direction_sha256']:
        raise RuntimeError('frozen original-DAS direction changed')
    result.update(completed=True, base_parameters_unchanged=True, frozen_direction_unchanged=True)
    qa.save(path, result)
    return result


def freeze_reused_control_grades(args, raw, stem):
    identity = raw['identity']
    if 'parent_control_file' not in identity: return
    parent_path = Path(identity['parent_control_file'])
    if qa.digest(parent_path) != identity['parent_control_sha256']:
        raise ValueError('reused parent generation changed before grading')
    parent = qa.read(parent_path)
    scored_path = parent_path.with_name(parent_path.stem + '_scored_items.json')
    scored = qa.read(scored_path)
    cfg = adapter_script.validate_judge_experiment(args.judge_exp)
    if (not scored.get('completed') or not scored.get('grading_completed')
            or scored.get('grading_source_generation_sha256') != qa.digest(parent_path)
            or scored.get('grading_judge_config_sha256') != qa.fingerprint(cfg)):
        raise ValueError('reused controls need complete original grades with unchanged judge settings')
    qa.verify_question_rows(scored['evaluations'], parent['evaluations'])
    by_id = {r['pair_id']: r for r in scored['evaluations']}
    selected = [by_id[r['pair_id']] for r in raw['evaluations']]
    conditions = identity['reused_conditions']
    for original, current in zip(selected, raw['evaluations']):
        if any(original.get(k) != current.get(k) for k in ('pair_id', 'question', 'gold', 'incorrect', 'best_answer')):
            raise ValueError('reused control question/reference changed before grading')
        for c in conditions:
            if original['outputs'][c] != current['outputs'][c] or original['finished'][c] != current['finished'][c]:
                raise ValueError('reused control response/completion flag changed before grading')
    from core.qa_control_cache import seed_control_cache
    seed = seed_control_cache(source_dir=args.controls_judge_cache, run_dir=args.run_dir,
                              judge_config=cfg, scored_rows=selected, conditions=conditions,
                              scored_sha256=qa.digest(scored_path), stem=stem)
    for name, value in seed.items(): setattr(args, name, value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('check', 'generate-tune', 'judge-check', 'judge-tune'))
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pair-ids', nargs='+')
    p.add_argument('--layer', type=int, help='explicit layer-only tuning follow-up; requires the finite frozen plan')
    p.add_argument('--layer-plan', type=Path)
    p.add_argument('--parent-controls', type=Path,
                   help='reuse only baseline/source-only/logit-mix from this raw artifact; both new-layer swaps are executed')
    p.add_argument('--controls-judge-cache', type=Path)
    p.add_argument('--judge-exp', default='E4_qa')
    args = p.parse_args()
    state = inputs(args)
    if args.phase.startswith('judge-'):
        _, meta, _, _, _, _, _, _, identity = state
        args.model = meta['model']; adapter_script.validate_judge_experiment(args.judge_exp)
        stem = args.phase.split('-', 1)[1]
        raw = qa.read(args.run_dir / (stem + '.json'))
        recorded = raw.get('identity', {})
        # Grading code may be repaired after generation. Verify and retain the
        # actual archived generation code; never relabel old output as produced
        # by today's code. Everything affecting the scientific run still binds.
        if recorded != {**identity, 'implementation': recorded.get('implementation')}:
            raise ValueError('generation source/cohort/settings differ from the requested grading run')
        for relative, expected in recorded['implementation']['files'].items():
            if qa.digest(args.run_dir / 'code_snapshot' / relative) != expected:
                raise ValueError('archived generation implementation was changed')
        if qa.digest(args.run_dir / 'code_snapshot/scripts/e29_qa_source_interchange.py') != recorded['implementation']['entry_script_sha256']:
            raise ValueError('archived generation entry script was changed')
        qa.save(args.run_dir / (stem + '_grading_implementation.json'),
                {'source_generation_sha256': qa.digest(args.run_dir / (stem + '.json')),
                 'grading_implementation': identity['implementation'],
                 'generation_implementation': recorded['implementation'],
                 'note': 'Generation bytes unchanged; missing/duplicate display indices are normalized only in the derived grading copy.'})
        freeze_reused_control_grades(args, raw, stem)
        qa.judge(args, {**meta, 'protocol_fingerprint': qa.fingerprint(recorded)}, stem + '.json', stem + '_scored')
    else:
        generate(args, state)


if __name__ == '__main__':
    main()
