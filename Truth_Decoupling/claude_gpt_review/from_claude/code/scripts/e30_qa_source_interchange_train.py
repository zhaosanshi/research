#!/usr/bin/env python3
"""Train a bounded supervised source interchange and evaluate on tuning only.

Training changes only a unit direction, at the original frozen DAS layer. The
source sees a fixed truthful instruction and the same gold past-token IDs while
training, and the same generated past-token IDs at inference. No test phase.
"""
import argparse
import copy
from pathlib import Path
import shutil

import torch

import common  # noqa: F401
import e25_qa_das as qa
import e27_qa_generation as legacy_adapter
import e29_qa_source_interchange as source_eval
from core import qa_source_interchange_training as training
from core.qa_generation_adapter import implementation_identity
from core.qa_precision import fp32_output_projection
from core.qa_source_interchange import greedy_source_generate
from core.qa_source_layer_plan import resolve_layer_plan, freeze_plan, plan_identity
from core.readout import rule_abstains


def source_state(args):
    ns, meta, bank, prepared = legacy_adapter.source_run(args.source_run)
    choice = qa.selection(ns, meta)['selected']['qa_das']
    if choice is None: raise ValueError('source has no previously frozen original-DAS layer')
    stored_layer = None
    if args.phase != 'train' and (args.run_dir / 'source_adapter.pt').exists():
        stored_layer = training.load(args.run_dir)['layer']
    layer, plan = resolve_layer_plan(args, meta, choice['layer'], source_eval.SOURCE_SYSTEM, stored_layer)
    freeze_plan(args.run_dir, plan)
    args._layer_plan = plan
    return ns, meta, bank, prepared, layer


def train(args, state, bundle=None):
    ns, meta, bank, prepared, layer = state
    records = prepared['e_train']
    if args.pilot is not None:
        if not 1 <= args.pilot <= len(records): raise ValueError('pilot must fit the original estimation cohort')
        records = records[:args.pilot]
    source = legacy_adapter.source_identity(ns, meta, [layer])
    identity = {'source': source, 'pilot': bool(meta['pilot'] or args.pilot is not None),
                'training_entry_sha256': qa.digest(__file__),
                'layer_selection': 'unchanged previously frozen original-DAS tuning choice',
                'estimation_cohort_size': len(prepared['e_train'])}
    if getattr(args, '_layer_plan', None) is not None:
        if len(prepared['e_train']) != args._layer_plan['plan']['training']['pairs']:
            raise ValueError('estimation cohort size differs from the frozen layer plan')
        identity['layer_plan'] = plan_identity(args._layer_plan)
        identity['layer_selection'] = 'one of two fixed 50%/75% generation-layer follow-up locations'
    if (args.run_dir / 'source_adapter.pt').exists() or (args.run_dir / 'source_adapter.json').exists():
        saved = training.load(args.run_dir)
        wanted = {'epochs': args.epochs, 'lr': args.lr, 'clip': args.clip, 'seed': args.seed,
                  'optimizer': 'Adam', 'alpha': 1.0, 'include_eos': False, 'fixed_center': False}
        if (saved['source'] != source or saved['layer'] != layer or saved['train_config'] != wanted
                or saved['training_pair_ids'] != [r['pair_id'] for r in records] or saved['initialization'] != args.init
                or saved['source_system'] != source_eval.SOURCE_SYSTEM):
            raise ValueError('existing adapter does not match the requested training; use a new directory')
        return saved
    if bundle is None: bundle = qa.open_bundle(meta, prepared)
    qa.verify_bundle(bundle, meta, prepared)
    result = training.train(bundle, records, meta['config'], bank['E_qa_das_L%d' % layer],
                            layer=layer, source_system=source_eval.SOURCE_SYSTEM, epochs=args.epochs,
                            lr=args.lr, clip=args.clip, seed=args.seed, init=args.init, identity=identity)
    training.save(result, args.run_dir)
    code_root = Path(__file__).resolve().parents[1]
    for name, expected in result['training_implementation']['files'].items():
        path = args.run_dir / 'training_code_snapshot' / name; path.parent.mkdir(parents=True, exist_ok=True)
        if qa.digest(code_root / name) != expected: raise RuntimeError('training implementation changed before archival')
        shutil.copyfile(code_root / name, path)
    if qa.digest(__file__) != result['training_entry_sha256']:
        raise RuntimeError('training entry script changed during training')
    shutil.copyfile(__file__, args.run_dir / 'training_code_snapshot/scripts/e30_qa_source_interchange_train.py')
    training.verify_training_snapshot(args.run_dir, result)
    return result


def verified_adapter(args, state):
    ns, meta, _, _, layer = state
    saved = training.load(args.run_dir)
    training.verify_training_snapshot(args.run_dir, saved)
    if (saved['model'] != meta['model'] or saved['model_config'] != meta['model_config'] or saved['layer'] != layer
            or saved['source'] != legacy_adapter.source_identity(ns, meta, [layer])
            or saved['source_system'] != source_eval.SOURCE_SYSTEM
            or saved['config'] != {k: meta['config'][k] for k in saved['config']}):
        raise ValueError('adapter source/model/layer/prompt differs from the current run')
    if saved.get('layer_plan') != plan_identity(getattr(args, '_layer_plan', None)):
        raise ValueError('source adapter and deployment layer plan differ')
    return saved


def controls(args, state, adapter, pilot):
    if args.controls_run is None: raise ValueError('evaluation requires the matching five-arm E29 --controls-run')
    _, meta, _, prepared, layer = state
    filename = 'check.json' if pilot else 'tune.json'
    path = args.controls_run / filename
    old = qa.read(path); identity = old['identity']
    questions = prepared['splits']['tune'][:3] if pilot else prepared['splits']['tune']
    if (old.get('completed') is not True or old['conditions'] != list(source_eval.CONDITIONS)
            or old['model'] != meta['model'] or identity['source'] != adapter['source'] or identity['layer'] != layer
            or identity['source_system'] != adapter['source_system'] or identity['base_system'] != adapter['config']['qa_system']
            or identity['user_template'] != adapter['config']['qa_user'] or identity['batch_size'] != 1
            or identity['max_new_tokens'] != adapter['config']['max_new_tokens']
            or [r['pair_id'] for r in old['evaluations']] != [q['pair_id'] for q in questions]):
        raise ValueError('the five control arms do not match the trained source-interchange deployment')
    qa.verify_question_rows([{**row, 'idx': i} for i, row in enumerate(old['evaluations'])], questions)
    if adapter.get('pilot') and not pilot:
        raise ValueError('a three-pair training pilot cannot enter the full tuning comparison')
    return old, path, questions


def generate(args, state, adapter, pilot, bundle=None):
    _, meta, _, prepared, layer = state
    old, control_path, questions = controls(args, state, adapter, pilot)
    filename = 'check.json' if pilot else 'tune.json'
    path = args.run_dir / filename
    method = 'trained_source_swap_' + adapter['initialization'] + '_init'
    implementation = training.source_implementation_identity(); implementation['entry_script_sha256'] = qa.digest(__file__)
    identity = {'protocol': training.PROTOCOL, 'model': meta['model'], 'source': adapter['source'],
                'adapter_sha256': qa.digest(args.run_dir / 'source_adapter.pt'),
                'control_generation_sha256': qa.digest(control_path), 'condition': method,
                'layer': layer, 'source_system': adapter['source_system'], 'config': adapter['config'],
                'batch_size': 1, 'alpha': 1.0, 'same_prefix_tokens': True, 'gold_used_at_inference': False,
                'question_ids': [q['pair_id'] for q in questions], 'implementation': implementation,
                'pilot': pilot, 'split': 'tune', 'exploratory_previously_exposed_tuning_set': True}
    if path.exists():
        stored = qa.read(path)
        if stored.get('identity') != identity: raise ValueError('generation request changed; use a new run')
        if stored.get('completed'): return stored
        raise ValueError('partial generated artifact exists; preserve it and use a new run')
    if bundle is None: bundle = qa.open_bundle(meta, prepared)
    qa.verify_bundle(bundle, meta, prepared)
    versions = [(p, p._version, p.requires_grad) for p in bundle.model.parameters()]
    result = copy.deepcopy(old)
    result.update(protocol=training.PROTOCOL, identity=identity, control_identity=old['identity'],
                  control_generation_source=str(control_path),
                  protocol_fingerprint=qa.fingerprint(identity), bank_protocol_fingerprint=qa.fingerprint(identity),
                  training_sha256=identity['adapter_sha256'], pilot=bool(pilot or adapter.get('pilot')),
                  intervention_position='same-prefix source coordinate at prompt-end and every later answer predictor; see per-row trace',
                  wiring_pilot=pilot, completed=False,
                  E={'method': 'supervised source interchange', 'initialization': adapter['initialization'],
                     'objective': adapter['objective'], 'layer': layer, 'alpha': 1.0}, C={'enabled': False})
    result['conditions'].append(method)
    result['condition_specs'].append({'name': method, 'kind': 'E', 'uses_reference_answer': False, 'eligible_for_selection': False})
    code_root = Path(__file__).resolve().parents[1]
    for name, expected in implementation['files'].items():
        target = args.run_dir / 'generation_code_snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
        if qa.digest(code_root / name) != expected: raise RuntimeError('generation code changed before execution')
        shutil.copyfile(code_root / name, target)
    shutil.copyfile(__file__, args.run_dir / 'generation_code_snapshot/scripts/e30_qa_source_interchange_train.py')
    with fp32_output_projection(bundle):
        for index, row in enumerate(result['evaluations']):
            row['idx'] = index
            base, source = source_eval.prompt_tokens(bundle, adapter['config'], row['question'])
            if row['prompt_token_ids'] != {'base': base, 'source': source}:
                raise ValueError('current encoded prompts differ from the five-arm controls')
            generated = greedy_source_generate(bundle, base, source, mode='source_interchange', layer=layer,
                                                direction=adapter['W'], max_new_tokens=adapter['config']['max_new_tokens'])
            row['outputs'][method] = generated['output']; row['finished'][method] = generated['finished']
            row['rule_abstain'][method] = bool(rule_abstains(generated['output']))
            row['generation_token_ids'][method] = generated['token_ids']; row['trace'][method] = generated['trace']
            if pilot:
                row['cache_reference_checks'][method] = source_eval.cache_reference_check(
                    bundle, base, source, layer, adapter['W'], generated['token_ids'])
                row['trained_same_prompt_identity_checks'] = source_eval.same_prompt_identity_check(
                    bundle, base, layer, adapter['W'], generated['token_ids'])
            print('SUPERVISED_SOURCE_SWAP', meta['model'], row['pair_id'], repr(generated['output']), flush=True)
            qa.save(path, result)
    if any(p._version != version or p.requires_grad != flag for p, version, flag in versions):
        raise RuntimeError('source-interchange evaluation changed base-model parameters')
    now = training.source_implementation_identity(); now['entry_script_sha256'] = qa.digest(__file__)
    if now != implementation:
        raise RuntimeError('generation implementation changed during execution')
    result.update(completed=True, base_parameters_unchanged=True)
    qa.save(path, result)
    return result


def freeze_control_grades(args, state, adapter, stem):
    old, raw_path, _ = controls(args, state, adapter, stem == 'check')
    scored_path = args.controls_run / (stem + '_scored_items.json')
    scored = qa.read(scored_path)
    cfg = legacy_adapter.validate_judge_experiment(args.judge_exp)
    if (scored.get('completed') is not True or scored.get('grading_completed') is not True
            or scored.get('grading_source_generation_sha256') != qa.digest(raw_path)
            or scored.get('grading_judge_config_sha256') != qa.fingerprint(cfg)
            or scored.get('conditions') != old['conditions']):
        raise ValueError('complete five-arm control grades with the same judge settings are required')
    qa.verify_question_rows(scored['evaluations'], old['evaluations'])
    generated = qa.read(args.run_dir / (stem + '.json'))
    qa.verify_question_rows(generated['evaluations'], old['evaluations'])
    if (generated.get('protocol_fingerprint') != qa.fingerprint(generated['identity'])
            or generated.get('bank_protocol_fingerprint') != generated['protocol_fingerprint']):
        raise ValueError('trained generation identity fingerprint differs')
    for original, current in zip(old['evaluations'], generated['evaluations']):
        if any(current['outputs'][c] != original['outputs'][c] or current['finished'][c] != original['finished'][c]
               for c in old['conditions']):
            raise ValueError('a copied control answer/completion flag changed in the six-arm comparison')
    for raw_row, row in zip(old['evaluations'], scored['evaluations']):
        if row['outputs'] != raw_row['outputs'] or row['finished'] != raw_row['finished']:
            raise ValueError('control grades do not belong to the original control answers')
        if any(row['judge'][c].get('label') not in ('correct', 'wrong', 'abstain')
               or row['judge'][c].get('rubric_id') != qa.QA_GRADER_VERSION for c in old['conditions']):
            raise ValueError('control grades contain unresolved or foreign-rubric entries')
    from core.qa_control_cache import seed_control_cache
    seed = seed_control_cache(source_dir=args.controls_judge_cache, run_dir=args.run_dir, judge_config=cfg,
                              scored_rows=scored['evaluations'], conditions=old['conditions'],
                              scored_sha256=qa.digest(scored_path), stem=stem)
    for name, value in seed.items(): setattr(args, name, value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('train', 'check', 'generate-tune', 'judge-check', 'judge-tune'))
    p.add_argument('--source-run', type=Path, required=True); p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--controls-run', type=Path)
    p.add_argument('--layer', type=int, help='only a layer declared in --layer-plan; no test-based selection')
    p.add_argument('--layer-plan', type=Path)
    p.add_argument('--controls-judge-cache', type=Path,
                   help='judge phases copy this cache into a run-local cache and require every reused control grade to agree')
    p.add_argument('--pilot', type=int); p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=.001); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--init', choices=('das', 'random'), default='das')
    p.add_argument('--judge-exp', default='E4_qa')
    args = p.parse_args(); state = source_state(args)
    if args.phase == 'train': train(args, state); return
    adapter = verified_adapter(args, state)
    if args.phase.startswith('judge-'):
        _, meta, _, _, _ = state
        args.model = meta['model']; legacy_adapter.validate_judge_experiment(args.judge_exp)
        stem = args.phase.split('-', 1)[1]; raw = qa.read(args.run_dir / (stem + '.json'))
        if raw['identity']['adapter_sha256'] != qa.digest(args.run_dir / 'source_adapter.pt'):
            raise ValueError('generated adapter differs from the saved checkpoint')
        freeze_control_grades(args, state, adapter, stem)
        qa.judge(args, {**meta, 'protocol_fingerprint': raw['bank_protocol_fingerprint']}, stem + '.json', stem + '_scored')
    else:
        generate(args, state, adapter, args.phase == 'check')


if __name__ == '__main__':
    main()
