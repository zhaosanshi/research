#!/usr/bin/env python3
"""Uniform seven-arm 256-token completion diagnostic; immutable 48-token parent."""
import argparse
from collections import Counter
import copy
from pathlib import Path
import sys

import torch

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE / 'scripts'), str(CODE)]
import e25_qa_das as qa
import e27_qa_generation as ga
import e29_qa_source_interchange as source_eval
from core import qa_source_interchange_training as ce_training
from core import qa_source_pair_preference as preference
from core.qa_precision import fp32_output_projection
from core.qa_source_interchange import greedy_source_generate
from core.readout import rule_abstains
from core.qa_control_cache import seed_control_cache

PROTOCOL = 'source_natural_completion_diagnostic_v1'
BUDGET = 256
CONDITIONS = [*source_eval.CONDITIONS, 'trained_source_swap_das_init', preference.METHOD]
CRITICAL_CODE = ('core/qa_source_interchange.py', 'core/model_io.py', 'core/qa_precision.py',
                 'scripts/e29_qa_source_interchange.py')
PILOT_IDS = {'llama3.1-8b': ['tqa:479', 'tqa:476', 'tqa:563'],
             'qwen3-8b': ['tqa:123', 'tqa:457', 'tqa:374']}


def load_state(args):
    ns, meta, bank, prepared = ga.source_run(args.source_run)
    ce, pref = ce_training.load(args.ce_run), preference.load(args.parent_run)
    parent = qa.read(args.parent_run / 'tune.json')
    grade = qa.read(args.parent_run / 'tune_scored_items.json')
    ce_raw = qa.read(args.ce_run / 'tune.json')
    source_identity = ga.source_identity(ns, meta, [pref['layer']])
    if (parent.get('completed') is not True or grade.get('completed') is not True or not grade.get('grading_completed')
            or parent['protocol'] != preference.PROTOCOL or parent['conditions'] != CONDITIONS
            or parent['model'] != meta['model'] or ce['model'] != meta['model'] or pref['model'] != meta['model']
            or ce['layer'] != pref['layer'] or ce['source'] != pref['source'] or pref['source'] != source_identity
            or ce['config'] != pref['config'] or pref['config']['max_new_tokens'] != 48
            or pref['source_system'] != source_eval.SOURCE_SYSTEM
            or parent['identity']['adapter_sha256'] != qa.digest(args.parent_run / 'preference_adapter.pt')
            or parent['identity']['control_generation_sha256'] != qa.digest(args.ce_run / 'tune.json')
            or pref['ce_parent']['checkpoint_sha256'] != qa.digest(args.ce_run / 'source_adapter.pt')
            or grade['grading_source_generation_sha256'] != qa.digest(args.parent_run / 'tune.json')):
        raise ValueError('natural completion needs the unchanged complete E34 parent, CE checkpoint and source')
    qa.verify_question_rows(parent['evaluations'], prepared['splits']['tune'])
    qa.verify_question_rows(grade['evaluations'], parent['evaluations'])
    for a, b in zip(parent['evaluations'], grade['evaluations']):
        if a['outputs'] != b['outputs'] or a['finished'] != b['finished']:
            raise ValueError('parent grades changed the original responses')
    for name in CRITICAL_CODE:
        if qa.digest(CODE / name) != parent['identity']['implementation']['files'][name]:
            raise ValueError('natural completion decoder or prompt code changed: ' + name)
    layer = pref['layer']
    original = torch.as_tensor(bank['E_qa_das_L%d' % layer]).float().clone()
    generator = torch.Generator(device='cpu').manual_seed(42 + layer)
    random = torch.nn.functional.normalize(torch.randn(meta['hidden_size'], generator=generator), dim=0)
    ancestor = ce_raw['control_identity']
    if (qa.fingerprint(original.tolist()) != ancestor['direction_sha256']
            or qa.fingerprint(random.tolist()) != ancestor['random_direction_sha256']):
        raise ValueError('frozen original-DAS/random direction does not reproduce its original bytes')
    weights = {'source_das_swap': original, 'source_random_swap': random,
               'trained_source_swap_das_init': ce['W'], preference.METHOD: pref['W']}
    ended_controls = {name: [r['pair_id'] for r in parent['evaluations'] if r['finished'][name]][:3] for name in CONDITIONS}
    if any(len(ids) != 3 for ids in ended_controls.values()): raise ValueError('need three finished controls per arm')
    pair_probes = [r['pair_id'] for r in parent['evaluations'] if not r['finished'][preference.METHOD]][:3]
    if pair_probes != PILOT_IDS[meta['model']]: raise ValueError('the declared three unfinished pilot questions changed')
    plan = {'protocol': PROTOCOL, 'model': meta['model'], 'layer': layer, 'source': source_identity,
            'config': pref['config'], 'source_system': pref['source_system'], 'question_ids': [r['pair_id'] for r in parent['evaluations']],
            'condition': preference.METHOD, 'same_prefix_tokens': True, 'batch_size': 1, 'alpha': 1.0, 'split': 'tune',
            'parent_generation_sha256': qa.digest(args.parent_run / 'tune.json'),
            'parent_scored_sha256': qa.digest(args.parent_run / 'tune_scored_items.json'),
            'adapter_sha256': qa.digest(args.parent_run / 'preference_adapter.pt'),
            'control_generation_sha256': qa.digest(args.parent_run / 'tune.json'),
            'ce_checkpoint_sha256': qa.digest(args.ce_run / 'source_adapter.pt'),
            'weight_fingerprints': {name: qa.fingerprint(w.tolist()) for name, w in weights.items()},
            'conditions': CONDITIONS, 'ended_control_ids': ended_controls, 'unfinished_pair_pilot_ids': pair_probes,
            'budget_override': {'parent_max_new_tokens': 48, 'max_new_tokens': BUDGET,
                                'only_python_generation_loop_limit_changes': True, 'training_configuration_unchanged': True},
            'decoder_code': {name: qa.digest(CODE / name) for name in CRITICAL_CODE},
            'entry_code_sha256': qa.digest(__file__), 'gold_used_at_inference': False,
            'no_forced_eos': True, 'no_text_trimming': True, 'all_questions_retained': True}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / 'completion_plan.json'
    if path.exists() and qa.read(path) != plan: raise ValueError('completion plan changed; use a new run directory')
    if not path.exists(): qa.save(path, plan)
    return {'ns': ns, 'meta': meta, 'prepared': prepared, 'parent': parent, 'grade': grade,
            'ce': ce, 'pref': pref, 'weights': weights, 'plan': plan}


def repeat_stat(tokens):
    count = Counter(tuple(tokens[i:i + 8]) for i in range(max(0, len(tokens) - 7)))
    return max(count.values(), default=0)


def one(args, state, bundle, row, name):
    base, source = source_eval.prompt_tokens(bundle, state['pref']['config'], row['question'])
    if row['prompt_token_ids'] != {'base': base, 'source': source}:
        raise ValueError('original prompt token IDs changed')
    mode = 'source_interchange' if name in state['weights'] else name
    result = greedy_source_generate(bundle, base, source, mode=mode, layer=state['pref']['layer'],
                                    direction=state['weights'].get(name), max_new_tokens=BUDGET)
    old_ids = row['generation_token_ids'][name]
    if row['finished'][name]:
        if (result['token_ids'] != old_ids or not result['finished'] or result['output'] != row['outputs'][name]):
            raise RuntimeError('already-finished response changed with an unused budget extension: ' + row['pair_id'] + '/' + name)
    elif len(old_ids) != 48 or result['token_ids'][:48] != old_ids:
        raise RuntimeError('original 48-token path changed: ' + row['pair_id'] + '/' + name)
    return result


def generate(args, state, pilot, bundle=None):
    plan, parent = state['plan'], state['parent']
    path = args.run_dir / ('pilot.json' if pilot else 'tune.json')
    if path.exists():
        stored = qa.read(path)
        if stored.get('identity') != plan: raise ValueError('existing completion identity differs')
        if stored.get('completed'): return stored
        raise ValueError('partial completion diagnostic exists; preserve it and use a new run')
    if not pilot:
        proof = qa.read(args.run_dir / 'pilot.json')
        if not proof.get('completed') or proof.get('identity') != plan or len(proof['finished_control_checks']) != 21:
            raise ValueError('finish the 21 exact finished controls and three extension probes first')
        expected={(name,pid) for name,ids in plan['ended_control_ids'].items() for pid in ids}
        observed={(r['condition'],r['pair_id']) for r in proof['finished_control_checks'] if r.get('tokens_and_text_exact') and r.get('finished')}
        if observed != expected or [r['pair_id'] for r in proof['extension_probes']] != plan['unfinished_pair_pilot_ids']:
            raise ValueError('pilot proof does not cover the declared finished and unfinished controls')
        if not all(r.get('first_48_token_ids_exact') and r['new']['finished'] for r in proof['extension_probes']):
            raise ValueError('pilot still contains unfinished/failed continuations; no automatic full release')
        if not getattr(args,'pilot_reviewed',False):
            raise ValueError('read the pilot complete texts for loops first, then pass --pilot-reviewed')
    if bundle is None: bundle = qa.open_bundle(state['meta'], state['prepared'])
    qa.verify_bundle(bundle, state['meta'], state['prepared'])
    versions = [(p, p._version, p.requires_grad) for p in bundle.model.parameters()]
    by_id = {r['pair_id']: r for r in parent['evaluations']}
    result = {'protocol': PROTOCOL, 'model': plan['model'], 'identity': plan, 'completed': False}
    with fp32_output_projection(bundle):
        if pilot:
            result.update(finished_control_checks=[], extension_probes=[])
            for name, ids in plan['ended_control_ids'].items():
                for pid in ids:
                    new = one(args, state, bundle, by_id[pid], name)
                    result['finished_control_checks'].append({'pair_id': pid, 'condition': name,
                        'tokens_and_text_exact': True, 'finished': True, 'tokens': len(new['token_ids'])})
                    qa.save(path, result)
            for pid in plan['unfinished_pair_pilot_ids']:
                row = by_id[pid]; new = one(args, state, bundle, row, preference.METHOD)
                result['extension_probes'].append({'pair_id': pid, 'question': row['question'], 'gold': row['gold'],
                    'incorrect': row.get('incorrect', []), 'condition': preference.METHOD,
                    'old_output': row['outputs'][preference.METHOD], 'old_token_ids': row['generation_token_ids'][preference.METHOD],
                    'new': new, 'first_48_token_ids_exact': True, 'maximum_repeated_8_token_ngram': repeat_stat(new['token_ids'])})
                print('COMPLETION_PILOT', pid, new['finished'], len(new['token_ids']), repr(new['output']), flush=True)
                qa.save(path, result)
        else:
            rows = []
            result.update(protocol_fingerprint=qa.fingerprint(plan), bank_protocol_fingerprint=qa.fingerprint(plan),
                control_identity=parent['identity'], parent_protocol=parent['protocol'], split='tune', pilot=False,
                wiring_pilot=False, diagnostic_only=False, diagnostic_budget_only=True,
                conditions=copy.deepcopy(parent['conditions']), condition_specs=copy.deepcopy(parent['condition_specs']),
                E=copy.deepcopy(parent['E']), C=copy.deepcopy(parent['C']),
                intervention_position='Original source-interchange coordinates and weights; only the external loop upper bound is 256.',
                evaluations=rows, completion_audit=[])
            for original in parent['evaluations']:
                row = {k: copy.deepcopy(original[k]) for k in ('pair_id', 'question', 'gold', 'incorrect', 'best_answer', 'idx',
                       'prompt_token_ids', 'outputs', 'finished', 'rule_abstain', 'generation_token_ids', 'trace') if k in original}
                for name in CONDITIONS:
                    if original['finished'][name]:
                        result['completion_audit'].append({'pair_id': row['pair_id'], 'condition': name,
                            'action': 'reused_original_naturally_finished', 'witness_control_ids': plan['ended_control_ids'][name]})
                        continue
                    new = one(args, state, bundle, original, name)
                    row['outputs'][name] = new['output']; row['finished'][name] = new['finished']
                    row['rule_abstain'][name] = bool(rule_abstains(new['output']))
                    row['generation_token_ids'][name] = new['token_ids']; row['trace'][name] = new['trace']
                    result['completion_audit'].append({'pair_id': row['pair_id'], 'condition': name,
                        'action': 'rerun_original_unfinished_at_uniform_256_budget', 'first_48_token_ids_exact': True,
                        'tokens': len(new['token_ids']), 'finished': new['finished'],
                        'maximum_repeated_8_token_ngram': repeat_stat(new['token_ids'])})
                    print('NATURAL_COMPLETION', row['pair_id'], name, new['finished'], len(new['token_ids']), repr(new['output']), flush=True)
                rows.append(row); qa.save(path, result)
    if any(p._version != v or p.requires_grad != flag for p, v, flag in versions):
        raise RuntimeError('completion diagnostic modified base-model parameters')
    if any(qa.digest(CODE/name)!=sha for name,sha in plan['decoder_code'].items()) or qa.digest(__file__)!=plan['entry_code_sha256']:
        raise RuntimeError('completion implementation changed during execution')
    if pilot:
        result['all_extension_probes_naturally_finished']=all(r['new']['finished'] for r in result['extension_probes'])
        result['full_not_automatically_released']=True
    else:
        result['pilot_review_acknowledged']=True
        result['unfinished_after_budget']={name:sum(not r['finished'][name] for r in result['evaluations']) for name in CONDITIONS}
    result.update(completed=True, base_parameters_unchanged=True)
    qa.save(path, result); return result



def judge(args,state):
    """Same whole text keeps its old semantic grade; changed tails use same legacy judge."""
    raw_path=args.run_dir/'tune.json';raw=qa.read(raw_path);parent=state['parent'];graded=state['grade'];plan=state['plan']
    if (not raw.get('completed') or raw.get('identity')!=plan or raw.get('conditions')!=CONDITIONS
            or raw.get('protocol_fingerprint')!=qa.fingerprint(plan)
            or raw.get('bank_protocol_fingerprint')!=qa.fingerprint(plan)):
        raise ValueError('complete matching seven-arm 256 generation before grading')
    qa.verify_question_rows(raw['evaluations'],parent['evaluations'])
    qa.verify_question_rows(graded['evaluations'],parent['evaluations'])
    cfg=ga.validate_judge_experiment('E4_qa_legacy_r20')
    if (cfg.get('full_personal_refusal_guard',False)
            or graded.get('grading_judge_config_sha256')!=qa.fingerprint(cfg)
            or graded.get('grading_source_generation_sha256')!=plan['parent_generation_sha256']
            or qa.digest(args.parent_run/'tune_scored_items.json')!=plan['parent_scored_sha256']):
        raise ValueError('all seven arms need the exact frozen legacy grader; guard is a separate all-arm table')
    frozen={};unchanged=changed=0
    for old,new,scored in zip(parent['evaluations'],raw['evaluations'],graded['evaluations']):
        if old['prompt_token_ids']!=new['prompt_token_ids'] or old['outputs']!=scored['outputs'] or old['finished']!=scored['finished']:
            raise ValueError('completion prompts or parent grade text changed')
        frozen[new['pair_id']]={}
        for name in CONDITIONS:
            ids=new['generation_token_ids'][name];old_ids=old['generation_token_ids'][name]
            if old['finished'][name]:
                if ids!=old_ids or not new['finished'][name] or new['outputs'][name]!=old['outputs'][name] or new['trace'][name]!=old['trace'][name]:
                    raise ValueError('finished control was changed rather than faithfully reused')
            elif len(old_ids)!=48 or ids[:48]!=old_ids or len(ids)>BUDGET:
                raise ValueError('completion does not retain the old 48-token prefix')
            if new['outputs'][name]==old['outputs'][name]:
                frozen[new['pair_id']][name]=scored['judge'][name];unchanged+=1
            else:changed+=1
    seeded=seed_control_cache(source_dir=args.parent_run/'judge_cache',run_dir=args.run_dir,judge_config=cfg,
        scored_rows=graded['evaluations'],conditions=CONDITIONS,scored_sha256=plan['parent_scored_sha256'],stem='tune')
    for key,value in seeded.items():setattr(args,key,value)
    args.frozen_control_grades=frozen;args.model=state['meta']['model'];args.judge_exp='E4_qa_legacy_r20'
    qa.save(args.run_dir/'grade_inheritance.json',{'parent_scored_sha256':plan['parent_scored_sha256'],
        'unchanged_whole_text_grades_frozen':unchanged,'changed_whole_texts_to_evaluate':changed,
        'same_legacy_policy_for_all_seven_arms':True,'guard_not_only_applied_to_new_method':True,
        'frozen_conditions_by_pair':{pid:list(g) for pid,g in frozen.items()}})
    qa.judge(args,{**state['meta'],'protocol_fingerprint':raw['bank_protocol_fingerprint']},'tune.json','tune_scored')
    return {'frozen':unchanged,'changed':changed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('pilot', 'generate', 'judge'))
    p.add_argument('--source-run', type=Path, required=True); p.add_argument('--ce-run', type=Path, required=True)
    p.add_argument('--parent-run', type=Path, required=True); p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--pilot-reviewed',action='store_true',help='acknowledge manual full-text/no-loop review after successful pilot; never changes outputs')
    args = p.parse_args(); state = load_state(args)
    if args.phase=='judge':judge(args,state)
    else:generate(args, state, args.phase == 'pilot')


if __name__ == '__main__': main()
