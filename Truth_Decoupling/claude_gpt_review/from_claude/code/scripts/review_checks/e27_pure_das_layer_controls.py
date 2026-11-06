#!/usr/bin/env python3
"""Frozen original-DAS layer/timing controls, tuning split only.

Default Llama: baseline + layers 10/29 x prefill/continuous, alpha fixed at 3.
No optimizer, extra content training, EOS objective, or adapted E27 vector is
loaded. Requested pilot questions are generated in their original complete
source batches, then only requested questions enter the separate judge phase.
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
from core.config import load_experiment

PROTOCOL = 'pure_das_layer_timing_controls_v1'
ALPHA = 3.0
SPLIT = 'tune'
CODE_FILES = ('core/intervene.py', 'core/model_io.py', 'core/qa_precision.py',
    'core/qa_context.py', 'core/qa_das.py', 'core/judge.py', 'core/qa_scoring.py',
    'scripts/e24_real_qa.py', 'scripts/e25_qa_das.py', 'scripts/e25_qa_context.py',
    'scripts/review_checks/e27_pure_das_layer_controls.py')


def tensor_hash(value):
    x = torch.as_tensor(value).detach().float().cpu().contiguous()
    return hashlib.sha256(json.dumps({'shape': list(x.shape), 'dtype': 'float32'}, sort_keys=True).encode()
                          + x.numpy().tobytes()).hexdigest()


def selected_context(prepared, limit, pair_ids, batch_size):
    rows = prepared['splits'][SPLIT]
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(rows)):
        raise ValueError('limit must fit the tuning split')
    wanted = list(pair_ids) if pair_ids else [r['pair_id'] for r in rows[:limit]]
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError('need unique requested tuning IDs')
    ids = [r['pair_id'] for r in rows]
    if not set(wanted) <= set(ids):
        raise ValueError('requested question is not in the tuning split')
    if pair_ids and limit is not None and len(wanted) != limit:
        raise ValueError('explicit IDs must match the declared pilot limit')
    positions = [i for i, r in enumerate(rows) if r['pair_id'] in set(wanted)]
    starts = sorted({i // batch_size * batch_size for i in positions})
    context = [r for start in starts for r in rows[start:start + batch_size]]
    selected = [r for r in rows if r['pair_id'] in set(wanted)]
    return selected, context, [{'source_batch_start': start,
        'source_batch_ids': [r['pair_id'] for r in rows[start:start + batch_size]]} for start in starts]


def inputs(args):
    ns = SimpleNamespace(run_dir=Path(args.source_run), model=args.model)
    meta, bank, prepared = qa.load_bank(ns)
    meta, bank = qa.runtime_bank(ns, meta, bank)
    layers = list(args.layers)
    if len(layers) != 2 or len(set(layers)) != 2 or not set(layers) <= set(meta['das_candidate_layers']):
        raise ValueError('exactly two distinct verified original-DAS layers required')
    if args.all_questions and args.pair_ids:
        raise ValueError('all-questions cannot be combined with selected pilot IDs')
    limit = None if args.all_questions else args.limit
    selected, context, batches = selected_context(prepared, limit, args.pair_ids, meta['config']['batch_size'])
    choices = [dict(qa.BASELINE)] + [qa.condition('qa_das', mode, layer, ALPHA)
        for layer in layers for mode in ('prefill', 'continuous')]
    for choice in choices:
        choice['eligible_for_selection'] = False
    reference_name = qa.condition('qa_das', 'continuous', layers[1], ALPHA)['name']
    reference_path = ns.run_dir / 'tune.json'
    reference = qa.read(reference_path)
    if (not reference.get('completed') or reference.get('model') != meta['model']
            or reference.get('split') != SPLIT or reference.get('training_sha256') != meta['training_sha256']
            or reference.get('bank_protocol_fingerprint') != meta['protocol_fingerprint']):
        raise ValueError('the source tuning generation does not match the verified bank/training')
    qa.verify_question_rows(reference['evaluations'], prepared['splits'][SPLIT])
    if any(name not in reference['conditions'] for name in ('baseline', reference_name)):
        raise ValueError('source tune.json must contain baseline and the second-layer continuous control')
    source_files = [ns.run_dir / name for name in ('bank.json','bank.npz','questions.json','training.json','tune.json')]
    source_files += [qa.checkpoint_path(ns.run_dir, l, meta['config']['seed']) for l in layers]
    identity = {'source_run': str(ns.run_dir.resolve()), 'source_bank_fingerprint': meta['fingerprint'],
        'source_runtime_fingerprint': meta['protocol_fingerprint'], 'model': meta['model'],
        'source_sha256': {str(p.resolve()): qa.digest(p) for p in source_files},
        'checkpoint_sha256': {str(l): qa.digest(qa.checkpoint_path(ns.run_dir,l,meta['config']['seed'])) for l in layers},
        'direction_sha256': {str(l): tensor_hash(bank['E_qa_das_L%d' % l]) for l in layers},
        'centres_sha256': {str(l): {kind: tensor_hash(bank[kind+'_center_L%d' % l]) for kind in ('prompt','decode')} for l in layers},
        'code_sha256': {name: qa.digest(CODE / name) for name in CODE_FILES},
        'layers': layers, 'alpha': ALPHA, 'modes': ['prefill','continuous'],
        'split': SPLIT, 'selected_ids': [r['pair_id'] for r in selected], 'batch_context': batches,
        'source_reference_condition': reference_name, 'reference_sha256': qa.digest(reference_path),
        'training_called_by_this_script': False, 'extra_content_adaptation': False,
        'uses_test_for_selection': False, 'pilot': bool(limit is not None or meta['pilot'])}
    return ns, meta, bank, prepared, choices, selected, context, reference, identity


def generation_meta(meta, identity):
    return {**meta, 'protocol_fingerprint': qa.fingerprint(identity),
            'pilot': identity['pilot']}


def prepare_output(args, ns, identity):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    original = ns.run_dir / 'questions.json'; target = args.run_dir / 'questions.json'
    if target.exists() and qa.digest(target) != qa.digest(original):
        raise ValueError('output questions.json differs from the original source bytes')
    if not target.exists():
        shutil.copyfile(original, target)
    for relative, digest in identity['code_sha256'].items():
        target = args.run_dir / 'code_snapshot' / relative
        if target.exists() and qa.digest(target) != digest:
            raise ValueError('a frozen execution-code snapshot differs; choose a new run')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(CODE / relative, target)
    target=args.run_dir/'identity.json'
    if target.exists():
        if qa.read(target)!=identity:
            raise ValueError('frozen control identity changed; use a new run')
    else:
        qa.save(target,identity,new=True)


def generate(args, state, bundle=None):
    ns, meta, bank, prepared, choices, selected, context, reference, identity = state
    prepare_output(args, ns, identity)
    emeta = generation_meta(meta, identity)
    if bundle is None:
        bundle = qa.open_bundle(meta, prepared)
    versions = [(p, p._version, p.requires_grad) for p in bundle.model.parameters()]
    before_vectors = {str(l): tensor_hash(bank['E_qa_das_L%d' % l]) for l in identity['layers']}
    # Complete source batches remain contiguous after concatenation. The final
    # source partial batch, if any, remains last and retains its original width.
    context_file='check_generations.json' if identity['pilot'] else 'tune_context.json'
    raw = qa.generate_questions(args, emeta, bank, context, choices,
        context_file, SPLIT, bundle=bundle)
    if any(p._version != v or p.requires_grad != flag for p,v,flag in versions):
        raise RuntimeError('base parameter values/version flags changed in an inference-only control')
    if before_vectors != {str(l): tensor_hash(bank['E_qa_das_L%d' % l]) for l in identity['layers']}:
        raise RuntimeError('a frozen source direction changed')
    source_by_id = {r['pair_id']:r for r in reference['evaluations']}
    wanted = set(identity['selected_ids'])
    output = copy.deepcopy(raw)
    output['evaluations'] = [r for r in output['evaluations'] if r['pair_id'] in wanted]
    for idx,row in enumerate(output['evaluations']): row['idx']=idx
    checks = []
    for row in output['evaluations']:
        for name in ('baseline', identity['source_reference_condition']):
            old=source_by_id[row['pair_id']]
            checks.append({'pair_id':row['pair_id'],'condition':name,
                'exact_text_equal':row['outputs'][name]==old['outputs'][name],
                'finished_equal':row['finished'][name]==old['finished'][name]})
    matches=all(c['exact_text_equal'] and c['finished_equal'] for c in checks)
    output.update(protocol=PROTOCOL, identity=identity, pilot=identity['pilot'], wiring_pilot=identity['pilot'],
        diagnostic_only=False, generation_context_file=context_file,
        generation_context_sha256=qa.digest(args.run_dir/context_file),
        generation_context_questions=len(context), selected_questions=len(selected),
        batch_context=identity['batch_context'], source_reproduction=checks,
        base_parameters_and_source_directions_unchanged=True, completed=matches,
        E={'method':'original frozen QA-DAS; no extra content adaptation', 'layers':identity['layers'],
           'alpha':ALPHA,'modes':identity['modes']}, C={'enabled':False})
    qa.verify_question_rows(output['evaluations'], selected)
    path=args.run_dir/'tune.json'
    if path.exists() and qa.read(path)!=output:
        raise ValueError('existing selected-output artifact differs; preserve it and use a new run')
    qa.save(path,output)
    if not matches:
        raise RuntimeError('baseline/second-layer continuous did not reproduce source text; do not interpret layer control')
    for row in output['evaluations']:
        print('SELECTED_PILOT' if identity['pilot'] else 'SELECTED_TUNE', row['pair_id'],
              json.dumps(row['outputs'],ensure_ascii=False), flush=True)
    print('PURE_DAS_LAYER_CONTROL_DONE',len(selected),'scored questions;',len(context),'source-context questions',flush=True)
    return output


def judge(args, state):
    ns,meta,bank,prepared,choices,selected,context,reference,identity=state
    raw=qa.read(args.run_dir/'tune.json')
    if raw.get('identity')!=identity or not raw.get('completed'):
        raise ValueError('finish the matching source-only generation before judging')
    if raw['generation_context_sha256']!=qa.digest(args.run_dir/raw['generation_context_file']):
        raise ValueError('generation context changed')
    qa.verify_question_rows(raw['evaluations'],selected)
    cfg=load_experiment(args.judge_exp)
    if not isinstance(cfg.get('judge'),dict) or not cfg['judge'].get('model'):
        raise ValueError('judge experiment requires the QA judge configuration, e.g. E4_qa')
    return qa.judge(args,generation_meta(meta,identity),'tune.json','tune_scored')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['generate','judge'])
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--model',default='llama3.1-8b')
    p.add_argument('--layers',type=int,nargs=2,default=[10,29])
    p.add_argument('--limit',type=int,default=3)
    p.add_argument('--pair-ids',nargs='+')
    p.add_argument('--all-questions',action='store_true')
    p.add_argument('--judge-exp',default='E4_qa')
    args=p.parse_args();state=inputs(args)
    if args.phase=='generate':generate(args,state)
    else:judge(args,state)


if __name__=='__main__':main()
