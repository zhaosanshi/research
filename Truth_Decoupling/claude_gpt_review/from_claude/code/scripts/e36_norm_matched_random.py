#!/usr/bin/env python3
"""Three untrained random orientations matched to the learned source write size.

Tuning only. Matching is to the hypothetical E34 write at the SAME state and
source, after BF16 storage. A scalar learned dose is borrowed explicitly; this
is an orientation null, not a source-information null or an untrained method.
"""
import argparse,copy
from pathlib import Path
import torch
import common
import e25_qa_das as qa
import e27_qa_generation as legacy
import e29_qa_source_interchange as source_eval
import e34_qa_source_pair_preference as pair_entry
from core import qa_source_pair_preference as preference
from core import qa_norm_matched_random as control
from core.qa_generation_adapter import implementation_identity
from core.qa_control_cache import seed_control_cache
from core.qa_precision import fp32_output_projection
from core.readout import rule_abstains

PROTOCOL='source_actual_norm_random_orientation_control_v1'
SEEDS=(42,43,44)
CRITICAL_CODE=('core/qa_source_interchange.py','core/model_io.py','core/qa_precision.py',
               'scripts/e29_qa_source_interchange.py')


def validate_parent_grades(raw,graded,judge_cfg,raw_sha256):
    if (not graded.get('completed') or not graded.get('grading_completed')
            or graded.get('model')!=raw['model'] or graded.get('split')!='tune'
            or graded.get('grading_source_generation_sha256')!=raw_sha256
            or graded.get('grading_judge_config_sha256')!=qa.fingerprint(judge_cfg)
            or graded.get('conditions')!=raw['conditions']
            or graded.get('condition_specs')!=raw['condition_specs']):
        raise ValueError('freeze complete matching seven-arm parent judgments with the same legacy judge first')
    qa.verify_question_rows(graded['evaluations'],raw['evaluations'])
    validate_rows(graded['evaluations'],raw['evaluations'],raw['conditions'],complete=True)
    for row in graded['evaluations']:
        if set(row.get('outcome',{}))!=set(raw['conditions']) or set(row.get('judge',{}))!=set(raw['conditions']):
            raise ValueError('parent has incomplete seven-arm grades')
        for name in raw['conditions']:
            grade=row['judge'][name]
            if (grade.get('rubric_id')!=qa.QA_GRADER_VERSION or grade.get('label') not in ('correct','wrong','abstain')
                    or row['outcome'][name]!=grade['label']):
                raise ValueError('unresolved or inconsistent parent semantic grade')


def load(args):
    ns,meta,bank,prepared=legacy.source_run(args.source_run)
    adapter=preference.load(args.parent_run)
    if adapter.get('pilot') or adapter['model']!=meta['model'] or adapter['model_config']!=meta['model_config']:
        raise ValueError('need complete matching E34 adapter')
    if adapter['source']!=legacy.source_identity(ns,meta,[adapter['layer']]):
        raise ValueError('adapter source changed')
    if (adapter['config']!={k:meta['config'][k] for k in adapter['config']}
            or adapter['source_system']!=source_eval.SOURCE_SYSTEM):
        raise ValueError('adapter prompts or budget differ from the declared source')
    pair_entry.verify_snapshot(args.parent_run/'training_code_snapshot',adapter['entry_implementation'])
    raw=qa.read(args.parent_run/'tune.json')
    graded=qa.read(args.parent_run/'tune_scored_items.json')
    names=[*pair_entry.PARENT_CONDITIONS,preference.METHOD]
    ids=[q['pair_id'] for q in prepared['splits']['tune']]
    if (raw.get('protocol')!=preference.PROTOCOL or not raw.get('completed') or raw.get('pilot')
            or raw['conditions']!=names or raw['identity']['adapter_sha256']!=qa.digest(args.parent_run/'preference_adapter.pt')
            or raw['identity']['source']!=adapter['source'] or raw['identity']['layer']!=adapter['layer']
            or raw['identity']['config']!=adapter['config'] or raw['identity']['source_system']!=adapter['source_system']
            or raw['identity']['batch_size']!=1 or raw['identity']['alpha']!=1.
            or raw['protocol_fingerprint']!=qa.fingerprint(raw['identity'])
            or raw.get('bank_protocol_fingerprint')!=raw['protocol_fingerprint']
            or raw['identity'].get('gold_used_at_inference') is not False
            or any(c.get('uses_reference_answer') is not False for c in raw['condition_specs'])
            or [r['pair_id'] for r in raw['evaluations']]!=ids):
        raise ValueError('parent generation identity/cohort differs')
    qa.verify_question_rows(raw['evaluations'],prepared['splits']['tune'])
    for relative in CRITICAL_CODE:
        if qa.digest(Path(__file__).resolve().parents[1]/relative)!=raw['identity']['implementation']['files'][relative]:
            raise ValueError('original E34 decoder/prompt implementation changed: '+relative)
    cfg=legacy.validate_judge_experiment(pair_entry.LEGACY_JUDGE)
    if cfg.get('full_personal_refusal_guard',False):
        raise ValueError('keep parent legacy grades unchanged; any guard is a uniform later side table')
    validate_parent_grades(raw,graded,cfg,qa.digest(args.parent_run/'tune.json'))
    return ns,meta,prepared,adapter,raw,graded


def plan(args,state):
    ns,meta,prepared,adapter,parent,graded=state
    rows=parent['evaluations'][:3] if args.pilot else parent['evaluations']
    obj={'protocol':PROTOCOL,'model':meta['model'],'source':adapter['source'],'layer':adapter['layer'],
         'source_system':adapter['source_system'],'config':adapter['config'],'batch_size':1,
         'max_new_tokens':adapter['config']['max_new_tokens'],'pilot':args.pilot,'split':'tune',
         'question_ids':[r['pair_id'] for r in rows],'seeds':list(SEEDS),
         'conditions':parent['conditions']+['size_matched_random_s%d'%s for s in SEEDS],
         'parent_generation_sha256':qa.digest(args.parent_run/'tune.json'),
         'parent_scored_sha256':qa.digest(args.parent_run/'tune_scored_items.json'),
         'target_adapter_sha256':qa.digest(args.parent_run/'preference_adapter.pt'),
         'training_called':False,'gold_used_at_inference':False,
         'learned_information_borrowed':'scalar actual write norm at the same current state/source; not orientation',
         'matching':'post-residual-cast L2; tolerance=.01*target_norm+.0001*state_norm; fail if exceeded',
         'trajectory_limit':'conditional state matching; different natural histories need not have equal marginal dose distributions',
         'random_direction_rule':'fixed normalize(randn(cpu generator seed+layer)); no gradient or fitting',
         'random_sign_rule':'sign of the same donor-minus-state projection onto its own fixed random axis',
         'judge_experiment':pair_entry.LEGACY_JUDGE,
         'judge_config_sha256':qa.fingerprint(legacy.validate_judge_experiment(pair_entry.LEGACY_JUDGE)),
         'implementation':implementation_identity()}
    obj['fingerprint']=qa.fingerprint(obj)
    return obj


def frozen_plan(args,state,create=False):
    expected=plan(args,state);path=args.run_dir/'plan.json'
    if path.exists():
        if qa.read(path)!=expected:raise ValueError('norm-control plan/config/source/code changed')
    elif create:
        args.run_dir.mkdir(parents=True,exist_ok=True);qa.save(path,expected)
        pair_entry.snapshot(args.run_dir/'code_snapshot',expected['implementation'])
    else:raise ValueError('prepare the norm-control plan first')
    return expected


def validate_rows(rows,parent_rows,names,complete=False):
    if [r['pair_id'] for r in rows]!=[r['pair_id'] for r in parent_rows]:
        raise ValueError('norm-control cohort/order differs')
    for row,parent in zip(rows,parent_rows):
        for key in ('question','gold','incorrect','best_answer','idx','prompt_token_ids'):
            if row.get(key)!=parent.get(key):raise ValueError('norm-control question/prompt changed')
        for name in parent['outputs']:
            for key in ('outputs','finished','rule_abstain','generation_token_ids','trace'):
                if row[key][name]!=parent[key][name]:raise ValueError('inherited parent data changed')
        if complete and any(set(row[key])!=set(names) for key in ('outputs','finished','rule_abstain','generation_token_ids','trace')):
            raise ValueError('incomplete norm-control conditions or execution traces')


def generate(args,state):
    ns,meta,prepared,adapter,parent,graded=state
    frozen=frozen_plan(args,state)
    rows=parent['evaluations'][:3] if args.pilot else parent['evaluations']
    path=args.run_dir/('check.json' if args.pilot else 'tune.json')
    identity={**frozen,'control_generation_sha256':frozen['parent_generation_sha256'],
              'source_mode':'matched_question','same_prefix_tokens':True,'alpha':1.}
    signature=qa.fingerprint(identity)
    if path.exists():
        result=qa.read(path)
        if result.get('identity')!=identity:raise ValueError('norm-control resume identity differs')
    else:
        result=copy.deepcopy(parent)
        result.update(protocol=PROTOCOL,identity=identity,control_identity=parent['identity'],
                      protocol_fingerprint=signature,bank_protocol_fingerprint=signature,
                      pilot=args.pilot,wiring_pilot=args.pilot,split='tune',completed=False,
                      conditions=frozen['conditions'],evaluations=copy.deepcopy(rows),
                      intervention_position='same predictors; fixed random orientation, actual stored norm matched to shadow E34',
                      E={'method':'actual-norm-matched untrained random orientation','training_called':False},C={'enabled':False})
        result['condition_specs']=copy.deepcopy(parent['condition_specs'])+[
            {'name':'size_matched_random_s%d'%s,'kind':'random_null','uses_reference_answer':False,
             'eligible_for_selection':False,'learned_size_borrowed':True,'seed':s} for s in SEEDS]
    validate_rows(result['evaluations'],rows,result['conditions'])
    bundle=qa.open_bundle(meta,prepared);qa.verify_bundle(bundle,meta,prepared)
    versions=[(p,p._version,p.requires_grad) for p in bundle.model.parameters()]
    vectors={s:control.fixed_random_direction(meta['hidden_size'],s,adapter['layer']) for s in SEEDS}
    fingerprints={s:pair_entry.tensor_fingerprint(w) for s,w in vectors.items()}
    with fp32_output_projection(bundle):
        for index,row in enumerate(result['evaluations']):
            row['idx']=index
            base,source=source_eval.prompt_tokens(bundle,adapter['config'],row['question'])
            if row['prompt_token_ids']!={'base':base,'source':source}:raise ValueError('control prompt tokenization differs')
            for seed in SEEDS:
                name='size_matched_random_s%d'%seed
                if name in row['outputs']:continue
                value=control.generate(bundle,base,source,layer=adapter['layer'],target_direction=adapter['W'],
                                       random_direction=vectors[seed],max_new_tokens=frozen['max_new_tokens'])
                row['outputs'][name]=value['output'];row['finished'][name]=value['finished']
                row['rule_abstain'][name]=bool(rule_abstains(value['output']))
                row['generation_token_ids'][name]=value['token_ids'];row['trace'][name]=value['trace']
                if args.pilot:print('NORM_NULL',meta['model'],row['pair_id'],name,repr(value['output']),flush=True)
            if args.pilot or (index+1)%16==0:
                qa.save(path,result);print('COLLECTED',index+1,len(rows),flush=True)
    if any(p._version!=v or p.requires_grad!=r for p,v,r in versions):raise RuntimeError('null changed base parameters')
    if fingerprints!={s:pair_entry.tensor_fingerprint(w) for s,w in vectors.items()}:raise RuntimeError('random vector changed')
    if implementation_identity()!=frozen['implementation']:raise RuntimeError('null runtime code changed')
    validate_rows(result['evaluations'],rows,result['conditions'],complete=True)
    result.update(completed=True,random_vectors_unchanged=True,base_parameters_unchanged=True,
                  random_vector_fingerprints={str(k):v for k,v in fingerprints.items()})
    qa.save(path,result)


def judge(args,state):
    frozen=frozen_plan(args,state)
    _,meta,_,_,parent,graded=state
    stem='check' if args.pilot else 'tune'
    raw=qa.read(args.run_dir/(stem+'.json'))
    rows=parent['evaluations'][:3] if args.pilot else parent['evaluations']
    if not raw.get('completed') or raw['identity']['fingerprint']!=frozen['fingerprint']:raise ValueError('generation incomplete or foreign')
    validate_rows(raw['evaluations'],rows,raw['conditions'],complete=True)
    cfg=legacy.validate_judge_experiment(pair_entry.LEGACY_JUDGE)
    selected=graded['evaluations'][:3] if args.pilot else graded['evaluations']
    seeded=seed_control_cache(source_dir=args.parent_run/'judge_cache',run_dir=args.run_dir,judge_config=cfg,
        scored_rows=selected,conditions=parent['conditions'],scored_sha256=frozen['parent_scored_sha256'],stem=stem)
    for k,v in seeded.items():setattr(args,k,v)
    args.model=meta['model'];args.judge_exp=pair_entry.LEGACY_JUDGE
    qa.judge(args,{**meta,'protocol_fingerprint':raw['bank_protocol_fingerprint']},stem+'.json',stem+'_scored')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','generate','judge'))
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--pilot',action='store_true')
    args=p.parse_args();state=load(args)
    if args.phase=='prepare':frozen_plan(args,state,True)
    elif args.phase=='generate':generate(args,state)
    else:judge(args,state)


if __name__=='__main__':main()
