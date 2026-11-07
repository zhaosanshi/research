#!/usr/bin/env python3
"""Retrain the fixed E34 operator with only source question attention keys masked."""
import argparse
import copy
from pathlib import Path
import sys

import torch

CODE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(CODE/'scripts'),str(CODE)]
import e25_qa_das as qa
import e27_qa_generation as legacy
import e29_qa_source_interchange as source_eval
import e34_qa_source_pair_preference as e34
from core import qa_source_pair_preference as informed
from core import qa_source_masked_preference as masked
from core.qa_source_question_mask import SOURCE_MODE,locate_question_mask,greedy_masked_source_generate
from core.qa_generation_adapter import implementation_identity
from core.qa_precision import fp32_output_projection
from core.qa_control_cache import seed_control_cache
from core.readout import rule_abstains

PARENTS=[*e34.PARENT_CONDITIONS,informed.METHOD]


def state(args):
    original=e34.source_state(args)
    ns,meta,bank,prepared,ce,_=original
    adapter=informed.load(args.e34_run);parent=qa.read(args.e34_run/'tune.json');grade=qa.read(args.e34_run/'tune_scored_items.json')
    cfg=legacy.validate_judge_experiment(e34.LEGACY_JUDGE)
    if (adapter['pilot'] or adapter['model']!=meta['model'] or adapter['layer']!=ce['layer']
            or adapter['source']!=ce['source'] or adapter['config']!=ce['config'] or adapter['source_system']!=ce['source_system']
            or adapter['training_pair_ids']!=[r['pair_id'] for r in prepared['e_train']]
            or not torch.equal(adapter['initial_W'],ce['initial_W'])
            or adapter['ce_parent']['checkpoint_sha256']!=qa.digest(args.ce_run/'source_adapter.pt')
            or adapter['ce_parent']['generation_sha256']!=qa.digest(args.ce_run/'tune.json')
            or parent.get('completed') is not True or parent.get('pilot') or parent['protocol']!=informed.PROTOCOL
            or parent['conditions']!=PARENTS or parent['model']!=meta['model']
            or parent['identity']['source']!=adapter['source'] or parent['identity']['config']!=adapter['config']
            or parent['identity']['source_system']!=adapter['source_system'] or parent['identity']['layer']!=adapter['layer']
            or parent['identity']['adapter_sha256']!=qa.digest(args.e34_run/'preference_adapter.pt')
            or parent.get('protocol_fingerprint')!=qa.fingerprint(parent['identity'])
            or grade.get('grading_completed') is not True or grade.get('completed') is not True
            or grade['conditions']!=PARENTS or grade['grading_judge_config_sha256']!=qa.fingerprint(cfg)
            or grade['grading_source_generation_sha256']!=qa.digest(args.e34_run/'tune.json')):
        raise ValueError('question-masked control requires the unchanged full fixed-layer E34 parent')
    e34.verify_snapshot(args.e34_run/'training_code_snapshot',adapter['entry_implementation'])
    qa.verify_question_rows(parent['evaluations'],prepared['splits']['tune'])
    inherited(parent['evaluations'],grade['evaluations'])
    for row in grade['evaluations']:
        for name in PARENTS:
            g=row['judge'][name]
            if g.get('rubric_id')!=qa.QA_GRADER_VERSION or g.get('label') not in ('correct','wrong','abstain') or row['outcome'][name]!=g['label']:
                raise ValueError('parent semantic grades are unresolved/foreign/inconsistent')
    return original,adapter,parent,grade,cfg


def inherited(old,new):
    qa.verify_question_rows(new,old)
    if len(new)!=len(old):raise ValueError('parent condition cohort changed')
    for a,b in zip(old,new):
        if a['prompt_token_ids']!=b['prompt_token_ids']:raise ValueError('parent prompt IDs changed')
        for key in ('outputs','finished','rule_abstain','generation_token_ids','trace'):
            if any(a[key][n]!=b[key][n] for n in PARENTS):raise ValueError('inherited parent content changed: '+key)


def plan(args,st,create=False):
    original,old,parent,grade,judge=st;prepared=original[3];ce=original[4]
    records=prepared['e_train'][:3 if args.pilot==3 else None]
    value={'protocol':masked.PROTOCOL,'method':masked.METHOD,'source_mode':SOURCE_MODE,
        'model':old['model'],'layer':old['layer'],'source':old['source'],'config':old['config'],'source_system':old['source_system'],
        'pilot':args.pilot==3,'training_pair_ids':[r['pair_id'] for r in records],
        'training_records_sha256':qa.fingerprint(records),'initial_W':e34.tensor_fingerprint(ce['initial_W']),
        'training':{'pairs':len(records),'epochs':1 if args.pilot==3 else 3,'lr':.001,'clip':1.,'seed':42,'alpha':1.},
        'ce_parent':old['ce_parent'],
        'informed_parent':{'checkpoint_sha256':qa.digest(args.e34_run/'preference_adapter.pt'),
            'manifest_sha256':qa.digest(args.e34_run/'preference_adapter.json'),
            'generation_sha256':qa.digest(args.e34_run/'tune.json'),'graded_items_sha256':qa.digest(args.e34_run/'tune_scored_items.json')},
        'conditions':[*PARENTS,masked.METHOD],'question_ids':[r['pair_id'] for r in prepared['splits']['tune']],
        'judge_config':judge,'implementation':implementation_identity(),
        'entry_sha256':qa.digest(__file__),
        'scope':'Source question attention keys masked; original absolute positions, system/roles/length/public answer prefix retained. Not completely information-free. Previously exposed tuning cohort only.'}
    value['fingerprint']=qa.fingerprint(value);path=args.run_dir/'plan.json'
    if path.exists():
        if qa.read(path)!=value:raise ValueError('masked-source fixed plan/source/code changed')
    elif create:args.run_dir.mkdir(parents=True,exist_ok=True);qa.save(path,value)
    else:raise ValueError('freeze plan before later phases')
    return value


def verified(args,st):
    p=plan(args,st);a=masked.load(args.run_dir);old=st[1]
    if (a['fixed_plan_sha256']!=qa.digest(args.run_dir/'plan.json') or a['source']!=p['source']
            or a['informed_parent']!=p['informed_parent'] or a['entry_implementation']!=p['implementation']
            or a['training_pair_ids']!=p['training_pair_ids'] or a['pilot']!=p['pilot']
            or a['layer']!=p['layer'] or a['config']!=p['config'] or a['source_system']!=p['source_system']
            or not torch.equal(a['initial_W'],old['initial_W'])):
        raise ValueError('masked-source checkpoint does not match plan or original initialization')
    e34.verify_snapshot(args.run_dir/'training_code_snapshot',a['entry_implementation']);return a


def train(args,st,bundle=None):
    p=plan(args,st,create=True)
    if (args.run_dir/'preference_adapter.pt').exists():return verified(args,st)
    original=st[0];meta,prepared,ce=original[1],original[3],original[4]
    if bundle is None:bundle=qa.open_bundle(meta,prepared)
    qa.verify_bundle(bundle,meta,prepared);torch.manual_seed(42)
    records=prepared['e_train'][:3 if p['pilot'] else None]
    a=masked.train(bundle,records,meta['config'],ce['initial_W'],layer=ce['layer'],source_system=ce['source_system'],pilot=p['pilot'],
        identity={'source':ce['source'],'fixed_plan_sha256':qa.digest(args.run_dir/'plan.json'),
                  'ce_parent':p['ce_parent'],'informed_parent':p['informed_parent'],'entry_implementation':p['implementation']})
    if implementation_identity()!=p['implementation'] or qa.digest(__file__)!=p['entry_sha256']:
        raise RuntimeError('masked control implementation changed during training')
    masked.save(a,args.run_dir);e34.snapshot(args.run_dir/'training_code_snapshot',p['implementation']);return a


def generate(args,st,a,pilot,bundle=None):
    p=plan(args,st);original,old,parent,grade,judge=st;meta,prepared=original[1],original[3]
    if a['pilot'] and not pilot:raise ValueError('pilot-trained mask direction cannot enter full192')
    if not pilot:
        check=qa.read(args.run_dir/'check.json')
        if not check.get('completed') or check['identity']['adapter_sha256']!=qa.digest(args.run_dir/'preference_adapter.pt'):
            raise ValueError('three-question generation check is required before full192')
    rows=copy.deepcopy(parent['evaluations'][:3] if pilot else parent['evaluations'])
    stem='check' if pilot else 'tune';path=args.run_dir/(stem+'.json')
    identity={'protocol':masked.PROTOCOL,'source_mode':SOURCE_MODE,'model':p['model'],'source':p['source'],'layer':p['layer'],
        'config':p['config'],'source_system':p['source_system'],'alpha':1.,'batch_size':1,'same_prefix_tokens':True,
        'condition':masked.METHOD,'question_ids':[r['pair_id'] for r in rows],'adapter_sha256':qa.digest(args.run_dir/'preference_adapter.pt'),
        'source_mask_manifest_sha256':a['source_mask_manifest_sha256'],'fixed_plan_sha256':qa.digest(args.run_dir/'plan.json'),
        'control_generation_sha256':p['informed_parent']['generation_sha256'],'informed_parent':p['informed_parent'],
        'implementation':p['implementation'],'gold_used_at_inference':False,'split':'tune','pilot':pilot,
        'controls_reused_not_rerun':PARENTS}
    if path.exists():
        saved=qa.read(path)
        if saved.get('identity')!=identity:raise ValueError('generation identity changed')
        inherited(parent['evaluations'][:len(rows)],saved['evaluations'])
        if saved.get('completed'):return saved
        raise ValueError('partial masked-source run preserved; use a fresh directory')
    if bundle is None:bundle=qa.open_bundle(meta,prepared)
    qa.verify_bundle(bundle,meta,prepared);versions=[(x,x._version,x.requires_grad) for x in bundle.model.parameters()]
    result=copy.deepcopy(parent);result.update(protocol=masked.PROTOCOL,source_mode=SOURCE_MODE,identity=identity,
        control_identity=parent['identity'],ancestor_control_identity=parent.get('control_identity'),
        control_generation_source=str(args.e34_run/'tune.json'),training_sha256=identity['adapter_sha256'],
        protocol_fingerprint=qa.fingerprint(identity),bank_protocol_fingerprint=qa.fingerprint(identity),
        conditions=[*PARENTS,masked.METHOD],condition_specs=copy.deepcopy(parent['condition_specs'])+[
            {'name':masked.METHOD,'kind':'E','uses_reference_answer':False,'eligible_for_selection':False,'source_mode':SOURCE_MODE}],
        completed=False,pilot=bool(pilot or a['pilot']),wiring_pilot=pilot,evaluations=rows,
        E={'method':masked.METHOD,'layer':a['layer'],'alpha':1.},C={'enabled':False})
    e34.snapshot(args.run_dir/'generation_code_snapshot',p['implementation'])
    if pilot:result['learned_direction_mask_verification']=masked.verify_mask_control(bundle,meta['config'],prepared['e_train'][:3],a['source_system'],a['layer'],a['W'])
    with fp32_output_projection(bundle):
        for row in rows:
            base,source=source_eval.prompt_tokens(bundle,a['config'],row['question'])
            m=locate_question_mask(bundle,a['config'],a['source_system'],row['question'])
            if row['prompt_token_ids']!={'base':base,'source':source} or m['source_prompt_ids']!=source:
                raise ValueError('source-mask prompt differs from inherited seven arms')
            g=greedy_masked_source_generate(bundle,base,source,source_prompt_attention_mask=m['source_prompt_attention_mask'],
                layer=a['layer'],direction=a['W'],max_new_tokens=a['config']['max_new_tokens'])
            name=masked.METHOD;row['outputs'][name]=g['output'];row['finished'][name]=g['finished']
            row['rule_abstain'][name]=bool(rule_abstains(g['output']));row['generation_token_ids'][name]=g['token_ids'];row['trace'][name]=g['trace']
            row['source_question_mask']=m
            print('MASKED_SOURCE',p['model'],row['pair_id'],repr(g['output']),flush=True);qa.save(path,result)
    inherited(parent['evaluations'][:len(rows)],rows)
    if any(x._version!=v or x.requires_grad!=g for x,v,g in versions):raise RuntimeError('base model changed')
    if implementation_identity()!=p['implementation']:raise RuntimeError('generation code changed')
    result.update(completed=True,base_parameters_unchanged=True);qa.save(path,result);return result


def judge(args,st,stem):
    p=plan(args,st);parent,grade,cfg=st[2],st[3],st[4];raw=qa.read(args.run_dir/(stem+'.json'))
    if not raw.get('completed') or raw['identity']['fixed_plan_sha256']!=qa.digest(args.run_dir/'plan.json'):
        raise ValueError('generate complete matching raw before grading')
    inherited(parent['evaluations'][:len(raw['evaluations'])],raw['evaluations'])
    if qa.digest(args.e34_run/'tune_scored_items.json')!=p['informed_parent']['graded_items_sha256']:
        raise ValueError('parent grades changed after freezing')
    seeded=seed_control_cache(source_dir=args.e34_run/'judge_cache',run_dir=args.run_dir,judge_config=cfg,
        scored_rows=grade['evaluations'],conditions=PARENTS,scored_sha256=p['informed_parent']['graded_items_sha256'],stem=stem)
    for key,value in seeded.items():setattr(args,key,value)
    by_id={r['pair_id']:r for r in grade['evaluations']}
    args.frozen_control_grades={r['pair_id']:{n:by_id[r['pair_id']]['judge'][n] for n in PARENTS} for r in raw['evaluations']}
    args.model=p['model'];args.judge_exp=e34.LEGACY_JUDGE
    qa.judge(args,{**st[0][1],'protocol_fingerprint':raw['bank_protocol_fingerprint']},stem+'.json',stem+'_scored')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','train','check','generate','judge-check','judge'))
    for name in ('source-run','ce-run','e34-run','run-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pilot',type=int,choices=(3,));a=p.parse_args();st=state(a)
    if a.phase=='prepare':plan(a,st,create=True)
    elif a.phase=='train':train(a,st)
    elif a.phase in ('judge-check','judge'):judge(a,st,'check' if a.phase=='judge-check' else 'tune')
    else:generate(a,st,verified(a,st),a.phase=='check')

if __name__=='__main__':main()
