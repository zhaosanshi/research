#!/usr/bin/env python3
"""Fixed E34 paired-content loss diagnostic, reusing six CE-parent control arms.

Only Llama L15 and Qwen L17, original CE initial_W, 92 pairs x 3 epochs (or
3 pairs x 1 pilot epoch), alpha=1/lr=.001/clip=1/seed=42. No test phase/grid.
The seventh arm alone is newly generated. Six previous outputs/traces/grades
are inherited byte-for-byte in content and bound to the CE-parent hashes.
"""
import argparse,copy
from pathlib import Path
import shutil
import torch

import common  # noqa: F401
import e25_qa_das as qa
import e27_qa_generation as legacy
import e29_qa_source_interchange as source_eval
from core import qa_source_pair_preference as preference
from core import qa_source_interchange_training as ce_training
from core.qa_generation_adapter import implementation_identity
from core.qa_precision import fp32_output_projection
from core.qa_source_interchange import greedy_source_generate,unit_direction
from core.qa_control_cache import seed_control_cache
from core.readout import rule_abstains

FIXED_LAYERS={'llama3.1-8b':15,'qwen3-8b':17}
FULL_PAIRS=92
TUNE_PAIRS=192
PARENT_CONDITIONS=(*source_eval.CONDITIONS,'trained_source_swap_das_init')
LEGACY_JUDGE='E4_qa_legacy_r20'


def tensor_fingerprint(value):
    import hashlib
    x=value.detach().cpu().contiguous()
    return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()}


def source_state(args):
    ns,meta,bank,prepared=legacy.source_run(args.source_run)
    if meta['model'] not in FIXED_LAYERS:raise ValueError('E34 is restricted to the two predeclared models')
    layer=FIXED_LAYERS[meta['model']]
    if args.pilot not in (None,3):raise ValueError('only the three-pair pilot or full fixed experiment is permitted')
    if len(prepared['e_train'])!=FULL_PAIRS or len(prepared['splits']['tune'])!=TUNE_PAIRS:
        raise ValueError('source must contain the fixed full estimation/tuning cohorts')
    ce=ce_training.load(args.ce_run);ce_training.verify_training_snapshot(args.ce_run,ce)
    expected_source=legacy.source_identity(ns,meta,[layer])
    config={k:meta['config'][k] for k in ('qa_system','qa_user','max_new_tokens','batch_size')}
    wanted={'epochs':3,'lr':.001,'clip':1.,'seed':42,'optimizer':'Adam','alpha':1.,'include_eos':False,'fixed_center':False}
    if (ce.get('pilot') or ce['model']!=meta['model'] or ce['model_config']!=meta['model_config']
            or ce['layer']!=layer or ce['source']!=expected_source or ce['initialization']!='das'
            or ce['source_system']!=source_eval.SOURCE_SYSTEM or ce['config']!=config or ce['train_config']!=wanted
            or ce['training_pair_ids']!=[r['pair_id'] for r in prepared['e_train']]):
        raise ValueError('CE parent does not match the fixed full-data same-operator comparison')
    expected_initial=unit_direction(bank['E_qa_das_L%d'%layer],meta['hidden_size']).cpu()
    if not torch.equal(ce['initial_W'],expected_initial):raise ValueError('CE initial_W is not the verified original DAS initialization')
    parent=qa.read(args.ce_run/'tune.json')
    if (parent.get('protocol')!=ce_training.PROTOCOL or parent['identity'].get('protocol')!=ce_training.PROTOCOL
            or parent.get('pilot') or not parent.get('completed') or parent['model']!=meta['model'] or parent['conditions']!=list(PARENT_CONDITIONS)
            or parent['identity']['source']!=expected_source or parent['identity']['layer']!=layer
            or parent['identity']['adapter_sha256']!=qa.digest(args.ce_run/'source_adapter.pt')
            or parent['identity']['config']!=config or parent['identity']['source_system']!=source_eval.SOURCE_SYSTEM
            or parent['identity']['batch_size']!=1 or parent['identity']['alpha']!=1.
            or parent['identity'].get('condition')!='trained_source_swap_das_init'
            or parent['identity'].get('same_prefix_tokens') is not True or parent['identity'].get('gold_used_at_inference') is not False
            or parent['identity']['question_ids']!=[r['pair_id'] for r in prepared['splits']['tune']]
            or parent.get('protocol_fingerprint')!=qa.fingerprint(parent['identity'])):
        raise ValueError('six-arm parent generation identity differs')
    qa.verify_question_rows(parent['evaluations'],prepared['splits']['tune'])
    return ns,meta,bank,prepared,ce,parent


def make_plan(args,state):
    ns,meta,bank,prepared,ce,parent=state
    judge=legacy.validate_judge_experiment(LEGACY_JUDGE)
    if judge.get('full_personal_refusal_guard',False):raise ValueError('E34 must inherit legacy grading for all seven arms; guard belongs in a later all-arm side table')
    scored_path=args.ce_run/'tune_scored_items.json';scored=qa.read(scored_path)
    if (not scored.get('completed') or not scored.get('grading_completed')
            or scored.get('grading_source_generation_sha256')!=qa.digest(args.ce_run/'tune.json')
            or scored.get('grading_judge_config_sha256')!=qa.fingerprint(judge)
            or scored.get('conditions')!=list(PARENT_CONDITIONS)):
        raise ValueError('freeze the complete same-policy CE-parent grades before E34 training')
    validate_inherited(parent['evaluations'],scored['evaluations'])
    for row in scored['evaluations']:
        for name in PARENT_CONDITIONS:
            grade=row['judge'][name]
            if (grade.get('label') not in ('correct','wrong','abstain') or grade.get('rubric_id')!=qa.QA_GRADER_VERSION
                    or row['outcome'][name]!=grade['label']):
                raise ValueError('parent grade is unresolved/foreign/inconsistent before plan freeze')
    plan={'protocol':preference.PROTOCOL,'method':preference.METHOD,'model':meta['model'],'layer':ce['layer'],
      'source':ce['source'],'source_system':ce['source_system'],'config':ce['config'],'pilot':args.pilot==3,
      'training_pair_ids':[r['pair_id'] for r in prepared['e_train'][:3 if args.pilot==3 else None]],
      'training_records_sha256':qa.fingerprint(prepared['e_train'][:3 if args.pilot==3 else None]),
      'training':{'pairs':3 if args.pilot==3 else FULL_PAIRS,'epochs':1 if args.pilot==3 else 3,'alpha':1.,'lr':.001,'clip':1.,'seed':42},
      'ce_parent':{'checkpoint_sha256':qa.digest(args.ce_run/'source_adapter.pt'),
                   'manifest_sha256':qa.digest(args.ce_run/'source_adapter.json'),
                   'generation_sha256':qa.digest(args.ce_run/'tune.json'),
                   'graded_items_sha256':qa.digest(scored_path)},
      'initial_W':tensor_fingerprint(ce['initial_W']),
      'initialization':'exact CE parent initial_W; never CE final W',
      'legacy_judge_experiment':LEGACY_JUDGE,'legacy_judge_config':judge,
      'conditions':[*PARENT_CONDITIONS,preference.METHOD],
      'question_ids':[r['pair_id'] for r in prepared['splits']['tune']],
      'implementation':implementation_identity(),
      'scope':'Previously exposed tuning questions only. Prefix normalization cancellation does not imply fixed answer rate. Not a standard DPO reproduction.'}
    plan['fingerprint']=qa.fingerprint(plan);return plan


def freeze_or_verify(args,state,create=False):
    plan=make_plan(args,state);path=args.run_dir/'plan.json'
    if path.exists():
        if qa.read(path)!=plan:raise ValueError('frozen E34 plan/source/implementation differs')
    elif create:
        args.run_dir.mkdir(parents=True,exist_ok=True);qa.save(path,plan)
    else:raise ValueError('freeze the fixed plan with prepare/train before any later phase')
    return plan


def snapshot(directory,identity):
    root=Path(__file__).resolve().parents[1]
    for rel,h in identity['files'].items():
        target=Path(directory)/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if qa.digest(root/rel)!=h:raise RuntimeError('implementation changed before snapshot')
        if target.exists() and qa.digest(target)!=h:raise ValueError('existing implementation snapshot changed')
        shutil.copyfile(root/rel,target)


def verify_snapshot(directory,identity):
    for rel,h in identity['files'].items():
        if not (Path(directory)/rel).is_file() or qa.digest(Path(directory)/rel)!=h:raise ValueError('source snapshot missing/changed: '+rel)


def train(args,state,bundle=None):
    plan=freeze_or_verify(args,state,create=True)
    ns,meta,bank,prepared,ce,parent=state
    if (args.run_dir/'preference_adapter.pt').exists() or (args.run_dir/'preference_adapter.json').exists():
        return verified_adapter(args,state)
    if bundle is None:bundle=qa.open_bundle(meta,prepared)
    qa.verify_bundle(bundle,meta,prepared)
    records=prepared['e_train'][:3 if plan['pilot'] else None]
    torch.manual_seed(42)
    result=preference.train(bundle,records,meta['config'],ce['initial_W'],layer=ce['layer'],source_system=ce['source_system'],
        pilot=plan['pilot'],identity={'source':ce['source'],'fixed_plan_sha256':qa.digest(args.run_dir/'plan.json'),
                                    'ce_parent':plan['ce_parent'],'entry_implementation':plan['implementation']})
    if not torch.equal(result['initial_W'],ce['initial_W']):raise RuntimeError('preference initialization differs bitwise from CE starting point')
    if implementation_identity()!=plan['implementation']:raise RuntimeError('implementation changed while training')
    preference.save(result,args.run_dir)
    snapshot(args.run_dir/'training_code_snapshot',plan['implementation'])
    verify_snapshot(args.run_dir/'training_code_snapshot',plan['implementation'])
    return result


def verified_adapter(args,state):
    plan=freeze_or_verify(args,state)
    adapter=preference.load(args.run_dir);ce=state[4]
    if (adapter['source']!=plan['source'] or adapter['layer']!=plan['layer'] or adapter['model']!=plan['model']
            or adapter['source_system']!=plan['source_system'] or adapter['config']!=plan['config']
            or adapter['fixed_plan_sha256']!=qa.digest(args.run_dir/'plan.json') or adapter['ce_parent']!=plan['ce_parent']
            or adapter['training_pair_ids']!=plan['training_pair_ids'] or adapter['pilot']!=plan['pilot']
            or adapter['entry_implementation']!=plan['implementation'] or not torch.equal(adapter['initial_W'],ce['initial_W'])):
        raise ValueError('preference adapter/plan/source/initialization differs')
    verify_snapshot(args.run_dir/'training_code_snapshot',adapter['entry_implementation'])
    return adapter


def control_rows(state,pilot):
    parent=state[5];questions=state[3]['splits']['tune'][:3] if pilot else state[3]['splits']['tune']
    rows=copy.deepcopy(parent['evaluations'][:len(questions)])
    qa.verify_question_rows(rows,questions)
    return rows,questions


def validate_inherited(parent_rows,new_rows):
    if len(parent_rows)!=len(new_rows):raise ValueError('inherited control cohort changed')
    qa.verify_question_rows(new_rows,parent_rows)
    fields=('outputs','finished','rule_abstain','generation_token_ids','trace')
    for old,new in zip(parent_rows,new_rows):
        if old['prompt_token_ids']!=new['prompt_token_ids']:raise ValueError('inherited prompt tokens changed')
        for field in fields:
            for condition in PARENT_CONDITIONS:
                if new[field][condition]!=old[field][condition]:raise ValueError('copied control field changed: '+field+'/'+condition)
        for condition,value in old.get('cache_reference_checks',{}).items():
            if new.get('cache_reference_checks',{}).get(condition)!=value:raise ValueError('inherited cache trace changed')


def generate(args,state,adapter,pilot,bundle=None):
    plan=freeze_or_verify(args,state);ns,meta,bank,prepared,ce,parent=state
    if adapter['pilot'] and not pilot:raise ValueError('three-pair training pilot cannot enter full generation')
    rows,questions=control_rows(state,pilot);stem='check' if pilot else 'tune';path=args.run_dir/(stem+'.json')
    identity={'protocol':preference.PROTOCOL,'model':meta['model'],'source':adapter['source'],'layer':adapter['layer'],
      'config':adapter['config'],'source_system':adapter['source_system'],'batch_size':1,'alpha':1.,
      'question_ids':[q['pair_id'] for q in questions],'adapter_sha256':qa.digest(args.run_dir/'preference_adapter.pt'),
      'control_generation_sha256':qa.digest(args.ce_run/'tune.json'),'condition':preference.METHOD,
      'fixed_plan_sha256':qa.digest(args.run_dir/'plan.json'),'implementation':plan['implementation'],
      'split':'tune','pilot':pilot,'same_prefix_tokens':True,'gold_used_at_inference':False,
      'controls_reused_not_rerun':list(PARENT_CONDITIONS)}
    if path.exists():
        saved=qa.read(path)
        if saved.get('identity')!=identity:raise ValueError('generation identity changed')
        validate_inherited(rows,saved['evaluations'])
        if saved.get('completed'):return saved
        raise ValueError('partial generation preserved; use a fresh run rather than silently mixing attempts')
    if bundle is None:bundle=qa.open_bundle(meta,prepared)
    qa.verify_bundle(bundle,meta,prepared)
    versions=[(p,p._version,p.requires_grad) for p in bundle.model.parameters()]
    result=copy.deepcopy(parent)
    result.update(protocol=preference.PROTOCOL,identity=identity,control_identity=parent['identity'],
      ancestor_control_identity=parent.get('control_identity'),control_generation_source=str(args.ce_run/'tune.json'),
      protocol_fingerprint=qa.fingerprint(identity),bank_protocol_fingerprint=qa.fingerprint(identity),
      training_sha256=identity['adapter_sha256'],pilot=bool(pilot or adapter['pilot']),wiring_pilot=pilot,completed=False,
      conditions=[*PARENT_CONDITIONS,preference.METHOD],evaluations=rows,
      E={'method':preference.METHOD,'layer':adapter['layer'],'alpha':1.,'initialization':'exact CE initial_W'},C={'enabled':False})
    result['condition_specs']=copy.deepcopy(parent['condition_specs'])+[{'name':preference.METHOD,'kind':'E','uses_reference_answer':False,'eligible_for_selection':False}]
    snapshot(args.run_dir/'generation_code_snapshot',plan['implementation'])
    with fp32_output_projection(bundle):
        for index,row in enumerate(rows):
            row['idx']=index;base,source=source_eval.prompt_tokens(bundle,adapter['config'],row['question'])
            if row['prompt_token_ids']!={'base':base,'source':source}:raise ValueError('new prompt tokens differ from the six control arms')
            generated=greedy_source_generate(bundle,base,source,mode='source_interchange',layer=adapter['layer'],
                                             direction=adapter['W'],max_new_tokens=adapter['config']['max_new_tokens'])
            method=preference.METHOD
            row['outputs'][method]=generated['output'];row['finished'][method]=generated['finished']
            row['rule_abstain'][method]=bool(rule_abstains(generated['output']))
            row['generation_token_ids'][method]=generated['token_ids'];row['trace'][method]=generated['trace']
            if pilot:
                row.setdefault('cache_reference_checks',{})[method]=source_eval.cache_reference_check(bundle,base,source,adapter['layer'],adapter['W'],generated['token_ids'])
                row['pair_preference_same_prompt_identity_checks']=source_eval.same_prompt_identity_check(bundle,base,adapter['layer'],adapter['W'],generated['token_ids'])
            print('PAIR_PREFERENCE_SOURCE_SWAP',meta['model'],row['pair_id'],repr(generated['output']),flush=True)
            qa.save(path,result)
    validate_inherited(parent['evaluations'][:len(rows)],rows)
    if any(p._version!=v or p.requires_grad!=g for p,v,g in versions):raise RuntimeError('generation changed base model parameters')
    if implementation_identity()!=plan['implementation']:raise RuntimeError('generation implementation changed')
    result.update(completed=True,base_parameters_unchanged=True);qa.save(path,result);return result


def freeze_control_grades(args,state,adapter,stem):
    plan=freeze_or_verify(args,state);parent=state[5];rows,questions=control_rows(state,stem=='check')
    scored_path=args.ce_run/'tune_scored_items.json';scored=qa.read(scored_path)
    cfg=legacy.validate_judge_experiment(LEGACY_JUDGE)
    if qa.digest(scored_path)!=plan['ce_parent']['graded_items_sha256']:raise ValueError('parent grades changed after the fixed E34 plan')
    if (not scored.get('completed') or not scored.get('grading_completed')
            or scored.get('grading_source_generation_sha256')!=qa.digest(args.ce_run/'tune.json')
            or scored.get('grading_judge_config_sha256')!=qa.fingerprint(cfg)
            or scored.get('conditions')!=list(PARENT_CONDITIONS)):
        raise ValueError('complete legacy six-arm grades with exact judge settings are required')
    qa.verify_question_rows(scored['evaluations'],parent['evaluations'])
    validate_inherited(parent['evaluations'],scored['evaluations'])
    for r in scored['evaluations']:
        for c in PARENT_CONDITIONS:
            if r['judge'][c].get('label') not in ('correct','wrong','abstain') or r['judge'][c].get('rubric_id')!=qa.QA_GRADER_VERSION:
                raise ValueError('unresolved/foreign legacy grade')
    generated=qa.read(args.run_dir/(stem+'.json'))
    if generated['identity']['adapter_sha256']!=qa.digest(args.run_dir/'preference_adapter.pt'):
        raise ValueError('graded generation uses a different preference checkpoint')
    if generated['identity']['control_generation_sha256']!=qa.digest(args.ce_run/'tune.json'):
        raise ValueError('graded generation inherited a different CE parent')
    validate_inherited(rows,generated['evaluations'])
    if generated.get('protocol_fingerprint')!=qa.fingerprint(generated['identity']) or generated.get('bank_protocol_fingerprint')!=generated['protocol_fingerprint']:
        raise ValueError('new generation fingerprint changed')
    selected=scored['evaluations'][:len(questions)]
    seed=seed_control_cache(source_dir=args.ce_run/'judge_cache',run_dir=args.run_dir,judge_config=cfg,
        scored_rows=selected,conditions=PARENT_CONDITIONS,scored_sha256=qa.digest(scored_path),stem=stem)
    for name,value in seed.items():setattr(args,name,value)
    return len(selected)*len(PARENT_CONDITIONS)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','train','check','generate-tune','judge-check','judge-tune'))
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--ce-run',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--pilot',type=int,choices=(3,));args=p.parse_args();state=source_state(args)
    if args.phase=='prepare':freeze_or_verify(args,state,create=True);return
    if args.phase=='train':train(args,state);return
    adapter=verified_adapter(args,state)
    if args.phase.startswith('judge-'):
        args.model=state[1]['model'];args.judge_exp=LEGACY_JUDGE;stem=args.phase.split('-',1)[1]
        count=freeze_control_grades(args,state,adapter,stem);print('FROZEN_CONTROL_GRADES',count,flush=True)
        raw=qa.read(args.run_dir/(stem+'.json'))
        qa.judge(args,{**state[1],'protocol_fingerprint':raw['bank_protocol_fingerprint']},stem+'.json',stem+'_scored')
    else:generate(args,state,adapter,args.phase=='check')

if __name__=='__main__':main()
