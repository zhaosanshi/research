#!/usr/bin/env python3
"""E40: Llama-only C14 extension after the prespecified E39 training gate pass.
Inherit ten E37 arms/grades exactly; generate four new arms, fixed E15/rho/budget.
"""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import e25_qa_das as qa
import e26_matched_coverage as joint
import e27_qa_generation as legacy
import e29_qa_source_interchange as source_eval
import e34_qa_source_pair_preference as pair_entry
import e37_source_commitment_order as parent_entry
from core import qa_source_pair_preference as preference
from core import qa_source_commitment as operation
from core.joint_steering import unit
from core.qa_control_cache import seed_control_cache
from core.qa_generation_adapter import implementation_identity
from core.qa_precision import fp32_output_projection
from core.readout import rule_abstains

PROTOCOL='source_e_commitment_neighbor_extension_v1'
MODEL='llama3.1-8b'
JUDGE='E4_qa'
ARM_FIELDS=('outputs','finished','rule_abstain','generation_token_ids','trace','gate_readout')


def new_conditions():
    out=[]
    for suffix,alpha in (('m1',-1.),('p1',1.)):
        for enabled in (False,True):
            kind='EC' if enabled else 'C'
            out.append({'name':kind+'_neighbor_'+suffix,'kind':kind,'e_enabled':enabled,
                'e':{'layer':15,'alpha':1.} if enabled else None,
                'c':{'layer':14,'side':'neighbor','alpha':alpha,'position':'base_prompt_end_only'},
                'uses_reference_answer':False})
    return out


def validate_inheritance(raw,parent,questions,complete=False):
    if raw['control_identity']!=parent['identity'] or raw['parent_protocol']!=parent['protocol']:
        raise ValueError('parent identity changed')
    qa.verify_question_rows(raw['evaluations'],questions[:len(raw['evaluations'])])
    if complete and len(raw['evaluations'])!=len(questions):raise ValueError('incomplete child cohort')
    if len(raw['evaluations'])>len(questions):raise ValueError('foreign child cohort')
    for row,old in zip(raw['evaluations'],parent['evaluations']):
        if {k:v for k,v in row.items() if k not in ARM_FIELDS}!={k:v for k,v in old.items() if k not in ARM_FIELDS}:
            raise ValueError('parent question/prompts/tokens/metadata changed')
        for field in ARM_FIELDS:
            if {k:row[field].get(k) for k in parent['conditions']}!=old[field]:
                raise ValueError('inherited parent arm changed: '+field)
    parent_entry.validate_rows(raw,questions,complete)


def state(a):
    ns,meta,_,prepared=legacy.source_run(a.source_run)
    adapter=preference.load(a.e_run);pair_entry.verify_snapshot(a.e_run/'training_code_snapshot',adapter['entry_implementation'])
    cm,bank,cd=joint.load_run(SimpleNamespace(run_dir=a.c_run,model=MODEL))
    parent=qa.read(a.parent_run/'tune.json');graded=qa.read(a.parent_grades)
    lock=qa.read(a.parent_run/'reference_lock_audit.json');manifest=qa.read(a.parent_run/'reference_lock_manifest.json')
    li=lock['identity']
    if (li!=manifest['identity'] or li['model']!=MODEL or li['reference_conflicts']!=0
        or li['changed_final_labels']!=0 or lock['changed_labels'] or not li['all_existing_v1_results_preserved']
        or li['current_generation_sha256']!=qa.digest(a.parent_run/'tune.json')
        or li['current_grade_sha256']!=qa.digest(a.parent_grades)
        or manifest['files'].get('tune.json')!=li['current_generation_sha256']
        or manifest['files'].get('tune_scored_items.json')!=li['current_grade_sha256']):
        raise ValueError('complete zero-drift reference-locked parent required')
    if (meta['model']!=MODEL or adapter['model']!=MODEL or adapter['layer']!=15 or adapter.get('pilot')
        or len(adapter['training_pair_ids'])!=92 or adapter['source']!=legacy.source_identity(ns,meta,[15])
        or cm['model_config']!=meta['model_config'] or adapter['model_config']!=meta['model_config']
        or not parent.get('completed') or parent.get('pilot') or parent.get('protocol')!=operation.PROTOCOL
        or parent['model']!=MODEL or len(parent['evaluations'])!=192 or parent['conditions']!=[c['name'] for c in parent_entry.conditions(MODEL)]
        or parent['identity']!=qa.read(a.parent_run/'plan.json')
        or parent['identity']['adapter_sha256']!=qa.digest(a.e_run/'preference_adapter.pt')
        or parent['identity']['source']!=adapter['source'] or parent['identity']['config']!=adapter['config']
        or parent['identity']['max_new_tokens']!=48 or parent['identity']['e_layer']!=15
        or parent['identity']['c_bank_meta_sha256']!=qa.digest(a.c_run/'bank.json')
        or parent['identity']['c_bank_sha256']!=qa.digest(a.c_run/'bank.npz')
        or parent['identity']['c_data_sha256']!=qa.digest(a.c_run/'data.json')):
        raise ValueError('E40 needs the complete fixed Llama E37/E34/corrected-C parent')
    pair_entry.verify_snapshot(a.parent_run/'code_snapshot',parent['identity']['implementation'])
    code=Path(__file__).resolve().parents[1]
    for rel,h in parent['identity']['implementation']['files'].items():
        if rel.startswith('core/') or rel in ('scripts/common.py','scripts/e29_qa_source_interchange.py'):
            if qa.digest(code/rel)!=h:raise ValueError('fixed parent runtime bytes changed: '+rel)
    parent_entry.validate_rows(parent,prepared['splits']['tune'],True)
    judge=legacy.validate_judge_experiment(JUDGE)
    if (not graded.get('grading_completed') or graded.get('grading_source_generation_sha256')!=qa.digest(a.parent_run/'tune.json')
        or graded.get('grading_judge_config_sha256')!=qa.fingerprint(judge)
        or graded['conditions']!=parent['conditions'] or graded['condition_specs']!=parent['condition_specs']
        or not judge.get('full_personal_refusal_guard')):
        raise ValueError('freeze complete reference-locked parent grades under the same uniform judge')
    qa.verify_question_rows(graded['evaluations'],parent['evaluations'])
    for r,old in zip(graded['evaluations'],parent['evaluations']):
        for field in ARM_FIELDS:
            if r[field]!=old[field]:raise ValueError('parent graded text/token/trace differs from raw')
        for name in parent['conditions']:
            if r['judge'][name]['label'] not in ('correct','wrong','abstain') or r['outcome'][name]!=r['judge'][name]['label']:
                raise ValueError('unresolved/inconsistent parent grade')
    p=qa.read(a.e39_run/'plan.json');cal=qa.read(a.e39_run/'calibration.json');gate=qa.read(a.e39_run/'train_gate.json');report=qa.read(a.e39_run/'report.json')
    if (p['protocol']!='e39_neighbor_c_train_function_v1' or p['model']!=MODEL or p['c_layer']!=14
        or p['e_layer_reference_only']!=15 or p['source_sha256']['e34/preference_adapter.pt']!=qa.digest(a.e_run/'preference_adapter.pt')
        or p['source_sha256']['C/bank.json']!=qa.digest(a.c_run/'bank.json')
        or p['source_sha256']['C/bank.npz']!=qa.digest(a.c_run/'bank.npz')
        or p['source_sha256']['e37_calibration/c_calibration.json']!=parent['identity']['c_calibration_sha256']
        or cal['plan_sha256']!=qa.digest(a.e39_run/'plan.json') or cal['states_sha256']!=qa.digest(a.e39_run/'native_states.pt')
        or gate['plan_sha256']!=cal['plan_sha256'] or not gate['completed'] or gate['check_first3']
        or gate['calibration_sha256']!=qa.digest(a.e39_run/'calibration.json')
        or report['raw_sha256']!=qa.digest(a.e39_run/'train_gate.json') or not report['summary']['qualifies_training_function']
        or cal['fit']['rho']!=p['rho'] or cal['fit']['absolute_error']>cal['fit']['tolerance']
        or [r['pair_id'] for r in gate['rows']]!=adapter['training_pair_ids']):
        raise ValueError('neighboring C lacks the frozen training-only functional qualification')
    expected_train=[{k:r[k] for k in ('pair_id','question')} for r in prepared['e_train']]
    if p['questions']!=expected_train or [{k:r[k] for k in ('pair_id','question')} for r in gate['rows']]!=expected_train:
        raise ValueError('E39 question text/cohort differs from fixed E34 training')
    if not np.isfinite(cal['fit']['coefficient']) or cal['fit']['coefficient']<=0:
        raise ValueError('invalid neighboring C coefficient')
    for name,sign in [('C_minus',-1),('C_plus',1)]:
        delta=np.asarray([r['conditions'][name]['yes_minus_no']-r['conditions']['baseline']['yes_minus_no'] for r in gate['rows']])
        if not np.isfinite(delta).all() or (sign*delta>0).sum()<83 or np.median(abs(delta))<.1:
            raise ValueError('E39 does not independently pass the predeclared function criteria')
    direction=torch.as_tensor(unit(bank['qa_das_C_L14']),dtype=torch.float32)
    if joint.fingerprint(direction.tolist())!=p['C_direction_sha256']:raise ValueError('neighbor C vector changed')
    return dict(ns=ns,meta=meta,prepared=prepared,adapter=adapter,cmeta=cm,C=direction,parent=parent,graded=graded,
        judge=judge,coefficient=cal['fit']['coefficient'],rho=p['rho'],e39_plan=p)


def plan(a,s):
    parent=s['parent'];questions=s['prepared']['splits']['tune'][:3] if a.pilot else s['prepared']['splits']['tune']
    identity=copy.deepcopy(parent['identity'])
    identity.update(protocol=PROTOCOL,pilot=a.pilot,question_ids=[r['pair_id'] for r in questions],
        all_arms_freshly_generated=False,all_controls_inherited_exactly=True,
        parent_generation_sha256=qa.digest(a.parent_run/'tune.json'),parent_scored_sha256=qa.digest(a.parent_grades),
        parent_plan_sha256=qa.digest(a.parent_run/'plan.json'),
        parent_reference_lock_audit_sha256=qa.digest(a.parent_run/'reference_lock_audit.json'),
        parent_reference_lock_manifest_sha256=qa.digest(a.parent_run/'reference_lock_manifest.json'),neighbor_c_layer=14,neighbor_c_coefficient=s['coefficient'],
        neighbor_rho=s['rho'],implementation=implementation_identity())
    for name in ('plan','report','calibration','train_gate'):
        identity['e39_'+('gate' if name=='train_gate' else name)+'_sha256']=qa.digest(a.e39_run/(name+'.json'))
    identity['e39_states_sha256']=qa.digest(a.e39_run/'native_states.pt')
    identity['condition_specs']=copy.deepcopy(parent['condition_specs'])+new_conditions()
    identity['conditions']=[c['name'] for c in identity['condition_specs']]
    identity['c_layers']['neighbor']=14;identity['c_coefficients']['neighbor']=s['coefficient']
    identity['c_directions']['neighbor']={'layer':14,'scale':s['e39_plan']['C_original_scale'],
        'vector':pair_entry.tensor_fingerprint(s['C']),'qualification_report_sha256':identity['e39_report_sha256']}
    identity['reference_use']='inherit all ten parent arms exactly; same frozen parent grades; no claims they were freshly generated'
    identity.pop('fingerprint',None);identity['fingerprint']=qa.fingerprint(identity)
    return identity


def frozen(a,s,create=False):
    expected=plan(a,s);path=a.run_dir/'plan.json'
    if path.exists():
        if qa.read(path)!=expected:raise ValueError('frozen extension source/grades/code changed')
    elif create:
        a.run_dir.mkdir(parents=True,exist_ok=True);pair_entry.snapshot(a.run_dir/'code_snapshot',expected['implementation']);qa.save(path,expected)
    else:raise ValueError('prepare extension first')
    pair_entry.verify_snapshot(a.run_dir/'code_snapshot',expected['implementation'])
    return expected


def parameters(c,s):
    return {'e_layer':15 if c['e_enabled'] else None,'e_direction':s['adapter']['W'] if c['e_enabled'] else None,
        'c_layer':14,'c_direction':s['C'],'c_scale':s['coefficient'],'c_alpha':c['c']['alpha']}


def generate(a,s,bundle=None):
    identity=frozen(a,s);parent=s['parent'];questions=s['prepared']['splits']['tune'][:3] if a.pilot else s['prepared']['splits']['tune']
    stem='check' if a.pilot else 'tune';path=a.run_dir/(stem+'.json')
    if path.exists():
        result=qa.read(path)
        if result['identity']!=identity:raise ValueError('foreign extension generation')
        validate_inheritance(result,parent,questions,result.get('completed',False))
        if result.get('completed'):return result
    else:
        result={k:copy.deepcopy(parent[k]) for k in ('model','E','C','intervention_position')}
        result.update(protocol=PROTOCOL,identity=identity,protocol_fingerprint=qa.fingerprint(identity),bank_protocol_fingerprint=qa.fingerprint(identity),
            control_identity=copy.deepcopy(parent['identity']),parent_protocol=parent['protocol'],pilot=a.pilot,wiring_pilot=a.pilot,
            split='tune',completed=False,conditions=identity['conditions'],condition_specs=identity['condition_specs'],evaluations=[],
            all_controls_freshly_generated=False,all_controls_inherited_exactly=True,pilot_control_reproduction=[])
    bundle=qa.open_bundle(s['meta'],s['prepared']) if bundle is None else bundle;qa.verify_bundle(bundle,s['meta'],s['prepared'])
    versions=[(p,p._version,p.requires_grad) for p in bundle.model.parameters()]
    vectors=(pair_entry.tensor_fingerprint(s['adapter']['W']),pair_entry.tensor_fingerprint(s['C']))
    with fp32_output_projection(bundle):
        for i,question in enumerate(questions):
            old=parent['evaluations'][i]
            if i==len(result['evaluations']):result['evaluations'].append(copy.deepcopy(old))
            row=result['evaluations'][i]
            base,source=source_eval.prompt_tokens(bundle,s['adapter']['config'],question['question'])
            gate_text=bundle.build_chat(identity['gate_system'],identity['gate_user'].format(question=question['question']))
            gate_ids=bundle.encode([gate_text])['input_ids'][0].tolist();labels=joint.gate_map(bundle,gate_text,s['cmeta']['config'])
            if row['prompt_token_ids']!={'base':base,'source':source} or row['gate_prompt_ids']!=gate_ids:
                raise ValueError('parent prompt tokens changed')
            if a.pilot and not any(x['pair_id']==row['pair_id'] for x in result['pilot_control_reproduction']):
                witnesses={}
                for control,enabled in [('baseline',False),('E_source',True)]:
                    value=operation.generate(bundle,base,source,max_new_tokens=48,e_layer=15 if enabled else None,e_direction=s['adapter']['W'] if enabled else None)
                    if (value['token_ids']!=old['generation_token_ids'][control] or value['output']!=old['outputs'][control]
                        or value['finished']!=old['finished'][control] or value['trace']!=old['trace'][control]):
                        raise RuntimeError('zero-C baseline/E fails exact parent reproduction')
                    witnesses[control]=value
                result['pilot_control_reproduction'].append({'pair_id':row['pair_id'],'zero_C_exact_parent':True,'regenerated':witnesses})
            for c in new_conditions():
                name=c['name']
                if name in row['outputs']:continue
                kw=parameters(c,s);value=operation.generate(bundle,base,source,max_new_tokens=48,**kw)
                decoder=operation.SourceCommitmentDecoder(bundle,gate_ids,source,**kw);logits,trace=decoder.next_logits()
                row['gate_readout'][name]={'yes_id':labels['yes'],'no_id':labels['no'],'yes_minus_no':float(logits[labels['yes']]-logits[labels['no']]),
                    'p_yes':float(logits.softmax(-1)[labels['yes']]),'p_no':float(logits.softmax(-1)[labels['no']]),'trace':trace}
                row['outputs'][name]=value['output'];row['finished'][name]=value['finished'];row['rule_abstain'][name]=bool(rule_abstains(value['output']))
                row['generation_token_ids'][name]=value['token_ids'];row['trace'][name]=value['trace']
                if a.pilot:print('E40 PILOT',row['pair_id'],name,'finished',value['finished'],repr(value['output']),flush=True)
            if a.pilot or (i+1)%8==0:
                qa.save(path,result);print('E40 GENERATED',i+1,len(questions),flush=True)
    if any(p._version!=v or p.requires_grad!=g for p,v,g in versions):raise RuntimeError('extension mutated model parameters')
    if vectors!=(pair_entry.tensor_fingerprint(s['adapter']['W']),pair_entry.tensor_fingerprint(s['C'])):raise RuntimeError('extension mutated E/C directions')
    if implementation_identity()!=identity['implementation'] or plan(a,state(a))!=identity:raise RuntimeError('extension source/grade/code changed while executing')
    validate_inheritance(result,parent,questions,True)
    result.update(completed=True,base_parameters_unchanged=True,directions_unchanged=True)
    qa.save(path,result);return result


def judge(a,s):
    identity=frozen(a,s);stem='check' if a.pilot else 'tune';raw=qa.read(a.run_dir/(stem+'.json'))
    questions=s['prepared']['splits']['tune'][:3] if a.pilot else s['prepared']['splits']['tune']
    if not raw.get('completed') or raw['identity']!=identity:raise ValueError('complete exact extension before grading')
    validate_inheritance(raw,s['parent'],questions,True)
    grades=s['graded']['evaluations'][:len(questions)]
    seeded=seed_control_cache(source_dir=a.controls_judge_cache,run_dir=a.run_dir,judge_config=s['judge'],
        scored_rows=grades,conditions=s['parent']['conditions'],scored_sha256=identity['parent_scored_sha256'],stem=stem)
    for key,value in seeded.items():setattr(a,key,value)
    a.model=MODEL;a.judge_exp=JUDGE
    qa.save(a.run_dir/(stem+'_grade_inheritance.json'),{'parent_scored_sha256':identity['parent_scored_sha256'],
        'frozen_parent_conditions':s['parent']['conditions'],'frozen_parent_grade_count':10*len(grades),'new_conditions':[c['name'] for c in new_conditions()],
        'uniform_judge_config_sha256':qa.fingerprint(s['judge'])})
    qa.judge(a,{**s['meta'],'protocol_fingerprint':raw['bank_protocol_fingerprint']},stem+'.json',stem+'_scored')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','generate','judge'))
    for name in ('source-run','e-run','c-run','parent-run','parent-grades','controls-judge-cache','e39-run','run-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pilot',action='store_true');a=p.parse_args();s=state(a)
    if a.phase=='prepare':frozen(a,s,True)
    elif a.phase=='generate':generate(a,s)
    else:judge(a,s)

if __name__=='__main__':main()
