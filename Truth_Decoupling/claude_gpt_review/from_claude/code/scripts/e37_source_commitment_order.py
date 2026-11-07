#!/usr/bin/env python3
"""R21 fixed source-E / prompt-only-C depth-order diagnostic, no training/grid.

Ten freshly generated arms. E is the fixed E34 paired-preference checkpoint;
C comes from the corrected QA ability bank at predeclared 25%/75% depths.
Every arm uses the same model-specific budget (Llama48, Qwen256) and current
semantic/refusal policy. Existing raw answers are checked, never inherited.
"""
import argparse
import copy
from pathlib import Path
import shutil
from types import SimpleNamespace
import contextlib

import torch
import common
import e25_qa_das as qa
import e26_matched_coverage as joint
import e27_qa_generation as legacy
import e29_qa_source_interchange as source_eval
import e34_qa_source_pair_preference as pair_entry
from e35_qa_gate_prompt_diagnostic import ABILITY_SYSTEM, ABILITY_USER
from core import qa_source_pair_preference as preference
from core import qa_source_commitment as operation
from core.joint_steering import unit
from core.qa_generation_adapter import implementation_identity
from core.qa_precision import fp32_output_projection
from core.readout import rule_abstains
from core.qa_reference_grades import reference_index,verify_matching_grades
from core.qa_control_cache import seed_control_cache


PROTOCOL = operation.PROTOCOL
MODEL_PLAN = {'llama3.1-8b': {'e_layer':15,'before':7,'after':23,'budget':48},
              'qwen3-8b': {'e_layer':17,'before':8,'after':26,'budget':256}}
JUDGE = 'E4_qa'


def conditions(model):
    p = MODEL_PLAN[model]
    out = [{'name':'baseline','kind':'base','e_enabled':False,'e':None,'c':None,'uses_reference_answer':False},
           {'name':'E_source','kind':'E','e_enabled':True,'e':{'layer':p['e_layer'],'alpha':1.},'c':None,'uses_reference_answer':False}]
    for side in ('before','after'):
        for suffix,alpha in (('m1',-1.),('p1',1.)):
            for enabled in (False,True):
                kind = 'EC' if enabled else 'C'
                out.append({'name':kind+'_'+side+'_'+suffix,'kind':kind,'e_enabled':enabled,
                    'e':{'layer':p['e_layer'],'alpha':1.} if enabled else None,
                    'c':{'layer':p[side],'side':side,'alpha':alpha,'position':'base_prompt_end_only'},
                    'uses_reference_answer':False})
    return out


def state(args):
    ns,meta,_,prepared = legacy.source_run(args.source_run)
    if meta['model'] not in MODEL_PLAN: raise ValueError('only the two fixed model plans are registered')
    p = MODEL_PLAN[meta['model']]
    adapter = preference.load(args.e_run)
    pair_entry.verify_snapshot(args.e_run/'training_code_snapshot',adapter['entry_implementation'])
    cfg = {k:meta['config'][k] for k in ('qa_system','qa_user','max_new_tokens','batch_size')}
    if (adapter.get('pilot') or adapter['model'] != meta['model'] or adapter['model_config'] != meta['model_config']
            or adapter['layer'] != p['e_layer'] or adapter['source'] != legacy.source_identity(ns,meta,[p['e_layer']])
            or adapter['config'] != cfg or adapter['source_system'] != source_eval.SOURCE_SYSTEM
            or len(adapter['training_pair_ids']) != 92 or adapter['train_config']['epochs'] != 3):
        raise ValueError('E is not the fixed full E34 checkpoint/source')
    cmeta,bank,cdata = joint.load_run(SimpleNamespace(run_dir=args.c_run,model=meta['model']))
    if (cmeta.get('pilot') or cmeta['model_config'] != meta['model_config']
            or cmeta['config'].get('qa_gate_system') != ABILITY_SYSTEM
            or cmeta['config'].get('qa_gate_user') != ABILITY_USER
            or any(cmeta['qa_config'][k] != cfg[k] for k in ('qa_system','qa_user'))
            or cmeta.get('generation_adapters')):
        raise ValueError('C requires the corrected full QA ability bank with original prompt/model identity')
    questions = prepared['splits']['tune']
    if len(questions) != 192 or len(cdata['qa']['tune']) != 192:
        raise ValueError('the fixed diagnostic cohort is all 192 tuning questions')
    cmap = {r['pair_id']:r['question'] for r in cdata['qa']['tune']}
    if cmap != {r['pair_id']:r['question'] for r in questions}:
        raise ValueError('C and E sources have different QA tuning cohorts')
    cvalues = {}
    for side in ('before','after'):
        layer = p[side]; key = 'qa_das_C_L%d'%layer
        stat = cmeta['c_estimation']['qa']['methods']['das'][str(layer)]
        if not stat['available'] or stat['n_estimation'] != len(cdata['qa']['train']):
            raise ValueError('C direction was not estimated on the full fixed estimation cohort')
        w = torch.as_tensor(unit(bank[key]),dtype=torch.float32)
        scale = float(bank[key+'_scale'])
        if abs(scale-stat['scale']) > 1e-10 or not scale > 0:
            raise ValueError('C bank and metadata scale differ')
        cvalues[side] = {'direction':w,'scale':scale,'stat':stat}
    reference = qa.read(args.reference_run/'tune.json')
    if (not reference.get('completed') or reference.get('pilot') or reference['model'] != meta['model']
            or reference['identity']['source'] != adapter['source']
            or reference['identity']['layer'] != adapter['layer']
            or reference['identity']['config'] != cfg
            or reference['identity']['adapter_sha256'] != qa.digest(args.e_run/'preference_adapter.pt')
            or reference['identity'].get('budget_override',{}).get('max_new_tokens',cfg['max_new_tokens']) != p['budget']
            or not {'baseline',preference.METHOD}.issubset(reference['conditions'])):
        raise ValueError('reference baseline/E must use this exact checkpoint/source and model-specific budget')
    qa.verify_question_rows(reference['evaluations'],questions)
    judge = legacy.validate_judge_experiment(JUDGE)
    if judge.get('full_personal_refusal_guard') is not True:
        raise ValueError('all ten arms require the corrected uniform refusal policy')
    reference_index(reference,qa.read(args.reference_run/'tune_scored_items.json'),
        raw_sha256=qa.digest(args.reference_run/'tune.json'),
        reference_config=legacy.validate_judge_experiment(pair_entry.LEGACY_JUDGE),target_config=judge)
    return ns,meta,prepared,adapter,cmeta,cvalues,reference,judge


def calibration_identity(args,s):
    _,meta,prepared,adapter,_,cvalues,_,_ = s
    rows=[{'pair_id':r['pair_id'],'question':r['question']} for r in prepared['e_train']]
    if len(rows)!=92 or [r['pair_id'] for r in rows]!=adapter['training_pair_ids']:
        raise ValueError('C calibration must use exactly the same fixed 92 E training questions')
    return {'protocol':'train_prompt_actual_relative_C_norm_v1','model':meta['model'],
        'model_config':meta['model_config'],'source':adapter['source'],
        'adapter_sha256':qa.digest(args.e_run/'preference_adapter.pt'),
        'c_bank_meta_sha256':qa.digest(args.c_run/'bank.json'),'c_bank_sha256':qa.digest(args.c_run/'bank.npz'),
        'config':adapter['config'],'batch_size':1,'questions':rows,
        'c_layers':{side:MODEL_PLAN[meta['model']][side] for side in ('before','after')},
        'original_scales':{side:v['scale'] for side,v in cvalues.items()},
        'C_vectors':{side:pair_entry.tensor_fingerprint(v['direction']) for side,v in cvalues.items()},
        'uses_answer_generation':False,'uses_semantic_labels':False,'E_intervention_enabled':False,
        'implementation':implementation_identity()}


def load_calibration(args,s):
    if args.original_sigma:return None
    if args.calibration_dir is None:raise ValueError('frozen calibration directory required')
    value=qa.read(args.calibration_dir/'c_calibration.json')
    if (not value.get('completed') or value.get('identity')!=calibration_identity(args,s)
            or value.get('states_sha256')!=qa.digest(args.calibration_dir/'unmodified_prompt_states.pt')
            or len(value.get('prompt_token_ids',[]))!=92
            or value['calibration'].get('protocol')!='train_prompt_actual_relative_C_norm_v1'):
        raise ValueError('C calibration identity/actual-state file differs')
    for side in ('before','after'):
        point=value['calibration']['layers'][side]
        if (not 0<point['coefficient']<=s[5][side]['scale']
                or point['absolute_median_error']>value['calibration']['tolerance']):
            raise ValueError('invalid calibrated C coefficient or matching tolerance')
    return value


def calibrate(args,s,bundle=None):
    _,meta,prepared,adapter,_,cvalues,_,_=s
    identity=calibration_identity(args,s);args.run_dir.mkdir(parents=True,exist_ok=True)
    path=args.run_dir/'c_calibration.json'
    if path.exists():
        old=qa.read(path)
        if old.get('identity')!=identity or not old.get('completed'):raise ValueError('existing calibration differs or is incomplete')
        return old
    pair_entry.snapshot(args.run_dir/'code_snapshot',identity['implementation'])
    qa.save(args.run_dir/'calibration_plan.json',identity)
    bundle=qa.open_bundle(meta,prepared) if bundle is None else bundle;qa.verify_bundle(bundle,meta,prepared)
    versions=[(p,p._version,p.requires_grad) for p in bundle.model.parameters()]
    states={side:[] for side in ('before','after')};token_ids=[]
    with fp32_output_projection(bundle),torch.no_grad():
        for index,row in enumerate(identity['questions']):
            base,_=source_eval.prompt_tokens(bundle,adapter['config'],row['question']);token_ids.append(base)
            captured={}
            def make_hook(side):
                def hook(module,inputs,output):
                    if side in captured:raise RuntimeError('calibration layer called twice')
                    h=output[0] if isinstance(output,tuple) else output
                    captured[side]=h[0,-1].detach().cpu().clone()
                return hook
            with contextlib.ExitStack() as stack:
                for side,layer in identity['c_layers'].items():
                    handle=bundle.layer(layer).register_forward_hook(make_hook(side));stack.callback(handle.remove)
                inputs=torch.tensor([base],device=bundle.device,dtype=torch.long)
                bundle.model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False)
            if set(captured)!=set(states):raise RuntimeError('C calibration missed a requested layer')
            for side in states:states[side].append(captured[side])
            if (index+1)%16==0:print('E37 CALIBRATION CAPTURED',index+1,92,flush=True)
        states={side:torch.stack(rows) for side,rows in states.items()}
        result=operation.calibrate_coefficients({k:v.to(bundle.device) for k,v in states.items()},
                   {k:v['direction'] for k,v in cvalues.items()},identity['original_scales'])
    if any(p._version!=v or p.requires_grad!=r for p,v,r in versions):raise RuntimeError('calibration changed model weights')
    if implementation_identity()!=identity['implementation']:raise RuntimeError('calibration source code changed')
    torch.save(states,args.run_dir/'unmodified_prompt_states.pt')
    value={'identity':identity,'completed':True,'prompt_token_ids':token_ids,'calibration':result,
           'states_sha256':qa.digest(args.run_dir/'unmodified_prompt_states.pt'),
           'state_shapes':{k:list(v.shape) for k,v in states.items()},'state_dtypes':{k:str(v.dtype) for k,v in states.items()},
           'base_parameters_unchanged':True}
    qa.save(path,value)
    print('E37 CALIBRATED',meta['model'],'rho',result['rho'],
          {k:{'coefficient':v['coefficient'],'original_median':v['original_profile']['median'],
              'calibrated_median':v['calibrated_profile']['median']} for k,v in result['layers'].items()},flush=True)
    return value


def plan(args,s):
    ns,meta,prepared,adapter,cmeta,cvalues,reference,judge = s
    p = MODEL_PLAN[meta['model']]; questions = prepared['splits']['tune'][:3] if args.pilot else prepared['splits']['tune']
    calibration = load_calibration(args,s)
    result = {'protocol':PROTOCOL,'model':meta['model'],'model_config':meta['model_config'],
        'source':adapter['source'],'source_system':adapter['source_system'],'layer':p['e_layer'],'e_layer':p['e_layer'],
        'config':adapter['config'],'max_new_tokens':p['budget'],'batch_size':1,'alpha':1.,
        'same_prefix_tokens':True,'gold_used_at_inference':False,'all_arms_freshly_generated':True,
        'source_has_C':False,'training_called':False,'pilot':args.pilot,'split':'tune',
        'adapter_sha256':qa.digest(args.e_run/'preference_adapter.pt'),
        'adapter_manifest_sha256':qa.digest(args.e_run/'preference_adapter.json'),
        'c_bank_meta_sha256':qa.digest(args.c_run/'bank.json'),'c_bank_sha256':qa.digest(args.c_run/'bank.npz'),
        'c_bank_fingerprint':cmeta['fingerprint'],'c_data_sha256':qa.digest(args.c_run/'data.json'),
        'c_layers':{side:p[side] for side in ('before','after')},
        'c_directions':{side:{'layer':p[side],'scale':v['scale'],'vector':pair_entry.tensor_fingerprint(v['direction']),
                             'estimation':v['stat']} for side,v in cvalues.items()},
        'c_alphas':[-1.,1.],'c_position':'base_prompt_end_only',
        'calibration_mode':'original_sigma_diagnostic' if args.original_sigma else 'train_relative_norm_matched',
        'c_calibration_sha256':None if args.original_sigma else qa.digest(args.calibration_dir/'c_calibration.json'),
        'c_coefficients':{side:cvalues[side]['scale'] if args.original_sigma else calibration['calibration']['layers'][side]['coefficient']
                          for side in ('before','after')},
        'c_orthogonality_scope':'existing QA bank projects off its old same-layer DAS E; not the new cross-layer E34',
        'order_scope':'physical depth, C direction and estimation scale jointly vary; not a pure operator permutation',
        'question_ids':[r['pair_id'] for r in questions],
        'conditions':[c['name'] for c in conditions(meta['model'])],
        'condition_specs':conditions(meta['model']),
        'reference_generation_sha256':qa.digest(args.reference_run/'tune.json'),
        'reference_grading_sha256':qa.digest(args.reference_run/'tune_scored_items.json'),
        'reference_grade_policy':'same complete semantic request; all reference arms locked after uniform current refusal guard',
        'reference_use':'verify newly generated baseline/E tokens, never copy original outputs',
        'gate_system':ABILITY_SYSTEM,'gate_user':ABILITY_USER,
        'gate_scope':'diagnostic first Yes-No logits; answers never depend on the gate result',
        'judge_experiment':JUDGE,'judge_config_sha256':qa.fingerprint(judge),
        'implementation':implementation_identity()}
    result['fingerprint'] = qa.fingerprint(result)
    return result


def frozen(args,s,create=False):
    expected = plan(args,s); path = args.run_dir/'plan.json'
    if path.exists():
        if qa.read(path) != expected: raise ValueError('fixed E37 plan/source/code changed')
    elif create:
        args.run_dir.mkdir(parents=True,exist_ok=True)
        pair_entry.snapshot(args.run_dir/'code_snapshot',expected['implementation'])
        qa.save(path,expected)
    else: raise ValueError('prepare E37 before execution')
    return expected


def parameters(condition,adapter,cvalues,coefficients):
    kw = {'e_layer':adapter['layer'] if condition['e_enabled'] else None,
          'e_direction':adapter['W'] if condition['e_enabled'] else None}
    c = condition['c']
    if c:
        v = cvalues[c['side']]
        kw.update(c_layer=c['layer'],c_direction=v['direction'],c_scale=coefficients[c['side']],c_alpha=c['alpha'])
    return kw


def validate_rows(raw,questions,complete=False):
    if complete and len(raw['evaluations'])!=len(questions):raise ValueError('incomplete E37 cohort')
    if len(raw['evaluations'])>len(questions):raise ValueError('unexpected E37 questions')
    qa.verify_question_rows(raw['evaluations'],questions[:len(raw['evaluations'])])
    names = set(raw['conditions'])
    for row in raw['evaluations']:
        for field in ('outputs','finished','rule_abstain','generation_token_ids','trace','gate_readout'):
            if not set(row[field]).issubset(names) or (complete and set(row[field]) != names):
                raise ValueError('E37 row has missing/foreign condition fields')
        for name, tokens in row['generation_token_ids'].items():
            tr = row['trace'][name]
            if not tokens or len(tokens) != len(tr) or any(t['accepted_prefix_ids'] != tokens[:i] or t['next_top1_id'] != tokens[i]
                    for i,t in enumerate(tr)):
                raise ValueError('E37 generation trace/token prefix differs')


def generate(args,s,bundle=None):
    ns,meta,prepared,adapter,cmeta,cvalues,reference,judge = s
    identity = frozen(args,s); questions = prepared['splits']['tune'][:3] if args.pilot else prepared['splits']['tune']
    path = args.run_dir/('check.json' if args.pilot else 'tune.json')
    if path.exists():
        result = qa.read(path)
        if result.get('identity') != identity: raise ValueError('existing E37 run has a different identity')
        validate_rows(result,questions,complete=result.get('completed',False))
        if result.get('completed'): return result
    else:
        result = {'protocol':PROTOCOL,'model':meta['model'],'identity':identity,
            'protocol_fingerprint':qa.fingerprint(identity),'bank_protocol_fingerprint':qa.fingerprint(identity),
            'pilot':args.pilot,'wiring_pilot':args.pilot,'split':'tune','completed':False,
            'conditions':identity['conditions'],'condition_specs':identity['condition_specs'],'evaluations':[],
            'intervention_position':'E: every base predictor, live independent source; C: original base prompt final position only',
            'E':{'method':preference.METHOD,'fixed_checkpoint':True},'C':{'enabled':True,'estimation':'corrected QA ability bank'},
            'all_controls_freshly_generated':True}
    bundle = qa.open_bundle(meta,prepared) if bundle is None else bundle
    qa.verify_bundle(bundle,meta,prepared)
    versions = [(p,p._version,p.requires_grad) for p in bundle.model.parameters()]
    directions = {'E':pair_entry.tensor_fingerprint(adapter['W']),**{k:pair_entry.tensor_fingerprint(v['direction']) for k,v in cvalues.items()}}
    refs = {r['pair_id']:r for r in reference['evaluations']}
    with fp32_output_projection(bundle):
        for i,question in enumerate(questions):
            if i == len(result['evaluations']):
                old = refs[question['pair_id']]
                row = {k:copy.deepcopy(old[k]) for k in ('pair_id','question','gold','incorrect','best_answer') if k in old}
                row.update(idx=i,**{k:{} for k in ('outputs','finished','rule_abstain','generation_token_ids','trace','gate_readout')})
                result['evaluations'].append(row)
            row = result['evaluations'][i]
            base,source = source_eval.prompt_tokens(bundle,adapter['config'],row['question'])
            if refs[row['pair_id']]['prompt_token_ids'] != {'base':base,'source':source}:
                raise ValueError('reference prompt token IDs changed')
            row['prompt_token_ids'] = {'base':base,'source':source}
            gate_text = bundle.build_chat(ABILITY_SYSTEM,ABILITY_USER.format(question=row['question']))
            gate_ids = bundle.encode([gate_text])['input_ids'][0].tolist()
            labels = joint.gate_map(bundle,gate_text,cmeta['config'])
            row['gate_prompt_ids'] = gate_ids
            for condition in identity['condition_specs']:
                name = condition['name']
                if name in row['outputs']: continue
                kw = parameters(condition,adapter,cvalues,identity['c_coefficients'])
                value = operation.generate(bundle,base,source,max_new_tokens=identity['max_new_tokens'],**kw)
                if name in ('baseline','E_source'):
                    original = 'baseline' if name=='baseline' else preference.METHOD
                    old = refs[row['pair_id']]
                    if (value['token_ids'] != old['generation_token_ids'][original] or value['output'] != old['outputs'][original]
                            or value['finished'] != old['finished'][original]):
                        raise RuntimeError('fresh baseline/E fails exact same-budget reference reproduction: '+row['pair_id']+'/'+name)
                decoder = operation.SourceCommitmentDecoder(bundle,gate_ids,source,**kw)
                logits,trace = decoder.next_logits()
                margin = float(logits[labels['yes']]-logits[labels['no']])
                row['gate_readout'][name] = {'yes_id':labels['yes'],'no_id':labels['no'],'yes_minus_no':margin,
                    'p_yes':float(logits.softmax(-1)[labels['yes']]),'p_no':float(logits.softmax(-1)[labels['no']]),'trace':trace}
                row['outputs'][name] = value['output'];row['finished'][name] = value['finished']
                row['rule_abstain'][name] = bool(rule_abstains(value['output']))
                row['generation_token_ids'][name] = value['token_ids'];row['trace'][name] = value['trace']
                if args.pilot: print('E37 PILOT',meta['model'],row['pair_id'],name,'finished',value['finished'],repr(value['output']),flush=True)
            if args.pilot or (i+1)%8 == 0:
                qa.save(path,result);print('E37 COLLECTED',i+1,len(questions),flush=True)
    if any(p._version!=v or p.requires_grad!=r for p,v,r in versions): raise RuntimeError('E37 changed model weights')
    if directions != {'E':pair_entry.tensor_fingerprint(adapter['W']),**{k:pair_entry.tensor_fingerprint(v['direction']) for k,v in cvalues.items()}}:
        raise RuntimeError('E37 changed E/C directions')
    if implementation_identity() != identity['implementation']: raise RuntimeError('E37 code changed during execution')
    validate_rows(result,questions,True)
    result.update(completed=True,base_parameters_unchanged=True,directions_unchanged=True,
                  baseline_E_fresh_exact_reference=True)
    qa.save(path,result);return result


def judge(args,s):
    identity = frozen(args,s);meta=s[1];stem='check' if args.pilot else 'tune'
    raw=qa.read(args.run_dir/(stem+'.json'));questions=s[2]['splits']['tune'][:3] if args.pilot else s[2]['splits']['tune']
    if not raw.get('completed') or raw['identity']!=identity: raise ValueError('complete matching generation required')
    validate_rows(raw,questions,True)
    reference=reference_index(s[6],qa.read(args.reference_run/'tune_scored_items.json'),
        raw_sha256=identity['reference_generation_sha256'],
        reference_config=legacy.validate_judge_experiment(pair_entry.LEGACY_JUDGE),target_config=s[7])
    seeded=seed_control_cache(source_dir=args.reference_run/'judge_cache',run_dir=args.run_dir,judge_config=s[7],
        scored_rows=reference['policy_converted_rows'],conditions=s[6]['conditions'],
        scored_sha256=identity['reference_grading_sha256'],stem=stem)
    # Arm names differ from the parent. Validate by the complete semantic
    # request below, including C/EC outputs matching ANY of the seven parents.
    args.judge_cache_dir=seeded['judge_cache_dir'];args.judge_exp=JUDGE;args.model=meta['model']
    qa.judge(args,{**meta,'protocol_fingerprint':raw['bank_protocol_fingerprint']},stem+'.json',stem+'_scored')
    verification=verify_matching_grades(raw,qa.read(args.run_dir/(stem+'_scored_items.json')),reference,
        raw_sha256=qa.digest(args.run_dir/(stem+'.json')),target_config=s[7])
    verification.update(reference_generation_sha256=identity['reference_generation_sha256'],
                        reference_grading_sha256=identity['reference_grading_sha256'])
    qa.save(args.run_dir/(stem+'_reference_grade_verification.json'),verification)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('calibrate','prepare','generate','judge'))
    for name in ('source-run','e-run','c-run','reference-run','run-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    g=p.add_mutually_exclusive_group();g.add_argument('--original-sigma',action='store_true');g.add_argument('--calibration-dir',type=Path)
    args=p.parse_args();s=state(args)
    if args.original_sigma and not args.pilot:p.error('original sigma is only a three-question diagnostic')
    if args.phase!='calibrate' and not args.original_sigma and args.calibration_dir is None:p.error('the main run requires frozen train-only C calibration')
    if args.phase=='calibrate':calibrate(args,s)
    elif args.phase=='prepare':frozen(args,s,True)
    elif args.phase=='generate':generate(args,s)
    else:judge(args,s)


if __name__=='__main__':main()
