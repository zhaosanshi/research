#!/usr/bin/env python3
"""E39: one predeclared neighboring C site, 92 E34 training questions only.
No answer generation, correctness labels, E writes, layer search or tuning phase.
"""
import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import e26_matched_coverage as joint
from core.experiment_provenance import implementation_identity, require_unchanged
from core.joint_steering import fingerprint, unit
from core.qa_source_commitment import SourceCommitmentDecoder, relative_write_profile

PROTOCOL='e39_neighbor_c_train_function_v1'
FIXED={'llama3.1-8b':{'e_layer':15,'c_layer':14,'rho':.018418746069073677},
       'qwen3-8b':{'e_layer':17,'c_layer':16,'rho':.010587390512228012}}
NAMES={'baseline':None,'C_minus':-1.,'C_plus':1.}


def sources(a):
    meta,bank,data=joint.load_run(SimpleNamespace(run_dir=a.c_run,model=a.model))
    cal=joint.read(a.e37_calibration_dir/'c_calibration.json');identity=cal['identity']
    adapter=joint.read(a.e_run/'preference_adapter.json');fixed=FIXED[a.model]
    if (not cal.get('completed') or meta['pilot'] is not None
        or identity['model']!=a.model or adapter['model']!=a.model or adapter.get('pilot')
        or adapter['protocol']!='supervised_source_pair_preference_v1' or adapter['layer']!=fixed['e_layer']
        or adapter['model_config']!=meta['model_config'] or identity['model_config']!=meta['model_config']
        or identity['source']!=adapter['source'] or identity['config']!=adapter['config']
        or identity['adapter_sha256']!=joint.digest(a.e_run/'preference_adapter.pt')
        or adapter['checkpoint_sha256']!=identity['adapter_sha256']
        or identity['c_bank_meta_sha256']!=joint.digest(a.c_run/'bank.json')
        or identity['c_bank_sha256']!=joint.digest(a.c_run/'bank.npz')
        or cal['states_sha256']!=joint.digest(a.e37_calibration_dir/'unmodified_prompt_states.pt')
        or cal['calibration']['rho']!=fixed['rho']):
        raise ValueError('foreign/changed E34, corrected C bank, or frozen E37 target')
    rows=identity['questions'];expected={str(r['pair_id']):r['question'] for r in data['qa']['train']}
    if (len(rows)!=92 or len({r['pair_id'] for r in rows})!=92
        or [r['pair_id'] for r in rows]!=adapter['training_pair_ids']
        or any(set(r)!={'pair_id','question'} or expected.get(str(r['pair_id']))!=r['question'] for r in rows)
        or {str(r['pair_id']) for r in rows}&{str(r['pair_id']) for r in data['qa']['tune']}):
        raise ValueError('requires exactly 92 frozen E34 training questions, no labels or tuning rows')
    key=f'qa_das_C_L{fixed["c_layer"]}';stat=meta['c_estimation']['qa']['methods']['das'][str(fixed['c_layer'])]
    if not stat['available'] or stat['n_estimation']!=96 or float(bank[key+'_scale'])!=stat['scale']:
        raise ValueError('neighboring C is unavailable or not from the existing full bank')
    direction=torch.tensor(unit(bank[key]),dtype=torch.float32)
    return meta,bank,data,cal,adapter,rows,direction


def source_hashes(a):
    return {f'{label}/{n}':joint.digest(folder/n) for label,folder,names in [
        ('e34',a.e_run,('preference_adapter.json','preference_adapter.pt')),
        ('C',a.c_run,('bank.json','bank.npz','data.json')),
        ('e37_calibration',a.e37_calibration_dir,('c_calibration.json','unmodified_prompt_states.pt'))] for n in names}


def make_plan(a):
    meta,bank,data,cal,adapter,rows,direction=sources(a);fixed=FIXED[a.model]
    result={'protocol':PROTOCOL,'model':a.model,'split':'E34_train92_only','questions':rows,
        'e_layer_reference_only':fixed['e_layer'],'c_layer':fixed['c_layer'],'rho':fixed['rho'],
        'C_original_scale':float(bank[f'qa_das_C_L{fixed["c_layer"]}_scale']),
        'C_direction_sha256':fingerprint(direction.tolist()),'C_bank_fingerprint':meta['fingerprint'],
        'source_sha256':source_hashes(a),'implementation':implementation_identity(__file__),
        'answer_prompt_config':{k:adapter['config'][k] for k in ('qa_system','qa_user')},
        'gate_system':meta['config']['qa_gate_system'],'gate_user':meta['config']['qa_gate_user'],
        'batch_size':1,'alpha':[-1.,1.],'coefficient_bracket':'[0, original_sigma]; fail if frozen rho is not bracketed; never lower rho','bisection_iterations':48,'relative_tolerance':.005,'absolute_tolerance':1e-7,
        'qualification':{'minimum_sign_fraction_each':.9,'minimum_median_absolute_yes_no_change_each':.1},
        'uses_correctness_labels':False,'generates_answers':False,'E_intervention_enabled':False,
        'claims_limit':'One neighboring layer, training-only function diagnostic; no held-out effectiveness or layer-grid claim.'}
    result['fingerprint']=fingerprint(result);return result


def prepare(a):
    plan=make_plan(a);a.run_dir.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    for rel,h in plan['implementation']['files'].items():
        dest=a.run_dir/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and joint.digest(dest)!=h:raise ValueError('existing snapshot differs')
        shutil.copyfile(root/rel,dest)
    joint.save(a.run_dir/'plan.json',plan,immutable=True)
    print('E39 PREPARED',a.model,'C',plan['c_layer'],'92 training questions',flush=True)
    return plan


def frozen(a):
    plan=joint.read(a.run_dir/'plan.json')
    if plan!=make_plan(a):raise ValueError('prepared E39 protocol/source/code changed')
    for rel,h in plan['implementation']['files'].items():
        if joint.digest(a.run_dir/'source_snapshot'/rel)!=h:raise ValueError('saved code snapshot changed')
    return plan


def match_fixed_target(states,direction,original_scale,rho):
    if states.dtype!=torch.bfloat16 or not math.isfinite(rho) or rho<=0:
        raise ValueError('requires original BF16 states and a finite positive fixed target')
    original=relative_write_profile(states,direction,original_scale)
    if original['median']<rho:raise ValueError('original dose cannot bracket the frozen target; do not change target or search layers')
    low,high=0.,original_scale;best=(abs(original['median']-rho),high,original);search=[]
    for _ in range(48):
        mid=(low+high)/2;profile=relative_write_profile(states,direction,mid);error=abs(profile['median']-rho)
        search.append({'coefficient':mid,'median':profile['median']})
        if error<best[0]:best=(error,mid,profile)
        if profile['median']<rho:low=mid
        else:high=mid
    tolerance=.005*rho+1e-7
    if best[0]>tolerance:raise RuntimeError('fixed target cannot be matched within frozen tolerance')
    return {'rho':rho,'coefficient':best[1],'original_scale':original_scale,'original_profile':original,
        'matched_profile':best[2],'absolute_error':best[0],'tolerance':tolerance,'search':search}


def calibrate(a,bundle=None):
    plan=frozen(a);meta,_,_,old,adapter,rows,direction=sources(a)
    if (a.run_dir/'calibration.json').exists() or (a.run_dir/'native_states.pt').exists():raise FileExistsError('calibration is immutable')
    bundle=joint.open_bundle(meta) if bundle is None else bundle
    versions=[(p,p._version,p.requires_grad) for p in bundle.model.parameters()]
    captured=[];tokens=[];prompts=[]
    with torch.no_grad(),joint.fp32_output_projection(bundle):
        for i,r in enumerate(rows):
            text=bundle.build_chat(plan['answer_prompt_config']['qa_system'],plan['answer_prompt_config']['qa_user'].format(question=r['question']))
            inputs=bundle.encode([text]);ids=inputs['input_ids'][0].tolist()
            if ids!=old['prompt_token_ids'][i]:raise ValueError('answer prompt tokenization changed from E37 calibration')
            storage=[]
            def hook(module,inp,out):
                h=out[0] if isinstance(out,tuple) else out;storage.append(h[0,-1].detach().clone())
            with ExitStack() as stack:
                handle=bundle.layer(plan['c_layer']).register_forward_hook(hook);stack.callback(handle.remove)
                bundle.model(**inputs,use_cache=False)
            if len(storage)!=1:raise RuntimeError('neighbor layer must be captured exactly once')
            captured.append(storage[0]);tokens.append(ids);prompts.append(text)
        states=torch.stack(captured);fit=match_fixed_target(states,direction.to(states.device),plan['C_original_scale'],plan['rho'])
    if any(p._version!=v or p.requires_grad!=g for p,v,g in versions):raise RuntimeError('calibration mutated model parameters')
    require_unchanged(plan['implementation'],__file__)
    torch.save(states.cpu(),a.run_dir/'native_states.pt')
    result={'protocol':PROTOCOL,'model':a.model,'plan_sha256':joint.digest(a.run_dir/'plan.json'),
        'completed':True,'states_sha256':joint.digest(a.run_dir/'native_states.pt'),'states_dtype':str(states.dtype),
        'state_shape':list(states.shape),'input_token_ids':tokens,'prompts':prompts,'fit':fit,'base_parameters_unchanged':True}
    joint.save(a.run_dir/'calibration.json',result,immutable=True)
    print('E39 CALIBRATED',a.model,'coef',fit['coefficient'],'rho',fit['rho'],'matched',fit['matched_profile']['median'],flush=True)
    return result


def load_fit(a,plan):
    fit=joint.read(a.run_dir/'calibration.json')
    if (not fit['completed'] or fit['protocol']!=PROTOCOL or fit['model']!=a.model
        or fit['plan_sha256']!=joint.digest(a.run_dir/'plan.json')
        or fit['states_sha256']!=joint.digest(a.run_dir/'native_states.pt')
        or fit['fit']['rho']!=plan['rho'] or fit['fit']['absolute_error']>fit['fit']['tolerance']):raise ValueError('calibration changed')
    _,_,_,_,_,_,direction=sources(a)
    states=torch.load(a.run_dir/'native_states.pt',map_location='cpu',weights_only=True)
    if not isinstance(states,torch.Tensor) or states.dtype!=torch.bfloat16 or states.shape!=(92,len(direction)):
        raise ValueError('native calibration states malformed')
    coefficient=fit['fit']['coefficient']
    if not math.isfinite(coefficient) or not 0<coefficient<=plan['C_original_scale']:
        raise ValueError('invalid frozen coefficient')
    profile=relative_write_profile(states,direction,coefficient)
    if abs(profile['median']-plan['rho'])>.005*plan['rho']+1e-7:
        raise ValueError('coefficient no longer matches original BF16 state target')
    return fit


def gate(a,check=False,bundle=None):
    plan=frozen(a);fit=load_fit(a,plan);meta,_,_,_,_,questions,direction=sources(a)
    bundle=joint.open_bundle(meta) if bundle is None else bundle
    versions=[(p,p._version,p.requires_grad) for p in bundle.model.parameters()]
    if check:questions=questions[:3]
    else:
        probe=joint.read(a.run_dir/'check.json')
        if not probe.get('completed') or not probe.get('check_first3') or probe.get('plan_sha256')!=joint.digest(a.run_dir/'plan.json') or probe.get('calibration_sha256')!=joint.digest(a.run_dir/'calibration.json'):
            raise ValueError('complete frozen first-three check before full training diagnostic')
        summarize(probe,plan)
    path=a.run_dir/('check.json' if check else 'train_gate.json')
    identity={'protocol':PROTOCOL,'model':a.model,'plan_sha256':joint.digest(a.run_dir/'plan.json'),
        'calibration_sha256':joint.digest(a.run_dir/'calibration.json'),'check_first3':check,'question_ids':[r['pair_id'] for r in questions]}
    raw=joint.read(path) if path.exists() else {**identity,'completed':False,'rows':[]}
    if any(raw.get(k)!=v for k,v in identity.items()) or [r['pair_id'] for r in raw['rows']]!=identity['question_ids'][:len(raw['rows'])]:raise ValueError('foreign gate resume')
    with torch.no_grad(),joint.fp32_output_projection(bundle):
        for r in questions[len(raw['rows']):]:
            text=bundle.build_chat(plan['gate_system'],plan['gate_user'].format(question=r['question']))
            ids=bundle.encode([text])['input_ids'][0].tolist();labels=joint.gate_map(bundle,text,meta['config'])
            row={**r,'prompt':text,'input_ids':ids,'gate_labels':labels,'conditions':{}}
            for name,alpha in NAMES.items():
                kw={} if alpha is None else {'c_layer':plan['c_layer'],'c_direction':direction,'c_scale':fit['fit']['coefficient'],'c_alpha':alpha}
                decoder=SourceCommitmentDecoder(bundle,ids,ids,**kw)
                logits,trace=decoder.next_logits();probs=logits.float().softmax(-1);top=int(logits.argmax())
                margin=float(logits[labels['yes']]-logits[labels['no']])
                if not math.isfinite(margin):raise ValueError('nonfinite gate score')
                row['conditions'][name]={'yes_minus_no':margin,'p_yes':float(probs[labels['yes']]),'p_no':float(probs[labels['no']]),
                    'top1_id':top,'top1_text':bundle.tokenizer.decode([top]),'top1_is_yes_no':top in (labels['yes'],labels['no']), 'trace':trace}
            if not check and len(raw['rows'])<3 and row!=probe['rows'][len(raw['rows'])]:
                raise RuntimeError('first-three prompt/logit/trace changed during full diagnostic')
            raw['rows'].append(row);joint.save(path,raw)
            if check:print('E39 CHECK',a.model,r['pair_id'],repr(r['question']),{k:{f:v[f] for f in ('yes_minus_no','top1_text','p_yes','p_no')} for k,v in row['conditions'].items()},flush=True)
    if any(p._version!=v or p.requires_grad!=g for p,v,g in versions):raise RuntimeError('gate mutated model parameters')
    require_unchanged(plan['implementation'],__file__)
    raw['completed']=True;joint.save(path,raw);return raw


def summarize(raw,plan):
    expected=plan['questions'][:3] if raw['check_first3'] else plan['questions']
    if not raw['completed'] or len(raw['rows'])!=len(expected):raise ValueError('incomplete gate cohort')
    for r,q in zip(raw['rows'],expected):
        if any(r[k]!=v for k,v in q.items()) or set(r['conditions'])!=set(NAMES):raise ValueError('gate identity differs')
        for name,alpha in NAMES.items():
            v=r['conditions'][name];t=v['trace']
            if t['source_forward_calls']!=0 or len(t['c_writes'])!=(0 if alpha is None else 1):raise ValueError('unexpected intervention or source stream')
            if alpha is not None and (t['c_writes'][0]['layer']!=plan['c_layer'] or t['c_writes'][0]['alpha']!=alpha):raise ValueError('wrong C write')
            if not math.isfinite(v['yes_minus_no']) or not 0<=v['p_yes']<=1 or not 0<=v['p_no']<=1 or v['p_yes']+v['p_no']>1+1e-6:raise ValueError('invalid gate probabilities')
    result={}
    for name,sign in [('C_minus',-1),('C_plus',1)]:
        delta=np.array([r['conditions'][name]['yes_minus_no']-r['conditions']['baseline']['yes_minus_no'] for r in raw['rows']]);count=int((sign*delta>0).sum());n=len(delta)
        result[name]={'n':n,'correct_sign_count':count,'opposite_sign_count':int((sign*delta<0).sum()),'zero_count':int((delta==0).sum()),
            'correct_sign_fraction':count/n,'median_absolute_change':float(np.median(abs(delta))),'median_signed_change':float(np.median(sign*delta)),
            'mean_change':float(delta.mean()),'passes_sign':count/n>=.9,'passes_magnitude':float(np.median(abs(delta)))>=.1,
            'actual_delta_over_base':{k:float(fun([r['conditions'][name]['trace']['c_writes'][0]['stored_delta_over_base'] for r in raw['rows']])) for k,fun in [('min',np.min),('median',np.median),('max',np.max)]}}
    result['format']={name:{'top1_yes_no_count':sum(r['conditions'][name]['top1_is_yes_no'] for r in raw['rows']),
        'yes_no_mass_min':min(r['conditions'][name]['p_yes']+r['conditions'][name]['p_no'] for r in raw['rows']),
        'yes_no_mass_median':float(np.median([r['conditions'][name]['p_yes']+r['conditions'][name]['p_no'] for r in raw['rows']]))} for name in NAMES}
    result['qualifies_training_function']=all(result[name]['passes_sign'] and result[name]['passes_magnitude'] for name in ('C_minus','C_plus'))
    return result


def report(a):
    plan=frozen(a);fit=load_fit(a,plan);raw=joint.read(a.run_dir/'train_gate.json')
    if raw.get('protocol')!=PROTOCOL or raw.get('model')!=a.model or raw['plan_sha256']!=joint.digest(a.run_dir/'plan.json') or raw['calibration_sha256']!=joint.digest(a.run_dir/'calibration.json') or raw['check_first3']:raise ValueError('foreign or partial report')
    result={'protocol':PROTOCOL,'model':a.model,'split':'E34_train92_only','c_layer':plan['c_layer'],'rho':plan['rho'],
        'plan_sha256':joint.digest(a.run_dir/'plan.json'),'raw_sha256':joint.digest(a.run_dir/'train_gate.json'),
        'coefficient':fit['fit']['coefficient'],'criteria':plan['qualification'],'summary':summarize(raw,plan),
        'interpretation':'Training-only function qualification, not evidence of answer correctness or E+C gains.'}
    joint.save(a.run_dir/'report.json',result);print('E39 REPORT',a.model,result['summary'],flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','calibrate','check','gate','report'])
    p.add_argument('--model',required=True,choices=tuple(FIXED))
    for key in ('e-run','c-run','e37-calibration-dir','run-dir'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();{'prepare':prepare,'calibrate':calibrate,'check':lambda a:gate(a,True),'gate':gate,'report':report}[a.phase](a)

if __name__=='__main__':main()
