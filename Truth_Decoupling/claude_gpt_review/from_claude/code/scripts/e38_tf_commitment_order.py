#!/usr/bin/env python3
"""E38: fixed E with earlier/later C, in BOTH TF answer and ability forwards.

Exploratory tuning diagnosis only. Frozen 50%-ability E; C at 25/75% depth,
its own saved vector/scale, alpha +/-1; matched C-only controls. No test phase.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import e26_matched_coverage as joint
from core.experiment_provenance import implementation_identity, require_unchanged
from core.intervene import steer
from core.joint_steering import fingerprint, intervention_specs, unit
from core.qa_source_commitment import calibrate_coefficients
from core.model_io import forward_logits
from core.readout import build_label_token_map, label_logprobs_from_logits
from core.selective_eval import quota_select

PROTOCOL = 'e38_tf_fixed_e_c_order_v1'
METHODS = ('das','md')
ALPHAS = (-1.,1.)
COVERAGES = (.25,.5,.75)
SOURCE_FILES = ('bank.json','bank.npz','data.json','selection.json','tf_tune.json')
EXPECTED_E = {'llama3.1-8b':{'das':(11,3.),'md':(13,3.)}, 'qwen3-8b':{'das':(16,3.),'md':(21,2.)}}


def source(args):
    ns=SimpleNamespace(run_dir=args.source_run,model=args.model)
    meta,bank,data=joint.load_run(ns)
    if meta['pilot'] is not None or tuple(joint.evaluation_tasks(meta))!=('tf',) or not meta['config'].get('tf_gate_system'):
        raise ValueError('source must be the repaired complete TF-only ability run')
    selection=joint.frozen_selection(ns,meta)
    tune=joint.scored_rows(ns,meta,'tf','tune')[0]
    return meta,bank,data,selection,tune


def conditions(meta,bank,selection):
    specs={c['name']:c for c in meta['conditions']['tf']}
    out=[dict(name='baseline',kind='base',task='tf',method='base',e=None,c=None,source_control='baseline')]
    fixed={};clayers=[math.ceil(meta['num_layers']*f)-1 for f in (.25,.75)]
    for method in METHODS:
        name=selection['tasks']['tf']['choices']['0.5'][method+'_E']
        if name not in specs or specs[name]['kind']!='E' or specs[name]['method']!=method:
            raise ValueError('frozen E choice unavailable or foreign')
        e=deepcopy(specs[name]['e'])
        if (e.get('layer'),e.get('alpha'))!=EXPECTED_E[meta['model']][method] or e.get('mode')!='prefill':
            raise ValueError('predeclared E anchor changed')
        if not clayers[0]<e['layer']<clayers[1]:
            raise ValueError('predeclared C sites must bracket fixed E')
        fixed[method]={'source_condition':name,'e':e}
        out.append(dict(name=method+'_E',kind='E',task='tf',method=method,e=e,c=None,source_control=name))
        for layer in clayers:
            key=f'tf_{method}_C_L{layer}'
            if key not in bank or key+'_scale' not in bank or float(bank[key+'_scale'])<=0:
                raise ValueError('predeclared C vector/scale unavailable; do not replace its layer')
            for alpha in ALPHAS:
                for kind in ('C','EC'):
                    out.append(dict(name=f'{method}_{kind}_L{layer}_a{alpha:g}',kind=kind,task='tf',method=method,
                        e=e if kind=='EC' else None,c={'layer':layer,'alpha':alpha},source_control=None))
    if len(out)!=19 or len({c['name'] for c in out})!=19:
        raise ValueError('expected exactly 19 predeclared conditions')
    return out,fixed,clayers


def clean(row):
    return {'id':str(row['id']),'pair_id':str(row['pair_id']),'text':row['text']}



def calibration_rows(data):
    rows=[clean(r) for r in data['tf']['train']]
    groups=Counter(r['pair_id'] for r in rows)
    if len(rows)!=192 or len(groups)!=96 or set(groups.values())!={2} or set(groups)&{str(r['pair_id']) for r in data['tf']['tune']}:
        raise ValueError('calibration requires all 192 training statements / 96 pairs, disjoint from tuning')
    return rows


def calibrate(args,bundle=None):
    if args.dose_mode!='train_relative' or args.calibration_run is None:
        raise ValueError('calibration phase requires train_relative and an independent calibration-run')
    meta,bank,data,selection,tune=source(args)
    cs,fixed,layers=conditions(meta,bank,selection);rows=calibration_rows(data)
    identity=implementation_identity(__file__)
    bundle=joint.open_bundle(meta) if bundle is None else bundle
    joint.verify_tokens(bundle,meta)
    texts=[joint.tf_prompt(bundle,meta['prompts'],r['text'],allow_idk=False) for r in rows]
    states={side:[] for side in ('before','after')};tokens=[];masks=[]
    for start in range(0,len(texts),meta['config']['batch_size']):
        inputs=bundle.encode(texts[start:start+meta['config']['batch_size']]);captured={};handles=[]
        if not inputs['attention_mask'][:,-1].all():raise ValueError('left padding required')
        try:
            for side,layer in zip(('before','after'),layers):
                def record(module,inp,out,_side=side):
                    if _side in captured:raise RuntimeError('calibration layer called more than once')
                    h=out[0] if isinstance(out,tuple) else out
                    captured[_side]=h[:,-1].detach().clone()
                handles.append(bundle.layer(layer).register_forward_hook(record))
            with torch.no_grad(),joint.fp32_output_projection(bundle):
                forward_logits(bundle,inputs,logits_to_keep=1)
        finally:
            for handle in handles:handle.remove()
        if set(captured)!={'before','after'}:raise RuntimeError('incomplete native calibration capture')
        for side in captured:states[side].append(captured[side])
        tokens.extend(inputs['input_ids'].cpu().tolist());masks.extend(inputs['attention_mask'].cpu().tolist())
    states={side:torch.cat(chunks) for side,chunks in states.items()}
    methods={}
    for method in METHODS:
        directions={side:torch.tensor(unit(bank[f'tf_{method}_C_L{layer}']),dtype=torch.float32,device=states[side].device) for side,layer in zip(('before','after'),layers)}
        scales={side:float(bank[f'tf_{method}_C_L{layer}_scale']) for side,layer in zip(('before','after'),layers)}
        methods[method]=calibrate_coefficients(states,directions,scales)
    dest=args.calibration_run;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'calibration.json').exists() or (dest/'states.pt').exists():
        raise FileExistsError('calibration is immutable; use a new independent directory')
    torch.save({k:v.cpu() for k,v in states.items()},dest/'states.pt')
    result=dict(protocol='e38_train_relative_calibration_v1',model=args.model,bank_fingerprint=meta['fingerprint'],
        source_sha256={n:joint.digest(args.source_run/n) for n in SOURCE_FILES},implementation=identity,
        input_rows=rows,prompts=texts,input_ids=tokens,attention_mask=masks,batch_size=meta['config']['batch_size'],
        c_layers=layers,states_sha256=joint.digest(dest/'states.pt'),states_dtype={k:str(v.dtype) for k,v in states.items()},methods=methods,
        scope='all 192 training statements / 96 pairs; no outcomes/labels in model or calibration; common target per estimator, both signs')
    require_unchanged(identity,__file__);result['fingerprint']=fingerprint(result)
    joint.save(dest/'calibration.json',result,immutable=True)
    print('E38 CALIBRATED',args.model,{m:{'rho':r['rho'],'coefficients':{k:v['coefficient'] for k,v in r['layers'].items()}} for m,r in methods.items()},flush=True)
    return result


def load_calibration(args,meta,bank,data,layers):
    mode=getattr(args,'dose_mode','original_sigma')
    if mode=='original_sigma':
        if getattr(args,'calibration_run',None) is not None:raise ValueError('original dose cannot load calibration')
        if args.limit is None:raise ValueError('original sigma is retained only for the three-pair diagnostic')
        return None
    if mode!='train_relative' or args.calibration_run is None:raise ValueError('missing train-only calibration')
    p=args.calibration_run;result=joint.read(p/'calibration.json')
    if (result.get('protocol')!='e38_train_relative_calibration_v1' or result.get('model')!=args.model
        or result.get('fingerprint')!=fingerprint({k:v for k,v in result.items() if k!='fingerprint'})
        or result.get('bank_fingerprint')!=meta['fingerprint'] or result.get('c_layers')!=layers
        or result.get('input_rows')!=calibration_rows(data) or result.get('states_sha256')!=joint.digest(p/'states.pt')
        or result.get('source_sha256')!={n:joint.digest(args.source_run/n) for n in SOURCE_FILES}):
        raise ValueError('calibration identity/cohort/source changed')
    require_unchanged(result['implementation'],__file__)
    states=torch.load(p/'states.pt',map_location='cpu',weights_only=True)
    if set(states)!={'before','after'} or any(str(v.dtype)!=result['states_dtype'][k] or v.shape!=(len(result['input_rows']),meta['hidden_size']) or not torch.isfinite(v).all() for k,v in states.items()):
        raise ValueError('native calibration states malformed')
    for method in METHODS:
        fit=result['methods'][method]
        for side,layer in zip(('before','after'),layers):
            v=fit['layers'][side]
            if not math.isfinite(v['coefficient']) or not 0<v['coefficient']<=float(bank[f'tf_{method}_C_L{layer}_scale']) or v['absolute_median_error']>fit['tolerance']:
                raise ValueError('invalid frozen calibration coefficient')
    return result


def effective_bank(bank,plan):
    out=dict(bank)
    if plan['dose_mode']=='train_relative':
        for method,record in plan['calibration']['methods'].items():
            for side,layer in zip(('before','after'),plan['c_layers']):
                out[f'tf_{method}_C_L{layer}_scale']=np.asarray(record['layers'][side]['coefficient'])
    return out


def make_plan(args):
    if args.limit not in (None,3):
        raise ValueError('only the predeclared 3-pair pilot or all 192 pairs is allowed')
    meta,bank,data,selection,tune=source(args)
    cs,fixed,layers=conditions(meta,bank,selection)
    calibration=load_calibration(args,meta,bank,data,layers)
    cohort=data['tf']['tune'];pairs=list(dict.fromkeys(str(r['pair_id']) for r in cohort))
    if len(pairs)!=192 or len(cohort)!=384 or Counter(str(r['pair_id']) for r in cohort)!=Counter({p:2 for p in pairs}):
        raise ValueError('source must contain the fixed 192 complete TF pairs')
    keep=set(pairs[:args.limit]);inputs=[clean(r) for r in cohort if str(r['pair_id']) in keep]
    plan=dict(protocol=PROTOCOL,model=args.model,split='tune',exploratory=True,pilot_limit=args.limit,
        bank_fingerprint=meta['fingerprint'],dose_mode=getattr(args,'dose_mode','original_sigma'),
        calibration=calibration,calibration_json_sha256=joint.digest(args.calibration_run/'calibration.json') if calibration else None,source_sha256={n:joint.digest(args.source_run/n) for n in SOURCE_FILES},
        implementation=implementation_identity(__file__),fixed_E=fixed,c_layers=layers,c_alphas=list(ALPHAS),
        layer_rule='ceil(.25*N)-1 and ceil(.75*N)-1, each own saved C and scale',
        input_rows=inputs,conditions=cs,batch_size=meta['config']['batch_size'],seed=meta['config']['seed'],
        coverages=list(COVERAGES),write_contexts=['answer','gate'],precision='fp32 output head; native backbone',
        claims_limit='C position and its local vector/scale change together. No pure-time-order or held-out claim.')
    plan['fingerprint']=fingerprint(plan)
    return plan


def prepare(args):
    plan=make_plan(args);args.run_dir.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    for rel,h in plan['implementation']['files'].items():
        dest=args.run_dir/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and joint.digest(dest)!=h:
            raise ValueError('existing source snapshot differs')
        shutil.copyfile(root/rel,dest)
        if joint.digest(dest)!=h:
            raise ValueError('inconsistent source snapshot')
    joint.save(args.run_dir/'plan.json',plan,immutable=True)
    print('E38 PREPARED',args.model,'pairs',len(plan['input_rows'])//2,'conditions',len(plan['conditions']),flush=True)
    return plan


def load_plan(args):
    plan=joint.read(args.run_dir/'plan.json')
    if plan.get('protocol')!=PROTOCOL or plan.get('model')!=args.model or plan.get('split')!='tune':
        raise ValueError('foreign plan')
    require_unchanged(plan['implementation'],__file__)
    if make_plan(args)!=plan:
        raise ValueError('source/code/CLI changed after plan freeze')
    for rel,h in plan['implementation']['files'].items():
        if joint.digest(args.run_dir/'source_snapshot'/rel)!=h:
            raise ValueError('source snapshot changed')
    return plan


def traced_forward(bundle,texts,specs):
    """Use the production intervention hook, observe its actual rounded residual."""
    inputs=bundle.encode(texts)
    if not inputs['attention_mask'][:,-1].all():
        raise ValueError('left padding required')
    events=[];pending={};handles=[];before={}
    for spec in specs:
        original=spec.delta
        def measured(h_pos,*a,_s=spec,_f=original,**kw):
            delta=_f(h_pos,*a,**kw)
            if _s.layer in pending:
                raise ValueError('diagnostic requires at most one write per layer')
            if not torch.isfinite(h_pos).all() or not torch.isfinite(delta).all():
                raise ValueError('nonfinite requested intervention')
            pending[_s.layer]=(h_pos.detach().clone(),delta.detach().clone(),_s.name)
            events.append(_s.layer)
            return delta
        spec.delta=measured
    try:
        with torch.no_grad(),steer(bundle,specs) as hooks:
            for layer in hooks:
                def observed(module,inp,out,_layer=layer):
                    if _layer not in pending or _layer in before:
                        raise RuntimeError('missing or repeated actual write')
                    pre,delta,name=pending[_layer]
                    post=(out[0] if isinstance(out,tuple) else out)[:,-1].detach().float()
                    actual=post-pre
                    if not torch.isfinite(post).all():
                        raise ValueError('nonfinite actual post-write residual')
                    before[_layer]={'name':name,'layer':_layer,
                        'residual_l2_before':pre.norm(dim=1).cpu().tolist(),
                        'requested_delta_l2':delta.norm(dim=1).cpu().tolist(),
                        'actual_delta_l2':actual.norm(dim=1).cpu().tolist(),
                        'actual_delta_over_residual':(actual.norm(dim=1)/pre.norm(dim=1).clamp_min(1e-12)).cpu().tolist()}
                handles.append(bundle.layer(layer).register_forward_hook(observed))
            logits=forward_logits(bundle,inputs,logits_to_keep=1)[:,-1].detach().float().cpu().numpy()
            expected=sorted(s.layer for s in specs)
            if events!=expected or any(s.n_applied!=1 for s in specs) or any(h.call_count!=1 for h in hooks.values()):
                raise RuntimeError('actual execution order/count differs from declared layer order')
    finally:
        for h in handles:h.remove()
    if not np.isfinite(logits).all():
        raise ValueError('nonfinite output logits')
    traces=[]
    for i in range(len(texts)):
        traces.append([{k:(v[i] if isinstance(v,list) else v) for k,v in before[layer].items()} for layer in events])
    tokens=inputs['input_ids'].detach().cpu().tolist();mask=inputs['attention_mask'].detach().cpu().tolist()
    return logits,traces,[{'input_ids':x,'attention_mask':m} for x,m in zip(tokens,mask)]


def infer_chunk(bundle,meta,bank,rows,condition):
    # No labels, reference answers or outcomes enter this function.
    if any(set(r)!={'id','pair_id','text'} for r in rows):
        raise ValueError('inference accepts sanitized input rows only')
    gate_texts=[joint.gate_text(bundle,meta,'tf',r) for r in rows]
    lm=joint.gate_map(bundle,gate_texts[0],meta['config'])
    gl,gt,ginputs=traced_forward(bundle,gate_texts,intervention_specs(bank,condition,'gate'))
    answer_texts=[joint.tf_prompt(bundle,meta['prompts'],r['text'],allow_idk=False) for r in rows]
    variants=meta['prompts']['statement']['label_variants']
    am=build_label_token_map(bundle.tokenizer,answer_texts[0],{k:variants[k] for k in ('True','False')})
    al,at,ainputs=traced_forward(bundle,answer_texts,intervention_specs(bank,condition,'answer'))
    out=[]
    for i,r in enumerate(rows):
        lp=label_logprobs_from_logits(al[i],am,use_abstain=False);margin=float(lp['True']-lp['False'])
        probs=np.exp(gl[i].astype(float)-np.logaddexp.reduce(gl[i].astype(float)))
        yes,no=int(lm['yes']),int(lm['no']);top=int(np.argmax(gl[i]))
        out.append({**r,'output':'True' if margin>0 else 'False','predicted_true':bool(margin>0),
            'answer_margin':margin,'simple_score':abs(margin),'score':float(gl[i,yes]-gl[i,no]),'finished':True,
            'gate_p_yes':float(probs[yes]),'gate_p_no':float(probs[no]),'gate_top1_id':top,
            'gate_top1_text':bundle.tokenizer.decode([top]),'gate_label_ids':{'yes':yes,'no':no},
            'gate_prompt':gate_texts[i],'answer_prompt':answer_texts[i],
            'gate_input':ginputs[i],'answer_input':ainputs[i],'gate_writes':gt[i],'answer_writes':at[i]})
    return out


def validate_rows(rows,inputs,c,partial=False):
    if len(rows)>len(inputs) or [r.get('id') for r in rows] != [r['id'] for r in inputs[:len(rows) if partial else len(inputs)]]:
        raise ValueError('row cohort/order mismatch')
    expected=sorted([x['layer'] for x in (c['e'],c['c']) if x])
    for r,q in zip(rows,inputs):
        if any(r.get(k)!=v for k,v in q.items()) or r.get('finished') is not True:
            raise ValueError('question text/identity changed')
        for k in ('score','simple_score','answer_margin','gate_p_yes','gate_p_no'):
            if isinstance(r.get(k),bool) or not isinstance(r.get(k),(float,int)) or not math.isfinite(r[k]):
                raise ValueError('nonfinite/missing inference score')
        if r['simple_score']!=abs(r['answer_margin']) or r['predicted_true']!=(r['answer_margin']>0) or r['output']!=('True' if r['predicted_true'] else 'False'):
            raise ValueError('prediction/margin mismatch')
        if not 0<=r['gate_p_yes']<=1 or not 0<=r['gate_p_no']<=1 or r['gate_p_yes']+r['gate_p_no']>1+1e-8:
            raise ValueError('invalid gate probability')
        for context in ('gate','answer'):
            events=r[context+'_writes']
            if [v['layer'] for v in events]!=expected:
                raise ValueError('saved layer execution order mismatch')
            for v in events:
                for key in ('residual_l2_before','requested_delta_l2','actual_delta_l2','actual_delta_over_residual'):
                    if not math.isfinite(v[key]) or v[key]<0:
                        raise ValueError('invalid write norm')
            token=r[context+'_input'];ids=token['input_ids'];mask=token['attention_mask']
            if not ids or len(ids)!=len(mask) or any(type(v)!=int or v<0 for v in ids) or any(v not in (0,1) for v in mask) or mask[-1]!=1:
                raise ValueError('invalid prompt token audit')


def collect(args,bundle=None):
    plan=load_plan(args);meta,bank,data,selection,tune=source(args)
    bank=effective_bank(bank,plan)
    bundle=joint.open_bundle(meta) if bundle is None else bundle
    joint.verify_tokens(bundle,meta)
    ident={k:plan[k] for k in ('protocol','model','split','pilot_limit','implementation')}
    ident['plan_sha256']=joint.digest(args.run_dir/'plan.json')
    path=args.run_dir/'tune.json'
    out=joint.read(path) if path.exists() else {**ident,'completed':False,'rows':{},'control_reproduction':{}}
    if any(out.get(k)!=v for k,v in ident.items()) or not set(out['rows'])<=set(c['name'] for c in plan['conditions']):
        raise ValueError('foreign resume artifact')
    with joint.fp32_output_projection(bundle):
        for c in plan['conditions']:
            rows=out['rows'].setdefault(c['name'],[]);validate_rows(rows,plan['input_rows'],c,True)
            for start in range(len(rows),len(plan['input_rows']),plan['batch_size']):
                rows.extend(infer_chunk(bundle,meta,bank,plan['input_rows'][start:start+plan['batch_size']],c))
                validate_rows(rows,plan['input_rows'],c,True);joint.save(path,out)
            validate_rows(rows,plan['input_rows'],c)
            if c['source_control']:
                old={r['id']:r for r in tune[c['source_control']]}
                metrics={k:max(abs(r[k]-old[r['id']][k]) for r in rows) for k in ('score','simple_score')}
                metrics['prediction_mismatches']=sum(r['output']!=old[r['id']]['output'] for r in rows)
                out['control_reproduction'][c['name']]=metrics
                if plan['pilot_limit'] is None and (max(metrics.values())>1e-5):
                    raise RuntimeError('full-cohort base/E control fails exact source reproduction: '+str(metrics))
            print('E38 CONDITION',args.model,c['name'],len(rows),flush=True)
    require_unchanged(plan['implementation'],__file__)
    if make_plan(args)!=plan:raise RuntimeError('sources changed during collection')
    out['completed']=True;joint.save(path,out)
    return out


def selected(rows,q,selector,seed):
    rs=[{**r,'score':r['simple_score']} for r in rows] if selector=='simple' else rows
    return quota_select(rs,q,seed=seed)


def report(args):
    plan=load_plan(args);meta,bank,data,selection,tune=source(args);raw=joint.read(args.run_dir/'tune.json')
    if (not raw.get('completed') or any(raw.get(k)!=plan[k] for k in ('protocol','model','split','pilot_limit','implementation'))
        or raw.get('plan_sha256')!=joint.digest(args.run_dir/'plan.json') or set(raw['rows'])!={c['name'] for c in plan['conditions']}):
        raise ValueError('incomplete/foreign raw artifact')
    labels={str(r['id']):bool(r['label']) for r in data['tf']['tune']}
    records={};joined={}
    for c in plan['conditions']:
        rs=raw['rows'][c['name']];validate_rows(rs,plan['input_rows'],c)
        joined[c['name']]=[{**r,'outcome':'correct' if r['predicted_true']==labels[r['id']] else 'wrong'} for r in rs]
        r=joined[c['name']];stat={'condition':c,'n':len(r),'correct_all':sum(x['outcome']=='correct' for x in r),'quotas':{},'write_summary':{}}
        for context in ('gate','answer'):
            events=[v for x in r for v in x[context+'_writes']]
            stat['write_summary'][context]={key:{'min':min((v[key] for v in events),default=0),'max':max((v[key] for v in events),default=0)} for key in ('requested_delta_l2','actual_delta_l2','actual_delta_over_residual')}
        stat['gate_summary']={'margin_min':min(x['score'] for x in r),'margin_max':max(x['score'] for x in r),'yes_count':sum(x['score']>0 for x in r),'label_mass_min':min(x['gate_p_yes']+x['gate_p_no'] for x in r)}
        for selector in ('simple','commitment'):
            stat['quotas'][selector]={str(q):selected(r,q,selector,plan['seed']) for q in plan['coverages']}
        records[c['name']]=stat
    differences=[]
    for c in plan['conditions']:
        if c['kind']!='EC':continue
        e=c['method']+'_E';cn=c['name'].replace('_EC_','_C_')
        for selector in ('simple','commitment'):
            for q in map(str,plan['coverages']):
                views={k:records[name]['quotas'][selector][q] for k,name in [('base','baseline'),('E',e),('C',cn),('EC',c['name'])]}
                if not all(v['feasible'] for v in views.values()):raise ValueError('TF quota unexpectedly infeasible')
                selected_sets={k:set(v['selected_ids']) for k,v in views.items()}
                delivered={k:{r['id']:int(r['id'] in selected_sets[k] and r['outcome']=='correct') for r in joined[name]} for k,name in [('base','baseline'),('E',e),('C',cn),('EC',c['name'])]}
                pairs={}
                for r in plan['input_rows']:
                    v=pairs.setdefault(r['pair_id'],{'EC_minus_E':0,'C_minus_base':0})
                    v['EC_minus_E']+=delivered['EC'][r['id']]-delivered['E'][r['id']]
                    v['C_minus_base']+=delivered['C'][r['id']]-delivered['base'][r['id']]
                differences.append({'condition':c['name'],'selector':selector,'coverage':q,'k':views['E']['k'],
                    'correct':{k:v['correct'] for k,v in views.items()},'EC_minus_E':views['EC']['correct']-views['E']['correct'],
                    'C_minus_base':views['C']['correct']-views['base']['correct'],'paired_question_differences':pairs})
    result=dict(protocol=PROTOCOL,model=args.model,split='tune',exploratory=True,pilot_limit=plan['pilot_limit'],
        plan_sha256=joint.digest(args.run_dir/'plan.json'),generation_sha256=joint.digest(args.run_dir/'tune.json'),
        conditions=records,comparisons=differences,control_reproduction=raw['control_reproduction'],dose_mode=plan['dose_mode'],
        interpretation='Every predeclared point is retained. Tuning diagnostics, no selected winner or confirmatory p-value.')
    require_unchanged(plan['implementation'],__file__)
    joint.save(args.run_dir/'report.json',result)
    print('E38 REPORT',args.model,'conditions',len(records),'comparisons',len(differences),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['calibrate','prepare','collect','report']);p.add_argument('--model',required=True,choices=tuple(EXPECTED_E))
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,choices=[3]);p.add_argument('--dose-mode',choices=['original_sigma','train_relative'],default='train_relative')
    p.add_argument('--calibration-run',type=Path);a=p.parse_args()
    {'calibrate':calibrate,'prepare':prepare,'collect':collect,'report':report}[a.phase](a)

if __name__=='__main__':main()
