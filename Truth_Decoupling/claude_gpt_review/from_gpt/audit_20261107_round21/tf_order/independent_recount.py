#!/usr/bin/env python3
"""Independent E38 counts, whole-question resampling/reselection and source audit.
Does not import E38, its report functions, or core.selective_eval.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import csv
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def tie(i,seed):return hashlib.sha256(json.dumps(['actual_coverage_v1',seed,i],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def exact_pair_swap(differences):
    d=[int(x) for x in differences if x]
    distribution={0:1}
    for v in d:
        nxt={}
        for subtotal,ways in distribution.items():
            for sign in (-1,1):nxt[subtotal+sign*abs(v)]=nxt.get(subtotal+sign*abs(v),0)+ways
        distribution=nxt
    observed=abs(sum(d))
    return sum(v for k,v in distribution.items() if abs(k)>=observed)/2**len(d)


def one(run,source,n_boot):
    plan=read(run/'plan.json');raw=read(run/'tune.json');report=read(run/'report.json')
    bank=read(source/'bank.json');data=read(source/'data.json')
    assert plan['fingerprint']==fp({k:v for k,v in plan.items() if k!='fingerprint'})
    assert plan['bank_fingerprint']==bank['fingerprint']
    assert all(sha(source/n)==h for n,h in plan['source_sha256'].items())
    assert all(sha(run/'source_snapshot'/n)==h for n,h in plan['implementation']['files'].items())
    assert raw['completed'] and raw['plan_sha256']==sha(run/'plan.json')==report['plan_sha256']
    assert report['generation_sha256']==sha(run/'tune.json')
    assert all(raw[k]==plan[k] for k in ('protocol','model','split','pilot_limit','implementation'))
    assert set(raw['rows'])==set(report['conditions'])=={c['name'] for c in plan['conditions']}
    inputs=plan['input_rows'];ids=[r['id'] for r in inputs];n=len(ids)
    assert len(set(ids))==n
    groups=sorted({r['pair_id'] for r in inputs});index={g:i for i,g in enumerate(groups)}
    gi=np.asarray([index[r['pair_id']] for r in inputs]);assert set(Counter(gi).values())=={2}
    gold={str(r['id']):bool(r['label']) for r in data['tf']['tune']}
    freqs=np.random.default_rng(42).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=n_boot)
    records={};boot={};cells=[];calibration={};controls={}
    old_tune=read(source/'tf_tune.json')
    if plan['dose_mode']=='train_relative':
        import torch
        cal=plan['calibration'];cp=run.parent/'train_calibration'
        assert sha(cp/'calibration.json')==plan['calibration_json_sha256'] and read(cp/'calibration.json')==cal
        assert sha(cp/'states.pt')==cal['states_sha256']
        states=torch.load(cp/'states.pt',map_location='cpu',weights_only=True)
        assert cal['input_rows']==[{k:str(r[k]) for k in ('id','pair_id','text')} for r in data['tf']['train']]
        assert len(cal['input_rows'])==192 and len({r['pair_id'] for r in cal['input_rows']})==96
        assert all(str(h.dtype)==cal['states_dtype'][side] and h.shape==(192,bank['hidden_size']) for side,h in states.items())
        with np.load(source/'bank.npz',allow_pickle=False) as vectors:
            for method,fit in cal['methods'].items():
                for side,layer in zip(('before','after'),plan['c_layers']):
                    h=states[side];w=np.asarray(vectors[f'tf_{method}_C_L{layer}'],dtype=np.float64);w/=np.linalg.norm(w)
                    w=torch.tensor(w,dtype=torch.float32);coef=fit['layers'][side]['coefficient'];ratios=[]
                    for sign in (-1,1):
                        delta=sign*(w[:,None]@torch.tensor([coef],dtype=torch.float32))
                        actual=(h.float()+delta).to(h.dtype).float()-h.float()
                        ratios.extend((actual.norm(dim=-1)/h.float().norm(dim=-1)).tolist())
                    actual_median=float(np.median(ratios))
                    assert abs(actual_median-fit['rho'])<=fit['tolerance']
                    calibration[method+'_'+side]={'native_dtype':str(h.dtype),'n':len(h),'rho':fit['rho'],'recomputed_median':actual_median,'coefficient':coef,'negative_median':float(np.median(ratios[:len(h)])),'positive_median':float(np.median(ratios[len(h):]))}
    for c in plan['conditions']:
        name=c['name'];rows=raw['rows'][name]
        assert [r['id'] for r in rows]==ids
        if c['source_control']:
            old={r['id']:r for r in old_tune['rows'][c['source_control']]}
            checked={key:max(abs(r[key]-old[r['id']][key]) for r in rows) for key in ('score','simple_score')}
            checked['prediction_mismatches']=sum(r['output']!=old[r['id']]['output'] for r in rows)
            assert checked==raw['control_reproduction'][name]==report['control_reproduction'][name]
            if plan['pilot_limit'] is None:assert max(checked.values())<=1e-5
            controls[name]=checked
        correct=np.asarray([int(r['predicted_true']==gold[r['id']]) for r in rows])
        assert int(correct.sum())==report['conditions'][name]['correct_all']
        for r,q in zip(rows,inputs):
            assert all(r[k]==v for k,v in q.items())
            assert r['output']==('True' if r['answer_margin']>0 else 'False')
            assert r['simple_score']==abs(r['answer_margin'])
            for context in ('answer','gate'):
                assert [v['layer'] for v in r[context+'_writes']]==sorted(s['layer'] for s in (c['e'],c['c']) if s)
                assert all(np.isfinite(v[k]) and v[k]>=0 for v in r[context+'_writes'] for k in ('actual_delta_l2','requested_delta_l2','actual_delta_over_residual'))
        for selector,key in [('simple','simple_score'),('commitment','score')]:
            ordering=np.asarray(sorted(range(n),key=lambda i:(-rows[i][key],tie(ids[i],plan['seed']),ids[i])))
            counts=freqs[:,gi[ordering]];cum=counts.cumsum(axis=1)
            for q in plan['coverages']:
                k=int(Fraction(str(q))*n);mask=np.zeros(n,dtype=int);mask[ordering[:k]]=1
                delivered=mask*correct;assert int(delivered.sum())==report['conditions'][name]['quotas'][selector][str(q)]['correct']
                assert sorted(ids[i] for i in ordering[:k])==report['conditions'][name]['quotas'][selector][str(q)]['selected_ids']
                records[(name,selector,str(q))]={'correct':int(delivered.sum()),'delivered':delivered,'k':k}
                selected_count=np.minimum(counts,np.maximum(k-(cum-counts),0))
                boot[(name,selector,str(q))]=(selected_count*correct[ordering]).sum(axis=1)
                cells.append({'condition':name,'selector':selector,'coverage':q,'k':k,'correct':int(delivered.sum())})
    comparisons=[]
    for c in plan['conditions']:
        if c['kind']!='EC':continue
        name=c['name'];ename=c['method']+'_E';cname=name.replace('_EC_','_C_')
        for selector in ('simple','commitment'):
            for q in map(str,plan['coverages']):
                keys={arm:(cond,selector,q) for arm,cond in [('base','baseline'),('E',ename),('C',cname),('EC',name)]}
                counts={arm:records[key]['correct'] for arm,key in keys.items()}
                deltas=records[keys['EC']]['delivered']-records[keys['E']]['delivered']
                grouped=np.bincount(gi,weights=deltas,minlength=len(groups)).astype(int)
                sample=boot[keys['EC']]-boot[keys['E']]
                comparisons.append({'model':plan['model'],'method':c['method'],'condition':name,'c_layer':c['c']['layer'],
                    'e_layer':c['e']['layer'],'c_alpha':c['c']['alpha'],'order':'C_before_E' if c['c']['layer']<c['e']['layer'] else 'E_before_C',
                    'selector':selector,'coverage':q,'k':records[keys['E']]['k'],'correct':counts,
                    'EC_minus_E':counts['EC']-counts['E'],'C_minus_base':counts['C']-counts['base'],
                    'pair_swap_p_fixed_selection':exact_pair_swap(grouped),
                    'question_bootstrap_reselect_ci95_correct_count':np.quantile(sample,[.025,.975]).tolist(),
                    'n_question_groups':len(groups),'n_boot':n_boot})
    return {'model':plan['model'],'pilot_limit':plan['pilot_limit'],'source_plan_sha256':sha(run/'plan.json'),
        'raw_sha256':sha(run/'tune.json'),'report_sha256':sha(run/'report.json'),'cells_checked':len(cells),
        'calibration_independently_recomputed':calibration,'source_controls_independently_recomputed':controls,'cells':cells,'comparisons':comparisons,
        'interpretation':'Exploratory fixed-grid tuning audit. Group-swap p fixes observed masks; bootstrap resamples/reselects whole questions.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--phase',choices=['pilot3','full192'],required=True)
    p.add_argument('--source',type=Path,default=Path('/tmp/td_r20_20261107/Truth_Decoupling/claude_gpt_review/from_gpt/audit_20261107_round20/tf_ability/results'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--n-boot',type=int,default=2000);a=p.parse_args()
    out={m:one(a.results/m/a.phase,a.source/m/'tf_ability_full',a.n_boot) for m in ('llama3.1-8b','qwen3-8b')}
    # Frozen R21 family: 48 EC-vs-fixed-E comparisons across both models,
    # both estimators, both C sites, both signs and three quotas, per readout.
    for selector in ('simple','commitment'):
        family=sorted([r for model in out.values() for r in model['comparisons'] if r['selector']==selector],key=lambda r:r['pair_swap_p_fixed_selection'])
        assert len(family)==48
        running=0
        for i,r in enumerate(family):
            running=max(running,min(1,(len(family)-i)*r['pair_swap_p_fixed_selection']))
            r['holm_within_48_prespecified_points']=running
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print('INDEPENDENT PASS',a.phase,{m:{'cells':r['cells_checked'],'comparisons':len(r['comparisons'])} for m,r in out.items()})

if __name__=='__main__':main()
