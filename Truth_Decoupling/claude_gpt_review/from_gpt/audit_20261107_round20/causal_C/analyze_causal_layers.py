#!/usr/bin/env python3
"""Independent bookkeeping audit and descriptive summary of E31 tuning scans.
No experiment/core import; no test selection, no significance claim for winners.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'from_claude/code/scripts/review_checks'))
from e26_independent_recount import quota,read,digest,fingerprint,require


def native(rows):
    return {r['id']:r['outcome'] if r['score']>0 else 'abstain' for r in rows}


def check_counts(d,stored,label):
    if d is None:
        require(stored['feasible'] is False,label+' wrong feasibility');return
    c=Counter(d.values());a=c['correct']+c['wrong']
    require(stored['n']==len(d) and stored['answered']==a and stored['correct']==c['correct'],label+' counts differ')
    if 'wrong' in stored:require(stored['wrong']==c['wrong'],label+' wrong count differs')
    if 'selected_ids' in stored:require(set(stored['selected_ids'])=={i for i,v in d.items() if v!='abstain'},label+' masks differ')
    expected=c['correct']/a if a else None
    require(stored['accuracy_answered']==expected,label+' accuracy differs')


def ranks(values):
    a=np.asarray(values);order=np.argsort(a,kind='stable');out=np.zeros(len(a));i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out


def analyze(run,source_run):
    plan=read(run/'plan.json');raw=read(run/'gate_scores.json');report=read(run/'report.json')
    require(plan['pilot_limit'] is None and plan['split']=='tune' and plan['all_layers'] is False,'full preset-layer tuning run required')
    require(plan['fingerprint']==fingerprint({k:v for k,v in plan.items() if k!='fingerprint'}),'plan fingerprint changed')
    for name,h in plan['source_sha256'].items():require(digest(source_run/name)==h,'source changed '+name)
    for rel,h in plan['implementation']['files'].items():require(digest(run/'source_snapshot'/rel)==h,'snapshot changed '+rel)
    require(raw['completed'] and raw['plan_sha256']==report['plan_sha256']==digest(run/'plan.json'),'run incomplete/foreign')
    require(report['gate_scores_sha256']==digest(run/'gate_scores.json'),'report raw source changed')
    require(raw['implementation']==plan['implementation'],'implementation identity changed')
    bank=read(source_run/'bank.json');data=read(source_run/'data.json');selection=read(source_run/'selection.json')
    require(plan['bank_fingerprint']==bank['fingerprint'],'bank identity changed')
    output=[];summaries=[];checks={'conditions':0,'quota_cells':0,'native_points':0,'matched_count_controls':0}
    for task,tp in plan['tasks'].items():
        old=read(source_run/(task+'_tune'+('_scored' if task=='qa' else '')+'.json'))
        original={n:[r for r in rs if not r.get('gold_is_abstention')] for n,rs in old['rows'].items()}
        ids=[r['id'] for r in original['baseline']]
        require([x['id'] for x in tp['input_rows']]==ids,'tuning cohort changed')
        require(all(set(x)==({'id','pair_id','text'} if task=='tf' else {'id','pair_id','question'}) for x in tp['input_rows']),
                'model input contains answer or label fields')
        names={c['name'] for c in tp['conditions']};rr=raw['tasks'][task]['rows'];rp=report['tasks'][task]
        require(set(rr)==names==set(rp['conditions']),'missing/extra condition')
        joined={}
        for c in tp['conditions']:
            name=c['name'];rows=rr[name];require([x['id'] for x in rows]==ids,'gate row identity changed')
            by={x['id']:x for x in original[c['answer_condition']]}
            require(all(set(r)=={'id','pair_id','score'} and r['pair_id']==by[r['id']]['pair_id'] and math.isfinite(r['score']) for r in rows),
                    'invalid model score payload')
            joined[name]=[{**by[r['id']],'score':r['score']} for r in rows]
            if c['control']:
                diff=max(abs(r['score']-by[r['id']]['score']) for r in rows)
                require(diff<=1e-5 and diff==rp['control_source_score_max_abs_diff'][name],'control fails source reproduction')
            else:
                method=c['method'];n=bank['num_layers'];expected_layers=sorted(set([math.ceil(n*f)-1 for f in (.25,.5,.75)]+bank['c_layers'][task][method]))
                require(c['c']['layer'] in expected_layers and c['c']['alpha'] in (-1.,1.),'unplanned C write')
                fixed_name=selection['tasks'][task]['choices']['0.5'][method+'_E']
                expected_answer='baseline' if c['kind']=='C' else fixed_name
                require(c['answer_condition']==expected_answer,'candidate answer source changed')
                require((c['e'] is None) if c['kind']=='C' else c['e']==tp['fixed_E'][method]['e'],'E changed while scanning C')
        for c in tp['conditions']:
            name=c['name'];rows=joined[name];stored=rp['conditions'][name];checks['conditions']+=1
            dn=native(rows);check_counts(dn,stored['native_gate'],name+' native');checks['native_points']+=1
            qvalues={}
            for q in plan['coverages']:
                q=str(q);d=quota(rows,q,'score',plan['seed']);check_counts(d,stored['quotas'][q],name+'/'+q);checks['quota_cells']+=1
                qvalues[q]=None if d is None else sum(v=='correct' for v in d.values())
            controls={'base_gate':joined[task+'_base_gate'],
                      'base_simple':[{**r,'score':r['simple_score']} for r in joined[task+'_base_gate']]}
            if c['method'] in ('das','md'):
                er=joined[task+'_'+c['method']+'_E_gate']
                controls.update(fixed_E_gate=er,fixed_E_simple=[{**r,'score':r['simple_score']} for r in er])
            a=sum(v!='abstain' for v in dn.values());good=sum(v=='correct' for v in dn.values())
            require(set(stored['same_actual_count_controls'])==set(controls),'native matched controls changed')
            for label,rs in controls.items():
                dc=quota(rs,Fraction(a,len(rows)),'score',plan['seed']);s=stored['same_actual_count_controls'][label]
                check_counts(dc,s,name+' matched '+label);checks['matched_count_controls']+=1
                delta=good-sum(v=='correct' for v in dc.values()) if dc is not None else None
                require(s['delta_correct_native_minus_control']==delta,'matched difference changed')
            if c['control']:continue
            control_name=task+'_base_gate' if c['kind']=='C' else task+'_'+c['method']+'_E_gate';cr=joined[control_name]
            m=np.array([r['score'] for r in rows]);m0=np.array([r['score'] for r in cr])
            rec={'model':plan['model'],'task':task,'method':c['method'],'kind':c['kind'],'layer':c['c']['layer'],
                 'alpha':c['c']['alpha'],'condition':name,'original_read_layer':c['original_read_layer'],
                 'mean_gate_score_change':float((m-m0).mean()),'score_change_sd':float(np.std(m-m0)),
                 'score_rank_correlation':float(np.corrcoef(ranks(m),ranks(m0))[0,1]),
                 'native_answered':a,'native_correct':good,'native_accuracy':good/a if a else None,
                 'n':len(rows),'quota_correct':qvalues,'native_same_K_controls':stored['same_actual_count_controls'],'quota_comparisons':{}}
            for q in plan['coverages']:
                q=str(q);comp={}
                for label,rs in controls.items():
                    dc=quota(rs,q,'score',plan['seed']);correct=None if dc is None else sum(v=='correct' for v in dc.values())
                    comp[label]={'correct':correct,'delta':None if correct is None or qvalues[q] is None else qvalues[q]-correct}
                rec['quota_comparisons'][q]=comp
            output.append(rec)
        for method in ('das','md'):
            for kind in ('C','EC'):
                candidates=[r for r in output if r['model']==plan['model'] and r['task']==task and r['method']==method and r['kind']==kind]
                for q in map(str,plan['coverages']):
                    feasible=[r for r in candidates if r['quota_correct'][q] is not None]
                    best=min(feasible,key=lambda r:(-r['quota_correct'][q],abs(r['alpha']),r['layer'],r['alpha'],r['condition'])) if feasible else None
                    old=[r for r in feasible if r['original_read_layer']]
                    oldbest=max((r['quota_correct'][q] for r in old),default=None)
                    stored=rp['best_on_tune_only'][method+'_'+kind][q]
                    require((best is None and stored is None) or (best is not None and stored['condition']==best['condition']), 'best-on-tune selection differs')
                    summaries.append({'model':plan['model'],'task':task,'method':method,'kind':kind,'coverage':float(q),
                        'best_condition':None if best is None else best['condition'],'best_correct':None if best is None else best['quota_correct'][q],
                        'best_original_read_layer_correct':oldbest,'best_comparisons':None if best is None else best['quota_comparisons'][q],
                        'warning':'same-data tuning maximum, not a held-out gain'})
    return {'model':plan['model'],'checks':checks,'all_match':True,'rows':output,'best_on_tune_summaries':summaries,
            'sources':{n:digest(run/n) for n in ('plan.json','gate_scores.json','report.json')}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    results=[analyze(a.results_root/model/'full192',a.source_root/model) for model in ('llama3.1-8b','qwen3-8b')]
    out={'exploratory':True,'split':'tune','all_match':True,'scope':'gate-only C layer scan; frozen candidate answers; offline judge-defined actual-answer eligibility',
         'runs':results}
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    for run in results:
        print(run['model'],run['checks'])
        for r in run['best_on_tune_summaries']:
            if r['coverage']==.5 and r['method']=='das' and r['kind']=='EC':print(r)
if __name__=='__main__':main()
