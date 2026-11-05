from pathlib import Path
import os
import json, hashlib
from collections import Counter
import numpy as np
from scipy.stats import rankdata, binomtest

ROOT=Path(os.environ.get('TD_AUDIT_CLAUDE_ROOT', str(Path(__file__).resolve().parents[3] / 'from_claude')))
R=ROOT/'results/02_问答DAS_v1'
OUT=Path(os.environ.get('TD_AUDIT_OUT', str(Path(__file__).resolve().parent))); OUT.mkdir(parents=True, exist_ok=True)
checks=[]

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ck(name,actual,expected):
    ok=bool(np.allclose(actual,expected,rtol=0,atol=1e-12,equal_nan=True)) if isinstance(actual,(float,np.floating,list,np.ndarray)) and not isinstance(expected,(dict,list)) else actual==expected
    checks.append({'name':name,'pass':bool(ok),**({} if ok else {'actual':actual,'expected':expected})})
def auc(s,y):
    s,y=np.asarray(s),np.asarray(y)
    n=int(y.sum()); nn=len(y)-n
    return float((rankdata(s)[y==1].sum()-n*(n+1)/2)/(n*nn)) if n and nn else np.nan

def main_runs():
    result={}
    for variant,branch in [('v1','第16轮_全量'),('v1u','第16轮_DAS正结果搜索/v1u')]:
      for model in ['llama3.1-8b','qwen3-8b','gemma3-4b']:
        run=R/branch/model; name=variant+'/'+model
        q=read(run/'questions.json'); bank=read(run/'bank.json')
        split={k:[a['pair_id'] for a in v] for k,v in q['splits'].items()}
        ck(name+'/split_counts',{k:len(v) for k,v in split.items()},{'train':96,'tune':192,'test':529})
        ck(name+'/no_overlap',len(set(split['train'])&set(split['tune']))+len(set(split['train'])&set(split['test']))+len(set(split['tune'])&set(split['test'])),0)
        ck(name+'/est_pair_count',len(q['e_train']),92)
        ck(name+'/est_ids',bank['training_pair_ids'],[a['pair_id'] for a in q['e_train']])
        ck(name+'/questions_sha256',bank['questions_sha256'],sha(run/'questions.json'))
        proxy=read(run/'proxy_scores.json'); short=read(run/'shortlist.json'); sel=read(run/'selection.json')
        ck(name+'/proxy_tune_ids',proxy['question_ids'],[a['pair_id'] for a in q['e_tune']])
        vectors={c:np.array(v['correct'],bool) for c,v in proxy['conditions'].items()}; bv=vectors['baseline']
        def key(c):
          v=vectors[c['name']]
          return (-int(v.sum()),int((bv&~v).sum()),float(c['alpha']),c['layer'],c['name'])
        selected_proxy={}
        for method in ['qa_das','qa_md']:
          by={}
          for c in proxy['condition_specs']:
            if c.get('context')==method and c.get('mode')=='continuous':
              if c['layer'] not in by or key(c)<key(by[c['layer']]): by[c['layer']]=c
          selected_proxy[method]=sorted(by.values(),key=key)[:3]
        ck(name+'/proxy_shortlist',selected_proxy,short['candidates'])
        for f,h in sel['sources'].items(): ck(name+'/freeze_source/'+f,sha(run/f),h)
        phase_results={}
        for phase,rawfile,splitname in [('tune','tune.json','tune'),('test','test.json','test'),('diagnostics','tune_diagnostics.json','tune')]:
          if not (run/(phase+'_scored.json')).exists(): continue
          s=read(run/(phase+'_scored.json')); d=read(run/(phase+'_scored_items.json')); raw=read(run/rawfile)
          rows=d['evaluations']; ck(name+'/'+phase+'/ids',[e['pair_id'] for e in rows],split[splitname])
          ck(name+'/'+phase+'/completed',s['completed'],True)
          ck(name+'/'+phase+'/sourcehash',s['source_generation_sha256'],sha(run/rawfile))
          rb={e['pair_id']:e for e in raw['evaluations']}
          for e in rows:
            ck(name+'/'+phase+'/raw_text/'+e['pair_id'],e['outputs'],rb[e['pair_id']]['outputs'])
            ck(name+'/'+phase+'/labels/'+e['pair_id'],e['outcome'],{c:x['label'] for c,x in e['judge'].items()})
          valid=[e for e in rows if not e['gold_is_abstention']]
          n=len(valid); ck(name+'/'+phase+'/denominator',n,523 if splitname=='test' else 192)
          ck(name+'/'+phase+'/resolved',s['n_resolved_all_conditions'],n)
          base=np.array([e['outcome']['baseline'] for e in valid]); bs=(base=='correct').astype(int)
          rng=np.random.default_rng(0); boots=rng.choice(n,(2000,n),replace=True)
          table={}
          for c in d['conditions']:
            labs=np.array([e['outcome'][c] for e in valid]); counts=dict(Counter(labs)); trans={a+'->'+b:int(((base==a)&(labs==b)).sum()) for a in ('correct','wrong','abstain') for b in ('correct','wrong','abstain')}
            ck(name+'/'+phase+'/unresolved/'+c,s['rates'][c]['n_unresolved'],0)
            for status in ('correct','wrong','abstain'):
              ck(name+'/'+phase+'/'+c+'/'+status,float((labs==status).mean()),s['rates'][c]['rates'][status])
            delta=(labs=='correct').astype(int)-bs
            ci=np.percentile(delta[boots].mean(1),[2.5,97.5]).tolist()
            if c!='baseline':
              ck(name+'/'+phase+'/'+c+'/CI',np.array(ci),np.array(s['rates'][c]['delta_vs_baseline']['correct']['ci95']))
              ck(name+'/'+phase+'/'+c+'/transitions',trans,s['transitions_vs_baseline'][c])
            table[c]={'counts':counts,'n':n,'delta_correct':float(delta.mean()),'ci95':ci,'transitions':trans,'net_answered':int((labs!='abstain').sum()-(base!='abstain').sum())}
          if phase=='tune':
            specs={c['name']:c for c in d['condition_specs']}; chosen={}
            for method in ['qa_das','qa_md']:
              eligible=[c for c in d['condition_specs'] if c.get('context')==method and table[c['name']]['delta_correct']>0 and abs(table[c['name']]['net_answered'])<=5]
              def realkey(c):
                t=table[c['name']]; tr=t['transitions']; return(-round(t['delta_correct']*192),tr['correct->wrong']+tr['correct->abstain'],float(c['alpha']),c['layer'],c['name'])
              chosen[method]=min(eligible,key=realkey) if eligible else None
            ck(name+'/real_freeze',chosen,sel['selected'])
          if phase=='test':
            ck(name+'/test_only_frozen',d['condition_specs'],[{'name':'baseline','kind':'baseline','eligible_for_selection':False,'uses_reference_answer':False}]+[c for c in sel['selected'].values() if c is not None])
          phase_results[phase]=table
        result[name]={'selection':sel['selected'],'shortlist':short['candidates'],'phases':phase_results}
    return result

def measurements():
    base=R/'第16轮_DAS正结果搜索/measurements'; out={}
    for p in sorted(base.glob('*heldout*.json')):
      d=read(p); s=d['summary']; per=d.get('per_pair') or d.get('per_question'); ispair='per_pair' in d
      vals={k:np.array(v['correct'] if ispair else v,bool) for k,v in per.items()}; b=vals['baseline']; n=len(b)
      rng=np.random.default_rng(0); boots=rng.integers(0,n,(2000,n)); result={}
      model=s['model']; variant='v1u' if p.name.startswith('v1u_') else 'v1'; branch='第16轮_DAS正结果搜索/v1u' if variant=='v1u' else '第16轮_全量'
      run=R/branch/model; q=read(run/'questions.json'); sh=read(run/'shortlist.json')
      ck(p.name+'/picks',s['pre_specified_pick'],{k:v[0]['name'] if v else None for k,v in sh['candidates'].items()})
      ck(p.name+'/no_tune_overlap',len(set(d['pair_ids'])&{x['pair_id'] for x in q['splits']['tune']}),0)
      for k,v in vals.items():
        if k=='baseline': continue
        delta=v.astype(int)-b.astype(int); report=s['conditions'][k]
        ck(p.name+'/'+k+'/delta',float(delta.mean()),report['test_delta' if ispair else 'delta'])
        ci=np.percentile(delta[boots].mean(1),[2.5,97.5]); ck(p.name+'/'+k+'/CI',ci,np.array(report['test_delta_ci95' if ispair else 'delta_ci95']))
        pos=int((delta>0).sum()); neg=int((delta<0).sum())
        result[k]={'corrected':pos,'harmed':neg,'delta':float(delta.mean()),'ci95':ci.tolist(),'exact_two_sided_p':binomtest(pos,pos+neg,.5).pvalue if pos+neg else 1.0}
      picks=s['pre_specified_pick']; a,bm=vals[picks['qa_das']],vals[picks['qa_md']]; diff=a.astype(int)-bm.astype(int); pos=int((diff>0).sum()); neg=int((diff<0).sum())
      out[p.name]={'n':n,'conditions':result,'das_minus_md':{'delta':float(diff.mean()),'ci95':np.percentile(diff[boots].mean(1),[2.5,97.5]).tolist(),'discordant':[pos,neg],'exact_two_sided_p':binomtest(pos,pos+neg,.5).pvalue if pos+neg else 1.}}
    for p in sorted(base.glob('*own*detect*.json')):
      d=read(p)
      if 'test_scores' not in d:
        out[p.name]={'raw_scores_absent':True,'keys':list(d)}; continue
      model=d['model']; branch='第16轮_DAS正结果搜索/v1u' if p.name.startswith('v1u_') else '第16轮_全量'; run=R/branch/model
      raw=read(run/'test_scored_items.json')['evaluations']; answered=[e for e in raw if not e['gold_is_abstention'] and e['outcome']['baseline'] in ('correct','wrong')]
      ck(p.name+'/test_ids',d['test_ids'],[e['pair_id'] for e in answered]); y=np.array(d['test_labels']); ck(p.name+'/labels',y.tolist(),[int(e['outcome']['baseline']=='correct') for e in answered])
      n=len(y); rng=np.random.default_rng(0); boots=rng.integers(0,n,(2000,n)); scores={k:np.array(v) for k,v in d['test_scores'].items()}; auc_boot={}
      for k,s in scores.items():
        ck(p.name+'/'+k+'/auc',auc(s,y),d['picked'][k]['test_auroc']); auc_boot[k]=np.array([auc(s[i],y[i]) for i in boots]); ck(p.name+'/'+k+'/ci',np.percentile(auc_boot[k],[2.5,97.5]),np.array(d['picked'][k]['test_auroc_ci95']))
      for key,v in d.get('paired_auroc_differences_test',{}).items():
        a,b=key.split('_minus_'); ck(p.name+'/'+key+'/paired_ci',np.percentile(auc_boot[a]-auc_boot[b],[2.5,97.5]),np.array(v['ci95']))
      for family in ['das','md']:
        cand={k:v for k,v in d['all'].items() if v['family']==family}; pick=max(cand,key=lambda k:(cand[k]['tune_auroc'],-cand[k]['layer'])); ck(p.name+'/'+family+'/tune_pick',pick,d['picks'][family])
      out[p.name]={'n':n,'correct':int(y.sum()),'picks':d['picks'],'auroc':{k:auc(s,y) for k,s in scores.items()},'paired':d.get('paired_auroc_differences_test',{})}
    for p in sorted(base.glob('verify_own*.json')):
      d=read(p); raw=read(R/'第16轮_全量'/d['model']/'test_scored_items.json')['evaluations']; rows=[e for e in raw if not e['gold_is_abstention']]; ck(p.name+'/ids',d['pair_ids']['test'],[e['pair_id'] for e in rows])
      o=np.array([{'correct':1,'wrong':-1,'abstain':0}[r['outcome']['baseline']] for r in rows]); n=len(o); rng=np.random.default_rng(0); boots=rng.integers(0,n,(2000,n)); tab={}
      for k,v in d['margins']['test'].items():
        m=np.array([np.nan if x is None else x for x in v]); keep=np.where(~np.isnan(m),m>=0,False)|(o==0); strict=np.where(~np.isnan(m),m>0,False)|(o==0); sample=o*keep
        if k=='baseline': base_sample=sample
        r=d['readers'][k]['gates']['own_verdict']; ck(p.name+'/'+k+'/mean',float(sample.mean()),r['test_cmw_after'])
        result={'kept_correct':int(((o==1)&keep).sum()),'kept_wrong':int(((o==-1)&keep).sum()),'ties':int((m==0).sum()),'tie_ids':[rows[i]['pair_id'] for i in np.where(m==0)[0]],'current_cmw':float(sample.mean()),'strict_cmw':float((o*strict).mean())}
        if k!='baseline':
          diff=sample-base_sample; ci=np.percentile(diff[boots].mean(1),[2.5,97.5]); ck(p.name+'/'+k+'/delta',float(diff.mean()),r['vs_unsteered_verifier']['delta']); ck(p.name+'/'+k+'/CI',ci,np.array(r['vs_unsteered_verifier']['ci95'])); result['delta_vs_verifier']=float(diff.mean()); result['delta_ci95']=ci.tolist()
        tab[k]=result
      out[p.name]=tab
    return out

result={'runs':main_runs(),'measurements':measurements()}
result['checks']={'n':len(checks),'n_failed':sum(not x['pass'] for x in checks),'failures':[x for x in checks if not x['pass']]}
(OUT/'recomputed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=True))
print(json.dumps(result['checks'],ensure_ascii=False,indent=2))
for name,d in result['runs'].items():
 print(name,'selected',{k:v and v['name'] for k,v in d['selection'].items()})
 print('test',[(k,v['counts'],v['delta_correct']) for k,v in d['phases']['test'].items()])
for name,d in result['measurements'].items():
 if 'das_minus_md' in d: print(name,'DAS-MD',d['das_minus_md'])
 if name.startswith('verify_'): print(name,'ties',{k:v['ties'] for k,v in d.items()})
