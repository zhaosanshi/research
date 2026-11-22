from pathlib import Path
import json,sys,hashlib,collections
import numpy as np
sys.path[:0]=['Iteration_01/diag','Iteration_01/code/scripts/review_checks']
import iti_table as A, cap_score as CS, iti_gsm8k as I
from judge_two_axis_run import load_verified_single
model='gemma3-27b';base=Path(A.BASE)/model;fit=base/'faithful_seed42'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
disp,l,s=A.KEY[model]
Y=A.MP.prompt_split(disp,l,0,s)[0]; X=A.MP.prompt_split(disp,l,1,s)[0]
sp,txt,_,ids=X;cfg=A.ML.n1_pick(Y[0],Y[1],s);_,run,_=A.MP.usual(disp,1)
ch=read(fit/'choice.json'); oc=read(fit/'choice_all.json'); assert ch['stem']==oc['stem']
raw={}
for p in Path(run,'eval_extra').glob('judge_tune_iti_faithful_seed42*_tune'):
 if not (p/'judge2/single_pro.jsonl').is_file(): continue
 inp=read(p/'judge_input.json');_,cells,votes=load_verified_single(str(p),'pro',generation_path=str(p/'judge_input.json'),require_complete=False)
 for cond,source in inp['source_files'].items():
  if not cond.startswith('ITI') and 'iti' not in cond.lower():continue
  gen=fit/'gen'/source['file']; assert sha(gen)==source['sha256']
  gd=read(gen);m={(str(x['pair_id'])):x for x in gd['rows']};v=[]
  ctab={c['pair_id']:(c,votes[cid]) for cid,c in cells.items() if c['condition']==cond}
  for q in Y[3]:
   c,r=ctab[q]; assert c['prediction']==m[q]['text']
   if r['status']!='scored' and (c['prediction'].strip() or r.get('basis')!='empty_output_no_judge_vote'): raise ValueError('unresolved development verdict')
   v.append((int(r.get('label')=='A')-int(r.get('label')=='B')) if r['status']=='scored' else 0)
  raw[(gd['setting']['k'],gd['setting']['alpha'])]=sum(v)
print('tune raw',len(raw),'best',sorted(raw.items(),key=lambda x:(-x[1],x[0]))[:4],flush=True)
p=Path(A.MP.work(run,'test','iti_faithful_seed42'))
gpath=fit/'gen'/('test_'+ch['stem']+'.json'); g=read(gpath); inp=read(p/'judge_input.json')
assert inp['source_files'][g['condition']]['sha256']==sha(gpath) and g['setting']['fit_file_sha256']==sha(base/'fit_faithful_seed42.pt')
_,cs,rv=load_verified_single(str(p),'pro',generation_path=str(p/'judge_input.json'),require_complete=False)
vote={x['pair_id']:(x,rv[i]) for i,x in cs.items() if x['condition']==g['condition']}; rr={x['pair_id']:x for x in g['rows']}
assert len(rr)==len(g['rows']) and list(rr)==g['question_ids']
iti=[]; flags=[]
for i in ids:
 x,v=vote[i]; assert rr[i]['text']==x['prediction'] and rr[i]['rule_abstain']==x['rule_abstain']; assert v['status']=='scored' or not x['prediction'].strip()
 iti.append((v.get('label')=='A')-(v.get('label')=='B'));flags.append(not x['rule_abstain'] and bool(x['prediction'].strip()))
print('raw ITI',len(ids),sum(flags),sum(iti),collections.Counter(iti),flush=True)
for label,syscfg in [('E',('E',s)),('N1',cfg)]:
 st,ans,_=A.S.system(sp,txt,syscfg); z=[(x=='R')-(x=='W') for x in st];k=min(sum(flags),sum(ans)); u=sorted(x for x,y in zip(iti,flags) if y);v=sorted(x for x,y in zip(z,ans) if y)
 joint=[(a,b) for a,b,pa,pb in zip(z,iti,ans,flags) if pa and pb]
 print(label,syscfg,'net,issued',sum(z),sum(ans),'diff',A.ps.ci([a-b for a,b in zip(z,iti)]),'equal range',k,(sum(v[:k])-sum(u[-k:]),sum(v[-k:])-sum(u[:k])),'both',len(joint),A.ps.ci([a-b for a,b in joint]),flush=True)
gsmids=read(I.IDS)['ids'];gold=CS.gold_of('gsm8k')
def mathscore(path):
 d=read(path); assert [r['id'] for r in d['rows']]==gsmids and d['completed']; return [(CS.judge_row('gsm8k',r,gold(r['id']))[0]) for r in d['rows']]
m0=mathscore(Path(I.ROOT)/model/'b8/neutral.json');mi=mathscore(fit/'gsm8k'/(ch['stem']+'.json')); me=mathscore(Path(I.ROOT)/model/'b8/neutral--E28-w-p1-at-prompt.json');v=lambda a:(a=='R')-(a=='W')
print('GSM8K count',len(gsmids),'base ITI E',*[collections.Counter(x) for x in [m0,mi,me]],'E-ITI net',A.ps.ci([v(x)-v(y) for x,y in zip(me,mi)]),flush=True)
m0=read(base/'mmlu_k0.json');mi=read(fit/'mmlu'/(ch['stem']+'.json'));r={a['id']:(a['pick']==a['gold']) for a in m0['rows']};t={a['id']:(a['pick']==a['gold']) for a in mi['rows']};dev,rep=A.PM.halves(list(r));assert len(r)==len(t)==len(m0['rows'])==len(mi['rows'])==14042 and r.keys()==t.keys()
print('MMLU split',[(len(k),sum(t[q]-r[q] for q in k),A.PM.boot([t[q]-r[q] for q in k])) for k in [dev,rep]],flush=True)
for b in sorted(Path('Iteration_01/baselines/caa').glob('*')):
 print('CAA snapshot',b.name,'accept', (b/'acceptance/ACCEPT_DONE').exists(),'mmlu',len([p for p in (b/'mmlu').glob('L*.json') if not p.name.endswith('.receipt.json')]),'tune',len(list(b.glob('run_caa/eval_extra/gen/x--*tune.json'))),'judges',len(list(b.glob('run_caa/eval_extra/judge_*/judge2/single_pro.jsonl'))),'choices',len(list(b.glob('choice*.json'))),'tests',len(list(b.glob('run_caa/eval_extra/gen/*test.json'))),'gsm',len(list((b/'gsm8k').glob('*.json'))))
