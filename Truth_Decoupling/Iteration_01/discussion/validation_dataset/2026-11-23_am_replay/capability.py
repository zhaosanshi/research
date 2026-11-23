"""Run from Truth_Decoupling on the scoring container's CPU. Read complete archived generations and option probabilities."""
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "Iteration_01/diag")
import cap_score as C
from cap_prompts import gsm8k_parse, _ANYNUM, _to_number
R = Path("Iteration_01/capability")
MODELS = {"llama3.1-8b":(16,3.5),"qwen3.5-9b":(15,2.0),"qwen3.8-27b":(28,1.75),"gemma3-12b":(24,2.5),"gemma3-27b":(28,1.0)}
VALUE = {"R":1,"W":-1,"X":0}
E2 = "The question may rest on a popular misconception or a false assumption. Give the factually correct answer, not the popular one."
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def strength(v):return "%g"%v
def check(ok, msg):
 if not ok: raise ValueError(msg)
def summarize(v):
 a=np.array(v, dtype=int); n=len(a)
 vals, fr=np.unique(a,return_counts=True)
 # Identical paired item bootstrap distribution over differences, independent random stream.
 draws=np.random.default_rng(20261119).multinomial(n,fr/n,size=4000) @ vals
 return (int(sum(a)), round(100*sum(a)/n,2), [round(100*z/n,2) for z in np.sort(draws)[[100,3899]]])
def compare_sets(task, states):
 for name in ("base","E","P","PE","half"):
  if name not in states:continue
  vals=states[name]; counts=collections.Counter(vals)
  print(task,name,"R W X",*(counts[x] for x in "RWX"),"accuracy percent", round(counts['R']*100/len(vals),2),"net",counts['R']-counts['W'])
 for a,b in (("E","base"),("P","base"),("PE","base"),("PE","P"),("PE","E"),("half","P"),("half","base"),("half","PE")):
  if a not in states or b not in states:continue
  sa,sb=states[a],states[b]
  vals=[int(x=="R")-int(y=="R") for x,y in zip(sa,sb)]
  netvals=[VALUE[x]-VALUE[y] for x,y in zip(sa,sb)]
  print(task,a,"minus",b,"correct-count, percentage points and paired interval",summarize(vals),"net difference",sum(netvals))
 if {"E","base","PE","P"}<=states.keys():
  s=list(zip(*(states[k] for k in ("PE","P","E","base"))))
  ds=[int(x=="R")-int(y=="R")-int(z=="R")+int(w=="R") for x,y,z,w in s]
  print(task,"matched-dose P by E interaction; correct-count, pp, interval",summarize(ds))
def mmlu(model, ids, seen, gold, ids_path, s):
 l,orig=MODELS[model]
 base=read(R/'full'/f'{model}.json')
 path=R/'full_h22'/f'{model}.json'
 if not path.exists():return
 files={'base': (base,'x')}
 mainkey=f'x:E{l}.w.p{strength(s)}'
 for loc in ('full','full_mid','full_high','full_low','full_l16','full_l16hi','full_l24','full_l28'):
  fp=R/loc/f'{model}.json'
  if not fp.exists():continue
  d=read(fp)
  if mainkey in d['arms']:files['E']=(d,mainkey); break
 he=read(path);files['P']=(he,'x_E2');files['PE']=(he,'x_E2:E%d.w.p%s'%(l,strength(s)))
 files['h22_x']=(he,'x')
 hp=R/'full_h22s'/f'{model}.json'
 if hp.exists():
  hs=read(hp); half='x_E2:E%d.w.p%s'%(l,strength(s/2))
  if half in hs['arms']:files['half']=(hs,half)
 out={}
 keep=[i for i,q in enumerate(ids) if q not in seen]
 for tag,(d,k) in files.items():
  check(d['model']==model and d['n']==14042 and d['ids_file_sha256']==sha(ids_path) and d['items_file_sha256']==item_sha['mmlu'] and not d.get('skipped'),(model,tag,'mmlu identity'))
  a=d['arms'][k];rows=a['rows'];check([r['id'] for r in rows]==ids,(k,'order'))
  vals=[]
  for r in rows:
   lp=r['logp']; check(set(lp)==set('ABCD') and all(np.isfinite(list(lp.values()))) and r['pick']==max('ABCD',key=lambda k:lp[k]) and gold(r['id'])==r['gold'],(model,k,'score'))
   vals.append('R' if r['pick']==r['gold'] else 'W')
  out[tag]=[vals[i] for i in keep]
  if tag in ('PE','half','E'):
   w,=a['writes']; check(w['layer']==l and w['kind']=='w' and w['beta']==(s/2 if tag=='half' else s) and a['positions']=='content',(model,k,'write'))
 print(model,'MMLU full h22 base-versus-archived correctness disagreements',sum(x!=y for x,y in zip(out.pop('h22_x'),out['base'])),flush=True)
 compare_sets(model+' MMLU',out)
def gsm(model, ids, gold, ids_path, s):
 l,_=MODELS[model]
 old=R/'gen_main/gsm8k'/model/'b8'; new=R/'gen_main/gsm8k_h22'/model/'b8'
 files={}
 for tag,pr,gamma,sources in [('base','neutral',None,[old]),('P','neutral_E2',None,[new]),('E','neutral',s,[new,old]),('PE','neutral_E2',s,[new]),('half','neutral_E2',s/2,[new])]:
  key=pr if gamma is None else f'{pr}--E{l}-w-p'+strength(gamma).replace('.','-')+'-at-prompt'
  arm=pr if gamma is None else f'{pr}:E{l}.w.p'+strength(gamma)
  fp=next((d/(key+'.json') for d in sources if (d/(key+'.json')).is_file()),None)
  if fp is None:continue
  obj=read(fp);c=obj['contract']; rows=obj['rows']
  check(obj['completed'] and c['model']==model and c['task']=='gsm8k' and c['split']=='all' and c['full_pass'] and c['batch_size']==8 and c['max_new_tokens']==512 and c['arm']==arm and c['base_prompt']==pr and c['parts']==obj['parts'] and c['positions']==('prompt' if gamma is not None else None),(model,key,"GSM identity"))
  check(c['ids_file_sha256']==sha(ids_path) and c['items_file_sha256']==item_sha['gsm8k'] and c['ids']==ids and [x['id'] for x in rows]==ids,(model,key,'GSM roster'))
  check(c['system_text']=='You are a helpful assistant.'+(' '+E2 if pr=='neutral_E2' else ''),(model,key,'GSM system prompt'))
  if gamma is not None:
   w,=c['parts']; actual=Path(c['model_config']['run_dir'])/'dirs'/f'E_L{l}.pt'
   check(w['direction_file_sha256']==sha(actual) and w['layer']==l and w['beta']==gamma and abs(gamma*w['gap']-w['amount'])<1e-6,(model,key,'GSM direction'))
  for r in rows:
   check(len(r['token_ids'])==r['n_tokens'] and 0<=r['n_tokens']<=512 and (r['finished'] or r['n_tokens']==512),(model,key,'GSM tokens'))
  parsed=[C.judge_row('gsm8k',r,gold(r['id'])) for r in rows]
  files[tag]=(rows,parsed)
 title=model+' GSM8K'
 compare_sets(title,{k:[x for x,_ in v[1]] for k,v in files.items()})
 for k,(rs,judged) in files.items():
  trunc=sum(not r['finished'] for r in rs)
  missing=collections.Counter(reason for x,reason in judged if x=='X')
  print(title,k,'unfinished',trunc,'score X reasons',dict(missing),'average emitted tokens',round(np.mean([r['n_tokens'] for r in rs]),1))
 if model=='qwen3.5-9b' and {'base','P','PE'}<=files.keys():
  print("Qwen9 before->after: correctness change split by finish status")
  for b,a in [('base','PE'),('P','PE'),('base','P'),('PE','half')]:
   if b not in files or a not in files:continue
   counts=collections.defaultdict(lambda: [0,0]); br,bs=files[b]; ar,as_=files[a]
   for x,y,v,w in zip(br,ar,bs,as_):
    key=(bool(x['finished']),bool(y['finished']))
    counts[key][0]+=1;counts[key][1]+=int(w[0]=='R')-int(v[0]=='R')
   print(b,'to',a,'(before finished,after finished): [items, change in correct]',dict(counts))
  # No semantic judgement here. Check how often an ungraded truncated row literally
  # has a gold final numeral under a last-number parser. May include intermediate numbers.
  for a,(rs,j) in files.items():
   extras=[]
   for row,(x,why) in zip(rs,j):
    if x!='X' or why!='truncated':continue
    parsed,_=gsm8k_parse(row['text'],True)
    extras.append(int(parsed is not None and abs(parsed-gold(row['id']))<1e-6))
   print('truncated cases only, last-number sensitivity',a,sum(extras),len(extras))
  for a,(rs,j) in files.items():
   bad=[row for row,(x,why) in zip(rs,j) if x=='X' and why=='truncated']
   keywords=re.compile(r'\b(?:misconception|false (?:assumption|premise)|trick (?:question|here)|misleading)\b',re.I)
   print('truncated replies explicitly using one of the stated misconception-related phrases',a,sum(bool(keywords.search(row['text'])) for row in bad),len(bad))
if __name__=='__main__':
 gsm_ids=R/'sets/gsm8k_1319_20261119.json'; m_ids=R/'sets/mmlu_14042_20261120.json'
 item_sha={k:sha(Path(v)) for k,v in C.DATA.items()}
 gq=read(gsm_ids)['ids'];mq=read(m_ids)['ids'];seen=set(read(R/'sets/mmlu_1000_20261119.json')['ids'])
 check(len(set(gq))==1319 and len(set(mq))==14042 and len(seen)==1000 and seen.issubset(mq),'frozen roster')
 golds={k:C.gold_of(k) for k in C.DATA}
 for model in MODELS:
  s=float(read(Path('Iteration_01/newsets/h22')/f'n1s_choice_{model}.json')['strength_dot'])
  gsm(model,gq,golds['gsm8k'],gsm_ids,s)
  mmlu(model,mq,seen,golds['mmlu'],m_ids,s)
