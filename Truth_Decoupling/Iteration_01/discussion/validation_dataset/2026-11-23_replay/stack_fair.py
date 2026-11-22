"""Read-only H/I audit. Run from Truth_Decoupling in scoring container. No newly generated answers or judgments."""
import sys,json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
sys.path[:0]=['Iteration_01/diag','Iteration_01/code/scripts/newsets','Iteration_01/code/scripts/review_checks']
import n1_on_sentence as N, qaqa_pilot_table as Q, tq_same_n_decomp as T
from dataclasses import dataclass
@dataclass(frozen=True)
class Cell:
    text:str
    issued:bool
    v:int

def cc(text,a,v):
    if not isinstance(text,str) or v not in (-1,0,1):raise ValueError('bad cell')
    return Cell(text.strip(),bool(a),int(v))

def raw_set(sp,tx,s,ids):
    def get(q,arm):return cc(tx.ec[q,arm],not sp.blank[q,arm],N.V[sp.st[q,arm]])
    return {q:[get(q,'baseline_x'),get(q,sp.cE(s)),*[get(q,sp.cEC(s,l)) for l in N.ps.LAD]] for q in ids}

def selected(row,cfg):
    p,e,*arms=row
    if e.issued or cfg[0]=='E':return e
    cap=cfg[2] if p.issued else cfg[3]
    if cap is None:return e
    for c in arms[:N.ps.LAD.index(cap)+1]:
        if c.issued:
            return e if p.issued and N.MP.ro.norm(c.text)==N.MP.ro.norm(p.text) else c
    return e

def fit(rows,s):
    sets=[('E',s)]+[('N1',s,x,y) for x in N.ML.CAPS for y in N.ML.CAPS if (x,y)!=(None,None)]
    return max(sets,key=lambda cfg:sum(selected(r,cfg).v for r in rows.values()))

def tq(m):
    display,l,main=N.A.KEY[m]; candidates=[]
    for s in (main,N.half(main)):
        phase=N.restricted(m,0); sp,tx=N.build(m,0,'x_E2',s,phase)
        assert list(sp.qs)==list(phase)
        rows=raw_set(sp,tx,s,phase); cfg=fit(rows,s)
        score=sum(selected(rs,cfg).v for rs in rows.values())
        candidates.append((score,-float(N.dot(s)),s,cfg))
    _,_,s,cfg=max(candidates); ch=json.load(open(N.CHOICE%m))
    if ch['model']!=m or ch['strength']!=s or tuple(ch['cfg'])!=cfg:raise ValueError('stored choice disagrees with tune')
    ids=N.restricted(m,1);sp,tx=N.build(m,1,'x_E2',s,ids);rows=raw_set(sp,tx,s,sp.qs)
    D=N.MP.MT.Data(display,1);work=D.m['d'][1];run=Path(D.m['ec'][1].split('/eval_extra/')[0]);dir_=run/'eval_extra'
    fs,fb=N.ps.load(work,'pro'); ft=N.fr.raw_texts(work); fkey='baseline_conf_E2'
    cdoc=json.loads((dir_/'confidence_prompts_test_conf_E2.json').read_bytes());conf=cdoc['targets']['prompt:conf_E2']['rows']
    if hashlib.sha256((run/'frozen_test.json').read_bytes()).hexdigest()!=cdoc['frozen_file_sha256'] or cdoc['partial'] or [r['pair_id'] for r in conf]!=cdoc['question_ids']:raise ValueError('wrong confidence source')
    conf={r['pair_id']:r for r in conf}
    sd=T.forced(str(run),l,main,ids)
    ws=N.MP.work(str(run),'test','pospromptdl%d'%l);ft2=N.fr.raw_texts(ws);fk2='E_L%d_conf-w-p%s'%(l,main)
    qrows={}
    for q,rs in rows.items():
        f=cc(ft[q,fkey],not fb[q,fkey],N.V[fs[q,fkey]])
        if hashlib.sha1(f.text.encode()).hexdigest()!=conf[q]['text_sha1']:raise ValueError('wrong score text')
        f2=cc(ft2[q,fk2],sd[q][1],sd[q][0])
        qrows[q]=(rs,selected(rs,cfg),f,conf[q]['mean_logp'],f2,sd[q][2],main==s)
    print('TQA verified tune',m,s,cfg)
    return qrows

def qq(m):
    base=Path(Q.S.NS);bank=base/'bank/qaqa_full_bank_20261122.jsonl';pilot=set(json.loads((base/'splits/qaqa_pilot_20261122.json').read_bytes())['ids'])
    ch=json.load(open(N.CHOICE%m));s=ch['strength_dot'];el,_,cl=Q.SET[m][:3]
    R=Q.Raw(m,'qaqafull',str(bank),['test_full570_h22n','test_full570_h22n1','test_full570_h22z'])
    R.gdir=base/'gen_prompt/h22/qaqafull'/m/'test'
    arms=['x_E2',f'x_E2:E{el}.w.p{s}']+[f'x_E2:E{el}.w.p{s}+C{cl}.w.m{k}' for k in ['0.25','0.5','1','2']]
    for a in arms:R.need(a)
    src=Q.Raw(m,'qaqafull',str(bank),['test_full570','test_full570_tuned','test_full570_strong'])
    main=Q.SET[m][1];fa='conf:E%d.w.p%s'%(el,main)
    for a in ('conf_E2',fa):src.need(a)
    conf=src.confidence('qaqafull','conf_E2');cf2=src.confidence('qaqafull',fa)
    if set(R.ids)!=set(src.ids) or len(pilot)!=100 or not pilot.issubset(R.ids):raise ValueError('roster')
    get=lambda b,a,q:cc(b.text(a,q),not b.blank(a,q),b.cell(a,q))
    rows={q:[get(R,a,q) for a in arms] for q in R.ids}
    cfg=fit({q:rows[q] for q in R.ids if q in pilot},ch['strength'])
    print('QQ verified tune',m,cfg)
    return {q:(rs,selected(rs,cfg),get(src,'conf_E2',q),conf[q],get(src,fa,q),cf2[q],main==s) for q,rs in rows.items() if q not in pilot}

def bounds(x,y,ids):
    xv=[x[q].v for q in ids if x[q].issued];yv=[y[q].v for q in ids if y[q].issued];n=min(len(xv),len(yv))
    xv.sort();yv.sort()
    return n,sum(xv[:n])-sum(yv[-n:]),sum(xv[-n:])-sum(yv[:n])
def run(label,data):
    ids=sorted(set.intersection(*(set(v) for v in data.values())))
    names=list(data)
    for subset in (names,names[:3]):
        common=sorted(set.intersection(*(set(data[m]) for m in subset)))
        by={};both={}
        for m in subset:
            rows=data[m]
            obs={q:rows[q][1] for q in common}; p={q:rows[q][0][0] for q in common}; e={q:rows[q][0][1] for q in common}
            sys_={'SN1':obs,'S0':p,'SE':e}
            for which,field,rank in [('forcedP',2,3),('forcedE',4,5)]:
                for strat in [False,True]:
                    want={q for q in common if not e[q].issued and obs[q].issued};take=set();lack=0
                    for origin in ((False,True) if strat else (None,)):
                        region=[q for q in common if not e[q].issued and (origin is None or p[q].issued==origin)]
                        quota=sum(q in want for q in region)
                        items=[q for q in region if rows[q][field].issued and rows[q][rank] is not None and (not strat or not p[q].issued or N.MP.ro.norm(rows[q][field].text)!=N.MP.ro.norm(p[q].text))]
                        if any(not math.isfinite(rows[q][rank]) for q in items):raise ValueError('invalid confidence')
                        items.sort(key=lambda q:-rows[q][rank])
                        take.update(items[:quota]);lack+=max(0,quota-len(items))
                    if lack: print('infeasible',label,m,which,strat,lack);continue
                    sys_[which+str(strat)]={q:(rows[q][field] if q in take else e[q]) for q in common}
            pat=Counter()
            for q in common:
                if obs[q].text==p[q].text and obs[q].v!=p[q].v:pat[(obs[q].v,p[q].v)]+=1
            print(label,m,'S0,SE,SN1',[(sum(u[q].v for q in common),sum(u[q].issued for q in common)) for u in (p,e,obs)],'S0=SN1 same text verdict conflict:',dict(pat),'S0/same-count bound',bounds(obs,p,common),'same dose forced E',rows[common[0]][6])
            by[m]=sys_
        for name in ['S0','SE','forcedPFalse','forcedPTrue','forcedEFalse','forcedETrue']:
            if not all(name in d for d in by.values()):continue
            vec=[]
            for q in common:
                row=[]
                for m,d in by.items():
                    aa,bb=d['SN1'][q],d[name][q]
                    row.append((aa.v-bb.v, (aa.v-bb.v) if aa.text!=bb.text else 0, (aa.v-bb.v) if aa.issued and bb.issued else 0, int(aa.issued and bb.issued)))
                vec.append(list(map(sum,zip(*row))))
            ci=N.ps.ci([r[0] for r in vec])
            print('RESULT',label,len(subset),len(common),'SN1-'+name,'raw',ci,'same text unified',sum(r[1] for r in vec),'jointly issued',sum(r[2] for r in vec),'pairs',sum(r[3] for r in vec))
        limits=[bounds(d['SN1'],d['S0'],common) for d in by.values()]; print('pooled limits SN1-S0',label,len(subset),list(map(sum,zip(*limits))))

def closest(label,data,names):
    import random
    ids=sorted(set.intersection(*(set(data[m]) for m in names)))
    n=len(ids); rng=random.Random(20261116);weights=np.array([np.bincount([rng.randrange(n) for _ in range(n)],minlength=n) for _ in range(4000)]+[np.ones(n,int)])
    for arm,fi,cf in [('forcedP',2,3),('forcedE',4,5)]:
        base=np.zeros(n,dtype=int);groups=[];k=0;keepers=0;shorts={}
        for m in names:
            rr=data[m];shorts[m]=[]
            for own in [True,False]:
                ix=[i for i,q in enumerate(ids) if (not rr[q][0][1].issued and (not rr[q][0][0].issued)==own)]
                rel=[i for i in ix if rr[ids[i]][1].issued]
                pool=[i for i in ix if rr[ids[i]][fi].issued and rr[ids[i]][cf]!=None and (own or N.MP.ro.norm(rr[ids[i]][fi].text)!=N.MP.ro.norm(rr[ids[i]][0][0].text))]
                keep=sorted(pool,key=lambda i:-rr[ids[i]][cf])[:len(rel)]
                for i in rel:
                    a,e=rr[ids[i]][1],rr[ids[i]][0][1];base[i]+=a.v-e.v
                for i in keep:
                    f,e=rr[ids[i]][fi],rr[ids[i]][0][1];base[i]-=f.v-e.v
                gap=len(rel)-len(keep)
                if gap:groups.append((gap,rel,np.array([rr[ids[i]][1].v-rr[ids[i]][0][1].v for i in rel])))
                shorts[m].append(gap);k+=len(keep)
            keepers+=sum(rr[q][0][1].issued for q in ids)
        lo=weights@base;hi=lo.copy()
        for gap,ix,sc in groups:
            effect=np.sort(weights[:,ix]*sc,axis=1);lo-=effect[:,-gap:].sum(1);hi-=effect[:,:gap].sum(1)
        print(label,','.join(names),arm,keepers+k,'samples/outer',lo[-1],hi[-1],np.sort(lo[:-1])[100],np.sort(hi[:-1])[3899],shorts,flush=True)
for l,read in [('TQA',tq),('QA2',qq)]:
    data={m:read(m) for m in N.ORDER}
    for ms in [list(data),list(data)[:3],[m for m in data if next(iter(data[m].values()))[6]]]:
        closest(l,data,ms)
