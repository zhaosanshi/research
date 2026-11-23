"""CPU-only audit from Truth_Decoupling in the scoring container. Prints aggregate counts, no answer text.
Keep the plain honesty prompt's original answer where submitted. Rebuild E, EC and C-only fill candidates from native cells.
"""
import collections
from pathlib import Path
import random
import json
import sys

import numpy as np

sys.path[:0]=['Iteration_01/diag','Iteration_01/code/scripts/newsets','Iteration_01/code/scripts/review_checks']
import n1_on_sentence as N
import n1_on_sentence2 as N2
import qaqa_pilot_table as Q
import selfjudge_probe as SP
import selfjudge_stack_pick as SKP
import fill_only as F

MODELS=tuple(N.ORDER)
LAD=tuple(N.ps.LAD)
ROOT=Path(Q.S.NS)
BANK=str(ROOT/'bank/qaqa_full_bank_20261122.jsonl')


def cell(text,v,issued):
    if not isinstance(text,str) or v not in (-1,0,1):
        raise ValueError('invalid answer cell')
    return text.strip(),int(v),bool(issued)


def pick_ec(p,e,ladder,caps):
    if e[2]:
        return e
    cap = caps[0] if p[2] else caps[1]
    if cap is None:
        return e
    for v in ladder[:LAD.index(cap)+1]:
        if v[2]:
            if p[2] and N.MP.ro.norm(p[0])==N.MP.ro.norm(v[0]):
                return e
            return v
    return e


def tq_rows(model):
    choice=N2.choice(model)
    strength=choice['strength']
    cap=F.sc_cap(model)
    if cap!=F.REF_SCCAP[model]:
        raise ValueError('C tune cap mismatch')
    out={}
    for i,split in [(0,'tune'),(1,'test')]:
        ids=N.restricted(model,i)
        sp,tx=N.build(model,i,'x_E2',strength,ids)
        if len(set(ids))!=len(ids) or set(ids)!=set(sp.qs):
            raise ValueError((model,split,'missing native roster'))
        if split=='tune':
            cfg=N.ML.n1_pick(sp,tx,strength)
            if list(cfg)!=choice['cfg']:
                raise ValueError((model,'EC tune cap mismatch'))
        raw_c=N2.c_cells(model,i,ids)
        words_c=F.c_texts(model,i)
        step=SP.read_json(Path(SP.OUT)/model/'items.json')['items']
        question={x['id']:x['question'] for x in step if x['split']==split}
        D=N.MP.MT.Data(N.A.KEY[model][0],i)
        ds,db=N.ps.load(D.m['d'][i],'pro')
        txt=N.fr.raw_texts(D.m['d'][i])
        def get(q,k):
            return cell(tx.ec[q,k],N.V[sp.st[q,k]],not sp.blank[q,k])
        rows={}
        for q in ids:
            p=get(q,'baseline_x');e=get(q,sp.cE(strength))
            cs=[get(q,sp.cEC(strength,k)) for k in LAD]
            ec=pick_ec(p,e,cs,cfg[2:] if cfg[0]=='N1' else (None,None))
            c=p
            if not p[2]:
                for k in LAD[:LAD.index(cap)+1]:
                    if raw_c[k][q][1]:
                        c=cell(words_c[q,k],raw_c[k][q][0],True)
                        break
            fp=cell(txt[q,'baseline_conf_E2'],N.V[ds[q,'baseline_conf_E2']],not db[q,'baseline_conf_E2'])
            rows[q]=(question[q],{'S':p,'E':e,'EC':ec,'C':c,'prompt':fp})
        out[split]=rows
    return out


def qa_rows(model):
    cand=N2.choice(model);s=cand['strength_dot']
    base=Q.Raw(model,'qaqafull',BANK,('test_full570','test_full570_tuned','test_full570_strong'))
    raw=Q.Raw(model,'qaqafull',BANK,('test_full570_h22n','test_full570_h22n1','test_full570_h22z'))
    raw.gdir=ROOT/'gen_prompt/h22/qaqafull'/model/'test'
    rows=[json.loads(l) for l in Path(BANK).read_text().splitlines()]
    ids=raw.ids; pilot=set(json.loads(Path(SP.QAQA_PILOT).read_bytes())['ids'])
    if ids!=base.ids or ids!=[b['id'] for b in rows] or len(set(ids))!=570 or len(pilot)!=100 or not pilot.issubset(ids):
        raise ValueError('bad evaluation roster')
    e=raw.e(s);prefix=e.replace('x:','x_E2:')
    arms=['x_E2',prefix,*[raw.ec(s,k).replace('x:','x_E2:') for k in Q.LAD]]
    for a in arms:raw.need(a)
    base.need('conf_E2')
    get=lambda r,a,q:cell(r.text(a,q),r.cell(a,q),not r.blank(a,q))
    data={q:[get(raw,a,q) for a in arms] for q in ids}
    # Native candidate subsets; choose by tune-only net, existing order breaks ties.
    choices=[(a,b) for a in Q.CAPS for b in Q.CAPS]
    def as_cap(c):return None if c is None else str(c).replace('.','-')
    def pick(q,duo):
        p,e,*c=data[q]
        return pick_ec(p,e,c,tuple(map(as_cap,duo)))
    # This grid includes the both-None E-only policy.
    best=max(choices,key=lambda caps:sum(pick(q,caps)[1] for q in ids if q in pilot))
    out={'tune':{},'test':{}}
    for q,meta in zip(ids,rows):
        p,e,*_=data[q]
        out['tune' if q in pilot else 'test'][q]=(meta['question'],{'S':p,'E':e,'EC':pick(q,best),'prompt':get(base,'conf_E2',q)})
    return out


def get_scores(model):
    # SKP.lookup does not load prospective FO data. Check the explicit new source if already present.
    result=SKP.lookup(model)
    path=Path(SP.OUT)/model/'fo_items_probe.json'
    if path.is_file():
        _,meta,z=SP.read_probe(model,source='fo_items')
        if meta.get('batch_size')!=1:
            raise ValueError('bad new-source batch')
        for j,k in enumerate(z['statement_sha1']):
            c={v:float(z[v+'__margin'][j]) for v in meta['conditions']}
            if str(k) in result and result[str(k)]!=c:
                raise ValueError('non-reproducible margin')
            result[str(k)]=c
    return {k:v['baseline'] for k,v in result.items()}


def dropped_range(items, k, weights, honor_scores):
    """Observed missing ranks: enumerate how many unknown rows enter the dropped prefix; optimize over which ones.
    Set honor_scores=False for an envelope over all score orders. Each column is indexed by its question.
    On a bootstrap draw extrema range over every admissible fixed full-cohort selection.
    """
    if not k:
        z=np.zeros(len(weights),int)
        return z,z
    known=sorted((r for r in items if honor_scores and r[2] is not None),key=lambda r:(r[2],r[0]))
    missing=[r for r in items if not honor_scores or r[2] is None]
    def weighted(rows):
        if not rows:return np.empty((len(weights),0),int)
        return weights[:,[r[0] for r in rows]]*np.asarray([r[1] for r in rows])
    def prefixes(a):
        return np.column_stack((np.zeros(len(weights),int),np.cumsum(a,axis=1)))
    fixed=prefixes(weighted(known)[:,:k])
    unknown=np.sort(weighted(missing),axis=1)
    low,high=prefixes(unknown[:,:k]),prefixes(unknown[:,::-1][:,:k])
    choices=range(max(0,k-len(known)),min(len(missing),k)+1)
    return (np.min(np.stack([fixed[:,k-j]+low[:,j] for j in choices]),axis=0),
            np.max(np.stack([fixed[:,k-j]+high[:,j] for j in choices]),axis=0))


def dynamic_range(items, needed, weights, scores):
    """Extrema of the sum removed if counts are rematched inside each draw.
    For every permissible number of unknown scores before the cutoff, the rest
    is the cheapest known-score prefix. Unknown answers may take any order.
    """
    known=sorted((x for x in items if scores and x[2] is not None),key=lambda x:(x[2],x[0]))
    other=[x for x in items if not scores or x[2] is None]
    size=len(weights)
    counts=np.array([weights[:,[x[0] for x in other if x[1]==v]].sum(axis=1)
                     for v in (-1,0,1)])
    known_n=np.array([weights[:,x[0]] for x in known],dtype=int).T if known else np.empty((size,0),int)
    # Build the achievable prefix sums at every possible cutoff, vectorized over draws.
    maxdrop=int(needed.max())
    before=np.zeros(size,dtype=int); prefix=np.zeros((size,maxdrop+1),dtype=int)
    levels=np.arange(1,maxdrop+1)
    for col,x in enumerate(known):
        previous=before.copy(); before+=known_n[:,col]
        if len(levels):
            added=np.clip(levels[None,:]-previous[:,None],0,known_n[:,col,None])
            prefix[:,1:]+=added*x[1]
    lo=np.full(size,10**6,dtype=int);hi=-lo
    for nk in range(maxdrop+1):
        nu=needed-nk; valid=(nu>=0)&(nu<=counts.sum(axis=0))&(nk<=before)
        if not valid.any():continue
        min_unknown=-np.minimum(nu,counts[0])+np.maximum(0,nu-counts[0]-counts[1])
        max_unknown=np.minimum(nu,counts[2])-np.maximum(0,nu-counts[2]-counts[1])
        lo[valid]=np.minimum(lo[valid],(prefix[:,nk]+min_unknown)[valid])
        hi[valid]=np.maximum(hi[valid],(prefix[:,nk]+max_unknown)[valid])
    if (lo>hi).any():raise ValueError("no matching allocation")
    return lo,hi


def replay(label, banks, scores, ids, arms, harmonize=False):
    n=len(ids);rng=random.Random(20261116 if label.startswith('TQA') else 20261119)
    W=np.array([np.ones(n,dtype=int)]+[np.bincount([rng.randrange(n) for _ in range(n)],minlength=n) for _ in range(4000)],dtype=int)
    sets={}
    for model,rows in banks.items():
        views={a:[] for a in arms}
        for j,q in enumerate(ids):
            question,cells=rows[q]
            s=cells['S']
            votes={}
            for a in ('S','E','EC','C','prompt'):
                if a in cells:votes.setdefault(cells[a][0],cells[a][1])
            for a in arms:
                c=s if s[2] or a=='S' else cells[a]
                text,net,issued=c
                if harmonize:net=votes[text]
                sc=scores[model].get(SP.sha1(SP.statement(question,text))) if issued and not s[2] else None
                views[a].append((j,net,sc,issued,not s[2]))
        sets[model]=views
    print(label,'models',len(banks),'questions',n,'common verdicts',harmonize)
    per_model=list(sets)
    for a in arms:
        counts=[];hits=collections.Counter(); ds=np.zeros(n,int)
        for model,v in sets.items():
            counts.append(sum(r[3] and r[4] for r in v[a]))
            ds+=np.array([r[1]-s[1] for r,s in zip(v[a],v['S'])])
            for r,s in zip(v[a],v['S']):
                if r[3] and r[4]:
                    hits[r[1]]+=1
        if a!='S':
            bs=np.sort((W@ds)[1:]);print(a,'fills',counts,'positive/wrong/other',[hits[x] for x in [1,-1,0]],'net improvement vs S',int(sum(ds)),[int(bs[100]),int(bs[3899])])
    for candidates in ([('EC','E'),('EC','C'),('EC','E','C'),('EC','prompt')] if 'C' in arms else [('EC','E'),('EC','prompt')]):
        lower={a:np.zeros(len(W),int) for a in candidates};upper={a:v.copy() for a,v in lower.items()}
        upper_any={a:v.copy() for a,v in lower.items()};lower_any={a:v.copy() for a,v in lower.items()}
        target=[]; miss=[]; fixed_natural={a:0 for a in candidates}
        relaxed_lo={a:np.zeros(len(W),int) for a in candidates};relaxed_hi={a:v.copy() for a,v in relaxed_lo.items()}
        for model,v in sets.items():
            all_fills={a:[r for r in v[a] if r[3] and r[4]] for a in candidates}
            n_keep=min(map(len,all_fills.values()));target.append(n_keep);miss.append({})
            counts_by_draw={a:W[:,[x[0] for x in items]].sum(axis=1) for a,items in all_fills.items()}
            min_draw=np.min(np.stack(list(counts_by_draw.values())),axis=0)
            for a,issued in all_fills.items():
                before=W@np.array([r[1]-s[1] for r,s in zip(v[a],v['S'])])
                trim=len(issued)-n_keep
                miss[-1][a]=sum(x[2] is None for x in issued) if trim else 0
                fixed_natural[a]+=int(before[0])
                lo,hi=dropped_range(issued,trim,W,True); la,ha=dropped_range(issued,trim,W,False)
                lower[a]+=before-hi;upper[a]+=before-lo;lower_any[a]+=before-ha;upper_any[a]+=before-la
                if not harmonize and label.endswith("test") and candidates==("EC","C"):
                    dl,dh=dynamic_range(issued,counts_by_draw[a]-min_draw,W,True)
                    relaxed_lo[a]+=before-dh;relaxed_hi[a]+=before-dl
        print('shared arms',candidates,'min fill quota per model',target,'matched fills',sum(target),'missing when cut',miss)
        a=candidates[0]
        for b in candidates[1:]:
            for mode,l,u in [('known ordering',lower,upper),('any ordering',lower_any,upper_any)]:
                ll=l[a]-u[b];hh=u[a]-l[b]
                print(' ',a,'minus',b,mode,'sample',ll[0],hh[0],'paired outer',np.sort(ll[1:])[100],np.sort(hh[1:])[3899])
            if not harmonize and label.endswith("test") and candidates==("EC","C"):
                lower_rematch = relaxed_lo[a]-relaxed_hi[b]
                upper_rematch = relaxed_hi[a]-relaxed_lo[b]
                print(' ',a,'minus',b,'rematch per model in each paired draw; known order exact sample and outer',
                      lower_rematch[0],upper_rematch[0],np.sort(lower_rematch[1:])[100],np.sort(upper_rematch[1:])[3899])
    print('',flush=True)


if __name__=='__main__':
    scores={m:get_scores(m) for m in MODELS}
    for task,loader in [('TQA',tq_rows),('QA2',qa_rows)]:
        banks={m:loader(m) for m in MODELS}
        for split in ('test','tune'):
            part={m:dat[split] for m,dat in banks.items()}; common=sorted(set.intersection(*(set(dat) for dat in part.values())))
            if task=='QA2':common=[q for q in next(iter(part.values())) if q in set(common)]
            assert len(common)==({'TQA':{'test':520,'tune':192},'QA2':{'test':470,'tune':100}}[task][split])
            arms=['S','E','EC','prompt']+(['C'] if task=='TQA' else [])
            for canonical in (False,True) if split=='test' else (False,):
                replay(task+' '+split,part,scores,common,arms,canonical)
