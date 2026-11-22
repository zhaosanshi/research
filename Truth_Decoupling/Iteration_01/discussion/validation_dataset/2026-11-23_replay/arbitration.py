"""CPU-only replay from Truth_Decoupling. No external judge call or generated answer."""
import sys,random,hashlib,collections,itertools
from pathlib import Path
import numpy as np
sys.path[:0]=["Iteration_01/diag","Iteration_01/code/scripts/newsets"]
import selfjudge_probe as P
P.SOURCES.update(('tj1_s0','tj1_conf','tj1_n1','tj1_chk','tj_s0','tj_conf'))
MODELS=('llama3.1-8b','qwen3.5-9b','qwen3.8-27b','gemma3-12b','gemma3-27b')

def source(model,key):
    doc,meta,z=P.read_probe(model,source=key)
    conds=tuple(meta['conditions']); is_one=key.startswith('tj1_')
    if is_one and (meta.get('batch_size')!=1 or meta.get('path_identity_batch8_first')!=64):
        raise ValueError('wrong singleton calibration')
    names=set()
    ans=[x for x in doc['items'] if x['answered']]
    if key=='tj1_chk' and (len(ans)!=16 or set(meta['check_b'])!=set(conds) or
             any(meta['check_b'][c][k]!=0 for c in conds for k in ('second_pass_max_abs_diff','shuffled_pass_max_abs_diff'))):
        raise ValueError("incomplete repeated checks")
    idx={r['id']:j for j,r in enumerate(ans)}
    out={}
    for x in doc['items']:
        s=x['split']; q=x['id'].split('|',1)[-1]
        if s!='test':continue
        v=x.get('value', P.V.get(x.get('state')))
        if q in out or v not in (-1,0,1):raise ValueError('bad item')
        if x['answered']:
            j=idx[x['id']]; h=str(z['statement_sha1'][j])
            if h!=P.sha1(P.statement(x['question'],x['answer'])):raise ValueError('bad statement')
            mg={c:float(z[c+'__lpT'][j]-z[c+'__lpF'][j]) for c in conds}
        else:
            h=P.sha1(P.statement(x['question'],x['answer']));mg={}
        out[q]=(v,bool(x['answered']),h,mg,x['neg'])
    return out,doc,meta

def load(model):
    S=source(model,'tj1_s0')[0]; T=source(model,'tj1_conf')[0]; N=source(model,'tj1_n1')[0]
    _,doc,_=source(model,'')
    source(model,'tj1_chk')
    rows={}
    for x in doc['items']:
        q=x['id']
        if x['split']!='test' or q not in S or q not in T: continue
        n=N.get(q,(P.V[x['n1_state']],False,None,{},False))
        if n[:2]!=(P.V[x['n1_state']],bool(x['n1_answered'])):raise ValueError('wrong candidate')
        rows[q]=[S[q],n,T[q]]
    return rows

def evaluate(row,c,order):
    S,N,T=row
    def choose(a,b):
        if a[1] and (not b[1] or a[2]==b[2] or not (b[3][c]>a[3][c])): return a
        return b
    def z(x):
        if x[1] and not x[4] and x[3][c]<0: return (0,False,x[2],x[3],x[4])
        return x
    A=choose(S,N); B=choose(S,T); fill=S if S[1] else N
    if order:
        consensus={}
        for i in order:
            y=row[i]
            if y[2] is not None:consensus.setdefault(y[2],y[0])
        harmon=lambda x: (consensus.get(x[2],x[0]),*x[1:])
        S,N,T,A,B,fill=map(harmon,(S,N,T,A,B,fill))
    allv=dict(S=S,N=N,T=T,A=A,B=B,fill=fill)
    for k,x in list(allv.items()):allv[k+'z']=z(x)
    return allv

def score(rows,ids,c,order):
    data=[evaluate(rows[q],c,order) for q in ids]
    pairs=[('A','S'),('A','fill'),('A','B'),('Az','Sz'),('Az','fillz'),('Az','Bz')]
    result={}
    for a,b in pairs:
        x=[r[a] for r in data];y=[r[b] for r in data]
        va=np.array([r[0] for r in x]);vb=np.array([r[0] for r in y])
        cana=sorted([i for i in range(len(x)) if x[i][1]],key=lambda i:x[i][3][c]);canb=sorted([i for i in range(len(y)) if y[i][1]],key=lambda i:y[i][3][c])
        result[a,b,'natural']=va-vb
        if len(cana)>len(canb):va[cana[:len(cana)-len(canb)]]=0
        elif len(canb)>len(cana):vb[canb[:len(canb)-len(cana)]]=0
        result[a,b,'equal']=va-vb
    return result

def main():
    data={m:load(m) for m in MODELS}
    ids=sorted(set.intersection(*(set(v) for v in data.values()))); assert len(ids)==520
    rng=random.Random(20261116); n=len(ids)
    W=np.array([np.bincount([rng.randrange(n) for _ in range(n)],minlength=n) for _ in range(4000)])
    for order in (None,(0,1,2),(2,1,0)):
        allr=[score(data[m],ids,'baseline',order) for m in MODELS]
        for key in allr[0]:
            v=sum([r[key] for r in allr]);draws=np.sort(W@v)
            print('J',order,key,sum(v),int(draws[100]),int(draws[3899]),[int(sum(r[key])) for r in allr])
    for model in MODELS:
        dup=collections.Counter();cross=collections.defaultdict(list)
        overlap=collections.Counter()
        for q in ids:
            S,N,T=data[model][q]
            for tag,x in zip('SNT',(S,N,T)):
                if x[1]:cross[q,x[2]].append((tag,x[0],x[3]['baseline']))
            if S[1] and N[1] and S[2]!=N[2] and S[0]*N[0]==-1:
                if S[0]==1:overlap['S_correct']+=1
                if N[0]==1:overlap['N_correct']+=1
                if N[3]['baseline']>S[3]['baseline']:overlap['saved' if N[0]==1 else 'spoiled']+=1
        for k,listv in cross.items():
            if len(listv)>1:
                dup['dup']+=1;dup['margin different']+=len({x[2] for x in listv})>1
                dup['verdict different']+=len({x[1] for x in listv})>1
        print(model,dict(overlap),dict(dup))

def common_target():
    data={m:load(m) for m in MODELS}
    ids=sorted(set.intersection(*(set(v) for v in data.values())))
    n=len(ids);rng=random.Random(20261116)
    W=np.array([np.bincount([rng.randrange(n) for _ in range(n)],minlength=n) for _ in range(4000)])
    for control in ('S','fill','B'):
        items=[]
        for m in MODELS:
            d=[evaluate(data[m][q],'baseline',None) for q in ids]
            def top(arm,k):
                issued=[i for i,r in enumerate(d) if r[arm][1]]
                if len(issued)<k: raise ValueError('unreachable')
                ranked=sorted(issued,key=lambda i:d[i][arm][3]['baseline'])
                omit=set(ranked[:len(issued)-k])
                return np.array([0 if i in omit else r[arm][0] for i,r in enumerate(d)])
            target=min(sum(r[a][1] for r in d) for a in ('Az',control+'z'))
            bare=top('A',target)-top(control,target)
            shared=top('Az',target)-top(control+'z',target)
            items.append((target,bare,shared,top('Az',target)-top('A',target),top(control+'z',target)-top(control,target)))
        for j,name in ((1,'select top K with self-score on BOTH sides (no exemption)'),(2,'apply exempting Z to BOTH sides then trim to K')):
            v=sum(x[j] for x in items);bs=np.sort(W@v)
            print(control,sum(x[0] for x in items),name,int(v.sum()),int(bs[100]),int(bs[3899]))
        print(control,'Z vs top K for A and control',sum(sum(x[3]) for x in items),sum(sum(x[4]) for x in items),'per-model',[(sum(x[3]),sum(x[4])) for x in items]); v=sum(x[2]-x[1] for x in items);bs=np.sort(W@v)
        print(control,'difference between the two variants',int(v.sum()),int(bs[100]),int(bs[3899]))


if __name__=='__main__':
    main()
    common_target()
