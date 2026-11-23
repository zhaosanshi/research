"""CPU only from Truth_Decoupling in scoring container. Replay J6/J7 from bound original scores and native answer records."""
from pathlib import Path
from collections import Counter
import hashlib, json, random, sys
import numpy as np
sys.path[:0]=["Iteration_01/diag","Iteration_01/code/scripts/newsets","Iteration_01/code/scripts/review_checks"]
import selfjudge_stack_pick as K
import selfjudge_stack as S
import selfjudge_probe as P
import selfjudge_two_answers as T
import main_table_prompt as MP

def margin_cache(model):
    M={}; tags={}
    for src in ('tj1_s0','tj1_conf','tj1_n1','st1_items','sp_items'):
        doc,meta,z=P.read_probe(model,source=src)
        if meta['batch_size']!=1 or meta['model']!=model: raise ValueError("wrong source")
        for i,h in enumerate(z['statement_sha1']):
            h=str(h);m={c:float(z[c+'__lpT'][i]-z[c+'__lpF'][i]) for c in meta['conditions']}
            if any(not np.isfinite(v) for v in m.values()) or h in M and m!=M[h]: raise ValueError('conflicting singleton')
            M[h]=m
            tags[h]=tags.get(h,0)+1
    return M

def decode(r,M):
    g,a,statement,neg,_,_=r
    return (g,a,M[P.sha1(statement)]["baseline"] if a else -np.inf,neg,statement)

def select(a,b):
    if a[1] and not b[1]:return a
    if b[1] and (not a[1] or a[4]==b[4] or a[2]<=b[2]):return b
    return a if a[1] else b

def z(r):
    return (0,False,-np.inf,False,None) if r[1] and r[2]<0 and not r[3] else r

def systems(d,M,unified):
    st,e,o,p=[decode(d[x],M) for x in ('St','E','O','S0')]
    if unified:
        g={}
        for x in (e,o,st,p):
            if x[1]:g.setdefault(x[4],x[0])
        def u(x): return (g[x[4]],*x[1:]) if x[1] else x
        st,e,o,p=map(u,(st,e,o,p))
    b=select(st,e);tr=st[1] and st[3] and e[1] and st[4]!=e[4]
    if not st[1] and e[1]: change='fill E'
    elif not e[1] and st[1]:change='fill stack'
    elif st[1] and e[1] and st[4]!=e[4]:change='choose both'
    else:change='same/default'
    return dict(St=st,E=e,O=o,P=p,B=b,Efirst=e if e[1] or not st[1] else st,Stfirst=st if st[1] else e,KeepE=b if e[1] else e,StackOnE=st if e[1] and st[1] else e,T=select(p,o),D=b if tr else st,
                Bz=z(b),Stz=z(st),Ez=z(e),Tz=z(select(p,o))),change

def vect(rows,ids,a,b):
    x=[rows[q][a] for q in ids]; y=[rows[q][b] for q in ids]
    nat=np.asarray([u[0]-v[0] for u,v in zip(x,y)])
    xs,ys=[x for x in range(len(x)) if rows[ids[x]][a][1]],[x for x in range(len(y)) if rows[ids[x]][b][1]]
    t=min(len(xs),len(ys)); A=np.array([s[0] for s in x]); B=np.array([s[0] for s in y])
    for v,data,who in ((A,x,xs),(B,y,ys)):
        drop=sorted(who,key=lambda i:(data[i][2],i))[:len(who)-t]
        v[drop]=0
    return t,nat,A-B

from functools import lru_cache
@lru_cache(maxsize=12)
def weights(n,seed):
    rand=random.Random(seed)
    w=np.array([np.bincount([rand.randrange(n) for _ in range(n)],minlength=n) for i in range(4000)])
    return w

def jci(rows,seed):
    a=np.sort(weights(len(rows),seed)@rows);return (int(rows.sum()),int(a[100]),int(a[3899]))

def main():
    modes=list(K.ORDER);M={m:margin_cache(m) for m in modes}
    sets={m:K.all_sets(m) for m in modes}
    for unified in (False,True):
        for key in ('TruthfulQA test','qaqa','triviaqa test','triviaqa test2','popqa test','popqa test2'):
            common=sorted(set.intersection(*(set(sets[m][key][0]) for m in modes)))
            groups=[('all',common)]
            if key=='qaqa':
                raw=sets[modes[0]][key]; info=raw[2]
                if any(info[q]!=sets[m][key][2][q] for m in modes for q in common): raise ValueError('qtype differs')
                qs=[q for q in raw[0] if q in set(common) and not info[q][1]]
                assert len(qs)==470
                groups=[('all',qs)]+[(t,[q for q in qs if info[q][0]==t]) for t in K.SP.QTYPES]
            for typ,ids in groups:
                obs={m:{} for m in modes}
                splits=Counter(); disputed=Counter()
                for m in modes:
                    for q in ids:
                        d=sets[m][key][1][q]; row,ch=systems(d,M[m],unified);obs[m][q]=row
                        splits['over_St',ch]+=row['B'][0]-row['St'][0]
                        splits['over_E',ch]+=row['B'][0]-row['E'][0]
                        splits['count',ch]+=1
                        if row["St"][1] and row["E"][1] and row["St"][4]==row["E"][4] and d["St"][0]!=d["E"][0]:disputed['same-text conflicting']+=1
                print('J7',key,typ,'common',len(ids),'unified',unified)
                for a,b in (('B','St'),('B','E'),('B','T'),('KeepE','E'),('KeepE','StackOnE'),('B','Efirst'),('B','Stfirst'),('D','B'),('Bz','Ez'),('Bz','Tz')):
                    vs=[vect(obs[m],ids,a,b) for m in modes]
                    print(a+'-'+b,'natural / equal count / equal',jci(sum(v[1] for v in vs),20261116 if key.startswith('TruthfulQA') else 20261119),sum(v[0] for v in vs),jci(sum(v[2] for v in vs),20261116 if key.startswith('TruthfulQA') else 20261119))
                print('counts and changes',dict(splits),dict(disputed),flush=True)

if __name__=="__main__": main()
