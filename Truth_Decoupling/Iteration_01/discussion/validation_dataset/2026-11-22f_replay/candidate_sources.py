"""Run in the scoring container at the Truth_Decoupling root. Read-only; prints no question text.
'right' requires an issued answer and an R verdict. Fixed candidates are the first C rung already published (0.25)
and the original must-answer E at its main-table strength. Missing passes propagate, never count as failures.
"""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,"Iteration_01/diag")
import tq_same_n_decomp as T
import qaqa_pilot_table as Q

BANK=Path(Q.S.NS)/'bank/qaqa_full_bank_20261122.jsonl'
WORK=('test_full570','test_full570_tuned','test_full570_strong')
def cell(text,issued,v):
    if not isinstance(text,str) or int(v) not in (-1,0,1): raise ValueError('bad record')
    return (hashlib.sha256(text.strip().encode()).hexdigest(),bool(issued),int(v))
def qq(model):
    raw=Q.Raw(model,'qaqafull',str(BANK),WORK);s=Q.SET[model][1];e=raw.e(s)
    arms=[e,'x','conf','conf:'+e.split(':',1)[1]]+[raw.ec(s,k) for k in Q.LAD]
    for a in arms: raw.need(a)
    skips=set(json.loads((Path(Q.S.NS)/'splits/qaqa_pilot_20261122.json').read_bytes())['ids'])
    if len(raw.ids)!=570 or len(set(raw.ids))!=570 or len(skips)!=100 or not skips.issubset(raw.ids): raise ValueError('raw roster')
    result={'dev':{},'test':{}}
    for q in raw.ids:
        cs=[cell(raw.text(a,q),not raw.blank(a,q),raw.cell(a,q)) for a in arms]
        result['dev' if q in skips else 'test'][q]=(raw.qtype[q],cs)
    return result
def tq(model):
    display,l,s=T.A.KEY[model]
    src,=[x for x in T.ML.sources(display)[2] if x[0]==l]
    result={}
    for i,part in enumerate(('dev','test')):
        split=('tune','test')[i]
        batch,err=T.MP.prompt_split(display,l,i,s)
        if err: raise ValueError(err)
        sp,tx,_,ids=batch
        ids=[q for q in ids if q in set(src[1+2*i].qs)]
        if len(set(ids))!=len(ids): raise ValueError('duplicate id')
        dat,run,_=T.MP.usual(display,i)
        w=dat.m['d'][i]
        raw_s,raw_b=T.ps.load(w,"pro");raw_t=T.MP.fr.raw_texts(w)
        fcond=f'E_L{l}_conf-w-p{s}'; fw=T.MP.work(run,split,'pospromptdl%d'%l)
        ftest=None
        if T.MP.has(fw):
            fs,fb=T.ps.load(fw,'pro');ft=T.MP.fr.raw_texts(fw)
            if i==1: ftest=T.forced(run,l,s,ids)
        else: fs,fb,ft={},{},{}
        out={}
        for q in ids:
            def p(c): return cell(tx.ec[q,c],not sp.blank[q,c],T.V[sp.st[q,c]])
            def r(st,bl,txt,c):
                if st.get((q,c),'?')=='?': return None
                return cell(txt[q,c],not bl[q,c],T.V[st[q,c]])
            forced=r(fs,fb,ft,fcond)
            if ftest is not None and (forced is None or forced[1:]!=(ftest[q][1],ftest[q][0])): raise ValueError('forced identity')
            out[q]=('-', [p(sp.cE(s)),p('baseline_x'),r(raw_s,raw_b,raw_t,'baseline_conf'),forced]+[p(sp.cEC(s,k)) for k in T.ps.LAD])
        result[part]=out
    return result

def features(c,k=4):
    # c[0]=E; c[1]=original; c[2]=forced original; c[3]=forced E; 4: are the four EC rungs.
    bucket='E-blank' if not c[0][1] else 'E-wrong' if c[0][2]==-1 else 'other'
    if bucket=='other':return None
    base=c[:4+k]
    for required in (0,1,2,*range(4,4+k)):
        if base[required] is None:raise ValueError('insufficient coverage')
    values=collections.defaultdict(set)
    prior={}
    for obj in base:
        if obj is None:continue
        key,issued,v=obj
        values[key].add(v==1)
        prior.setdefault(key,v==1)
    uncertain=[key for key,hits in values.items() if len(hits)>1]
    def indicators(verdict):
        right=lambda obj: obj is not None and obj[1] and verdict[obj[0]]
        cc=any(map(right,base[4:]));r0=right(base[1]);f0=right(base[2])
        out=[int(cc),int(r0 or f0),int(cc and not(r0 or f0))]
        if base[3] is None: return out
        f=right(base[3]); rival=f or r0 or f0
        return out+[int(cc and not f),int(f and not cc),int(cc and f),int(not cc and not f),int(cc and not rival),int(rival and not cc)]
    prev=indicators(prior)
    options=[indicators({**prior,**dict(zip(uncertain,votes))}) for votes in itertools.product((False,True),repeat=len(uncertain))]
    return bucket,prev,[min(i) for i in zip(*options)],[max(i) for i in zip(*options)],int(bool(uncertain))

def report(data,task):
    for part in ('dev','test'):
        roster=sorted(set.intersection(*(set(x[part]) for x in data.values())))
        print('roster',task,part,len(roster),flush=True)
        for k in (1,4):
            totals={}
            for m in T.ORDER:
                per=collections.defaultdict(list)
                for q in roster:
                    typ,c=data[m][part][q]
                    f=features(c,k)
                    if f:
                        b,*f=f
                        for t in ('all',typ) if typ!='-' else ('all',):
                            per[b,t].append(f)
                for key,rows in per.items():
                    totals.setdefault(key,[]).extend(rows)
                    if key[1]=='all':
                        print('by-model',task,part,m,k,key[0],len(rows),*[sum(x) for x in zip(*(r[0] for r in rows))])
            for key,rows in sorted(totals.items()):
                lengths={len(r[0]) for r in rows}
                if len(lengths)!=1: raise ValueError('some arms missing on only part of roster')
                sums=lambda n: [sum(x) for x in zip(*(r[n] for r in rows))]
                print('pooled',task,part,k,*key,'total',len(rows),'harmonized',sums(0),'lower',sums(1),'upper',sums(2),'disputed_rows',sum(r[3] for r in rows),flush=True)

def selected(task,data):
    # One C attempt per rerun, dose chosen only from its own development questions.
    # No correctness label from the test bank chooses a dose.
    common={p:sorted(set.intersection(*(set(d[p]) for d in data.values()))) for p in ('dev','test')}
    print('selected dose: maximum development net on whole roster / maximum number of issued RIGHT on E blanks; ties lowest dose')
    for rule in ('dev_net_whole_roster','dev_right_among_blanks'):
        doses={}
        for m in T.ORDER:
            rows=data[m]['dev']
            cells=[rows[q][1] for q in rows]
            # Use the model's native tune roster, not the common roster, to select its own dose.
            if task=='qq':
                fit=Q.Raw(m,'qaqa',Q.TUNE_BANK,['test_tune100']);s=Q.SET[m][1];e=fit.e(s)
                arms=[e]+[fit.ec(s,k) for k in Q.LAD]
                for arm in arms:fit.need(arm)
                if set(fit.ids)!=set(rows):raise ValueError('pilot and 570-run selection rosters disagree')
                cells=[]
                for q in fit.ids:
                    cc=[cell(fit.text(a,q),not fit.blank(a,q),fit.cell(a,q)) for a in arms]
                    cells.append([cc[0],None,None,None,*cc[1:]])
            values=[sum(c[j][2] for c in cells) if rule=='dev_net_whole_roster'
                    else sum((not c[0][1]) and c[j][1] and c[j][2]==1 for c in cells) for j in range(4,8)]
            doses[m]=4+max(range(4),key=lambda j:values[j])
            print(task,rule,m,'rung',doses[m]-3,'dev_values',values)
        for part,ids in common.items():
            for b in ('E-blank','E-wrong'):
                rows=[]
                for q in ids:
                    total=[]
                    for m in T.ORDER:
                        c=data[m][part][q][1]
                        one=c[:4]+[c[doses[m]]]
                        f=features(one,1)
                        if f and f[0]==b:total.append(f[1])
                    dims=9 if part=='test' or task=='qq' else 3
                    rows.append([sum(t[j] for t in total) for j in range(dims)])
                sums=[sum(x) for x in zip(*rows)]
                if len(sums)==9:
                    print(task,rule,part,b,'counts',sums,'Cexclusive-minus-forcedexclusive',T.ps.ci([r[3]-r[4] for r in rows]),
                          'Cexclusive-minus-other3exclusive',T.ps.ci([r[7]-r[8] for r in rows]))
                else:print(task,rule,part,b,'incomplete (forced E missing), partial counts',sums)

if __name__ == '__main__':
    print('columns: C right, either no-write source right, C alone vs two no-write sources; when forced E exists: C only vs forced E, forced E only vs C, both, neither, C only vs all three, other three only vs C')
    for task,reader in [('qq',qq),('tq',tq)]:
        data = {m: reader(m) for m in T.ORDER}
        report(data, task)
        selected(task, data)
