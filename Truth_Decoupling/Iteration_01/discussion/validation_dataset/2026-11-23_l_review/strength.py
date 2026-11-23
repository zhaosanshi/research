"""CPU-only replay from Truth_Decoupling in scoring container, independent choice and cap decisions."""
import itertools
import json
import random
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,"Iteration_01/diag")
import n1_on_sentence3 as I
import policy_select2 as P
import release_origin as R

CAPS = list(itertools.product(range(5),repeat=2))
ROOT = Path(I.NSD)
PILOT = set(json.loads(Path("Iteration_01/newsets/splits/qaqa_pilot_20261122.json").read_bytes())["ids"])
assert len(PILOT)==100


def tqa(model,i,s):
    ids=I.N.restricted(model,i)
    sp,tx=I.tq_build(model,i,s,ids)
    e=sp.cE(I.N.dash(s)); c=[sp.cEC(I.N.dash(s),k) for k in P.LAD]
    def cell(q,a):
        return I.V[sp.st[q,a]], not sp.blank[q,a], R.norm(tx.ec[q,a])
    return {q:[cell(q,a) for a in ("baseline_x",e,*c)] for q in sp.qs}


def qaqa(model,s):
    raw=I.qaqa_raw(model,s)
    e=raw.e(s); cs=[raw.ec(s,k) for k in I.LADK]
    return {q:[(raw.cell(a,q),not raw.blank(a,q), R.norm(raw.text(a,q))) for a in ("x",e,*cs)] for q in raw.ids},raw.qtype


def answer(row,caps):
    p,e,*cs=row
    if e[1]: return e[:2]
    end=caps[0 if p[1] else 1]
    for c in cs[:end]:
        if c[1]:
            if p[1] and c[2]==p[2]: return e[:2]
            return c[:2]
    return e[:2]


def choose(rows,ids):
    # Index 0 corresponds to no fill, then 0.25, 0.5, 1, 2; first maximizer wins, as in the plan.
    scores=[sum(answer(rows[q],c)[0] for q in ids) for c in CAPS]
    best=scores.index(max(scores))
    return CAPS[best], scores[best]


def main():
    data={}
    for m in I.ORDER:
        s,h=I.strengths(m); frozen=json.loads(Path(I.CHOICE3 % m).read_bytes())
        if (frozen["model"],frozen["s_half"],frozen["s_sel"]) != (m,h,s): raise ValueError("wrong saved choice")
        terms={}; caps={}; archive={}
        for dose in (s,h):
            train=tqa(m,0,dose);q,types=qaqa(m,dose)
            if set(train)!=set(I.N.restricted(m,0)) or len(q)!=570 or not PILOT.issubset(q): raise ValueError("tune roster")
            tqc,ts=choose(train,train)
            qc,qs=choose(q,PILOT)
            el=I.Q.SET[m][0];arm=f"x_E2:E{el}.w.p{dose}"
            fact=[sum(a[0] for a in I.fact_arm(task,m,"tune",arm)[0].values()) for task in ("triviaqa","popqa")]
            total=ts+qs+sum(fact); rec=frozen["rows"][dose]
            if [ts,qs,*fact,total]!=[rec[k] for k in ("truthfulqa_tune_sn1","qaqa_pilot_sn1","triviaqa_tune_se","popqa_tune_se","total")]:raise ValueError(("choice mismatch",m,dose))
            terms[dose]=total;caps[dose]=(tqc,qc);archive[dose]=q
        pick=h if terms[h]>=terms[s] else s
        if pick!=frozen["chosen"]:raise ValueError(("selected wrong strength",m))
        print("chosen",m,s,h,terms[s],terms[h],pick,flush=True)
        by_task={}
        for tag,dose in (("old",s),("new",pick)):
            tqc,qc=caps[dose]
            test=tqa(m,1,dose)
            T={q:(answer(rs,tqc),rs[1][:2]) for q,rs in test.items()}
            QQ={q:(answer(rs,qc),rs[1][:2]) for q,rs in archive[dose].items() if q not in PILOT}
            by_task[tag]={"TQA":T,"QA2":QQ}
            for task,split in I.FACT:
                vals,_=I.fact_arm(task,m,split,f"x_E2:E{el}.w.p{dose}")
                by_task[tag][task+' '+split]={q:(cell,cell) for q,cell in vals.items()}
        data[m]=by_task
    for task in data[I.ORDER[0]]["new"]:
        common=sorted(set.intersection(*(set(data[m][k][task]) for k in ('old','new') for m in I.ORDER)))
        groups=["all"] if task!="QA2" else ["all","no_failure","failure","unverifiable"]
        for typ in groups:
            qs=[q for q in common if typ=="all" or types[q]==typ]
            arrays={}
            for tag in ('old','new'):
                arrays[tag]=np.array([[[data[m][tag][task][q][idx][0] for q in qs] for idx in (0,1)] for m in I.ORDER]).sum(axis=0)
            rng=random.Random(20261116 if task=="TQA" else 20261119)
            w=np.array([np.bincount([rng.randrange(len(qs)) for j in qs],minlength=len(qs)) for _ in range(4000)])
            v=arrays['new'][0]-arrays['old'][0]
            sims=np.sort(w@v)
            print(task,typ,len(qs),"old/new net (combo if available else SE)",int(arrays['old'][0].sum()),int(arrays['new'][0].sum()),"delta",int(v.sum()),(int(sims[100]),int(sims[3899])),flush=True)

if __name__=="__main__":main()
