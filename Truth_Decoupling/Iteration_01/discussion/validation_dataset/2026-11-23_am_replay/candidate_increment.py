"""Post hoc: on the original E submission mask, attribute answer replacement to the stack first pass or a C release.
No training, generation or scoring calls. Run from Truth_Decoupling on the scoring host.
"""
import sys,json,random
from pathlib import Path
import numpy as np
sys.path[:0]=['Iteration_01/diag','Iteration_01/code/scripts/newsets','Iteration_01/code/scripts/review_checks']
import selfjudge_stack_pick as K
import selfjudge_stack as S
import n1_on_sentence2 as N2
import qaqa_pilot_table as Q

def pick(st,e,margins):
    if not e[1]:return e[0]
    if not st[1] or st[2]==e[2]:return e[0]
    get=lambda item: margins[K.SP.sha1(item[2])]['baseline']
    return st[0] if get(st)>get(e) else e[0]

def data(m,task):
    M=K.lookup(m)
    if task=="TQA":
        ids,raw,_=K.tqa(m)['test']
        sr,_,_,_=S.sn1(m,1)
        se={q:r['SE'][1] for q,r in sr.items()}
        groups={q:'all' for q in ids}
    else:
        ids,raw,types=K.qaqa(m)['qaqa'];ids=[q for q in ids if not types[q][1]]
        groups={q:types[q][0] for q in ids}
        root=Path(Q.S.NS); bank=str(root/'bank/qaqa_full_bank_20261122.jsonl')
        R=Q.Raw(m,'qaqafull',bank,['test_full570_h22n','test_full570_h22n1','test_full570_h22z'])
        R.gdir=root/'gen_prompt/h22/qaqafull'/m/'test'
        arm=f'x_E2:E{Q.SET[m][0]}.w.p{N2.choice(m)["strength_dot"]}'
        R.need(arm);se={q: not R.blank(arm,q) for q in ids}
        for q in ids:
            if se[q] and raw[q]['St'][5] != R.text(arm,q):raise ValueError((m,q,'SE text'))
    out={}
    for q in ids:
        d=raw[q];e=d['E'];st=d['St']
        y=pick(st,e,M)-e[0]
        release=st[1] and not se[q]; overlap=e[1] and st[1] and e[2]!=st[2]
        out[q]=(groups[q],(int(e[1]),y, y if not release else 0,y if release else 0,int(release and overlap)))
    return ids,out

def main():
    for task in ('TQA','QA2'):
        models={m:data(m,task) for m in K.ORDER}
        shared=set.intersection(*(set(x[0]) for x in models.values()))
        ids=sorted(shared) if task=='TQA' else [q for q in models[K.ORDER[0]][0] if q in shared]
        for lab in (('all',) if task=='TQA' else ('all',*K.SP.QTYPES)):
            qs=[q for q in ids if lab=='all' or models[K.ORDER[0]][1][q][0]==lab]
            tab=np.array([[np.sum([d[q][1][j] for _,d in models.values()]) for j in range(5)] for q in qs])
            n=len(qs);rng=random.Random(20261116 if task=='TQA' else 20261119)
            weights=np.array([np.bincount([rng.randrange(n) for _ in qs],minlength=n) for _ in range(4000)])
            sums=tab.sum(0); ci=np.sort(weights@tab[:,1:4],axis=0)
            print(task,lab,"questions, E answers, selectable C-release differing from E",n,sums[0],sums[4],
                  "total / P+E first-pass / C-release gain vs E",[(int(x),int(a),int(b)) for x,a,b in zip(sums[1:4],ci[100],ci[3899])],flush=True)
if __name__=='__main__':main()
