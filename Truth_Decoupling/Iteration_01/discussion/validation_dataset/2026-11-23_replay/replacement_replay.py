"""Offline independent H replay. Run from Truth_Decoupling in the scoring container."""
import sys,json,hashlib,contextlib,io
from pathlib import Path
import numpy as np
sys.path[:0]=['Iteration_01/diag','Iteration_01/code/scripts/newsets','Iteration_01/code/scripts/review_checks']
import replacement_2x2 as H,n1_on_sentence as N,qaqa_pilot_table as Q

def tq(m):
    display,l,s=N.A.KEY[m]; i=1
    D=N.MP.MT.Data(display,i); run=D.m['ec'][i].split('/eval_extra/')[0];cl=D.m['c']
    ops=[(N.MP.work(run,'test','h22c4'),'baseline_x_E2'),(D.m['d'][i],'baseline_conf_E2'),(N.MP.work(run,'test',f'pospromptdl{l}'),f'E_L{l}_conf-w-p{s}'),(N.MP.work(run,'test',f'posprompt2l{l}'),f'X_x--E{l}-w-p{s}-and-C{cl}-w-m0-25-at-prompt'),(N.MP.work(run,'test','h22c4'),f'C_L{cl}_x_E2-w-m0-25')]
    found=[]
    for w,a in ops:
        v,b=N.ps.load(w,'pro');z=N.fr.raw_texts(w)
        found.append({q:(N.V[score],not b[q,arm],z[q,arm].strip()) for (q,arm),score in v.items() if arm==a and score!='?'})
    return {q:[v[q] for v in found] for q in D.sp.qs if all(q in v for v in found)}
def qq(m):
    root=Path(Q.S.NS);bank=root/'bank/qaqa_full_bank_20261122.jsonl';l,s,cl=Q.SET[m][:3]
    base=Q.Raw(m,'qaqafull',str(bank),['test_full570','test_full570_tuned','test_full570_strong'])
    new4=Q.Raw(m,'qaqafull',str(bank),['test_full570_h22'])
    new0=Q.Raw(m,'qaqafull',str(bank),['test_full570_h22z'])
    for raw in [new4,new0]:raw.gdir=root/'gen_prompt/h22/qaqafull'/m/'test'
    ops=[(new0,'x_E2'),(base,'conf_E2'),(base,'conf:'+base.e(s).split(':')[1]),(base,base.ec(s,'0.25')),(new4,f'x_E2:C{cl}.w.m0.25')]
    for raw,a in ops:raw.need(a)
    return {q:[(raw.cell(a,q),not raw.blank(a,q),raw.text(a,q).strip()) for raw,a in ops] for q in base.ids if q not in H.PILOT}
for label,read in [('TQA',tq),('QA2',qq)]:
    data={m:read(m) for m in N.ORDER}
    ids=sorted(set.intersection(*(set(g) for g in data.values()))) if label=='TQA' else list(data[N.ORDER[0]])
    for restrict in (False,True) if label=='TQA' else (False,):
        if restrict: ids=sorted(set(ids).intersection(*(set(N.restricted(m,1)) for m in N.ORDER)))
        for shared in (False,True):
            diff=[];raws=[];n=0;tot=np.zeros(5,dtype=np.int64);confs=0
            for q in ids:
                w=[0,0]; ss=[0,0]
                for m in N.ORDER:
                    vals=data[m][q]; issued=[x[1] for x in vals]
                    if shared and not all(issued[1:]): continue
                    n+=1; v=[x[0] for x in vals];tot+=np.array(v)
                    txt={}; vc=[txt.setdefault(t,score) for score,_,t in vals]
                    confs+=int(vc!=v);w[0]+=v[2]-v[4];w[1]+=v[3]-v[4]-v[2]+v[1];ss[0]+=vc[2]-vc[4];ss[1]+=vc[3]-vc[4]-vc[2]+vc[1]
                raws.append(w);diff.append(ss)
            ci=lambda cols:[Q.S.joint_ci([[v[i] for v in cols]]) for i in range(2)]
            print(label,restrict,len(ids),shared,n,'values',tot.tolist(),'raw 2-4/interact',ci(raws),'harmonised',ci(diff),'conflict items',confs,flush=True)
