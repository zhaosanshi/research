"""CPU only; from the Truth_Decoupling root. Re-rank archived batch-one answers at the same per-model quota."""
import sys
import random
import numpy as np
sys.path[:0]=['Iteration_01/diag','Iteration_01/code/scripts/newsets']
import selfjudge_stack as stack

def kept(items,k):
    eligible=[i for i,(_,issued,_) in enumerate(items) if issued]
    if k>len(eligible): raise ValueError("unreachable count")
    drop=set(sorted(eligible,key=lambda i:(items[i][2],i))[:len(eligible)-k])
    return np.array([0 if i in drop else x[0] for i,x in enumerate(items)])

def main():
    bank={m:stack.load(m,stack.TA.load(m))['test'] for m in stack.ORDER}
    for models in [tuple(stack.ORDER[:3]), tuple(stack.ORDER)]:
        ids=sorted(set.intersection(*(set(bank[m][0]) for m in models)))
        if len(ids)!=(521 if len(models)==3 else 520): raise ValueError("unexpected roster")
        rng=random.Random(20261116)
        n=len(ids)
        weights=np.asarray([np.bincount([rng.randrange(n) for _ in range(n)],minlength=n) for _ in range(4000)])
        for arms in [('S0z','SN1z'),('S0z','SEz','SN1z'),('SEz','SN1z')]:
            target=[];scores=[]
            for m in models:
                rows=bank[m][1]
                cand=[[rows[q]['baseline'][a] for q in ids] for a in arms]
                k=min(sum(r[1] for r in sheet) for sheet in cand)
                target.append(k)
                scores.append([kept(sheet,k) for sheet in cand])
            values=np.array(scores)
            print("models",len(models),"shared questions",len(ids),"arms",arms,"counts",target,"matched",sum(target),"nets",values.sum(axis=(0,2)).tolist())
            for i in range(len(arms)):
                for j in range(i):
                    ds=values[:,i]-values[:,j]
                    byq=ds.sum(axis=0);bs=np.sort(weights@byq)
                    print(" ",arms[i],"-",arms[j],int(byq.sum()),[int(bs[100]),int(bs[3899])],"by model",ds.sum(axis=1).tolist())
if __name__=='__main__': main()
